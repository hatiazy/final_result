from __future__ import annotations

"""Generate the six-column execution-date output from remote spot data only.

This module deliberately contains no local-reference comparison.  It runs the
already-frozen V55/V80/V156/V189 engines, keeps the latest extreme predictions
even when their future O2O labels are not available yet, then aligns all
outputs to the actual execution-date grid and writes one compact CSV.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPOT_PATTERN = "/home/hzy/cta/IC数据更新*最终固化版/现货最终版/CSI500_SPOT_md_eod_raw*最终版.parquet"
OUTPUT_ROOT = PACKAGE_ROOT / "runtime_outputs"

COMPACT_COLUMNS = ["实际执行日", "三状态", "+1反转", "-1反转", "大涨", "大跌"]
NONZERO_COLUMNS = ["date", "three_state", "minus_exit_signal", "plus_exit_signal", "final_three_state"]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return str(value)


def _resolve_one(raw: str | Path, label: str) -> Path:
    text = str(raw).strip()
    raw_path = Path(text).expanduser()
    if not raw_path.is_absolute():
        raise ValueError(f"{label} 必须使用绝对路径：{text}")
    if any(token in text for token in "*?["):
        import glob

        matches = sorted(
            Path(item).expanduser().resolve()
            for item in glob.glob(text)
            if Path(item).is_file()
        )
        if len(matches) != 1:
            raise FileNotFoundError(f"{label} 通配路径必须唯一匹配一个文件；matches={[str(item) for item in matches]}")
        return matches[0]
    path = raw_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在：{path}")
    return path


def resolve_spot(explicit: str | Path | None = None) -> Path:
    """Resolve only the remote spot input; no local fallback is allowed."""
    configured = explicit if explicit is not None else os.environ.get("COMPANY_SPOT_PATH", DEFAULT_SPOT_PATTERN)
    return _resolve_one(configured, "远端现货")


def _resolve_absolute_directory(raw: str | Path, label: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} 必须使用绝对路径：{raw}")
    return path.resolve()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if (~compact).any():
        parsed.loc[~compact] = pd.to_datetime(text.loc[~compact], errors="coerce", format="mixed")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    return parsed.dt.normalize()


def spot_calendar(path: Path) -> pd.DatetimeIndex:
    frame = _read_table(path)
    aliases = ("trade_dt", "trade_date", "date", "datetime", "交易日")
    lower = {str(column).strip().lower(): column for column in frame.columns}
    candidates: list[tuple[int, str, pd.DatetimeIndex]] = []
    for priority, alias in enumerate(aliases):
        column = lower.get(alias.lower())
        if column is None:
            continue
        dates = _parse_dates(frame[column]).dropna().drop_duplicates().sort_values()
        if len(dates) >= 2:
            candidates.append((priority, str(column), pd.DatetimeIndex(dates)))
    if isinstance(frame.index, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(frame.index.normalize()).drop_duplicates().sort_values()
        if len(dates) >= 2:
            candidates.append((len(aliases), "<index>", dates))
    if not candidates:
        raise ValueError(f"远端现货没有至少两个可解析交易日；字段={list(frame.columns)}")
    return max(candidates, key=lambda item: (len(item[2]), -item[0]))[2]


def load_nonzero(path: Path) -> pd.DataFrame:
    frame = _read_table(path)
    missing = sorted(set(NONZERO_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"非零冻结结果缺少列：{missing}")
    frame = frame.loc[:, NONZERO_COLUMNS].copy()
    frame["date"] = _parse_dates(frame["date"])
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("非零冻结结果的 date 无效或重复")
    for column in NONZERO_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int8")
    return frame.sort_values("date").reset_index(drop=True)


def load_extreme(path: Path) -> pd.DataFrame:
    frame = _read_table(path)
    required = {"date", "predicted"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"大涨大跌冻结结果缺少列：{missing}")
    frame = frame.loc[:, ["date", "predicted"]].copy()
    frame["date"] = _parse_dates(frame["date"])
    frame["predicted"] = pd.to_numeric(frame["predicted"], errors="raise").astype("int8")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("大涨大跌冻结结果的 date 无效或重复")
    return frame.sort_values("date").reset_index(drop=True)


def _run_child(script: Path, args: list[str], env: dict[str, str], label: str, log_path: Path) -> None:
    command = [sys.executable, str(script), *args]
    print(f"[{label}] 启动", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PACKAGE_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"{label} 失败，退出码={code}；详见 {log_path}")


def build_compact_summary(spot_path: Path, output_dir: Path) -> dict[str, Any]:
    calendar = spot_calendar(spot_path)
    next_execution = pd.Series(calendar[1:].to_numpy(), index=calendar[:-1])

    nonzero = load_nonzero(output_dir / "remote_nonzero_five_columns.csv")
    result = nonzero.loc[:, ["date", "three_state", "plus_exit_signal", "minus_exit_signal"]].rename(
        columns={
            "date": "实际执行日",
            "three_state": "三状态",
            "plus_exit_signal": "+1反转",
            "minus_exit_signal": "-1反转",
        }
    )

    # The nonzero exporter already emits its effective-date grid.  Check that
    # this grid is exactly the next actual spot date for every formation row,
    # with only the final formation row allowed to use the next business day
    # until a new spot row is available.  This prevents a later fillna(0) from
    # silently creating an execution-day placeholder.
    first_execution = pd.Timestamp(nonzero["date"].min())
    expected_nonzero_dates = calendar[
        (calendar >= first_execution) & (calendar <= calendar[-1])
    ]
    if pd.Timestamp(nonzero["date"].max()) > pd.Timestamp(calendar[-1]):
        expected_nonzero_dates = expected_nonzero_dates.append(
            pd.DatetimeIndex([pd.Timestamp(calendar[-1]) + pd.offsets.BDay(1)])
        )
    if not nonzero["date"].equals(pd.Series(expected_nonzero_dates, name="date")):
        raise ValueError(
            "非零信号执行日网格不是形成日后的下一实际交易日；"
            f"实际={nonzero['date'].min()}->{nonzero['date'].max()}，"
            f"期望={expected_nonzero_dates.min()}->{expected_nonzero_dates.max()}"
        )

    for side, label in (("up", "大涨"), ("down", "大跌")):
        prediction_path = output_dir / f"remote_{side}_extreme_predictions.csv"
        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"缺少{label}最新运行预测文件：{prediction_path}；"
                "请先用当前版本的 run_extreme_side.py 生成，不能只使用历史评价文件"
            )
        extreme = load_extreme(prediction_path)
        mapped_dates = extreme["date"].map(next_execution)
        pending = mapped_dates.isna()
        if pending.any():
            pending_formations = extreme.loc[pending, "date"]
            if len(pending_formations) != 1 or pd.Timestamp(pending_formations.iloc[0]) != pd.Timestamp(calendar[-1]):
                raise ValueError(
                    f"{label}存在无法映射到实际执行日的形成日："
                    f"{pending_formations.dt.strftime('%Y-%m-%d').tolist()}"
                )
            # The latest formation close is allowed to be ahead of the spot
            # file.  Keep the same next-weekday display convention as the
            # nonzero exporter until the next actual spot row arrives.
            mapped_dates.loc[pending] = pending_formations + pd.offsets.BDay(1)
        if not np.all(pd.to_datetime(mapped_dates).to_numpy() > extreme["date"].to_numpy()):
            raise ValueError(f"{label}存在未向后移位的信号日期，拒绝生成六列表")
        mapped = pd.DataFrame({
            "实际执行日": mapped_dates,
            label: extreme["predicted"].astype("int8"),
        })
        if mapped["实际执行日"].duplicated().any():
            raise ValueError(f"{label} 映射到实际执行日后出现重复日期")
        if set(mapped["实际执行日"]) != set(pd.to_datetime(result["实际执行日"])):
            raise ValueError(
                f"{label}形成日映射后的执行日集合与非零执行日集合不一致；"
                "拒绝用 0 填补缺失信号"
            )
        result = result.merge(mapped, on="实际执行日", how="outer")

    if result[COMPACT_COLUMNS[1:]].isna().any().any():
        raise ValueError("六列表存在未由形成日信号覆盖的执行日，拒绝填入默认 0")
    for column in ["三状态", "+1反转", "-1反转", "大涨", "大跌"]:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype("int8")
    result["实际执行日"] = pd.to_datetime(result["实际执行日"], errors="raise").dt.strftime("%Y-%m-%d")
    result = result.sort_values("实际执行日").reset_index(drop=True)
    result = result.loc[:, COMPACT_COLUMNS]

    output_path = output_dir.parent / "最终执行日简表.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    first_execution = pd.Timestamp(nonzero["date"].min())
    first_formation = calendar[calendar < first_execution][-1]
    latest_formation = pd.Timestamp(calendar[-1])
    latest_execution = latest_formation + pd.offsets.BDay(1)
    record = {
        "operation": "只用远端现货和包内冻结参数生成六列表；不读取本地参考，不做比较",
        "spot_path": str(spot_path),
        "engine_output_dir": str(output_dir),
        "output_path": str(output_path),
        "columns": COMPACT_COLUMNS,
        "rows": int(len(result)),
        "date_min": str(result["实际执行日"].min()) if len(result) else None,
        "date_max": str(result["实际执行日"].max()) if len(result) else None,
        "freeze": {
            "minus_exit": {"version": "V55", "threshold": 0.815842643776842, "confirm_days": 2, "min_state_age": 1},
            "plus_exit": {"version": "V80", "threshold": 0.4920211306764882, "confirm_days": 1, "min_state_age": 5},
            "big_down": {"version": "V156", "candidate": "base_0621", "threshold": 0.6832148298881413},
            "big_up": {"version": "V189", "candidate": "base_1839", "threshold": 0.8135114753699175},
        },
        "execution_date_rule": "每行均为形成日收盘后的计算结果；实际执行日是下一实际现货交易日；最新形成日尚无下一现货行时按下一工作日映射，数值仍来自最新形成日计算",
        "date_mapping": {
            "verified": True,
            "formation_date_min": first_formation.strftime("%Y-%m-%d"),
            "formation_date_max": latest_formation.strftime("%Y-%m-%d"),
            "execution_date_min": str(result["实际执行日"].min()),
            "execution_date_max": str(result["实际执行日"].max()),
            "latest_formation_to_execution": f"{latest_formation.strftime('%Y-%m-%d')} -> {latest_execution.strftime('%Y-%m-%d')}",
            "nonzero_grid_exact": True,
            "extreme_formation_to_execution_exact": True,
            "missing_signal_values_filled": False,
        },
        "extreme_prediction_input": "remote_up_extreme_predictions.csv / remote_down_extreme_predictions.csv",
        "candidate_rebuild": False,
        "comparison_performed": False,
    }
    (output_dir.parent / "最终执行日简表_生成记录.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return record


def generate_compact_output(
    spot_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    spot = resolve_spot(spot_path)
    configured_output = output_dir if output_dir is not None else os.environ.get("UPLOAD_OUTPUT_DIR", str(OUTPUT_ROOT))
    root_output = _resolve_absolute_directory(configured_output, "六列表输出目录")
    root_output.mkdir(parents=True, exist_ok=True)
    engine_output = root_output / "_engine_outputs"
    engine_output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["COMPANY_SPOT_PATH"] = str(spot)
    env["1545_SPOT_PATH"] = str(spot)
    env["UPLOAD_OUTPUT_DIR"] = str(engine_output)
    env["PYTHONUNBUFFERED"] = "1"

    _run_child(PACKAGE_ROOT / "src" / "run_nonzero_remote.py", [], env, "非零冻结", engine_output / "run_nonzero.log")
    for side, label in (("down", "大跌冻结"), ("up", "大涨冻结")):
        _run_child(
            PACKAGE_ROOT / "src" / "run_extreme_side.py",
            [side, str(spot), str(engine_output)],
            env,
            label,
            engine_output / f"run_extreme_{side}.log",
        )
    return build_compact_summary(spot, engine_output)


def main() -> None:
    parser = argparse.ArgumentParser(description="只用远端现货和冻结参数生成六列表")
    parser.add_argument("--spot", default=None, help="可选：远端现货文件；不填则使用 COMPANY_SPOT_PATH 或默认远端路径")
    parser.add_argument("--output", default=None, help="可选：输出目录；默认是 runtime_outputs")
    args = parser.parse_args()
    record = generate_compact_output(args.spot, args.output)
    print(f"六列表已生成：{record['output_path']}")
    print(f"行数：{record['rows']}；执行日：{record['date_min']} -> {record['date_max']}")


if __name__ == "__main__":
    main()
