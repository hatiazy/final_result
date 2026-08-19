from __future__ import annotations

"""Main audit for the independent remote validation upload package.

The signal engines read only the spot file.  The remote three-state file is
read afterwards as an audit baseline, never as a model input.
"""

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
DEFAULT_THREE_STATE_PATH = "/home/hzy/cta/三状态冻结/IC_1545_three_state_and_downside_warning.csv"
LOCAL_SMOKE_SPOT = PACKAGE_ROOT / "data" / "local_smoke" / "CSI500_SPOT_000905_XSHG_20070115_20260817_audited_pure_spot_fixture.parquet"
REPORT_EXPECTED = PACKAGE_ROOT / "expected" / "report_freeze"
LOCAL_EXPECTED = PACKAGE_ROOT / "expected" / "local_smoke"
OUTPUT_ROOT = PACKAGE_ROOT / "runtime_outputs"

NONZERO_COLUMNS = ["date", "three_state", "minus_exit_signal", "plus_exit_signal", "final_three_state"]
EXTREME_COLUMNS = ["date", "close", "score", "predicted", "actual_extreme", "correct", "direction_correct", "o2o_bp", "signed_o2o_bp", "phase"]


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
    if any(token in text for token in "*?["):
        import glob

        matches = sorted(Path(item).expanduser().resolve() for item in glob.glob(text) if Path(item).is_file())
        if len(matches) != 1:
            raise FileNotFoundError(f"{label} 通配路径必须唯一匹配一个文件；matches={[str(item) for item in matches]}")
        return matches[0]
    path = Path(text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在: {path}")
    return path


def _resolve_spot(explicit: str | Path | None) -> tuple[Path, str]:
    if explicit is not None:
        return _resolve_one(explicit, "现货"), "explicit"
    configured = os.environ.get("COMPANY_SPOT_PATH", DEFAULT_SPOT_PATTERN)
    try:
        return _resolve_one(configured, "现货"), "remote_default_or_environment"
    except FileNotFoundError:
        if LOCAL_SMOKE_SPOT.is_file():
            return LOCAL_SMOKE_SPOT.resolve(), "local_smoke_fallback"
        raise


def _resolve_baseline(explicit: str | Path | None, source_kind: str) -> tuple[Path, str]:
    if explicit is not None:
        return _resolve_one(explicit, "远端三状态"), "explicit"
    configured = os.environ.get("REMOTE_THREE_STATE_PATH", DEFAULT_THREE_STATE_PATH)
    try:
        return _resolve_one(configured, "远端三状态"), "remote_default_or_environment"
    except FileNotFoundError:
        if source_kind == "local_smoke_fallback":
            return (LOCAL_EXPECTED / "本地三状态基准.csv").resolve(), "local_smoke_fallback"
        raise


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def _dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def _calendar_dates(values: pd.Series) -> pd.Series:
    """Parse daily trading dates using the same compact-date rules as the engines."""
    text = values.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    noncompact = ~compact
    if noncompact.any():
        parsed.loc[noncompact] = pd.to_datetime(text.loc[noncompact], errors="coerce", format="mixed")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    return parsed.dt.normalize()


def _load_nonzero(path: Path) -> pd.DataFrame:
    frame = _read_table(path)
    missing = sorted(set(NONZERO_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"非零退出结果缺少列: {missing}")
    frame = frame.loc[:, NONZERO_COLUMNS].copy()
    frame["date"] = _dates(frame["date"])
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("非零退出结果 date 无效或重复")
    for column in NONZERO_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int8")
    return frame.sort_values("date").reset_index(drop=True)


def _load_baseline(path: Path) -> pd.DataFrame:
    frame = _read_table(path)
    date_column = next((x for x in ["date", "effective_date", "trade_dt", "trade_date"] if x in frame.columns), None)
    state_column = next((x for x in ["three_state", "state", "base_state", "final_three_state"] if x in frame.columns), None)
    if date_column is None or state_column is None:
        raise ValueError(f"远端三状态缺少日期或状态列: {list(frame.columns)}")
    out = frame[[date_column, state_column]].rename(columns={date_column: "date", state_column: "remote_three_state"}).copy()
    out["date"] = _dates(out["date"])
    out["remote_three_state"] = pd.to_numeric(out["remote_three_state"], errors="raise").astype("int8")
    if out["date"].isna().any() or out["date"].duplicated().any():
        raise ValueError("远端三状态 date 无效或重复")
    return out.sort_values("date").reset_index(drop=True)


def _load_extreme(path: Path) -> pd.DataFrame:
    frame = _read_table(path)
    missing = sorted(set(EXTREME_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"大涨大跌逐日结果缺少列: {missing}")
    frame = frame.loc[:, EXTREME_COLUMNS].copy()
    frame["date"] = _dates(frame["date"])
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("大涨大跌逐日结果 date 无效或重复")
    for column in ["predicted", "actual_extreme", "correct", "direction_correct"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int8")
    for column in ["close", "score", "o2o_bp", "signed_o2o_bp"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["phase"] = frame["phase"].astype("string").str.lower()
    return frame.sort_values("date").reset_index(drop=True)


def _spot_calendar(path: Path) -> pd.DatetimeIndex:
    frame = _read_table(path)
    aliases = ("trade_dt", "trade_date", "date", "datetime", "交易日")
    lower_columns = {str(column).strip().lower(): column for column in frame.columns}
    candidates: list[tuple[int, str, pd.DatetimeIndex]] = []
    for priority, alias in enumerate(aliases):
        column = lower_columns.get(alias.lower())
        if column is None:
            continue
        dates = _calendar_dates(frame[column]).dropna().drop_duplicates().sort_values()
        if len(dates) >= 2:
            candidates.append((priority, str(column), pd.DatetimeIndex(dates)))

    # Some vendor exports keep the trading date in the index rather than a named column.
    if isinstance(frame.index, pd.DatetimeIndex):
        index_dates = pd.DatetimeIndex(frame.index.normalize()).dropna().drop_duplicates().sort_values()
        if len(index_dates) >= 2:
            candidates.append((len(aliases), "<index>", index_dates))

    if not candidates:
        available = [str(column) for column in frame.columns]
        raise ValueError(
            "现货没有至少两个可解析交易日，无法生成实际执行日映射；"
            f"候选字段={aliases}，实际字段={available}"
        )

    # Prefer the candidate with the most complete calendar.  This handles files that
    # carry a short/placeholder `date` column alongside the real `trade_dt` field.
    _, _, calendar = max(candidates, key=lambda item: (len(item[2]), -item[0]))
    return calendar


def _build_execution_summary(
    spot_path: Path,
    nonzero: pd.DataFrame,
    down: pd.DataFrame,
    up: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """Write one compact execution-date view after all detailed comparisons."""
    calendar = _spot_calendar(spot_path)
    next_execution = pd.Series(calendar[1:].to_numpy(), index=calendar[:-1])
    result = nonzero.loc[:, ["date", "three_state", "plus_exit_signal", "minus_exit_signal"]].rename(
        columns={
            "date": "实际执行日",
            "three_state": "三状态",
            "plus_exit_signal": "+1反转",
            "minus_exit_signal": "-1反转",
        }
    ).copy()

    mapped_prediction_counts: dict[str, int] = {}
    for side, label in (("up", "大涨"), ("down", "大跌")):
        source = up if side == "up" else down
        side_frame = pd.DataFrame({
            "实际执行日": source["date"].map(next_execution),
            label: source["predicted"].astype(int),
        }).dropna(subset=["实际执行日"])
        if side_frame["实际执行日"].duplicated().any():
            raise ValueError(f"{label}预测映射到实际执行日后出现重复日期")
        mapped_prediction_counts[label] = int(side_frame[label].sum())
        result = result.merge(side_frame, on="实际执行日", how="outer")

    if result["三状态"].isna().any():
        missing = result.loc[result["三状态"].isna(), "实际执行日"].dt.strftime("%Y-%m-%d").tolist()[:10]
        raise ValueError(f"大涨大跌预测的实际执行日无法与三状态日期对齐: {missing}")
    for column in ["+1反转", "-1反转", "大涨", "大跌"]:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype("int8")
    result["三状态"] = pd.to_numeric(result["三状态"], errors="raise").astype("int8")
    result = result.sort_values("实际执行日").reset_index(drop=True)
    result["实际执行日"] = result["实际执行日"].dt.strftime("%Y-%m-%d")
    columns = ["实际执行日", "三状态", "+1反转", "-1反转", "大涨", "大跌"]
    result.loc[:, columns].to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "path": str(output_path),
        "columns": columns,
        "rows": int(len(result)),
        "date_min": str(result["实际执行日"].min()) if len(result) else None,
        "date_max": str(result["实际执行日"].max()) if len(result) else None,
        "nonzero_plus_reversal_days": int(result["+1反转"].sum()),
        "nonzero_minus_reversal_days": int(result["-1反转"].sum()),
        "big_up_prediction_days": int(result["大涨"].sum()),
        "big_down_prediction_days": int(result["大跌"].sum()),
        "mapped_prediction_counts": mapped_prediction_counts,
        "valid": bool(result["实际执行日"].is_unique and result["三状态"].isin([-1, 0, 1]).all()),
        "meaning": "非零退出信号沿用有效执行日；大涨/大跌由形成日预测映射到下一实际现货交易日。所有信号列为0/1。",
    }


def _compare(left: pd.DataFrame, right: pd.DataFrame, columns: list[str], label: str, float_atol: float = 1e-12) -> tuple[pd.DataFrame, dict[str, Any]]:
    left = left.loc[:, columns].copy()
    right = right.loc[:, columns].copy()
    left = left.sort_values("date").reset_index(drop=True)
    right = right.sort_values("date").reset_index(drop=True)
    merged = left.merge(right, on="date", how="outer", suffixes=("_generated", "_local"), indicator=True)
    compare_columns: list[str] = []
    for column in columns:
        if column == "date":
            continue
        generated = merged.get(f"{column}_generated")
        local = merged.get(f"{column}_local")
        if generated is None or local is None:
            continue
        if pd.api.types.is_numeric_dtype(generated) or pd.api.types.is_numeric_dtype(local):
            equal = np.isclose(
                pd.to_numeric(generated, errors="coerce"),
                pd.to_numeric(local, errors="coerce"),
                rtol=0.0,
                atol=float_atol,
                equal_nan=True,
            )
        else:
            equal = generated.astype("string").eq(local.astype("string")).to_numpy()
        equal = np.asarray(equal, dtype=bool)
        equal &= merged["_merge"].eq("both").to_numpy()
        merged[f"match_{column}"] = equal
        compare_columns.append(column)
    match_columns = [f"match_{column}" for column in compare_columns]
    merged["all_columns_match"] = merged[match_columns].all(axis=1) if match_columns else False
    merged["row_status"] = np.where(
        merged["_merge"].eq("both"),
        np.where(merged["all_columns_match"], "MATCH", "MISMATCH"),
        merged["_merge"].str.upper(),
    )
    common = merged["_merge"].eq("both")
    metrics = {
        "label": label,
        "generated_rows": int(len(left)),
        "local_rows": int(len(right)),
        "common_rows": int(common.sum()),
        "generated_only_rows": int(merged["_merge"].eq("left_only").sum()),
        "local_only_rows": int(merged["_merge"].eq("right_only").sum()),
        "mismatch_rows_on_common_dates": int((common & ~merged["all_columns_match"]).sum()),
        "no_date_or_value_difference": bool(len(merged) > 0 and merged["_merge"].eq("both").all() and merged["all_columns_match"].all()),
    }
    return merged, metrics


def _run_child(script: Path, args: list[str], env: dict[str, str], log_path: Path, label: str) -> None:
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
        raise RuntimeError(f"{label} 失败，退出码={code}，完整日志={log_path}")


def run_remote_validation(
    spot_path: str | Path | None = None,
    three_state_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    spot, source_kind = _resolve_spot(spot_path)
    baseline, baseline_kind = _resolve_baseline(three_state_path, source_kind)
    output = Path(output_dir or os.environ.get("UPLOAD_OUTPUT_DIR", str(OUTPUT_ROOT))).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_root = LOCAL_EXPECTED if source_kind == "local_smoke_fallback" else REPORT_EXPECTED

    env = os.environ.copy()
    env["COMPANY_SPOT_PATH"] = str(spot)
    env["1545_SPOT_PATH"] = str(spot)
    env["UPLOAD_OUTPUT_DIR"] = str(output)
    env["PYTHONUNBUFFERED"] = "1"
    _run_child(PACKAGE_ROOT / "src" / "run_nonzero_remote.py", [], env, output / "run_nonzero.log", "非零退出")
    for side in ("down", "up"):
        _run_child(
            PACKAGE_ROOT / "src" / "run_extreme_side.py",
            [side, str(spot), str(output)],
            env,
            output / f"run_extreme_{side}.log",
            f"大涨大跌-{side}",
        )

    generated_nonzero = _load_nonzero(output / "remote_nonzero_five_columns.csv")
    local_nonzero = _load_nonzero(expected_root / "本地非零五列参考结果.csv")
    nonzero_compare, nonzero_audit = _compare(generated_nonzero, local_nonzero, NONZERO_COLUMNS, "远端非零五列 vs 本地参考")
    nonzero_compare.to_csv(output / "非零五列逐日对比.csv", index=False, encoding="utf-8-sig")

    generated_base = generated_nonzero[["date", "three_state"]].rename(columns={"three_state": "generated_three_state"})
    remote_base = _load_baseline(baseline)
    state_compare = generated_base.merge(remote_base, on="date", how="outer", indicator=True)
    state_compare["match"] = state_compare["generated_three_state"].eq(state_compare["remote_three_state"])
    state_compare["match"] = state_compare["match"].fillna(False)
    state_compare["row_status"] = np.where(state_compare["_merge"].eq("both"), np.where(state_compare["match"], "MATCH", "MISMATCH"), state_compare["_merge"].str.upper())
    state_compare.to_csv(output / "三状态逐日对比.csv", index=False, encoding="utf-8-sig")
    state_common = state_compare["_merge"].eq("both")
    state_audit = {
        "baseline_path": str(baseline),
        "baseline_source_kind": baseline_kind,
        "generated_rows": int(len(generated_base)),
        "baseline_rows": int(len(remote_base)),
        "common_rows": int(state_common.sum()),
        "generated_only_rows": int(state_compare["_merge"].eq("left_only").sum()),
        "baseline_only_rows": int(state_compare["_merge"].eq("right_only").sum()),
        "mismatch_rows_on_common_dates": int((state_common & ~state_compare["match"]).sum()),
        "no_date_or_value_difference": bool(len(state_compare) > 0 and state_compare["_merge"].eq("both").all() and state_compare["match"].all()),
    }

    extreme_audits: dict[str, dict[str, Any]] = {}
    for side in ("down", "up"):
        generated = _load_extreme(output / f"remote_{side}_extreme_daily.csv")
        local = _load_extreme(expected_root / f"{side}_大涨大跌逐日参考.csv")
        compare, audit = _compare(generated, local, EXTREME_COLUMNS, f"远端{side}侧 vs 本地参考")
        compare.to_csv(output / f"大涨大跌_{side}_逐日对比.csv", index=False, encoding="utf-8-sig")
        extreme_audits[side] = audit

    execution_summary = _build_execution_summary(
        spot,
        generated_nonzero,
        _load_extreme(output / "remote_down_extreme_daily.csv"),
        _load_extreme(output / "remote_up_extreme_daily.csv"),
        output / "最终执行日简表.csv",
    )

    manifest = {
        "package": "05_上传包_日期解析修正版_非零退出与大涨大跌远端验证",
        "spot_path": str(spot),
        "spot_source_kind": source_kind,
        "remote_three_state_path": str(baseline),
        "expected_reference_root": str(expected_root),
        "output_dir": str(output),
        "engine_input_policy": "两个引擎只读取现货文件；三状态文件仅在引擎结束后作为逐日审计基准读取。",
        "generated_files": {
            "nonzero": str(output / "remote_nonzero_five_columns.csv"),
            "nonzero_compare": str(output / "非零五列逐日对比.csv"),
            "state_compare": str(output / "三状态逐日对比.csv"),
            "extreme_down": str(output / "remote_down_extreme_daily.csv"),
            "extreme_up": str(output / "remote_up_extreme_daily.csv"),
            "extreme_down_compare": str(output / "大涨大跌_down_逐日对比.csv"),
            "extreme_up_compare": str(output / "大涨大跌_up_逐日对比.csv"),
            "execution_summary": str(output / "最终执行日简表.csv"),
        },
        "nonzero_audit": nonzero_audit,
        "state_audit": state_audit,
        "extreme_audits": extreme_audits,
        "execution_summary": execution_summary,
        "success": bool(
            nonzero_audit["no_date_or_value_difference"]
            and state_audit["no_date_or_value_difference"]
            and all(item["no_date_or_value_difference"] for item in extreme_audits.values())
            and execution_summary["valid"]
        ),
    }
    manifest_path = output / "最终一致性结论.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print("\nREMOTE_VALIDATION_AUDIT_END", flush=True)
    print(json.dumps({"success": manifest["success"], "nonzero": nonzero_audit, "state": state_audit, "extreme": extreme_audits}, ensure_ascii=False, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    run_remote_validation()
