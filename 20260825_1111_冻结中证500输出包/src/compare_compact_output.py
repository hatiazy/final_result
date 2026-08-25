from __future__ import annotations

"""Run the two independent 02 checks for the frozen eight-column package."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PACKAGE_ROOT / "runtime_outputs"
LOCAL_REFERENCE = PACKAGE_ROOT / "本地结果对照" / "01_只用现货生成八列" / "最终执行日简表.csv"
DEFAULT_REMOTE_THREE_STATE = str(OUTPUT_ROOT / "最终执行日简表.csv")
COMPACT_COLUMNS = ["实际执行日", "三状态", "+1反转", "-1反转", "0转-1", "0转+1", "大涨", "大跌"]
SIGNAL_COLUMNS = COMPACT_COLUMNS[1:]


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return str(value)


def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if (~compact).any():
        parsed.loc[~compact] = pd.to_datetime(text.loc[~compact], errors="coerce", format="mixed")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    return parsed.dt.normalize()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def _resolve_absolute_path(raw: str | Path, label: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label}必须使用绝对路径：{raw}")
    return path.resolve()


def _read_compact(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在：{path}")
    frame = _read_table(path)
    missing = [column for column in COMPACT_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in COMPACT_COLUMNS]
    if missing or extra:
        raise ValueError(f"{label}列不符合八列表；缺少={missing}；多出={extra}")
    frame = frame.loc[:, COMPACT_COLUMNS].copy()
    frame["实际执行日"] = _parse_dates(frame["实际执行日"])
    if frame["实际执行日"].isna().any():
        raise ValueError(f"{label}存在无法解析的实际执行日")
    if frame["实际执行日"].duplicated().any():
        raise ValueError(f"{label}存在重复实际执行日")
    for column in SIGNAL_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    if not frame["三状态"].isin([-1, 0, 1]).all():
        raise ValueError(f"{label}的三状态不在 -1/0/1 内")
    if not frame[["+1反转", "-1反转", "0转-1", "0转+1", "大涨", "大跌"]].isin([0, 1]).all().all():
        raise ValueError(f"{label}的六个信号不是 0/1")
    return frame.sort_values("实际执行日").reset_index(drop=True)


def _read_remote_three_state(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"远端三状态文件不存在：{path}")
    frame = _read_table(path)
    date_column = next(
        (column for column in ["date", "effective_date", "trade_dt", "trade_date", "实际执行日"] if column in frame.columns),
        None,
    )
    state_column = next(
        (column for column in ["three_state", "state", "base_state", "final_three_state", "三状态"] if column in frame.columns),
        None,
    )
    if date_column is None or state_column is None:
        raise ValueError(f"远端三状态文件缺少日期或三状态列：{list(frame.columns)}")
    result = frame.loc[:, [date_column, state_column]].rename(
        columns={date_column: "实际执行日", state_column: "三状态_远端"}
    ).copy()
    result["实际执行日"] = _parse_dates(result["实际执行日"])
    result["三状态_远端"] = pd.to_numeric(result["三状态_远端"], errors="raise").astype("int64")
    if result["实际执行日"].isna().any():
        raise ValueError("远端三状态文件存在无法解析的日期")
    if result["实际执行日"].duplicated().any():
        raise ValueError("远端三状态文件存在重复日期")
    if not result["三状态_远端"].isin([-1, 0, 1]).all():
        raise ValueError("远端三状态文件包含非法状态值")
    return result.sort_values("实际执行日").reset_index(drop=True)


def _resolve_remote_three_state(explicit: str | Path | None) -> Path:
    configured = explicit if explicit is not None else os.environ.get(
        "REMOTE_THREE_STATE_PATH", DEFAULT_REMOTE_THREE_STATE
    )
    return _resolve_absolute_path(configured, "远端三状态")


def _compare_full_eight_columns(
    generated: pd.DataFrame,
    local_reference: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]], bool]:
    merged = generated.merge(
        local_reference,
        on="实际执行日",
        how="outer",
        suffixes=("_生成", "_本地冻结"),
        indicator=True,
    )
    common = merged["_merge"].eq("both")
    field_metrics: dict[str, dict[str, int]] = {}
    match_columns: list[str] = []
    for column in SIGNAL_COLUMNS:
        match_name = f"match_{column}"
        merged[match_name] = (
            merged[f"{column}_生成"].eq(merged[f"{column}_本地冻结"]) & common
        )
        match_columns.append(match_name)
        field_metrics[column] = {
            "mismatch_rows_on_common_dates": int((common & ~merged[match_name]).sum()),
            "match_rows_on_common_dates": int((common & merged[match_name]).sum()),
        }
    merged["all_columns_match"] = merged[match_columns].all(axis=1)
    merged["row_status"] = merged.apply(
        lambda row: (
            "MATCH" if row["_merge"] == "both" and bool(row["all_columns_match"])
            else "MISMATCH" if row["_merge"] == "both"
            else str(row["_merge"]).upper()
        ),
        axis=1,
    )
    exact = bool(
        len(generated) == len(local_reference)
        and common.all()
        and merged.loc[common, "all_columns_match"].all()
    )
    return merged, field_metrics, exact


def _compare_remote_three_state(
    generated: pd.DataFrame,
    remote_three_state: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]], bool]:
    merged = generated.merge(
        remote_three_state,
        on="实际执行日",
        how="outer",
        indicator=True,
    )
    common = merged["_merge"].eq("both")
    merged["match_三状态"] = (
        merged["三状态"].eq(merged["三状态_远端"]) & common
    )
    merged["all_columns_match"] = merged["match_三状态"]
    merged["row_status"] = merged.apply(
        lambda row: (
            "MATCH" if row["_merge"] == "both" and bool(row["all_columns_match"])
            else "MISMATCH" if row["_merge"] == "both"
            else str(row["_merge"]).upper()
        ),
        axis=1,
    )
    metrics = {
        "三状态": {
            "mismatch_rows_on_common_dates": int((common & ~merged["match_三状态"]).sum()),
            "match_rows_on_common_dates": int((common & merged["match_三状态"]).sum()),
        }
    }
    exact = bool(
        len(generated) == len(remote_three_state)
        and common.all()
        and merged.loc[common, "match_三状态"].all()
    )
    return merged, metrics, exact


def compare_compact_output(
    generated_path: str | Path | None = None,
    local_reference_path: str | Path | None = None,
    three_state_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    output = _resolve_absolute_path(output_dir or OUTPUT_ROOT, "检查输出目录")
    output.mkdir(parents=True, exist_ok=True)
    generated_file = _resolve_absolute_path(
        generated_path or output / "最终执行日简表.csv", "生成八列表"
    )
    local_file = _resolve_absolute_path(local_reference_path or LOCAL_REFERENCE, "本地冻结八列表")
    remote_state_file = _resolve_remote_three_state(three_state_path)

    generated = _read_compact(generated_file, "生成八列表")
    local_reference = _read_compact(local_file, "本地冻结八列表")
    remote_three_state = _read_remote_three_state(remote_state_file)

    local_comparison, field_metrics, local_exact = _compare_full_eight_columns(
        generated, local_reference
    )
    state_comparison, state_metrics, state_exact = _compare_remote_three_state(
        generated, remote_three_state
    )
    local_comparison_path = output / "八列表逐日对比.csv"
    state_comparison_path = output / "三状态逐日对比.csv"
    local_comparison.to_csv(local_comparison_path, index=False, encoding="utf-8-sig")
    state_comparison.to_csv(state_comparison_path, index=False, encoding="utf-8-sig")
    local_comparison.loc[local_comparison["row_status"].ne("MATCH")].head(50).to_csv(
        output / "八列表不一致示例.csv", index=False, encoding="utf-8-sig"
    )
    state_comparison.loc[state_comparison["row_status"].ne("MATCH")].head(50).to_csv(
        output / "三状态不一致示例.csv", index=False, encoding="utf-8-sig"
    )

    local_common = local_comparison["_merge"].eq("both")
    state_common = state_comparison["_merge"].eq("both")
    conclusion = {
        "operation": "同时读取01生成八列表、包内本地冻结八列表和远端三状态；执行两组独立逐日对比",
        "generated_path": str(generated_file),
        "local_reference_path": str(local_file),
        "remote_three_state_path": str(remote_state_file),
        "eight_column_comparison_path": str(local_comparison_path),
        "three_state_comparison_path": str(state_comparison_path),
        "columns": COMPACT_COLUMNS,
        "generated_rows": int(len(generated)),
        "local_reference_rows": int(len(local_reference)),
        "remote_three_state_rows": int(len(remote_three_state)),
        "local_common_rows": int(local_common.sum()),
        "state_common_rows": int(state_common.sum()),
        "local_generated_only_rows": int(local_comparison["_merge"].eq("left_only").sum()),
        "local_reference_only_rows": int(local_comparison["_merge"].eq("right_only").sum()),
        "state_generated_only_rows": int(state_comparison["_merge"].eq("left_only").sum()),
        "state_baseline_only_rows": int(state_comparison["_merge"].eq("right_only").sum()),
        "generated_date_min": generated["实际执行日"].min().strftime("%Y-%m-%d") if len(generated) else None,
        "generated_date_max": generated["实际执行日"].max().strftime("%Y-%m-%d") if len(generated) else None,
        "local_reference_date_min": local_reference["实际执行日"].min().strftime("%Y-%m-%d") if len(local_reference) else None,
        "local_reference_date_max": local_reference["实际执行日"].max().strftime("%Y-%m-%d") if len(local_reference) else None,
        "remote_three_state_date_min": remote_three_state["实际执行日"].min().strftime("%Y-%m-%d") if len(remote_three_state) else None,
        "remote_three_state_date_max": remote_three_state["实际执行日"].max().strftime("%Y-%m-%d") if len(remote_three_state) else None,
        "field_metrics": field_metrics,
        "state_metrics": state_metrics,
        "all_dates_and_signals_match": local_exact,
        "all_dates_and_three_state_match": state_exact,
        "success": bool(local_exact and state_exact),
        "meaning": "success=True 同时表示生成八列表与包内本地冻结八列表八列完全一致，且生成八列表中的三状态与远端三状态按实际执行日完全一致。",
    }
    conclusion_path = output / "八列表一致性结论.json"
    conclusion_path.write_text(
        json.dumps(conclusion, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(conclusion, ensure_ascii=False, indent=2, default=_json_default))
    return conclusion


def main() -> None:
    parser = argparse.ArgumentParser(description="同时比较本地冻结八列表和远端三状态")
    parser.add_argument("--generated", default=None, help="可选：01生成的八列表绝对路径")
    parser.add_argument("--local", default=None, help="可选：包内本地冻结八列表绝对路径")
    parser.add_argument("--three-state", default=None, help="可选：远端三状态绝对路径")
    parser.add_argument("--output", default=None, help="可选：对比输出目录绝对路径")
    args = parser.parse_args()
    compare_compact_output(args.generated, args.local, args.three_state, args.output)


if __name__ == "__main__":
    main()
