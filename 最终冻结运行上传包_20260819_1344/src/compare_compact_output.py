from __future__ import annotations

"""Compare the remote six-column output with the bundled local freeze reference."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PACKAGE_ROOT / "runtime_outputs"
LOCAL_REFERENCE = PACKAGE_ROOT / "expected" / "local_freeze" / "最终执行日简表_本地冻结参考.csv"
COMPACT_COLUMNS = ["实际执行日", "三状态", "+1反转", "-1反转", "大涨", "大跌"]
SIGNAL_COLUMNS = ["三状态", "+1反转", "-1反转", "大涨", "大跌"]


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


def _read_compact(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in COMPACT_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in COMPACT_COLUMNS]
    if missing or extra:
        raise ValueError(f"{label}列不符合六列表；缺少={missing}；多出={extra}")
    frame = frame.loc[:, COMPACT_COLUMNS].copy()
    frame["实际执行日"] = pd.to_datetime(frame["实际执行日"], errors="coerce").dt.normalize()
    if frame["实际执行日"].isna().any():
        raise ValueError(f"{label}存在无法解析的实际执行日")
    if frame["实际执行日"].duplicated().any():
        duplicates = frame.loc[frame["实际执行日"].duplicated(), "实际执行日"].dt.strftime("%Y-%m-%d").tolist()[:10]
        raise ValueError(f"{label}存在重复实际执行日：{duplicates}")
    for column in SIGNAL_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    invalid_state = ~frame["三状态"].isin([-1, 0, 1])
    invalid_binary = ~frame[["+1反转", "-1反转", "大涨", "大跌"]].isin([0, 1]).all(axis=1)
    if invalid_state.any() or invalid_binary.any():
        raise ValueError(f"{label}包含非法状态值或信号值")
    return frame.sort_values("实际执行日").reset_index(drop=True)


def _resolve_absolute_path(raw: str | Path, label: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label}必须使用绝对路径：{raw}")
    return path.resolve()


def compare_compact_output(
    remote_path: str | Path | None = None,
    local_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    output = _resolve_absolute_path(output_dir or OUTPUT_ROOT, "检查输出目录")
    output.mkdir(parents=True, exist_ok=True)
    remote_file = _resolve_absolute_path(remote_path or output / "最终执行日简表.csv", "远端六列表")
    local_file = _resolve_absolute_path(local_path or LOCAL_REFERENCE, "本地冻结六列表")
    remote = _read_compact(remote_file, "远端六列表")
    local = _read_compact(local_file, "本地冻结六列表")

    merged = remote.merge(
        local,
        on="实际执行日",
        how="outer",
        suffixes=("_远端", "_本地"),
        indicator=True,
    )
    common = merged["_merge"].eq("both")
    match_columns: list[str] = []
    field_metrics: dict[str, dict[str, Any]] = {}
    for column in SIGNAL_COLUMNS:
        generated = merged[f"{column}_远端"]
        reference = merged[f"{column}_本地"]
        match = generated.eq(reference) & common
        match_name = f"match_{column}"
        merged[match_name] = match
        match_columns.append(match_name)
        field_metrics[column] = {
            "mismatch_rows_on_common_dates": int((common & ~match).sum()),
            "match_rows_on_common_dates": int((common & match).sum()),
        }

    merged["all_columns_match"] = merged[match_columns].all(axis=1) if match_columns else False
    merged["row_status"] = np.where(
        merged["_merge"].eq("both"),
        np.where(merged["all_columns_match"], "MATCH", "MISMATCH"),
        merged["_merge"].str.upper(),
    )
    comparison_path = output / "六列表逐日对比.csv"
    merged.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    all_common_match = bool(common.all() and merged.loc[common, "all_columns_match"].all())
    conclusion = {
        "operation": "读取远端六列表，与包内本地冻结六列表逐日逐列比较",
        "remote_path": str(remote_file),
        "local_reference_path": str(local_file),
        "comparison_path": str(comparison_path),
        "columns": COMPACT_COLUMNS,
        "remote_rows": int(len(remote)),
        "local_rows": int(len(local)),
        "common_rows": int(common.sum()),
        "remote_only_rows": int(merged["_merge"].eq("left_only").sum()),
        "local_only_rows": int(merged["_merge"].eq("right_only").sum()),
        "remote_date_min": remote["实际执行日"].min().strftime("%Y-%m-%d") if len(remote) else None,
        "remote_date_max": remote["实际执行日"].max().strftime("%Y-%m-%d") if len(remote) else None,
        "local_date_min": local["实际执行日"].min().strftime("%Y-%m-%d") if len(local) else None,
        "local_date_max": local["实际执行日"].max().strftime("%Y-%m-%d") if len(local) else None,
        "field_metrics": field_metrics,
        "all_dates_and_signals_match": all_common_match,
        "success": all_common_match and len(remote) == len(local),
        "meaning": "success=True 表示六列、所有共同实际执行日和全部信号值完全一致；否则查看六列表逐日对比.csv 的 row_status 和 match_* 列。",
    }
    conclusion_path = output / "六列表一致性结论.json"
    conclusion_path.write_text(
        json.dumps(conclusion, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )

    mismatch_sample = merged.loc[merged["row_status"].ne("MATCH")].head(50).copy()
    mismatch_sample.to_csv(output / "六列表不一致示例.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(conclusion, ensure_ascii=False, indent=2, default=_json_default))
    return conclusion


def main() -> None:
    parser = argparse.ArgumentParser(description="比较远端六列表和本地冻结六列表")
    parser.add_argument("--remote", default=None, help="可选：远端六列表路径")
    parser.add_argument("--local", default=None, help="可选：本地冻结六列表路径")
    parser.add_argument("--output", default=None, help="可选：对比输出目录")
    args = parser.parse_args()
    compare_compact_output(args.remote, args.local, args.output)


if __name__ == "__main__":
    main()
