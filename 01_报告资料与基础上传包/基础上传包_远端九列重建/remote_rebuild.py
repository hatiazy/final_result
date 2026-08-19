"""Remote-only rebuild of the merged nine-column signal view.

The two frozen company packages are isolated in subprocesses because both
contain top-level modules such as ``pool_registry``.  The parent process only
merges their five-column exports and performs deterministic comparisons.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_SPOT_PATTERN = "/home/hzy/cta/IC数据更新*最终固化版/现货最终版/CSI500_SPOT_md_eod_raw*最终版.parquet"
DEFAULT_THREE_STATE_PATH = "/home/hzy/cta/三状态冻结/IC_1545_three_state_and_downside_warning.csv"
REFERENCE_NINE_PATH = PACKAGE_ROOT / "reference" / "本地九列参考结果.csv"

NINE_COLUMNS = [
    "date",
    "three_state",
    "minus_exit_signal",
    "plus_exit_signal",
    "reversal_final_three_state",
    "minus_entry_signal",
    "plus_entry_signal",
    "transfer_final_three_state",
    "combined_final_three_state",
]

REVERSAL_COLUMNS = [
    "date",
    "three_state",
    "minus_exit_signal",
    "plus_exit_signal",
    "final_three_state",
]
TRANSFER_COLUMNS = [
    "date",
    "three_state",
    "minus_entry_signal",
    "plus_entry_signal",
    "final_three_state",
]


def _resolve_one(raw: str | Path, label: str) -> Path:
    text = str(raw).strip()
    if any(token in text for token in "*?["):
        matches = sorted(Path(item).expanduser().resolve() for item in Path("/").glob(text.lstrip("/")))
        matches = [item for item in matches if item.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(f"{label} 通配路径必须唯一匹配一个文件；matches={[str(item) for item in matches]}")
        return matches[0]
    path = Path(text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在: {path}")
    return path


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    raise ValueError(f"只支持 Parquet/CSV/TSV 输入: {path}")


def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}", na=False)
    dates = pd.to_datetime(values, errors="coerce")
    if compact.any():
        dates.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    dates = dates.dt.normalize()
    if dates.isna().any():
        raise ValueError("日期列包含无法解析的值")
    return dates


def _pick_column(frame: pd.DataFrame, candidates: list[str], role: str) -> str:
    exact = {str(column): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(f"输入缺少 {role} 列；候选={candidates}；实际={list(frame.columns)}")


def _load_five_column(path: Path, expected: list[str], label: str) -> pd.DataFrame:
    frame = _read_table(path)
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} 缺少列: {missing}; 实际列={list(frame.columns)}")
    frame = frame.loc[:, expected].copy()
    frame["date"] = _parse_dates(frame["date"])
    if frame["date"].duplicated().any():
        raise ValueError(f"{label} date 存在重复")
    for column in expected[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int8")
    if not frame["three_state"].isin([-1, 0, 1]).all():
        raise ValueError(f"{label} three_state 不在 -1/0/1 内")
    signal_columns = [column for column in expected if column.endswith("_signal")]
    for column in signal_columns:
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"{label} {column} 不在 0/1 内")
    return frame.sort_values("date").reset_index(drop=True)


def _load_remote_three_state(path: Path) -> pd.DataFrame:
    frame = _read_table(path)
    date_column = _pick_column(frame, ["date", "effective_date", "trade_dt", "trade_date", "formation_date"], "远端三状态日期")
    state_column = _pick_column(frame, ["three_state", "state", "base_state", "final_three_state"], "远端三状态")
    result = frame.loc[:, [date_column, state_column]].rename(columns={date_column: "date", state_column: "remote_three_state"}).copy()
    result["date"] = _parse_dates(result["date"])
    result["remote_three_state"] = pd.to_numeric(result["remote_three_state"], errors="raise").astype("int8")
    if result["date"].duplicated().any():
        raise ValueError("远端三状态 date 存在重复")
    if not result["remote_three_state"].isin([-1, 0, 1]).all():
        raise ValueError("远端三状态不在 -1/0/1 内")
    return result.sort_values("date").reset_index(drop=True)


def _apply_reversal(base: pd.Series, minus: pd.Series, plus: pd.Series) -> pd.Series:
    output = base.to_numpy(dtype=np.int8).copy()
    output[(base.to_numpy() == -1) & (minus.to_numpy(dtype=bool))] = 0
    output[(base.to_numpy() == 1) & (plus.to_numpy(dtype=bool))] = 0
    return pd.Series(output, index=base.index, dtype="int8")


def _apply_transfer(base: pd.Series, minus: pd.Series, plus: pd.Series) -> pd.Series:
    output = base.to_numpy(dtype=np.int8).copy()
    conflict = minus.to_numpy(dtype=bool) & plus.to_numpy(dtype=bool)
    base_values = base.to_numpy(dtype=np.int8)
    output[(base_values == 0) & minus.to_numpy(dtype=bool) & ~conflict] = -1
    output[(base_values == 0) & plus.to_numpy(dtype=bool) & ~conflict] = 1
    return pd.Series(output, index=base.index, dtype="int8")


def build_nine_columns(reversal: pd.DataFrame, transfer: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not reversal["date"].equals(transfer["date"]):
        raise ValueError("1545 与长0五列结果的 date 网格不一致")
    if not reversal["three_state"].equals(transfer["three_state"]):
        raise ValueError("1545 与长0五列结果的 three_state 逐日不一致")
    base = reversal["three_state"].astype("int8")
    reversal_final = _apply_reversal(base, reversal["minus_exit_signal"], reversal["plus_exit_signal"])
    transfer_final = _apply_transfer(base, transfer["minus_entry_signal"], transfer["plus_entry_signal"])
    combined_final = base.copy()
    combined_final = _apply_reversal(combined_final, reversal["minus_exit_signal"], reversal["plus_exit_signal"])
    combined_final = _apply_transfer(combined_final, transfer["minus_entry_signal"], transfer["plus_entry_signal"])
    result = pd.DataFrame(
        {
            "date": reversal["date"],
            "three_state": base,
            "minus_exit_signal": reversal["minus_exit_signal"],
            "plus_exit_signal": reversal["plus_exit_signal"],
            "reversal_final_three_state": reversal_final,
            "minus_entry_signal": transfer["minus_entry_signal"],
            "plus_entry_signal": transfer["plus_entry_signal"],
            "transfer_final_three_state": transfer_final,
            "combined_final_three_state": combined_final,
        }
    )
    audit = {
        "rows": int(len(result)),
        "date_min": result["date"].min(),
        "date_max": result["date"].max(),
        "columns": list(result.columns),
        "base_state_counts": {str(value): int((base == value).sum()) for value in (-1, 0, 1)},
        "signal_counts": {column: int(result[column].sum()) for column in result.columns if column.endswith("_signal")},
        "entry_conflict_days": int((result["minus_entry_signal"].eq(1) & result["plus_entry_signal"].eq(1) & result["three_state"].eq(0)).sum()),
        "source_final_reproduction": {
            "reversal": bool(reversal_final.equals(reversal["final_three_state"])),
            "transfer": bool(transfer_final.equals(transfer["final_three_state"])),
        },
    }
    return result, audit


def _compare_frames(left: pd.DataFrame, right: pd.DataFrame, columns: list[str], left_prefix: str, right_prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    left = left.copy()
    right = right.copy()
    left["date"] = _parse_dates(left["date"])
    right["date"] = _parse_dates(right["date"])
    left = left.drop_duplicates("date").sort_values("date")
    right = right.drop_duplicates("date").sort_values("date")
    merged = left.merge(right, on="date", how="outer", suffixes=(f"_{left_prefix}", f"_{right_prefix}"), indicator=True)
    compare_columns: list[str] = []
    for column in columns:
        if column == "date":
            continue
        lcol = f"{column}_{left_prefix}"
        rcol = f"{column}_{right_prefix}"
        if lcol not in merged.columns or rcol not in merged.columns:
            compare_columns.append(column)
            continue
        equal = merged[lcol].eq(merged[rcol])
        equal = equal.fillna(False) | (merged[lcol].isna() & merged[rcol].isna())
        merged[f"match_{column}"] = equal
        compare_columns.append(column)
    match_columns = [f"match_{column}" for column in compare_columns if f"match_{column}" in merged.columns]
    merged["all_columns_match"] = merged[match_columns].all(axis=1) if match_columns else False
    merged["row_status"] = np.where(merged["_merge"].eq("both"), np.where(merged["all_columns_match"], "MATCH", "MISMATCH"), merged["_merge"].str.upper())
    common = merged["_merge"].eq("both")
    metrics = {
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "common_rows": int(common.sum()),
        "left_only_rows": int(merged["_merge"].eq("left_only").sum()),
        "right_only_rows": int(merged["_merge"].eq("right_only").sum()),
        "mismatch_rows_on_common_dates": int((common & ~merged["all_columns_match"]).sum()),
        "all_common_rows_exact": bool(common.any() and (common & merged["all_columns_match"]).sum() == common.sum()),
        "no_date_or_value_difference": bool(len(merged) > 0 and merged["_merge"].eq("both").all() and merged["all_columns_match"].all()),
    }
    return merged, metrics


def _stream_child(script: Path, env: dict[str, str], log_path: Path, label: str) -> None:
    command = [sys.executable, str(script)]
    print(f"[{label}] 启动: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=PACKAGE_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{label} 子进程失败，退出码={return_code}；完整日志={log_path}")


def run_remote_audit(
    spot_path: str | Path | None = None,
    three_state_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    reference_nine_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run both frozen engines, merge nine columns, and write exact audits."""

    spot_raw = spot_path or os.environ.get("COMPANY_SPOT_PATH", DEFAULT_SPOT_PATTERN)
    baseline_raw = three_state_path or os.environ.get("REMOTE_THREE_STATE_PATH", DEFAULT_THREE_STATE_PATH)
    output_raw = output_dir or os.environ.get("REMOTE_OUTPUT_DIR", str(PACKAGE_ROOT / "远端输出"))
    reference_raw = reference_nine_path or os.environ.get("LOCAL_REFERENCE_NINE_PATH", str(REFERENCE_NINE_PATH))
    spot = _resolve_one(spot_raw, "远端现货")
    baseline_path = _resolve_one(baseline_raw, "远端三状态")
    output = Path(output_raw).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = _resolve_one(reference_raw, "本地九列参考结果")

    env = os.environ.copy()
    env["1545_SPOT_PATH"] = str(spot)
    env["COMPANY_SPOT_PATH"] = str(spot)
    env["REMOTE_OUTPUT_DIR"] = str(output)
    env["PYTHONUNBUFFERED"] = "1"
    _stream_child(PACKAGE_ROOT / "run_1545_remote.py", env, output / "run_1545.log", "1545")
    _stream_child(PACKAGE_ROOT / "run_long0_remote.py", env, output / "run_long0.log", "长0")

    reversal_path = output / "remote_1545_five_columns.csv"
    transfer_path = output / "remote_long0_five_columns.csv"
    reversal = _load_five_column(reversal_path, REVERSAL_COLUMNS, "远端1545五列")
    transfer = _load_five_column(transfer_path, TRANSFER_COLUMNS, "远端长0五列")
    nine, merge_audit = build_nine_columns(reversal, transfer)
    nine_path = output / "remote_combined_nine_columns.csv"
    nine.to_csv(nine_path, index=False, encoding="utf-8-sig")

    remote_state = _load_remote_three_state(baseline_path)
    generated_base = nine[["date", "three_state"]].copy()
    generated_1545 = reversal[["date", "three_state"]].rename(columns={"three_state": "generated_1545_three_state"})
    generated_long0 = transfer[["date", "three_state"]].rename(columns={"three_state": "generated_long0_three_state"})
    base_compare = generated_base.rename(columns={"three_state": "generated_nine_three_state"}).merge(remote_state, on="date", how="outer", indicator=True)
    base_compare = base_compare.merge(generated_1545, on="date", how="outer").merge(generated_long0, on="date", how="outer")
    base_compare["match_generated_1545_vs_remote"] = base_compare["generated_1545_three_state"].eq(base_compare["remote_three_state"])
    base_compare["match_generated_long0_vs_remote"] = base_compare["generated_long0_three_state"].eq(base_compare["remote_three_state"])
    base_compare["match_generated_1545_vs_long0"] = base_compare["generated_1545_three_state"].eq(base_compare["generated_long0_three_state"])
    base_compare["match_nine_vs_remote"] = base_compare["generated_nine_three_state"].eq(base_compare["remote_three_state"])
    base_compare["any_difference"] = ~(
        base_compare[["match_generated_1545_vs_remote", "match_generated_long0_vs_remote", "match_generated_1545_vs_long0", "match_nine_vs_remote"]].all(axis=1)
    )
    base_compare_path = output / "三状态逐日对比.csv"
    base_compare.to_csv(base_compare_path, index=False, encoding="utf-8-sig")
    base_common = base_compare["_merge"].eq("both")
    state_audit = {
        "remote_three_state_path": str(baseline_path),
        "generated_1545_rows": int(len(reversal)),
        "generated_long0_rows": int(len(transfer)),
        "remote_rows": int(len(remote_state)),
        "common_rows_with_nine_grid": int(base_common.sum()),
        "remote_only_rows": int((base_compare["_merge"] == "right_only").sum()),
        "generated_only_rows": int((base_compare["_merge"] == "left_only").sum()),
        "generated_1545_vs_remote_mismatch_rows": int((base_common & ~base_compare["match_generated_1545_vs_remote"]).sum()),
        "generated_long0_vs_remote_mismatch_rows": int((base_common & ~base_compare["match_generated_long0_vs_remote"]).sum()),
        "generated_1545_vs_long0_mismatch_rows": int((base_common & ~base_compare["match_generated_1545_vs_long0"]).sum()),
        "all_three_state_checks_exact": bool(base_common.any() and not base_compare.loc[base_common, "any_difference"].any() and base_compare["_merge"].eq("both").all()),
        "generated_1545_counts": {str(value): int((reversal["three_state"] == value).sum()) for value in (-1, 0, 1)},
        "generated_long0_counts": {str(value): int((transfer["three_state"] == value).sum()) for value in (-1, 0, 1)},
        "remote_counts": {str(value): int((remote_state["remote_three_state"] == value).sum()) for value in (-1, 0, 1)},
    }

    local_reversal_reference = _load_five_column(
        PACKAGE_ROOT / "reference" / "本地1545五列参考结果.csv",
        REVERSAL_COLUMNS,
        "本地1545五列参考结果",
    )
    local_transfer_reference = _load_five_column(
        PACKAGE_ROOT / "reference" / "本地长0五列参考结果.csv",
        TRANSFER_COLUMNS,
        "本地长0五列参考结果",
    )
    local_reversal_compare, local_reversal_audit = _compare_frames(
        reversal,
        local_reversal_reference,
        REVERSAL_COLUMNS,
        "remote",
        "local",
    )
    local_transfer_compare, local_transfer_audit = _compare_frames(
        transfer,
        local_transfer_reference,
        TRANSFER_COLUMNS,
        "remote",
        "local",
    )
    local_reversal_compare_path = output / "本地1545五列逐日对比.csv"
    local_transfer_compare_path = output / "本地长0五列逐日对比.csv"
    local_reversal_compare.to_csv(local_reversal_compare_path, index=False, encoding="utf-8-sig")
    local_transfer_compare.to_csv(local_transfer_compare_path, index=False, encoding="utf-8-sig")

    reference_frame = _read_table(reference)
    missing_reference = sorted(set(NINE_COLUMNS) - set(reference_frame.columns))
    if missing_reference:
        raise ValueError(f"本地九列参考结果缺少列: {missing_reference}")
    reference_frame = reference_frame.loc[:, NINE_COLUMNS].copy()
    local_compare, local_audit = _compare_frames(nine, reference_frame, NINE_COLUMNS, "remote", "local")
    local_compare_path = output / "本地九列逐日对比.csv"
    local_compare.to_csv(local_compare_path, index=False, encoding="utf-8-sig")

    manifest = {
        "package": "远端九列重建上传包_20260817",
        "spot_path": str(spot),
        "remote_three_state_path": str(baseline_path),
        "reference_nine_path": str(reference),
        "output_dir": str(output),
        "generated_files": {
            "remote_1545_five_columns": str(reversal_path),
            "remote_long0_five_columns": str(transfer_path),
            "remote_combined_nine_columns": str(nine_path),
            "three_state_daily_comparison": str(base_compare_path),
            "local_1545_five_daily_comparison": str(local_reversal_compare_path),
            "local_long0_five_daily_comparison": str(local_transfer_compare_path),
            "local_nine_daily_comparison": str(local_compare_path),
        },
        "merge_audit": merge_audit,
        "state_audit": state_audit,
        "local_nine_audit": local_audit,
        "local_five_audits": {
            "1545": local_reversal_audit,
            "long0": local_transfer_audit,
        },
        "success": bool(
            state_audit["all_three_state_checks_exact"]
            and local_reversal_audit["no_date_or_value_difference"]
            and local_transfer_audit["no_date_or_value_difference"]
            and local_audit["no_date_or_value_difference"]
        ),
        "policy": "四信号均为 event-only：只改信号当天，后续日期恢复基础三状态；同日双入口冲突保留 0。",
    }
    print("\nREMOTE_NINE_COLUMN_AUDIT_END")
    print(f"success={manifest['success']}")
    print(f"all_three_state_checks_exact={state_audit['all_three_state_checks_exact']}")
    print(f"local_1545_five_exact={local_reversal_audit['no_date_or_value_difference']}")
    print(f"local_long0_five_exact={local_transfer_audit['no_date_or_value_difference']}")
    print(f"local_nine_exact={local_audit['no_date_or_value_difference']}")
    return {
        "nine": nine,
        "state_compare": base_compare,
        "local_1545_compare": local_reversal_compare,
        "local_long0_compare": local_transfer_compare,
        "local_compare": local_compare,
        "manifest": manifest,
    }


if __name__ == "__main__":
    run_remote_audit()
