from __future__ import annotations

"""Compare the current remote run outputs with packaged local references.

This is the authoritative local/remote comparison for the standalone package.
It does not generate signals and it does not read any remote baseline data:
the remote side is the output produced by 01--06 in the current package, and
the local side is the immutable CSV reference under ``本地结果对照``.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PACKAGE_ROOT / "本地结果对照"
DEFAULT_OUTPUT = PACKAGE_ROOT / "runtime_outputs_07_consistency"


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


def _absolute(raw: str | Path, label: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} 必须使用绝对路径：{raw}")
    return path.resolve()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    date_tokens = ("日期", "date", "trade_dt", "trade_date", "形成日", "执行日")
    for column in result.columns:
        name = str(column).lower()
        series = result[column]
        if any(token.lower() in name for token in date_tokens):
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            if parsed.notna().sum() == len(series):
                result[column] = parsed.dt.strftime("%Y-%m-%d")
                continue
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() == len(series) and len(series) > 0:
            result[column] = numeric.astype(float)
        else:
            result[column] = series.astype("string")
    return result


def _sort_for_compare(frame: pd.DataFrame) -> pd.DataFrame:
    date_columns = [
        column
        for column in frame.columns
        if any(token.lower() in str(column).lower() for token in ("日期", "date", "trade_dt", "trade_date"))
    ]
    if date_columns:
        return frame.sort_values(date_columns, kind="stable", na_position="last").reset_index(drop=True)
    return frame.reset_index(drop=True)


def _compare_table(label: str, remote_path: Path, local_path: Path, output_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "table": label,
        "remote_path": str(remote_path),
        "local_reference_path": str(local_path),
        "status": "PENDING",
        "remote_rows": None,
        "local_rows": None,
        "remote_columns": None,
        "local_columns": None,
        "mismatch_cells": None,
        "mismatch_rows": None,
        "detail_path": None,
    }
    if not remote_path.is_file():
        result["status"] = "MISSING_REMOTE"
        return result
    if not local_path.is_file():
        result["status"] = "MISSING_LOCAL_REFERENCE"
        return result

    remote = _sort_for_compare(_normalise(_read_csv(remote_path)))
    local = _sort_for_compare(_normalise(_read_csv(local_path)))
    result["remote_rows"] = int(len(remote))
    result["local_rows"] = int(len(local))
    result["remote_columns"] = [str(column) for column in remote.columns]
    result["local_columns"] = [str(column) for column in local.columns]

    if list(remote.columns) != list(local.columns):
        result["status"] = "COLUMN_MISMATCH"
        return result

    row_count = min(len(remote), len(local))
    mismatch_mask = pd.DataFrame(False, index=range(row_count), columns=remote.columns)
    for column in remote.columns:
        left = remote[column].iloc[:row_count].reset_index(drop=True)
        right = local[column].iloc[:row_count].reset_index(drop=True)
        left_na = left.isna()
        right_na = right.isna()
        both_na = left_na & right_na
        numeric_left = pd.to_numeric(left, errors="coerce")
        numeric_right = pd.to_numeric(right, errors="coerce")
        numeric_pair = numeric_left.notna() & numeric_right.notna()
        equal_numeric = pd.Series(False, index=left.index)
        if numeric_pair.any():
            equal_numeric.loc[numeric_pair] = np.isclose(
                numeric_left.loc[numeric_pair].to_numpy(dtype=float),
                numeric_right.loc[numeric_pair].to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-8,
                equal_nan=True,
            )
        equal_text = left.astype("string").eq(right.astype("string"))
        mismatch_mask[column] = ~(both_na | equal_numeric | (~numeric_pair & equal_text))

    mismatch_rows = mismatch_mask.any(axis=1)
    result["mismatch_cells"] = int(mismatch_mask.to_numpy().sum())
    result["mismatch_rows"] = int(mismatch_rows.sum()) + abs(len(remote) - len(local))

    if result["mismatch_cells"] == 0 and len(remote) == len(local):
        result["status"] = "PASS"
        return result

    result["status"] = "VALUE_MISMATCH" if result["mismatch_cells"] else "ROW_COUNT_MISMATCH"
    if row_count:
        detail = pd.concat(
            [remote.iloc[:row_count].add_suffix("_远端"), local.iloc[:row_count].add_suffix("_本地")],
            axis=1,
        )
        detail.insert(0, "mismatch_row", mismatch_rows.to_numpy())
        detail = detail.loc[detail["mismatch_row"]].head(200)
        detail_path = output_dir / f"不一致明细_{label}.csv"
        detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
        result["detail_path"] = str(detail_path)
    return result


def _table_pairs(package_root: Path, reference_root: Path) -> list[tuple[str, Path, Path]]:
    remote_04 = package_root / "runtime_outputs_04_returns"
    local_04 = reference_root / "02_O2O加算收益与持有段计数分析"
    remote_05 = package_root / "runtime_outputs_05_yearly"
    local_05 = reference_root / "03_逐年分析与2026表现解释"
    remote_06 = package_root / "runtime_outputs_06_mechanism" / "tables"
    local_06 = reference_root / "04_历史相似行情与机制分解分析" / "tables"
    pairs: list[tuple[str, Path, Path]] = [
        (
            "01_事件八列",
            package_root / "runtime_outputs_holding_period" / "最终执行日简表.csv",
            reference_root / "01_含持有期零段反转八列表" / "最终执行日简表.csv",
        ),
        (
            "01_含持有期八列",
            package_root / "runtime_outputs_holding_period" / "含持有期八列表.csv",
            reference_root / "01_含持有期零段反转八列表" / "含持有期八列表.csv",
        ),
        (
            "01_连续诊断输出",
            package_root / "runtime_outputs_holding_period" / "连续诊断输出.csv",
            reference_root / "01_含持有期零段反转八列表" / "连续诊断输出.csv",
        ),
    ]
    for filename in (
        "O2O加算逐日收益与状态.csv",
        "O2O加算风险指标.csv",
        "持有段明细.csv",
        "持有段计数_按系列.csv",
        "年度段计数与信号计数_原始.csv",
    ):
        pairs.append((f"02_{filename[:-4]}", remote_04 / filename, local_04 / filename))
    for filename in (
        "逐年_信号与持有段计数.csv",
        "逐年_加入四个反转分析.csv",
        "逐年_原始三状态分析.csv",
        "2026_逐日表现分解.csv",
        "2026_逐月表现分解.csv",
    ):
        pairs.append((f"03_{filename[:-4]}", remote_05 / filename, local_05 / filename))
    for filename in (
        "历史相似行情候选_60日.csv",
        "历史相似行情特征对比_z值.csv",
        "机制诊断逐日面板.csv",
        "滚动相对与固定锚定代理_年度对比.csv",
        "年度行情机制与收益对比.csv",
    ):
        pairs.append((f"04_{filename[:-4]}", remote_06 / filename, local_06 / filename))
    return pairs


def run_consistency_check(
    package_root: str | Path | None = None,
    reference_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    package = _absolute(package_root or PACKAGE_ROOT, "包根目录")
    reference = _absolute(reference_root or package / "本地结果对照", "本地结果对照目录")
    output = _absolute(output_dir or package / "runtime_outputs_07_consistency", "07 输出目录")
    output.mkdir(parents=True, exist_ok=True)

    rows = [
        _compare_table(label, remote, local, output)
        for label, remote, local in _table_pairs(package, reference)
    ]
    summary = pd.DataFrame(rows)
    summary_path = output / "一致性检查摘要.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    success = bool(len(summary) > 0 and summary["status"].eq("PASS").all())
    conclusion = {
        "operation": "比较当前远端运行输出与包内本地结果对照；不读取远端基准文件，不生成信号",
        "package_root": str(package),
        "reference_root": str(reference),
        "output_dir": str(output),
        "table_count": int(len(summary)),
        "pass_count": int(summary["status"].eq("PASS").sum()),
        "failed_tables": summary.loc[summary["status"].ne("PASS"), "table"].tolist(),
        "success": success,
        "summary_path": str(summary_path),
        "rule": "日期、列名、行数完全一致；数值允许绝对误差1e-8和相对误差1e-10；缺失或不同截点不算一致",
    }
    conclusion_path = output / "一致性检查结论.json"
    conclusion_path.write_text(
        json.dumps(conclusion, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(conclusion, ensure_ascii=False, indent=2, default=_json_default))
    return conclusion


def main() -> None:
    parser = argparse.ArgumentParser(description="比较远端本次运行输出与包内本地结果对照")
    parser.add_argument("--package-root", default=None)
    parser.add_argument("--reference-root", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_consistency_check(args.package_root, args.reference_root, args.output)


if __name__ == "__main__":
    main()
