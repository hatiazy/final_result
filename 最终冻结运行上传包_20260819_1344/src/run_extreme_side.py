from __future__ import annotations

"""Run one frozen pure-spot extreme-event side in an isolated process."""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "packages" / "extreme" / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from company_pool_runner import run_company_side  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("side", choices=["down", "up"])
    parser.add_argument("spot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.spot = args.spot.expanduser()
    args.output = args.output.expanduser()
    if not args.spot.is_absolute() or not args.output.is_absolute():
        raise ValueError("大涨/大跌运行入口的现货路径和输出目录都必须是绝对路径")
    args.spot = args.spot.resolve()
    args.output = args.output.resolve()
    if not args.spot.is_file():
        raise FileNotFoundError(f"现货输入不存在：{args.spot}")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["EXTREME_RUNTIME_OUTPUT_DIR"] = str(args.output)
    result = run_company_side(args.side, input_path=args.spot, progress=True)
    if result.get("status") != "COMPLETED":
        raise RuntimeError(f"{args.side} 侧没有完成冻结: {result.get('status')}")

    plot = pd.DataFrame(result["periods"]["plot_data"])
    required = ["date", "close", "score", "predicted", "actual_extreme", "correct", "direction_correct", "o2o_bp", "signed_o2o_bp", "phase"]
    missing = sorted(set(required) - set(plot.columns))
    if missing:
        raise ValueError(f"{args.side} 侧逐日结果缺少列: {missing}")
    daily = plot.loc[:, required].copy()
    daily["predicted"] = daily["predicted"].astype(int)
    daily["actual_extreme"] = daily["actual_extreme"].astype(int)
    daily["correct"] = daily["correct"].astype(int)
    daily["direction_correct"] = daily["direction_correct"].astype(int)
    daily.to_csv(args.output / f"remote_{args.side}_extreme_daily.csv", index=False, encoding="utf-8-sig")

    # The historical daily file is intentionally limited to rows with a
    # realized O2O label.  The runtime feed must also contain the latest
    # unlabelled prediction, because that is the signal for the next open.
    prediction = pd.DataFrame(result["periods"]["prediction_data"])
    prediction_required = ["date", "score", "predicted", "phase"]
    missing_prediction = sorted(set(prediction_required) - set(prediction.columns))
    if missing_prediction:
        raise ValueError(f"{args.side} 侧运行预测结果缺少列: {missing_prediction}")
    prediction = prediction.loc[:, prediction_required].copy()
    prediction["predicted"] = prediction["predicted"].astype(int)
    prediction.to_csv(
        args.output / f"remote_{args.side}_extreme_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "status": result["status"],
        "side": result["side"],
        "pool_rule_id": result.get("pool_rule_id"),
        "selection_rule": result.get("selection_rule"),
        "test_used_for_selection": result.get("test_used_for_selection"),
        "input_audit": result.get("input_audit"),
        "pool": result.get("pool"),
        "freeze": result.get("freeze"),
        "periods": {
            key: value
            for key, value in result.get("periods", {}).items()
            if key not in {"plot_data", "prediction_data"}
        },
        "daily_rows": int(len(daily)),
        "daily_date_min": str(daily["date"].min()) if len(daily) else None,
        "daily_date_max": str(daily["date"].max()) if len(daily) else None,
        "prediction_rows": int(len(prediction)),
        "prediction_date_min": str(prediction["date"].min()) if len(prediction) else None,
        "prediction_date_max": str(prediction["date"].max()) if len(prediction) else None,
        "prediction_date_role": "formation date; the compact exporter maps it to the next actual execution date",
    }
    (args.output / f"remote_{args.side}_extreme_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"[大涨大跌-{args.side}] output={args.output / f'remote_{args.side}_extreme_daily.csv'}", flush=True)
    print(f"[大涨大跌-{args.side}] rows={len(daily)} date={summary['daily_date_min']} -> {summary['daily_date_max']}", flush=True)


if __name__ == "__main__":
    main()
