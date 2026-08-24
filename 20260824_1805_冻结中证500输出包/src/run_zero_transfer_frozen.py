from __future__ import annotations

"""Run the two frozen zero-segment transfer signals from one raw spot file.

This is a production entry point, not a candidate scanner.  It never reads a
local signal result, a precomputed state file, or a future return label.  The
two frozen thresholds and release thresholds are constants from the authorised
V38/V57 freeze; the score path is evaluated causally through the latest spot
row, so the newest formation day can produce the next execution-day signal.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "packages" / "zero_transfer" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from spot_panel import load_spot_panel  # noqa: E402
from zero_transfer.logic_features import compute_logic_scores  # noqa: E402


FROZEN_ZERO_TRANSFER: dict[str, dict[str, Any]] = {
    "down": {
        "direction": -1,
        "signal_column": "minus_entry_signal",
        "source_version": "V38",
        "candidate_id": "V38_down_s01_a1_q0.85_c1_H03",
        "core_logic_name": "成交量趋势突破确认",
        "method_key": "volume_breakout",
        "score_column": "score_01",
        "score_variant_number": 1,
        "score_variant": {"window": 4, "direction_or_recovery_lag": 1},
        "min_zero_age": 1,
        "entry_quantile": 0.85,
        "confirmation_days": 1,
        "threshold_from_development": 1.2294466376933055,
        "release_quantile": 0.70,
        "release_threshold": 0.6645986031220402,
        "holding_package": {
            "package_id": "H03",
            "min_hold_days": 3,
            "max_hold_days": 10,
            "release_quantile_gap": 0.15,
            "cooldown_days": 3,
        },
    },
    "up": {
        "direction": 1,
        "signal_column": "plus_entry_signal",
        "source_version": "V57",
        "candidate_id": "V57_up_s01_a1_q0.85_c1_H04",
        "core_logic_name": "Recurrence Quantification",
        "method_key": "rqa",
        "score_column": "score_01",
        "score_variant_number": 1,
        "score_variant": {"embedding_m": 2, "delay": 2, "reserve_variant": 1},
        "min_zero_age": 1,
        "entry_quantile": 0.85,
        "confirmation_days": 1,
        "threshold_from_development": 4.671980676328502,
        "release_quantile": 0.65,
        "release_threshold": 4.606280193236715,
        "holding_package": {
            "package_id": "H04",
            "min_hold_days": 5,
            "max_hold_days": 20,
            "release_quantile_gap": 0.20,
            "cooldown_days": 5,
        },
    },
}


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


def _resolve_absolute(raw: str | Path, label: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label}必须使用绝对路径：{raw}")
    return path.resolve()


def _resolve_spot(raw: str | Path | None) -> Path:
    configured = raw if raw is not None else os.environ.get("COMPANY_SPOT_PATH", "").strip()
    if configured:
        path = _resolve_absolute(configured, "COMPANY_SPOT_PATH")
    else:
        root = Path("/home/hzy/cta")
        matches = sorted(root.glob("IC数据更新*最终固化版/现货最终版/CSI500_SPOT_md_eod_raw*最终版.parquet"))
        if len(matches) != 1:
            raise FileNotFoundError(
                "无法唯一定位远端现货；请通过 COMPANY_SPOT_PATH 或 --spot 传入绝对路径，"
                f"默认匹配={root / 'IC数据更新*最终固化版/现货最终版/CSI500_SPOT_md_eod_raw*最终版.parquet'}"
            )
        path = matches[0].resolve()
    if not path.is_file():
        raise FileNotFoundError(f"远端现货不存在：{path}")
    return path


def _resolve_output(raw: str | Path | None) -> Path:
    configured = raw if raw is not None else os.environ.get(
        "UPLOAD_OUTPUT_DIR", str(PACKAGE_ROOT / "runtime_outputs")
    )
    output = _resolve_absolute(configured, "UPLOAD_OUTPUT_DIR")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _confirmed(above: np.ndarray, days: int) -> np.ndarray:
    if days <= 1:
        return above.copy()
    return (
        pd.Series(above.astype("int8"))
        .rolling(days, min_periods=days)
        .sum()
        .eq(days)
        .to_numpy()
        & above
    )


def _frozen_signal_path(
    panel: pd.DataFrame,
    score: np.ndarray,
    freeze: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply frozen entry/holding rules without using future labels.

    ``o2o_h1`` exists in the research panel for post-freeze evaluation, but it
    is deliberately absent from this eligibility mask.  A live run must be
    able to evaluate the latest formation close before its future execution
    return exists.
    """

    if len(score) != len(panel):
        raise ValueError("冻结评分数组与现货研究面板行数不一致")
    min_age = int(freeze["min_zero_age"])
    threshold = float(freeze["threshold_from_development"])
    same_zero = panel["state"].eq(0).to_numpy()
    eligible = same_zero & panel["state_age"].ge(min_age).to_numpy()
    above = eligible & np.isfinite(score) & (score >= threshold)
    confirmation = _confirmed(above, int(freeze["confirmation_days"]))
    holding = freeze["holding_package"]
    selected = np.zeros(len(panel), dtype=bool)
    holding_path = np.zeros(len(panel), dtype=bool)
    active = False
    age = 0
    cooldown = 0
    min_hold = int(holding["min_hold_days"])
    max_hold = int(holding["max_hold_days"])
    cooldown_days = int(holding["cooldown_days"])
    release = float(freeze["release_threshold"])
    for i in range(len(panel)):
        if cooldown:
            cooldown -= 1
        if active:
            age += 1
            leave = age >= max_hold or (
                age >= min_hold
                and (
                    not same_zero[i]
                    or not np.isfinite(score[i])
                    or score[i] < release
                )
            )
            if leave:
                active = False
                age = 0
                cooldown = cooldown_days
            else:
                holding_path[i] = True
        if not active and cooldown == 0 and confirmation[i]:
            selected[i] = True
            active = True
            age = 1
            holding_path[i] = True
    return selected, holding_path


def _effective_dates(panel: pd.DataFrame) -> tuple[pd.DatetimeIndex, bool]:
    formation = pd.to_datetime(panel["formation_date"], errors="raise").dt.normalize()
    effective = pd.Series(
        pd.to_datetime(panel["effective_date"], errors="coerce").dt.normalize().to_numpy(),
        index=panel.index,
        dtype="datetime64[ns]",
    )
    pending = effective.isna()
    if pending.any():
        pending_positions = np.flatnonzero(pending.to_numpy())
        if len(pending_positions) != 1 or pending_positions[0] != len(effective) - 1:
            raise RuntimeError("只有最后一个形成日允许暂时没有下一实际现货交易日")
        effective.iloc[-1] = formation.iloc[-1] + pd.offsets.BDay(1)
    if not (effective.to_numpy() > formation.to_numpy()).all():
        raise RuntimeError("零段反转存在形成日不早于执行日的日期映射")
    if effective.duplicated().any():
        raise RuntimeError("零段反转执行日映射出现重复日期")
    return pd.DatetimeIndex(effective), bool(pending.any())


def run_zero_transfer_frozen(
    spot_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    source = _resolve_spot(spot_path)
    output = _resolve_output(output_dir)
    print("[零段反转] 读取唯一远端现货并从冻结八状态公式生成基础状态", flush=True)
    spot, panel, spot_audit = load_spot_panel(source)
    effective, pending_effective = _effective_dates(panel)
    signals: dict[str, np.ndarray] = {}
    holding_paths: dict[str, np.ndarray] = {}
    latest_scores: dict[str, float | None] = {}
    for side in ("down", "up"):
        freeze = FROZEN_ZERO_TRANSFER[side]
        print(
            f"[零段反转-{side}] 运行冻结 {freeze['candidate_id']}（不扫描候选、不读取未来标签）",
            flush=True,
        )
        scores = compute_logic_scores(
            str(freeze["source_version"]),
            spot,
            panel,
            int(freeze["direction"]),
        )
        column = str(freeze["score_column"])
        if column not in scores.columns:
            raise ValueError(f"{side} 冻结评分列不存在：{column}")
        score = scores[column].to_numpy(dtype=float)
        latest_scores[side] = float(score[-1]) if np.isfinite(score[-1]) else None
        selected, holding_path = _frozen_signal_path(panel, score, freeze)
        signals[side] = selected
        holding_paths[side] = holding_path

    formation = pd.to_datetime(panel["formation_date"], errors="raise").dt.strftime("%Y-%m-%d")
    execution = pd.Series(effective, index=panel.index).dt.strftime("%Y-%m-%d")
    result = pd.DataFrame(
        {
            "formation_date": formation,
            "execution_date": execution,
            "minus_entry_signal": signals["down"].astype("int8"),
            "plus_entry_signal": signals["up"].astype("int8"),
        }
    )
    required = ["formation_date", "execution_date", "minus_entry_signal", "plus_entry_signal"]
    if list(result.columns) != required:
        raise AssertionError("零段反转运行结果列顺序被改变")
    if result[required].isna().any().any():
        raise AssertionError("零段反转结果出现缺失值；禁止用默认值填补")
    if not result["minus_entry_signal"].isin([0, 1]).all() or not result["plus_entry_signal"].isin([0, 1]).all():
        raise AssertionError("零段反转结果只能包含 0/1")
    signal_path = output / "remote_zero_transfer_predictions.csv"
    result.to_csv(signal_path, index=False, encoding="utf-8-sig")
    metadata = {
        "operation": "只用唯一远端现货和包内 V38/V57 冻结参数生成零段反转最新执行信号",
        "input_file": str(source),
        "input_contract": "one_raw_spot_file_only",
        "spot_audit": spot_audit,
        "output_file": str(signal_path),
        "columns": required,
        "rows": int(len(result)),
        "date_min": str(result["execution_date"].min()),
        "date_max": str(result["execution_date"].max()),
        "freeze": FROZEN_ZERO_TRANSFER,
        "signal_counts": {
            "minus_entry_days": int(result["minus_entry_signal"].sum()),
            "plus_entry_days": int(result["plus_entry_signal"].sum()),
            "minus_holding_days": int(holding_paths["down"].sum()),
            "plus_holding_days": int(holding_paths["up"].sum()),
        },
        "latest_formation_audit": {
            "formation_date": str(formation.iloc[-1]),
            "execution_date": str(execution.iloc[-1]),
            "base_three_state": int(panel["state"].iloc[-1]),
            "down_score": latest_scores["down"],
            "up_score": latest_scores["up"],
            "down_threshold": float(FROZEN_ZERO_TRANSFER["down"]["threshold_from_development"]),
            "up_threshold": float(FROZEN_ZERO_TRANSFER["up"]["threshold_from_development"]),
            "minus_entry_signal": int(result["minus_entry_signal"].iloc[-1]),
            "plus_entry_signal": int(result["plus_entry_signal"].iloc[-1]),
            "computed_from_spot_through": spot_audit["date_max"],
            "future_o2o_label_used": False,
        },
        "date_mapping": {
            "formation_rule": "formation_date t is computed from the t-day close and earlier spot rows",
            "execution_rule": "execution_date is the next actual trading row t+1; latest missing row uses the next weekday only as the executable date display",
            "formation_strictly_before_execution": True,
            "pending_latest_execution_date_display": pending_effective,
            "latest_formation_to_execution": f"{formation.iloc[-1]} -> {execution.iloc[-1]}",
            "signal_values_filled": False,
            "future_o2o_label_used_for_signal": False,
        },
        "runtime_contract": {
            "candidate_scan": False,
            "candidate_reselection": False,
            "local_signal_result_read": False,
            "remote_three_state_baseline_read": False,
            "future_returns_or_evaluation_records_read": False,
            "test_used_for_selection": False,
        },
    }
    (output / "remote_zero_transfer_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(
        f"[零段反转] output={signal_path}; rows={len(result)}; execution={result['execution_date'].min()} -> {result['execution_date'].max()}",
        flush=True,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="只用远端现货和零段反转冻结参数生成最新信号")
    parser.add_argument("--spot", default=None, help="远端现货绝对路径")
    parser.add_argument("--output", default=None, help="输出目录绝对路径")
    args = parser.parse_args()
    run_zero_transfer_frozen(args.spot, args.output)


if __name__ == "__main__":
    main()
