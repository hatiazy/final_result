from __future__ import annotations

"""Build the causal continuous sidecar for the daily frozen output.

The eight-column execution table is intentionally left unchanged.  This
sidecar exposes the continuous evidence already used by the frozen spot-only
1545 engine, plus score/threshold distances for V38/V57 and V156/V189.  Every
row is keyed by formation close and mapped to the next actual execution row;
the last row may use the next weekday only as the executable-date display.
No future O2O/C2C label is read or used here.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ZERO_TRANSFER_SOURCE = PACKAGE_ROOT / "packages" / "zero_transfer" / "src"
if str(ZERO_TRANSFER_SOURCE) not in sys.path:
    sys.path.insert(0, str(ZERO_TRANSFER_SOURCE))

from spot_panel import load_spot_panel  # noqa: E402


EXTREME_DEFAULT_THRESHOLDS = {
    "up": 0.8135114753699175,
    "down": 0.6832148298881413,
}

OUTPUT_COLUMNS = [
    "形成日",
    "实际执行日",
    "三状态",
    "state_strength",
    "state_phase",
    "fast_engine",
    "slow_engine",
    "direction_score",
    "direction_score_continuous",
    "direction_score_band",
    "risk_high_count",
    "-1转0_score",
    "-1转0_冻结阈值",
    "-1转0_距阈值",
    "+1转0_score",
    "+1转0_冻结阈值",
    "+1转0_距阈值",
    "0转-1_score",
    "0转-1_冻结阈值",
    "0转-1_距阈值",
    "0转-1_释放阈值",
    "0转-1_距释放阈值",
    "0转-1_事件信号",
    "0转+1_score",
    "0转+1_冻结阈值",
    "0转+1_距阈值",
    "0转+1_释放阈值",
    "0转+1_距释放阈值",
    "0转+1_事件信号",
    "大涨_score",
    "大涨_冻结阈值",
    "大涨_距阈值",
    "大涨_预测",
    "大跌_score",
    "大跌_冻结阈值",
    "大跌_距阈值",
    "大跌_预测",
]


def _absolute(raw: str | Path, label: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} 必须使用绝对路径：{raw}")
    return path.resolve()


def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if (~compact).any():
        parsed.loc[~compact] = pd.to_datetime(text.loc[~compact], errors="coerce", format="mixed")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    if parsed.isna().any():
        raise ValueError("连续诊断输入存在无法解析的日期")
    return parsed.dt.normalize()


def _effective_dates(panel: pd.DataFrame) -> tuple[pd.Series, bool]:
    formation = _parse_dates(panel["formation_date"])
    effective = pd.Series(
        pd.to_datetime(panel["effective_date"], errors="coerce").dt.normalize().to_numpy(),
        index=panel.index,
        dtype="datetime64[ns]",
    )
    pending = effective.isna()
    if pending.any():
        positions = np.flatnonzero(pending.to_numpy())
        if len(positions) != 1 or positions[0] != len(effective) - 1:
            raise RuntimeError("连续诊断只有最后一个形成日允许缺少下一实际交易日")
        effective.iloc[-1] = formation.iloc[-1] + pd.offsets.BDay(1)
    if not (effective.to_numpy() > formation.to_numpy()).all():
        raise RuntimeError("连续诊断存在形成日不早于执行日的行")
    if effective.duplicated().any():
        raise RuntimeError("连续诊断执行日出现重复")
    return effective, bool(pending.any())


def _enforce_threshold_side(values: pd.Series, condition: pd.Series) -> pd.Series:
    values = pd.Series(values, index=condition.index, dtype=float)
    condition = pd.Series(condition, index=values.index).fillna(False).astype(bool)
    half_below = np.nextafter(0.5, 0.0)
    return pd.Series(
        np.where(condition, values.clip(0.5, 1.0), values.clip(0.0, half_below)),
        index=values.index,
    ).fillna(0.0)


def _ge_support(values: pd.Series, threshold: float, index: pd.Index) -> pd.Series:
    values = pd.Series(values, index=index, dtype=float)
    raw = pd.Series(
        np.where(
            values <= threshold,
            0.5 * values / max(float(threshold), 1e-12),
            0.5 + 0.5 * (values - threshold) / max(1.0 - float(threshold), 1e-12),
        ),
        index=index,
    ).clip(0, 1)
    return _enforce_threshold_side(raw, values.ge(threshold))


def _le_support(values: pd.Series, threshold: float, index: pd.Index) -> pd.Series:
    values = pd.Series(values, index=index, dtype=float)
    return 1.0 - _ge_support(values, threshold, index)


def _change_le_support(values: pd.Series, threshold: float, scale: float, index: pd.Index) -> pd.Series:
    values = pd.Series(values, index=index, dtype=float)
    raw = (0.5 + 0.5 * ((threshold - values) / scale).clip(-1, 1)).clip(0, 1)
    return _enforce_threshold_side(raw, values.le(threshold))


def _bool_support(values: pd.Series, index: pd.Index) -> pd.Series:
    return pd.Series(values, index=index).fillna(False).astype(float)


def _state_strength(panel: pd.DataFrame) -> pd.Series:
    """Mirror the reference 11 rule-evidence strength formula."""
    idx = panel.index
    signal = pd.to_numeric(panel["state"], errors="raise").astype("int8")
    axis = pd.to_numeric(panel["direction_score"], errors="coerce").clip(0, 1)
    slow = pd.to_numeric(panel["slow_engine"], errors="coerce").clip(0, 1)
    fast = pd.to_numeric(panel["fast_engine"], errors="coerce").clip(0, 1)
    trend = pd.to_numeric(panel["趋势状态"], errors="coerce").clip(0, 1)
    volume_price = pd.to_numeric(panel["量价状态"], errors="coerce").clip(0, 1)
    position = pd.to_numeric(panel["位置修复状态"], errors="coerce").clip(0, 1)
    intraday = pd.to_numeric(panel["跳空/日内承接状态"], errors="coerce").clip(0, 1)
    low_count = pd.concat(
        [pd.to_numeric(panel[column], errors="coerce") for column in (
            "位置修复状态", "跳空/日内承接状态", "趋势状态", "量价状态",
        )],
        axis=1,
    ).le(1 / 3).sum(axis=1).astype(float).clip(0, 4)
    high_count = pd.concat(
        [pd.to_numeric(panel[column], errors="coerce") for column in (
            "位置修复状态", "跳空/日内承接状态", "趋势状态", "量价状态",
        )],
        axis=1,
    ).ge(2 / 3).sum(axis=1).astype(float).clip(0, 4)
    delta1 = axis.diff()
    delta2 = axis.diff(2)
    rebound = panel["rebound_veto"].astype(bool)
    exit_veto = panel["heat_reversal_exit_veto"].astype(bool)
    long_positive_context = panel["long_positive_context"].astype(bool)

    negative_components = pd.DataFrame({
        "axis_le_030": _le_support(axis, 0.30, idx),
        "slow_le_030": _le_support(slow, 0.30, idx),
        "fast_le_042": _le_support(fast, 0.42, idx),
        "low_count_ge_3": _ge_support(low_count / 4.0, 3 / 4, idx),
        "delta2_axis_le_0": _change_le_support(delta2, 0.0, 0.20, idx),
        "no_long_positive_context": _bool_support(~long_positive_context, idx),
        "no_rebound_veto": _bool_support(~rebound, idx),
    }, index=idx)
    positive_components = pd.DataFrame({
        "axis_ge_070": _ge_support(axis, 0.70, idx),
        "trend_ge_070": _ge_support(trend, 0.70, idx),
        "volume_ge_070": _ge_support(volume_price, 0.70, idx),
        "position_ge_042": _ge_support(position, 0.42, idx),
        "intraday_ge_042": _ge_support(intraday, 0.42, idx),
        "no_exit_veto": _bool_support(~exit_veto, idx),
    }, index=idx)
    negative_support = negative_components.min(axis=1)
    positive_support = positive_components.min(axis=1)
    result = pd.Series(
        np.select(
            [signal.eq(-1), signal.eq(0), signal.eq(1)],
            [-negative_support, positive_support - negative_support, positive_support],
            default=np.nan,
        ),
        index=idx,
        name="state_strength",
        dtype=float,
    ).clip(-1, 1)
    return result.fillna(0.0)


def _state_phase(panel: pd.DataFrame) -> pd.Series:
    signal = pd.to_numeric(panel["state"], errors="raise").astype("int8")
    run_id = signal.ne(signal.shift()).cumsum()
    run_length = signal.groupby(run_id).cumcount().add(1)
    phase = pd.Series("continuation", index=panel.index, dtype="string")
    phase.loc[run_length.le(2)] = "entry"
    phase.loc[panel["pending_transition"].astype(bool)] = "pending_switch"
    return phase


def _summary_threshold(engine_dir: Path, side: str) -> float:
    path = engine_dir / f"remote_{side}_extreme_summary.json"
    default = float(EXTREME_DEFAULT_THRESHOLDS[side])
    if not path.is_file():
        return default
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("freeze", {}).get("score_threshold_fitted_development", default)
    value = float(value)
    if not np.isclose(value, default, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{side} 大涨/大跌冻结阈值与包内冻结值不一致：{value} != {default}")
    return value


def _load_extreme(engine_dir: Path, side: str, formation: pd.Series) -> pd.DataFrame:
    path = engine_dir / f"remote_{side}_extreme_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少{side}大涨/大跌最新预测文件：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"date", "score", "predicted"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} 缺少连续值字段：{missing}")
    frame = frame.loc[:, ["date", "score", "predicted"]].copy()
    frame["formation_date"] = _parse_dates(frame["date"])
    frame["score"] = pd.to_numeric(frame["score"], errors="raise")
    frame["predicted"] = pd.to_numeric(frame["predicted"], errors="raise").astype("int8")
    if frame["formation_date"].duplicated().any():
        raise ValueError(f"{path.name} 的形成日重复")
    return frame.set_index("formation_date").reindex(formation.to_numpy())


def _load_zero_continuous(engine_dir: Path, formation: pd.Series) -> pd.DataFrame:
    path = engine_dir / "remote_zero_transfer_continuous.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少零段反转连续分数文件：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "formation_date", "minus_score", "plus_score", "minus_threshold", "plus_threshold",
        "minus_release_threshold", "plus_release_threshold", "minus_entry_signal", "plus_entry_signal",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} 缺少连续值字段：{missing}")
    frame["formation_date"] = _parse_dates(frame["formation_date"])
    if frame["formation_date"].duplicated().any():
        raise ValueError("零段反转连续分数形成日重复")
    frame = frame.set_index("formation_date")
    return frame.reindex(formation.to_numpy())


def _load_nonzero_continuous(engine_dir: Path, formation: pd.Series) -> pd.DataFrame:
    path = engine_dir / "remote_nonzero_continuous.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少非零反转连续分数文件：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "formation_date", "minus_score", "plus_score", "minus_threshold", "plus_threshold",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} 缺少连续值字段：{missing}")
    frame["formation_date"] = _parse_dates(frame["formation_date"])
    if frame["formation_date"].duplicated().any():
        raise ValueError("非零连续分数形成日重复")
    return frame.set_index("formation_date").reindex(formation.to_numpy())


def build_continuous_diagnostics(
    spot_path: str | Path,
    engine_output_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = _absolute(spot_path, "现货输入")
    engine = _absolute(engine_output_dir, "冻结引擎输出目录")
    output = _absolute(output_dir, "连续诊断输出目录")
    if not source.is_file():
        raise FileNotFoundError(f"现货输入不存在：{source}")
    if not engine.is_dir():
        raise FileNotFoundError(f"冻结引擎输出目录不存在：{engine}")
    output.mkdir(parents=True, exist_ok=True)

    _spot, panel, spot_audit = load_spot_panel(source)
    effective, pending_latest = _effective_dates(panel)
    formation = _parse_dates(panel["formation_date"])
    signal = pd.to_numeric(panel["state"], errors="raise").astype("int8")
    state_strength = _state_strength(panel)
    state_phase = _state_phase(panel)

    zero = _load_zero_continuous(engine, formation)
    nonzero = _load_nonzero_continuous(engine, formation)
    up = _load_extreme(engine, "up", formation)
    down = _load_extreme(engine, "down", formation)
    up_threshold = _summary_threshold(engine, "up")
    down_threshold = _summary_threshold(engine, "down")

    out = pd.DataFrame({
        "形成日": formation,
        "实际执行日": effective,
        "三状态": signal,
        "state_strength": state_strength.to_numpy(),
        "state_phase": state_phase.to_numpy(),
        "fast_engine": pd.to_numeric(panel["fast_engine"], errors="raise").clip(0, 1).to_numpy(),
        "slow_engine": pd.to_numeric(panel["slow_engine"], errors="raise").clip(0, 1).to_numpy(),
        "direction_score": pd.to_numeric(panel["direction_score"], errors="raise").clip(0, 1).to_numpy(),
        "direction_score_continuous": pd.to_numeric(panel["direction_score_continuous"], errors="raise").clip(0, 1).to_numpy(),
        "direction_score_band": pd.to_numeric(panel["direction_score_band"], errors="raise").clip(0, 1).to_numpy(),
        "risk_high_count": pd.to_numeric(panel["risk_high_count"], errors="raise").to_numpy(),
        "-1转0_score": pd.to_numeric(nonzero["minus_score"], errors="coerce").to_numpy(),
        "-1转0_冻结阈值": pd.to_numeric(nonzero["minus_threshold"], errors="raise").to_numpy(),
        "-1转0_距阈值": (pd.to_numeric(nonzero["minus_score"], errors="coerce") - pd.to_numeric(nonzero["minus_threshold"], errors="raise")).to_numpy(),
        "+1转0_score": pd.to_numeric(nonzero["plus_score"], errors="coerce").to_numpy(),
        "+1转0_冻结阈值": pd.to_numeric(nonzero["plus_threshold"], errors="raise").to_numpy(),
        "+1转0_距阈值": (pd.to_numeric(nonzero["plus_score"], errors="coerce") - pd.to_numeric(nonzero["plus_threshold"], errors="raise")).to_numpy(),
        "0转-1_score": pd.to_numeric(zero["minus_score"], errors="coerce").to_numpy(),
        "0转-1_冻结阈值": pd.to_numeric(zero["minus_threshold"], errors="raise").to_numpy(),
        "0转-1_距阈值": (pd.to_numeric(zero["minus_score"], errors="coerce") - pd.to_numeric(zero["minus_threshold"], errors="raise")).to_numpy(),
        "0转-1_释放阈值": pd.to_numeric(zero["minus_release_threshold"], errors="raise").to_numpy(),
        "0转-1_距释放阈值": (pd.to_numeric(zero["minus_score"], errors="coerce") - pd.to_numeric(zero["minus_release_threshold"], errors="raise")).to_numpy(),
        "0转-1_事件信号": pd.to_numeric(zero["minus_entry_signal"], errors="raise").astype("int8").to_numpy(),
        "0转+1_score": pd.to_numeric(zero["plus_score"], errors="coerce").to_numpy(),
        "0转+1_冻结阈值": pd.to_numeric(zero["plus_threshold"], errors="raise").to_numpy(),
        "0转+1_距阈值": (pd.to_numeric(zero["plus_score"], errors="coerce") - pd.to_numeric(zero["plus_threshold"], errors="raise")).to_numpy(),
        "0转+1_释放阈值": pd.to_numeric(zero["plus_release_threshold"], errors="raise").to_numpy(),
        "0转+1_距释放阈值": (pd.to_numeric(zero["plus_score"], errors="coerce") - pd.to_numeric(zero["plus_release_threshold"], errors="raise")).to_numpy(),
        "0转+1_事件信号": pd.to_numeric(zero["plus_entry_signal"], errors="raise").astype("int8").to_numpy(),
        "大涨_score": pd.to_numeric(up["score"], errors="coerce").to_numpy(),
        "大涨_冻结阈值": up_threshold,
        "大涨_距阈值": (pd.to_numeric(up["score"], errors="coerce") - up_threshold).to_numpy(),
        "大涨_预测": pd.to_numeric(up["predicted"], errors="raise").astype("int8").to_numpy(),
        "大跌_score": pd.to_numeric(down["score"], errors="coerce").to_numpy(),
        "大跌_冻结阈值": down_threshold,
        "大跌_距阈值": (pd.to_numeric(down["score"], errors="coerce") - down_threshold).to_numpy(),
        "大跌_预测": pd.to_numeric(down["predicted"], errors="raise").astype("int8").to_numpy(),
    })
    out = out.loc[:, OUTPUT_COLUMNS]
    out["形成日"] = pd.to_datetime(out["形成日"])
    out["实际执行日"] = pd.to_datetime(out["实际执行日"])
    out = out.sort_values("实际执行日").reset_index(drop=True)

    if out["实际执行日"].duplicated().any() or not out["实际执行日"].is_monotonic_increasing:
        raise AssertionError("连续诊断执行日重复或未排序")
    if not out["三状态"].isin([-1, 0, 1]).all():
        raise AssertionError("连续诊断三状态出现非法值")
    if not out["state_strength"].between(-1, 1).all():
        raise AssertionError("state_strength 超出 [-1, 1]")
    if not out[["大涨_score", "大跌_score"]].tail(1).notna().all().all():
        raise AssertionError("最新形成日缺少大涨/大跌连续分数，拒绝生成占位诊断")
    if not out[["-1转0_score", "+1转0_score", "0转-1_score", "0转+1_score"]].tail(1).notna().all().all():
        raise AssertionError("最新形成日缺少冻结反转连续分数，拒绝生成占位诊断")

    # V80 is a Development-fitted supervised score. The frozen scorer does
    # not emit an in-sample score for the early Development rows, so its two
    # historical sidecar fields are structurally not-applicable there. Keep
    # them as NaN instead of inventing 0 or carrying a later value backwards.
    # Any other missing continuous field is an actual production error.
    structural_missing_fields = {
        column: int(out[column].isna().sum())
        for column in OUTPUT_COLUMNS
        if int(out[column].isna().sum()) > 0
    }
    allowed_structural_missing = {"+1转0_score", "+1转0_距阈值"}
    unexpected_missing = set(structural_missing_fields) - allowed_structural_missing
    if unexpected_missing:
        raise AssertionError(f"连续诊断存在未登记缺失字段，拒绝生成：{sorted(unexpected_missing)}")
    latest_runtime_fields = [column for column in OUTPUT_COLUMNS if column not in {"形成日", "实际执行日", "state_phase"}]
    latest_row_all_runtime_fields_present = bool(out[latest_runtime_fields].tail(1).notna().all().all())
    if not latest_row_all_runtime_fields_present:
        raise AssertionError("最新执行日诊断字段存在缺失，拒绝生成占位输出")

    output_path = output / "连续诊断输出.csv"
    csv_out = out.copy()
    csv_out["形成日"] = csv_out["形成日"].dt.strftime("%Y-%m-%d")
    csv_out["实际执行日"] = csv_out["实际执行日"].dt.strftime("%Y-%m-%d")
    csv_out.to_csv(output_path, index=False, encoding="utf-8-sig")

    metadata = {
        "operation": "从唯一远端现货和本次冻结引擎连续分数生成诊断旁表；不改变八列表",
        "input_spot": str(source),
        "engine_output_dir": str(engine),
        "output_file": str(output_path),
        "columns": OUTPUT_COLUMNS,
        "rows": int(len(out)),
        "date_min": str(csv_out["实际执行日"].min()),
        "date_max": str(csv_out["实际执行日"].max()),
        "latest_formation_to_execution": f"{csv_out['形成日'].iloc[-1]} -> {csv_out['实际执行日'].iloc[-1]}",
        "latest_formation_date": str(csv_out["形成日"].iloc[-1]),
        "pending_latest_execution_display": pending_latest,
        "state_strength_definition": "reference-11 frozen-rule evidence support; not a return forecast",
        "state_phase_definition": "entry for first two days of a state run; continuation thereafter; pending_switch overrides when a frozen transition is waiting for confirmation",
        "extreme_score_definition": "latest V156/V189 score from formation close; margin = score - frozen Development threshold",
        "zero_transfer_score_definition": "latest V38/V57 score from formation close; margins are score minus frozen entry/release thresholds",
        "nonzero_score_definition": "latest V55/V80 score from formation close; margin is score minus frozen Development threshold",
        "future_o2o_label_used": False,
        "signal_values_filled": False,
        "missing_value_policy": "structural_na_only; never fill defaults, backfill, or placeholders",
        "structural_missing_fields": structural_missing_fields,
        "structural_missing_reason": {
            "+1转0_score": "V80 冻结分类器在其尚未进入前向评分域的早期 Development 行不输出 in-sample score；该字段对这些历史行不适用，不参与精简八列，也不用于当前最新信号。",
            "+1转0_距阈值": "随 +1转0_score 一起不适用；不以 0、阈值或后续日期值填充。",
        },
        "latest_row_all_runtime_fields_present": latest_row_all_runtime_fields_present,
        "spot_audit": {
            "date_min": spot_audit["date_min"],
            "date_max": spot_audit["date_max"],
            "future_values_used_in_state": spot_audit["future_values_used_in_state"],
            "external_state_file_read": spot_audit["external_state_file_read"],
            "external_three_state_baseline_read": spot_audit["external_three_state_baseline_read"],
        },
    }
    metadata_path = output / "连续诊断输出_运行记录.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return metadata


__all__ = ["OUTPUT_COLUMNS", "build_continuous_diagnostics"]
