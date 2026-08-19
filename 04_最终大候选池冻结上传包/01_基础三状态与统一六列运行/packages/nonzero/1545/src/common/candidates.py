from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .registry import (
    CONFIRM_DAYS,
    COOLDOWN_DAYS,
    MIN_STATE_AGES,
    RAW_CANDIDATES_PER_SIDE,
    SCORE_QUANTILES,
)
from .scores import exit_sign, side_state


DEV_END = pd.Timestamp("2022-12-31")
VALID_END = pd.Timestamp("2024-12-31")


@dataclass(frozen=True)
class CandidateSpec:
    score_variant: str
    threshold_quantile: float
    threshold_value: float
    min_state_age: int
    confirm_days: int
    cooldown_days: int

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.score_variant}_q{int(round(self.threshold_quantile * 100)):02d}"
            f"_a{self.min_state_age}_c{self.confirm_days}_cd{self.cooldown_days}"
        )


def _phase_masks(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    index = pd.DatetimeIndex(panel.index)
    exit_date = pd.to_datetime(panel["exit_h1_date"], errors="coerce")
    development = (index <= DEV_END) & exit_date.le(DEV_END).to_numpy()
    validation = (
        (index > DEV_END)
        & (index <= VALID_END)
        & exit_date.le(VALID_END).to_numpy()
    )
    return {
        "Development": np.asarray(development, dtype=bool),
        "Validation": np.asarray(validation, dtype=bool),
        "Development+Validation": np.asarray(development | validation, dtype=bool),
    }


def _confirmed(condition: np.ndarray, days: int) -> np.ndarray:
    if days == 1:
        return condition.copy()
    rolling = pd.Series(condition.astype(np.int8)).rolling(days, min_periods=days).sum()
    return rolling.eq(days).to_numpy()


def _cooldown(signal: np.ndarray, days: int) -> np.ndarray:
    if days == 0:
        return signal.copy()
    output = np.zeros(len(signal), dtype=bool)
    next_allowed = 0
    for position in np.flatnonzero(signal):
        if position < next_allowed:
            continue
        output[position] = True
        next_allowed = int(position) + days + 1
    return output


def build_candidate_signal(
    panel: pd.DataFrame,
    score: pd.Series,
    spec: CandidateSpec,
    side: str,
) -> np.ndarray:
    matching = panel["base_state"].eq(side_state(side)).to_numpy()
    finite = score.notna().to_numpy()
    raw = matching & finite & score.ge(spec.threshold_value).to_numpy()
    confirmed = _confirmed(raw, spec.confirm_days)
    age_eligible = panel["state_age"].ge(spec.min_state_age).to_numpy()
    return _cooldown(confirmed & age_eligible, spec.cooldown_days)


def _signal_hash(signal: np.ndarray) -> str:
    packed = np.packbits(signal.astype(np.uint8), bitorder="little")
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else np.nan


def _stage_metrics(
    signal: np.ndarray,
    eligible: np.ndarray,
    improvement: np.ndarray,
    phase: np.ndarray,
) -> dict[str, float | int]:
    valid = phase & np.isfinite(improvement)
    trigger = signal & valid
    denominator = eligible & valid
    values = improvement[trigger]
    count = int(trigger.sum())
    eligible_count = int(denominator.sum())
    return {
        "n": count,
        "eligible_n": eligible_count,
        "coverage": float(count / eligible_count) if eligible_count else np.nan,
        "mean_bp": _safe_mean(values) * 10_000,
        "win_rate": float(np.mean(values > 0)) if count else np.nan,
    }


def _joint_score(dev: dict[str, Any], val: dict[str, Any], pooled: dict[str, Any]) -> float:
    if min(dev["n"], val["n"]) < 1:
        return np.nan
    dev_s = dev["mean_bp"] * dev["n"] / (dev["n"] + 8.0)
    val_s = val["mean_bp"] * val["n"] / (val["n"] + 6.0)
    pool_s = pooled["mean_bp"] * pooled["n"] / (pooled["n"] + 12.0)
    coverage_penalty = 2.0 * abs(dev["coverage"] - val["coverage"])
    return float(
        0.35 * dev_s
        + 0.45 * val_s
        + 0.10 * pool_s
        + 0.10 * min(dev_s, val_s)
        - coverage_penalty
    )


def _candidate_metrics(
    panel: pd.DataFrame,
    score: pd.Series,
    signal: np.ndarray,
    spec: CandidateSpec,
    side: str,
    phase_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    improvement = (exit_sign(side) * panel["o2o_h1"]).to_numpy(float)
    eligible = (
        panel["base_state"].eq(side_state(side)).to_numpy()
        & panel["state_age"].ge(spec.min_state_age).to_numpy()
        & score.notna().to_numpy()
    )
    dev = _stage_metrics(signal, eligible, improvement, phase_masks["Development"])
    val = _stage_metrics(signal, eligible, improvement, phase_masks["Validation"])
    pooled = _stage_metrics(signal, eligible, improvement, phase_masks["Development+Validation"])
    score_value = _joint_score(dev, val, pooled)
    formal_pass = bool(
        dev["n"] >= 12
        and val["n"] >= 8
        and pooled["n"] >= 24
        and dev["mean_bp"] > 0
        and val["mean_bp"] > 0
    )
    return {
        **asdict(spec),
        "candidate_id": spec.candidate_id,
        "development_n": dev["n"],
        "development_eligible_n": dev["eligible_n"],
        "development_coverage": dev["coverage"],
        "development_h1_improvement_bp": dev["mean_bp"],
        "development_h1_win_rate": dev["win_rate"],
        "validation_n": val["n"],
        "validation_eligible_n": val["eligible_n"],
        "validation_coverage": val["coverage"],
        "validation_h1_improvement_bp": val["mean_bp"],
        "validation_h1_win_rate": val["win_rate"],
        "pooled_n": pooled["n"],
        "pooled_eligible_n": pooled["eligible_n"],
        "pooled_coverage": pooled["coverage"],
        "pooled_h1_improvement_bp": pooled["mean_bp"],
        "pooled_h1_win_rate": pooled["win_rate"],
        "joint_score": score_value,
        "formal_pass": formal_pass,
        "selection_metric": "O2O_H1_exit_improvement_only",
    }


def freeze_candidates(
    panel: pd.DataFrame,
    scores: pd.DataFrame,
    side: str,
    version_id: str,
    core_logic_name: str,
) -> dict[str, Any]:
    if panel.index.max() > VALID_END:
        raise AssertionError("freeze_candidates 只能接收截至 2024-12-31 的物理截断数据")
    if not panel.index.equals(scores.index):
        raise AssertionError("panel/scores index mismatch")
    if len(scores.columns) != 8:
        raise AssertionError("每版必须恰好8个评分变体")
    if (panel["phase"] == "Test").any():
        raise AssertionError("冻结输入含 Test 行")

    phase_masks = _phase_masks(panel)
    dv_mask = panel.index.to_series().le(VALID_END).to_numpy()
    registry_rows: list[dict[str, Any]] = []
    representative_by_hash: dict[str, tuple[CandidateSpec, np.ndarray]] = {}
    threshold_audit: list[dict[str, Any]] = []

    for score_variant in scores.columns:
        score = scores[score_variant]
        threshold_population = (
            panel["base_state"].eq(side_state(side))
            & panel.index.to_series().le(DEV_END)
            & panel["exit_h1_date"].le(DEV_END)
            & score.notna()
        )
        population = score.loc[threshold_population]
        threshold_audit.append({
            "score_variant": score_variant,
            "development_threshold_rows": int(len(population)),
            "score_min": float(population.min()) if len(population) else np.nan,
            "score_max": float(population.max()) if len(population) else np.nan,
            "score_unique": int(population.nunique()) if len(population) else 0,
        })
        for quantile in SCORE_QUANTILES:
            threshold = float(population.quantile(quantile)) if len(population) else np.nan
            for min_age in MIN_STATE_AGES:
                for confirm in CONFIRM_DAYS:
                    for cooldown in COOLDOWN_DAYS:
                        spec = CandidateSpec(
                            score_variant=score_variant,
                            threshold_quantile=float(quantile),
                            threshold_value=threshold,
                            min_state_age=int(min_age),
                            confirm_days=int(confirm),
                            cooldown_days=int(cooldown),
                        )
                        if np.isfinite(threshold):
                            signal = build_candidate_signal(panel, score, spec, side)
                        else:
                            signal = np.zeros(len(panel), dtype=bool)
                        digest = _signal_hash(signal[dv_mask])
                        representative = representative_by_hash.get(digest)
                        if representative is None:
                            representative_by_hash[digest] = (spec, signal)
                            representative_id = spec.candidate_id
                        else:
                            representative_id = representative[0].candidate_id
                        registry_rows.append({
                            **asdict(spec),
                            "candidate_id": spec.candidate_id,
                            "signal_hash_dev_validation": digest,
                            "representative_candidate_id": representative_id,
                            "is_representative": representative is None,
                            "dev_validation_trigger_n": int((signal & phase_masks["Development+Validation"]).sum()),
                        })

    raw_registry = pd.DataFrame(registry_rows)
    if len(raw_registry) != RAW_CANDIDATES_PER_SIDE:
        raise AssertionError(f"候选数错误: {len(raw_registry)} != {RAW_CANDIDATES_PER_SIDE}")

    metric_rows: list[dict[str, Any]] = []
    for digest, (spec, signal) in representative_by_hash.items():
        row = _candidate_metrics(panel, scores[spec.score_variant], signal, spec, side, phase_masks)
        row["signal_hash_dev_validation"] = digest
        metric_rows.append(row)
    unique_metrics = pd.DataFrame(metric_rows)
    if unique_metrics.empty:
        ranked = unique_metrics.copy()
    else:
        ranked = unique_metrics.loc[unique_metrics["joint_score"].notna()].sort_values(
            ["joint_score", "pooled_n", "candidate_id"], ascending=[False, False, True]
        ).reset_index(drop=True)
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    top1: dict[str, Any] | None = None
    reason: str | None = None
    if len(ranked):
        top1 = ranked.iloc[0].to_dict()
        top1.update({
            "version_id": version_id,
            "side": side,
            "base_state": side_state(side),
            "action": "-1→0" if side == "minus" else "1→0",
            "core_logic_name": core_logic_name,
            "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "selection_data_end": str(panel.index.max().date()),
            "model_fit_period": "Development only",
            "threshold_fit_period": "Development only",
            "ranking_period": "Development+Validation",
            "test_used_for_selection": False,
        })
        if any("test" in str(key).lower() for key in top1):
            allowed = {"test_used_for_selection"}
            offending = [key for key in top1 if "test" in str(key).lower() and key not in allowed]
            if offending:
                raise AssertionError(f"冻结Top1意外含Test字段: {offending}")
    else:
        reason = "没有同时具备至少1个Development和1个Validation触发的可计算候选"

    unique_count = int(len(representative_by_hash))
    pool_audit = {
        "version_id": version_id,
        "side": side,
        "core_logic_name": core_logic_name,
        "raw_candidate_count": int(len(raw_registry)),
        "deduplicated_signal_count": unique_count,
        "duplicate_signal_count": int(len(raw_registry) - unique_count),
        "duplicate_signal_ratio": float(1.0 - unique_count / len(raw_registry)),
        "computable_ranked_count": int(len(ranked)),
        "formal_pass_count": int(ranked["formal_pass"].sum()) if len(ranked) else 0,
        "parameter_dimensions": {
            "score_variant": 8,
            "development_threshold_quantile": 9,
            "minimum_state_age": 4,
            "confirmation_days": 3,
            "cooldown_days": 3,
        },
        "threshold_audit": threshold_audit,
        "top1_available": top1 is not None,
        "no_top1_reason": reason,
        "selection_metric": "O2O_H1_exit_improvement_only",
        "c2c_used_for_selection": False,
        "h3_used_for_selection": False,
        "natural_switch_used_for_selection": False,
        "test_used_for_selection": False,
    }
    return {
        "raw_registry": raw_registry,
        "unique_metrics": unique_metrics,
        "ranked": ranked,
        "top20": ranked.head(20).copy(),
        "top1": top1,
        "pool_audit": pool_audit,
    }


def top1_spec(top1: dict[str, Any]) -> CandidateSpec:
    return CandidateSpec(
        score_variant=str(top1["score_variant"]),
        threshold_quantile=float(top1["threshold_quantile"]),
        threshold_value=float(top1["threshold_value"]),
        min_state_age=int(top1["min_state_age"]),
        confirm_days=int(top1["confirm_days"]),
        cooldown_days=int(top1["cooldown_days"]),
    )


def apply_frozen_top1(
    panel: pd.DataFrame,
    scores: pd.DataFrame,
    top1: dict[str, Any],
    side: str,
) -> pd.Series:
    if bool(top1.get("test_used_for_selection")):
        raise AssertionError("拒绝应用声称使用Test选择的Top1")
    spec = top1_spec(top1)
    signal = build_candidate_signal(panel, scores[spec.score_variant], spec, side)
    return pd.Series(signal, index=panel.index, name="exit_to_zero")


def json_safe_top1(top1: dict[str, Any] | None) -> dict[str, Any] | None:
    if top1 is None:
        return None
    return json.loads(json.dumps(top1, ensure_ascii=False, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(type(value).__name__)

