"""Company-side runner for the post-90 side-specific full-grid pools.

The package ships a compact version/side registry rather than local candidate
results.  Each admitted version/side contributes its full preregistered grid;
the company rebuilds all scores and thresholds, ranks candidates independently
inside one side using company Development+Validation, freezes one Top1, and
only then evaluates company Test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zero_transfer.design_registry import (  # noqa: E402
    CONFIRMATION_DAYS,
    ENTRY_QUANTILES,
    HOLDING_PACKAGES,
    JOINT_SCORE_WEIGHTS,
    MINIMUM_COMPUTABILITY,
    MIN_ZERO_AGES,
    SCORE_VARIANTS_BY_VERSION,
)
from zero_transfer.logic_features import compute_logic_scores  # noqa: E402
from zero_transfer.logic_registry import LOGIC_BY_VERSION  # noqa: E402
from pool_registry import POOL_RULE_ID, POOL_VERSION_SIDES  # noqa: E402
from state_adjustment_diagnostics import build_state_adjustment_diagnostics  # noqa: E402


PHASES = {
    "development": (pd.Timestamp("2018-01-01"), pd.Timestamp("2022-12-31")),
    "validation": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
    "test": (pd.Timestamp("2025-01-01"), pd.Timestamp("2100-01-01")),
}

# The upload package uses the same full-grid, Dev/Val-only quality protocol as
# the local audit.  A candidate is never discarded before its event-only state
# and actual holding path have been checked; Test is read only after freezing.
STATE_QUALITY_SHORTLIST_SIZE = 10**9
QUALITY_MIN_SELECTED_DAYS = 20
HOLDING_QUALITY_MIN_SEGMENT_WIN = 0.55
STATE_QUALITY_DOMINANCE_FIELDS = (
    ("development", "mean_directional_o2o_h1"),
    ("validation", "mean_directional_o2o_h1"),
    ("development", "mean_directional_o2o_h3"),
    ("validation", "mean_directional_o2o_h3"),
    ("development", "rank_ic"),
    ("validation", "rank_ic"),
    ("state_quality_development", "target_daily_mean"),
    ("state_quality_validation", "target_daily_mean"),
    ("state_quality_development", "target_segment_win"),
    ("state_quality_validation", "target_segment_win"),
    ("state_quality_development", "negative_zero_one_day_share"),
    ("state_quality_validation", "negative_zero_one_day_share"),
)

ProgressFn = Callable[[str], None]


def _candidate_id(version: str, side: str, score_number: int, min_age: int, quantile: float, confirmation: int, holding: str) -> str:
    return f"{version}_{side}_s{score_number:02d}_a{min_age}_q{quantile:.2f}_c{confirmation}_{holding}"


def _candidates(side: str) -> list[dict[str, Any]]:
    if side not in ("down", "up"):
        raise ValueError(side)
    rows: list[dict[str, Any]] = []
    for version in POOL_VERSION_SIDES[side]:
        spec = LOGIC_BY_VERSION[version]
        for score_number, score_config in enumerate(SCORE_VARIANTS_BY_VERSION[version]):
            for min_age in MIN_ZERO_AGES:
                for quantile in ENTRY_QUANTILES:
                    for confirmation in CONFIRMATION_DAYS:
                        for holding in HOLDING_PACKAGES:
                            rows.append(
                                {
                                    "candidate_id": _candidate_id(version, side, score_number, min_age, float(quantile), confirmation, holding.package_id),
                                    "method_key": spec.method_key,
                                    "core_logic_name": spec.core_logic_name,
                                    "score_variant_number": int(score_number),
                                    "score_variant": dict(score_config),
                                    "min_zero_age": int(min_age),
                                    "entry_quantile": float(quantile),
                                    "confirmation_days": int(confirmation),
                                    "holding_package": holding.to_dict(),
                                    "source_version": version,
                                    "source_side": side,
                                }
                            )
    expected = len(POOL_VERSION_SIDES[side]) * 6144
    if len(rows) != expected or len({row["candidate_id"] for row in rows}) != len(rows):
        raise AssertionError(f"{side} generated {len(rows)} candidates; expected {expected} unique candidates")
    return rows


def _phase_mask(panel: pd.DataFrame, phase: str) -> np.ndarray:
    if phase == "full":
        return np.ones(len(panel), dtype=bool)
    start, end = PHASES[phase]
    dates = pd.to_datetime(panel["formation_date"])
    return dates.ge(start).to_numpy() & dates.le(end).to_numpy()


def _eligible(panel: pd.DataFrame, min_age: int, mask: np.ndarray) -> np.ndarray:
    return (
        panel["state"].eq(0).to_numpy()
        & panel["state_age"].ge(min_age).to_numpy()
        & panel["o2o_h1"].notna().to_numpy()
        & mask
    )


def _confirmed(above: np.ndarray, days: int) -> np.ndarray:
    if days <= 1:
        return above.copy()
    return pd.Series(above.astype(int)).rolling(days, min_periods=days).sum().eq(days).to_numpy() & above


def _signal_path(panel: pd.DataFrame, score: np.ndarray, candidate: dict[str, Any], *, return_holding: bool = False) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, np.ndarray]:
    min_age = int(candidate["min_zero_age"])
    quantile = float(candidate["entry_quantile"])
    dev = _eligible(panel, min_age, _phase_mask(panel, "development"))
    values = score[dev & np.isfinite(score)]
    threshold = float(np.quantile(values, quantile)) if len(values) else np.nan
    if not np.isfinite(threshold):
        return np.zeros(len(panel), dtype=bool), threshold
    eligible = _eligible(panel, min_age, np.ones(len(panel), dtype=bool))
    same_zero = panel["state"].eq(0).to_numpy()
    above = eligible & same_zero & np.isfinite(score) & (score >= threshold)
    confirmation = _confirmed(above, int(candidate["confirmation_days"]))
    holding = candidate["holding_package"]
    release_q = max(0.05, quantile - float(holding["release_quantile_gap"]))
    release = float(np.quantile(values, release_q)) if len(values) else threshold
    selected = np.zeros(len(panel), dtype=bool)
    active = False
    age = 0
    cooldown = 0
    holding_path = np.zeros(len(panel), dtype=bool)
    min_hold = int(holding["min_hold_days"])
    max_hold = int(holding["max_hold_days"])
    cooldown_days = int(holding["cooldown_days"])
    for i in range(len(panel)):
        if cooldown:
            cooldown -= 1
        if active:
            age += 1
            leave = age >= max_hold or (
                age >= min_hold
                and (not same_zero[i] or not np.isfinite(score[i]) or score[i] < release)
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
    if return_holding:
        return selected, threshold, holding_path
    return selected, threshold


def _rank_ic(score: np.ndarray, target: np.ndarray) -> float:
    valid = np.isfinite(score) & np.isfinite(target)
    if valid.sum() < 5:
        return np.nan
    return float(pd.Series(score[valid]).rank().corr(pd.Series(target[valid]).rank()))


def _metrics(panel: pd.DataFrame, score: np.ndarray, selected: np.ndarray, side: int, phase: str, min_age: int) -> dict[str, Any]:
    mask = _phase_mask(panel, phase)
    eligible = _eligible(panel, min_age, mask)
    chosen = selected & mask
    h1 = side * panel["o2o_h1"].to_numpy(dtype=float)
    h3 = side * panel["o2o_h3"].to_numpy(dtype=float)
    target = h1[chosen]
    future = h3[chosen]
    next_state = pd.to_numeric(panel["next_frozen_state"], errors="coerce").astype(float).to_numpy()
    next_finite = np.isfinite(next_state)
    target_transition = next_finite & (next_state == float(side))
    opposite_transition = next_finite & (next_state == float(-side))
    exit_transition = next_finite & (next_state != 0.0)
    eligible_next = eligible & next_finite
    target_count = int(np.logical_and(chosen, target_transition).sum())
    opposite_count = int(np.logical_and(chosen, opposite_transition).sum())
    exit_count = int(np.logical_and(chosen, exit_transition).sum())
    target_rate = float(target_count / chosen.sum()) if chosen.sum() else np.nan
    eligible_target_rate = float(target_transition[eligible_next].mean()) if eligible_next.sum() else np.nan
    result: dict[str, Any] = {
        "selected_days": int(chosen.sum()),
        "eligible_days": int(eligible.sum()),
        "coverage": float(chosen.sum() / eligible.sum()) if eligible.sum() else np.nan,
        "mean_directional_o2o_h1": float(np.nanmean(target)) if len(target) else np.nan,
        "mean_directional_o2o_h3": float(np.nanmean(future)) if np.isfinite(future).any() else np.nan,
        "h1_hit_rate": float(np.mean(target > 0)) if len(target) else np.nan,
        "rank_ic": _rank_ic(score[chosen], target),
        "path_consistency_h1_h3": float(np.mean(np.sign(target[np.isfinite(future)]) == np.sign(future[np.isfinite(future)]))) if np.isfinite(future).any() else np.nan,
        "rapid_restore_rate": float(panel.loc[chosen, "next_frozen_state"].eq(0).mean()) if chosen.sum() else np.nan,
        "target_transition_rate": target_rate,
        "opposite_transition_rate": float(opposite_count / chosen.sum()) if chosen.sum() else np.nan,
        "transition_event_rate": float(exit_count / chosen.sum()) if chosen.sum() else np.nan,
        "transition_purity": float(target_count / exit_count) if exit_count else np.nan,
        "eligible_target_transition_rate": eligible_target_rate,
        "target_transition_lift_vs_eligible": target_rate - eligible_target_rate if np.isfinite(target_rate) and np.isfinite(eligible_target_rate) else np.nan,
        "test_used_for_selection": False,
    }
    eligible_target = h1[eligible]
    result["improvement_vs_all_eligible_zero"] = result["mean_directional_o2o_h1"] - (float(np.nanmean(eligible_target)) if len(eligible_target) else np.nan)
    return result


def _persistent_own_state(panel: pd.DataFrame, holding: np.ndarray, side: int) -> np.ndarray:
    """Causally carry one side's frozen holding path through base-zero runs."""

    base = pd.to_numeric(panel["state"], errors="raise").astype(int).to_numpy()
    zero = base == 0
    positions = np.arange(len(base), dtype=int)
    run_id = np.cumsum(~zero).astype(int)
    run_count = int(run_id.max()) + 1 if len(run_id) else 0
    first_signal = np.full(run_count, len(base), dtype=int)
    signal_positions = np.flatnonzero(zero & np.asarray(holding, dtype=bool))
    if len(signal_positions):
        np.minimum.at(first_signal, run_id[signal_positions], signal_positions)
    state = base.copy()
    carry = zero & (positions >= first_signal[run_id])
    state[carry] = int(side)
    # ``run_id`` labels each contiguous zero run; a signal starts carrying
    # the side only from its first holding day in that run.
    return state


def _event_own_state(panel: pd.DataFrame, entry_signal: np.ndarray, side: int) -> np.ndarray:
    """Relabel only the actual entry-signal day in a base-zero state.

    This is the production state-adjustment semantics.  A later day in the
    same base-zero run stays 0 unless it has its own entry signal; the
    continuous ``holding_signal`` is intentionally not accepted here.
    """

    base = pd.to_numeric(panel["state"], errors="raise").astype(int).to_numpy()
    signal = np.asarray(entry_signal, dtype=bool)
    if len(signal) != len(base):
        raise ValueError("entry signal must align with the research panel")
    state = base.copy()
    zero_signal = (base == 0) & signal
    state[zero_signal] = int(side)
    return state


def _state_quality_phase(panel: pd.DataFrame, state: np.ndarray, side: int, phase: str) -> dict[str, Any]:
    """Return causal own-side state-path quality metrics for Dev/Val only."""

    mask = _phase_mask(panel, phase)
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        return {
            "zero_segment_count": 0,
            "zero_mean_length": np.nan,
            "zero_one_day_share": np.nan,
            "target_daily_mean": np.nan,
            "target_daily_win": np.nan,
            "target_segment_win": np.nan,
            "target_segment_count": 0,
            "negative_zero_one_day_share": np.nan,
        }
    lo, hi = int(positions[0]), int(positions[-1])
    values = np.asarray(state, dtype=int)
    starts = np.r_[True, values[1:] != values[:-1]]
    start_positions = np.flatnonzero(starts)
    end_positions = np.r_[start_positions[1:] - 1, len(values) - 1]
    segment_values = values[start_positions]
    lengths = end_positions - start_positions + 1
    complete = (end_positions >= lo) & (start_positions <= hi)
    zero_lengths = lengths[complete & (segment_values == 0)]
    returns = panel["o2o_h1"].to_numpy(dtype=float, na_value=np.nan)
    target_mask = mask & (values == int(side)) & np.isfinite(returns)
    directional = returns[target_mask] * float(side)
    segment_returns: list[float] = []
    for start, end, value in zip(start_positions[complete], end_positions[complete], segment_values[complete]):
        if int(value) != int(side):
            continue
        segment_values_raw = returns[int(start) : int(end) + 1]
        segment_values_raw = segment_values_raw[np.isfinite(segment_values_raw)]
        if len(segment_values_raw):
            segment_returns.append(float(np.prod(1.0 + float(side) * segment_values_raw) - 1.0))
    segment_array = np.asarray(segment_returns, dtype=float)
    return {
        "zero_segment_count": int(len(zero_lengths)),
        "zero_mean_length": float(np.mean(zero_lengths)) if len(zero_lengths) else np.nan,
        "zero_one_day_share": float(np.mean(zero_lengths == 1)) if len(zero_lengths) else np.nan,
        "target_daily_mean": float(np.mean(directional)) if len(directional) else np.nan,
        "target_daily_win": float(np.mean(directional > 0.0)) if len(directional) else np.nan,
        "target_segment_win": float(np.mean(segment_array > 0.0)) if len(segment_array) else np.nan,
        "target_segment_count": int(len(segment_array)),
        "negative_zero_one_day_share": -float(np.mean(zero_lengths == 1)) if len(zero_lengths) else np.nan,
    }


def _state_quality_for_candidate(panel: pd.DataFrame, entry_signal: np.ndarray, side: int) -> dict[str, Any]:
    state = _event_own_state(panel, entry_signal, side)
    return {
        "development": _state_quality_phase(panel, state, side, "development"),
        "validation": _state_quality_phase(panel, state, side, "validation"),
    }


def _state_quality_gate(row: dict[str, Any]) -> bool:
    """Strict full-grid Dev/Val gate; no Test field is read here."""

    for phase in ("development", "validation"):
        signal = row[phase]
        state = row.get(f"state_quality_{phase}", {})
        if not (
            signal.get("selected_days", 0) >= QUALITY_MIN_SELECTED_DAYS
            and np.isfinite(signal["mean_directional_o2o_h1"])
            and signal["mean_directional_o2o_h1"] > 0.0
            and np.isfinite(signal["mean_directional_o2o_h3"])
            and signal["mean_directional_o2o_h3"] > 0.0
            and np.isfinite(signal.get("h1_hit_rate", np.nan))
            and signal["h1_hit_rate"] >= 0.50
            and np.isfinite(signal["target_transition_lift_vs_eligible"])
            and signal["target_transition_lift_vs_eligible"] > 0.0
            and np.isfinite(signal["rank_ic"])
            and signal["rank_ic"] >= 0.0
            and np.isfinite(signal["transition_purity"])
            and signal["transition_purity"] >= 0.60
            and np.isfinite(state.get("target_daily_mean", np.nan))
            and state["target_daily_mean"] > 0.0
            and np.isfinite(state.get("target_daily_win", np.nan))
            and state["target_daily_win"] >= 0.50
            and np.isfinite(state.get("target_segment_win", np.nan))
            and state["target_segment_win"] >= 0.50
            and np.isfinite(state.get("zero_one_day_share", np.nan))
            and state["zero_one_day_share"] <= 0.15
            and np.isfinite(state.get("zero_mean_length", np.nan))
            and 3.0 <= state["zero_mean_length"] <= 25.0
        ):
            return False
        holding = row.get(f"holding_quality_{phase}", {})
        if not (
            holding.get("segment_count", 0) >= 5
            and np.isfinite(holding.get("segment_win_rate", np.nan))
            and holding["segment_win_rate"] >= HOLDING_QUALITY_MIN_SEGMENT_WIN
            and np.isfinite(holding.get("mean_segment_directional_return", np.nan))
            and holding["mean_segment_directional_return"] > 0.0
            and np.isfinite(holding.get("daily_mean_directional_return", np.nan))
            and holding["daily_mean_directional_return"] > 0.0
            and np.isfinite(holding.get("mean_length", np.nan))
            and 2.0 <= holding["mean_length"] <= 12.0
            and np.isfinite(holding.get("one_day_share", np.nan))
            and holding["one_day_share"] <= 0.10
        ):
            return False
    return True


def _state_quality_dominates(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    """Compare return/rank and state-path quality, not event prevalence.

    Transition rate and lift remain hard positive gates in
    ``_state_quality_gate``.  They are not dominance dimensions because a
    less fragmented holding path can legitimately fire fewer, better-timed
    transitions; treating prevalence as dominance would systematically favour
    over-triggered one-day remnants.
    """

    candidate_values: list[float] = []
    baseline_values: list[float] = []
    for scope, field in STATE_QUALITY_DOMINANCE_FIELDS:
        candidate_scope = candidate.get(scope, {})
        baseline_scope = baseline.get(scope, {})
        candidate_value = float(candidate_scope.get(field, np.nan))
        baseline_value = float(baseline_scope.get(field, np.nan))
        if not np.isfinite(candidate_value) or not np.isfinite(baseline_value):
            return False
        candidate_values.append(candidate_value)
        baseline_values.append(baseline_value)
    candidate_array = np.asarray(candidate_values, dtype=float)
    baseline_array = np.asarray(baseline_values, dtype=float)
    return bool(np.all(candidate_array >= baseline_array) and np.any(candidate_array > baseline_array))


def _holding_quality_phase(panel: pd.DataFrame, holding: np.ndarray, side: int, phase: str) -> dict[str, Any]:
    """Measure actual contiguous holding segments for a Dev/Val quality gate."""

    mask = _phase_mask(panel, phase)
    values = np.asarray(holding, dtype=bool)
    returns = panel["o2o_h1"].to_numpy(dtype=float, na_value=np.nan)
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    ends = np.r_[starts[1:] - 1, len(values) - 1]
    segment_returns: list[float] = []
    lengths: list[int] = []
    for start, end in zip(starts, ends):
        if not values[start] or not mask[int(start) : int(end) + 1].any():
            continue
        raw = returns[int(start) : int(end) + 1]
        raw = raw[np.isfinite(raw)]
        if len(raw):
            segment_returns.append(float(np.prod(1.0 + float(side) * raw) - 1.0))
            lengths.append(int(end - start + 1))
    values_return = np.asarray(segment_returns, dtype=float)
    phase_returns = returns[mask & values]
    directional = phase_returns * float(side)
    holding_days = int((mask & values).sum())
    return {
        "holding_days": holding_days,
        "segment_count": int(len(values_return)),
        "mean_length": float(np.mean(lengths)) if lengths else np.nan,
        "one_day_share": float(np.mean(np.asarray(lengths) == 1)) if lengths else np.nan,
        "mean_segment_directional_return": float(np.mean(values_return)) if len(values_return) else np.nan,
        "segment_win_rate": float(np.mean(values_return > 0.0)) if len(values_return) else np.nan,
        "daily_mean_directional_return": float(np.mean(directional)) if len(directional) else np.nan,
        "daily_win_rate": float(np.mean(directional > 0.0)) if len(directional) else np.nan,
    }


def _holding_quality_for_candidate(panel: pd.DataFrame, holding: np.ndarray, side: int) -> dict[str, Any]:
    return {
        "development": _holding_quality_phase(panel, holding, side, "development"),
        "validation": _holding_quality_phase(panel, holding, side, "validation"),
    }


def _holding_quality_gate(row: dict[str, Any]) -> bool:
    """Check the strict actual holding-segment component on Dev/Val."""

    for phase in ("development", "validation"):
        metrics = row.get(f"holding_quality_{phase}", {})
        if not (
            metrics.get("segment_count", 0) >= 5
            and np.isfinite(metrics.get("segment_win_rate", np.nan))
            and metrics["segment_win_rate"] >= HOLDING_QUALITY_MIN_SEGMENT_WIN
            and np.isfinite(metrics.get("mean_segment_directional_return", np.nan))
            and metrics["mean_segment_directional_return"] > 0.0
            and np.isfinite(metrics.get("one_day_share", np.nan))
            and metrics["one_day_share"] <= 0.10
            and np.isfinite(metrics.get("mean_length", np.nan))
            and 2.0 <= metrics["mean_length"] <= 12.0
            and np.isfinite(metrics.get("daily_mean_directional_return", np.nan))
            and metrics["daily_mean_directional_return"] > 0.0
        ):
            return False
    return True


_QUALITY_SCORE_WEIGHTS = {
    "h1": 0.13,
    "h3": 0.10,
    "hit": 0.05,
    "lift": 0.12,
    "purity": 0.08,
    "state_daily": 0.11,
    "state_segment_win": 0.08,
    "holding_win": 0.12,
    "holding_return": 0.10,
    "holding_daily": 0.05,
    "zero_continuity": 0.06,
}


def _quality_dimensions(row: dict[str, Any]) -> dict[str, float]:
    """Worst-phase Dev/Val dimensions for quality-aware freezing."""

    def finite(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def worst(scope: str, field: str) -> float:
        if scope == "metrics":
            values = (row["development"], row["validation"])
        elif scope == "state":
            values = (row["state_quality_development"], row["state_quality_validation"])
        else:
            values = (row["holding_quality_development"], row["holding_quality_validation"])
        return float(min(finite(values[0].get(field)), finite(values[1].get(field))))

    return {
        "h1": worst("metrics", "mean_directional_o2o_h1"),
        "h3": worst("metrics", "mean_directional_o2o_h3"),
        "hit": worst("metrics", "h1_hit_rate"),
        "lift": worst("metrics", "target_transition_lift_vs_eligible"),
        "purity": worst("metrics", "transition_purity"),
        "state_daily": worst("state", "target_daily_mean"),
        "state_segment_win": worst("state", "target_segment_win"),
        "holding_win": worst("holding", "segment_win_rate"),
        "holding_return": worst("holding", "mean_segment_directional_return"),
        "holding_daily": worst("holding", "daily_mean_directional_return"),
        "zero_continuity": 1.0 - max(
            finite(row["state_quality_development"].get("zero_one_day_share")),
            finite(row["state_quality_validation"].get("zero_one_day_share")),
        ),
    }


def _quality_percentile_score(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    names = tuple(_QUALITY_SCORE_WEIGHTS)
    arrays = {
        name: pd.Series([row.get("quality_dimensions", {}).get(name, np.nan) for row in rows], dtype=float)
        for name in names
    }
    ranks = {name: values.rank(pct=True, na_option="bottom").to_numpy(dtype=float) for name, values in arrays.items()}
    for position, row in enumerate(rows):
        row["quality_selection_score"] = float(sum(_QUALITY_SCORE_WEIGHTS[name] * ranks[name][position] for name in names))


def _quality_rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(bool(row.get("research_gate_pass", False))),
        float(row.get("quality_selection_score", -np.inf)),
        float(row.get("quality_dimensions", {}).get("holding_win", -np.inf)),
        float(row.get("quality_dimensions", {}).get("zero_continuity", -np.inf)),
        int(row["candidate"].get("selected_total", 0)),
        str(row["candidate"]["candidate_id"]),
    )


def _score_cache(
    spot: pd.DataFrame,
    panel: pd.DataFrame,
    candidates: Iterable[dict[str, Any]],
    side: int,
    progress: ProgressFn | None = None,
) -> dict[str, pd.DataFrame]:
    versions = sorted({str(row["source_version"]) for row in candidates}, key=lambda value: int(value[1:]))
    cache: dict[str, pd.DataFrame] = {}
    for number, version in enumerate(versions, start=1):
        if progress:
            progress(f"阶段 2/5：计算核心逻辑分数 {number}/{len(versions)}（{version}）")
        scores = compute_logic_scores(version, spot, panel, side)
        if "formation_date" in scores.columns:
            scores = scores.set_index("formation_date")
        scores.index = pd.to_datetime(scores.index)
        cache[version] = scores.reindex(pd.to_datetime(panel["formation_date"])).reset_index(drop=True)
    return cache


def _selection_score(rows: list[dict[str, Any]], score_key: str = "selection_score") -> None:
    def safe_nanmean(values: Iterable[float]) -> float:
        array = np.asarray(list(values), dtype=float)
        return float(np.nanmean(array)) if np.isfinite(array).any() else np.nan

    def safe_nanmin(values: Iterable[float]) -> float:
        array = np.asarray(list(values), dtype=float)
        return float(np.nanmin(array)) if np.isfinite(array).any() else np.nan

    metric_names = {
        "dev_mean_directional_o2o_h1": "development",
        "val_mean_directional_o2o_h1": "validation",
        "pooled_mean_directional_o2o_h1": None,
        "worst_phase_mean_directional_o2o_h1": None,
        "pooled_h1_hit_rate": None,
        "worst_phase_rank_ic": None,
        "pooled_mean_directional_o2o_h3": None,
        "h1_h3_path_consistency": None,
        "pooled_target_transition_rate": None,
        "pooled_transition_purity": None,
        "one_minus_rapid_restore_rate": None,
    }
    raw: dict[str, list[float]] = {name: [] for name in metric_names}
    for row in rows:
        dev, val = row["development"], row["validation"]
        raw["dev_mean_directional_o2o_h1"].append(dev["mean_directional_o2o_h1"])
        raw["val_mean_directional_o2o_h1"].append(val["mean_directional_o2o_h1"])
        raw["pooled_mean_directional_o2o_h1"].append(safe_nanmean([dev["mean_directional_o2o_h1"], val["mean_directional_o2o_h1"]]))
        raw["worst_phase_mean_directional_o2o_h1"].append(safe_nanmin([dev["mean_directional_o2o_h1"], val["mean_directional_o2o_h1"]]))
        raw["pooled_h1_hit_rate"].append(safe_nanmean([dev["h1_hit_rate"], val["h1_hit_rate"]]))
        raw["worst_phase_rank_ic"].append(safe_nanmin([dev["rank_ic"], val["rank_ic"]]))
        raw["pooled_mean_directional_o2o_h3"].append(safe_nanmean([dev["mean_directional_o2o_h3"], val["mean_directional_o2o_h3"]]))
        raw["h1_h3_path_consistency"].append(safe_nanmean([dev["path_consistency_h1_h3"], val["path_consistency_h1_h3"]]))
        raw["pooled_target_transition_rate"].append(safe_nanmean([dev["target_transition_rate"], val["target_transition_rate"]]))
        raw["pooled_transition_purity"].append(safe_nanmean([dev["transition_purity"], val["transition_purity"]]))
        raw["one_minus_rapid_restore_rate"].append(1.0 - safe_nanmean([dev["rapid_restore_rate"], val["rapid_restore_rate"]]))
    ranks = {name: pd.Series(values, dtype=float).rank(pct=True, na_option="bottom").to_numpy() for name, values in raw.items()}
    for i, row in enumerate(rows):
        row[score_key] = float(sum(JOINT_SCORE_WEIGHTS[name] * ranks[name][i] for name in metric_names))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _signal_overlap_rows(
    logic_records: list[dict[str, Any]],
    selected_by_method: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Describe overlap of the frozen local-Top1 entry paths.

    This is a diagnostic on the actual selected-day paths, not a selection
    criterion.  It is intentionally computed after local Dev/Val freezing and
    never feeds back into either the official global Top1 or local Top1.
    """
    record_by_method = {str(row["method_key"]): row for row in logic_records}
    methods = sorted(selected_by_method)
    rows: list[dict[str, Any]] = []
    for left_pos, left_method in enumerate(methods):
        left = np.asarray(selected_by_method[left_method], dtype=bool)
        for right_method in methods[left_pos + 1 :]:
            right = np.asarray(selected_by_method[right_method], dtype=bool)
            intersection = int(np.logical_and(left, right).sum())
            union = int(np.logical_or(left, right).sum())
            left_days = int(left.sum())
            right_days = int(right.sum())
            smaller_days = min(left_days, right_days)
            left_centered = left.astype(float) - float(left.mean())
            right_centered = right.astype(float) - float(right.mean())
            denominator = float(np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered)))
            phi = float(np.dot(left_centered, right_centered) / denominator) if denominator > 0 else np.nan
            left_record = record_by_method[left_method]
            right_record = record_by_method[right_method]
            rows.append(
                {
                    "left_method_key": left_method,
                    "left_core_logic_name": left_record["core_logic_name"],
                    "left_source_version": left_record["source_version"],
                    "left_candidate_id": left_record["candidate_id"],
                    "right_method_key": right_method,
                    "right_core_logic_name": right_record["core_logic_name"],
                    "right_source_version": right_record["source_version"],
                    "right_candidate_id": right_record["candidate_id"],
                    "left_selected_days": left_days,
                    "right_selected_days": right_days,
                    "intersection_days": intersection,
                    "union_days": union,
                    "jaccard_overlap": float(intersection / union) if union else np.nan,
                    "overlap_rate_of_smaller": float(intersection / smaller_days) if smaller_days else np.nan,
                    "binary_phi_correlation": phi,
                    "exact_same_path": bool(np.array_equal(left, right)),
                }
            )
    return rows


def run_company_side(
    side: str,
    spot: pd.DataFrame,
    panel: pd.DataFrame,
    output_dir: str | Path,
    *,
    show_progress: bool = True,
    progress_every: int = 10000,
) -> dict[str, Any]:
    """Rebuild one side's full grid, freeze one company Top1, then show Test.

    Progress is printed at coarse checkpoints so a remote Notebook user can
    see that the long candidate scan is advancing.
    """
    if side not in ("down", "up"):
        raise ValueError("side must be down or up")
    side_value = -1 if side == "down" else 1
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def progress(message: str) -> None:
        if show_progress:
            print(f"[{side}] {message}", flush=True)

    progress("阶段 1/5：构造本侧完整候选网格")
    candidates = _candidates(side)
    progress(f"阶段 1/5 完成：候选数 {len(candidates):,}")
    scores = _score_cache(spot, panel, candidates, side_value, progress=progress)
    rows: list[dict[str, Any]] = []
    # Keep one row for every raw candidate so percentile ranks match the local
    # full 6,144-grid audit.  Only rows meeting ordinary Dev/Val computability
    # are used by the legacy transition diagnostics; non-computable rows remain
    # explicit quality-gate failures rather than disappearing before ranking.
    quality_rows: list[dict[str, Any]] = []
    progress("阶段 3/5：在本侧候选内扫描 Development+Validation")
    for number, candidate in enumerate(candidates, start=1):
        score_frame = scores[candidate["source_version"]]
        score = score_frame.iloc[:, int(candidate["score_variant_number"])].to_numpy(dtype=float)
        selected, threshold = _signal_path(panel, score, candidate)
        dev = _metrics(panel, score, selected, side_value, "development", int(candidate["min_zero_age"]))
        val = _metrics(panel, score, selected, side_value, "validation", int(candidate["min_zero_age"]))
        candidate["selected_total"] = int(selected.sum())
        row = {"candidate": candidate, "threshold": threshold, "development": dev, "validation": val}
        quality_rows.append(row)
        if dev["selected_days"] >= MINIMUM_COMPUTABILITY["development_selected_days"] and val["selected_days"] >= MINIMUM_COMPUTABILITY["validation_selected_days"]:
            rows.append(row)
        if show_progress and (number == 1 or number % max(1, progress_every) == 0 or number == len(candidates)):
            print(f"[{side}] 阶段 3/5：已扫描 {number:,}/{len(candidates):,}，Dev/Val 可计算 {len(rows):,}", flush=True)
    if not rows:
        raise ValueError(f"No company-computable candidates for {side}")
    _selection_score(rows)
    logic_top_rows: list[dict[str, Any]] = []
    for method_key in sorted({row["candidate"]["method_key"] for row in rows}):
        logic_rows = [row for row in rows if row["candidate"]["method_key"] == method_key]
        _selection_score(logic_rows, score_key="logic_selection_score")
        logic_rows.sort(
            key=lambda row: (
                -float(row["logic_selection_score"]),
                -float(np.nanmean([row["development"]["target_transition_rate"], row["validation"]["target_transition_rate"]])) if np.isfinite(np.nanmean([row["development"]["target_transition_rate"], row["validation"]["target_transition_rate"]])) else np.inf,
                -float(row["validation"]["mean_directional_o2o_h1"]) if np.isfinite(row["validation"]["mean_directional_o2o_h1"]) else np.inf,
                str(row["candidate"]["candidate_id"]),
            )
        )
        logic_top_rows.append(logic_rows[0])
    rows.sort(
        key=lambda row: (
            -float(row["selection_score"]),
            -float(np.nanmean([row["development"]["target_transition_rate"], row["validation"]["target_transition_rate"]])) if np.isfinite(np.nanmean([row["development"]["target_transition_rate"], row["validation"]["target_transition_rate"]])) else np.inf,
            -float(np.nanmean([row["development"]["target_transition_lift_vs_eligible"], row["validation"]["target_transition_lift_vs_eligible"]])) if np.isfinite(np.nanmean([row["development"]["target_transition_lift_vs_eligible"], row["validation"]["target_transition_lift_vs_eligible"]])) else np.inf,
            -float(row["validation"]["mean_directional_o2o_h1"]) if np.isfinite(row["validation"]["mean_directional_o2o_h1"]) else np.inf,
            str(row["candidate"]["candidate_id"]),
        )
    )
    # Keep the transition-aware ranking as the baseline, then inspect only a
    # bounded Dev/Val shortlist for a strict state-quality dominator.  This is
    # deliberately after the primary ranking and never reads Test metrics.
    baseline_top = rows[0]
    state_quality_shortlist = quality_rows[: min(STATE_QUALITY_SHORTLIST_SIZE, len(quality_rows))]
    progress(
        f"阶段 3/5：对排名前 {len(state_quality_shortlist):,} 个候选做 Dev/Val 状态质量审计"
    )
    for number, row in enumerate(state_quality_shortlist, start=1):
        dev = row["development"]
        val = row["validation"]
        plausible = all(
            phase.get("selected_days", 0) >= QUALITY_MIN_SELECTED_DAYS
            and np.isfinite(phase.get("mean_directional_o2o_h1", np.nan))
            and phase["mean_directional_o2o_h1"] > 0.0
            and np.isfinite(phase.get("mean_directional_o2o_h3", np.nan))
            and phase["mean_directional_o2o_h3"] > 0.0
            and np.isfinite(phase.get("target_transition_lift_vs_eligible", np.nan))
            and phase["target_transition_lift_vs_eligible"] > 0.0
            for phase in (dev, val)
        )
        if not plausible:
            row["state_quality_development"] = {}
            row["state_quality_validation"] = {}
            row["holding_quality_development"] = {}
            row["holding_quality_validation"] = {}
            row["quality_dimensions"] = {}
            row["research_gate_pass"] = False
            continue
        state_candidate = row["candidate"]
        state_score = scores[state_candidate["source_version"]].iloc[:, int(state_candidate["score_variant_number"])].to_numpy(dtype=float)
        state_selected, _, state_holding = _signal_path(panel, state_score, state_candidate, return_holding=True)
        state_quality = _state_quality_for_candidate(panel, state_selected, side_value)
        holding_quality = _holding_quality_for_candidate(panel, state_holding, side_value)
        row["state_quality_development"] = state_quality["development"]
        row["state_quality_validation"] = state_quality["validation"]
        row["holding_quality_development"] = holding_quality["development"]
        row["holding_quality_validation"] = holding_quality["validation"]
        if show_progress and (number == 1 or number == len(state_quality_shortlist) or number % 128 == 0):
            print(
                f"[{side}] 阶段 3/5：状态质量已审计 {number:,}/{len(state_quality_shortlist):,}",
                flush=True,
            )
    # Quality dimensions and the strict research gate are computed for every
    # Dev/Val-computable row, not only for the transition-primary Top20.  This
    # is the production counterpart of the local full-grid reselection.
    for row in state_quality_shortlist:
        if row.get("state_quality_development"):
            row["quality_dimensions"] = _quality_dimensions(row)
            row["research_gate_pass"] = _state_quality_gate(row)
    _quality_percentile_score(quality_rows)
    baseline_state_quality_gate_pass = _state_quality_gate(baseline_top)
    gate_candidates = [
        row
        for row in state_quality_shortlist
        if row.get("research_gate_pass", False)
    ]
    holding_quality_candidates = [
        row
        for row in gate_candidates
        if _holding_quality_gate(row)
    ]
    dominators = [
        row
        for row in gate_candidates
        if _state_quality_dominates(row, baseline_top)
    ]
    state_quality_override_applied = bool(holding_quality_candidates) and not any(
        row is baseline_top for row in holding_quality_candidates
    )
    if holding_quality_candidates:
        top = max(holding_quality_candidates, key=_quality_rank_key)
        selection_policy = "QUALITY_AWARE_FULL_GRID_DEV_VAL_ONLY"
        progress(
            f"阶段 3/5：在 {len(holding_quality_candidates):,} 个三阶段质量候选中采用 {top['candidate']['candidate_id']}"
        )
    elif dominators:
        # ``rows`` is already in the primary ranking order, so taking the
        # first dominator preserves the original score as a tie-breaker.
        top = next(row for row in rows if any(row is item for item in dominators))
        selection_policy = "STATE_QUALITY_DOMINANCE_GUARD_DEV_VAL_ONLY"
        progress(
            f"阶段 3/5：发现 {len(dominators):,} 个严格状态质量支配候选；采用 {top['candidate']['candidate_id']}"
        )
    elif not baseline_state_quality_gate_pass and holding_quality_candidates:
        top = holding_quality_candidates[0]
        selection_policy = "STATE_AND_HOLDING_QUALITY_GATE_FALLBACK_DEV_VAL_ONLY"
        progress(
            f"阶段 3/5：主排序 Top1 未通过事件状态质量门；在 {len(holding_quality_candidates):,} 个 Dev/Val 双质量候选中采用 {top['candidate']['candidate_id']}"
        )
    elif not baseline_state_quality_gate_pass and gate_candidates:
        # If the primary Top1 itself fails the event-only state-quality gate,
        # do not let a fragmented zero path win merely because its transition
        # score is a little higher.  Choose the highest-ranked gate-passing
        # candidate, still using only Dev/Val and preserving the primary score
        # as the tie-breaker.  This is a quality fallback, never a Test read.
        top = gate_candidates[0]
        selection_policy = "STATE_QUALITY_GATE_FALLBACK_DEV_VAL_ONLY"
        progress(
            f"阶段 3/5：主排序 Top1 未通过事件状态质量门；在 {len(gate_candidates):,} 个 Dev/Val 通过候选中采用 {top['candidate']['candidate_id']}"
        )
    else:
        top = baseline_top
        selection_policy = "TRANSITION_AWARE_DV_PRIMARY_NO_DOMINATOR"
        progress("阶段 3/5：未发现严格状态质量支配候选；保留转移主排序 Top1")
    candidate = top["candidate"]
    score = scores[candidate["source_version"]].iloc[:, int(candidate["score_variant_number"])].to_numpy(dtype=float)
    selected, threshold, holding_selected = _signal_path(panel, score, candidate, return_holding=True)
    selected_state = _event_own_state(panel, selected, side_value)
    selected_state_quality = {
        phase: _state_quality_phase(panel, selected_state, side_value, phase)
        for phase in ("full", "development", "validation", "test")
    }
    progress(f"阶段 4/5：冻结前排序完成；本侧 Top1 候选为 {candidate['candidate_id']}，Test 仍锁定")

    top20 = []
    display_rows = [top] + [row for row in rows if row is not top][:19]
    for row in display_rows:
        top20.append(
            {
                "candidate_id": row["candidate"]["candidate_id"],
                "source_version": row["candidate"]["source_version"],
                "method_key": row["candidate"]["method_key"],
                "score_variant_number": row["candidate"]["score_variant_number"],
                "min_zero_age": row["candidate"]["min_zero_age"],
                "entry_quantile": row["candidate"]["entry_quantile"],
                "confirmation_days": row["candidate"]["confirmation_days"],
                "holding_package": row["candidate"]["holding_package"]["package_id"],
                "selection_score": row["selection_score"],
                "development": row["development"],
                "validation": row["validation"],
                "state_quality_development": row.get("state_quality_development"),
                "state_quality_validation": row.get("state_quality_validation"),
                "state_quality_gate_pass": _state_quality_gate(row) if "state_quality_development" in row else False,
                "holding_quality_development": row.get("holding_quality_development"),
                "holding_quality_validation": row.get("holding_quality_validation"),
                "holding_quality_gate_pass": _holding_quality_gate(row) if "holding_quality_development" in row else False,
                "state_quality_dominates_baseline": any(row is item for item in dominators),
            }
        )
    (output / "company_top20.csv").write_text(pd.DataFrame(top20).to_csv(index=False), encoding="utf-8")
    pre_freeze = {
        "selection_scope": "company_independent_within_side_only_after_post_90_registry",
        "side": side,
        "pool_rule_id": POOL_RULE_ID,
        "candidate_count_input": len(candidates),
        "candidate_count_computable": len(rows),
        "selection_policy": selection_policy,
        "baseline_top_candidate_id": baseline_top["candidate"]["candidate_id"],
        "baseline_state_quality_gate_pass": baseline_state_quality_gate_pass,
        "state_quality_gate_candidate_count": len(gate_candidates),
        "holding_quality_gate_candidate_count": len(holding_quality_candidates),
        "state_quality_override_applied": state_quality_override_applied,
        "state_quality_shortlist_count": len(state_quality_shortlist),
        "state_quality_dominator_count": len(dominators),
        "holding_quality_min_segment_win": HOLDING_QUALITY_MIN_SEGMENT_WIN,
        "minimum_computability": dict(MINIMUM_COMPUTABILITY),
        "selected_candidate_preview": {
            "candidate_id": candidate["candidate_id"],
            "source_version": candidate["source_version"],
            "source_side": candidate["source_side"],
            "method_key": candidate["method_key"],
            "core_logic_name": candidate["core_logic_name"],
            "score_variant_number": candidate["score_variant_number"],
            "score_variant": candidate["score_variant"],
            "min_zero_age": candidate["min_zero_age"],
            "entry_quantile": candidate["entry_quantile"],
            "confirmation_days": candidate["confirmation_days"],
            "holding_package": candidate["holding_package"],
            "threshold_from_development": threshold,
        },
        "development": top["development"],
        "validation": top["validation"],
        "state_quality": selected_state_quality,
        "top20_file": "company_top20.csv",
        "logic_top1_file": "company_logic_top1.csv",
        "logic_count": len(logic_top_rows),
        "test_locked_until_after_freeze": True,
        "test_used_for_selection": False,
    }
    (output / "company_pre_freeze.json").write_text(json.dumps(_jsonable(pre_freeze), ensure_ascii=False, indent=2), encoding="utf-8")

    progress("阶段 5/5：冻结参数已写入，现开始计算冻结后的 Test")
    test = _metrics(panel, score, selected, side_value, "test", int(candidate["min_zero_age"]))
    full = _metrics(panel, score, selected, side_value, "full", int(candidate["min_zero_age"]))
    progress(f"阶段 5/5：计算 {len(logic_top_rows)} 个逻辑局部 Top1 的冻结后 Test 诊断")
    logic_records: list[dict[str, Any]] = []
    logic_flat: list[dict[str, Any]] = []
    logic_selected_by_method: dict[str, np.ndarray] = {}
    for logic_row in logic_top_rows:
        logic_candidate = logic_row["candidate"]
        logic_score = scores[logic_candidate["source_version"]].iloc[:, int(logic_candidate["score_variant_number"])].to_numpy(dtype=float)
        logic_selected, logic_threshold = _signal_path(panel, logic_score, logic_candidate)
        logic_selected_by_method[str(logic_candidate["method_key"])] = logic_selected.copy()
        logic_full = _metrics(panel, logic_score, logic_selected, side_value, "full", int(logic_candidate["min_zero_age"]))
        logic_test = _metrics(panel, logic_score, logic_selected, side_value, "test", int(logic_candidate["min_zero_age"]))
        record = {
            "method_key": logic_candidate["method_key"],
            "core_logic_name": logic_candidate["core_logic_name"],
            "source_version": logic_candidate["source_version"],
            "candidate_id": logic_candidate["candidate_id"],
            "score_variant_number": logic_candidate["score_variant_number"],
            "score_variant": logic_candidate["score_variant"],
            "min_zero_age": logic_candidate["min_zero_age"],
            "entry_quantile": logic_candidate["entry_quantile"],
            "confirmation_days": logic_candidate["confirmation_days"],
            "holding_package": logic_candidate["holding_package"],
            "threshold_from_development": logic_threshold,
            "logic_selection_score": logic_row["logic_selection_score"],
            "development": logic_row["development"],
            "validation": logic_row["validation"],
            "full": logic_full,
            "test": logic_test,
            "selection_scope": "within_method_key_using_development_validation_only",
            "test_used_for_selection": False,
        }
        logic_records.append(record)
        flat = {
            key: value
            for key, value in record.items()
            if key not in {"score_variant", "holding_package", "development", "validation", "full", "test"}
        }
        flat["holding_package"] = logic_candidate["holding_package"]["package_id"]
        for phase_name, metrics in (("development", record["development"]), ("validation", record["validation"]), ("full", record["full"]), ("test", record["test"])):
            for metric_name, metric_value in metrics.items():
                flat[f"{phase_name}_{metric_name}"] = metric_value
        logic_flat.append(_jsonable(flat))
    overlap_rows = _signal_overlap_rows(logic_records, logic_selected_by_method)
    overlap_values = [row["jaccard_overlap"] for row in overlap_rows if np.isfinite(row["jaccard_overlap"])]
    overlap_summary = {
        "selection_scope": "frozen_local_top1_entry_path_diagnostic_only",
        "side": side,
        "logic_count": len(logic_records),
        "pair_count": len(overlap_rows),
        "test_used_for_selection": False,
        "official_global_top1_unchanged": bool(not state_quality_override_applied),
        "heuristic_high_overlap_jaccard_threshold": 0.50,
        "heuristic_high_overlap_pair_count": int(sum(value >= 0.50 for value in overlap_values)),
        "max_jaccard_overlap": float(max(overlap_values)) if overlap_values else None,
        "csv_file": "company_logic_signal_overlap.csv",
        "rows": overlap_rows,
    }
    (output / "company_logic_signal_overlap.json").write_text(json.dumps(_jsonable(overlap_summary), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(overlap_rows).to_csv(output / "company_logic_signal_overlap.csv", index=False)
    logic_summary = {
        "selection_scope": "one_local_top1_per_method_key_selected_on_development_validation_only",
        "side": side,
        "logic_count": len(logic_records),
        "test_used_for_selection": False,
        "official_global_top1_unchanged": bool(not state_quality_override_applied),
        "csv_file": "company_logic_top1.csv",
        "signal_overlap_file": "company_logic_signal_overlap.csv",
        "signal_overlap_json_file": "company_logic_signal_overlap.json",
        "rows": logic_records,
    }
    (output / "company_logic_top1.json").write_text(json.dumps(_jsonable(logic_summary), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(logic_flat).to_csv(output / "company_logic_top1.csv", index=False)
    dates = pd.to_datetime(panel["formation_date"])
    phase_dates = {
        "full": {"start": str(dates.min().date()), "end": str(dates.max().date())},
    }
    for name, bounds in PHASES.items():
        phase_mask = _phase_mask(panel, name)
        phase_dates[name] = (
            {"start": str(dates[phase_mask].min().date()), "end": str(dates[phase_mask].max().date())}
            if phase_mask.any()
            else {"start": str(bounds[0].date()), "end": str(bounds[1].date())}
        )
    progress(f"阶段 5/5 完成：Test selected_days={test['selected_days']}，directional_O2O_H1={test['mean_directional_o2o_h1']:.6f}")
    # Post-freeze state relabelling audit.  It is written outside the
    # selection path and may be combined with the opposite side after the
    # other company notebook has run.
    signal_path = output.parent / f"company_signal_{side}.parquet"
    pd.DataFrame({"formation_date": pd.to_datetime(panel["formation_date"]), "entry_signal": selected.astype(np.int8), "holding_signal": holding_selected.astype(np.int8)}).to_parquet(signal_path, index=False)
    opposite = "up" if side == "down" else "down"
    opposite_path = output.parent / f"company_signal_{opposite}.parquet"
    if opposite_path.is_file():
        opposite_frame = pd.read_parquet(opposite_path)
        opposite_frame["formation_date"] = pd.to_datetime(opposite_frame["formation_date"], errors="raise")
        own_frame = pd.read_parquet(signal_path)
        own_frame["formation_date"] = pd.to_datetime(own_frame["formation_date"], errors="raise")
        joined = own_frame.merge(opposite_frame, on="formation_date", how="outer", suffixes=("_own", "_opposite")).sort_values("formation_date")
        aligned_panel = panel.copy().sort_values("formation_date").reset_index(drop=True)
        if joined["formation_date"].tolist() != pd.to_datetime(aligned_panel["formation_date"]).tolist():
            raise ValueError("company side signal dates do not align for state-adjustment diagnostics")
        if side == "down":
            down_entry = joined["entry_signal_own"].fillna(0).to_numpy(dtype=bool)
            up_entry = joined["entry_signal_opposite"].fillna(0).to_numpy(dtype=bool)
            down_holding = joined["holding_signal_own"].fillna(0).to_numpy(dtype=bool)
            up_holding = joined["holding_signal_opposite"].fillna(0).to_numpy(dtype=bool)
        else:
            down_entry = joined["entry_signal_opposite"].fillna(0).to_numpy(dtype=bool)
            up_entry = joined["entry_signal_own"].fillna(0).to_numpy(dtype=bool)
            down_holding = joined["holding_signal_opposite"].fillna(0).to_numpy(dtype=bool)
            up_holding = joined["holding_signal_own"].fillna(0).to_numpy(dtype=bool)
        adjustment = build_state_adjustment_diagnostics(
            aligned_panel,
            down_entry,
            up_entry,
            down_holding=down_holding,
            up_holding=up_holding,
        )
        adjustment["signal_semantics"] = "formal_state_entry_signal_only; actual_holding_path_audited_separately"
        adjustment["signal_sources"] = {"down": str(output.parent / "company_signal_down.parquet"), "up": str(output.parent / "company_signal_up.parquet")}
        adjustment["pair_complete"] = True
        adjustment_path = output.parent / "company_state_adjustment_diagnostics.json"
    else:
        zeros = np.zeros(len(panel), dtype=bool)
        adjustment = build_state_adjustment_diagnostics(
            panel,
            selected if side == "down" else zeros,
            selected if side == "up" else zeros,
            down_holding=holding_selected if side == "down" else zeros,
            up_holding=holding_selected if side == "up" else zeros,
        )
        adjustment["signal_semantics"] = "formal_state_entry_signal_only; actual_holding_path_audited_separately"
        adjustment["signal_sources"] = {side: str(signal_path), opposite: None}
        adjustment["pair_complete"] = False
        adjustment_path = output / "company_state_adjustment_diagnostics.json"
    adjustment_path.write_text(json.dumps(_jsonable(adjustment), ensure_ascii=False, indent=2), encoding="utf-8")
    selected_holding_quality = {
        phase: _holding_quality_phase(panel, holding_selected, side_value, phase)
        for phase in ("full", "development", "validation", "test")
    }
    freeze = {
        "selection_scope": "company_independent_within_side_only_after_post_90_registry",
        "side": side,
        "pool_rule_id": POOL_RULE_ID,
        "admitted_source_versions": list(POOL_VERSION_SIDES[side]),
        "candidate_id": candidate["candidate_id"],
        "selection_policy": selection_policy,
        "baseline_top_candidate_id": baseline_top["candidate"]["candidate_id"],
        "baseline_state_quality_gate_pass": baseline_state_quality_gate_pass,
        "state_quality_gate_candidate_count": len(gate_candidates),
        "holding_quality_gate_candidate_count": len(holding_quality_candidates),
        "state_quality_override_applied": state_quality_override_applied,
        "state_quality_shortlist_count": len(state_quality_shortlist),
        "state_quality_dominator_count": len(dominators),
        "holding_quality_min_segment_win": HOLDING_QUALITY_MIN_SEGMENT_WIN,
        "source_version": candidate["source_version"],
        "source_side": side,
        "core_logic_name": candidate["core_logic_name"],
        "method_key": candidate["method_key"],
        "score_variant_number": candidate["score_variant_number"],
        "score_variant": candidate["score_variant"],
        "min_zero_age": candidate["min_zero_age"],
        "entry_quantile": candidate["entry_quantile"],
        "confirmation_days": candidate["confirmation_days"],
        "holding_package": candidate["holding_package"],
        "threshold_from_development": threshold,
        "candidate_count_input": len(candidates),
        "candidate_count_computable": len(rows),
        "development": top["development"],
        "validation": top["validation"],
        "state_quality": selected_state_quality,
        "holding_quality": selected_holding_quality,
        "full": full,
        "test": test,
        "phase_dates": phase_dates,
        "test_used_for_selection": False,
        "pre_freeze_summary_file": "company_pre_freeze.json",
        "top20_file": "company_top20.csv",
        "logic_top1_file": "company_logic_top1.csv",
        "logic_top1_json_file": "company_logic_top1.json",
        "logic_signal_overlap_file": "company_logic_signal_overlap.csv",
        "logic_signal_overlap_json_file": "company_logic_signal_overlap.json",
        "logic_count": len(logic_records),
        "state_adjustment_diagnostics_file": str(adjustment_path.name if adjustment_path.parent == output else adjustment_path),
        "state_adjustment_pair_complete": bool(adjustment.get("pair_complete")),
    }
    (output / "company_freeze.json").write_text(json.dumps(_jsonable(freeze), ensure_ascii=False, indent=2), encoding="utf-8")
    # When the opposite side has already finished, the pair-level diagnostic
    # is now complete.  Synchronize the earlier side's freeze metadata so
    # both side notebooks point to the same root-level audit artifact.
    if bool(adjustment.get("pair_complete")):
        for side_name in (side, opposite):
            side_dir = output.parent / ("01_down" if side_name == "down" else "02_up")
            side_freeze_path = side_dir / "company_freeze.json"
            if not side_freeze_path.is_file():
                continue
            side_freeze = json.loads(side_freeze_path.read_text(encoding="utf-8"))
            side_freeze["state_adjustment_pair_complete"] = True
            side_freeze["state_adjustment_diagnostics_file"] = adjustment_path.name
            side_freeze_path.write_text(json.dumps(_jsonable(side_freeze), ensure_ascii=False, indent=2), encoding="utf-8")
    return freeze


__all__ = ["run_company_side", "POOL_RULE_ID", "POOL_VERSION_SIDES"]
