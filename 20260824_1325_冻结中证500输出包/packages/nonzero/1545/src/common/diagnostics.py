"""Post-freeze diagnostics that never participate in candidate selection.

These diagnostics are intentionally separate from :mod:`candidates`: they are
computed only after a version-local Top1 has been written and are used to make
the notebooks explicit about uncertainty, state recovery, and the cost of
exiting during a strong continuation segment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .scores import exit_sign, side_state


def moving_block_bootstrap_mean(
    values: np.ndarray,
    *,
    repetitions: int = 200,
    block_length: int = 5,
    seed: int = 1545,
) -> dict[str, Any]:
    """Return a fixed-seed moving-block bootstrap for an ordered event series."""

    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    result: dict[str, Any] = {
        "repetitions": int(repetitions),
        "block_length": int(block_length),
        "seed": int(seed),
        "selection_use": False,
    }
    if len(clean) == 0:
        result.update({"n": 0, "mean_bp": np.nan, "p05_bp": np.nan, "median_bp": np.nan, "p95_bp": np.nan, "probability_positive": np.nan})
        return result
    if len(clean) == 1:
        value = float(clean[0] * 10_000)
        result.update({"n": 1, "mean_bp": value, "p05_bp": value, "median_bp": value, "p95_bp": value, "probability_positive": float(clean[0] > 0)})
        return result

    rng = np.random.default_rng(seed)
    block = max(1, min(int(block_length), len(clean)))
    starts = np.arange(len(clean))
    samples = np.empty(int(repetitions), dtype=float)
    for rep in range(int(repetitions)):
        picked: list[float] = []
        while len(picked) < len(clean):
            start = int(rng.choice(starts))
            for offset in range(block):
                picked.append(float(clean[(start + offset) % len(clean)]))
                if len(picked) >= len(clean):
                    break
        samples[rep] = float(np.mean(picked[: len(clean)]))
    samples_bp = samples * 10_000
    result.update({
        "n": int(len(clean)),
        "mean_bp": float(np.mean(clean) * 10_000),
        "p05_bp": float(np.quantile(samples_bp, 0.05)),
        "median_bp": float(np.quantile(samples_bp, 0.50)),
        "p95_bp": float(np.quantile(samples_bp, 0.95)),
        "probability_positive": float(np.mean(samples > 0)),
    })
    return result


def _state_at_dates(panel: pd.DataFrame, dates: pd.Series) -> pd.Series:
    lookup = panel["base_state"].copy()
    normalized = pd.to_datetime(dates, errors="coerce")
    values = lookup.reindex(normalized.to_numpy())
    return pd.Series(values.to_numpy(), index=dates.index, dtype="float64")


def recovery_diagnostics(panel: pd.DataFrame, signal: pd.Series, side: str) -> dict[str, Any]:
    """Measure whether a frozen exit is followed by a return to the source state."""

    triggered = signal.astype(bool)
    source = side_state(side)
    result: dict[str, Any] = {"side": side, "source_state": source, "n": int(triggered.sum())}
    if not triggered.any():
        for horizon in (1, 2, 3):
            result[f"recovery_original_state_rate_day_{horizon}"] = np.nan
        result["rapid_recovery_within_1_to_3_days_rate"] = np.nan
        return result

    state_frames: list[pd.Series] = []
    for horizon in (1, 2, 3):
        date_col = "effective_date" if horizon == 1 else f"exit_h{horizon - 1}_date"
        states = _state_at_dates(panel, panel.loc[triggered, date_col])
        same = states.eq(source)
        result[f"recovery_original_state_rate_day_{horizon}"] = float(same.mean()) if len(same) else np.nan
        state_frames.append(same.rename(f"day_{horizon}"))
    combined = pd.concat(state_frames, axis=1).fillna(False)
    result["rapid_recovery_within_1_to_3_days_rate"] = float(combined.any(axis=1).mean()) if len(combined) else np.nan
    return result


def excellent_segment_diagnostics(panel: pd.DataFrame, signal: pd.Series, side: str, period_mask: np.ndarray) -> dict[str, Any]:
    """Report early exits that occur during unusually strong continuation days.

    ``continuation_return`` is the signed return from maintaining the source
    state, so a positive value is a day on which exiting to zero gives up the
    original directional move.  The 75th percentile is calculated within the
    requested frozen diagnostic period and is never used for selection.
    """

    source = side_state(side)
    valid = (
        period_mask
        & panel["base_state"].eq(source).to_numpy()
        & panel["o2o_h1"].notna().to_numpy()
    )
    continuation = (-exit_sign(side) * panel["o2o_h1"]).to_numpy(float)
    population = continuation[valid]
    population = population[np.isfinite(population)]
    threshold = float(np.quantile(population, 0.75)) if len(population) else np.nan
    triggered = signal.to_numpy(bool) & valid & np.isfinite(continuation)
    values = continuation[triggered]
    return {
        "population_n": int(len(population)),
        "excellent_continuation_threshold_bp": threshold * 10_000 if np.isfinite(threshold) else np.nan,
        "trigger_n": int(len(values)),
        "excellent_segment_trigger_n": int(np.sum(values >= threshold)) if np.isfinite(threshold) else 0,
        "excellent_segment_damage_rate": float(np.mean(values >= threshold)) if len(values) and np.isfinite(threshold) else np.nan,
        "mean_continuation_return_given_trigger_bp": float(np.mean(values) * 10_000) if len(values) else np.nan,
    }


def post_freeze_diagnostics(panel: pd.DataFrame, signal: pd.Series, side: str, period_mask: np.ndarray) -> dict[str, Any]:
    """Build all diagnostics for a single frozen side and period."""

    improvement = (exit_sign(side) * panel["o2o_h1"]).to_numpy(float)
    triggered = signal.to_numpy(bool) & period_mask & np.isfinite(improvement)
    return {
        "bootstrap_o2o_h1": moving_block_bootstrap_mean(improvement[triggered]),
        "recovery": recovery_diagnostics(panel, signal.where(period_mask, False), side),
        "excellent_segment": excellent_segment_diagnostics(panel, signal, side, period_mask),
        "selection_use": False,
    }
