"""Post-freeze diagnostics for replacing zero-state days with frozen signals.

The diagnostics deliberately run *after* a side signal has been frozen.  They
answer whether relabelling a zero day as -1/+1 improves holding-segment
lengths, directional win rates, and the return distribution of each state.  No
field from this module is used to rank candidates or admit a version to the
upload pool.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PHASES = {
    "full": (pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")),
    "development": (pd.Timestamp("2018-01-01"), pd.Timestamp("2022-12-31")),
    "validation": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
    "test": (pd.Timestamp("2025-01-01"), pd.Timestamp("2100-01-01")),
}


def _period_mask(panel: pd.DataFrame, period: str) -> np.ndarray:
    dates = pd.to_datetime(panel["formation_date"], errors="raise")
    start, end = PHASES[period]
    return dates.ge(start).to_numpy() & dates.le(end).to_numpy()


def _segment_frame(state: np.ndarray, dates: pd.Series) -> pd.DataFrame:
    values = np.asarray(state, dtype=int)
    if len(values) == 0:
        return pd.DataFrame(columns=["state", "start", "end", "length", "start_pos", "end_pos"])
    starts = np.r_[True, values[1:] != values[:-1]]
    starts_idx = np.flatnonzero(starts)
    ends_idx = np.r_[starts_idx[1:] - 1, len(values) - 1]
    return pd.DataFrame(
        {
            "state": values[starts_idx],
            "start": pd.to_datetime(dates.iloc[starts_idx]).to_numpy(),
            "end": pd.to_datetime(dates.iloc[ends_idx]).to_numpy(),
            "length": ends_idx - starts_idx + 1,
            "start_pos": starts_idx,
            "end_pos": ends_idx,
        }
    )


def _selected_segments(state: np.ndarray, dates: pd.Series, period_mask: np.ndarray) -> pd.DataFrame:
    """Return complete state segments that intersect a diagnostic period."""
    segments = _segment_frame(state, dates)
    if len(segments) == 0:
        return segments
    # A phase boundary must not turn the visible tail of a long holding into
    # a false one-day segment.  Keep the complete segment when it intersects
    # the requested period.
    period_dates = pd.to_datetime(dates).to_numpy()
    in_period = np.zeros(len(segments), dtype=bool)
    for pos, row in segments.iterrows():
        in_period[pos] = bool((period_mask & (period_dates >= row["start"]) & (period_dates <= row["end"])).any())
    return segments.loc[in_period].reset_index(drop=True)


def _segment_stats(state: np.ndarray, dates: pd.Series, period_mask: np.ndarray) -> dict[str, Any]:
    # Segment metrics use complete segments whose formation dates intersect the
    # requested period.  This avoids counting only the visible tail of a
    # segment as a one-day position at a phase boundary.
    segments = _selected_segments(state, dates, period_mask)
    if len(segments) == 0:
        return {str(s): {"segment_count": 0} for s in (-1, 0, 1)}
    output: dict[str, Any] = {}
    for value in (-1, 0, 1):
        lengths = pd.to_numeric(segments.loc[segments["state"] == value, "length"], errors="coerce").dropna().to_numpy(dtype=float)
        output[str(value)] = {
            "segment_count": int(len(lengths)),
            "mean_length": float(np.mean(lengths)) if len(lengths) else np.nan,
            "median_length": float(np.median(lengths)) if len(lengths) else np.nan,
            "p25_length": float(np.quantile(lengths, 0.25)) if len(lengths) else np.nan,
            "p75_length": float(np.quantile(lengths, 0.75)) if len(lengths) else np.nan,
            "p95_length": float(np.quantile(lengths, 0.95)) if len(lengths) else np.nan,
            "one_day_share": float(np.mean(lengths == 1)) if len(lengths) else np.nan,
            "max_length": int(np.max(lengths)) if len(lengths) else None,
        }
    return output


def _segment_return_stats(panel: pd.DataFrame, state: np.ndarray, dates: pd.Series, period_mask: np.ndarray) -> dict[str, Any]:
    """Return complete-holding, rather than per-day, directional outcomes.

    A segment is a win when its compounded O2O_H1 return in the segment's
    direction is positive.  This is diagnostic only; no segment statistic is
    used in candidate selection or pool admission.
    """
    segments = _selected_segments(state, dates, period_mask)
    returns = pd.to_numeric(panel["o2o_h1"], errors="coerce").astype(float).to_numpy()
    output: dict[str, Any] = {}
    for value in (-1, 0, 1):
        segment_returns: list[float] = []
        for _, row in segments.loc[segments["state"] == value].iterrows():
            raw = returns[int(row["start_pos"]): int(row["end_pos"]) + 1]
            raw = raw[np.isfinite(raw)]
            if len(raw) == 0:
                continue
            directional = raw * float(value) if value != 0 else raw
            segment_returns.append(float(np.prod(1.0 + directional) - 1.0))
        values = np.asarray(segment_returns, dtype=float)
        output[str(value)] = {
            "n": int(len(values)),
            "mean_segment_directional_return": float(np.mean(values)) if len(values) else np.nan,
            "median_segment_directional_return": float(np.median(values)) if len(values) else np.nan,
            "p05_segment_directional_return": float(np.quantile(values, 0.05)) if len(values) else np.nan,
            "p95_segment_directional_return": float(np.quantile(values, 0.95)) if len(values) else np.nan,
            "segment_win_rate_directional": float(np.mean(values > 0.0)) if len(values) else np.nan,
        }
    return output


def _return_stats(panel: pd.DataFrame, state: np.ndarray, period_mask: np.ndarray) -> dict[str, Any]:
    returns = pd.to_numeric(panel["o2o_h1"], errors="coerce").astype(float).to_numpy()
    output: dict[str, Any] = {}
    for value in (-1, 0, 1):
        mask = period_mask & (state == value) & np.isfinite(returns)
        raw = returns[mask]
        directional = raw * float(value) if value != 0 else raw
        output[str(value)] = {
            "n": int(len(raw)),
            "mean_return": float(np.mean(raw)) if len(raw) else np.nan,
            "mean_directional_return": float(np.mean(directional)) if len(raw) else np.nan,
            "median_return": float(np.median(raw)) if len(raw) else np.nan,
            "p05_return": float(np.quantile(raw, 0.05)) if len(raw) else np.nan,
            "p25_return": float(np.quantile(raw, 0.25)) if len(raw) else np.nan,
            "p75_return": float(np.quantile(raw, 0.75)) if len(raw) else np.nan,
            "p95_return": float(np.quantile(raw, 0.95)) if len(raw) else np.nan,
            "win_rate_directional": float(np.mean(directional > 0.0)) if len(raw) else np.nan,
        }
    return output


def _state_counts(state: np.ndarray, period_mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(state, dtype=int)[period_mask]
    total = len(values)
    return {
        str(value): {"days": int(np.sum(values == value)), "share": float(np.mean(values == value)) if total else np.nan}
        for value in (-1, 0, 1)
    }


def _holding_path_period_stats(
    panel: pd.DataFrame,
    holding_signal: np.ndarray,
    side: int,
    dates: pd.Series,
    period_mask: np.ndarray,
) -> dict[str, Any]:
    """Measure the actual frozen holding path, separate from state carry.

    ``persistent_relabel`` intentionally carries a signal through the rest of
    a base-zero run to audit a possible three-state rewrite.  That is not the
    same object as the actual trade holding path.  This helper keeps only the
    contiguous ``holding_signal`` days, so a holding-segment win rate cannot be
    diluted by zero days that were relabelled for the state audit.
    """

    holding = np.asarray(holding_signal, dtype=bool)
    if len(holding) != len(panel):
        raise ValueError("holding path must align with the research panel")
    path_state = np.where(holding, int(side), 0).astype(int)
    segment_stats = _segment_stats(path_state, dates, period_mask).get(str(side), {})
    segment_return_stats = _segment_return_stats(panel, path_state, dates, period_mask).get(str(side), {})
    return_stats = _return_stats(panel, path_state, period_mask).get(str(side), {})
    holding_days = int(np.logical_and(holding, period_mask).sum())
    phase_days = int(period_mask.sum())
    return {
        "side": int(side),
        "holding_days": holding_days,
        "phase_days": phase_days,
        "holding_coverage": float(holding_days / phase_days) if phase_days else np.nan,
        "segment_stats": segment_stats,
        "segment_return_stats": segment_return_stats,
        "return_stats": return_stats,
    }


def _persistent_adjusted_state(base: np.ndarray, down_signal: np.ndarray, up_signal: np.ndarray) -> np.ndarray:
    output = np.asarray(base, dtype=int).copy()
    position = 0
    for i, value in enumerate(output):
        value = int(value)
        if value != 0:
            position = 0
            continue
        down = bool(down_signal[i])
        up = bool(up_signal[i])
        if down and up:
            # A same-day conflict is not allowed to manufacture a direction.
            position = 0
            output[i] = 0
        elif down:
            position = -1
            output[i] = -1
        elif up:
            position = 1
            output[i] = 1
        elif position:
            output[i] = position
    return output


def _persistent_adjusted_state_conflict_continue(
    base: np.ndarray,
    down_signal: np.ndarray,
    up_signal: np.ndarray,
) -> np.ndarray:
    """Causally continue the prior direction through a same-day conflict.

    This is an audit-only alternative to the conservative conflict policy in
    :func:`_persistent_adjusted_state`.  A conflict is assigned to the
    already active direction when one exists; a first conflict with no active
    direction remains neutral.  The decision uses only the current signals
    and the preceding state, never the state or return after the conflict.
    """
    output = np.asarray(base, dtype=int).copy()
    position = 0
    for i, value in enumerate(output):
        value = int(value)
        if value != 0:
            position = 0
            continue
        down = bool(down_signal[i])
        up = bool(up_signal[i])
        if down and up:
            if position:
                output[i] = position
            else:
                output[i] = 0
            continue
        if down:
            position = -1
            output[i] = -1
        elif up:
            position = 1
            output[i] = 1
        elif position:
            output[i] = position
    return output


def _event_adjusted_state(base: np.ndarray, down_signal: np.ndarray, up_signal: np.ndarray) -> np.ndarray:
    output = np.asarray(base, dtype=int).copy()
    conflict = np.asarray(down_signal, dtype=bool) & np.asarray(up_signal, dtype=bool)
    output[np.asarray(down_signal, dtype=bool) & ~conflict & (output == 0)] = -1
    output[np.asarray(up_signal, dtype=bool) & ~conflict & (output == 0)] = 1
    return output


def _same_direction_gap_bridge(state: np.ndarray, max_gap: int, blocked: np.ndarray | None = None) -> np.ndarray:
    """Retrospectively bridge short neutral gaps between equal directions.

    This is deliberately diagnostic-only: it uses the state after the gap to
    decide whether the gap should be filled, so it is not causal and must not
    be used to create a live signal. Conflicts or blocked days prevent a bridge.
    """
    output = np.asarray(state, dtype=int).copy()
    blocked_values = np.asarray(blocked, dtype=bool) if blocked is not None else np.zeros(len(output), dtype=bool)
    if len(blocked_values) != len(output):
        raise ValueError("gap bridge block mask must align with state")
    if max_gap <= 0:
        return output
    i = 0
    while i < len(output):
        if output[i] != 0:
            i += 1
            continue
        j = i
        while j < len(output) and output[j] == 0:
            j += 1
        if (
            i > 0
            and j < len(output)
            and j - i <= max_gap
            and output[i - 1] in (-1, 1)
            and output[i - 1] == output[j]
            and not blocked_values[i:j].any()
        ):
            output[i:j] = output[i - 1]
        i = j
    return output


def build_state_adjustment_diagnostics(
    panel: pd.DataFrame,
    down_signal: np.ndarray,
    up_signal: np.ndarray,
    *,
    down_holding: np.ndarray | None = None,
    up_holding: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return event-only state adjustment and separate holding diagnostics.

    ``down_signal`` and ``up_signal`` are the frozen *entry* signals.  They
    are the only signals allowed to alter the formal three-state series: one
    zero day becomes -1/+1 on the signal day, and later zero days remain zero.
    ``down_holding``/``up_holding`` are optional actual holding paths and are
    used only for the independent holding-segment audit.  Keeping these two
    inputs separate prevents a persistent holding path from silently being
    interpreted as a persistent state relabel.
    """

    base = pd.to_numeric(panel["state"], errors="raise").astype(int).to_numpy()
    down = np.asarray(down_signal, dtype=bool)
    up = np.asarray(up_signal, dtype=bool)
    if len(base) != len(down) or len(base) != len(up):
        raise ValueError("state adjustment signals must align with the research panel")
    down_hold = down if down_holding is None else np.asarray(down_holding, dtype=bool)
    up_hold = up if up_holding is None else np.asarray(up_holding, dtype=bool)
    if len(base) != len(down_hold) or len(base) != len(up_hold):
        raise ValueError("holding paths must align with the research panel")
    dates = pd.Series(pd.to_datetime(panel["formation_date"], errors="raise")).reset_index(drop=True)
    base = base.copy()
    event = _event_adjusted_state(base, down, up)
    persistent = _persistent_adjusted_state(base, down, up)
    persistent_conflict_continue = _persistent_adjusted_state_conflict_continue(base, down, up)
    conflict = down & up
    persistent_gap_bridge_1 = _same_direction_gap_bridge(persistent, 1, conflict)
    persistent_gap_bridge_2 = _same_direction_gap_bridge(persistent, 2, conflict)
    signal_rows = {
        "down_signal_days": int(down.sum()),
        "up_signal_days": int(up.sum()),
        "conflict_days": int(conflict.sum()),
        "down_signal_outside_zero_days": int((down & (base != 0)).sum()),
        "up_signal_outside_zero_days": int((up & (base != 0)).sum()),
    }
    result: dict[str, Any] = {
        "selection_use": False,
        "formal_state_adjustment_mode": "event_relabel_entry_day_only",
        "holding_path_audit_mode": "holding_signal_separate_from_formal_state_relabel",
        "definition": {
            "event_relabel": "only the frozen signal day in state 0 is relabelled to -1/+1",
            "persistent_relabel": "audit-only counterfactual: a frozen entry signal carries through the rest of its base-zero run; never used for the formal state output",
            "persistent_relabel_conflict_continue": "causal audit-only alternative: when both signals fire in state 0, continue the already active direction; a first conflict with no active direction stays 0",
            "directional_return": "state * O2O_H1 for -1/+1; raw O2O_H1 for state 0",
            "one_day_holding_flag": "segment length == 1; diagnostic only",
            "segment_win": "a complete holding segment is a win when compounded directional O2O_H1 return is positive",
            "gap_bridge": "retrospective diagnostic only: fill at most 1 or 2 neutral days when both neighboring states have the same direction; conflicts are never bridged",
        },
        "signal_rows": signal_rows,
        "holding_rows": {
            "down_holding_days": int(down_hold.sum()),
            "up_holding_days": int(up_hold.sum()),
            "down_holding_outside_zero_days": int((down_hold & (base != 0)).sum()),
            "up_holding_outside_zero_days": int((up_hold & (base != 0)).sum()),
        },
        "holding_paths": {"down": {}, "up": {}},
        "periods": {},
    }
    for period in PHASES:
        mask = _period_mask(panel, period)
        result["holding_paths"]["down"][period] = _holding_path_period_stats(panel, down_hold, -1, dates, mask)
        result["holding_paths"]["up"][period] = _holding_path_period_stats(panel, up_hold, 1, dates, mask)
        base_segments = _segment_stats(base, dates, mask)
        event_segments = _segment_stats(event, dates, mask)
        persistent_segments = _segment_stats(persistent, dates, mask)
        conflict_continue_segments = _segment_stats(persistent_conflict_continue, dates, mask)
        bridge_1_segments = _segment_stats(persistent_gap_bridge_1, dates, mask)
        bridge_2_segments = _segment_stats(persistent_gap_bridge_2, dates, mask)
        base_segment_returns = _segment_return_stats(panel, base, dates, mask)
        event_segment_returns = _segment_return_stats(panel, event, dates, mask)
        persistent_segment_returns = _segment_return_stats(panel, persistent, dates, mask)
        conflict_continue_segment_returns = _segment_return_stats(panel, persistent_conflict_continue, dates, mask)
        bridge_1_segment_returns = _segment_return_stats(panel, persistent_gap_bridge_1, dates, mask)
        bridge_2_segment_returns = _segment_return_stats(panel, persistent_gap_bridge_2, dates, mask)
        base_returns = _return_stats(panel, base, mask)
        event_returns = _return_stats(panel, event, mask)
        persistent_returns = _return_stats(panel, persistent, mask)
        conflict_continue_returns = _return_stats(panel, persistent_conflict_continue, mask)
        bridge_1_returns = _return_stats(panel, persistent_gap_bridge_1, mask)
        bridge_2_returns = _return_stats(panel, persistent_gap_bridge_2, mask)
        result["periods"][period] = {
            "base": {
                "state_counts": _state_counts(base, mask),
                "segment_stats": base_segments,
                "segment_return_stats": base_segment_returns,
                "return_stats": base_returns,
            },
            "event_relabel": {
                "state_counts": _state_counts(event, mask),
                "segment_stats": event_segments,
                "segment_return_stats": event_segment_returns,
                "return_stats": event_returns,
            },
            "persistent_relabel": {
                "state_counts": _state_counts(persistent, mask),
                "segment_stats": persistent_segments,
                "segment_return_stats": persistent_segment_returns,
                "return_stats": persistent_returns,
            },
            "persistent_relabel_conflict_continue": {
                "state_counts": _state_counts(persistent_conflict_continue, mask),
                "segment_stats": conflict_continue_segments,
                "segment_return_stats": conflict_continue_segment_returns,
                "return_stats": conflict_continue_returns,
            },
            "persistent_relabel_gap_bridge_1": {
                "state_counts": _state_counts(persistent_gap_bridge_1, mask),
                "segment_stats": bridge_1_segments,
                "segment_return_stats": bridge_1_segment_returns,
                "return_stats": bridge_1_returns,
            },
            "persistent_relabel_gap_bridge_2": {
                "state_counts": _state_counts(persistent_gap_bridge_2, mask),
                "segment_stats": bridge_2_segments,
                "segment_return_stats": bridge_2_segment_returns,
                "return_stats": bridge_2_returns,
            },
            "improvement": {
                "event_vs_base": _improvement(base_segments, event_segments, base_segment_returns, event_segment_returns, base_returns, event_returns),
                "persistent_vs_base": _improvement(base_segments, persistent_segments, base_segment_returns, persistent_segment_returns, base_returns, persistent_returns),
                "conflict_continue_vs_persistent": _improvement(
                    persistent_segments,
                    conflict_continue_segments,
                    persistent_segment_returns,
                    conflict_continue_segment_returns,
                    persistent_returns,
                    conflict_continue_returns,
                ),
                "gap_bridge_1_vs_persistent": _improvement(persistent_segments, bridge_1_segments, persistent_segment_returns, bridge_1_segment_returns, persistent_returns, bridge_1_returns),
                "gap_bridge_2_vs_persistent": _improvement(persistent_segments, bridge_2_segments, persistent_segment_returns, bridge_2_segment_returns, persistent_returns, bridge_2_returns),
            },
        }
    full = result["periods"]["full"]
    event_improvement = full["improvement"]["event_vs_base"]
    event_segments = full["event_relabel"]["segment_stats"]
    holding_down = result["holding_paths"]["down"]["full"]
    holding_up = result["holding_paths"]["up"]["full"]
    persistent_improvement = full["improvement"]["persistent_vs_base"]
    persistent_segments = full["persistent_relabel"]["segment_stats"]
    directional_states = ("-1", "1")
    zero_length_reduced = (
        float(full["event_relabel"]["segment_stats"]["0"].get("mean_length", np.nan))
        < float(full["base"]["segment_stats"]["0"].get("mean_length", np.nan))
    )
    event_directional_day_mean_improved = all(
        float(event_improvement[state].get("mean_directional_return_delta", np.nan)) >= 0.0
        for state in directional_states
    )
    event_directional_segment_win_improved = all(
        float(event_improvement[state].get("segment_win_rate_directional_delta", np.nan)) >= 0.0
        for state in directional_states
    )
    holding_one_day_values = [
        float(holding_down.get("segment_stats", {}).get("one_day_share", np.nan)),
        float(holding_up.get("segment_stats", {}).get("one_day_share", np.nan)),
    ]
    holding_one_day_share_ok = bool(
        len(holding_one_day_values) == 2
        and all(np.isfinite(value) and value <= 0.05 for value in holding_one_day_values)
    )
    holding_segment_win_values = {
        "down": holding_down.get("segment_return_stats", {}).get("segment_win_rate_directional"),
        "up": holding_up.get("segment_return_stats", {}).get("segment_win_rate_directional"),
    }
    holding_segment_win_ok = bool(
        all(np.isfinite(float(value)) and float(value) >= 0.50 for value in holding_segment_win_values.values())
    )
    event_directional_one_day_share = {
        state: event_segments[state].get("one_day_share") for state in directional_states
    }
    result["quality_assessment"] = {
        "status": "PASS_EVENT_ZERO_SHORTENING_HOLDING_PATH_QUALITY_RETURN_MIXED"
        if zero_length_reduced and holding_one_day_share_ok and holding_segment_win_ok and not (event_directional_day_mean_improved and event_directional_segment_win_improved)
        else "PASS_ALL_CHECKS"
        if zero_length_reduced and holding_one_day_share_ok and holding_segment_win_ok and event_directional_day_mean_improved and event_directional_segment_win_improved
        else "CHECK_REQUIRED",
        "selection_use": False,
        "formal_state_adjustment_mode": "event_relabel_entry_day_only",
        "zero_mean_segment_length_reduced_by_event": bool(zero_length_reduced),
        "event_directional_one_day_share": event_directional_one_day_share,
        "event_directional_daily_mean_return_uniformly_improved": bool(event_directional_day_mean_improved),
        "event_directional_segment_win_uniformly_improved": bool(event_directional_segment_win_improved),
        "actual_holding_directional_one_day_share_max_le_5pct": bool(holding_one_day_share_ok),
        "actual_holding_directional_segment_win_ge_50pct": bool(holding_segment_win_ok),
        "actual_holding_segment_win_rates": holding_segment_win_values,
        "actual_holding_paths": {
            "down": holding_down,
            "up": holding_up,
        },
        # Backward-compatible audit fields.  These are explicitly prefixed so
        # they cannot be mistaken for the production event-only assessment.
        "persistent_zero_mean_segment_length_reduced_audit": bool(
            float(full["persistent_relabel"]["segment_stats"]["0"].get("mean_length", np.nan))
            < float(full["base"]["segment_stats"]["0"].get("mean_length", np.nan))
        ),
        "persistent_directional_one_day_share_audit": {
            state: persistent_segments[state].get("one_day_share") for state in directional_states
        },
        "persistent_directional_daily_mean_return_uniformly_improved_audit": all(
            float(persistent_improvement[state].get("mean_directional_return_delta", np.nan)) >= 0.0
            for state in directional_states
        ),
        "persistent_directional_segment_win_uniformly_improved_audit": all(
            float(persistent_improvement[state].get("segment_win_rate_directional_delta", np.nan)) >= 0.0
            for state in directional_states
        ),
        "persistent_gap_bridge_1_zero_segment_count": int(full["persistent_relabel_gap_bridge_1"]["segment_stats"]["0"].get("segment_count", 0)),
        "persistent_gap_bridge_2_zero_segment_count": int(full["persistent_relabel_gap_bridge_2"]["segment_stats"]["0"].get("segment_count", 0)),
        "persistent_gap_bridge_1_zero_one_day_share": full["persistent_relabel_gap_bridge_1"]["segment_stats"]["0"].get("one_day_share"),
        "persistent_gap_bridge_2_zero_one_day_share": full["persistent_relabel_gap_bridge_2"]["segment_stats"]["0"].get("one_day_share"),
        "persistent_conflict_continue_zero_segment_count": int(full["persistent_relabel_conflict_continue"]["segment_stats"]["0"].get("segment_count", 0)),
        "persistent_conflict_continue_zero_one_day_share": full["persistent_relabel_conflict_continue"]["segment_stats"]["0"].get("one_day_share"),
        "persistent_conflict_continue_directional_daily_mean_uniformly_improved": all(
            float(full["improvement"]["conflict_continue_vs_persistent"][state].get("mean_directional_return_delta", np.nan)) >= 0.0
            for state in directional_states
        ),
        "persistent_conflict_continue_directional_segment_win_uniformly_improved": all(
            float(full["improvement"]["conflict_continue_vs_persistent"][state].get("segment_win_rate_directional_delta", np.nan)) >= 0.0
            for state in directional_states
        ),
        "interpretation": "diagnostic only; zero-state shortening does not automatically prove directional return improvement",
    }
    return result


def _improvement(
    base_segments: dict[str, Any],
    adjusted_segments: dict[str, Any],
    base_segment_returns: dict[str, Any],
    adjusted_segment_returns: dict[str, Any],
    base_returns: dict[str, Any],
    adjusted_returns: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for value in (-1, 0, 1):
        key = str(value)
        bseg, aseg = base_segments.get(key, {}), adjusted_segments.get(key, {})
        bsegret, asegret = base_segment_returns.get(key, {}), adjusted_segment_returns.get(key, {})
        bret, aret = base_returns.get(key, {}), adjusted_returns.get(key, {})
        output[key] = {
            "mean_segment_length_delta": _delta(aseg.get("mean_length"), bseg.get("mean_length")),
            "one_day_share_delta": _delta(aseg.get("one_day_share"), bseg.get("one_day_share")),
            "mean_directional_return_delta": _delta(aret.get("mean_directional_return"), bret.get("mean_directional_return")),
            "win_rate_directional_delta": _delta(aret.get("win_rate_directional"), bret.get("win_rate_directional")),
            "mean_segment_directional_return_delta": _delta(asegret.get("mean_segment_directional_return"), bsegret.get("mean_segment_directional_return")),
            "segment_win_rate_directional_delta": _delta(asegret.get("segment_win_rate_directional"), bsegret.get("segment_win_rate_directional")),
            "p05_return_delta": _delta(aret.get("p05_return"), bret.get("p05_return")),
            "p95_return_delta": _delta(aret.get("p95_return"), bret.get("p95_return")),
        }
    return output


def _delta(adjusted: Any, base: Any) -> float:
    try:
        a, b = float(adjusted), float(base)
        return a - b if np.isfinite(a) and np.isfinite(b) else np.nan
    except (TypeError, ValueError):
        return np.nan


__all__ = ["build_state_adjustment_diagnostics"]
