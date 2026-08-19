from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import FrozenRecord, source_whitelist
from .features import feature_lineage_table
from .metrics import (
    _joint_score,
    _phase_metrics,
    _signal_hash,
    annual_metrics,
    block_bootstrap_metrics,
    bootstrap_summary,
    error_diagnostics,
    evaluate_frozen_candidate,
    label_thresholds,
    score_bins,
    select_top1,
)
from .pipeline import PreparedResearch, write_json


SCHEMA_BASE_DIMS: dict[str, tuple[int, ...]] = {
    "dynamic_quantile": (8, 8, 4),
    "regime": (8, 8, 4),
    "range_vol": (8, 8, 4),
    "tree": (8, 4, 4, 4),
    "evt": (8, 8, 4),
    "event": (8, 8, 4),
    "online": (8, 8, 4),
    "liquidity": (8, 8, 4),
    "complexity": (8, 8, 4),
    "distribution": (8, 4, 4, 4),
    "bayes": (8, 4, 4, 4),
    "pattern": (8, 4, 4, 4),
    "conformal": (8, 8, 4),
}

COVERAGE_GRIDS: dict[str, tuple[float, ...]] = {
    "dynamic_quantile": (0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.105, 0.115, 0.130, 0.150),
    "regime": (0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.105, 0.115, 0.130, 0.150),
    "range_vol": (0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.105, 0.115, 0.130, 0.150),
    "tree": (0.025, 0.040, 0.055, 0.070, 0.085, 0.100, 0.115, 0.130),
    "evt": (0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.105, 0.115, 0.130, 0.150),
    "event": (0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.105, 0.115, 0.130, 0.150),
    "online": (0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.105, 0.115, 0.130, 0.150),
    "liquidity": (0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.105, 0.115, 0.130, 0.150),
    "complexity": (0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.105, 0.115, 0.130, 0.150),
    "distribution": (0.025, 0.040, 0.055, 0.070, 0.085, 0.100, 0.115, 0.130),
    "bayes": (0.040, 0.075, 0.110, 0.150),
    "pattern": (0.025, 0.040, 0.055, 0.070, 0.085, 0.100, 0.115, 0.130),
    "conformal": (0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.105, 0.115, 0.130, 0.150),
}

LINKS = (
    (0.75, 0.25, 0.00),
    (0.55, 0.30, 0.15),
    (0.40, 0.40, 0.20),
    (0.25, 0.50, 0.25),
)


def load_reserve_metadata(path: str | Path | None = None) -> dict[str, dict[str, str]]:
    here = Path(__file__).resolve()
    packaged = (here.parents[1] / "reserve_metadata.csv").resolve()
    candidate = packaged if path is None else Path(path).expanduser().resolve()
    if candidate != packaged:
        raise ValueError("company package accepts only its bundled reserve registry")
    if not candidate.is_file():
        raise FileNotFoundError("bundled reserve registry not found")
    with candidate.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["reserve_id"]: row for row in rows}


def reserve_ids(metadata: dict[str, dict[str, str]]) -> list[str]:
    ids = sorted(metadata, key=lambda value: int(value[1:]))
    if ids != [f"V{i:02d}" for i in range(51, 91)]:
        raise AssertionError("reserve metadata must contain contiguous V51-V90")
    return ids


def _finite(values: Any, fill: float = 0.5) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)


def _series(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        raise KeyError(f"reserve feature missing: {name}")
    return _finite(frame[name].to_numpy())


def _bounded(values: np.ndarray, span: int = 252) -> np.ndarray:
    series = pd.Series(_finite(values))
    mean = series.ewm(span=span, adjust=False, min_periods=min(30, max(5, span // 4)).__int__()).mean()
    variance = (series - mean).pow(2).ewm(span=span, adjust=False, min_periods=min(30, max(5, span // 4)).__int__()).mean()
    z = (series - mean) / np.sqrt(variance + 1e-8)
    return 1 / (1 + np.exp(-z.clip(-8, 8).to_numpy()))


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return _finite(pd.Series(values).rolling(window, min_periods=min(window, max(5, window // 2))).mean().to_numpy())


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    return _finite(pd.Series(values).rolling(window, min_periods=min(window, max(5, window // 2))).std(ddof=0).to_numpy(), 1e-6)


def _rolling_last_rank(values: np.ndarray, window: int = 252, min_periods: int = 60) -> np.ndarray:
    """Causal empirical rank of the current value inside its trailing window."""
    series = pd.Series(_finite(values))

    def rank_last(window_values: np.ndarray) -> float:
        finite = window_values[np.isfinite(window_values)]
        if not len(finite):
            return 0.5
        target = float(window_values[-1])
        less = float(np.sum(finite < target))
        equal = float(np.sum(finite == target))
        return (less + 0.5 * equal) / len(finite)

    return _finite(series.rolling(window, min_periods=min_periods).apply(rank_last, raw=True).to_numpy())


def _duration_since_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, bool)
    out = np.zeros(len(state), dtype=float)
    duration = 0.0
    previous = False
    for i, current in enumerate(state):
        duration = duration + 1.0 if current and previous else 1.0 if current else 0.0
        out[i] = duration
        previous = bool(current)
    return out


def _ewm(values: np.ndarray, halflife: float) -> np.ndarray:
    return _finite(pd.Series(values).ewm(halflife=halflife, adjust=False, min_periods=max(5, int(2 * halflife))).mean().to_numpy())


def _directional(values: np.ndarray, side: str) -> np.ndarray:
    score = _bounded(values)
    return 1 - score if side == "down" else score


def _risk(values: np.ndarray) -> np.ndarray:
    return _bounded(np.abs(values))


def _lagged(values: np.ndarray, lag: int) -> np.ndarray:
    arr = _finite(values)
    out = np.full(len(arr), np.nan, dtype=float)
    if lag < len(arr):
        out[lag:] = arr[:-lag]
    return out


def _method_primitives(frame: pd.DataFrame, method: str, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ret = _series(frame, "ret_1")
    oo = _series(frame, "oo_ret_1")
    absret = np.abs(ret)
    tr = _series(frame, "true_range_pct")
    vol = _series(frame, "vol_20")
    amount = _series(frame, "amount_z_20")
    close_loc = _series(frame, "close_location")
    signed = _directional(ret, side)
    direction5 = _directional(_series(frame, "ret_5"), side)
    direction20 = _directional(_series(frame, "ret_20"), side)
    risk = _risk(absret + 0.5 * tr + 0.2 * _series(frame, "amihud"))
    scale = _risk(vol + tr)

    def banks(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        result = []
        for values_i in values:
            arr = np.column_stack([_bounded(values_i, span=span) for span in (32, 64, 128, 252, 384, 512, 768, 1024)])
            result.append(arr)
        return tuple(result)  # type: ignore[return-value]

    if method in {"caviar_dynamic_quantile", "care_dynamic_expectile", "gas_score_driven_tail"}:
        q = np.column_stack([_directional(_ewm(oo, h), side) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        shock = np.column_stack([_risk(_ewm(absret, h)) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        asym = np.column_stack([_directional(_ewm(ret * (absret + 1e-6), h), side) for h in (2, 4, 8, 16)])
        return q, shock, asym, np.column_stack([_bounded(q[:, i] * shock[:, i], span=64) for i in range(8)])

    if method == "bayesian_online_changepoint":
        innovations = ret - _rolling_mean(ret, 32)
        z = np.abs(innovations) / (_rolling_std(ret, 32) + 1e-6)
        change = np.column_stack([_bounded(z * h, span=span) for h, span in zip((0.5, 0.75, 1, 1.5, 2, 3, 4, 6), (32, 64, 96, 128, 192, 256, 384, 512))])
        run = np.column_stack([_bounded(_ewm((z < h).astype(float), h * 8), span=128) for h in (1, 1.5, 2, 3, 4, 5, 6, 8)])
        return change, run, np.column_stack([_bounded(z * h, span=64) for h in (0.5, 0.75, 1, 1.5)]), np.column_stack([_bounded(scale * h, span=span) for h, span in zip((0.5, 0.75, 1, 1.5), (64, 128, 256, 512))])

    if method == "har_range_distribution":
        range_base = np.column_stack([
            _series(frame, "parkinson_vol_20"),
            _series(frame, "rs_vol_20"),
            _series(frame, "yz_vol_20"),
            tr,
        ]).mean(axis=1)
        daily = np.column_stack([_rolling_mean(range_base, w) for w in (3, 5, 10, 20, 30, 60, 120, 252)])
        weekly = np.column_stack([_rolling_mean(range_base, w) for w in (5, 10, 20, 40, 60, 120, 180, 252)])
        monthly = np.column_stack([_rolling_mean(range_base, w) for w in (20, 30, 60, 90, 120, 180, 252, 384)])
        return banks([daily.mean(axis=1), weekly.mean(axis=1), monthly.mean(axis=1), signed * range_base])

    if method in {"quantile_regression_forest", "quantile_gradient_boosting", "ngboost_student_t", "bart_rare_event", "gaussian_process_tail_classifier"}:
        state = np.column_stack([signed, direction5, direction20, risk, scale, _bounded(amount), _bounded(close_loc), _bounded(ret * amount)])
        local = np.column_stack([_bounded(_rolling_mean(state[:, i], w), span=252) for i in range(8) for w in (20, 60, 120, 252)])
        local = local.reshape(len(frame), 8, 4).mean(axis=2)
        conditional = np.column_stack([_bounded(state[:, i] - _rolling_mean(state[:, i], w), span=128) for i, w in zip(range(8), (20, 30, 60, 90, 120, 180, 252, 384))])
        uncertainty = np.column_stack([_risk(_rolling_std(ret, w) + _rolling_std(absret, w)) for w in (10, 20, 30, 60, 90, 120, 180, 252)])
        return state, local, conditional, uncertainty

    if method == "filtered_evt_pot":
        filtered = absret / (_ewm(absret, 32) + 1e-6)
        exceed = np.maximum(filtered[:, None] - np.array([1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0])[None, :], 0)
        return np.column_stack([_bounded(exceed[:, i]) for i in range(8)]), np.column_stack([_bounded(_ewm(exceed[:, i], h)) for i, h in enumerate((2, 4, 8, 16, 32, 64, 128, 256))]), np.column_stack([_directional(ret * (exceed[:, i] + 1e-6), side) for i in range(4)]), np.column_stack([_bounded(filtered * h, span=span) for h, span in zip((0.5, 0.75, 1, 1.5), (32, 64, 128, 256))])

    if method == "discrete_hawkes_tail_intensity":
        event = (np.abs(ret) > pd.Series(absret).shift(1).rolling(252, min_periods=60).quantile(0.90).to_numpy()).astype(float)
        intensity = np.column_stack([_bounded(_ewm(event, h)) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        signed_event = np.column_stack([_bounded(_ewm(((-ret if side == "down" else ret) * event), h)) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        return intensity, signed_event, np.column_stack([_bounded(_ewm(event * absret, h)) for h in (2, 4, 8, 16)]), np.column_stack([_bounded(scale * h, span=span) for h, span in zip((0.5, 0.75, 1, 1.5), (64, 128, 256, 512))])

    if method in {"dynamic_model_averaging", "hedge_expert_weights"}:
        experts = np.column_stack([signed, direction5, direction20, _bounded(close_loc), _bounded(amount), risk, scale, _bounded(ret * amount)])
        losses = np.abs(experts - _directional(ret, side)[:, None])
        weights = []
        for h in (2, 4, 8, 16, 32, 64, 128, 256):
            penalty = _ewm(losses.mean(axis=1), h)
            weights.append(np.exp(-penalty))
        expert_mix = np.column_stack([_bounded(experts @ np.array([0.05, 0.08, 0.12, 0.15, 0.18, 0.17, 0.15, 0.10])) for _ in range(8)])
        return experts, np.column_stack(weights), expert_mix, np.column_stack([_bounded(losses.mean(axis=1), span=span) for span in (32, 64, 128, 256)])

    if method in {"corwin_schultz_spread_state", "roll_spread_state", "abdi_ranaldo_spread_state"}:
        high = _series(frame, "high")
        low = _series(frame, "low")
        close = _series(frame, "close")
        if method == "roll_spread_state":
            spread = -pd.Series(ret).rolling(30, min_periods=10).cov(pd.Series(ret).shift(1)).to_numpy()
        elif method == "abdi_ranaldo_spread_state":
            spread = (high - low) / np.maximum(close, 1e-6) * (1 - np.clip(close_loc, 0, 1))
        else:
            beta = np.log(np.maximum(high, 1e-6) / np.maximum(low, 1e-6)) ** 2
            spread = np.sqrt(np.maximum(beta, 0)) * (1 + np.abs(close_loc - 0.5))
        spread = _finite(spread)
        return np.column_stack([_bounded(_rolling_mean(spread, w)) for w in (2, 5, 10, 20, 30, 60, 120, 252)]), np.column_stack([_bounded(_ewm(spread, h)) for h in (2, 4, 8, 16, 32, 64, 128, 256)]), np.column_stack([_bounded(signed * spread * h, span=span) for h, span in zip((0.5, 0.75, 1, 1.5), (32, 64, 128, 256))]), np.column_stack([_bounded(spread * h, span=span) for h, span in zip((0.5, 0.75, 1, 1.5), (64, 128, 256, 512))])

    if method == "rolling_permutation_entropy":
        signs = np.sign(oo)
        transitions = np.abs(np.diff(signs, prepend=signs[0]))
        entropy = np.column_stack([_bounded(_rolling_mean(transitions, w)) for w in (5, 10, 20, 30, 60, 90, 120, 252)])
        return entropy, np.column_stack([_bounded(_rolling_std(oo, w)) for w in (5, 10, 20, 30, 60, 90, 120, 252)]), np.column_stack([_bounded(signed * entropy[:, i], span=128) for i in range(4)]), np.column_stack([_bounded(1 - entropy[:, i], span=span) for i, span in enumerate((32, 64, 128, 256))])

    if method == "filtered_historical_simulation":
        # Filtered historical simulation: standardise realised O2O shocks by a
        # causal scale, then use the trailing empirical tail rank.  This is
        # deliberately different from conformal calibration below.
        halflives = (2, 4, 8, 16, 32, 64, 128, 256)
        scales = np.column_stack([_ewm(np.abs(oo), h) for h in halflives])
        standardized = np.column_stack([oo / (scales[:, i] + 1e-6) for i in range(8)])
        empirical_tail = np.column_stack([_rolling_last_rank(standardized[:, i]) for i in range(8)])
        tail_direction = 1.0 - empirical_tail if side == "down" else empirical_tail
        resampled_scale = np.column_stack([
            _bounded(_ewm(np.abs(standardized[:, i]), h))
            for i, h in enumerate(halflives)
        ])
        directional_residual = np.column_stack([
            _directional(standardized[:, i] * (1.0 + 0.15 * i), side)
            for i in range(4)
        ])
        uncertainty = np.column_stack([
            _bounded(_rolling_std(standardized[:, i], window))
            for i, window in enumerate((20, 30, 60, 90))
        ])
        return tail_direction, resampled_scale, directional_residual, uncertainty

    if method == "adaptive_conformal_tail":
        # Adaptive conformal tail: update a past-only nonconformity cutoff from
        # recent exceedance error, instead of using a fixed empirical rank.
        halflives = (2, 4, 8, 16, 32, 64, 128, 256)
        scales = np.column_stack([_ewm(absret, h) for h in halflives])
        nonconformity = np.column_stack([absret / (scales[:, i] + 1e-6) for i in range(8)])
        cutoff = np.column_stack([
            pd.Series(nonconformity[:, i]).rolling(252, min_periods=60).quantile(0.90).to_numpy()
            for i in range(8)
        ])
        exceedance = (nonconformity > cutoff).astype(float)
        adaptation = np.column_stack([
            _ewm(exceedance[:, i] - 0.10, span)
            for i, span in enumerate((16, 24, 32, 48, 64, 96, 128, 192))
        ])
        adaptive_excess = np.column_stack([
            _bounded((nonconformity[:, i] - cutoff[:, i] * (1.0 + adaptation[:, i])) / (cutoff[:, i] + 1e-6))
            for i in range(8)
        ])
        directional_excess = np.column_stack([
            _directional(ret / (scales[:, i] + 1e-6), side)
            for i in range(4)
        ])
        calibration_uncertainty = np.column_stack([
            _bounded(_rolling_std(exceedance[:, i], window))
            for i, window in enumerate((20, 30, 60, 90))
        ])
        return adaptive_excess, np.column_stack([_bounded(_ewm(exceedance[:, i], h)) for i, h in enumerate(halflives)]), directional_excess, calibration_uncertainty

    if method == "time_weighted_cqr":
        # Time-weighted conformalised quantile-regression proxy: a causal
        # exponentially weighted centre and scale define a side-specific
        # interval breach, with recent residuals receiving larger weight.
        halflives = (2, 4, 8, 16, 32, 64, 128, 256)
        centre = np.column_stack([_ewm(oo, h) for h in halflives])
        residual = np.column_stack([np.abs(oo - centre[:, i]) for i in range(8)])
        weighted_scale = np.column_stack([_ewm(residual[:, i], h) for i, h in enumerate(halflives)])
        interval_width = np.column_stack([
            _bounded(weighted_scale[:, i] * (0.75 + 0.25 * i))
            for i in range(8)
        ])
        directional_breach = np.column_stack([
            _directional((oo - centre[:, i]) / (weighted_scale[:, i] + 1e-6), side)
            for i in range(4)
        ])
        interval_uncertainty = np.column_stack([
            _bounded(_rolling_std(residual[:, i], window))
            for i, window in enumerate((20, 30, 60, 90))
        ])
        return interval_width, np.column_stack([_bounded(_ewm(residual[:, i], h)) for i, h in enumerate(halflives)]), directional_breach, interval_uncertainty

    if method == "conformal_test_martingale":
        # Online conformal test martingale: rolling p-values are transformed
        # by a betting function and accumulated causally as log evidence.
        halflives = (2, 4, 8, 16, 32, 64, 128, 256)
        scales = np.column_stack([_ewm(absret, h) for h in halflives])
        nonconformity = np.column_stack([absret / (scales[:, i] + 1e-6) for i in range(8)])
        pvalues = np.column_stack([1.0 - _rolling_last_rank(nonconformity[:, i]) for i in range(8)])
        betting_log = np.column_stack([
            _ewm(np.log(np.clip(1.0 + 1.5 * (pvalues[:, i] - 0.5), 1e-4, 4.0)), h)
            for i, h in enumerate(halflives)
        ])
        directional_pvalues = np.column_stack([
            _directional(ret / (scales[:, i] + 1e-6), side)
            for i in range(4)
        ])
        martingale_instability = np.column_stack([
            _bounded(_rolling_std(betting_log[:, i], window))
            for i, window in enumerate((20, 30, 60, 90))
        ])
        return np.column_stack([_bounded(pvalues[:, i]) for i in range(8)]), np.column_stack([_bounded(betting_log[:, i]) for i in range(8)]), directional_pvalues, martingale_instability

    if method == "filtered_markov_switching":
        vol_state = _bounded(vol)
        trend_state = _bounded(_series(frame, "ret_20"))
        transition = np.abs(np.diff(vol_state, prepend=vol_state[0])) + np.abs(np.diff(trend_state, prepend=trend_state[0]))
        return np.column_stack([_bounded(_ewm(transition, h)) for h in (2, 4, 8, 16, 32, 64, 128, 256)]), np.column_stack([_bounded(_ewm(vol_state, h)) for h in (2, 4, 8, 16, 32, 64, 128, 256)]), np.column_stack([_bounded(_ewm(trend_state, h)) for h in (2, 4, 8, 16)]), np.column_stack([_bounded(np.abs(np.diff(vol_state, prepend=vol_state[0])) * h, span=span) for h, span in zip((0.5, 0.75, 1, 1.5), (32, 64, 128, 256))])

    if method == "hidden_semi_markov_duration":
        # Semi-Markov duration: state occupancy and duration hazard are
        # explicit, rather than reusing the Markov filtered transition bank.
        vol_state = _bounded(vol)
        trend_state = _bounded(_series(frame, "ret_20"))
        target_state = (vol_state >= 0.65) | (
            trend_state <= 0.35 if side == "down" else trend_state >= 0.65
        )
        duration = _duration_since_state(target_state)
        hazard = np.column_stack([
            _bounded(1.0 / (duration + h), span=128)
            for h in (0.5, 1, 2, 4, 8, 16, 32, 64)
        ])
        persistence = np.column_stack([
            _bounded(_ewm(target_state.astype(float), h))
            for h in (2, 4, 8, 16, 32, 64, 128, 256)
        ])
        directional_state = np.column_stack([
            _directional((trend_state - 0.5) * (1.0 + 0.25 * i), side)
            for i in range(4)
        ])
        transition_hazard = np.column_stack([
            _bounded(np.abs(np.diff(target_state.astype(float), prepend=target_state[0])) * h, span=span)
            for h, span in zip((0.5, 0.75, 1, 1.5), (32, 64, 128, 256))
        ])
        return hazard, persistence, directional_state, transition_hazard

    if method == "acd_extreme_duration":
        cut = pd.Series(absret).shift(1).rolling(252, min_periods=60).quantile(0.90).to_numpy()
        event = absret > _finite(cut, np.nanmedian(absret))
        duration = np.zeros(len(frame), dtype=float)
        current = 0
        for i, flag in enumerate(event):
            current = 0 if flag else current + 1
            duration[i] = current
        return np.column_stack([_bounded(1 / (duration + h), span=128) for h in (0.5, 1, 2, 4, 8, 16, 32, 64)]), np.column_stack([_bounded(_ewm(event.astype(float), h)) for h in (2, 4, 8, 16, 32, 64, 128, 256)]), np.column_stack([_directional(ret * event, side) for _ in range(4)]), np.column_stack([_bounded(scale * h, span=span) for h, span in zip((0.5, 0.75, 1, 1.5), (32, 64, 128, 256))])

    if method in {"rough_fractional_range_kernel", "figarch_tail_scale", "multiplicative_error_range", "range_measurement_garch", "component_garch_tail"}:
        range_value = (tr + _series(frame, "parkinson_vol_20") + _series(frame, "rs_vol_20")) / 3
        if method == "rough_fractional_range_kernel":
            kernels = np.column_stack([_bounded(_ewm(range_value, h)) for h in (1.5, 2, 3, 5, 8, 13, 21, 34)])
        elif method == "figarch_tail_scale":
            kernels = np.column_stack([_bounded(_rolling_mean(range_value, w) ** (0.3 + 0.08 * i)) for i, w in enumerate((5, 10, 20, 30, 60, 90, 120, 252))])
        elif method == "multiplicative_error_range":
            kernels = np.column_stack([_bounded(range_value / (_ewm(range_value, h) + 1e-6)) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        elif method == "range_measurement_garch":
            kernels = np.column_stack([_bounded(_ewm(absret, h) * (1 + _bounded(range_value))) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        else:
            fast = np.column_stack([_ewm(range_value, h) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
            slow = np.column_stack([_ewm(range_value, h * 8) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
            kernels = np.column_stack([_bounded(fast[:, i] / (slow[:, i] + 1e-6)) for i in range(8)])
        return kernels, np.column_stack([_bounded(_ewm(kernels[:, i], h)) for i, h in enumerate((2, 4, 8, 16, 32, 64, 128, 256))]), np.column_stack([_directional(ret * kernels[:, i], side) for i in range(4)]), np.column_stack([_bounded(kernels[:, i] * h, span=span) for i, (h, span) in enumerate(zip((0.5, 0.75, 1, 1.5), (32, 64, 128, 256)))])

    if method in {"variance_ratio", "rolling_variance_ratio"}:
        vr = []
        for w in (5, 10, 20, 30, 60, 90, 120, 252):
            multi = pd.Series(oo).rolling(w, min_periods=max(5, w // 2)).var(ddof=0).to_numpy()
            one = pd.Series(oo).rolling(max(3, w // 5), min_periods=3).var(ddof=0).to_numpy()
            vr.append(_bounded(multi / (5 * one + 1e-8)))
        vr = np.column_stack(vr)
        return vr, np.column_stack([_bounded(_rolling_mean(vr[:, i], w)) for i, w in enumerate((5, 10, 20, 30, 60, 90, 120, 252))]), np.column_stack([_bounded(signed * vr[:, i], span=128) for i in range(4)]), np.column_stack([_bounded(np.abs(vr[:, i] - 0.5), span=span) for i, span in enumerate((32, 64, 128, 256))])

    if method in {"truncated_path_signature", "recurrence_quantification", "causal_matrix_profile_discord", "dev_only_shapelets", "deterministic_minirocket_tail", "wavelet_scattering_tail", "causal_ssa_residual"}:
        path = np.column_stack([oo, tr, _bounded(amount), close_loc])
        if method == "truncated_path_signature":
            interactions = path[:, 0] * path[:, 1]
            base = np.column_stack([_bounded(_rolling_mean(interactions, w)) for w in (3, 5, 10, 20, 30, 60, 120, 252)])
        elif method == "recurrence_quantification":
            lagged = np.column_stack([_lagged(path[:, j], 5) for j in range(path.shape[1])])
            distance = np.linalg.norm(path - lagged, axis=1)
            base = np.column_stack([_bounded(_rolling_mean(distance, w)) for w in (5, 10, 20, 30, 60, 90, 120, 252)])
        elif method == "causal_matrix_profile_discord":
            distances = []
            for lag in (3, 5, 10, 20, 30, 60, 90, 120):
                lagged = np.column_stack([_lagged(path[:, j], lag) for j in range(path.shape[1])])
                distances.append(_bounded(np.linalg.norm(path - lagged, axis=1)))
            base = np.column_stack(distances)
        elif method == "dev_only_shapelets":
            base = np.column_stack([_bounded(np.abs(oo - _rolling_mean(oo, w))) for w in (3, 5, 10, 20, 30, 60, 90, 120)])
        elif method == "deterministic_minirocket_tail":
            kernels = [np.array([1, -1]), np.array([1, 0, -1]), np.array([1, -2, 1]), np.array([-1, 1, 1, -1])]
            base = np.column_stack([_bounded(pd.Series(oo).rolling(len(kernel), min_periods=2).apply(lambda x: float(np.dot(x, kernel[-len(x):])), raw=True).to_numpy()) for kernel in kernels for _ in (0, 1)][:8])
        elif method == "wavelet_scattering_tail":
            base = np.column_stack([_bounded(np.abs(_ewm(oo, h) - _ewm(oo, h * 2))) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        else:
            base = np.column_stack([_bounded(oo - _rolling_mean(oo, w)) for w in (5, 10, 20, 30, 60, 90, 120, 252)])
        return base, np.column_stack([_bounded(_ewm(base[:, i], h)) for i, h in enumerate((2, 4, 8, 16, 32, 64, 128, 256))]), np.column_stack([_directional(ret * base[:, i], side) for i in range(4)]), np.column_stack([_bounded(base[:, i] * h, span=span) for i, (h, span) in enumerate(zip((0.5, 0.75, 1, 1.5), (32, 64, 128, 256)))])

    if method == "spot_transfer_entropy":
        sign = (oo > 0).astype(float)
        transition = np.abs(sign - _lagged(sign, 1))
        te = np.column_stack([_bounded(_rolling_mean(transition * _bounded(amount), w)) for w in (5, 10, 20, 30, 60, 90, 120, 252)])
        return te, np.column_stack([_bounded(_ewm(te[:, i], h)) for i, h in enumerate((2, 4, 8, 16, 32, 64, 128, 256))]), np.column_stack([_directional(ret * te[:, i], side) for i in range(4)]), np.column_stack([_bounded(1 - te[:, i], span=span) for i, span in enumerate((32, 64, 128, 256))])

    if method == "catch22_tail_classifier":
        ac = []
        for lag in (1, 2, 3, 5, 10, 20, 30, 60):
            ac.append(_bounded(pd.Series(oo).rolling(120, min_periods=30).corr(pd.Series(oo).shift(lag)).to_numpy()))
        ac = np.column_stack(ac)
        return ac, np.column_stack([_bounded(_rolling_std(oo, w)) for w in (5, 10, 20, 30, 60, 90, 120, 252)]), np.column_stack([_bounded(signed * ac[:, i], span=128) for i in range(4)]), np.column_stack([_bounded(_risk(ret) * h, span=span) for h, span in zip((0.5, 0.75, 1, 1.5), (32, 64, 128, 256))])

    raise KeyError(f"reserve method not implemented: {method}")


def candidate_parameter_table(meta: dict[str, str], side: str) -> pd.DataFrame:
    schema = meta["candidate_schema"]
    dims = SCHEMA_BASE_DIMS[schema]
    coverage = COVERAGE_GRIDS[schema]
    rows = []
    base_index = 0
    if len(dims) == 3:
        for a0 in range(dims[0]):
            for a1 in range(dims[1]):
                for a2 in range(dims[2]):
                    base_id = f"base_{base_index + 1:04d}"
                    for cov in coverage:
                        rows.append({
                            "candidate_id": f"{base_id}_cov_{cov:.3f}",
                            "base_candidate_id": base_id,
                            "base_index": base_index,
                            "axis_1": a0,
                            "axis_2": a1,
                            "axis_3": a2,
                            "axis_4": 0,
                            "coverage_config": cov,
                            "side": side,
                            "reserve_id": meta["reserve_id"],
                            "core_logic_name": meta["core_logic_name"],
                            "candidate_schema": schema,
                        })
                    base_index += 1
    else:
        for a0 in range(dims[0]):
            for a1 in range(dims[1]):
                for a2 in range(dims[2]):
                    for a3 in range(dims[3]):
                        base_id = f"base_{base_index + 1:04d}"
                        for cov in coverage:
                            rows.append({
                                "candidate_id": f"{base_id}_cov_{cov:.3f}",
                                "base_candidate_id": base_id,
                                "base_index": base_index,
                                "axis_1": a0,
                                "axis_2": a1,
                                "axis_3": a2,
                                "axis_4": a3,
                                "coverage_config": cov,
                                "side": side,
                                "reserve_id": meta["reserve_id"],
                                "core_logic_name": meta["core_logic_name"],
                                "candidate_schema": schema,
                            })
                        base_index += 1
    frame = pd.DataFrame(rows)
    expected = int(np.prod(dims)) * len(coverage)
    if len(frame) != expected or frame.candidate_id.nunique() != expected:
        raise AssertionError(f"reserve candidate table mismatch: {len(frame)} != {expected}")
    if expected < 2048:
        raise AssertionError("reserve candidate table is below the small-sample minimum")
    return frame


def _link_score(c0: np.ndarray, c1: np.ndarray, c2: np.ndarray, link: int) -> np.ndarray:
    weights = np.asarray(LINKS[link], dtype=float)
    raw = weights[0] * c0 + weights[1] * c1 + weights[2] * c2
    if link == 0:
        return np.clip(raw, 0, 1)
    if link == 1:
        return np.sqrt(np.clip(raw, 0, 1))
    if link == 2:
        return np.clip(raw * (0.65 + 0.7 * c2), 0, 1)
    return np.clip(np.maximum(c0, c1) * (0.5 + 0.5 * c2), 0, 1)


def score_matrix(frame: pd.DataFrame, meta: dict[str, str], side: str) -> tuple[np.ndarray, pd.DataFrame]:
    banks = _method_primitives(frame, meta["core_logic_name"], side)
    if any(bank.shape[0] != len(frame) for bank in banks):
        raise AssertionError("reserve primitive bank row mismatch")
    dims = SCHEMA_BASE_DIMS[meta["candidate_schema"]]
    scores = []
    metadata = []
    base_index = 0
    if len(dims) == 3:
        for a0 in range(dims[0]):
            for a1 in range(dims[1]):
                for a2 in range(dims[2]):
                    scores.append(_link_score(banks[0][:, a0], banks[1][:, a1], banks[2][:, a2], a2))
                    metadata.append({"base_candidate_id": f"base_{base_index + 1:04d}", "base_index": base_index, "axis_1": a0, "axis_2": a1, "axis_3": a2, "axis_4": 0})
                    base_index += 1
    else:
        for a0 in range(dims[0]):
            for a1 in range(dims[1]):
                for a2 in range(dims[2]):
                    for a3 in range(dims[3]):
                        score = 0.45 * banks[0][:, a0] + 0.30 * banks[1][:, a1] + 0.15 * banks[2][:, a2] + 0.10 * banks[3][:, a3]
                        scores.append(np.clip(score, 0, 1))
                        metadata.append({"base_candidate_id": f"base_{base_index + 1:04d}", "base_index": base_index, "axis_1": a0, "axis_2": a1, "axis_3": a2, "axis_4": a3})
                        base_index += 1
    matrix = np.column_stack(scores)
    meta_frame = pd.DataFrame(metadata)
    if matrix.shape[1] != len(meta_frame):
        raise AssertionError("reserve score matrix metadata mismatch")
    return matrix, meta_frame


def score_candidate_pool(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    development_scores: np.ndarray,
    validation_scores: np.ndarray,
    parameters: pd.DataFrame,
    side: str,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    if development_scores.shape[1] != parameters.base_index.nunique():
        raise AssertionError("reserve base score count does not match candidate table")
    y_scale = float(development.future_open_to_open_return_1d.std(ddof=0))
    rows = []
    for parameter in parameters.itertuples(index=False):
        j = int(parameter.base_index)
        dev_score = development_scores[:, j]
        val_score = validation_scores[:, j]
        threshold = float(np.nanquantile(dev_score, 1 - float(parameter.coverage_config)))
        dev_active = dev_score >= threshold
        val_active = val_score >= threshold
        dev = _phase_metrics(development, dev_score, dev_active, side, thresholds)
        val = _phase_metrics(validation, val_score, val_active, side, thresholds)
        row = parameter._asdict()
        row.update({f"dev_{key}": value for key, value in dev.items()})
        row.update({f"val_{key}": value for key, value in val.items()})
        row["score_threshold_fitted_development"] = threshold
        row["joint_score"] = _joint_score(dev, val, y_scale)
        row["signal_hash_dev_validation"] = _signal_hash(dev_active, val_active)
        row["eligible_min_samples"] = bool(dev["n_signal"] >= 20 and val["n_signal"] >= 15)
        rows.append(row)
    result = pd.DataFrame(rows)
    result = result.sort_values(["signal_hash_dev_validation", "joint_score", "candidate_id"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
    result["signal_duplicate_rank"] = result.groupby("signal_hash_dev_validation").cumcount()
    result["is_unique_signal"] = result.signal_duplicate_rank.eq(0)
    leader = result.loc[result.is_unique_signal, ["signal_hash_dev_validation", "candidate_id"]].rename(columns={"candidate_id": "signal_duplicate_of"})
    result = result.merge(leader, on="signal_hash_dev_validation", how="left")
    result = result.sort_values(["is_unique_signal", "eligible_min_samples", "joint_score", "val_precision", "candidate_id"], ascending=[False, False, False, False, True], kind="mergesort").reset_index(drop=True)
    result["selection_rank"] = np.arange(1, len(result) + 1)
    return result


def _phase_name(date: pd.Series) -> np.ndarray:
    return np.select(
        [date.dt.year.between(2018, 2022), date.dt.year.between(2023, 2024), date.dt.year.ge(2025)],
        ["development", "validation", "test"],
        default="other",
    )


def _prediction_frame(frame: pd.DataFrame, score: np.ndarray, active: np.ndarray, side: str, threshold: float) -> pd.DataFrame:
    cols = ["date", "entry_date", "label_exit_date", "max_feature_date", "future_open_to_open_return_1d", "future_close_to_close_return_1d"]
    out = frame[cols].copy()
    out["phase"] = _phase_name(out.date)
    out["side"] = side
    out["score"] = score
    out["score_threshold_fitted_development"] = threshold
    out["alert"] = np.asarray(active, bool).astype(int)
    out["test_used_for_selection"] = False
    return out.loc[out.date.ge("2018-01-01")].reset_index(drop=True)


def _write_phase_audit(prepared: PreparedResearch, test: pd.DataFrame, thresholds: dict[str, float], output_dir: Path) -> None:
    rows = []
    for name, frame in (("development", prepared.development), ("validation", prepared.validation), ("test_frozen_observation_only", test)):
        y = frame.future_open_to_open_return_1d.astype(float)
        valid = y.notna()
        rows.append({
            "phase": name,
            "n_rows": int(len(frame)),
            "n_labeled": int(valid.sum()),
            "down_label_n": int((y[valid] <= thresholds["q10"]).sum()),
            "down_label_ratio": float((y[valid] <= thresholds["q10"]).mean()) if valid.any() else np.nan,
            "up_label_n": int((y[valid] >= thresholds["q90"]).sum()),
            "up_label_ratio": float((y[valid] >= thresholds["q90"]).mean()) if valid.any() else np.nan,
            "q10_fitted_development": thresholds["q10"],
            "q90_fitted_development": thresholds["q90"],
            "test_used_for_selection": False,
        })
    pd.DataFrame(rows).to_csv(output_dir / "phase_sample_and_label_audit.csv", index=False)


def run_reserve_version_side(
    prepared: PreparedResearch,
    meta: dict[str, str],
    side: str,
    output_dir: str | Path,
    bootstrap_draws: int = 0,
) -> dict[str, Any]:
    if side not in {"down", "up"}:
        raise ValueError("side must be down or up")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the reserve version's result contract aligned with V01-V50 and the
    # target Notebook checklist: every side records the source whitelist,
    # causal feature lineage, and the prepared input audit before scoring.
    write_json(output_dir / "data_audit.json", prepared.data_audit)
    source_whitelist().to_csv(output_dir / "source_whitelist.csv", index=False)
    feature_lineage_table().to_csv(output_dir / "feature_lineage.csv", index=False)
    thresholds = label_thresholds(prepared.development)
    parameters = candidate_parameter_table(meta, side)
    parameters.to_csv(output_dir / "candidate_parameters.csv", index=False)
    full_scores, base_metadata = score_matrix(prepared.frame, meta, side)
    base_metadata.to_csv(output_dir / "base_candidate_parameters.csv", index=False)
    dev_scores = full_scores[prepared.development.index.to_numpy(), :]
    val_scores = full_scores[prepared.validation.index.to_numpy(), :]
    candidates = score_candidate_pool(prepared.development, prepared.validation, dev_scores, val_scores, parameters, side, thresholds)
    candidates.to_csv(output_dir / "candidate_metrics_development_validation.csv", index=False)
    candidates.loc[candidates.is_unique_signal].head(20).to_csv(output_dir / "top20_development_validation.csv", index=False)
    frozen = select_top1(candidates)
    base_j = int(frozen.base_index)
    freeze_record = FrozenRecord(
        version=meta["reserve_id"],
        side=side,
        candidate_id=str(frozen.candidate_id),
        frozen_at_stage="development_validation",
        test_used_for_selection=False,
    )
    freeze_payload = {
        **freeze_record.__dict__,
        "frozen_first_before_test": True,
        "test_gate_used_for_enablement": False,
        "test_metrics_present_at_freeze_time": False,
        "candidate_selection_unit": "one_version_one_side",
        "cross_version_test_comparison_before_freeze": False,
        "test_feedback_to_later_versions": False,
        "label_thresholds_fitted_development": thresholds,
        "candidate_parameters": {key: frozen[key] for key in ("base_candidate_id", "base_index", "axis_1", "axis_2", "axis_3", "axis_4", "coverage_config", "score_threshold_fitted_development")},
        "core_logic_name": meta["core_logic_name"],
        "paper_title": meta["paper_title"],
        "source_url": meta["source_url"],
        "candidate_count_raw": int(len(parameters)),
        "candidate_count_unique_signal": int(candidates.is_unique_signal.sum()),
        "duplicate_signal_ratio": float(1 - candidates.is_unique_signal.mean()),
    }
    write_json(output_dir / "FROZEN_TOP1_BEFORE_TEST.json", freeze_payload)

    vault = prepared.new_test_vault()
    test = vault.unlock(freeze_record)
    if not vault.opened:
        raise AssertionError("reserve TestVault did not open after independent freeze")

    dev_score = dev_scores[:, base_j]
    val_score = val_scores[:, base_j]
    test_score = full_scores[test.index.to_numpy(), base_j]
    dev_metrics, dev_active = evaluate_frozen_candidate(prepared.development, dev_score, frozen, side, thresholds)
    val_metrics, val_active = evaluate_frozen_candidate(prepared.validation, val_score, frozen, side, thresholds)
    test_metrics, test_active = evaluate_frozen_candidate(test, test_score, frozen, side, thresholds)
    pd.DataFrame([
        {"phase": "development", **dev_metrics},
        {"phase": "validation", **val_metrics},
        {"phase": "test_frozen_observation_only", **test_metrics},
    ]).to_csv(output_dir / "frozen_top1_three_phase_metrics.csv", index=False)

    bins = []
    for name, phase_frame, phase_score in (
        ("development", prepared.development, dev_score),
        ("validation", prepared.validation, val_score),
        ("test_frozen_observation_only", test, test_score),
    ):
        for count in (5, 10):
            table = score_bins(phase_frame, phase_score, side, bins=count)
            if not table.empty:
                table.insert(0, "group_count", count)
                table.insert(0, "phase", name)
                bins.append(table)
    if bins:
        pd.concat(bins, ignore_index=True).to_csv(output_dir / "score_group_metrics_5_and_10.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "score_group_metrics_5_and_10.csv", index=False)

    observed_frame = prepared.frame.copy()
    outcomes = ["future_open_to_open_return_1d", "future_close_to_close_return_1d"]
    observed_frame.loc[test.index, outcomes] = test[outcomes]
    research_frame = observed_frame.loc[observed_frame.date.ge("2018-01-01") & observed_frame.future_open_to_open_return_1d.notna()].copy()
    score_all = full_scores[:, base_j]
    threshold = float(frozen.score_threshold_fitted_development)
    active_all = score_all >= threshold
    annual_metrics(research_frame, score_all[research_frame.index.to_numpy()], active_all[research_frame.index.to_numpy()], side, thresholds).to_csv(output_dir / "annual_metrics.csv", index=False)

    # Match the V01-V50 diagnostic contract: Bootstrap is computed only after
    # the side's own Dev+Val freeze and TestVault unlock.  It is observational
    # and never feeds candidate ranking or the frozen record.
    bootstrap_parts = []
    bootstrap_summaries = {}
    phase_payloads = (
        ("development", prepared.development, dev_active),
        ("validation", prepared.validation, val_active),
        ("test_frozen_observation_only", test, test_active),
    )
    version_number = int(str(meta["reserve_id"])[1:])
    for phase_number, (name, phase_frame, phase_active) in enumerate(phase_payloads, start=1):
        samples = block_bootstrap_metrics(
            phase_frame,
            phase_active,
            side,
            thresholds,
            seed=20260812 + version_number * 100 + phase_number * 10 + (0 if side == "down" else 1),
            draws=int(bootstrap_draws),
        )
        samples.insert(0, "phase", name)
        bootstrap_parts.append(samples)
        bootstrap_summaries[name] = bootstrap_summary(samples)
    if bootstrap_parts:
        pd.concat(bootstrap_parts, ignore_index=True).to_csv(output_dir / "bootstrap_samples.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "bootstrap_samples.csv", index=False)
    write_json(output_dir / "bootstrap_summary.json", bootstrap_summaries)

    errors = error_diagnostics(test, test_score, test_active, side, thresholds)
    errors.to_csv(output_dir / "test_error_diagnostics.csv", index=False)
    if errors.empty:
        pd.DataFrame(columns=["error_type"]).to_csv(output_dir / "test_error_scenario_summary.csv", index=False)
    else:
        numeric = errors.select_dtypes(include=[np.number]).columns.tolist()
        errors.groupby("error_type")[numeric].agg(["count", "mean", "median"]).to_csv(output_dir / "test_error_scenario_summary.csv")

    predictions = _prediction_frame(observed_frame, score_all, active_all, side, threshold)
    predictions.to_csv(output_dir / "frozen_top1_daily_predictions.csv", index=False)
    predictions.tail(10).to_csv(output_dir / "latest_10_scores_and_alerts.csv", index=False)
    _write_phase_audit(prepared, test, thresholds, output_dir)

    def finite(value: Any, default: float = 0.0) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if np.isfinite(value) else default

    pretest_pass = bool(
        int(dev_metrics["n_signal"]) >= 20
        and int(val_metrics["n_signal"]) >= 15
        and finite(dev_metrics["precision_lift"]) >= 1.5
        and finite(val_metrics["precision_lift"]) >= 1.5
        and finite(dev_metrics["signed_mean_o2o"]) > 0
        and finite(val_metrics["signed_mean_o2o"]) > 0
        and finite(dev_metrics["rank_ic"]) > 0
        and finite(val_metrics["rank_ic"]) > 0
        and finite(val_metrics["reverse_extreme_rate"], 1) <= 0.20
    )
    frozen_test_pass = bool(
        int(test_metrics["n_signal"]) >= 15
        and finite(test_metrics["precision_lift"]) >= 1.5
        and finite(test_metrics["signed_mean_o2o"]) > 0
        and finite(test_metrics["rank_ic"]) > 0
        and finite(test_metrics["reverse_extreme_rate"], 1) <= 0.20
    )
    labeled_predictions = predictions.loc[predictions.future_open_to_open_return_1d.notna()]
    latest_labeled = labeled_predictions.iloc[-1] if not labeled_predictions.empty else predictions.iloc[-1]
    summary = {
        "version": meta["reserve_id"],
        "side": side,
        "core_logic_name": meta["core_logic_name"],
        "title_zh": meta["title_zh"],
        "paper_title": meta["paper_title"],
        "authors": meta["authors"],
        "year": int(meta["year"]),
        "source_url": meta["source_url"],
        "candidate_count_raw": int(len(parameters)),
        "candidate_count_unique_signal": int(candidates.is_unique_signal.sum()),
        "duplicate_signal_ratio": float(1 - candidates.is_unique_signal.mean()),
        "candidate_schema_formula": meta["candidate_schema_formula"],
        "candidate_parameter_dimensions": meta["candidate_parameter_dimensions"],
        "frozen_candidate_id": str(frozen.candidate_id),
        "frozen_base_candidate_id": str(frozen.base_candidate_id),
        "frozen_score_threshold": threshold,
        "label_thresholds_development": thresholds,
        "development": dev_metrics,
        "validation": val_metrics,
        "test_frozen_observation_only": test_metrics,
        "pretest_research_pass": pretest_pass,
        "frozen_test_diagnostic_pass": frozen_test_pass,
        "formal_pass_after_frozen_test": bool(pretest_pass and frozen_test_pass),
        "test_used_for_selection": False,
        "cross_version_test_comparison_before_freeze": False,
        "test_feedback_to_later_versions": False,
        "uses_1545": False,
        "uses_nine_state": False,
        "latest_formation_date": predictions.iloc[-1].date,
        "latest_effective_date": predictions.iloc[-1].entry_date,
        "latest_exit_date": predictions.iloc[-1].label_exit_date,
        "latest_fully_labeled_formation_date": latest_labeled.date,
        "latest_fully_labeled_effective_date": latest_labeled.entry_date,
        "latest_fully_labeled_exit_date": latest_labeled.label_exit_date,
        "bootstrap_draws": int(bootstrap_draws),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def write_reserve_readme(version_dir: str | Path, meta: dict[str, str], summaries: dict[str, dict[str, Any]]) -> None:
    version_dir = Path(version_dir)
    lines = [
        f"# {meta['reserve_id']} 大涨大跌预测｜{meta['title_zh']}",
        "",
        f"- 核心逻辑：{meta['core_logic_name']}",
        f"- 论文：{meta['paper_title']}（{meta['authors']}，{meta['year']}）",
        f"- 来源：{meta['source_url']}",
        f"- 与 V01–V50 的结构差异：{meta['structural_difference']}",
        f"- 纯现货边界：{meta['pure_spot_fields']}",
        f"- 候选轴：{meta['candidate_parameter_dimensions']}",
        f"- 候选公式：{meta['candidate_schema_formula']}",
        "- Test 协议：本版本本侧独立 Dev+Val 冻结后才解锁 Test；其它版本的 Test 不参与本版选择。",
        "",
        "| 侧别 | 候选数/去重 | 冻结候选 | Dev Rank IC | Val Rank IC | Test Rank IC（冻结后观察） | 正式通过 |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for side in ("down", "up"):
        summary = summaries[side]
        lines.append(
            f"| {'大跌' if side == 'down' else '大涨'} | {summary['candidate_count_raw']}/{summary['candidate_count_unique_signal']} | "
            f"{summary['frozen_candidate_id']} | {summary['development']['rank_ic']:.4f} | "
            f"{summary['validation']['rank_ic']:.4f} | {summary['test_frozen_observation_only']['rank_ic']:.4f} | "
            f"{'是' if summary['formal_pass_after_frozen_test'] else '否'} |"
        )
    (version_dir / "README.md").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")


def run_reserve_version(
    prepared: PreparedResearch,
    meta: dict[str, str],
    version_dir: str | Path,
    bootstrap_draws: int = 0,
) -> dict[str, dict[str, Any]]:
    version_dir = Path(version_dir)
    summaries = {
        "down": run_reserve_version_side(prepared, meta, "down", version_dir / "results" / "down", bootstrap_draws),
        "up": run_reserve_version_side(prepared, meta, "up", version_dir / "results" / "up", bootstrap_draws),
    }
    write_reserve_readme(version_dir, meta, summaries)
    return summaries
