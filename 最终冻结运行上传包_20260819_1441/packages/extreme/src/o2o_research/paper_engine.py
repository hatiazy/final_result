from __future__ import annotations

"""Paper-inspired, spot-only binary constructions V135--V158.

The implementations are intentionally small-sample causal proxies.  They do
not import any derivative, cross-sectional, or four-state data.  A method
produces four eight-level primitive banks; the shared strict binary engine
then enumerates the same transparent base×coverage candidate pool used by the
earlier versions.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.preprocessing import SplineTransformer
from sklearn.preprocessing import StandardScaler
from scipy.stats import genpareto

from . import reserve_engine as base


PAPER_SCHEMA = {"paper8": (8, 8, 8, 8)}


def load_paper_metadata(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    here = Path(__file__).resolve()
    packaged = (here.parents[1] / "paper_metadata.json").resolve()
    candidate = packaged if path is None else Path(path).expanduser().resolve()
    if candidate != packaged:
        raise ValueError("company package accepts only its bundled paper registry")
    if not candidate.is_file():
        raise FileNotFoundError("bundled paper registry not found")
    rows = json.loads(candidate.read_text(encoding="utf-8"))
    expected = [f"V{i:02d}" for i in range(int(rows[0]["reserve_id"][1:]), int(rows[-1]["reserve_id"][1:]) + 1)]
    if [row["reserve_id"] for row in rows] != expected:
        raise AssertionError("paper metadata reserve ids must be contiguous")
    return {row["reserve_id"]: row for row in rows}


def _raw(frame: pd.DataFrame, name: str, fill: float = 0.0) -> np.ndarray:
    if name not in frame:
        raise KeyError(f"paper feature missing: {name}")
    return np.nan_to_num(frame[name].astype(float).to_numpy(), nan=fill, posinf=fill, neginf=fill)


def _bounded(values: np.ndarray, span: int = 128) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(values, float), nan=0.0, posinf=0.0, neginf=0.0)
    s = pd.Series(arr)
    mean = s.ewm(span=max(4, int(span)), adjust=False, min_periods=max(8, int(span) // 4)).mean()
    var = (s - mean).pow(2).ewm(span=max(4, int(span)), adjust=False, min_periods=max(8, int(span) // 4)).mean()
    z = ((s - mean) / np.sqrt(var + 1e-8)).clip(-8, 8)
    return (1.0 / (1.0 + np.exp(-z))).fillna(0.5).to_numpy(float)


def _direction(values: np.ndarray, side: str, span: int = 128) -> np.ndarray:
    score = _bounded(values, span=span)
    return 1.0 - score if side == "down" else score


def _static_tail_logit(
    frame: pd.DataFrame,
    side: str,
    columns: tuple[str, ...],
    c_value: float,
) -> np.ndarray:
    """Development-fitted sparse tail probability for V203.

    This is deliberately a block-fitted classifier, rather than the existing
    delayed-feedback online Bayesian logistic state (V199).  The fit sample is
    fixed to Development (2018--2022); Validation and Test are only scored by
    the frozen coefficients.  Missing warm-up features are imputed with
    Development medians and the standardized feature map is also frozen there.
    Thus no later label, including a masked Test outcome, can alter the score.
    """
    if "future_open_to_open_return_1d" not in frame or "date" not in frame:
        raise KeyError("V203 requires date and future_open_to_open_return_1d")
    dates = pd.to_datetime(frame["date"])
    train_mask = dates.between("2018-01-01", "2022-12-31").to_numpy()
    y = frame["future_open_to_open_return_1d"].astype(float).to_numpy()
    finite_train = train_mask & np.isfinite(y)
    if not finite_train.any():
        return np.full(len(frame), 0.10, dtype=float)
    q = float(np.nanquantile(y[finite_train], 0.10 if side == "down" else 0.90))
    labels = ((y <= q) if side == "down" else (y >= q)).astype(int)
    raw = frame.loc[:, list(columns)].astype(float).to_numpy()
    train_raw = raw[finite_train]
    medians = np.nanmedian(train_raw, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    raw = np.where(np.isfinite(raw), raw, medians[None, :])
    scaler = StandardScaler().fit(raw[finite_train])
    x = scaler.transform(raw)
    if np.unique(labels[finite_train]).size < 2:
        return np.full(len(frame), float(labels[finite_train].mean()), dtype=float)
    model = LogisticRegression(
        C=float(np.clip(c_value, 1e-4, 100.0)),
        penalty="l2",
        class_weight=None,
        solver="lbfgs",
        max_iter=600,
        random_state=0,
    )
    model.fit(x[finite_train], labels[finite_train])
    return np.clip(model.predict_proba(x)[:, 1], 0.0, 1.0)


def _static_spline_tail_logit(
    frame: pd.DataFrame,
    side: str,
    columns: tuple[str, ...],
    knots: int,
    c_value: float,
) -> np.ndarray:
    """Development-only additive spline tail classifier (V208)."""
    dates = pd.to_datetime(frame["date"])
    train_mask = dates.between("2018-01-01", "2022-12-31").to_numpy()
    y = frame["future_open_to_open_return_1d"].astype(float).to_numpy()
    finite_train = train_mask & np.isfinite(y)
    if not finite_train.any():
        return np.full(len(frame), 0.10, dtype=float)
    q = float(np.nanquantile(y[finite_train], 0.10 if side == "down" else 0.90))
    labels = ((y <= q) if side == "down" else (y >= q)).astype(int)
    raw = frame.loc[:, list(columns)].astype(float).to_numpy()
    medians = np.nanmedian(raw[finite_train], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    raw = np.where(np.isfinite(raw), raw, medians[None, :])
    try:
        transformer = SplineTransformer(
            n_knots=int(np.clip(knots, 3, 8)), degree=2,
            knots="quantile", include_bias=False,
        )
        x_train = transformer.fit_transform(raw[finite_train])
        x_all = transformer.transform(raw)
        model = LogisticRegression(
            C=float(np.clip(c_value, 1e-4, 100.0)), penalty="l2",
            solver="lbfgs", max_iter=800, class_weight=None, random_state=0,
        )
        model.fit(x_train, labels[finite_train])
        return np.clip(model.predict_proba(x_all)[:, 1], 0.0, 1.0)
    except Exception:
        return np.full(len(frame), float(labels[finite_train].mean()), dtype=float)


def _static_block_spline_tail(
    frame: pd.DataFrame,
    side: str,
    columns: tuple[str, ...],
    knots: int,
    c_value: float,
    late_weight: float,
) -> np.ndarray:
    """Temporal block ensemble of frozen spline tail experts (V209)."""
    dates = pd.to_datetime(frame["date"])
    train_mask = dates.between("2018-01-01", "2022-12-31").to_numpy()
    y = frame["future_open_to_open_return_1d"].astype(float).to_numpy()
    finite_train = train_mask & np.isfinite(y)
    if not finite_train.any():
        return np.full(len(frame), 0.10, dtype=float)
    q = float(np.nanquantile(y[finite_train], 0.10 if side == "down" else 0.90))
    labels = ((y <= q) if side == "down" else (y >= q)).astype(int)
    raw = frame.loc[:, list(columns)].astype(float).to_numpy()
    block_masks = (
        finite_train & dates.le("2020-12-31").to_numpy(),
        finite_train & dates.ge("2021-01-01").to_numpy(),
    )
    forecasts: list[np.ndarray] = []
    for block_mask in block_masks:
        if block_mask.sum() < 40 or np.unique(labels[block_mask]).size < 2:
            continue
        medians = np.nanmedian(raw[block_mask], axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        block_raw = np.where(np.isfinite(raw), raw, medians[None, :])
        try:
            transformer = SplineTransformer(
                n_knots=int(np.clip(knots, 3, 8)), degree=2,
                knots="quantile", include_bias=False,
            )
            x_block = transformer.fit_transform(block_raw[block_mask])
            x_all = transformer.transform(block_raw)
            model = LogisticRegression(
                C=float(np.clip(c_value, 1e-4, 100.0)), penalty="l2",
                solver="lbfgs", max_iter=800, class_weight=None, random_state=0,
            )
            model.fit(x_block, labels[block_mask])
            forecasts.append(np.clip(model.predict_proba(x_all)[:, 1], 0.0, 1.0))
        except Exception:
            continue
    if not forecasts:
        return np.full(len(frame), float(labels[finite_train].mean()), dtype=float)
    if len(forecasts) == 1:
        return forecasts[0]
    w = float(np.clip(late_weight, 0.0, 1.0))
    return np.clip((1.0 - w) * forecasts[0] + w * forecasts[1], 0.0, 1.0)


def _monotone_pattern(columns: tuple[str, ...], side: str) -> tuple[int, ...]:
    """Return a fixed shape prior for the SCAM-inspired V210 proxy.

    The direction is registered from the economic meaning of the spot state,
    never estimated on Validation/Test.  Directional ranks are increasing for
    the requested side; range/volatility/liquidity stress is increasing for
    both tails.  Shadows use the usual rejection asymmetry.  Unknown fields
    are deliberately unconstrained rather than assigned a data-dependent
    sign.
    """
    side_sign = 1 if side == "up" else -1
    direction_tokens = (
        "ret_", "gap_", "intraday_ret", "close_location", "trend_efficiency",
        "momentum_curvature", "range_position", "oo_ret_", "oo_down_share_",
        "oo_up_share_",
    )
    stress_tokens = (
        "vol_", "true_range", "downside_vol", "upside_vol", "jump_intensity",
        "risk_expansion", "tail_uncertainty", "compression_release", "amihud",
        "amount_z", "volume_ratio", "state_transition",
    )
    out: list[int] = []
    for name in columns:
        text = str(name)
        if "upper_shadow" in text:
            out.append(1 if side == "down" else -1)
        elif "lower_shadow" in text:
            out.append(-1 if side == "down" else 1)
        elif any(token in text for token in direction_tokens):
            out.append(side_sign)
        elif any(token in text for token in stress_tokens):
            out.append(1)
        else:
            out.append(0)
    return tuple(out)


def _static_monotone_tail_boost(
    frame: pd.DataFrame,
    side: str,
    columns: tuple[str, ...],
    leaf_nodes: int,
    learning_rate: float,
    l2_value: float,
) -> np.ndarray:
    """Shape-constrained tail probability (V210).

    This is a small, frozen proxy for shape-constrained additive/boosted
    tail models.  HistGradientBoosting is used only as a computationally
    convenient constrained learner: all monotonic signs are registered from
    spot-state semantics before fitting, with tiny trees and no interactions.
    The event target, imputation, and model are Development-only.
    """
    dates = pd.to_datetime(frame["date"])
    train_mask = dates.between("2018-01-01", "2022-12-31").to_numpy()
    y = frame["future_open_to_open_return_1d"].astype(float).to_numpy()
    finite_train = train_mask & np.isfinite(y)
    if not finite_train.any():
        return np.full(len(frame), 0.10, dtype=float)
    q = float(np.nanquantile(y[finite_train], 0.10 if side == "down" else 0.90))
    labels = ((y <= q) if side == "down" else (y >= q)).astype(int)
    raw = frame.loc[:, list(columns)].astype(float).to_numpy()
    medians = np.nanmedian(raw[finite_train], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    x = np.where(np.isfinite(raw), raw, medians[None, :])
    if np.unique(labels[finite_train]).size < 2:
        return np.full(len(frame), float(labels[finite_train].mean()), dtype=float)
    try:
        model = HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=int(np.clip(leaf_nodes, 2, 8)),
            learning_rate=float(np.clip(learning_rate, 0.005, 0.20)),
            min_samples_leaf=45,
            l2_regularization=float(np.clip(l2_value, 0.0, 20.0)),
            monotonic_cst=_monotone_pattern(columns, side),
            random_state=0,
        )
        model.fit(x[finite_train], labels[finite_train])
        return np.clip(model.predict_proba(x)[:, 1], 0.0, 1.0)
    except Exception:
        # A failed constrained fit must be a transparent constant fallback,
        # never a silently unconstrained model.
        return np.full(len(frame), float(labels[finite_train].mean()), dtype=float)


def _static_evt_hurdle_tail(
    frame: pd.DataFrame,
    side: str,
    columns: tuple[str, ...],
    tail_quantile: float,
    c_value: float,
    severity_weight: float,
) -> np.ndarray:
    """Two-stage conditional POT/GPD tail score (V211).

    A low-capacity logistic hurdle estimates the probability of crossing a
    Development POT threshold.  Conditional exceedances are then summarized
    by a frozen generalized-Pareto tail; the score is the hurdle probability
    multiplied by a bounded, current-state severity factor.  This is not a
    rolling event counter or a direct quantile-regression copy: the GPD shape
    and scale are fitted once on Development excesses and never updated from
    Validation/Test labels.
    """
    dates = pd.to_datetime(frame["date"])
    train_mask = dates.between("2018-01-01", "2022-12-31").to_numpy()
    y = frame["future_open_to_open_return_1d"].astype(float).to_numpy()
    finite_train = train_mask & np.isfinite(y)
    if not finite_train.any():
        return np.full(len(frame), 0.10, dtype=float)
    desired = y if side == "up" else -y
    q_level = float(np.clip(tail_quantile, 0.75, 0.97))
    threshold = float(np.nanquantile(desired[finite_train], q_level))
    labels = (desired >= threshold).astype(int)
    raw = frame.loc[:, list(columns)].astype(float).to_numpy()
    medians = np.nanmedian(raw[finite_train], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    raw = np.where(np.isfinite(raw), raw, medians[None, :])
    scaler = StandardScaler().fit(raw[finite_train])
    x = scaler.transform(raw)
    if np.unique(labels[finite_train]).size < 2:
        return np.full(len(frame), float(labels[finite_train].mean()), dtype=float)
    try:
        gate = LogisticRegression(
            C=float(np.clip(c_value, 1e-4, 100.0)), penalty="l2",
            solver="lbfgs", max_iter=600, random_state=0,
        )
        gate.fit(x[finite_train], labels[finite_train])
        probability = np.clip(gate.predict_proba(x)[:, 1], 1e-8, 1.0)
    except Exception:
        probability = np.full(len(frame), float(labels[finite_train].mean()), dtype=float)

    excess = desired[finite_train][labels[finite_train] == 1] - threshold
    excess = excess[np.isfinite(excess) & (excess >= 0.0)]
    if len(excess) < 12 or float(np.nanstd(excess)) <= 1e-10:
        shape, scale_tail = 0.0, float(np.nanmedian(excess)) if len(excess) else 1e-4
    else:
        try:
            shape, _, scale_tail = genpareto.fit(excess, floc=0.0)
            shape = float(np.clip(shape, -0.25, 0.80))
            scale_tail = float(max(scale_tail, 1e-6))
        except Exception:
            shape, scale_tail = 0.0, float(max(np.nanmedian(excess), 1e-4))

    # State severity is feature-only: the magnitude of the current standardized
    # spot state relative to its Development map.  The GPD factor is bounded so
    # it cannot turn a small event count into an arbitrarily sharp score.
    severity_state = np.clip(np.nanmean(np.abs(x), axis=1), 0.0, 6.0)
    gpd_scale = float(scale_tail / max(abs(threshold), 1e-4))
    shape_term = np.clip(1.0 + shape * severity_state, 0.25, 3.0)
    factor = 1.0 + float(np.clip(severity_weight, 0.0, 1.5)) * np.tanh(gpd_scale * severity_state) * shape_term
    return np.nan_to_num(np.clip(probability * factor, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)


def _static_tail_quantile(
    frame: pd.DataFrame,
    side: str,
    columns: tuple[str, ...],
    quantile: float,
    alpha: float,
) -> np.ndarray:
    """Development-only semiparametric conditional tail forecast (V204).

    The target is the signed O2O return, and the fitted lower/upper quantile
    is evaluated without any later refit.  This transfers the paper
    literature's conditional-quantile idea to the available spot OHLCV state
    while keeping the feature map and the coefficient fit block-causal.
    """
    dates = pd.to_datetime(frame["date"])
    train_mask = dates.between("2018-01-01", "2022-12-31").to_numpy()
    y = frame["future_open_to_open_return_1d"].astype(float).to_numpy()
    finite_train = train_mask & np.isfinite(y)
    if not finite_train.any():
        return np.zeros(len(frame), dtype=float)
    raw = frame.loc[:, list(columns)].astype(float).to_numpy()
    medians = np.nanmedian(raw[finite_train], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    raw = np.where(np.isfinite(raw), raw, medians[None, :])
    scaler = StandardScaler().fit(raw[finite_train])
    x = scaler.transform(raw)
    model = QuantileRegressor(
        quantile=float(np.clip(quantile, 0.02, 0.98)),
        alpha=float(np.clip(alpha, 1e-5, 10.0)),
        solver="highs",
    )
    model.fit(x[finite_train], y[finite_train])
    predicted = model.predict(x)
    return np.nan_to_num(-predicted if side == "down" else predicted, nan=0.0, posinf=0.0, neginf=0.0)


def _static_tail_boost(
    frame: pd.DataFrame,
    side: str,
    columns: tuple[str, ...],
    leaf_nodes: int,
    learning_rate: float,
) -> np.ndarray:
    """Low-capacity Development-only gradient tail probability (V205)."""
    dates = pd.to_datetime(frame["date"])
    train_mask = dates.between("2018-01-01", "2022-12-31").to_numpy()
    y = frame["future_open_to_open_return_1d"].astype(float).to_numpy()
    finite_train = train_mask & np.isfinite(y)
    if not finite_train.any():
        return np.full(len(frame), 0.10, dtype=float)
    q = float(np.nanquantile(y[finite_train], 0.10 if side == "down" else 0.90))
    labels = ((y <= q) if side == "down" else (y >= q)).astype(int)
    raw = frame.loc[:, list(columns)].astype(float).to_numpy()
    medians = np.nanmedian(raw[finite_train], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    x = np.where(np.isfinite(raw), raw, medians[None, :])
    if np.unique(labels[finite_train]).size < 2:
        return np.full(len(frame), float(labels[finite_train].mean()), dtype=float)
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=int(np.clip(leaf_nodes, 2, 8)),
        learning_rate=float(np.clip(learning_rate, 0.005, 0.20)),
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=0,
    )
    model.fit(x[finite_train], labels[finite_train])
    return np.clip(model.predict_proba(x)[:, 1], 0.0, 1.0)


def _jump_path(x: np.ndarray, means: np.ndarray, jump_penalty: float) -> np.ndarray:
    """Fit the hard statistical-jump path for a fixed set of centroids.

    The objective is the usual squared reconstruction loss plus a fixed cost
    for changing state.  This small dynamic-programming implementation keeps
    the method independent of an HMM smoother: labels inside Development may
    use the whole Development block for fitting, while later observations are
    scored by a separate one-step causal filter.
    """
    x = np.asarray(x, float)
    means = np.asarray(means, float)
    n, k = len(x), len(means)
    if n == 0:
        return np.empty(0, dtype=int)
    dist = ((x[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
    dp = np.full((n, k), np.inf, dtype=float)
    back = np.zeros((n, k), dtype=np.int16)
    dp[0] = dist[0]
    penalty = float(max(jump_penalty, 0.0))
    for i in range(1, n):
        prev = dp[i - 1]
        for state in range(k):
            costs = prev + penalty * (np.arange(k) != state)
            parent = int(np.argmin(costs))
            back[i, state] = parent
            dp[i, state] = dist[i, state] + costs[parent]
    labels = np.empty(n, dtype=int)
    labels[-1] = int(np.argmin(dp[-1]))
    for i in range(n - 1, 0, -1):
        labels[i - 1] = back[i, labels[i]]
    return labels


def _causal_jump_filter(
    x: np.ndarray,
    means: np.ndarray,
    jump_penalty: float,
    distance_scale: float | None = None,
) -> np.ndarray:
    """Return causal soft state posteriors for a fitted jump model."""
    x = np.asarray(x, float)
    means = np.asarray(means, float)
    n, k = len(x), len(means)
    if n == 0:
        return np.empty((0, k), dtype=float)
    penalty = float(max(jump_penalty, 0.0))
    # A jump penalty becomes a persistent transition prior.  This is only a
    # one-step recursion: no future feature or future label is consulted.
    change_weight = float(np.exp(-np.clip(penalty, 0.0, 20.0)))
    trans = np.full((k, k), change_weight, dtype=float)
    np.fill_diagonal(trans, 1.0)
    trans /= trans.sum(axis=1, keepdims=True)
    posterior = np.full(k, 1.0 / k, dtype=float)
    out = np.zeros((n, k), dtype=float)
    scale = float(distance_scale) if distance_scale is not None else 1.0
    scale = max(scale, 1e-3)
    for i, row in enumerate(x):
        prior = posterior @ trans if i else np.full(k, 1.0 / k, dtype=float)
        dist = ((means - row[None, :]) ** 2).sum(axis=1)
        logp = np.log(np.maximum(prior, 1e-300)) - dist / (2.0 * scale)
        logp -= np.max(logp)
        posterior = np.exp(logp)
        total = float(posterior.sum())
        posterior = posterior / total if total > 0 else np.full(k, 1.0 / k, dtype=float)
        out[i] = posterior
    return out


def _static_jump_regime_tail(
    frame: pd.DataFrame,
    side: str,
    columns: tuple[str, ...],
    n_states: int,
    jump_penalty: float,
) -> np.ndarray:
    """Statistical-jump regime with a frozen state-conditional tail rate.

    This is the spot-only implementation inspired by the statistical jump
    model literature (persistent unsupervised regimes followed by a
    supervised regime-to-tail map).  Centroids and the state-to-event rates
    are fitted only on Development.  Validation/Test use a one-step filtered
    posterior, so the regime forecast is causal rather than a smoothed HMM
    label that could see later observations.
    """
    dates = pd.to_datetime(frame["date"])
    train_mask = dates.between("2018-01-01", "2022-12-31").to_numpy()
    y = frame["future_open_to_open_return_1d"].astype(float).to_numpy()
    finite_train = train_mask & np.isfinite(y)
    raw = frame.loc[:, list(columns)].astype(float).to_numpy()
    med = np.nanmedian(raw[finite_train], axis=0) if finite_train.any() else np.nanmedian(raw, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    raw = np.where(np.isfinite(raw), raw, med[None, :])
    center = np.nanmean(raw[finite_train], axis=0) if finite_train.any() else np.nanmean(raw, axis=0)
    scale = np.nanstd(raw[finite_train], axis=0) if finite_train.any() else np.nanstd(raw, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    x = np.clip((raw - center[None, :]) / scale[None, :], -8.0, 8.0)
    if finite_train.sum() < max(40, int(n_states) * 12):
        return np.full(len(frame), 0.10, dtype=float)
    train_x = x[finite_train]
    k = int(np.clip(n_states, 2, 4))
    penalty = float(np.clip(jump_penalty, 0.01, 32.0))
    try:
        init = KMeans(n_clusters=k, n_init=8, random_state=0, max_iter=100).fit(train_x)
        means = np.asarray(init.cluster_centers_, float)
        labels = init.labels_.astype(int)
        for _ in range(8):
            labels = _jump_path(train_x, means, penalty)
            updated = means.copy()
            for state in range(k):
                mask = labels == state
                if mask.sum() >= 8:
                    updated[state] = train_x[mask].mean(axis=0)
            if np.allclose(updated, means, atol=1e-5, rtol=1e-4):
                means = updated
                break
            means = updated
    except Exception:
        return np.full(len(frame), 0.10, dtype=float)

    base_event = 0.10 if side in {"down", "up"} else 0.10
    q = float(np.nanquantile(y[finite_train], 0.10 if side == "down" else 0.90))
    event = ((y <= q) if side == "down" else (y >= q)) & finite_train
    prior = 8.0
    rates = np.full(k, base_event, dtype=float)
    for state in range(k):
        mask = labels == state
        n_state = int(mask.sum())
        if n_state:
            rates[state] = float((event[finite_train][mask].sum() + prior * base_event) / (n_state + prior))
    rates = np.clip(rates, 0.01, 0.99)
    train_distance_scale = float(np.nanmedian(((train_x - means[0]) ** 2).sum(axis=1)))
    posterior = _causal_jump_filter(x, means, penalty, train_distance_scale)
    return np.clip(posterior @ rates, 0.0, 1.0)


def _static_jump_regime_boost(
    frame: pd.DataFrame,
    side: str,
    columns: tuple[str, ...],
    n_states: int,
    leaf_nodes: int,
    learning_rate: float,
) -> np.ndarray:
    """Jump-model latent states followed by a frozen regime classifier (V207)."""
    dates = pd.to_datetime(frame["date"])
    train_mask = dates.between("2018-01-01", "2022-12-31").to_numpy()
    y = frame["future_open_to_open_return_1d"].astype(float).to_numpy()
    finite_train = train_mask & np.isfinite(y)
    raw = frame.loc[:, list(columns)].astype(float).to_numpy()
    med = np.nanmedian(raw[finite_train], axis=0) if finite_train.any() else np.nanmedian(raw, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    raw = np.where(np.isfinite(raw), raw, med[None, :])
    center = np.nanmean(raw[finite_train], axis=0) if finite_train.any() else np.nanmean(raw, axis=0)
    scale = np.nanstd(raw[finite_train], axis=0) if finite_train.any() else np.nanstd(raw, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    x = np.clip((raw - center[None, :]) / scale[None, :], -8.0, 8.0)
    if finite_train.sum() < max(60, int(n_states) * 20):
        return np.full(len(frame), 0.10, dtype=float)
    train_x = x[finite_train]
    k = int(np.clip(n_states, 2, 3))
    try:
        init = KMeans(n_clusters=k, n_init=8, random_state=0, max_iter=100).fit(train_x)
        means = np.asarray(init.cluster_centers_, float)
        labels = init.labels_.astype(int)
        for _ in range(8):
            labels = _jump_path(train_x, means, 0.50)
            updated = means.copy()
            for state in range(k):
                mask = labels == state
                if mask.sum() >= 10:
                    updated[state] = train_x[mask].mean(axis=0)
            if np.allclose(updated, means, atol=1e-5, rtol=1e-4):
                means = updated
                break
            means = updated
        if np.unique(labels).size < 2:
            return np.full(len(frame), 0.10, dtype=float)
        regime_model = HistGradientBoostingClassifier(
            max_iter=80,
            max_leaf_nodes=int(np.clip(leaf_nodes, 2, 8)),
            learning_rate=float(np.clip(learning_rate, 0.005, 0.20)),
            min_samples_leaf=60,
            l2_regularization=2.0,
            random_state=0,
        )
        regime_model.fit(train_x, labels)
        regime_prob = np.asarray(regime_model.predict_proba(x), float)
        classes = np.asarray(regime_model.classes_, int)
    except Exception:
        return np.full(len(frame), 0.10, dtype=float)

    q = float(np.nanquantile(y[finite_train], 0.10 if side == "down" else 0.90))
    event = ((y <= q) if side == "down" else (y >= q)) & finite_train
    base_event = 0.10
    rates = np.full(k, base_event, dtype=float)
    train_event = event[finite_train]
    prior = 10.0
    for state in range(k):
        mask = labels == state
        n_state = int(mask.sum())
        if n_state:
            rates[state] = float((train_event[mask].sum() + prior * base_event) / (n_state + prior))
    rates = np.clip(rates, 0.01, 0.99)
    mapped = np.zeros((len(frame), k), dtype=float)
    for j, state in enumerate(classes):
        if 0 <= int(state) < k:
            mapped[:, int(state)] = regime_prob[:, j]
    return np.clip(mapped @ rates, 0.0, 1.0)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(np.nan_to_num(values, nan=0.0)).rolling(
        int(window), min_periods=min(int(window), max(2, int(window) // 2))
    ).mean().fillna(0.0).to_numpy(float)


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(np.nan_to_num(values, nan=0.0)).rolling(
        int(window), min_periods=min(int(window), max(2, int(window) // 2))
    ).std(ddof=0).fillna(0.0).to_numpy(float)


def _rolling_quantile(values: np.ndarray, window: int, q: float) -> np.ndarray:
    return pd.Series(np.nan_to_num(values, nan=0.0)).rolling(
        int(window), min_periods=max(8, int(window) // 2)
    ).quantile(float(q)).fillna(0.0).to_numpy(float)


def _rank(values: np.ndarray, window: int = 252) -> np.ndarray:
    return base._rolling_last_rank(
        np.nan_to_num(values, nan=0.0),
        window=window,
        min_periods=min(60, max(20, window // 4)),
    )


def _ewm(values: np.ndarray, halflife: float) -> np.ndarray:
    return pd.Series(np.nan_to_num(values, nan=0.0)).ewm(
        halflife=max(1.0, float(halflife)), adjust=False,
        min_periods=max(5, int(2 * halflife)),
    ).mean().fillna(0.0).to_numpy(float)


def _midas(values: np.ndarray, window: int, curvature: float) -> np.ndarray:
    """Causal beta-weighted distributed lag, distinct from an EWM."""
    arr = np.nan_to_num(np.asarray(values, float), nan=0.0)
    out = np.zeros(len(arr), float)
    lags = np.arange(1, int(window) + 1, dtype=float)
    x = lags / max(float(window), 1.0)
    weights = np.power(np.maximum(x, 1e-6), max(0.1, curvature))
    weights = weights / weights.sum()
    for i in range(len(arr)):
        n = min(i, int(window))
        if n <= 0:
            continue
        out[i] = float(np.dot(arr[i - n : i][::-1], weights[:n]))
    return out


def _haar_detail(values: np.ndarray, short: int, long: int) -> np.ndarray:
    fast = _rolling_mean(values, short)
    slow = _rolling_mean(values, long)
    return fast - slow


def _spectral_cross(x: np.ndarray, y: np.ndarray, window: int, period: int) -> np.ndarray:
    """Causal low-order Fourier cross-energy proxy for quantile spectra."""
    x, y = np.nan_to_num(x, nan=0.0), np.nan_to_num(y, nan=0.0)
    out = np.zeros(len(x), float)
    phase = 2.0 * np.pi * np.arange(window, dtype=float) / max(2, int(period))
    c, s = np.cos(phase), np.sin(phase)
    for i in range(len(x)):
        left = max(0, i - int(window) + 1)
        xx, yy = x[left : i + 1], y[left : i + 1]
        n = len(xx)
        if n < max(12, window // 2):
            continue
        cc, ss = c[-n:], s[-n:]
        xc, xs = np.dot(xx, cc), np.dot(xx, ss)
        yc, ys = np.dot(yy, cc), np.dot(yy, ss)
        out[i] = (xc * yc + xs * ys) / (np.linalg.norm(xx) * np.linalg.norm(yy) + 1e-8)
    return out


def _conformal_tail(values: np.ndarray, window: int, side: str) -> np.ndarray:
    signed = -values if side == "down" else values
    center = _ewm(signed, max(4, window // 4))
    nonconformity = signed - center
    out = np.zeros(len(values), float)
    for i in range(len(values)):
        left = max(0, i - int(window) + 1)
        hist = nonconformity[left:i]
        if len(hist) < max(12, window // 4):
            continue
        # A recency-weighted conformal rank.  The current point is never in
        # the calibration set, so this remains a one-step-ahead score.
        recent = hist[-min(len(hist), 64):]
        rank = float(np.mean(recent <= nonconformity[i]))
        scale = float(np.median(np.abs(recent - np.median(recent))) + 1e-6)
        out[i] = rank * (1.0 + max(0.0, nonconformity[i]) / scale)
    return _bounded(out, span=max(32, window))


def _quantile_transfer_entropy_tail(
    frame: pd.DataFrame,
    side: str,
    source: np.ndarray,
    bins: int,
    half_life: float,
    source_lag: int,
    alpha: float,
) -> np.ndarray:
    """Causal quantile-transfer-entropy proxy for a tail event.

    The QTE literature decomposes a driver into quantile states and measures
    directional information transfer rather than a conditional mean.  Here we
    use a deliberately small, auditable online table: source quantile state ×
    most-recent observed tail state.  The observation for row ``j`` enters only
    at row ``j+2`` (the first point at which its O2O label is observable), so
    masked Test labels cannot influence Test scores.
    """
    x = np.asarray(source, float)
    n = len(x)
    rank = _rank(x, window=252)
    state = np.floor(np.clip(np.nan_to_num(rank, nan=0.5), 0.0, 0.999999) * int(bins)).astype(int)
    target = frame["future_open_to_open_return_1d"].astype(float).to_numpy()
    dates = pd.to_datetime(frame["date"]).to_numpy()
    dev = (dates < np.datetime64("2023-01-01")) & np.isfinite(target)
    if not np.any(dev):
        dev = np.isfinite(target)
    q = float(np.nanquantile(target[dev], 0.10 if side == "down" else 0.90))
    event = np.where(np.isfinite(target), (target <= q if side == "down" else target >= q).astype(float), np.nan)
    # Counts are exponentially discounted empirical transition probabilities;
    # alpha is a fixed prior mass, not fitted with Test outcomes.
    decay = float(np.exp(-np.log(2.0) / max(float(half_life), 1.0)))
    joint = np.zeros((int(bins), 2), float)
    marginal = np.zeros(2, float)
    prior = float(np.clip(alpha, 0.05, 8.0))
    out = np.full(n, 0.5, float)
    last_event = 0
    for i in range(n):
        joint *= decay
        marginal *= decay
        # The row i-2 label is the newest label available without using the
        # current row's future; source_lag lets the source state be delayed.
        j = i - 2
        if j >= 0 and np.isfinite(event[j]):
            sj = j - int(source_lag)
            if sj >= 0 and 0 <= state[sj] < int(bins):
                s = int(state[sj])
                e = int(event[j] > 0.5)
                joint[s, e] += 1.0
                marginal[e] += 1.0
                last_event = e
        s_now = int(state[max(0, i - int(source_lag))])
        if not 0 <= s_now < int(bins):
            s_now = int(bins // 2)
        p_cond = (joint[s_now, 1] + prior) / (joint[s_now].sum() + 2.0 * prior)
        p_base = (marginal[1] + prior) / (marginal.sum() + 2.0 * prior)
        # Conditional information gain, with a small probability component to
        # keep the score monotone when the table is still sparse.
        gain = np.log((p_cond + 1e-6) / (p_base + 1e-6))
        state_tail = (1.0 if last_event else 0.0) * 0.12
        out[i] = 0.62 * p_cond + 0.30 * np.tanh(gain) + state_tail
    return _bounded(out, span=max(32, int(half_life) * 4))


def _drawdown_speed_components(
    frame: pd.DataFrame,
    side: str,
    window: int,
    duration_power: float,
    acceleration_weight: float,
    activity_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal drawdown/drawup depth-speed-duration geometry.

    Drawdown-risk papers distinguish the depth of a decline from the speed and
    duration with which it forms.  This implementation keeps that geometry
    explicit: rolling peak/trough, underwater duration, speed, acceleration,
    and a volume/amount drawdown interaction.  It is not a generic momentum or
    event-count score.
    """
    close = np.maximum(_raw(frame, "close"), 1e-8)
    ret = _raw(frame, "ret_1")
    activity = 0.55 * np.log1p(np.maximum(_raw(frame, "volume"), 0.0)) + 0.45 * np.log1p(np.maximum(_raw(frame, "amount"), 0.0))
    s = pd.Series(close)
    rolling_peak = s.rolling(int(window), min_periods=max(8, int(window) // 4)).max().to_numpy(float)
    rolling_trough = s.rolling(int(window), min_periods=max(8, int(window) // 4)).min().to_numpy(float)
    if side == "down":
        depth = np.clip((rolling_peak - close) / np.maximum(rolling_peak, 1e-8), 0.0, 1.0)
        underwater = close < np.roll(rolling_peak, 1)
        underwater[0] = False
    else:
        depth = np.clip((close - rolling_trough) / np.maximum(rolling_trough, 1e-8), 0.0, 1.0)
        underwater = close > np.roll(rolling_trough, 1)
        underwater[0] = False
    duration = np.zeros(len(close), float)
    run = 0.0
    for i, flag in enumerate(underwater):
        run = run + 1.0 if bool(flag) else 0.0
        duration[i] = run
    speed = depth / np.power(np.maximum(duration, 1.0), max(float(duration_power), 0.05))
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
    acceleration = np.maximum(speed - np.roll(speed, 1), 0.0)
    acceleration[0] = 0.0
    activity_peak = pd.Series(activity).rolling(int(window), min_periods=max(8, int(window) // 4)).max().to_numpy(float)
    activity_drawdown = np.clip((activity_peak - activity) / np.maximum(np.abs(activity_peak), 1e-8), 0.0, 1.0)
    shock = np.abs(ret) * (1.0 + activity_drawdown)
    return (
        _bounded(depth, span=max(32, int(window))),
        _bounded(speed + 0.25 * depth, span=max(32, int(window))),
        _bounded(acceleration + float(acceleration_weight) * depth, span=max(32, int(window))),
        _bounded(depth * (1.0 + float(activity_weight) * activity_drawdown) + 0.20 * shock, span=max(32, int(window))),
    )


def _knn_analog(frame: pd.DataFrame, side: str, window: int, neighbors: int, scale: float) -> np.ndarray:
    """Causal analog score using only earlier spot states and known past O2O."""
    x = np.column_stack([
        _rank(_raw(frame, "ret_1"), 128),
        _rank(_raw(frame, "true_range_pct"), 128),
        _rank(_raw(frame, "close_location", 0.5), 128),
    ])
    target = _raw(frame, "future_open_to_open_return_1d", np.nan)
    # Test labels are masked by PreparedResearch.  Only historical rows with a
    # fully observed label and a two-row causal gap are eligible analogs.
    out = np.full(len(x), 0.5, float)
    for i in range(len(x)):
        right = i - 2
        left = max(0, right - int(window))
        if right - left < max(24, neighbors * 2):
            continue
        idx = np.arange(left, right, dtype=int)
        idx = idx[np.isfinite(target[idx])]
        if len(idx) < max(12, neighbors):
            continue
        d = np.sqrt(np.sum((x[idx] - x[i]) ** 2, axis=1))
        take = idx[np.argsort(d)[: min(int(neighbors), len(idx))]]
        dd = d[np.argsort(d)[: min(int(neighbors), len(idx))]]
        weights = np.exp(-(dd - dd.min()) / max(float(scale), 1e-3))
        weights /= weights.sum() + 1e-12
        signed = (-target[take]) if side == "down" else target[take]
        out[i] = float(_bounded(np.array([np.dot(weights, signed)]), span=32)[0])
    return out


def _local_ridge(frame: pd.DataFrame, side: str, window: int, ridge: float) -> np.ndarray:
    """Causal low-dimensional local autoregression with a fixed ridge."""
    features = np.column_stack([
        _raw(frame, "ret_1"), _raw(frame, "ret_5"), _raw(frame, "gap"),
        _raw(frame, "intraday_ret"), _raw(frame, "true_range_pct"),
        _raw(frame, "amount_z_20"), _raw(frame, "vol_20"),
    ])
    features = np.nan_to_num(features, nan=0.0)
    target = _raw(frame, "future_open_to_open_return_1d", np.nan)
    out = np.zeros(len(features), float)
    for i in range(len(features)):
        right = i - 2
        left = max(0, right - int(window))
        idx = np.arange(left, right, dtype=int)
        idx = idx[np.isfinite(target[idx])]
        if len(idx) < max(32, features.shape[1] * 4):
            continue
        x = features[idx]
        center = np.nanmedian(x, axis=0)
        spread = np.nanmedian(np.abs(x - center), axis=0) + 1e-5
        z = (x - center) / spread
        xi = (features[i] - center) / spread
        design = np.column_stack([np.ones(len(z)), z])
        try:
            coef = np.linalg.solve(design.T @ design + float(ridge) * np.eye(design.shape[1]), design.T @ target[idx])
            pred = float(np.r_[1.0, xi] @ coef)
        except np.linalg.LinAlgError:
            pred = 0.0
        out[i] = -pred if side == "down" else pred
    return _bounded(out, span=64)


def _res_caviar_path(
    signed_return: np.ndarray,
    signed_overnight: np.ndarray,
    range_pct: np.ndarray,
    alpha: float,
    beta: float,
    leverage: float,
) -> np.ndarray:
    """Causal RES-CAViaR-style recursion with overnight nowcasting inputs.

    This is a deliberately low-dimensional proxy rather than an MCMC fit.  It
    keeps the paper's joint VaR/ES idea operational for a small daily sample:
    an evolving tail level reacts to the previous signed return, overnight
    shock, and range scale, while the candidate axes vary the persistence and
    leverage response.
    """
    r = np.nan_to_num(np.asarray(signed_return, float), nan=0.0)
    oo = np.nan_to_num(np.asarray(signed_overnight, float), nan=0.0)
    rng = np.nan_to_num(np.asarray(range_pct, float), nan=0.0)
    out = np.zeros(len(r), float)
    seed = float(np.nanmedian(np.abs(r[: min(len(r), 64)]))) if len(r) else 0.0
    out[0] = max(seed, 1e-5)
    omega = max(seed * (1.0 - beta) * 0.5, 1e-6)
    for i in range(1, len(r)):
        shock = max(r[i - 1], 0.0)
        out[i] = omega + beta * out[i - 1] + alpha * (shock + 0.5 * abs(oo[i - 1]))
        out[i] += leverage * max(rng[i - 1] - out[i - 1], 0.0)
    return out


def _realized_sv_skewt_score(
    ret: np.ndarray,
    range_pct: np.ndarray,
    signed_overnight: np.ndarray,
    vol_span: int,
    skew_span: int,
    tail_shape: float,
    side: str,
) -> np.ndarray:
    """Causal realized-SV/skew-t proxy using OHLC-derived realized scale."""
    r = np.nan_to_num(np.asarray(ret, float), nan=0.0)
    rng = np.nan_to_num(np.asarray(range_pct, float), nan=0.0)
    oo = np.nan_to_num(np.asarray(signed_overnight, float), nan=0.0)
    realized = 0.5 * r * r + 0.5 * rng * rng
    variance = pd.Series(realized).ewm(
        span=max(4, int(vol_span)), adjust=False, min_periods=max(8, int(vol_span) // 3)
    ).mean().fillna(0.0).to_numpy(float)
    scale = np.sqrt(variance + 1e-8)
    z = r / scale
    z2 = pd.Series(z * z).ewm(
        span=max(4, int(skew_span)), adjust=False, min_periods=max(8, int(skew_span) // 3)
    ).mean().fillna(1.0).to_numpy(float)
    z3 = pd.Series(z * z * z).ewm(
        span=max(4, int(skew_span)), adjust=False, min_periods=max(8, int(skew_span) // 3)
    ).mean().fillna(0.0).to_numpy(float)
    skew = z3 / np.power(np.maximum(z2, 1e-6), 1.5)
    signed_shock = -r if side == "down" else r
    directional = _bounded(_ewm(signed_shock, max(4, int(skew_span) // 2)), span=2 * int(skew_span))
    skew_tail = np.maximum(0.0, (-skew if side == "down" else skew))
    score = np.log1p(np.maximum(variance, 0.0)) + tail_shape * skew_tail
    score += 0.35 * np.abs(oo) / (scale + 1e-6) + 0.35 * directional
    return score


def _reconciled_volatility(
    ret: np.ndarray,
    gap: np.ndarray,
    intra: np.ndarray,
    range_pct: np.ndarray,
    span: int,
    top_weight: float,
    cross_weight: float,
) -> np.ndarray:
    """Bottom-up/top-down forecast reconciliation for daily spot volatility."""
    ret2 = np.nan_to_num(np.asarray(ret, float), nan=0.0) ** 2
    gap2 = np.nan_to_num(np.asarray(gap, float), nan=0.0) ** 2
    intra2 = np.nan_to_num(np.asarray(intra, float), nan=0.0) ** 2
    rng2 = np.nan_to_num(np.asarray(range_pct, float), nan=0.0) ** 2
    ewm = lambda x: pd.Series(x).ewm(span=max(4, int(span)), adjust=False, min_periods=max(8, int(span) // 3)).mean().fillna(0.0).to_numpy(float)
    direct = ewm(ret2 + 0.25 * rng2)
    bottom = 0.30 * ewm(gap2) + 0.30 * ewm(intra2) + 0.25 * ewm(rng2) + 0.15 * ewm(ret2)
    cross = ewm(np.abs(np.nan_to_num(np.asarray(gap, float), nan=0.0) * np.nan_to_num(np.asarray(intra, float), nan=0.0)))
    return np.maximum(0.0, float(top_weight) * direct + (1.0 - float(top_weight)) * bottom + float(cross_weight) * cross)


def _amre_multicandle_score(
    frame: pd.DataFrame,
    side: str,
    window: int,
    range_weight: float,
    body_weight: float,
) -> np.ndarray:
    """Multi-candlestick AMRE-inspired spot-volatility efficiency proxy."""
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    hl = np.log(h / l)
    co = np.log(c / o)
    ho = np.log(h / o)
    lo = np.log(l / o)
    park = hl * hl / (4.0 * np.log(2.0))
    gk = 0.5 * hl * hl - (2.0 * np.log(2.0) - 1.0) * co * co
    rs = ho * np.log(h / c) + lo * np.log(l / c)
    candle = np.maximum(0.0, float(range_weight) * park + (1.0 - float(range_weight)) * gk)
    candle = np.maximum(0.0, candle + float(body_weight) * np.maximum(rs, 0.0))
    multi = pd.Series(candle).rolling(
        max(2, int(window)), min_periods=max(2, int(window) // 2)
    ).mean().fillna(0.0).to_numpy(float)
    baseline = pd.Series(multi).ewm(
        span=max(8, 3 * int(window)), adjust=False, min_periods=max(8, int(window))
    ).mean().fillna(0.0).to_numpy(float)
    shock = multi / (baseline + 1e-8)
    body = np.abs(co) / (hl + 1e-8)
    signed_body = (-co if side == "down" else co)
    direction = _bounded(_ewm(signed_body * (0.5 + body), max(4, int(window))), span=4 * int(window))
    return _bounded(np.log1p(np.maximum(shock, 0.0)) + 0.45 * direction, span=4 * int(window))


def _context_tree_score(
    ret: np.ndarray,
    side: str,
    depth: int,
    threshold: float,
    decay: float,
) -> np.ndarray:
    """Causal Bayesian-context-tree-style AR score for daily returns."""
    r = np.nan_to_num(np.asarray(ret, float), nan=0.0)
    n = len(r)
    scale_series = pd.Series(np.abs(r)).rolling(128, min_periods=32).quantile(0.70).shift(1)
    fallback = pd.Series(np.abs(r)).expanding(min_periods=1).median().shift(1)
    scale = scale_series.fillna(fallback).fillna(abs(float(r[0])) + 1e-6).to_numpy(float)
    signed = -r if side == "down" else r
    sums: dict[str, float] = {}
    sqs: dict[str, float] = {}
    weights: dict[str, float] = {}
    out = np.zeros(n, float)
    for i in range(n):
        symbols: list[str] = []
        for lag in range(1, int(depth) + 1):
            j = i - lag
            if j < 0:
                break
            cut = float(threshold) * max(scale[j], 1e-6)
            symbols.append("+" if r[j] > cut else "-" if r[j] < -cut else "0")
        key = "".join(symbols) or "root"
        # Back off from the full context to shorter suffixes, as in a context
        # tree, with exponentially decreasing prior weight.
        candidates = [key] + [key[k:] for k in range(1, len(key))] + ["root"]
        pred = 0.0
        total = 0.0
        for order, context in enumerate(candidates):
            w = float(decay) ** order
            denom = weights.get(context, 0.0)
            if denom > 0:
                pred += w * sums[context] / denom
                total += w
        if total > 0:
            pred /= total
        out[i] = pred
        # Current return becomes available only after today's score is formed.
        for order, context in enumerate(candidates):
            w = float(decay) ** order
            weights[context] = weights.get(context, 0.0) + w
            sums[context] = sums.get(context, 0.0) + w * signed[i]
            sqs[context] = sqs.get(context, 0.0) + w * signed[i] * signed[i]
    return _bounded(out, span=64)


def _factor_overnight_garch(
    gap: np.ndarray,
    intra: np.ndarray,
    signed_return: np.ndarray,
    alpha: float,
    beta: float,
    cross_weight: float,
    side: str,
) -> np.ndarray:
    """Two-factor overnight/intraday GARCH-Itô-inspired causal scale."""
    g = np.nan_to_num(np.asarray(gap, float), nan=0.0)
    x = np.nan_to_num(np.asarray(intra, float), nan=0.0)
    s = np.nan_to_num(np.asarray(signed_return, float), nan=0.0)
    hg = np.zeros(len(g), float)
    hx = np.zeros(len(x), float)
    seed = max(float(np.nanmedian(g * g + x * x)), 1e-8)
    hg[0] = hx[0] = seed * 0.5
    omega = seed * max(1e-4, 1.0 - beta - 0.5 * alpha)
    for i in range(1, len(g)):
        hg[i] = omega + alpha * g[i - 1] * g[i - 1] + beta * hg[i - 1]
        hx[i] = omega + alpha * x[i - 1] * x[i - 1] + beta * hx[i - 1]
    corr = pd.Series(g).rolling(64, min_periods=16).corr(pd.Series(x)).fillna(0.0).to_numpy(float)
    risk = hg + hx + float(cross_weight) * np.abs(corr) * 2.0 * np.sqrt(np.maximum(hg * hx, 0.0))
    # Use a causal surprise ratio rather than the raw tiny variance level;
    # otherwise the alpha axis collapses after the final logistic squashing.
    baseline = pd.Series(risk).ewm(span=max(16, int(8.0 / max(1e-3, 1.0 - beta))), adjust=False, min_periods=16).mean().fillna(seed).to_numpy(float)
    volatility_surprise = np.log1p(np.maximum(risk / (baseline + 1e-8) - 1.0, 0.0))
    directional = _bounded(_ewm(s, max(4, int(8 * (1.0 + beta)))), span=64)
    skew = np.maximum(0.0, _ewm(-s if side == "down" else s, 16))
    return _bounded(volatility_surprise + 0.35 * directional + 0.15 * _bounded(skew, span=32), span=64)


def _directional_change_path(
    frame: pd.DataFrame,
    side: str,
    theta: float,
    memory: int,
    contrarian: bool = False,
) -> np.ndarray:
    """Daily directional-change intrinsic-time state.

    The event clock is reconstructed from end-of-day close prices.  At each
    observation the running extremum is updated first; a confirmation is only
    emitted after a causal reversal of ``theta`` from that extremum.  The
    score combines event direction, overshoot distance and event age.  The
    optional contrarian form is used by the overshoot-hazard reserve and is
    intentionally not a calendar-time moving average.
    """
    close = np.maximum(_raw(frame, "close"), 1e-8)
    n = len(close)
    theta = max(float(theta), 1e-4)
    mode = 0
    extremum = float(close[0]) if n else 1.0
    age = 0
    event_flag = np.zeros(n, float)
    event_dir = np.zeros(n, float)
    overshoot = np.zeros(n, float)
    ages = np.zeros(n, float)
    for i in range(n):
        price = float(close[i])
        confirmed = 0.0
        if mode >= 0:
            if price >= extremum:
                extremum = price
            if price <= extremum * (1.0 - theta):
                mode = -1
                extremum = price
                age = 0
                confirmed = -1.0
        else:
            if price <= extremum:
                extremum = price
            if price >= extremum * (1.0 + theta):
                mode = 1
                extremum = price
                age = 0
                confirmed = 1.0
        age += 1
        event_flag[i] = abs(confirmed)
        event_dir[i] = float(mode)
        event_distance = abs(np.log(price / max(extremum, 1e-8))) / theta
        overshoot[i] = float(np.clip(event_distance, 0.0, 8.0))
        ages[i] = float(age)

    desired = -1.0 if side == "down" else 1.0
    aligned = desired * event_dir
    if contrarian:
        aligned = -aligned
    recency = np.exp(-ages / max(2.0, float(memory)))
    event_impulse = pd.Series(event_flag * desired * event_dir).ewm(
        span=max(4, int(memory)), adjust=False, min_periods=2
    ).mean().fillna(0.0).to_numpy(float)
    if contrarian:
        event_impulse = -event_impulse
    raw = 1.05 * aligned + 0.33 * np.tanh(overshoot / 2.0) + 0.22 * recency + 0.20 * event_impulse
    return _bounded(raw, span=max(16, 2 * int(memory)))


def _dc_overshoot_hazard(
    frame: pd.DataFrame,
    side: str,
    theta: float,
    memory: int,
    hazard_weight: float,
) -> np.ndarray:
    """Causal DC overshoot survival/hazard proxy.

    This separates the question "which way is the current DC state?" from
    "is its overshoot unusually old/large and therefore near a transition?".
    Completed event ages and magnitudes are accumulated only after they have
    occurred; no future confirmation is used to score the current row.
    """
    close = np.maximum(_raw(frame, "close"), 1e-8)
    n = len(close)
    theta = max(float(theta), 1e-4)
    mode = 0
    extremum = float(close[0]) if n else 1.0
    age = 0
    since_event = 0
    completed_ages: list[float] = []
    completed_sizes: list[float] = []
    out = np.zeros(n, float)
    desired = -1.0 if side == "down" else 1.0
    for i in range(n):
        price = float(close[i])
        confirmed = 0.0
        if mode >= 0:
            if price >= extremum:
                extremum = price
            if price <= extremum * (1.0 - theta):
                if age > 0:
                    completed_ages.append(float(age))
                    completed_sizes.append(float(abs(np.log(price / max(extremum, 1e-8)))))
                mode = -1
                extremum = price
                age = 0
                since_event = 0
                confirmed = -1.0
        else:
            if price <= extremum:
                extremum = price
            if price >= extremum * (1.0 + theta):
                if age > 0:
                    completed_ages.append(float(age))
                    completed_sizes.append(float(abs(np.log(price / max(extremum, 1e-8)))))
                mode = 1
                extremum = price
                age = 0
                since_event = 0
                confirmed = 1.0
        age += 1
        since_event += 1
        hist_age = completed_ages[-max(8, int(memory)):]
        hist_size = completed_sizes[-max(8, int(memory)):]
        expected_age = float(np.median(hist_age)) if hist_age else max(4.0, float(memory) / 4.0)
        expected_size = float(np.median(hist_size)) if hist_size else theta
        current_size = abs(np.log(price / max(extremum, 1e-8)))
        age_hazard = 1.0 - np.exp(-float(age) / max(expected_age, 1.0))
        size_hazard = 1.0 - np.exp(-current_size / max(expected_size, theta))
        # If the current state is aligned with the requested side, a large
        # hazard is a warning against continuation; if it is opposite, the
        # same hazard is a reversal opportunity.
        alignment = desired * float(mode)
        raw = alignment * (1.0 - float(hazard_weight) * age_hazard)
        raw += (-alignment) * float(hazard_weight) * (0.65 * age_hazard + 0.35 * size_hazard)
        raw += 0.15 * desired * confirmed
        out[i] = raw
    return _bounded(out, span=max(16, 2 * int(memory)))


def _volume_clock_toxicity(
    frame: pd.DataFrame,
    side: str,
    window: int,
    bucket_scale: float,
    shock_weight: float,
) -> np.ndarray:
    """Daily-only VPIN/BVC proxy in volume time.

    Intraday trades are unavailable by design.  We therefore infer a signed
    bulk-volume fraction from close location and candle body, and update the
    toxicity state with volume weights rather than equal calendar weights.
    """
    volume = np.maximum(_raw(frame, "volume"), 0.0)
    location = np.clip(_raw(frame, "close_location", 0.5), 0.0, 1.0)
    body = _raw(frame, "body_pct")
    tr = np.maximum(_raw(frame, "true_range_pct"), 1e-8)
    signed_fraction = np.clip(0.65 * (2.0 * location - 1.0) + 0.35 * np.tanh(body / tr), -1.0, 1.0)
    vol_scale = pd.Series(volume).rolling(max(8, int(window)), min_periods=max(8, int(window) // 2)).median()
    bucket = np.maximum(vol_scale.fillna(pd.Series(volume).expanding(min_periods=1).median()).to_numpy(float) * float(bucket_scale), 1e-8)
    # A volume-clock update: high-volume bars consume more of the bucket and
    # therefore receive more weight than low-volume calendar observations.
    state = np.zeros(len(volume), float)
    used = np.zeros(len(volume), float)
    for i in range(len(volume)):
        weight = np.clip(volume[i] / bucket[i], 0.05, 4.0)
        decay = np.exp(-weight / max(float(window), 2.0))
        if i == 0:
            state[i] = signed_fraction[i]
            used[i] = weight
        else:
            state[i] = decay * state[i - 1] + (1.0 - decay) * signed_fraction[i]
            used[i] = decay * used[i - 1] + (1.0 - decay) * weight
    imbalance = state
    toxicity = np.abs(imbalance)
    flow_change = np.abs(np.diff(imbalance, prepend=imbalance[0]))
    desired = -1.0 if side == "down" else 1.0
    direction = desired * imbalance
    volume_surprise = _rank(volume / (pd.Series(volume).rolling(max(8, int(window)), min_periods=max(8, int(window) // 2)).median().replace(0, np.nan)).fillna(0.0).to_numpy(float), window=252)
    raw = direction + float(shock_weight) * toxicity + 0.20 * flow_change + 0.12 * np.nan_to_num(volume_surprise, nan=0.5)
    return _bounded(raw, span=max(32, 2 * int(window)))


def _ohlc_state_regression(
    frame: pd.DataFrame,
    side: str,
    window: int,
    lag: int,
    ridge: float,
) -> np.ndarray:
    """Rolling ridge forecast on the paper's unconstrained OHLC coordinates.

    The raw OHLC constraints are removed with log/logit coordinates, then a
    small causal state regression is fitted to the exact O2O target.  For row
    ``i`` only labels through ``i-2`` are eligible, matching the two-row O2O
    availability gap.  This is not a generic feature model: the input state
    is exactly the four transformed OHLC coordinates from the structural VAR
    construction.
    """
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    span = np.maximum(h - l, 1e-8)
    def logit(x: np.ndarray) -> np.ndarray:
        z = np.clip(x, 1e-5, 1.0 - 1e-5)
        return np.log(z / (1.0 - z))
    state = np.column_stack([
        np.log(c / pc),
        np.log(h / l),
        logit((o - l) / span),
        logit((c - l) / span),
    ])
    target = _raw(frame, "future_open_to_open_return_1d", np.nan)
    out = np.zeros(len(state), float)
    p = max(1, int(lag))
    # Materialise the lagged constrained-state design once.  Coefficients are
    # then updated with a causal recursive ridge (RLS) filter.  This is the
    # online analogue of the paper's rolling VAR/state regression and keeps a
    # full 20-year score surface fast enough for a 4,096-base candidate grid.
    lagged = np.full((len(state), 4 * p), np.nan, float)
    for offset in range(p):
        start = p - 1 - offset
        lagged[p - 1 :, 4 * offset : 4 * (offset + 1)] = state[start : len(state) - offset]
    dimension = 1 + 4 * p
    beta = np.zeros(dimension, float)
    covariance = np.eye(dimension, dtype=float) / max(float(ridge), 1e-3)
    forgetting = float(np.exp(-1.0 / max(float(window), 4.0)))
    target_scale = 1e-3
    for i in range(len(state)):
        # At row i, target[i-2] is the newest O2O label whose two future opens
        # are already observed.  Test labels are masked, so they cannot update
        # this filter after the freeze boundary.
        update = i - 2
        if update >= 0 and np.isfinite(target[update]) and np.isfinite(lagged[update]).all():
            x_update = np.r_[1.0, np.clip(lagged[update], -12.0, 12.0)]
            px = covariance @ x_update
            denominator = forgetting + float(x_update @ px)
            gain = px / max(denominator, 1e-8)
            error = float(target[update] - x_update @ beta)
            beta += gain * error
            covariance = (covariance - np.outer(gain, x_update @ covariance)) / forgetting
            covariance = 0.5 * (covariance + covariance.T)
            target_scale = 0.98 * target_scale + 0.02 * max(abs(float(target[update])), 1e-5)
        if np.isfinite(lagged[i]).all():
            x_current = np.r_[1.0, np.clip(lagged[i], -12.0, 12.0)]
            pred = float(x_current @ beta)
            out[i] = (-pred if side == "down" else pred) / max(target_scale, 1e-5)
    return _bounded(out, span=max(32, int(window) // 2))


def _symbolic_grammar_score(frame: pd.DataFrame, side: str, family: int, scale: float) -> np.ndarray:
    """Interpretable nonlinear OHLCV grammar bank.

    Each expression deliberately combines at least three distinct causal
    inputs.  It is a fixed, auditable grammar inspired by symbolic-regression
    feature engineering, not a random genetic search whose Test results could
    silently become a selection channel.
    """
    desired = -1.0 if side == "down" else 1.0
    body = desired * _raw(frame, "body_pct")
    gap = desired * _raw(frame, "gap")
    intra = desired * _raw(frame, "intraday_ret")
    loc = np.clip(_raw(frame, "close_location", 0.5), 0.0, 1.0)
    rng = np.maximum(_raw(frame, "range_pct"), 0.0)
    volume = np.clip(_raw(frame, "volume_ratio_20"), -3.0, 5.0)
    amount = np.clip(_raw(frame, "amount_z_20"), -4.0, 4.0)
    tr = np.maximum(_raw(frame, "true_range_pct"), 1e-8)
    shadow = _raw(frame, "upper_shadow_share", 0.5) - _raw(frame, "lower_shadow_share", 0.5)
    trend = desired * _raw(frame, "ret_5")
    vscale = np.maximum(_rolling_std(body, max(8, int(scale))), 1e-5)
    x = body / vscale
    y = gap / (tr + 1e-8)
    z = (2.0 * loc - 1.0) * (1.0 + rng / (np.nanmedian(rng) + 1e-8))
    q = np.tanh(volume / 2.0) + 0.5 * np.tanh(amount / 2.0)
    formulas = [
        np.tanh(0.75 * x + 0.35 * y + 0.25 * z),
        np.tanh(0.55 * x * (1.0 + np.abs(z)) + 0.30 * q + 0.20 * trend / (vscale + 1e-8)),
        np.tanh(0.60 * y + 0.35 * z * np.tanh(x) - 0.20 * shadow),
        np.tanh(0.45 * x + 0.40 * q * np.sign(x + 1e-9) + 0.30 * np.tanh(trend / (vscale + 1e-8))),
        np.tanh(0.40 * (x + y) + 0.35 * np.tanh(z * q) + 0.20 * np.sign(body) * np.sqrt(np.abs(rng))),
        np.tanh(0.50 * x * np.tanh(q) + 0.35 * y * np.tanh(z) + 0.25 * trend),
        np.tanh(0.65 * np.tanh(x) + 0.25 * np.tanh(y * z) + 0.25 * np.tanh(q - shadow)),
        np.tanh(0.45 * x + 0.30 * y + 0.30 * z + 0.20 * q + 0.15 * np.tanh(shadow * x)),
    ]
    return _bounded(formulas[int(family) % len(formulas)], span=max(16, 4 * int(scale)))


def _order_imbalance_inventory(frame: pd.DataFrame, side: str, window: int, reversal: float, impact: float) -> np.ndarray:
    """Daily order-imbalance/inventory-pressure proxy.

    The lagged imbalance component captures persistent split orders while the
    contemporaneous component is assigned the inventory-reversal sign, as in
    the order-imbalance return mechanism.  It uses only bar geometry and
    volume; no bid/ask or derivative data are inferred.
    """
    location = np.clip(_raw(frame, "close_location", 0.5), 0.0, 1.0)
    body = _raw(frame, "body_pct")
    volume = np.maximum(_raw(frame, "volume"), 0.0)
    signed_flow = volume * np.clip(0.70 * (2.0 * location - 1.0) + 0.30 * np.tanh(body / (np.abs(body).mean() + 1e-8)), -1.0, 1.0)
    denom = pd.Series(volume).rolling(max(8, int(window)), min_periods=max(8, int(window) // 2)).sum().replace(0, np.nan)
    imbalance = pd.Series(signed_flow).rolling(max(2, int(window)), min_periods=max(2, int(window) // 2)).sum() / denom
    imbalance = imbalance.fillna(0.0).to_numpy(float)
    lagged = np.roll(imbalance, 1)
    lagged[0] = 0.0
    current = imbalance
    ret = _raw(frame, "ret_1")
    desired = -1.0 if side == "down" else 1.0
    impact_state = pd.Series(desired * ret * current).ewm(span=max(4, int(window)), adjust=False, min_periods=4).mean().fillna(0.0).to_numpy(float)
    raw = desired * (lagged - float(reversal) * current) + float(impact) * impact_state
    return _bounded(raw, span=max(16, 2 * int(window)))


def _overnight_jump_reversal(
    frame: pd.DataFrame,
    side: str,
    window: int,
    z_cut: float,
    memory: int,
    volume_weight: float,
) -> np.ndarray:
    """Causal overnight-jump shock and short-horizon reversal state."""
    gap = _raw(frame, "gap")
    intra = _raw(frame, "intraday_ret")
    amount = _raw(frame, "amount_z_20")
    volume = np.maximum(_raw(frame, "volume"), 0.0)
    gs = pd.Series(gap)
    center = gs.rolling(max(8, int(window)), min_periods=max(8, int(window) // 2)).median().shift(1)
    mad = (gs - center).abs().rolling(max(8, int(window)), min_periods=max(8, int(window) // 2)).median().shift(1)
    scale = mad.fillna(gs.abs().expanding(min_periods=8).median().shift(1)).fillna(1e-3).clip(lower=1e-5).to_numpy(float)
    center_arr = center.fillna(0.0).to_numpy(float)
    z = (gap - center_arr) / scale
    jump = np.abs(z) >= float(z_cut)
    desired = -1.0 if side == "down" else 1.0
    # A positive score means the overnight shock points toward the requested
    # reversal: negative gap for up, positive gap for down.
    reversal = -desired * z
    day_confirmation = desired * intra * (1.0 + 0.25 * np.tanh(np.abs(z)))
    volume_rank = _rank(np.log1p(volume), window=252)
    event = pd.Series(np.where(jump, reversal, 0.0)).ewm(
        span=max(4, int(memory)), adjust=False, min_periods=2
    ).mean().fillna(0.0).to_numpy(float)
    persistence = pd.Series(jump.astype(float)).rolling(
        max(4, int(memory)), min_periods=2
    ).mean().fillna(0.0).to_numpy(float)
    raw = event + 0.25 * day_confirmation + 0.20 * persistence + float(volume_weight) * np.nan_to_num(volume_rank, nan=0.5)
    raw += 0.08 * desired * np.tanh(amount / 2.0)
    return _bounded(raw, span=max(32, 2 * int(memory)))


def _overnight_daytime_tugwar(frame: pd.DataFrame, side: str, window: int, threshold: float, asymmetry: float) -> np.ndarray:
    """Rolling frequency of high/low opening reversals (tug-of-war state)."""
    gap = _raw(frame, "gap")
    intra = _raw(frame, "intraday_ret")
    tr = np.maximum(_raw(frame, "true_range_pct"), 1e-6)
    g = gap / tr
    x = intra / tr
    large = np.abs(g) >= float(threshold)
    high_open_reversal = large & (g > 0) & (x < 0)
    low_open_reversal = large & (g < 0) & (x > 0)
    pos = pd.Series(high_open_reversal.astype(float)).rolling(
        max(5, int(window)), min_periods=max(5, int(window) // 2)
    ).mean().fillna(0.0).to_numpy(float)
    neg = pd.Series(low_open_reversal.astype(float)).rolling(
        max(5, int(window)), min_periods=max(5, int(window) // 2)
    ).mean().fillna(0.0).to_numpy(float)
    desired = -1.0 if side == "down" else 1.0
    # The published asymmetry is strongest for the high-opening reversal; the
    # side orientation is kept explicit so it can be audited and rejected if
    # it collapses to a generic gap rule.
    raw = desired * (float(asymmetry) * pos - (1.0 - float(asymmetry)) * neg)
    raw += 0.25 * desired * pd.Series(g * x).rolling(
        max(5, int(window)), min_periods=max(5, int(window) // 2)
    ).mean().fillna(0.0).to_numpy(float)
    return _bounded(raw, span=max(16, 2 * int(window)))


def _cross_quantilogram_gap_intraday(frame: pd.DataFrame, side: str, window: int, q_gap: float, q_intra: float, lag: int) -> np.ndarray:
    """Causal binary cross-quantile dependence of gap and intraday returns."""
    gap = _raw(frame, "gap")
    intra = _raw(frame, "intraday_ret")
    gr = _rank(gap, window=max(60, int(window)))
    ir = _rank(intra, window=max(60, int(window)))
    a = gr <= float(q_gap)
    b = ir >= float(q_intra)
    if side == "down":
        a = gr >= 1.0 - float(q_gap)
        b = ir <= 1.0 - float(q_intra)
    out = np.zeros(len(gap), float)
    lag = max(1, int(lag))
    for i in range(len(gap)):
        right = i - lag
        left = max(0, right - int(window) + 1)
        if right - left < max(24, int(window) // 3):
            continue
        x = a[left : right + 1]
        y = b[left + lag : right + lag + 1]
        if len(x) != len(y) or len(x) < 16:
            continue
        p_x = float(np.mean(x))
        p_y = float(np.mean(y))
        joint = float(np.mean(x & y))
        # Centered conditional excess over the unconditional quantile base
        # rate; this is the binary cross-quantilogram analogue.
        out[i] = (joint - p_x * p_y) / np.sqrt(max(p_x * (1 - p_x) * p_y * (1 - p_y), 1e-6))
    return _bounded(out, span=max(32, int(window) // 2))


def _opening_reversal_liquidity(frame: pd.DataFrame, side: str, window: int, open_cut: float, memory: int, volume_weight: float) -> np.ndarray:
    """Opening-location reversal state with a causal liquidity filter."""
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    prev_h = np.roll(h, 1)
    prev_l = np.roll(l, 1)
    prev_c = np.roll(c, 1)
    prev_h[0] = h[0]
    prev_l[0] = l[0]
    prev_c[0] = c[0]
    location = np.clip((o - prev_l) / np.maximum(prev_h - prev_l, 1e-8), 0.0, 1.0)
    gap = o / np.maximum(prev_c, 1e-8) - 1.0
    intra = c / o - 1.0
    high_open = (location >= 1.0 - float(open_cut)) & (gap > 0) & (intra < 0)
    low_open = (location <= float(open_cut)) & (gap < 0) & (intra > 0)
    high_rate = pd.Series(high_open.astype(float)).rolling(max(5, int(window)), min_periods=max(5, int(window) // 2)).mean().fillna(0.0).to_numpy(float)
    low_rate = pd.Series(low_open.astype(float)).rolling(max(5, int(window)), min_periods=max(5, int(window) // 2)).mean().fillna(0.0).to_numpy(float)
    desired = -1.0 if side == "down" else 1.0
    # Positive high-opening reversals are the asymmetric paper channel; for a
    # downside call the low-opening counterpart is the natural mirror.
    raw = desired * (high_rate - low_rate)
    recent = pd.Series(desired * intra * np.abs(gap)).ewm(span=max(4, int(memory)), adjust=False, min_periods=2).mean().fillna(0.0).to_numpy(float)
    amount = np.nan_to_num(_rank(np.log1p(np.maximum(_raw(frame, "amount"), 0.0)), window=252), nan=0.5)
    # The scale for the gap channel must be available online.  A full-sample
    # mean would let later (including Test) gap magnitudes change earlier
    # scores, even though no labels were used.  Use a trailing mean and a
    # causal expanding warm-up fallback instead.
    gap_abs = pd.Series(np.abs(gap))
    gap_scale = gap_abs.rolling(252, min_periods=20).mean()
    gap_scale = gap_scale.combine_first(gap_abs.expanding(min_periods=1).mean()).fillna(1e-8).to_numpy(float)
    raw += 0.35 * recent + float(volume_weight) * amount * desired * np.tanh(gap / (gap_scale + 1e-8))
    return _bounded(raw, span=max(16, 2 * int(memory)))


def _har_semirange_leverage(frame: pd.DataFrame, side: str, memory: int, leverage: float, jump_weight: float) -> np.ndarray:
    """Daily OHLC proxy for signed semivariance HAR with leverage and jumps."""
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    v = np.maximum(_raw(frame, "volume"), 0.0)
    gap = np.log(o / pc)
    intra = np.log(c / o)
    ret = np.log(c / pc)
    rng = np.log(h / l)
    loc = np.clip((c - l) / np.maximum(h - l, 1e-8), 0.0, 1.0)
    down_energy = np.maximum(-gap, 0.0) ** 2 + np.maximum(-intra, 0.0) ** 2 + rng**2 * (1.0 - loc)
    up_energy = np.maximum(gap, 0.0) ** 2 + np.maximum(intra, 0.0) ** 2 + rng**2 * loc
    signed = -1.0 if side == "down" else 1.0
    primary = down_energy if side == "down" else up_energy
    opposite = up_energy if side == "down" else down_energy
    total = down_energy + up_energy
    volume_ratio = v / pd.Series(v).rolling(20, min_periods=5).median().replace(0.0, np.nan).to_numpy()
    volume_ratio = np.nan_to_num(volume_ratio, nan=1.0, posinf=4.0, neginf=1.0)
    flow = signed * ret * np.log1p(np.abs(volume_ratio - 1.0))

    def roll(x: np.ndarray, w: int) -> np.ndarray:
        return pd.Series(np.nan_to_num(x, nan=0.0)).rolling(
            max(2, int(w)), min_periods=max(2, int(w) // 2)
        ).mean().to_numpy()

    windows = (2, 3, 5, 8, 13, 22, 44, 66)
    b0 = [_bounded(roll(primary, w), span=max(16, 2 * w)) for w in windows]
    har = []
    for i, w in enumerate(windows):
        d = roll(primary, max(2, w // 2))
        wk = roll(primary, max(3, w))
        mo = roll(primary, max(5, 3 * w))
        jump = np.maximum(primary - roll(primary, max(8, 4 * w)), 0.0)
        har.append(_bounded(d * (0.60 + 0.05 * i) + wk * 0.28 + mo * 0.12 + float(jump_weight) * jump, span=max(24, 3 * w)))
    b1 = har
    lev = signed * np.minimum(ret, 0.0) if side == "down" else signed * np.maximum(ret, 0.0)
    lev = np.maximum(lev, 0.0) * np.maximum(rng, 0.0)
    asym = primary / (total + 1e-8)
    b2 = [_bounded(roll(lev, w) + (0.25 + 0.05 * i) * roll(asym, max(3, w)), span=max(24, 2 * w)) for i, w in enumerate(windows)]
    shock = np.maximum(primary - roll(primary, 22), 0.0) + np.maximum(np.abs(flow), 0.0) * float(jump_weight)
    b3 = [_bounded(roll(shock, w) + 0.20 * roll(opposite, max(3, w)), span=max(24, 2 * w)) for w in windows]
    return np.column_stack(b0), np.column_stack(b1), np.column_stack(b2), np.column_stack(b3)


def _two_tail_pot_hawkes(frame: pd.DataFrame, side: str, memory: int, asymmetry: float, magnitude_weight: float) -> np.ndarray:
    """Two-tail asymmetric self/cross-excitation proxy with excess sizes."""
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    ret = np.log(c / pc)
    abs_ret = np.abs(ret)
    base_cut = pd.Series(abs_ret).shift(1).rolling(max(16, int(memory)), min_periods=max(8, int(memory) // 2)).quantile(0.85).to_numpy()
    base_cut = np.maximum(np.nan_to_num(base_cut, nan=np.nanmedian(abs_ret) + 1e-6), 1e-6)
    down_excess = np.maximum((-ret - base_cut) / base_cut, 0.0)
    up_excess = np.maximum((ret - base_cut) / base_cut, 0.0)
    down_event = (down_excess > 0).astype(float)
    up_event = (up_excess > 0).astype(float)
    decays = (2.0, 3.0, 5.0, 8.0, 13.0, 22.0, 44.0, 66.0)

    def excitation(event: np.ndarray, other: np.ndarray, hl: float, cross: float) -> np.ndarray:
        own = _ewm(np.r_[0.0, event[:-1]], hl)
        oth = _ewm(np.r_[0.0, other[:-1]], hl)
        mag = _ewm(np.r_[0.0, event[:-1] * np.maximum(magnitude_weight, 0.0)], hl)
        return own + cross * oth + mag

    target_down = []
    target_up = []
    for hl in decays:
        target_down.append(excitation(down_event, up_event, hl, 1.0 / max(float(asymmetry), 0.25)))
        target_up.append(excitation(up_event, down_event, hl, float(asymmetry)))
    target = target_down if side == "down" else target_up
    own_excess = down_excess if side == "down" else up_excess
    opp_excess = up_excess if side == "down" else down_excess
    b0 = [_bounded(x + 0.35 * own_excess, span=max(24, int(3 * decays[i]))) for i, x in enumerate(target)]
    b1 = [_bounded(_ewm(np.r_[0.0, own_excess[:-1]], decays[i]) + 0.25 * x, span=max(24, int(3 * decays[i]))) for i, x in enumerate(target)]
    b2 = [_bounded(x + (0.15 + 0.04 * i) * _ewm(np.r_[0.0, opp_excess[:-1]], decays[i]), span=max(24, int(3 * decays[i]))) for i, x in enumerate(target)]
    shock = np.maximum(abs_ret / base_cut - 1.0, 0.0)
    b3 = [_bounded(x + 0.20 * _ewm(np.r_[0.0, shock[:-1]], decays[i]), span=max(24, int(3 * decays[i]))) for i, x in enumerate(target)]
    return np.column_stack(b0), np.column_stack(b1), np.column_stack(b2), np.column_stack(b3)


def _conditional_duration_pot(frame: pd.DataFrame, side: str, memory: int, threshold: float, persistence: float) -> np.ndarray:
    """ACD-POT-style duration and excess-magnitude state for daily bars."""
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    ret = np.log(c / pc)
    shock = np.abs(ret)
    cut = pd.Series(shock).shift(1).rolling(max(16, int(memory)), min_periods=max(8, int(memory) // 2)).quantile(float(threshold)).to_numpy()
    cut = np.maximum(np.nan_to_num(cut, nan=np.nanmedian(shock) + 1e-6), 1e-6)
    down = np.maximum((-ret - cut) / cut, 0.0)
    up = np.maximum((ret - cut) / cut, 0.0)
    event = down if side == "down" else up
    other = up if side == "down" else down
    flag = event > 0
    n = len(ret)
    duration = np.zeros(n, float)
    conditional = np.zeros(n, float)
    since_other = np.zeros(n, float)
    cond0 = max(4.0, float(memory) / 4.0)
    if n:
        conditional[0] = cond0
    for i in range(1, n):
        duration[i] = 0.0 if flag[i] else duration[i - 1] + 1.0
        since_other[i] = 0.0 if other[i] > 0 else since_other[i - 1] + 1.0
        observed = duration[i - 1] if flag[i - 1] else max(duration[i - 1], 1.0)
        conditional[i] = max(1.0, (1.0 - float(persistence)) * cond0 + float(persistence) * conditional[i - 1] + (1.0 - float(persistence)) * observed)
    hazard = 1.0 / (1.0 + duration / np.maximum(conditional, 1.0))
    overdue = duration / np.maximum(conditional, 1.0)
    recency = np.exp(-duration / max(2.0, float(memory)))
    magnitude = _ewm(event, max(2.0, float(memory) / 3.0))
    other_recency = np.exp(-since_other / max(2.0, float(memory)))
    b0 = [_bounded(hazard * (0.55 + 0.04 * i) + recency * (0.20 + 0.02 * i) + magnitude, span=max(24, int(2 * memory))) for i in range(8)]
    b1 = [_bounded(overdue / (1.0 + overdue) + (0.20 + 0.04 * i) * magnitude, span=max(24, int(2 * memory))) for i in range(8)]
    b2 = [_bounded(recency + (0.18 + 0.04 * i) * other_recency + 0.30 * magnitude, span=max(24, int(2 * memory))) for i in range(8)]
    b3 = [_bounded(magnitude * (0.55 + 0.05 * i) + hazard * 0.35 + np.maximum(event, 0.0), span=max(24, int(2 * memory))) for i in range(8)]
    return np.column_stack(b0), np.column_stack(b1), np.column_stack(b2), np.column_stack(b3)


def _gas_var_es(frame: pd.DataFrame, side: str, learning: float, alpha: float, es_ratio: float) -> np.ndarray:
    """Low-dimensional score-driven joint VaR/ES state."""
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    proxy = np.log(c / pc)
    y = _raw(frame, "future_open_to_open_return_1d", np.nan)
    signed_y = (-y if side == "down" else y)
    lag_target = np.r_[np.nan, np.nan, signed_y[:-2]]
    lag_proxy = np.r_[0.0, proxy[:-1]]
    observed = np.where(
        np.isfinite(lag_target),
        np.maximum(lag_target, 0.0),
        np.maximum(((-lag_proxy) if side == "down" else lag_proxy), 0.0),
    )
    n = len(observed)
    q = np.zeros(n, float)
    es = np.zeros(n, float)
    seed = float(np.nanmedian(observed[: min(n, 64)])) if n else 1e-4
    seed = max(seed, 1e-5)
    if n:
        q[0] = seed
        es[0] = seed * max(1.05, float(es_ratio))
    for i in range(1, n):
        previous = observed[i - 1]
        score = (1.0 if previous > q[i - 1] else 0.0) - float(alpha)
        exceed = max(previous - q[i - 1], 0.0)
        q[i] = max(1e-7, q[i - 1] + float(learning) * (score * max(q[i - 1], 1e-6) + 0.35 * exceed))
        es[i] = max(q[i], (1.0 - 0.5 * float(learning)) * es[i - 1] + 0.5 * float(learning) * max(previous, q[i]) * max(float(es_ratio), 1.0))
    current = np.maximum(((-proxy) if side == "down" else proxy), 0.0)
    scale = np.maximum(q, 1e-7)
    span = max(24, int(16 / max(float(learning), 0.01)))
    b0 = [_bounded(q * (0.75 + 0.04 * i) + current * 0.25, span=span) for i in range(8)]
    b1 = [_bounded(es * (0.70 + 0.05 * i) + current * 0.30, span=span) for i in range(8)]
    b2 = [_bounded(np.maximum(es - scale, 0.0) * (0.60 + 0.05 * i) + current, span=span) for i in range(8)]
    b3 = [_bounded(current * (0.55 + 0.05 * i) + es / scale, span=span) for i in range(8)]
    return np.column_stack(b0), np.column_stack(b1), np.column_stack(b2), np.column_stack(b3)


def _restricted_quantile_scale(frame: pd.DataFrame, side: str, window: int, lower_q: float, upper_q: float) -> np.ndarray:
    """Causal restricted quantile-scale dynamics for tail separation."""
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    o = np.maximum(_raw(frame, "open"), 1e-8)
    ret = np.log(c / pc)
    gap = np.log(o / pc)
    intra = np.log(c / o)
    scale_input = 0.55 * ret + 0.25 * gap + 0.20 * intra
    s = pd.Series(scale_input)
    low = s.shift(1).rolling(max(8, int(window)), min_periods=max(8, int(window) // 2)).quantile(float(lower_q)).to_numpy()
    high = s.shift(1).rolling(max(8, int(window)), min_periods=max(8, int(window) // 2)).quantile(float(upper_q)).to_numpy()
    low = np.nan_to_num(low, nan=np.nanmedian(scale_input))
    high = np.nan_to_num(high, nan=np.nanmedian(scale_input))
    width = np.maximum(high - low, 1e-6)
    center = 0.5 * (high + low)
    current = scale_input
    signed = -1.0 if side == "down" else 1.0
    tail = np.maximum(signed * (low if side == "down" else high), 0.0)
    center_pressure = np.maximum(-signed * center, 0.0)
    width_change = np.maximum(np.diff(width, prepend=width[0]), 0.0)
    current_tail = np.maximum(signed * (current - center), 0.0)
    windows = (8, 13, 22, 33, 44, 66, 99, 132)
    b0 = [_bounded(tail / width * (0.70 + 0.04 * i), span=max(32, w)) for i, w in enumerate(windows)]
    b1 = [_bounded(width * (0.60 + 0.05 * i) + width_change, span=max(32, w)) for i, w in enumerate(windows)]
    b2 = [_bounded(center_pressure * (0.55 + 0.05 * i) + current_tail, span=max(32, w)) for i, w in enumerate(windows)]
    b3 = [_bounded(current_tail * (0.55 + 0.05 * i) + tail / width, span=max(32, w)) for i, w in enumerate(windows)]
    return np.column_stack(b0), np.column_stack(b1), np.column_stack(b2), np.column_stack(b3)


def _heavy_range_volume_leverage(frame: pd.DataFrame, side: str, memory: int, measurement_weight: float, leverage: float) -> np.ndarray:
    """HEAVY-style latent return/measurement volatility state from OHLCV."""
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    v = np.maximum(_raw(frame, "volume"), 0.0)
    ret = np.log(c / pc)
    rng = np.log(h / l)
    median_logv = max(float(np.nanmedian(np.log1p(v))), 1e-6)
    measure = rng**2 + 0.25 * np.log1p(v) * np.maximum(rng**2, 1e-8) / median_logv
    return_var = ret**2
    signed = -1.0 if side == "down" else 1.0
    shock = np.maximum(signed * ret, 0.0) * np.maximum(rng, 0.0)
    windows = (3, 5, 8, 13, 22, 33, 44, 66)
    latent, observed, levs = [], [], []
    for w in windows:
        rv = _ewm(return_var, max(2.0, w / 3.0))
        m = _ewm(measure, max(2.0, w / 3.0))
        state = (1.0 - float(measurement_weight)) * rv + float(measurement_weight) * m
        latent.append(state)
        observed.append(measure / (m + 1e-8))
        levs.append(_ewm(shock, max(2.0, w / 3.0)))
    b0 = [_bounded(x, span=max(24, 2 * w)) for x, w in zip(latent, windows)]
    b1 = [_bounded(x * (0.65 + 0.04 * i) + float(leverage) * levs[i], span=max(24, 2 * w)) for i, (x, w) in enumerate(zip(latent, windows))]
    b2 = [_bounded(observed[i] * (0.60 + 0.05 * i) + latent[i], span=max(24, 2 * w)) for i, w in enumerate(windows)]
    b3 = [_bounded(levs[i] * (0.65 + 0.04 * i) + observed[i], span=max(24, 2 * w)) for i, w in enumerate(windows)]
    return np.column_stack(b0), np.column_stack(b1), np.column_stack(b2), np.column_stack(b3)


def _threshold_quantile_autoregression(frame: pd.DataFrame, side: str, window: int, threshold_q: float, persistence: float) -> np.ndarray:
    """Causal threshold quantile autoregression with two tail regimes."""
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    ret = np.log(c / pc)
    y = _raw(frame, "future_open_to_open_return_1d", np.nan)
    signed_y = (-y if side == "down" else y)
    lag_y = np.r_[np.nan, np.nan, signed_y[:-2]]
    proxy = -ret if side == "down" else ret
    state_input = np.where(np.isfinite(lag_y), lag_y, np.r_[0.0, proxy[:-1]])
    s = pd.Series(state_input)
    threshold = s.shift(1).rolling(max(8, int(window)), min_periods=max(8, int(window) // 2)).quantile(float(threshold_q)).to_numpy()
    threshold = np.nan_to_num(threshold, nan=np.nanmedian(state_input))
    regime = state_input >= threshold
    n = len(state_input)
    fast = np.zeros(n, float)
    slow = np.zeros(n, float)
    for i in range(n):
        if i == 0:
            fast[i] = max(state_input[i], 0.0)
            slow[i] = max(state_input[i], 0.0)
        else:
            fast[i] = (1.0 - float(persistence)) * max(state_input[i], 0.0) + float(persistence) * fast[i - 1]
            slow[i] = (1.0 - 0.5 * float(persistence)) * max(state_input[i], 0.0) + 0.5 * float(persistence) * slow[i - 1]
    current = np.maximum(proxy, 0.0)
    regime_score = regime.astype(float) * (fast + 0.35 * slow)
    transition = np.abs(np.diff(regime.astype(float), prepend=float(regime[0])))
    windows = (8, 13, 22, 33, 44, 66, 99, 132)
    b0 = [_bounded(regime_score * (0.60 + 0.05 * i) + current * 0.20, span=max(32, w)) for i, w in enumerate(windows)]
    b1 = [_bounded(fast * (0.65 + 0.04 * i) + slow * 0.25, span=max(32, w)) for i, w in enumerate(windows)]
    b2 = [_bounded((1.0 - regime.astype(float)) * slow * (0.55 + 0.05 * i) + current, span=max(32, w)) for i, w in enumerate(windows)]
    b3 = [_bounded(transition * (0.60 + 0.04 * i) + regime_score, span=max(32, w)) for i, w in enumerate(windows)]
    return np.column_stack(b0), np.column_stack(b1), np.column_stack(b2), np.column_stack(b3)


def _volume_conditioned_reversal(frame: pd.DataFrame, side: str, window: int, volume_cut: float, reversal_weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Volume-conditioned momentum/reversal state.

    The construction follows the paper idea that volume changes the persistence
    versus reversal response of a past price move.  It is not a raw volume
    anomaly: the signed return is explicitly gated by high/low volume states.
    """
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    o = np.maximum(_raw(frame, "open"), 1e-8)
    v = np.maximum(_raw(frame, "volume"), 0.0)
    ret = np.log(c / pc)
    gap = np.log(o / pc)
    intra = np.log(c / o)
    signed = -1.0 if side == "down" else 1.0
    sr = signed * ret
    vr = _rank(np.log1p(v), window=max(32, int(window) * 4))
    vr = np.nan_to_num(vr, nan=0.5)
    high = np.maximum(vr - float(volume_cut), 0.0) / max(1.0 - float(volume_cut), 1e-6)
    low = np.maximum(float(volume_cut) - vr, 0.0) / max(float(volume_cut), 1e-6)
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    momentum, reversal, conditional, flow = [], [], [], []
    for w in windows:
        m = _rolling_mean(sr, w)
        # High-volume moves are allowed to reverse; low-volume moves retain
        # the direction.  The coefficient is a pre-registered axis.
        cond = m * (1.0 - low) - float(reversal_weight) * m * high
        momentum.append(_bounded(m, span=max(16, 2 * w)))
        reversal.append(_bounded(-m, span=max(16, 2 * w)))
        conditional.append(_bounded(cond, span=max(16, 2 * w)))
        flow.append(_bounded(_rolling_mean(sr * (1.0 - high) - float(reversal_weight) * sr * high, w), span=max(16, 2 * w)))
    b0 = np.column_stack(conditional)
    b1 = np.column_stack([_bounded((1.0 - float(reversal_weight)) * _rolling_mean(sr, w) - float(reversal_weight) * _rolling_mean(sr, w) * high, span=max(16, 2 * w)) for w in windows])
    b2 = np.column_stack([_bounded(_rolling_mean(np.maximum(sr, 0.0) * low - np.maximum(sr, 0.0) * high, w), span=max(16, 2 * w)) for w in windows])
    b3 = np.column_stack([_bounded(flow[i] + 0.25 * _rolling_mean(signed * (gap - intra), w), span=max(16, 2 * w)) for i, w in enumerate(windows)])
    return b0, b1, b2, b3


def _panic_momentum_crash(frame: pd.DataFrame, side: str, window: int, panic_cut: float, rebound_weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crash-state conditional momentum/rebound direction proxy."""
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    ret = np.log(c / pc)
    signed = -1.0 if side == "down" else 1.0
    sr = signed * ret
    drawdown = c / pd.Series(c).rolling(max(22, int(window) * 2), min_periods=max(8, int(window))).max().to_numpy() - 1.0
    vol = pd.Series(ret).rolling(max(8, int(window)), min_periods=max(4, int(window) // 2)).std(ddof=0).to_numpy()
    vol_long = pd.Series(ret).rolling(max(33, int(window) * 4), min_periods=max(12, int(window))).std(ddof=0).to_numpy()
    vol_ratio = np.nan_to_num(vol / np.maximum(vol_long, 1e-8), nan=1.0, posinf=4.0, neginf=1.0)
    panic = np.maximum(-drawdown, 0.0) * np.maximum(vol_ratio - float(panic_cut), 0.0)
    panic_state = _bounded(panic, span=max(32, int(window) * 3))
    mom = _rolling_mean(sr, max(3, int(window) // 2))
    # In a panic state the opposite of the recent momentum is the rebound
    # channel; outside panic the ordinary signed momentum channel remains.
    rebound = np.maximum(-mom, 0.0) * panic_state * float(rebound_weight)
    continuation = np.maximum(mom, 0.0) * (1.0 - 0.5 * panic_state)
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([_bounded(rebound + continuation * (0.35 + 0.05 * i), span=max(24, w)) for i, w in enumerate(windows)])
    b1 = np.column_stack([_bounded(panic * (0.55 + 0.05 * i) + np.maximum(-_rolling_mean(sr, w), 0.0) * float(rebound_weight), span=max(24, w)) for i, w in enumerate(windows)])
    b2 = np.column_stack([_bounded(np.maximum(_rolling_mean(sr, w), 0.0) * (1.0 - panic_state) + rebound, span=max(24, w)) for w in windows])
    b3 = np.column_stack([_bounded(panic_state + _rolling_mean(np.abs(sr), w) * (0.25 + 0.04 * i), span=max(24, w)) for i, w in enumerate(windows)])
    return b0, b1, b2, b3


def _liquidity_imbalance_absorption(frame: pd.DataFrame, side: str, window: int, saturation: float, reversal_weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nonlinear daily order-flow impact/absorption reversal state."""
    o = np.maximum(_raw(frame, "open"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    v = np.maximum(_raw(frame, "volume"), 0.0)
    ret = np.log(c / pc)
    body = np.log(c / o)
    loc = np.clip((c - l) / np.maximum(h - l, 1e-8), 0.0, 1.0)
    signed = -1.0 if side == "down" else 1.0
    flow = np.clip(0.65 * (2.0 * loc - 1.0) + 0.35 * np.tanh(body / np.maximum(np.abs(np.log(h / l)), 1e-8)), -1.0, 1.0)
    signed_flow = signed * flow
    v_series = pd.Series(v)
    vol_med = v_series.rolling(max(8, int(window)), min_periods=max(4, int(window) // 2)).median()
    # Warm-up values use only the history available at that row; never use a
    # global median that would incorporate future/Test volume observations.
    vol_med = vol_med.combine_first(v_series.expanding(min_periods=1).median()).fillna(0.0)
    vscale = np.maximum(vol_med.to_numpy(float), 1e-8)
    impact = np.abs(ret) / np.maximum(np.log1p(v / vscale), 1e-4)
    impact_state = _bounded(impact, span=max(32, int(window) * 2))
    absorption = np.abs(signed_flow) / np.maximum(np.abs(ret), 1e-5)
    absorption_state = _bounded(absorption, span=max(32, int(window) * 2))
    # Large flow with weak price impact is absorption and favors reversal;
    # large impact with persistent flow favors continuation.
    reversal = np.maximum(absorption_state - float(saturation), 0.0) * np.maximum(-_rolling_mean(signed_flow, window), 0.0) * float(reversal_weight)
    continuation = np.maximum(_rolling_mean(signed_flow, window), 0.0) * impact_state
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([_bounded(reversal + continuation * (0.35 + 0.05 * i), span=max(24, w)) for i, w in enumerate(windows)])
    b1 = np.column_stack([_bounded(_rolling_mean(-signed_flow, w) * (0.55 + 0.04 * i) + absorption_state * float(reversal_weight), span=max(24, w)) for i, w in enumerate(windows)])
    b2 = np.column_stack([_bounded(_rolling_mean(signed_flow, w) * (0.55 + 0.04 * i) + impact_state, span=max(24, w)) for i, w in enumerate(windows)])
    b3 = np.column_stack([_bounded(absorption_state * (0.55 + 0.05 * i) + impact_state * 0.35, span=max(24, w)) for i, w in enumerate(windows)])
    return b0, b1, b2, b3


def _large_move_volume_followup(frame: pd.DataFrame, side: str, window: int, extreme_q: float, continuation_weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Large-move continuation/reversal conditional on abnormal volume."""
    o = np.maximum(_raw(frame, "open"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    v = np.maximum(_raw(frame, "volume"), 0.0)
    ret = np.log(c / pc)
    gap = np.log(o / pc)
    intra = np.log(c / o)
    signed = -1.0 if side == "down" else 1.0
    sr = signed * ret
    volume_rank = np.nan_to_num(_rank(np.log1p(v), window=max(32, int(window) * 4)), nan=0.5)
    cut = pd.Series(np.abs(ret)).shift(1).rolling(max(16, int(window) * 2), min_periods=max(8, int(window))).quantile(float(extreme_q)).to_numpy()
    cut = np.maximum(np.nan_to_num(cut, nan=np.nanmedian(np.abs(ret)) + 1e-6), 1e-6)
    excess = np.maximum(np.abs(ret) - cut, 0.0) / cut
    high = np.maximum(volume_rank - 0.60, 0.0) / 0.40
    low = np.maximum(0.40 - volume_rank, 0.0) / 0.40
    # Empirical setup: high-volume large moves tend to continue, while
    # low-volume large moves tend to reverse.  The sign is side-specific.
    conditional = sr * (float(continuation_weight) * high - (1.0 - float(continuation_weight)) * low)
    conditional *= (0.35 + excess)
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([_bounded(_rolling_mean(conditional, w), span=max(24, w)) for w in windows])
    b1 = np.column_stack([_bounded(_rolling_mean(conditional * (0.55 + 0.05 * i) + sr * excess, w), span=max(24, w)) for i, w in enumerate(windows)])
    b2 = np.column_stack([_bounded(_rolling_mean(sr * high - sr * low, w) + 0.25 * _rolling_mean(excess, w), span=max(24, w)) for w in windows])
    b3 = np.column_stack([_bounded(_rolling_mean(signed * (gap - intra) * (1.0 + excess), w) + 0.20 * _rolling_mean(sr, w), span=max(24, w)) for w in windows])
    return b0, b1, b2, b3


def _state_conditional_momentum_adaptation(
    frame: pd.DataFrame,
    side: str,
    window: int,
    state_cut: float,
    crash_cut: float,
    adaptation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """State-conditional momentum mapping inspired by adaptive momentum.

    The recent return is not assigned one fixed continuation sign.  In a calm
    trend state it is treated as momentum; in a high-volatility or opposite
    trend state the mapping shifts toward a contrarian response.  This differs
    from a volatility gate because the *sign* of the return map changes with
    the joint trend/volatility state.
    """
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    ret = np.log(c / pc)
    signed = (-ret if side == "down" else ret)
    fast = _ewm(ret, max(2.0, float(window) / 3.0))
    slow = _ewm(ret, max(4.0, float(window)))
    short_vol = _rolling_std(ret, max(5, int(window) // 2))
    long_vol = _rolling_std(ret, max(20, int(window) * 3))
    vol_ratio = short_vol / np.maximum(long_vol, 1e-8)
    trend_state = np.where(slow >= float(state_cut) * np.maximum(long_vol, 1e-8), 1.0,
                           np.where(slow <= -float(state_cut) * np.maximum(long_vol, 1e-8), -1.0, 0.0))
    stress = np.maximum(vol_ratio - float(crash_cut), 0.0)
    stress = np.clip(stress / max(1.0, 2.0 - float(crash_cut)), 0.0, 1.0)
    desired = -1.0 if side == "down" else 1.0
    aligned = np.maximum(desired * trend_state, 0.0)
    # Calm aligned states preserve momentum; stress/opposite states gradually
    # rotate the mapping toward reversal.  The transition term reacts faster.
    state_weight = np.clip(aligned * (1.0 - stress) + (1.0 - aligned) * float(adaptation), 0.0, 1.0)
    mapped = signed * state_weight - signed * (1.0 - state_weight) * float(adaptation)
    accel = desired * (fast - slow) / np.maximum(long_vol, 1e-8)
    transition = np.abs(np.diff(trend_state, prepend=trend_state[0]))
    regime_memory = _ewm(mapped, max(3.0, float(window) / 2.0))
    range_pct = np.maximum(_raw(frame, "true_range_pct"), 0.0)
    b0 = np.column_stack([_bounded(mapped * (0.65 + 0.04 * i) + 0.18 * accel, span=max(24, 2 * int(window))) for i in range(8)])
    b1 = np.column_stack([_bounded(regime_memory + (0.18 + 0.03 * i) * mapped * stress, span=max(24, 2 * int(window))) for i in range(8)])
    b2 = np.column_stack([_bounded(mapped * (1.0 - 0.35 * stress) + (0.12 + 0.04 * i) * desired * range_pct, span=max(24, 2 * int(window))) for i in range(8)])
    b3 = np.column_stack([_bounded(transition * (0.45 + 0.05 * i) + np.abs(mapped) * (0.30 + 0.03 * i), span=max(24, 2 * int(window))) for i in range(8)])
    return b0, b1, b2, b3


def _volume_visibility_premium(
    frame: pd.DataFrame,
    side: str,
    window: int,
    shock_cut: float,
    memory: int,
    continuation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """High-volume visibility/attention shock with causal decay."""
    o = np.maximum(_raw(frame, "open"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    v = np.maximum(_raw(frame, "volume"), 0.0)
    ret = np.log(c / pc)
    gap = np.log(o / pc)
    intra = np.log(c / o)
    loc = np.clip((c - l) / np.maximum(h - l, 1e-8), 0.0, 1.0)
    logv = np.log1p(v)
    baseline = pd.Series(logv).shift(1).rolling(max(8, int(window)), min_periods=max(8, int(window) // 2)).median()
    surprise = logv - baseline.fillna(pd.Series(logv).expanding(min_periods=1).median()).to_numpy(float)
    high = np.maximum(surprise - float(shock_cut), 0.0)
    low = np.maximum(-surprise - float(shock_cut), 0.0)
    high = np.tanh(high / 1.5)
    low = np.tanh(low / 1.5)
    attention = _ewm(high, max(2.0, float(memory)))
    desired = -1.0 if side == "down" else 1.0
    signed = desired * ret
    signed_intra = desired * intra
    # Visibility shocks are allowed to carry a signed move, while a low-volume
    # shock is a weaker contrarian channel; both are explicit axes.
    premium = signed * (float(continuation) * high + 0.35 * low)
    flow = desired * (0.60 * ret + 0.40 * intra)
    range_pct = np.maximum(np.log(h / l), 0.0)
    b0 = np.column_stack([_bounded(premium * (0.60 + 0.05 * i) + 0.15 * attention, span=max(24, 2 * int(memory))) for i in range(8)])
    b1 = np.column_stack([_bounded(flow * high * (0.55 + 0.05 * i) + 0.25 * _ewm(flow * high, max(3.0, float(memory) / 2.0)), span=max(24, 2 * int(memory))) for i in range(8)])
    b2 = np.column_stack([_bounded(attention * (0.55 + 0.04 * i) + premium * (1.0 + 0.15 * i), span=max(24, 2 * int(memory))) for i in range(8)])
    b3 = np.column_stack([_bounded(premium * (1.0 + range_pct) * (0.50 + 0.05 * i) + 0.20 * desired * (gap - intra), span=max(24, 2 * int(memory))) for i in range(8)])
    return b0, b1, b2, b3


def _volume_autocorrelation_inventory(
    frame: pd.DataFrame,
    side: str,
    window: int,
    volume_cut: float,
    reversal_weight: float,
    continuation_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Volume-dependent return autocorrelation/inventory response."""
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    ret = np.log(c / pc)
    v = np.maximum(_raw(frame, "volume"), 0.0)
    logv = np.log1p(v)
    vr = _rank(logv, window=max(64, int(window) * 4))
    vr = np.nan_to_num(vr, nan=0.5)
    high = np.maximum(vr - float(volume_cut), 0.0) / max(1.0 - float(volume_cut), 1e-6)
    low = np.maximum(float(volume_cut) - vr, 0.0) / max(float(volume_cut), 1e-6)
    r = pd.Series(ret)
    lag = r.shift(1)
    mu = r.rolling(max(8, int(window)), min_periods=max(8, int(window) // 2)).mean()
    mu_lag = lag.rolling(max(8, int(window)), min_periods=max(8, int(window) // 2)).mean()
    cov = ((r - mu) * (lag - mu_lag)).rolling(max(8, int(window)), min_periods=max(8, int(window) // 2)).mean()
    var = ((r - mu) ** 2).rolling(max(8, int(window)), min_periods=max(8, int(window) // 2)).mean()
    rho = np.clip(np.nan_to_num((cov / np.maximum(var, 1e-8)).to_numpy(float), nan=0.0), -1.0, 1.0)
    desired = -1.0 if side == "down" else 1.0
    signed = desired * ret
    # High volume weakens serial continuation and makes a reversal response
    # more plausible; low volume preserves the inventory continuation channel.
    reversal = -signed * high * (float(reversal_weight) + 0.35 * np.maximum(-rho, 0.0))
    continuation = signed * low * (float(continuation_weight) + 0.35 * np.maximum(rho, 0.0))
    raw = reversal + continuation
    range_pct = np.maximum(_raw(frame, "true_range_pct"), 0.0)
    b0 = np.column_stack([_bounded(_ewm(raw, max(2.0, float(window) / (1.0 + 0.12 * i))), span=max(24, 2 * int(window))) for i in range(8)])
    b1 = np.column_stack([_bounded(reversal * (0.55 + 0.05 * i) + continuation, span=max(24, 2 * int(window))) for i in range(8)])
    b2 = np.column_stack([_bounded(raw + (0.12 + 0.04 * i) * desired * range_pct * (high + low), span=max(24, 2 * int(window))) for i in range(8)])
    b3 = np.column_stack([_bounded((1.0 - np.abs(rho)) * np.abs(raw) * (0.55 + 0.05 * i) + 0.20 * desired * ret, span=max(24, 2 * int(window))) for i in range(8)])
    return b0, b1, b2, b3


def _delayed_liquidity_shock(
    frame: pd.DataFrame,
    side: str,
    window: int,
    shock_cut: float,
    memory: int,
    reversal_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Distributed-lag response to an Amihud-style liquidity shock."""
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    ret = np.log(c / pc)
    amount = np.maximum(_raw(frame, "amount"), 0.0)
    volume = np.maximum(_raw(frame, "volume"), 0.0)
    dollar = np.maximum(amount, np.nanmedian(amount) * 1e-6 + 1e-8)
    impact = np.abs(ret) / np.log1p(dollar)
    baseline = pd.Series(impact).shift(1).rolling(max(8, int(window)), min_periods=max(8, int(window) // 2)).median()
    base = baseline.fillna(pd.Series(impact).expanding(min_periods=1).median()).to_numpy(float)
    ratio = impact / np.maximum(base, 1e-8)
    shock = np.maximum(ratio - float(shock_cut), 0.0)
    shock = np.tanh(shock)
    lagged = np.r_[0.0, shock[:-1]]
    delayed = _ewm(lagged, max(2.0, float(memory)))
    normalization = np.maximum(delayed - shock, 0.0)
    persistence = np.maximum(shock - delayed, 0.0)
    desired = -1.0 if side == "down" else 1.0
    signed = desired * ret
    reversal = -signed * normalization * float(reversal_weight)
    continuation = signed * persistence * (1.0 - 0.35 * float(reversal_weight))
    raw = reversal + continuation
    vol_state = _rank(np.log1p(volume), window=252)
    vol_state = np.nan_to_num(vol_state, nan=0.5)
    b0 = np.column_stack([_bounded(_ewm(raw, max(2.0, float(memory) / (1.0 + 0.10 * i))), span=max(24, 2 * int(memory))) for i in range(8)])
    b1 = np.column_stack([_bounded(reversal * (0.55 + 0.05 * i) + continuation, span=max(24, 2 * int(memory))) for i in range(8)])
    b2 = np.column_stack([_bounded(normalization * (0.60 + 0.04 * i) - persistence * 0.25 + 0.15 * signed * vol_state, span=max(24, 2 * int(memory))) for i in range(8)])
    b3 = np.column_stack([_bounded(raw * (0.55 + 0.05 * i) + np.abs(shock) * (0.20 + 0.03 * i), span=max(24, 2 * int(memory))) for i in range(8)])
    return b0, b1, b2, b3


def _lagged_week_reversal(
    frame: pd.DataFrame,
    side: str,
    lag: int,
    window: int,
    volume_cut: float,
    confirmation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Explicit one-week-lag reversal channel from Chinese-market evidence."""
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    v = np.maximum(_raw(frame, "volume"), 0.0)
    ret = np.log(c / pc)
    lagged = pd.Series(ret).shift(int(lag)).fillna(0.0).to_numpy(float)
    lag_abs = np.abs(lagged)
    cut = pd.Series(np.abs(ret)).shift(int(lag) + 1).rolling(max(16, int(window)), min_periods=max(8, int(window) // 2)).quantile(0.65).to_numpy()
    cut = np.maximum(np.nan_to_num(cut, nan=np.nanmedian(np.abs(ret)) + 1e-6), 1e-6)
    event = np.tanh(lag_abs / cut)
    vol_rank = _rank(np.log1p(v), window=252)
    vol_lag = np.roll(vol_rank, int(lag)); vol_lag[: int(lag)] = 0.5
    quiet = np.maximum(float(volume_cut) - vol_lag, 0.0) / max(float(volume_cut), 1e-6)
    desired = -1.0 if side == "down" else 1.0
    reversal = -desired * lagged * event * (0.65 + 0.35 * quiet)
    current_confirm = desired * (0.60 * ret + 0.40 * np.log(c / o))
    delayed = _ewm(reversal, max(2.0, float(window) / 2.0))
    range_pct = np.log(np.maximum(h, l) / np.maximum(l, 1e-8))
    raw = reversal + float(confirmation) * current_confirm
    b0 = np.column_stack([_bounded(raw * (0.60 + 0.05 * i), span=max(24, 2 * int(window))) for i in range(8)])
    b1 = np.column_stack([_bounded(delayed + (0.18 + 0.03 * i) * reversal, span=max(24, 2 * int(window))) for i in range(8)])
    b2 = np.column_stack([_bounded(reversal * quiet * (0.55 + 0.05 * i) + 0.20 * current_confirm, span=max(24, 2 * int(window))) for i in range(8)])
    b3 = np.column_stack([_bounded(np.abs(reversal) * (0.55 + 0.05 * i) + current_confirm * range_pct, span=max(24, 2 * int(window))) for i in range(8)])
    return b0, b1, b2, b3


def _extreme_event_decay(
    frame: pd.DataFrame,
    side: str,
    window: int,
    extreme_q: float,
    memory: int,
    continuation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Power-law-like post-extreme response with event age and decay state."""
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    ret = np.log(c / pc)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    v = np.maximum(_raw(frame, "volume"), 0.0)
    absret = np.abs(ret)
    cut = pd.Series(absret).shift(1).rolling(max(16, int(window)), min_periods=max(8, int(window) // 2)).quantile(float(extreme_q)).to_numpy()
    cut = np.maximum(np.nan_to_num(cut, nan=np.nanmedian(absret) + 1e-6), 1e-6)
    event = (absret >= cut).astype(float)
    desired = -1.0 if side == "down" else 1.0
    signed_event = desired * ret * event
    age = np.zeros(len(ret), float)
    for i in range(1, len(ret)):
        age[i] = 0.0 if event[i] > 0 else age[i - 1] + 1.0
    decay = np.exp(-age / max(2.0, float(memory)))
    impulse = _ewm(event, max(2.0, float(memory)))
    event_direction = _ewm(signed_event, max(2.0, float(memory) / 2.0))
    range_ratio = np.log(h / l) / np.maximum(c / pc - 1.0 + 1.0, 1e-8)
    range_ratio = np.nan_to_num(range_ratio, nan=0.0, posinf=4.0, neginf=0.0)
    volume_state = np.nan_to_num(_rank(np.log1p(v), window=252), nan=0.5)
    reversal = -event_direction * decay * (1.0 - 0.35 * float(continuation))
    follow = desired * ret * impulse * float(continuation)
    raw = reversal + follow
    b0 = np.column_stack([_bounded(raw * (0.60 + 0.05 * i), span=max(24, 2 * int(memory))) for i in range(8)])
    b1 = np.column_stack([_bounded(reversal * (0.55 + 0.05 * i) + follow, span=max(24, 2 * int(memory))) for i in range(8)])
    b2 = np.column_stack([_bounded(impulse * decay * (0.55 + 0.04 * i) + reversal, span=max(24, 2 * int(memory))) for i in range(8)])
    b3 = np.column_stack([_bounded(np.abs(event_direction) * (0.60 + 0.04 * i) + volume_state * np.abs(range_ratio) * 0.15, span=max(24, 2 * int(memory))) for i in range(8)])
    return b0, b1, b2, b3


def _negative_tail_transition_hazard(
    frame: pd.DataFrame,
    side: str,
    window: int,
    state_cut: float,
    age_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a causal three-state transition hazard for the target tail.

    The state is formed from the current return scaled by a trailing volatility
    estimate.  Transition counts are shifted one row before rolling, so the
    transition ending on the current row is not used to predict the next row.
    This is a Markov transition hazard, rather than a static trend/volatility
    gate or a duration-only tail score.
    """
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    ret = np.log(c / pc)
    scale = _rolling_std(ret, max(20, int(window) * 2))
    z = ret / np.maximum(scale, 1e-6)
    state = np.select([z <= -float(state_cut), z >= float(state_cut)], [-1, 1], default=0).astype(int)
    src = pd.Series(state).shift(1)
    dst = pd.Series(state)
    w = max(32, int(window))
    min_periods = max(12, w // 3)
    target = -1 if side == "down" else 1
    opposite = -target

    def transition_prob(destination: int) -> np.ndarray:
        probs = []
        for source_state in (-1, 0, 1):
            source_mask = src.eq(source_state)
            denom = source_mask.shift(1).rolling(w, min_periods=min_periods).sum()
            count = (source_mask & dst.eq(destination)).shift(1).rolling(w, min_periods=min_periods).sum()
            probs.append((count / np.maximum(denom, 1.0)).fillna(1.0 / 3.0).to_numpy(float))
        return np.select([state == -1, state == 0, state == 1], probs, default=1.0 / 3.0)

    p_target = transition_prob(target)
    p_opposite = transition_prob(opposite)
    p_neutral = transition_prob(0)
    edge = p_target - p_opposite
    entropy = -(p_target * np.log(np.maximum(p_target, 1e-8))
                + p_opposite * np.log(np.maximum(p_opposite, 1e-8))
                + p_neutral * np.log(np.maximum(p_neutral, 1e-8))) / np.log(3.0)
    age = np.zeros(len(state), dtype=float)
    for i in range(1, len(state)):
        age[i] = age[i - 1] + 1.0 if state[i] == state[i - 1] else 1.0
    age_boost = 1.0 - np.exp(-age / max(2.0, float(window) * max(float(age_weight), 0.1)))
    hazard = np.clip(p_target * (0.60 + 0.40 * age_boost) + 0.20 * edge, 0.0, 1.0)
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([_bounded(hazard * (0.70 + 0.04 * i), span=max(24, w)) for i in range(8)])
    b1 = np.column_stack([_bounded(edge * (0.65 + 0.05 * i), span=max(24, w)) for i in range(8)])
    b2 = np.column_stack([_bounded(hazard * (1.0 - entropy) * (0.55 + 0.05 * i), span=max(24, w)) for i in range(8)])
    b3 = np.column_stack([_bounded(age_boost * p_target * (0.50 + 0.05 * i) + 0.15 * np.abs(edge), span=max(24, windows[i])) for i in range(8)])
    return b0, b1, b2, b3


def _crash_rebound_contrarian_switch(
    frame: pd.DataFrame,
    side: str,
    window: int,
    extreme_q: float,
    switch_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Track a target-direction crash, rebound fraction, and switch state.

    The signal is explicitly two-stage: a target-side extreme move creates a
    shock, and subsequent opposite movement determines whether continuation or
    contrarian response is currently more plausible.  This is not the single
    volatility gate used by older momentum-crash versions.
    """
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    ret = np.log(c / pc)
    desired = -1.0 if side == "down" else 1.0
    aligned = desired * ret
    cut = pd.Series(np.abs(ret)).shift(1).rolling(max(16, int(window)), min_periods=max(8, int(window) // 2)).quantile(float(extreme_q)).to_numpy()
    cut = np.maximum(np.nan_to_num(cut, nan=np.nanmedian(np.abs(ret)) + 1e-6), 1e-6)
    event = (aligned >= cut).astype(float)
    excess = np.maximum(aligned - cut, 0.0) * event
    age = np.zeros(len(ret), dtype=float)
    last_shock = np.zeros(len(ret), dtype=float)
    rebound_acc = np.zeros(len(ret), dtype=float)
    for i in range(len(ret)):
        if event[i] > 0:
            age[i] = 0.0
            last_shock[i] = excess[i]
            rebound_acc[i] = 0.0
        elif i > 0:
            age[i] = age[i - 1] + 1.0
            last_shock[i] = last_shock[i - 1]
            rebound_acc[i] = rebound_acc[i - 1] + max(-aligned[i], 0.0)
    memory = max(2.0, float(window) * 0.45)
    decay = np.exp(-age / memory)
    shock = _ewm(last_shock, memory)
    rebound = np.clip(rebound_acc / np.maximum(last_shock + shock, 1e-6), 0.0, 2.0)
    continuation = shock * decay * np.clip(1.0 - rebound, 0.0, 1.0)
    contrarian = shock * decay * np.clip(rebound - 0.15, 0.0, 1.5)
    current = _ewm(aligned, max(2.0, memory / 2.0))
    switch = float(switch_weight) * continuation - (1.0 - float(switch_weight)) * contrarian
    switch += 0.20 * current * (0.5 + 0.5 * decay)
    range_pct = np.maximum(_raw(frame, "true_range_pct"), 0.0)
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([_bounded(switch * (0.60 + 0.05 * i), span=max(24, int(window))) for i in range(8)])
    b1 = np.column_stack([_bounded(continuation * (0.60 + 0.05 * i) - contrarian, span=max(24, int(window))) for i in range(8)])
    b2 = np.column_stack([_bounded(rebound * decay * (0.50 + 0.05 * i) + desired * ret * 0.18, span=max(24, int(window))) for i in range(8)])
    b3 = np.column_stack([_bounded(np.abs(switch) * (0.55 + 0.04 * i) + range_pct * decay * 0.15, span=max(24, windows[i])) for i in range(8)])
    return b0, b1, b2, b3


def _anchor_barrier_distance(
    frame: pd.DataFrame,
    side: str,
    window: int,
    break_cut: float,
    anchor_mix: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal distance and reaction to rolling/expanding psychological anchors."""
    c = np.maximum(_raw(frame, "close"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    o = np.maximum(_raw(frame, "open"), 1e-8)
    if side == "up":
        rolling_anchor = pd.Series(c).rolling(int(window), min_periods=max(8, int(window) // 2)).max().shift(1).to_numpy()
        expanding_anchor = pd.Series(c).expanding(min_periods=2).max().shift(1).to_numpy()
        anchor = float(anchor_mix) * rolling_anchor + (1.0 - float(anchor_mix)) * expanding_anchor
        relative = (c - anchor) / np.maximum(anchor, 1e-8)
        close_reaction = (c - o) / np.maximum(o, 1e-8)
    else:
        rolling_anchor = pd.Series(c).rolling(int(window), min_periods=max(8, int(window) // 2)).min().shift(1).to_numpy()
        expanding_anchor = pd.Series(c).expanding(min_periods=2).min().shift(1).to_numpy()
        anchor = float(anchor_mix) * rolling_anchor + (1.0 - float(anchor_mix)) * expanding_anchor
        relative = (anchor - c) / np.maximum(anchor, 1e-8)
        close_reaction = (o - c) / np.maximum(o, 1e-8)
    relative = np.nan_to_num(relative, nan=0.0, posinf=0.0, neginf=0.0)
    approach = np.exp(-np.abs(relative) / max(float(break_cut), 1e-3))
    breakout = np.maximum(relative - float(break_cut), 0.0)
    rejection = np.maximum(-relative, 0.0) * np.maximum(close_reaction, 0.0)
    velocity = relative - np.r_[0.0, relative[:-1]]
    barrier_state = np.tanh(breakout * 8.0) + 0.35 * np.tanh(velocity * 8.0) - 0.25 * np.tanh(rejection * 8.0)
    anchor_rank = _rank(np.abs(relative), window=max(64, int(window) * 2))
    anchor_rank = np.nan_to_num(anchor_rank, nan=0.5)
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([_bounded(barrier_state * (0.65 + 0.05 * i), span=max(24, int(window))) for i in range(8)])
    b1 = np.column_stack([_bounded((breakout - rejection) * (0.60 + 0.05 * i) + 0.15 * velocity, span=max(24, windows[i])) for i in range(8)])
    b2 = np.column_stack([_bounded(approach * (0.55 + 0.04 * i) + barrier_state * 0.35, span=max(24, int(window))) for i in range(8)])
    b3 = np.column_stack([_bounded(anchor_rank * np.abs(barrier_state) * (0.50 + 0.05 * i) + np.abs(close_reaction) * 0.15, span=max(24, windows[i])) for i in range(8)])
    return b0, b1, b2, b3


def _variance_ratio_state(
    frame: pd.DataFrame,
    side: str,
    window: int,
    horizon: int,
    vr_cut: float,
    mapping_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lo–MacKinlay-style variance-ratio state with causal sign mapping."""
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    ret = np.log(c / pc)
    r = pd.Series(ret)
    qret = r.rolling(max(2, int(horizon)), min_periods=max(2, int(horizon))).sum()
    numerator = qret.rolling(int(window), min_periods=max(16, int(window) // 2)).var(ddof=0)
    denominator = float(max(2, int(horizon))) * r.rolling(int(window), min_periods=max(16, int(window) // 2)).var(ddof=0)
    vr = (numerator / np.maximum(denominator, 1e-10)).replace([np.inf, -np.inf], np.nan).fillna(1.0).to_numpy(float)
    vr_state = np.tanh((vr - 1.0) / max(float(vr_cut), 1e-3))
    desired = -1.0 if side == "down" else 1.0
    signed = desired * ret
    continuation = np.maximum(vr_state, 0.0)
    reversal = np.maximum(-vr_state, 0.0)
    mapped = signed * (1.0 + float(mapping_weight) * continuation) - signed * float(mapping_weight) * reversal
    disagreement = np.abs(np.diff(vr_state, prepend=vr_state[0]))
    risk = np.abs(ret) * (1.0 + np.abs(vr_state))
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([_bounded(mapped * (0.60 + 0.05 * i), span=max(24, int(window))) for i in range(8)])
    b1 = np.column_stack([_bounded(vr_state * signed * (0.55 + 0.05 * i), span=max(24, windows[i])) for i in range(8)])
    b2 = np.column_stack([_bounded(mapped * (1.0 - 0.25 * disagreement) + 0.12 * risk * (0.5 + 0.04 * i), span=max(24, int(window))) for i in range(8)])
    b3 = np.column_stack([_bounded(np.abs(vr_state) * (0.55 + 0.04 * i) + disagreement * 0.20, span=max(24, windows[i])) for i in range(8)])
    return b0, b1, b2, b3


def _support_break_retest_failure(
    frame: pd.DataFrame,
    side: str,
    window: int,
    break_cut: float,
    decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two-stage barrier break followed by retest and failure-to-recover."""
    c = np.maximum(_raw(frame, "close"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    v = np.maximum(_raw(frame, "volume"), 0.0)
    if side == "down":
        level = pd.Series(l).rolling(int(window), min_periods=max(8, int(window) // 2)).min().shift(1).to_numpy()
        desired = -1.0
    else:
        level = pd.Series(h).rolling(int(window), min_periods=max(8, int(window) // 2)).max().shift(1).to_numpy()
        desired = 1.0
    level = np.nan_to_num(level, nan=np.nanmedian(c))
    distance = desired * (c - level) / np.maximum(level, 1e-8)
    break_event = distance >= float(break_cut)
    age = np.zeros(len(c), dtype=float)
    held_level = np.full(len(c), np.nan, dtype=float)
    max_retest = np.zeros(len(c), dtype=float)
    failure = np.zeros(len(c), dtype=float)
    for i in range(len(c)):
        if break_event[i]:
            age[i] = 0.0
            held_level[i] = level[i]
            max_retest[i] = 0.0
            failure[i] = max(float(distance[i]), 0.0)
        elif i > 0 and np.isfinite(held_level[i - 1]):
            age[i] = age[i - 1] + 1.0
            held_level[i] = held_level[i - 1]
            retrace = max(-desired * (c[i] - held_level[i]) / max(abs(held_level[i]), 1e-8), 0.0)
            max_retest[i] = max(max_retest[i - 1], retrace)
            away = max(desired * (c[i] - held_level[i]) / max(abs(held_level[i]), 1e-8), 0.0)
            failure[i] = away * max_retest[i]
    memory = np.exp(-age / max(2.0, float(window) * max(float(decay), 0.1)))
    volume_rank = np.nan_to_num(_rank(np.log1p(v), window=252), nan=0.5)
    valid = np.isfinite(held_level).astype(float)
    state = (np.maximum(distance, 0.0) + 1.5 * failure) * memory * valid
    retest_state = max_retest * memory * valid
    rejection = desired * np.log(np.maximum(c, 1e-8) / np.maximum(np.r_[c[0], c[:-1]], 1e-8))
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([_bounded(state * (0.60 + 0.05 * i), span=max(24, int(window))) for i in range(8)])
    b1 = np.column_stack([_bounded(failure * (0.55 + 0.05 * i) + state * 0.35, span=max(24, windows[i])) for i in range(8)])
    b2 = np.column_stack([_bounded(retest_state * (0.55 + 0.04 * i) + rejection * 0.20, span=max(24, int(window))) for i in range(8)])
    b3 = np.column_stack([_bounded(state * (0.50 + 0.05 * i) * (0.65 + 0.35 * volume_rank), span=max(24, windows[i])) for i in range(8)])
    return b0, b1, b2, b3


def _tail_asymmetry_leverage(
    frame: pd.DataFrame,
    side: str,
    window: int,
    tail_q: float,
    leverage_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Time-varying tail-mass asymmetry with a return-volatility leverage state."""
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    ret = np.log(c / pc)
    shifted = pd.Series(ret).shift(1)
    w = max(32, int(window))
    lo = shifted.rolling(w, min_periods=max(16, w // 2)).quantile(float(tail_q)).to_numpy()
    hi = shifted.rolling(w, min_periods=max(16, w // 2)).quantile(1.0 - float(tail_q)).to_numpy()
    lo = np.nan_to_num(lo, nan=np.nanmedian(ret))
    hi = np.nan_to_num(hi, nan=np.nanmedian(ret))
    neg_mass = pd.Series((shifted.to_numpy(float) <= lo).astype(float)).rolling(w, min_periods=max(16, w // 2)).mean().fillna(0.0).to_numpy()
    pos_mass = pd.Series((shifted.to_numpy(float) >= hi).astype(float)).rolling(w, min_periods=max(16, w // 2)).mean().fillna(0.0).to_numpy()
    desired = -1.0 if side == "down" else 1.0
    target_mass = neg_mass if side == "down" else pos_mass
    opposite_mass = pos_mass if side == "down" else neg_mass
    tail_balance = target_mass - opposite_mass
    center = pd.Series(ret).rolling(w, min_periods=max(16, w // 2)).mean().fillna(0.0).to_numpy()
    scale = _rolling_std(ret, w)
    skew = pd.Series((ret - center) ** 3).rolling(w, min_periods=max(16, w // 2)).mean().fillna(0.0).to_numpy() / np.maximum(scale ** 3, 1e-8)
    leverage = desired * (-ret) * _rank(np.abs(ret), window=max(64, w * 2))
    leverage = np.nan_to_num(leverage, nan=0.0)
    signed = desired * ret
    raw = tail_balance + float(leverage_weight) * leverage + 0.20 * signed * target_mass
    asymmetry = tail_balance - 0.25 * desired * skew
    risk = np.abs(skew) + np.abs(leverage)
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([_bounded(raw * (0.60 + 0.05 * i), span=max(24, w)) for i in range(8)])
    b1 = np.column_stack([_bounded(asymmetry * (0.55 + 0.05 * i), span=max(24, windows[i])) for i in range(8)])
    b2 = np.column_stack([_bounded(tail_balance * (1.0 + 0.10 * i) + leverage * 0.30, span=max(24, w)) for i in range(8)])
    b3 = np.column_stack([_bounded(risk * (0.50 + 0.05 * i) + np.abs(signed) * target_mass * 0.15, span=max(24, windows[i])) for i in range(8)])
    return b0, b1, b2, b3


def _liquidity_conditioned_reversal(
    frame: pd.DataFrame,
    side: str,
    window: int,
    shock_q: float,
    volatility_weight: float,
    low_turnover_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive short-term reversal conditioned on volatility and turnover.

    The causal proxy follows the documented liquidity-provision pattern: high
    volatility produces a faster initial reversal, while low turnover lets a
    reversal persist.  It uses a short and a long response channel instead of
    a static volume gate, making the construction distinct from V156/V168.
    """
    o = np.maximum(_raw(frame, "open"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    amount = np.maximum(_raw(frame, "amount"), 0.0)
    volume = np.maximum(_raw(frame, "volume"), 0.0)
    gap = np.log(o / pc)
    intra = np.log(c / o)
    ret = np.log(c / pc)
    desired = -1.0 if side == "down" else 1.0
    reversal_shock = -desired * intra
    cutoff = pd.Series(np.abs(intra)).shift(1).rolling(max(16, int(window)), min_periods=max(8, int(window) // 2)).quantile(float(shock_q)).to_numpy()
    cutoff = np.maximum(np.nan_to_num(cutoff, nan=np.nanmedian(np.abs(intra)) + 1e-6), 1e-6)
    excess = np.maximum(np.abs(intra) - cutoff, 0.0) / cutoff
    event = (reversal_shock > 0).astype(float) * np.tanh(excess)
    log_amount = np.log1p(amount)
    turnover = np.nan_to_num(_rank(log_amount, window=252), nan=0.5)
    vol_state = np.nan_to_num(_rank(_rolling_std(intra, max(8, int(window))), window=252), nan=0.5)
    impact = np.abs(intra) / np.maximum(log_amount, 1e-4)
    shock = reversal_shock * (0.50 + excess) * event
    fast = _ewm(shock, max(2.0, float(window) / 3.0))
    slow = _ewm(shock, max(4.0, float(window) * 1.5))
    fast_weight = np.clip(float(volatility_weight) * vol_state, 0.0, 1.0)
    slow_weight = np.clip(float(low_turnover_weight) * (1.0 - turnover), 0.0, 1.0)
    adaptive = fast * fast_weight + slow * slow_weight + shock * (0.25 + 0.25 * (1.0 - fast_weight))
    confirmation = desired * gap * np.tanh(np.abs(intra) / cutoff)
    range_state = np.abs(ret) / np.maximum(_rolling_std(ret, max(8, int(window))), 1e-6)
    raw = adaptive + 0.16 * confirmation + 0.12 * desired * ret * np.tanh(range_state)
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([_bounded(adaptive * (0.60 + 0.05 * i), span=max(24, int(window))) for i in range(8)])
    b1 = np.column_stack([_bounded(fast * (0.55 + 0.05 * i) + slow * (0.35 + 0.03 * i), span=max(24, windows[i])) for i in range(8)])
    b2 = np.column_stack([_bounded(raw * (0.60 + 0.04 * i) + impact * 0.10 * fast_weight, span=max(24, int(window))) for i in range(8)])
    b3 = np.column_stack([_bounded((fast_weight + slow_weight) * np.abs(shock) * (0.50 + 0.05 * i) + np.abs(confirmation) * 0.15, span=max(24, windows[i])) for i in range(8)])
    return b0, b1, b2, b3


def _overnight_decline_cycle_hazard(
    frame: pd.DataFrame,
    side: str,
    window: int,
    cycle_cut: float,
    persistence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """State-machine proxy for overnight/intraday decline-reversal cycles."""
    o = np.maximum(_raw(frame, "open"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    gap = np.log(o / pc)
    intra = np.log(c / o)
    tr = np.maximum(_raw(frame, "true_range_pct"), 1e-6)
    desired = -1.0 if side == "down" else 1.0
    ng = gap / tr
    ni = intra / tr
    gap_event = np.abs(ng) >= float(cycle_cut)
    # The cycle channel distinguishes same-direction pressure from a failed
    # intraday reversal.  It is a sequence-state score, not a rolling gap mean.
    same_direction = np.sign(ng) == np.sign(ni)
    target_leg = desired * (gap + intra)
    reversal_leg = -desired * (gap * np.sign(intra + 1e-12))
    cycle_event = gap_event.astype(float) * same_direction.astype(float)
    phase = _ewm(cycle_event * target_leg, max(2.0, float(window) / 2.0))
    failure = _ewm(gap_event.astype(float) * np.maximum(-desired * intra, 0.0), max(2.0, float(window) / 3.0))
    carry = _ewm(cycle_event * desired * intra, max(3.0, float(window)))
    raw = float(persistence) * carry + (1.0 - float(persistence)) * failure + 0.18 * reversal_leg
    alternation = np.abs(np.diff(np.sign(ng - ni), prepend=0.0))
    pressure = np.abs(ng) + np.abs(ni)
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([_bounded(raw * (0.60 + 0.05 * i), span=max(24, int(window))) for i in range(8)])
    b1 = np.column_stack([_bounded(phase * (0.55 + 0.05 * i) + failure, span=max(24, windows[i])) for i in range(8)])
    b2 = np.column_stack([_bounded((failure - carry) * (0.60 + 0.04 * i) + desired * intra * 0.15, span=max(24, int(window))) for i in range(8)])
    b3 = np.column_stack([_bounded(alternation * (0.50 + 0.05 * i) + pressure * np.abs(raw) * 0.08, span=max(24, windows[i])) for i in range(8)])
    return b0, b1, b2, b3


def _square_root_impact_inventory(
    frame: pd.DataFrame,
    side: str,
    window: int,
    impact_cut: float,
    absorption_weight: float,
    memory: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Square-root volume-impact and inventory absorption proxy."""
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    v = np.maximum(_raw(frame, "volume"), 0.0)
    ret = np.log(c / pc)
    body = np.log(c / o)
    location = np.clip((c - l) / np.maximum(h - l, 1e-8), 0.0, 1.0)
    desired = -1.0 if side == "down" else 1.0
    flow = v * (0.70 * (2.0 * location - 1.0) + 0.30 * np.tanh(body / (np.abs(body).mean() + 1e-8)))
    impact = np.abs(ret) / np.maximum(np.sqrt(v + 1.0), 1.0)
    impact_rank = np.nan_to_num(_rank(impact, window=max(64, int(window) * 2)), nan=0.5)
    absorption = np.abs(flow) / np.maximum(np.abs(ret) * np.sqrt(v + 1.0), 1e-5)
    absorption_rank = np.nan_to_num(_rank(absorption, window=max(64, int(window) * 2)), nan=0.5)
    signed = desired * ret
    continuation = signed * impact_rank
    reversal = -signed * absorption_rank * float(absorption_weight)
    shock = np.maximum(impact_rank - float(impact_cut), 0.0)
    raw = reversal * (0.50 + shock) + continuation * (1.0 - absorption_rank) * 0.45
    delayed = _ewm(raw, max(2.0, float(memory)))
    range_state = np.log(np.maximum(h, l) / np.maximum(l, 1e-8))
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([_bounded(raw * (0.60 + 0.05 * i), span=max(24, int(window))) for i in range(8)])
    b1 = np.column_stack([_bounded(delayed * (0.55 + 0.05 * i) + reversal, span=max(24, windows[i])) for i in range(8)])
    b2 = np.column_stack([_bounded(reversal * (0.60 + 0.04 * i) + continuation * 0.25, span=max(24, int(window))) for i in range(8)])
    b3 = np.column_stack([_bounded(np.abs(raw) * (0.50 + 0.05 * i) + range_state * shock * 0.15, span=max(24, windows[i])) for i in range(8)])
    return b0, b1, b2, b3


def _intraday_pressure_state_switch(
    frame: pd.DataFrame,
    side: str,
    window: int,
    pressure_cut: float,
    stress_cut: float,
    memory: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Regime-conditioned intraday pressure mapping.

    In calm/moderate states the pressure channel is mapped to a side-specific
    momentum or reversal response; in stressed states it is attenuated and the
    opposite channel is allowed only when a close-location exhaustion pattern
    is present.  This is a pressure-state switch, not a generic return sign or
    volatility gate.
    """
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    intra = np.log(c / o)
    gap = np.log(o / pc)
    ret = np.log(c / pc)
    rng = np.maximum(np.log(h / l), 1e-6)
    location = np.clip((c - l) / np.maximum(h - l, 1e-8), 0.0, 1.0)
    pressure = (intra + 0.35 * gap) / rng
    vol = _rolling_std(intra, max(8, int(window)))
    vol_rank = np.nan_to_num(_rank(vol, window=252), nan=0.5)
    calm = np.clip((float(stress_cut) - vol_rank) / max(float(stress_cut), 1e-3), 0.0, 1.0)
    stressed = np.clip((vol_rank - float(stress_cut)) / max(1.0 - float(stress_cut), 1e-3), 0.0, 1.0)
    strong = np.tanh(np.maximum(np.abs(pressure) - float(pressure_cut), 0.0))
    desired = -1.0 if side == "down" else 1.0
    aligned = desired * pressure
    # Downside uses exhaustion/reversal of positive pressure; upside uses the
    # continuation channel for positive pressure.  Both are softened under
    # high stress and require a directional close-location confirmation.
    if side == "down":
        primary = -aligned * (0.60 + 0.40 * calm)
        exhaustion = np.maximum(2.0 * location - 1.0, 0.0) * np.maximum(pressure, 0.0)
    else:
        primary = aligned * (0.60 + 0.40 * calm)
        exhaustion = np.maximum(2.0 * location - 1.0, 0.0) * np.maximum(pressure, 0.0)
    stress_reversal = -aligned * stressed * np.maximum(1.0 - 2.0 * location, 0.0)
    memory_state = _ewm(primary * strong + 0.35 * exhaustion + 0.20 * stress_reversal, max(2.0, float(memory)))
    ret_confirm = desired * ret * (0.40 + 0.60 * calm)
    raw = memory_state + 0.18 * ret_confirm + 0.12 * exhaustion * (1.0 - stressed)
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([_bounded(raw * (0.60 + 0.05 * i), span=max(24, int(memory))) for i in range(8)])
    b1 = np.column_stack([_bounded(primary * strong * (0.55 + 0.05 * i) + exhaustion * 0.25, span=max(24, windows[i])) for i in range(8)])
    b2 = np.column_stack([_bounded(memory_state * (0.60 + 0.04 * i) + ret_confirm * 0.20, span=max(24, int(memory))) for i in range(8)])
    b3 = np.column_stack([_bounded((calm * strong + stressed * np.abs(stress_reversal)) * (0.50 + 0.05 * i) + np.abs(exhaustion) * 0.15, span=max(24, windows[i])) for i in range(8)])
    return b0, b1, b2, b3


def _causal_rank_consensus_tail(
    frame: pd.DataFrame,
    side: str,
    window: int,
    consensus_power: float,
    risk_cut: float,
    memory: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed forecast-combination consensus of four spot-only experts.

    Each expert is ranked only against its own trailing history.  The product
    and trimmed-minimum channels deliberately reward agreement rather than a
    single dominant feature, following forecast-combination intuition while
    keeping all weights and transformations pre-registered.
    """
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    amount = np.maximum(_raw(frame, "amount"), 0.0)
    gap = np.log(o / pc)
    intra = np.log(c / o)
    ret = np.log(c / pc)
    tr = np.maximum(np.log(h / l), 1e-6)
    loc = np.clip((c - l) / np.maximum(h - l, 1e-8), 0.0, 1.0)
    desired = -1.0 if side == "down" else 1.0
    reversal = _rank(-desired * intra, window=max(64, int(window)))
    momentum = _rank(desired * (0.60 * ret + 0.40 * gap), window=max(64, int(window)))
    risk = _rank(np.abs(intra) + 0.5 * tr + 0.20 * np.abs(ret), window=max(64, int(window)))
    flow = _rank(desired * (2.0 * loc - 1.0) + 0.20 * np.tanh(np.log1p(amount)), window=max(64, int(window)))
    experts = np.column_stack([
        np.nan_to_num(reversal, nan=0.5),
        np.nan_to_num(momentum, nan=0.5),
        np.nan_to_num(risk, nan=0.5),
        np.nan_to_num(flow, nan=0.5),
    ])
    # Use a risk-conditioned blend: high risk is only useful when the
    # direction experts agree, preventing a pure volatility detector.
    directional = np.column_stack([experts[:, 0], experts[:, 1], experts[:, 3]])
    trimmed = np.sort(directional, axis=1)[:, 1]
    geometric = np.prod(np.clip(directional, 1e-4, 1.0), axis=1) ** (1.0 / 3.0)
    agreement = np.clip(trimmed * geometric, 0.0, 1.0)
    risk_gate = np.clip((experts[:, 2] - float(risk_cut)) / max(1.0 - float(risk_cut), 1e-3), 0.0, 1.0)
    raw = np.power(np.clip(0.55 * agreement + 0.45 * agreement * risk_gate, 0.0, 1.0), max(0.25, float(consensus_power)))
    raw = _ewm(raw, max(2.0, float(memory)))
    disagreement = np.std(directional, axis=1)
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([_bounded(raw * (0.65 + 0.04 * i), span=max(24, int(memory))) for i in range(8)])
    b1 = np.column_stack([_bounded(agreement * (0.60 + 0.05 * i) - disagreement * 0.25, span=max(24, windows[i])) for i in range(8)])
    b2 = np.column_stack([_bounded(geometric * (0.55 + 0.05 * i) + risk_gate * 0.20, span=max(24, int(window))) for i in range(8)])
    b3 = np.column_stack([_bounded((1.0 - disagreement) * (0.50 + 0.05 * i) + np.abs(desired * ret) * 0.10, span=max(24, windows[i])) for i in range(8)])
    return b0, b1, b2, b3


def _online_expectile_tail_link(
    frame: pd.DataFrame,
    side: str,
    window: int,
    expectile: float,
    learning: float,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal online asymmetric-least-squares tail link.

    This is a small-sample CARE/expectile proxy rather than a full-sample
    regression.  At formation row ``i`` the parameter state can only consume
    the label at row ``i-2`` (the first row at which that O2O outcome is
    fully known).  Test outcomes are masked by ``PreparedResearch`` and thus
    automatically stop the update loop in the Test period.
    """
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    volume = np.maximum(_raw(frame, "volume"), 0.0)
    amount = np.maximum(_raw(frame, "amount"), 0.0)
    desired = -1.0 if side == "down" else 1.0
    ret = np.log(c / pc)
    gap = np.log(o / pc)
    intra = np.log(c / o)
    true_range = np.maximum(np.log(h / l), 1e-6)
    location = np.clip((c - l) / np.maximum(h - l, 1e-8), 0.0, 1.0)

    # Rank-normalized state variables keep the recursive fit numerically
    # stable without using a future or a full-sample standardizer.
    state = np.column_stack([
        2.0 * _rank(desired * ret, window=252) - 1.0,
        2.0 * _rank(desired * gap, window=252) - 1.0,
        2.0 * _rank(desired * intra, window=252) - 1.0,
        2.0 * _rank(true_range, window=252) - 1.0,
        2.0 * _rank(np.log1p(volume), window=252) - 1.0,
        2.0 * _rank(np.log1p(amount), window=252) - 1.0,
        2.0 * location - 1.0,
    ])
    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.column_stack([np.ones(len(state)), state])

    # A current-day OHLC return scale is an observable normalization only;
    # it is never computed from the future O2O label.
    scale_series = pd.Series(np.abs(ret)).rolling(
        max(16, int(window)), min_periods=max(8, int(window) // 2)
    ).median().shift(1)
    fallback = pd.Series(np.abs(ret)).expanding(min_periods=8).median().shift(1)
    scale_base = scale_series.combine_first(fallback).fillna(np.nanmedian(np.abs(ret)) + 1e-5)
    scale = np.maximum(scale_base.to_numpy(float), 1e-5)
    target = _raw(frame, "future_open_to_open_return_1d", np.nan)
    signed_target = desired * target

    n, p = len(x), x.shape[1]
    beta = np.zeros(p, float)
    beta[0] = 0.0
    pred = np.zeros(n, float)
    residual_state = np.zeros(n, float)
    residual_scale = np.ones(n, float)
    positive_excess = np.zeros(n, float)
    confidence = np.zeros(n, float)
    resid_ewm = 0.0
    abs_resid_ewm = 1.0
    excess_ewm = 0.0
    forgetting = float(np.exp(-1.0 / max(2.0, float(window))))
    # The asymmetric least-squares weight is the defining expectile feature.
    tau = float(np.clip(expectile, 0.55, 0.97))
    step = float(np.clip(learning, 0.005, 0.50))
    ridge_value = max(float(ridge), 1e-4)

    for i in range(n):
        # Score before incorporating any label whose exit is today.  This
        # makes the state a genuine t-day prediction.
        pred[i] = float(np.dot(x[i], beta))
        residual_state[i] = resid_ewm
        residual_scale[i] = max(abs_resid_ewm, 0.15)
        positive_excess[i] = excess_ewm
        confidence[i] = pred[i] / residual_scale[i]

        j = i - 2
        if j < 0 or not np.isfinite(signed_target[j]):
            continue
        y_scaled = float(np.clip(signed_target[j] / scale[j], -8.0, 8.0))
        error = float(np.clip(y_scaled - pred[j], -8.0, 8.0))
        asym_weight = tau if error >= 0.0 else (1.0 - tau)
        norm = ridge_value + float(np.dot(x[j], x[j]))
        # Exponentially forgetting stochastic ALS/RLS update.  No target
        # after row i-2 is visible here, including all Test rows.
        gain = step * asym_weight / norm
        beta = forgetting * beta + gain * error * x[j]
        resid_ewm = forgetting * resid_ewm + (1.0 - forgetting) * error
        abs_resid_ewm = forgetting * abs_resid_ewm + (1.0 - forgetting) * abs(error)
        excess_ewm = forgetting * excess_ewm + (1.0 - forgetting) * max(error, 0.0)

    # Four distinct channels: the predicted tail level, asymmetric forecast
    # innovation, exceedance pressure, and confidence/feature agreement.
    current_signed = desired * ret / scale
    directional_agreement = np.mean(state[:, :3], axis=1)
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([
        _bounded(pred * (0.70 + 0.05 * i) + 0.10 * current_signed, span=max(24, int(window) * 2))
        for i in range(8)
    ])
    b1 = np.column_stack([
        _bounded((pred - residual_state) * (0.60 + 0.04 * i) + current_signed * 0.12, span=max(24, windows[i]))
        for i in range(8)
    ])
    b2 = np.column_stack([
        _bounded(positive_excess * (0.65 + 0.05 * i) + np.maximum(pred, 0.0) * 0.25, span=max(24, int(window)))
        for i in range(8)
    ])
    b3 = np.column_stack([
        _bounded(confidence * (0.50 + 0.05 * i) + directional_agreement * 0.20, span=max(24, windows[i]))
        for i in range(8)
    ])
    return b0, b1, b2, b3


def _local_expectile_state(
    frame: pd.DataFrame,
    side: str,
    window: int,
    expectile: float,
    kernel_scale: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Locally weighted causal expectile state (lCARE-style proxy).

    At row ``i`` the local sample ends at ``i-2``.  A few asymmetric-least-
    squares iterations are enough for a robust low-dimensional local tail
    estimate and avoid fitting a high-dimensional model on the small sample.
    """
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    volume = np.maximum(_raw(frame, "volume"), 0.0)
    desired = -1.0 if side == "down" else 1.0
    ret = np.log(c / pc)
    gap = np.log(o / pc)
    intra = np.log(c / o)
    true_range = np.maximum(np.log(h / l), 1e-6)
    location = np.clip((c - l) / np.maximum(h - l, 1e-8), 0.0, 1.0)
    state = np.column_stack([
        2.0 * _rank(desired * ret, window=252) - 1.0,
        2.0 * _rank(desired * gap, window=252) - 1.0,
        2.0 * _rank(desired * intra, window=252) - 1.0,
        2.0 * _rank(true_range, window=252) - 1.0,
        2.0 * _rank(np.log1p(volume), window=252) - 1.0,
        2.0 * location - 1.0,
    ])
    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    target = _raw(frame, "future_open_to_open_return_1d", np.nan)
    signed_target = desired * target
    scale_series = pd.Series(np.abs(ret)).rolling(
        max(16, int(window)), min_periods=max(8, int(window) // 2)
    ).median().shift(1)
    fallback = pd.Series(np.abs(ret)).expanding(min_periods=8).median().shift(1)
    scale = np.maximum(
        scale_series.combine_first(fallback).fillna(np.nanmedian(np.abs(ret)) + 1e-5).to_numpy(float),
        1e-5,
    )
    y = np.clip(signed_target / scale, -8.0, 8.0)
    n = len(state)
    expectile = float(np.clip(expectile, 0.55, 0.97))
    kernel_scale = max(float(kernel_scale), 0.03)
    iterations = max(1, int(iterations))
    estimate = np.zeros(n, float)
    dispersion = np.ones(n, float)
    exceedance = np.zeros(n, float)
    local_count = np.zeros(n, float)

    for i in range(n):
        right = i - 2
        left = max(0, right - int(window))
        if right <= left:
            continue
        idx = np.arange(left, right, dtype=int)
        valid = np.isfinite(y[idx])
        idx = idx[valid]
        if len(idx) < max(12, int(window) // 4):
            continue
        d = np.sqrt(np.sum((state[idx] - state[i]) ** 2, axis=1) / max(state.shape[1], 1))
        recency = np.exp(-(right - 1 - idx) / max(2.0, float(window) * 0.70))
        weights = np.exp(-d / kernel_scale) * recency
        weights = np.maximum(weights, 1e-8)
        e = float(np.sum(weights * y[idx]) / np.sum(weights))
        for _ in range(iterations):
            asym = np.where(y[idx] >= e, expectile, 1.0 - expectile)
            ww = weights * asym
            e = float(np.sum(ww * y[idx]) / np.sum(ww))
        resid = y[idx] - e
        disp = float(np.sqrt(np.sum(weights * resid * resid) / np.sum(weights)))
        positive = np.maximum(resid, 0.0)
        estimate[i] = e
        dispersion[i] = max(disp, 0.05)
        exceedance[i] = float(np.sum(weights * positive) / np.sum(weights))
        local_count[i] = min(1.0, len(idx) / max(float(window), 1.0))

    current_signed = desired * ret / scale
    agreement = np.mean(state[:, :3], axis=1)
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([
        _bounded(estimate * (0.70 + 0.05 * i) + 0.10 * current_signed, span=max(24, int(window)))
        for i in range(8)
    ])
    b1 = np.column_stack([
        _bounded((estimate + exceedance) * (0.60 + 0.04 * i), span=max(24, windows[i]))
        for i in range(8)
    ])
    b2 = np.column_stack([
        _bounded(np.maximum(estimate, 0.0) / dispersion * (0.55 + 0.05 * i), span=max(24, int(window)))
        for i in range(8)
    ])
    b3 = np.column_stack([
        _bounded((estimate / dispersion + 0.20 * agreement) * (0.50 + 0.05 * i) * local_count, span=max(24, windows[i]))
        for i in range(8)
    ])
    return b0, b1, b2, b3


def _range_volume_location_tail(
    frame: pd.DataFrame,
    side: str,
    window: int,
    tail_cut: float,
    interaction_weight: float,
    memory: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal copula-style range/volume/close-location tail dependence.

    The construction is a conditional tail-coincidence state: it asks whether
    range, trading activity and close location jointly occupy the target-side
    tail, then compares that joint frequency with the product of the marginal
    frequencies.  It is not a single volume gate or a linear order-imbalance
    score and uses no cross-sectional data.
    """
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    volume = np.maximum(_raw(frame, "volume"), 0.0)
    amount = np.maximum(_raw(frame, "amount"), 0.0)
    range_rank = np.nan_to_num(_rank(np.log(h / l), window=252), nan=0.5)
    activity_rank = np.nan_to_num(_rank(0.55 * np.log1p(volume) + 0.45 * np.log1p(amount), window=252), nan=0.5)
    location = np.clip((c - l) / np.maximum(h - l, 1e-8), 0.0, 1.0)
    # The target-side location tail is high close-location for the down side
    # and low close-location for the up side, based on the bar-auction proxy.
    location_signal = location if side == "down" else 1.0 - location
    window = max(16, int(window))
    cut = float(np.clip(tail_cut, 0.55, 0.92))
    range_tail = np.maximum((1.0 - range_rank) if side == "down" else range_rank, 0.0)
    activity_tail = np.maximum((1.0 - activity_rank) if side == "down" else activity_rank, 0.0)
    location_tail = np.maximum(location_signal, 0.0)
    e_r = (range_tail >= cut).astype(float)
    e_a = (activity_tail >= cut).astype(float)
    e_l = (location_tail >= cut).astype(float)
    joint = e_r * e_a * e_l
    roll = lambda z, w: pd.Series(z).shift(1).rolling(
        max(8, int(w)), min_periods=max(8, int(w) // 2)
    ).mean().fillna(0.0).to_numpy(float)
    p_r, p_a, p_l = roll(e_r, window), roll(e_a, window), roll(e_l, window)
    p_joint = roll(joint, window)
    expected = np.maximum(p_r * p_a * p_l, 1e-5)
    dependence = np.clip(p_joint / expected, 0.0, 8.0)
    dependence = np.tanh(np.log1p(dependence))
    event_strength = np.power(np.clip(range_tail * activity_tail * location_tail, 0.0, 1.0), 1.0 + 0.25 * float(interaction_weight))
    signed_bar = np.abs(np.log(c / np.maximum(_raw(frame, "prev_close"), 1e-8)))
    signed_bar_series = pd.Series(signed_bar)
    signed_bar_scale = signed_bar_series.rolling(
        window, min_periods=max(8, window // 2)
    ).median()
    signed_bar_scale = signed_bar_scale.combine_first(
        signed_bar_series.expanding(min_periods=1).median()
    ).fillna(1e-6).to_numpy(float)
    event_strength *= np.tanh(signed_bar / (signed_bar_scale + 1e-6))
    smoothed = _ewm(event_strength * (0.55 + float(interaction_weight) * dependence), max(2.0, float(memory)))
    marginal_agreement = np.minimum(np.minimum(range_tail, activity_tail), location_tail)
    disagreement = np.std(np.column_stack([range_tail, activity_tail, location_tail]), axis=1)
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([
        _bounded(event_strength * (0.65 + 0.05 * i) + 0.18 * dependence, span=max(24, window))
        for i in range(8)
    ])
    b1 = np.column_stack([
        _bounded(smoothed * (0.60 + 0.05 * i) + event_strength * 0.20, span=max(24, windows[i]))
        for i in range(8)
    ])
    b2 = np.column_stack([
        _bounded(dependence * (0.55 + 0.05 * i) + marginal_agreement * 0.25, span=max(24, window))
        for i in range(8)
    ])
    b3 = np.column_stack([
        _bounded((marginal_agreement - 0.35 * disagreement) * (0.50 + 0.05 * i) + dependence * 0.15, span=max(24, windows[i]))
        for i in range(8)
    ])
    return b0, b1, b2, b3


def _caviar_spot_tail_link(
    frame: pd.DataFrame,
    side: str,
    window: int,
    quantile: float,
    learning: float,
    state_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CAViaR-style causal autoregressive quantile link for one side.

    The state is a one-sided conditional quantile of the signed O2O return.
    It is updated with a pinball-gradient recursion, so it is not a rolling
    empirical quantile or a static volatility gate.  At row ``i`` the update
    can consume only the label at ``i-2``; the prepared Test rows have their
    labels masked and therefore cannot alter the recursion after the freeze
    boundary.  Current OHLCV state enters only through a pre-registered
    leverage term, analogous to the asymmetric/symmetric CAViaR update forms.
    """
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    volume = np.maximum(_raw(frame, "volume"), 0.0)
    amount = np.maximum(_raw(frame, "amount"), 0.0)
    desired = -1.0 if side == "down" else 1.0
    ret = np.log(c / pc)
    gap = np.log(o / pc)
    intra = np.log(c / o)
    true_range = np.maximum(np.log(h / l), 1e-8)
    location = np.clip((c - l) / np.maximum(h - l, 1e-8), 0.0, 1.0)
    target_location = location if side == "down" else 1.0 - location

    w = max(16, int(window))
    quantile = float(np.clip(quantile, 0.60, 0.98))
    learning = float(np.clip(learning, 0.005, 0.35))
    state_weight = float(np.clip(state_weight, 0.0, 1.5))
    # Scale uses only the prior path.  This keeps the recursive quantile
    # comparable across calm and volatile periods without a full-sample fit.
    scale_series = pd.Series(np.abs(ret)).rolling(
        w, min_periods=max(8, w // 2)
    ).median().shift(1)
    fallback = pd.Series(np.abs(ret)).expanding(min_periods=8).median().shift(1)
    scale = scale_series.combine_first(fallback).fillna(np.nanmedian(np.abs(ret)) + 1e-5)
    scale = np.maximum(scale.to_numpy(float), 1e-5)
    target = desired * _raw(frame, "future_open_to_open_return_1d", np.nan)
    y = target / scale

    # CAViaR's observable innovation state: current signed pressure, range,
    # activity and close-location are all available before the next open.
    pressure = np.maximum(desired * ret / scale, 0.0)
    range_rank = np.nan_to_num(_rank(true_range, window=252), nan=0.5)
    activity_rank = np.nan_to_num(
        _rank(0.55 * np.log1p(volume) + 0.45 * np.log1p(amount), window=252),
        nan=0.5,
    )
    state = np.clip(
        0.42 * np.tanh(pressure)
        + 0.22 * range_rank
        + 0.18 * activity_rank
        + 0.18 * target_location,
        0.0,
        1.0,
    )
    state += 0.10 * np.tanh(np.abs(desired * gap / scale))
    state = np.clip(state, 0.0, 1.0)

    n = len(ret)
    q_history = np.zeros(n, float)
    q = 0.0
    exceedance = 0.0
    exceedance_history = np.zeros(n, float)
    innovation_history = np.zeros(n, float)
    decay = float(np.exp(-1.0 / max(2.0, float(w))))
    for i in range(n):
        # Symmetric/asymmetric CAViaR proxy: autoregressive quantile level
        # plus a current observable leverage term.  The score is formed before
        # the newly available label at i-2 is incorporated.
        q_history[i] = max(q, 0.0)
        exceedance_history[i] = exceedance
        innovation_history[i] = max(q_history[i] - (1.0 - quantile), 0.0)
        j = i - 2
        if j < 0 or not np.isfinite(y[j]):
            continue
        y_j = float(np.clip(y[j], -12.0, 12.0))
        q_j = q_history[j]
        # Pinball gradient: positive when the current quantile is too low and
        # negative when it is too high.  The step is scaled by the prior
        # realized range, never by the future O2O outcome.
        gradient = quantile - float(y_j <= q_j)
        step_scale = 0.35 + 0.65 * np.tanh(abs(ret[j]) / scale[j])
        q += learning * step_scale * gradient
        q += learning * 0.20 * state_weight * max(pressure[j] - q, 0.0)
        q = float(np.clip(q, 0.0, 12.0))
        hit = float(y_j > q_j)
        exceedance = decay * exceedance + (1.0 - decay) * hit

    # Four interpretable banks: the autoregressive quantile, leverage-adjusted
    # quantile, exceedance pressure, and a tail/innovation confidence channel.
    # Each bank is deliberately a different CAViaR observable, not a copied
    # static feature gate.
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    q_level = np.nan_to_num(q_history, nan=0.0)
    exceed = np.nan_to_num(exceedance_history, nan=0.0)
    innovation = np.nan_to_num(innovation_history, nan=0.0)
    b0 = np.column_stack([
        _bounded(q_level * (0.70 + 0.05 * i), span=max(24, w))
        for i in range(8)
    ])
    b1 = np.column_stack([
        _bounded((q_level + state_weight * state * (0.25 + 0.05 * i)) * (0.60 + 0.04 * i), span=max(24, windows[i]))
        for i in range(8)
    ])
    b2 = np.column_stack([
        _bounded(q_level * (0.55 + 0.05 * i) * (1.0 + 0.80 * exceed) + innovation * 0.25, span=max(24, w))
        for i in range(8)
    ])
    b3 = np.column_stack([
        _bounded((innovation + 0.40 * exceed + 0.25 * state * q_level) * (0.50 + 0.05 * i), span=max(24, windows[i]))
        for i in range(8)
    ])
    return b0, b1, b2, b3


def _bayesian_state_tail(
    frame: pd.DataFrame,
    side: str,
    window: int,
    event_cut: float,
    prior_strength: float,
    memory: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal Bayesian state-conditional tail posterior.

    A target-side state is defined from the joint rank of range, trading
    activity and close location.  Unlike V189's unsupervised tail dependence,
    this method updates a conjugate-style posterior of the signed O2O outcome
    conditional on that state, with recency discounting and prior shrinkage.
    At row ``i`` only observations through ``i-2`` enter the posterior.  The
    state itself is observable on the current spot bar, so this is a compact
    dynamic-GLM/Bayesian-forecasting proxy rather than a full-sample classifier.
    """
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    o = np.maximum(_raw(frame, "open"), 1e-8)
    volume = np.maximum(_raw(frame, "volume"), 0.0)
    amount = np.maximum(_raw(frame, "amount"), 0.0)
    desired = -1.0 if side == "down" else 1.0
    ret = np.log(c / pc)
    gap = np.log(o / pc)
    true_range = np.maximum(np.log(h / l), 1e-8)
    location = np.clip((c - l) / np.maximum(h - l, 1e-8), 0.0, 1.0)
    range_rank = np.nan_to_num(_rank(true_range, window=252), nan=0.5)
    activity_rank = np.nan_to_num(
        _rank(0.55 * np.log1p(volume) + 0.45 * np.log1p(amount), window=252),
        nan=0.5,
    )
    # Calm/high-close is the down-side liquidity state; stressed/low-close is
    # its up-side counterpart.  This mapping is side-specific and not a
    # generic signed-return gate.
    range_tail = 1.0 - range_rank if side == "down" else range_rank
    activity_tail = 1.0 - activity_rank if side == "down" else activity_rank
    location_tail = location if side == "down" else 1.0 - location
    event_strength = np.clip(
        0.42 * range_tail + 0.33 * activity_tail + 0.25 * location_tail,
        0.0,
        1.0,
    )
    event_flag = (
        (range_tail >= float(event_cut))
        & (activity_tail >= float(event_cut))
        & (location_tail >= float(event_cut))
    ).astype(float)

    scale = pd.Series(np.abs(ret)).rolling(
        max(16, int(window)), min_periods=max(8, int(window) // 2)
    ).median().shift(1)
    scale = scale.fillna(pd.Series(np.abs(ret)).expanding(min_periods=8).median().shift(1))
    scale = np.maximum(scale.fillna(np.nanmedian(np.abs(ret)) + 1e-5).to_numpy(float), 1e-5)
    signed = desired * _raw(frame, "future_open_to_open_return_1d", np.nan) / scale

    w = max(16, int(window))
    memory = max(3, int(memory))
    prior_strength = max(float(prior_strength), 0.25)
    n = len(ret)
    posterior_prob = np.full(n, 0.5, float)
    posterior_mean = np.zeros(n, float)
    effective_count = np.zeros(n, float)
    uncertainty = np.ones(n, float)
    for i in range(n):
        end = i - 1  # include j=i-2, the newest fully known O2O label
        left = max(0, end - w)
        if end <= left:
            continue
        idx = np.arange(left, end, dtype=int)
        valid = np.isfinite(signed[idx])
        idx = idx[valid]
        if len(idx) < max(12, w // 4):
            continue
        age = end - 1 - idx
        weights = np.exp(-age / float(memory)) * event_flag[idx]
        total = float(weights.sum())
        if total <= 1e-8:
            continue
        success = (signed[idx] > 0.0).astype(float)
        # Beta(1/2, 1/2)-style neutral prior, with a separate shrinkage axis;
        # the posterior is used only as a state-conditional direction signal.
        posterior_prob[i] = float(
            (np.dot(weights, success) + 0.5 * prior_strength)
            / (total + prior_strength)
        )
        clipped = np.clip(signed[idx], -8.0, 8.0)
        posterior_mean[i] = float(np.dot(weights, clipped) / (total + prior_strength))
        effective_count[i] = total
        uncertainty[i] = 1.0 / np.sqrt(total + prior_strength)

    conditional_tail = np.tanh(np.maximum(posterior_mean, 0.0) / 2.0)
    prob_edge = np.clip(2.0 * (posterior_prob - 0.5), -1.0, 1.0)
    posterior_state = np.clip(
        event_strength * (0.55 * np.maximum(prob_edge, 0.0) + 0.45 * conditional_tail),
        0.0,
        1.0,
    )
    confidence = np.tanh(effective_count / max(4.0, float(prior_strength) * 2.0))
    # Four posterior channels: direction probability, conditional tail mean,
    # state-weighted posterior, and a confidence-adjusted tail channel.
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([
        _bounded((posterior_prob - 0.5) * (0.70 + 0.05 * i) + 0.20 * event_strength, span=max(24, w))
        for i in range(8)
    ])
    b1 = np.column_stack([
        _bounded(conditional_tail * (0.60 + 0.05 * i) + 0.25 * event_strength * np.maximum(prob_edge, 0.0), span=max(24, windows[i]))
        for i in range(8)
    ])
    b2 = np.column_stack([
        _bounded(posterior_state * (0.55 + 0.05 * i), span=max(24, w))
        for i in range(8)
    ])
    b3 = np.column_stack([
        _bounded((posterior_state + 0.30 * confidence * conditional_tail) * (0.50 + 0.05 * i), span=max(24, windows[i]))
        for i in range(8)
    ])
    return b0, b1, b2, b3


def _online_boa_spot_experts(
    frame: pd.DataFrame,
    side: str,
    window: int,
    learning: float,
    share: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Delayed-feedback Bernstein/Hedge-style aggregation of spot experts.

    Four deliberately different, low-capacity spot experts issue a tail
    probability on each bar: signed pressure, overnight/intraday tug-of-war,
    range/activity state, and close-location liquidity absorption.  Expert
    weights are updated only when the O2O label at ``i-2`` is available, using
    a bounded squared loss and a fixed-share refresh.  This is structurally
    different from V186's static rank consensus: the mixture can move quickly
    toward the expert that is working in the current regime while retaining a
    nonzero weight on the alternatives.  Test labels are masked by the
    preparation layer, so no post-freeze feedback can enter the scores.
    """
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    volume = np.maximum(_raw(frame, "volume"), 0.0)
    amount = np.maximum(_raw(frame, "amount"), 0.0)
    desired = -1.0 if side == "down" else 1.0
    ret = np.log(c / pc)
    gap = np.log(o / pc)
    intra = np.log(c / o)
    true_range = np.maximum(np.log(h / l), 1e-8)
    location = np.clip((c - l) / np.maximum(h - l, 1e-8), 0.0, 1.0)
    activity = 0.55 * np.log1p(volume) + 0.45 * np.log1p(amount)
    w = max(16, int(window))
    learning = float(np.clip(learning, 0.01, 2.0))
    share = float(np.clip(share, 0.0, 0.35))
    temperature = max(float(temperature), 0.20)

    scale = pd.Series(np.abs(ret)).rolling(
        w, min_periods=max(8, w // 2)
    ).median().shift(1)
    scale = scale.fillna(pd.Series(np.abs(ret)).expanding(min_periods=8).median().shift(1))
    scale = np.maximum(scale.fillna(np.nanmedian(np.abs(ret)) + 1e-5).to_numpy(float), 1e-5)
    pressure = desired * ret / scale
    gap_state = desired * gap / scale
    intra_state = desired * intra / scale
    range_rank = np.nan_to_num(_rank(true_range, window=252), nan=0.5)
    activity_rank = np.nan_to_num(_rank(activity, window=252), nan=0.5)
    location_tail = location if side == "down" else 1.0 - location
    range_tail = 1.0 - range_rank if side == "down" else range_rank
    activity_tail = 1.0 - activity_rank if side == "down" else activity_rank

    # Expert 1: directional pressure with a bounded shock response.
    e_pressure = 0.5 + 0.5 * np.tanh(0.70 * pressure + 0.20 * np.tanh(gap_state))
    # Expert 2: opening/intraday tug-of-war, emphasizing a failed opening move.
    e_tug = 0.5 + 0.5 * np.tanh(0.55 * gap_state - 0.45 * intra_state)
    # Expert 3: target-side range/activity stress or calm state.
    e_state = np.clip(0.55 * range_tail + 0.30 * activity_tail + 0.15 * location_tail, 0.0, 1.0)
    # Expert 4: close-location flow/absorption with a target-side activity tilt.
    flow = 2.0 * location - 1.0
    e_flow = 0.5 + 0.5 * np.tanh(desired * (0.75 * flow + 0.25 * np.tanh(intra_state)) + 0.15 * (activity_tail - 0.5))
    experts = np.column_stack([
        np.clip(e_pressure, 0.0, 1.0),
        np.clip(e_tug, 0.0, 1.0),
        np.clip(e_state, 0.0, 1.0),
        np.clip(e_flow, 0.0, 1.0),
    ])
    experts = np.nan_to_num(experts, nan=0.5, posinf=1.0, neginf=0.0)

    target = desired * _raw(frame, "future_open_to_open_return_1d", np.nan) / scale
    target_prob = 0.5 + 0.5 * np.tanh(np.clip(target / temperature, -8.0, 8.0))
    n = len(experts)
    weights = np.full(4, 0.25, float)
    weight_history = np.zeros((n, 4), float)
    mixture = np.zeros(n, float)
    disagreement = np.zeros(n, float)
    leader = np.zeros(n, float)
    effective_weight = np.zeros(n, float)
    update_decay = float(np.exp(-1.0 / max(2.0, float(w))))
    # The BOA-style second-order correction is kept bounded for stability on
    # daily returns; ``learning`` and ``share`` remain pre-registered axes.
    for i in range(n):
        weight_history[i] = weights
        mixture[i] = float(weights @ experts[i])
        disagreement[i] = float(np.sqrt(np.maximum(weights @ ((experts[i] - mixture[i]) ** 2), 0.0)))
        leader[i] = float(np.max(experts[i]))
        effective_weight[i] = float(np.max(weights))
        j = i - 2
        if j < 0 or not np.isfinite(target_prob[j]):
            continue
        losses = (experts[j] - float(target_prob[j])) ** 2
        mean_loss = float(weights @ losses)
        excess = losses - mean_loss
        variance = float(weights @ (excess * excess))
        eta = learning / max(1.0 + 2.0 * np.sqrt(variance), 1e-6)
        update = np.exp(np.clip(-eta * (excess + 0.20 * losses), -6.0, 6.0))
        weights = weights * update
        weights = (1.0 - share) * weights + share * 0.25
        weights = np.nan_to_num(weights, nan=0.25, posinf=1.0, neginf=0.0)
        total = float(weights.sum())
        weights = weights / total if total > 1e-10 else np.full(4, 0.25, float)
        # A small, causal forgetting of the incumbent mixture keeps the update
        # responsive without allowing one early expert to lock the state.
        if update_decay < 1.0:
            weights = update_decay * weights + (1.0 - update_decay) * 0.25
            weights /= max(float(weights.sum()), 1e-10)

    agreement = np.clip(1.0 - 2.0 * disagreement, 0.0, 1.0)
    leader_bonus = np.clip(leader - mixture, 0.0, 1.0)
    confidence = np.clip(0.5 * agreement + 0.5 * effective_weight, 0.0, 1.0)
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    # Four aggregation channels: raw BOA mixture, agreement-gated mixture,
    # leader-aware mixture, and confidence/dispersion adjusted mixture.
    b0 = np.column_stack([
        _bounded(mixture * (0.70 + 0.05 * i), span=max(24, w))
        for i in range(8)
    ])
    b1 = np.column_stack([
        _bounded(mixture * (0.55 + 0.05 * i) * (0.65 + 0.35 * agreement), span=max(24, windows[i]))
        for i in range(8)
    ])
    b2 = np.column_stack([
        _bounded((mixture + leader_bonus * (0.18 + 0.04 * i)) * (0.60 + 0.04 * i), span=max(24, w))
        for i in range(8)
    ])
    b3 = np.column_stack([
        _bounded((mixture * confidence + 0.20 * leader_bonus + 0.10 * agreement) * (0.50 + 0.05 * i), span=max(24, windows[i]))
        for i in range(8)
    ])
    return b0, b1, b2, b3


def _hawkes_spot_jump_intensity(
    frame: pd.DataFrame,
    side: str,
    window: int,
    decay: float,
    threshold: float,
    mark_power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal marked self-exciting jump intensity from daily spot bars.

    This is a small-sample daily proxy for a marked Hawkes process.  The
    events are target-side standardized return/range shocks; their magnitudes
    excite a decaying target intensity, while all-side activity supplies a
    cross-excitation baseline.  It uses no future O2O labels, derivatives, or
    cross-sectional information and is deliberately a recurrence rather than
    a fitted high-dimensional point-process model.
    """
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    volume = np.maximum(_raw(frame, "volume"), 0.0)
    amount = np.maximum(_raw(frame, "amount"), 0.0)
    desired = -1.0 if side == "down" else 1.0
    ret = np.log(c / pc)
    gap = np.log(o / pc)
    intra = np.log(c / o)
    tr = np.maximum(np.log(h / l), 1e-8)
    activity = 0.55 * np.log1p(volume) + 0.45 * np.log1p(amount)
    w = max(16, int(window))
    decay = float(np.clip(decay, 0.01, 0.50))
    threshold = float(np.clip(threshold, 0.30, 2.50))
    mark_power = float(np.clip(mark_power, 0.50, 2.00))

    abs_scale = pd.Series(np.abs(ret)).rolling(w, min_periods=max(8, w // 2)).median().shift(1)
    abs_scale = abs_scale.fillna(pd.Series(np.abs(ret)).expanding(min_periods=8).median().shift(1))
    scale = np.maximum(abs_scale.fillna(np.nanmedian(np.abs(ret)) + 1e-5).to_numpy(float), 1e-5)
    signed = desired * ret / scale
    signed_gap = desired * gap / scale
    signed_intra = desired * intra / scale
    range_z = tr / np.maximum(_ewm(tr, max(4, w // 4)), 1e-5)
    activity_rank = np.nan_to_num(_rank(activity, 252), nan=0.5)
    location = np.clip((c - l) / np.maximum(h - l, 1e-8), 0.0, 1.0)
    target_location = location if side == "down" else 1.0 - location

    # A current bar is an observable event; its mark is bounded so a single
    # limit-like return cannot numerically dominate the recurrence.
    raw_mark = np.maximum(signed, 0.0) + 0.35 * np.maximum(range_z - 1.0, 0.0)
    event = (raw_mark >= threshold).astype(float)
    mark = event * np.power(np.clip(raw_mark, 0.0, 8.0), mark_power)
    abs_mark = np.maximum(np.abs(ret) / scale, 0.0) + 0.25 * np.maximum(range_z - 1.0, 0.0)

    n = len(ret)
    target_intensity = np.zeros(n, float)
    all_intensity = np.zeros(n, float)
    event_memory = np.zeros(n, float)
    prev_target = prev_all = 0.0
    rho = float(np.exp(-1.0 / max(2.0, float(w) * decay)))
    for i in range(n):
        prev_target *= rho
        prev_all *= rho
        prev_target += mark[i]
        prev_all += np.power(np.clip(abs_mark[i], 0.0, 8.0), max(0.5, mark_power * 0.75))
        target_intensity[i] = prev_target
        all_intensity[i] = prev_all
        event_memory[i] = event[i]

    target_share = target_intensity / np.maximum(all_intensity, 1e-8)
    burst = np.log1p(target_intensity) * (0.55 + 0.45 * target_share)
    cross = np.log1p(all_intensity) * (0.35 + 0.65 * target_share)
    current_impulse = np.clip(0.60 * signed + 0.20 * signed_gap - 0.15 * signed_intra, -8.0, 8.0)
    target_flow = np.clip(0.55 * target_location + 0.45 * activity_rank, 0.0, 1.0)

    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([
        _bounded(burst * (0.70 + 0.05 * i) + 0.10 * current_impulse, span=max(24, w))
        for i in range(8)
    ])
    b1 = np.column_stack([
        _bounded((burst + 0.35 * np.maximum(current_impulse, 0.0)) * (0.58 + 0.05 * i), span=max(24, windows[i]))
        for i in range(8)
    ])
    b2 = np.column_stack([
        _bounded((target_share - 0.5) * (0.85 + 0.08 * i) + 0.35 * event_memory, span=max(24, w))
        for i in range(8)
    ])
    b3 = np.column_stack([
        _bounded((cross + target_flow) * (0.45 + 0.05 * i), span=max(24, windows[i]))
        for i in range(8)
    ])
    return b0, b1, b2, b3


def _bayesian_online_spot_changepoint(
    frame: pd.DataFrame,
    side: str,
    max_run: int,
    hazard: float,
    shock_cut: float,
    memory: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal Bayesian run-length/change-point state from spot observations.

    This is a deliberately small online approximation to Bayesian online
    change-point detection: a finite run-length posterior is propagated with a
    constant hazard, and the predictive likelihood is a robust Gaussian
    innovation using only the current OHLCV-derived observation.  No future
    O2O label is needed.  The four banks expose change probability, post-change
    target direction, run age and shock-adjusted activity separately so that a
    later base combination cannot silently collapse to one threshold.
    """
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    volume = np.maximum(_raw(frame, "volume"), 0.0)
    amount = np.maximum(_raw(frame, "amount"), 0.0)
    desired = -1.0 if side == "down" else 1.0
    ret = np.log(c / pc)
    gap = np.log(o / pc)
    intra = np.log(c / o)
    tr = np.maximum(np.log(h / l), 1e-8)
    activity = 0.55 * np.log1p(volume) + 0.45 * np.log1p(amount)
    m = int(np.clip(max_run, 8, 96))
    hazard = float(np.clip(hazard, 0.005, 0.40))
    shock_cut = float(np.clip(shock_cut, 0.40, 3.00))
    memory = float(np.clip(memory, 2.0, 128.0))

    scale_series = pd.Series(np.abs(ret)).rolling(max(8, int(memory)), min_periods=8).median()
    scale = np.maximum(scale_series.shift(1).fillna(pd.Series(np.abs(ret)).expanding(min_periods=8).median().shift(1)).fillna(np.nanmedian(np.abs(ret)) + 1e-5).to_numpy(float), 1e-5)
    z = desired * ret / scale
    z_gap = desired * gap / scale
    z_intra = desired * intra / scale
    vol_z = tr / np.maximum(_ewm(tr, max(4, int(memory // 3))), 1e-5)
    act_rank = np.nan_to_num(_rank(activity, 252), nan=0.5)
    obs = np.clip(0.58 * z + 0.22 * z_gap + 0.20 * z_intra, -8.0, 8.0)

    n = len(obs)
    run_prob = np.zeros((n, m), float)
    run_prob[0, 0] = 1.0
    change = np.zeros(n, float)
    run_age = np.zeros(n, float)
    post_mean = np.zeros(n, float)
    post_scale = np.ones(n, float)
    mu = np.zeros(m, float)
    var = np.ones(m, float)
    for i in range(n):
        if i > 0:
            prev = run_prob[i - 1]
            pred_var = np.maximum(var, 0.12)
            innov = np.maximum(np.abs(obs[i] - mu), 0.0)
            likelihood = np.exp(-0.5 * np.minimum(innov * innov / pred_var, 24.0)) / np.sqrt(pred_var + 1e-8)
            likelihood = np.maximum(likelihood, 1e-12)
            growth = prev * (1.0 - hazard) * likelihood
            reset = float(np.sum(prev * hazard * likelihood))
            current = np.zeros(m, float)
            current[0] = reset
            current[1:] = growth[:-1]
            total = float(current.sum())
            run_prob[i] = current / total if total > 1e-20 else np.r_[1.0, np.zeros(m - 1)]
            # Update the small conjugate-like state summaries causally.  The
            # posterior arrays are intentionally not fed any future labels.
            new_mu = np.zeros(m, float)
            new_var = np.ones(m, float)
            new_mu[0] = obs[i]
            new_var[0] = max(0.25, abs(obs[i]) * 0.15 + 0.15)
            for r in range(1, m):
                weight = np.clip(run_prob[i, r], 0.0, 1.0)
                prior_mu = mu[r - 1]
                prior_var = max(var[r - 1], 0.12)
                gain = 1.0 / (2.0 + r / max(memory, 2.0))
                new_mu[r] = (1.0 - gain) * prior_mu + gain * obs[i]
                new_var[r] = max(0.10, (1.0 - gain) * prior_var + gain * (obs[i] - prior_mu) ** 2)
            mu, var = new_mu, new_var
        change[i] = float(run_prob[i, 0])
        run_age[i] = float(np.sum(run_prob[i] * np.arange(m, dtype=float)) / max(float(run_prob[i].sum()), 1e-12))
        post_mean[i] = float(run_prob[i] @ mu)
        post_scale[i] = float(np.sqrt(max(run_prob[i] @ var, 0.10)))

    surprise = np.clip(np.abs(obs - post_mean) / np.maximum(post_scale, 0.2), 0.0, 8.0)
    target_surprise = np.maximum(obs, 0.0)
    direction_state = 0.5 + 0.5 * np.tanh(post_mean / np.maximum(post_scale, 0.2))
    activity_state = np.clip(0.55 * act_rank + 0.45 * np.tanh(np.maximum(vol_z - 1.0, 0.0)), 0.0, 1.0)
    age_hazard = np.clip(1.0 - np.exp(-run_age / max(memory, 2.0)), 0.0, 1.0)
    target_change = np.clip(change * (0.65 + 0.35 * activity_state) + 0.20 * np.tanh(target_surprise), 0.0, 1.0)

    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([
        _bounded(target_change * (0.75 + 0.05 * i) + 0.10 * direction_state, span=max(24, int(memory)))
        for i in range(8)
    ])
    b1 = np.column_stack([
        _bounded((target_change + 0.35 * np.maximum(direction_state - 0.5, 0.0)) * (0.58 + 0.05 * i), span=max(24, windows[i]))
        for i in range(8)
    ])
    b2 = np.column_stack([
        _bounded((age_hazard + 0.35 * np.maximum(direction_state - 0.5, 0.0)) * (0.55 + 0.05 * i), span=max(24, int(memory)))
        for i in range(8)
    ])
    b3 = np.column_stack([
        _bounded((target_change * (0.55 + 0.45 * activity_state) + 0.18 * surprise) * (0.50 + 0.05 * i), span=max(24, windows[i]))
        for i in range(8)
    ])
    return b0, b1, b2, b3


def _kalman_spot_dynamic_tail(
    frame: pd.DataFrame,
    side: str,
    discount: float,
    prior_scale: float,
    observation_scale: float,
    feature_mix: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Delayed-feedback dynamic linear tail model for spot-only O2O.

    A five-coordinate state-space regression is updated with the O2O label
    only after the two-row availability gap.  The coefficient state follows a
    discounted random walk (a small dynamic linear model); prediction rows
    never consume their own or any future target.  This is intentionally
    lower-dimensional than a rolling forest or a free-form local regression.
    """
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    volume = np.maximum(_raw(frame, "volume"), 0.0)
    amount = np.maximum(_raw(frame, "amount"), 0.0)
    desired = -1.0 if side == "down" else 1.0
    ret = np.log(c / pc)
    gap = np.log(o / pc)
    intra = np.log(c / o)
    true_range = np.maximum(np.log(h / l), 1e-8)
    location = np.clip((c - l) / np.maximum(h - l, 1e-8), 0.0, 1.0)
    activity = 0.55 * np.log1p(volume) + 0.45 * np.log1p(amount)
    w = 32
    abs_scale = pd.Series(np.abs(ret)).rolling(w, min_periods=8).median().shift(1)
    abs_scale = abs_scale.fillna(pd.Series(np.abs(ret)).expanding(min_periods=8).median().shift(1))
    scale = np.maximum(abs_scale.fillna(np.nanmedian(np.abs(ret)) + 1e-5).to_numpy(float), 1e-5)
    activity_state = np.nan_to_num(_rank(activity, 252), nan=0.5)
    range_state = np.nan_to_num(_rank(true_range, 252), nan=0.5)
    f = np.column_stack([
        np.ones(len(ret), float),
        np.clip(desired * ret / scale, -8.0, 8.0),
        np.clip(desired * gap / scale, -8.0, 8.0),
        np.clip(desired * intra / scale, -8.0, 8.0),
        np.clip(0.50 * (desired * (2.0 * location - 1.0)) + feature_mix * (activity_state - range_state), -1.0, 1.0),
    ])
    target_raw = frame["future_open_to_open_return_1d"].astype(float).to_numpy() if "future_open_to_open_return_1d" in frame else np.full(len(frame), np.nan)
    target = desired * target_raw / scale
    d = float(np.clip(discount, 0.94, 0.999))
    prior_scale = float(np.clip(prior_scale, 0.05, 4.0))
    obs_scale = float(np.clip(observation_scale, 0.10, 4.0))
    mix = float(np.clip(feature_mix, 0.0, 1.0))

    n, p = len(f), f.shape[1]
    beta = np.zeros(p, float)
    covariance = np.eye(p, dtype=float) * prior_scale
    pred_mean = np.zeros(n, float)
    pred_sd = np.ones(n, float)
    posterior_fit = np.zeros(n, float)
    innovation_state = np.zeros(n, float)
    # The observation-scale state is causal and shrinks to a stable prior.
    obs_var = obs_scale * obs_scale
    for i in range(n):
        x = f[i]
        covariance = covariance / d
        mean_i = float(x @ beta)
        variance_i = float(obs_var + x @ covariance @ x)
        pred_mean[i] = mean_i
        pred_sd[i] = np.sqrt(max(variance_i, 1e-5))
        posterior_fit[i] = min(1.0, i / 80.0)
        j = i - 2
        if j < 0 or not np.isfinite(target[j]):
            continue
        xj = f[j]
        pred_j = float(xj @ beta)
        q = float(obs_var + xj @ covariance @ xj)
        if not np.isfinite(q) or q <= 1e-8:
            continue
        innovation = float(np.clip(target[j] - pred_j, -6.0, 6.0))
        gain = (covariance @ xj) / q
        beta = beta + gain * innovation
        covariance = covariance - np.outer(gain, xj @ covariance)
        covariance = 0.5 * (covariance + covariance.T)
        covariance.flat[:: p + 1] = np.maximum(covariance.flat[:: p + 1], 1e-6)
        obs_var = 0.97 * obs_var + 0.03 * np.clip(innovation * innovation, 0.05, 16.0)
        innovation_state[i] = innovation

    z = np.clip(pred_mean / np.maximum(pred_sd, 0.05), -8.0, 8.0)
    confidence = np.clip(1.0 / np.maximum(pred_sd, 0.15), 0.0, 4.0)
    positive_z = np.maximum(z, 0.0)
    stability = np.clip(posterior_fit * (0.65 + 0.35 * np.tanh(confidence)), 0.0, 1.0)
    responsive = np.clip(0.60 * positive_z + 0.25 * np.maximum(innovation_state, 0.0) + 0.15 * stability, 0.0, 8.0)
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([
        _bounded(z * (0.72 + 0.05 * i), span=24 + 8 * i)
        for i in range(8)
    ])
    b1 = np.column_stack([
        _bounded(positive_z * (0.55 + 0.06 * i) + 0.08 * confidence, span=max(24, windows[i]))
        for i in range(8)
    ])
    b2 = np.column_stack([
        _bounded(responsive * (0.50 + 0.05 * i), span=max(24, 32 + 8 * i))
        for i in range(8)
    ])
    b3 = np.column_stack([
        _bounded((z * stability + 0.20 * confidence) * (0.48 + 0.05 * i), span=max(24, windows[i]))
        for i in range(8)
    ])
    return b0, b1, b2, b3


def _causal_quantile_partition_tail(
    frame: pd.DataFrame,
    side: str,
    window: int,
    bins: int,
    recency: float,
    prior_strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal quantile-partition forest proxy for target tail events.

    Each day is assigned to fixed causal rank leaves of four spot states.  A
    rolling, recency-weighted Beta-smoothed event rate is estimated within
    single-feature and pair-feature leaves, then exposed as four banks.  The
    event cut is frozen from Development labels only; labels enter a leaf only
    after the two-row O2O availability delay.  This is a deterministic,
    low-capacity quantile-forest proxy, not a random forest with Test fitting.
    """
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    volume = np.maximum(_raw(frame, "volume"), 0.0)
    amount = np.maximum(_raw(frame, "amount"), 0.0)
    desired = -1.0 if side == "down" else 1.0
    ret = np.log(c / pc)
    gap = np.log(o / pc)
    intra = np.log(c / o)
    tr = np.maximum(np.log(h / l), 1e-8)
    activity = 0.55 * np.log1p(volume) + 0.45 * np.log1p(amount)
    loc = np.clip((c - l) / np.maximum(h - l, 1e-8), 0.0, 1.0)
    # All state coordinates are current-or-past causal ranks.
    states = np.column_stack([
        np.nan_to_num(_rank(desired * gap, 252), nan=0.5),
        np.nan_to_num(_rank(tr, 252), nan=0.5),
        np.nan_to_num(_rank(activity, 252), nan=0.5),
        np.nan_to_num(_rank(loc if side == "down" else 1.0 - loc, 252), nan=0.5),
    ])
    target = frame["future_open_to_open_return_1d"].astype(float).to_numpy() if "future_open_to_open_return_1d" in frame else np.full(len(frame), np.nan)
    dates = pd.to_datetime(frame["date"]).to_numpy() if "date" in frame else np.arange(len(frame))
    dev_mask = (dates < np.datetime64("2023-01-01")) & np.isfinite(target)
    if not np.any(dev_mask):
        dev_mask = np.isfinite(target)
    q = float(np.nanquantile(target[dev_mask], 0.10 if side == "down" else 0.90))
    event = (target <= q) if side == "down" else (target >= q)
    event = np.asarray(event, bool)
    base_event = float(np.mean(event[dev_mask])) if np.any(dev_mask) else 0.10
    base_event = float(np.clip(base_event, 0.02, 0.30))
    w = max(64, int(window))
    b = int(np.clip(bins, 2, 10))
    decay = max(float(recency), 0.05)
    prior_strength = float(np.clip(prior_strength, 0.25, 8.0))
    n = len(states)
    single = np.full(n, base_event, float)
    pair = np.full(n, base_event, float)
    magnitude = np.zeros(n, float)
    support = np.zeros(n, float)
    agreement = np.zeros(n, float)
    pair_defs = ((0, 1), (0, 2), (1, 2), (1, 3))
    for i in range(n):
        right = i - 2
        if right <= 0:
            continue
        left = max(0, right - w)
        idx = np.arange(left, right, dtype=int)
        valid = np.isfinite(target[idx])
        idx = idx[valid]
        if len(idx) < max(24, b * 3):
            continue
        xh = states[idx]
        bins_hist = np.minimum(b - 1, np.floor(np.clip(xh, 0.0, 0.999999) * b).astype(int))
        current_bin = np.minimum(b - 1, np.floor(np.clip(states[i], 0.0, 0.999999) * b).astype(int))
        age = (right - 1) - idx
        weights = np.exp(-age / max(4.0, float(w) * decay))
        y = event[idx].astype(float)
        signed_excess = np.maximum((desired * target[idx]) - max(0.0, desired * q), 0.0)
        probs_single = []
        probs_pair = []
        counts = []
        for j in range(4):
            mask = bins_hist[:, j] == current_bin[j]
            ww = weights * mask
            den = float(ww.sum())
            probs_single.append(float((ww @ y + prior_strength * base_event) / (den + prior_strength)))
            counts.append(den)
        for j0, j1 in pair_defs:
            mask = (bins_hist[:, j0] == current_bin[j0]) & (bins_hist[:, j1] == current_bin[j1])
            ww = weights * mask
            den = float(ww.sum())
            probs_pair.append(float((ww @ y + prior_strength * base_event) / (den + prior_strength)))
        single[i] = float(np.mean(probs_single))
        pair[i] = float(np.mean(probs_pair))
        p_all = np.asarray(probs_single + probs_pair, float)
        agreement[i] = float(1.0 - np.clip(np.std(p_all) * 3.0, 0.0, 1.0))
        support[i] = float(np.clip(np.mean(counts) / max(8.0, float(w) / 4.0), 0.0, 1.0))
        magnitude[i] = float((weights @ signed_excess) / (weights.sum() + 1e-8))

    confidence = np.clip(0.50 * support + 0.50 * agreement, 0.0, 1.0)
    # Convert probabilities to stable bounded scores around the frozen base
    # rate; this prevents a sparse leaf from becoming a degenerate 0/1 gate.
    logit_single = np.log(np.clip(single, 1e-4, 1 - 1e-4) / np.clip(1.0 - single, 1e-4, 1.0))
    logit_pair = np.log(np.clip(pair, 1e-4, 1 - 1e-4) / np.clip(1.0 - pair, 1e-4, 1.0))
    base_logit = np.log(base_event / max(1e-6, 1.0 - base_event))
    excess_single = logit_single - base_logit
    excess_pair = logit_pair - base_logit
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([_bounded(excess_single * (0.70 + 0.05 * i), span=max(24, w // 2)) for i in range(8)])
    b1 = np.column_stack([_bounded(excess_pair * (0.60 + 0.05 * i), span=max(24, windows[i])) for i in range(8)])
    b2 = np.column_stack([_bounded((excess_pair * confidence + 0.20 * np.maximum(magnitude, 0.0)) * (0.55 + 0.05 * i), span=max(24, w // 2)) for i in range(8)])
    b3 = np.column_stack([_bounded((0.55 * excess_single + 0.45 * excess_pair) * (0.50 + 0.05 * i) * (0.55 + 0.45 * confidence), span=max(24, windows[i])) for i in range(8)])
    return b0, b1, b2, b3


def _ordinal_open_transition_tail(
    frame: pd.DataFrame,
    side: str,
    window: int,
    bins: int,
    recency: float,
    prior_strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal ordinal open-return transition posterior for O2O extremes.

    The state is the causal rank of the preceding open-to-open move.  For a
    current state transition, a rolling weighted table estimates the target
    q10/q90 event rate, with a duration-conditioned backoff and Beta shrinkage.
    The target row is admitted only at i-2, matching the O2O availability gap.
    This is a small Markov/ordinal transition mechanism rather than a static
    momentum score or a generic feature ensemble.
    """
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    oo = o / np.roll(o, 1) - 1.0
    oo[0] = np.nan
    if "oo_ret_1_rank252" in frame:
        rank_1 = np.nan_to_num(_raw(frame, "oo_ret_1_rank252", 0.5), nan=0.5)
    else:
        rank_1 = np.nan_to_num(_rank(oo, 252), nan=0.5)
    if "oo_ret_5_rank252" in frame:
        rank_5 = np.nan_to_num(_raw(frame, "oo_ret_5_rank252", 0.5), nan=0.5)
    else:
        rank_5 = np.nan_to_num(_rank(pd.Series(oo).rolling(5, min_periods=2).sum().to_numpy(), 252), nan=0.5)
    # A slight long-memory component prevents a one-day state from becoming
    # a raw sign gate while retaining a compact ordinal state space.
    state_value = np.clip(0.70 * rank_1 + 0.30 * rank_5, 0.0, 1.0)
    wbins = int(np.clip(bins, 3, 8))
    state = np.minimum(wbins - 1, np.floor(np.clip(state_value, 0.0, 0.999999) * wbins).astype(int))
    previous = np.roll(state, 1)
    previous[0] = state[0]
    transition = previous * wbins + state
    # Duration is the current ordinal state's causal run length.
    duration = np.zeros(len(state), float)
    for i in range(1, len(state)):
        duration[i] = duration[i - 1] + 1.0 if state[i] == state[i - 1] else 0.0

    target = frame["future_open_to_open_return_1d"].astype(float).to_numpy() if "future_open_to_open_return_1d" in frame else np.full(len(frame), np.nan)
    dates = pd.to_datetime(frame["date"]).to_numpy() if "date" in frame else np.arange(len(frame))
    dev_mask = (dates < np.datetime64("2023-01-01")) & np.isfinite(target)
    if not np.any(dev_mask):
        dev_mask = np.isfinite(target)
    q = float(np.nanquantile(target[dev_mask], 0.10 if side == "down" else 0.90))
    event = (target <= q) if side == "down" else (target >= q)
    event = np.asarray(event, bool)
    base_rate = float(np.clip(np.mean(event[dev_mask]) if np.any(dev_mask) else 0.10, 0.02, 0.30))
    w = max(64, int(window))
    decay = max(float(recency), 0.05)
    prior_strength = float(np.clip(prior_strength, 0.25, 8.0))
    n = len(state)
    transition_prob = np.full(n, base_rate, float)
    state_prob = np.full(n, base_rate, float)
    duration_prob = np.full(n, base_rate, float)
    excess_mag = np.zeros(n, float)
    for i in range(n):
        right = i - 2
        if right <= 0:
            continue
        left = max(0, right - w)
        idx = np.arange(left, right, dtype=int)
        valid = np.isfinite(target[idx])
        idx = idx[valid]
        if len(idx) < max(24, wbins * 3):
            continue
        age = (right - 1) - idx
        weights = np.exp(-age / max(4.0, float(w) * decay))
        y = event[idx].astype(float)
        cur_state = state[i]
        cur_trans = transition[i]
        trans_mask = transition[idx] == cur_trans
        state_mask = state[idx] == cur_state
        # Same-state duration bucket backs off to the transition posterior.
        dur_bucket = min(5, int(duration[i] // max(1.0, wbins / 2.0)))
        hist_dur_bucket = np.minimum(5, (duration[idx] // max(1.0, wbins / 2.0)).astype(int))
        dur_mask = (hist_dur_bucket == dur_bucket) & state_mask
        def posterior(mask: np.ndarray) -> tuple[float, float]:
            ww = weights * mask
            den = float(ww.sum())
            return float((ww @ y + prior_strength * base_rate) / (den + prior_strength)), den
        p_trans, den_trans = posterior(trans_mask)
        p_state, den_state = posterior(state_mask)
        p_dur, den_dur = posterior(dur_mask)
        transition_prob[i] = p_trans
        state_prob[i] = p_state
        duration_prob[i] = p_dur
        signed_excess = np.maximum(((-1.0 if side == "down" else 1.0) * target[idx]) - max(0.0, (-1.0 if side == "down" else 1.0) * q), 0.0)
        excess_mag[i] = float((weights * trans_mask @ signed_excess) / (den_trans + 1e-8)) if den_trans > 0 else 0.0

    base_logit = np.log(base_rate / max(1e-6, 1.0 - base_rate))
    def logit_excess(p: np.ndarray) -> np.ndarray:
        pp = np.clip(p, 1e-4, 1.0 - 1e-4)
        return np.log(pp / (1.0 - pp)) - base_logit
    e_trans, e_state, e_dur = map(logit_excess, (transition_prob, state_prob, duration_prob))
    consensus = 0.50 * e_trans + 0.30 * e_state + 0.20 * e_dur
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([_bounded(e_trans * (0.70 + 0.05 * i), span=max(24, w // 2)) for i in range(8)])
    b1 = np.column_stack([_bounded((e_trans + 0.25 * e_state) * (0.58 + 0.05 * i), span=max(24, windows[i])) for i in range(8)])
    b2 = np.column_stack([_bounded((consensus + 0.20 * np.maximum(excess_mag, 0.0)) * (0.55 + 0.05 * i), span=max(24, w // 2)) for i in range(8)])
    b3 = np.column_stack([_bounded((0.60 * consensus + 0.40 * e_dur) * (0.50 + 0.05 * i), span=max(24, windows[i])) for i in range(8)])
    return b0, b1, b2, b3


def _online_bayesian_logistic_tail(
    frame: pd.DataFrame,
    side: str,
    discount: float,
    learning: float,
    ridge: float,
    feature_mix: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Delayed-feedback online Bayesian-logistic tail-event state.

    A five-coordinate, class-imbalanced logistic state is initialized at the
    Development tail prevalence and updated by a diagonal Laplace/Newton step
    only when the corresponding O2O event is two rows old.  Discounting and a
    ridge prior keep the model responsive without fitting a high-dimensional
    classifier.  All predictors are current-day pure-spot states.
    """
    o = np.maximum(_raw(frame, "open"), 1e-8)
    h = np.maximum(_raw(frame, "high"), 1e-8)
    l = np.maximum(_raw(frame, "low"), 1e-8)
    c = np.maximum(_raw(frame, "close"), 1e-8)
    pc = np.maximum(_raw(frame, "prev_close"), 1e-8)
    volume = np.maximum(_raw(frame, "volume"), 0.0)
    amount = np.maximum(_raw(frame, "amount"), 0.0)
    desired = -1.0 if side == "down" else 1.0
    ret = np.log(c / pc)
    gap = np.log(o / pc)
    intra = np.log(c / o)
    tr = np.maximum(np.log(h / l), 1e-8)
    loc = np.clip((c - l) / np.maximum(h - l, 1e-8), 0.0, 1.0)
    activity = 0.55 * np.log1p(volume) + 0.45 * np.log1p(amount)
    scale_series = pd.Series(np.abs(ret)).rolling(32, min_periods=8).median().shift(1)
    scale = np.maximum(scale_series.fillna(pd.Series(np.abs(ret)).expanding(min_periods=8).median().shift(1)).fillna(np.nanmedian(np.abs(ret)) + 1e-5).to_numpy(float), 1e-5)
    gap_z = np.clip(desired * gap / scale, -6.0, 6.0)
    intra_z = np.clip(desired * intra / scale, -6.0, 6.0)
    ret_z = np.clip(desired * ret / scale, -6.0, 6.0)
    range_rank = np.nan_to_num(_rank(tr, 252), nan=0.5)
    activity_rank = np.nan_to_num(_rank(activity, 252), nan=0.5)
    location_state = desired * (2.0 * loc - 1.0)
    features = np.column_stack([
        np.ones(len(ret), float),
        ret_z,
        gap_z,
        intra_z,
        np.clip(0.55 * (range_rank - 0.5) + 0.45 * feature_mix * (activity_rank - 0.5) + 0.20 * location_state, -2.0, 2.0),
    ])
    target = frame["future_open_to_open_return_1d"].astype(float).to_numpy() if "future_open_to_open_return_1d" in frame else np.full(len(frame), np.nan)
    dates = pd.to_datetime(frame["date"]).to_numpy() if "date" in frame else np.arange(len(frame))
    dev_mask = (dates < np.datetime64("2023-01-01")) & np.isfinite(target)
    if not np.any(dev_mask):
        dev_mask = np.isfinite(target)
    q = float(np.nanquantile(target[dev_mask], 0.10 if side == "down" else 0.90))
    labels = ((target <= q) if side == "down" else (target >= q)).astype(float)
    base_rate = float(np.clip(np.mean(labels[dev_mask]) if np.any(dev_mask) else 0.10, 0.02, 0.30))
    discount = float(np.clip(discount, 0.94, 0.999))
    learning = float(np.clip(learning, 0.05, 1.5))
    ridge = float(np.clip(ridge, 0.01, 2.0))
    feature_mix = float(np.clip(feature_mix, 0.0, 1.0))
    n, p = len(features), features.shape[1]
    beta = np.zeros(p, float)
    beta[0] = np.log(base_rate / max(1e-6, 1.0 - base_rate))
    precision = np.full(p, ridge, float)
    probability = np.full(n, base_rate, float)
    margin = np.zeros(n, float)
    confidence = np.zeros(n, float)
    update_count = 0.0
    for i in range(n):
        beta[1:] *= discount
        x = features[i]
        eta = float(np.clip(x @ beta, -8.0, 8.0))
        p_i = 1.0 / (1.0 + np.exp(-eta))
        probability[i] = p_i
        margin[i] = eta - beta[0]
        confidence[i] = 1.0 - np.exp(-update_count / max(16.0, 1.0 / max(1e-5, 1.0 - discount)))
        j = i - 2
        if j < 0 or not np.isfinite(labels[j]):
            continue
        xj = features[j]
        eta_j = float(np.clip(xj @ beta, -8.0, 8.0))
        p_j = 1.0 / (1.0 + np.exp(-eta_j))
        residual = float(labels[j] - p_j)
        # Diagonal observed information plus ridge prior is a stable Laplace
        # approximation for a tiny online logistic state.
        info = p_j * (1.0 - p_j) * (xj * xj) + ridge
        precision = discount * precision + learning * info
        beta = beta + learning * residual * xj / np.maximum(precision, 1e-5)
        beta = np.clip(beta, -8.0, 8.0)
        update_count += 1.0

    logit_excess = np.clip(margin, -8.0, 8.0)
    positive_margin = np.maximum(logit_excess, 0.0)
    event_conf = np.clip(0.55 * confidence + 0.45 * np.tanh(np.maximum(precision.mean(), 0.0)), 0.0, 1.0)
    impulse = np.clip(0.55 * positive_margin + 0.25 * np.maximum(gap_z, 0.0) + 0.20 * np.maximum(intra_z, 0.0), 0.0, 8.0)
    windows = (3, 5, 8, 13, 21, 34, 55, 89)
    b0 = np.column_stack([_bounded(logit_excess * (0.72 + 0.05 * i), span=32 + 8 * i) for i in range(8)])
    b1 = np.column_stack([_bounded((positive_margin + 0.20 * np.maximum(probability - base_rate, 0.0)) * (0.58 + 0.05 * i), span=max(24, windows[i])) for i in range(8)])
    b2 = np.column_stack([_bounded(impulse * (0.52 + 0.05 * i), span=max(24, 32 + 8 * i)) for i in range(8)])
    b3 = np.column_stack([_bounded((logit_excess * event_conf + 0.20 * np.maximum(probability - base_rate, 0.0)) * (0.50 + 0.05 * i), span=max(24, windows[i])) for i in range(8)])
    return b0, b1, b2, b3


def _banks(frame: pd.DataFrame, method: str, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    oo = _raw(frame, "oo_ret_1")
    ret = _raw(frame, "ret_1")
    gap = _raw(frame, "gap")
    intra = _raw(frame, "intraday_ret")
    tr = _raw(frame, "true_range_pct")
    amount = _raw(frame, "amount_z_20")
    signed_oo = -oo if side == "down" else oo
    signed_ret = -ret if side == "down" else ret
    risk = np.abs(oo) + 0.5 * np.abs(ret) + 0.5 * tr

    if method == "static_sparse_tail_logit":
        # V203 is a four-expert, Development-fitted tail classifier.  Each
        # bank is a different causal spot feature family; the eight columns
        # vary only the pre-registered L2 strength.  The model is not an
        # online update and therefore cannot consume Validation/Test labels.
        c_grid = (0.003, 0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0)
        feature_groups = (
            (
                "oo_ret_1_rank252", "oo_ret_5_rank252", "oo_vol_5_rank252",
                "oo_vol_20_rank252", "oo_down_share_20_rank252", "oo_up_share_20_rank252",
                "gap_rank252", "intraday_ret_rank252",
            ),
            (
                "downside_vol_20_rank252", "upside_vol_20_rank252", "drawdown_60_rank252",
                "jump_intensity_20_rank252", "risk_expansion_rank", "tail_uncertainty_rank",
                "compression_release_rank", "amihud_rank252",
            ),
            (
                "ret_1_rank252", "ret_3_rank252", "ret_5_rank252", "ret_20_rank252",
                "trend_efficiency_20_rank252", "momentum_curvature_rank252",
                "range_position_20_rank252", "range_position_60_rank252",
            ),
            (
                "close_location_rank252", "upper_shadow_share_rank252", "lower_shadow_share_rank252",
                "true_range_pct_rank252", "amount_z_20_rank252", "volume_ratio_20_rank252",
                "gap_intraday_divergence_rank", "state_transition_rank",
            ),
        )
        banks = [
            np.column_stack([
                _static_tail_logit(frame, side, tuple(columns), c_value)
                for c_value in c_grid
            ])
            for columns in feature_groups
        ]
        return tuple(banks)

    if method == "spline_additive_tail_logit":
        # V208 is a low-capacity generalized-additive alternative to the
        # linear V203 and interaction-heavy V205.  Quantile knots are fitted
        # on Development only; each bank remains a separate spot state family.
        spline_grid = (
            (3, 0.01), (3, 0.10), (4, 0.03), (4, 0.30),
            (5, 0.03), (5, 0.30), (6, 0.10), (6, 1.00),
        )
        feature_groups = (
            (
                "ret_1_rank252", "gap_rank252", "intraday_ret_rank252",
                "close_location_rank252", "trend_efficiency_20_rank252",
                "momentum_curvature_rank252",
            ),
            (
                "true_range_pct_rank252", "vol_5_rank252", "vol_20_rank252",
                "downside_vol_20_rank252", "jump_intensity_20_rank252",
                "risk_expansion_rank",
            ),
            (
                "oo_ret_1_rank252", "oo_ret_5_rank252", "oo_vol_5_rank252",
                "oo_vol_20_rank252", "oo_down_share_20_rank252", "oo_up_share_20_rank252",
            ),
            (
                "amount_z_20_rank252", "volume_ratio_20_rank252", "amihud_rank252",
                "upper_shadow_share_rank252", "lower_shadow_share_rank252",
                "gap_intraday_divergence_rank",
            ),
        )
        banks = [
            np.column_stack([
                _static_spline_tail_logit(frame, side, tuple(columns), knots, c_value)
                for knots, c_value in spline_grid
            ])
            for columns in feature_groups
        ]
        return tuple(banks)

    if method == "temporal_block_spline_ensemble":
        # V209 averages two pre-registered time-block experts instead of
        # refitting a single spline on all Development.  The late-block
        # weight is fixed before Validation/Test and provides a direct,
        # small-sample forecast-combination axis.
        block_grid = (
            (3, 0.01, 0.25), (3, 0.10, 0.50), (4, 0.03, 0.50), (4, 0.30, 0.75),
            (5, 0.03, 0.25), (5, 0.30, 0.75), (6, 0.10, 0.50), (6, 1.00, 0.75),
        )
        feature_groups = (
            (
                "ret_1_rank252", "gap_rank252", "intraday_ret_rank252",
                "close_location_rank252", "trend_efficiency_20_rank252",
                "momentum_curvature_rank252",
            ),
            (
                "true_range_pct_rank252", "vol_5_rank252", "vol_20_rank252",
                "downside_vol_20_rank252", "jump_intensity_20_rank252",
                "risk_expansion_rank",
            ),
            (
                "oo_ret_1_rank252", "oo_ret_5_rank252", "oo_vol_5_rank252",
                "oo_vol_20_rank252", "oo_down_share_20_rank252", "oo_up_share_20_rank252",
            ),
            (
                "amount_z_20_rank252", "volume_ratio_20_rank252", "amihud_rank252",
                "upper_shadow_share_rank252", "lower_shadow_share_rank252",
                "gap_intraday_divergence_rank",
            ),
        )
        banks = [
            np.column_stack([
                _static_block_spline_tail(frame, side, tuple(columns), knots, c_value, late_weight)
                for knots, c_value, late_weight in block_grid
            ])
            for columns in feature_groups
        ]
        return tuple(banks)

    if method == "shape_constrained_tail_boost":
        # V210 applies a pre-registered monotonic shape prior to a tiny
        # gradient tail learner.  It is deliberately not the unconstrained
        # V205 tree: no feature can reverse its registered direction, and no
        # tree interaction is promoted by a Test-derived search.
        model_grid = (
            (2, 0.02, 0.5), (2, 0.05, 1.0), (3, 0.02, 1.0), (3, 0.05, 2.0),
            (3, 0.10, 2.0), (4, 0.02, 4.0), (4, 0.05, 4.0), (5, 0.02, 8.0),
        )
        feature_groups = (
            (
                "ret_1_rank252", "ret_5_rank252", "ret_20_rank252",
                "trend_efficiency_20_rank252", "momentum_curvature_rank252",
                "range_position_20_rank252",
            ),
            (
                "true_range_pct_rank252", "vol_5_rank252", "vol_20_rank252",
                "downside_vol_20_rank252", "jump_intensity_20_rank252",
                "risk_expansion_rank",
            ),
            (
                "gap_rank252", "intraday_ret_rank252", "close_location_rank252",
                "upper_shadow_share_rank252", "lower_shadow_share_rank252",
                "gap_intraday_divergence_rank",
            ),
            (
                "amount_z_20_rank252", "volume_ratio_20_rank252", "amihud_rank252",
                "state_transition_rank", "compression_release_rank",
                "tail_uncertainty_rank",
            ),
        )
        banks = [
            np.column_stack([
                _static_monotone_tail_boost(frame, side, tuple(columns), leaf_nodes, learning_rate, l2_value)
                for leaf_nodes, learning_rate, l2_value in model_grid
            ])
            for columns in feature_groups
        ]
        return tuple(banks)

    if method == "conditional_evt_hurdle_tail":
        # V211 is a POT/GPD hurdle model: the first stage predicts threshold
        # crossing and the second stage supplies frozen excess-tail severity.
        evt_grid = (
            (0.75, 0.01, 0.25), (0.80, 0.03, 0.35), (0.85, 0.10, 0.45), (0.88, 0.30, 0.55),
            (0.90, 0.03, 0.70), (0.92, 0.10, 0.85), (0.95, 0.30, 1.00), (0.97, 1.00, 1.20),
        )
        feature_groups = (
            (
                "ret_1_rank252", "ret_5_rank252", "ret_20_rank252",
                "trend_efficiency_20_rank252", "momentum_curvature_rank252",
                "range_position_20_rank252",
            ),
            (
                "true_range_pct_rank252", "vol_5_rank252", "vol_20_rank252",
                "downside_vol_20_rank252", "jump_intensity_20_rank252",
                "risk_expansion_rank",
            ),
            (
                "gap_rank252", "intraday_ret_rank252", "close_location_rank252",
                "upper_shadow_share_rank252", "lower_shadow_share_rank252",
                "gap_intraday_divergence_rank",
            ),
            (
                "amount_z_20_rank252", "volume_ratio_20_rank252", "amihud_rank252",
                "state_transition_rank", "compression_release_rank",
                "tail_uncertainty_rank",
            ),
        )
        banks = [
            np.column_stack([
                _static_evt_hurdle_tail(frame, side, tuple(columns), tail_quantile, c_value, severity_weight)
                for tail_quantile, c_value, severity_weight in evt_grid
            ])
            for columns in feature_groups
        ]
        return tuple(banks)

    if method == "static_semiparametric_tail_quantile":
        # V204 uses conditional quantile regression rather than a binary
        # event classifier.  Quantile levels are the only model axis; the
        # four banks deliberately use different spot-state families.
        q_grid = (
            (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25)
            if side == "down" else
            (0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95)
        )
        alpha_grid = (0.001, 0.001, 0.003, 0.003, 0.01, 0.01, 0.03, 0.03)
        feature_groups = (
            (
                "ret_1_rank252", "gap_rank252", "intraday_ret_rank252", "true_range_pct_rank252",
                "close_location_rank252", "amount_z_20_rank252", "drawdown_60_rank252",
                "oo_ret_5_rank252", "oo_vol_20_rank252", "risk_expansion_rank",
            ),
            (
                "ret_1_rank252", "gap_rank252", "intraday_ret_rank252", "true_range_pct_rank252",
                "close_location_rank252", "amount_z_20_rank252", "ret_20_rank252", "vol_20_rank252",
            ),
            (
                "oo_ret_1_rank252", "oo_ret_5_rank252", "oo_vol_5_rank252", "oo_vol_20_rank252",
                "oo_down_share_20_rank252", "oo_up_share_20_rank252", "gap_rank252", "intraday_ret_rank252",
            ),
            (
                "downside_vol_20_rank252", "upside_vol_20_rank252", "drawdown_60_rank252",
                "jump_intensity_20_rank252", "tail_uncertainty_rank", "compression_release_rank",
                "gap_intraday_divergence_rank", "state_transition_rank",
            ),
        )
        banks = [
            np.column_stack([
                _bounded(
                    _static_tail_quantile(frame, side, tuple(columns), q_value, alpha),
                    span=128,
                )
                for q_value, alpha in zip(q_grid, alpha_grid)
            ])
            for columns in feature_groups
        ]
        return tuple(banks)

    if method == "low_capacity_gradient_tail":
        # V205 follows the tail-risk ML literature with a deliberately tiny
        # gradient model.  The eight axes are fixed leaf/learning-rate pairs;
        # no depth, seed, feature subset or Test-derived early stopping is
        # searched.  Four banks keep overnight, tail, trend and candle-flow
        # information separate before the common convex score surface.
        model_grid = (
            (2, 0.02), (2, 0.05), (3, 0.02), (3, 0.05),
            (3, 0.10), (5, 0.02), (5, 0.05), (7, 0.02),
        )
        feature_groups = (
            (
                "ret_1_rank252", "gap_rank252", "intraday_ret_rank252", "true_range_pct_rank252",
                "close_location_rank252", "amount_z_20_rank252", "drawdown_60_rank252",
                "oo_ret_5_rank252", "oo_vol_20_rank252", "risk_expansion_rank",
            ),
            (
                "downside_vol_20_rank252", "upside_vol_20_rank252", "drawdown_60_rank252",
                "jump_intensity_20_rank252", "tail_uncertainty_rank", "compression_release_rank",
                "amihud_rank252", "state_transition_rank",
            ),
            (
                "ret_1_rank252", "ret_3_rank252", "ret_5_rank252", "ret_20_rank252",
                "trend_efficiency_20_rank252", "momentum_curvature_rank252",
                "range_position_20_rank252", "range_position_60_rank252",
            ),
            (
                "oo_ret_1_rank252", "oo_ret_5_rank252", "oo_vol_5_rank252", "oo_vol_20_rank252",
                "oo_down_share_20_rank252", "oo_up_share_20_rank252", "gap_intraday_divergence_rank",
                "volume_ratio_20_rank252",
            ),
        )
        banks = [
            np.column_stack([
                _static_tail_boost(frame, side, tuple(columns), leaf_nodes, learning_rate)
                for leaf_nodes, learning_rate in model_grid
            ])
            for columns in feature_groups
        ]
        return tuple(banks)

    if method == "statistical_jump_regime_tail":
        # V206 adapts the statistical-jump-model idea: persistent unsupervised
        # regimes are learned on Development and then mapped to a supervised
        # tail event rate.  The eight columns vary only the state count and
        # jump penalty; each of the four banks uses a distinct spot-state
        # family.  No HMM smoothing or later-label update is allowed.
        jump_grid = (
            (2, 0.10), (2, 0.25), (2, 0.50), (2, 1.00),
            (3, 0.10), (3, 0.25), (3, 0.50), (3, 1.00),
        )
        feature_groups = (
            (
                "ret_1_rank252", "ret_5_rank252", "ret_20_rank252",
                "trend_efficiency_20_rank252", "momentum_curvature_rank252",
                "range_position_20_rank252",
            ),
            (
                "true_range_pct_rank252", "vol_5_rank252", "vol_20_rank252",
                "downside_vol_20_rank252", "jump_intensity_20_rank252",
                "risk_expansion_rank",
            ),
            (
                "gap_rank252", "intraday_ret_rank252", "close_location_rank252",
                "upper_shadow_share_rank252", "lower_shadow_share_rank252",
                "gap_intraday_divergence_rank",
            ),
            (
                "amount_z_20_rank252", "volume_ratio_20_rank252", "amihud_rank252",
                "state_transition_rank", "compression_release_rank",
                "tail_uncertainty_rank",
            ),
        )
        banks = [
            np.column_stack([
                _static_jump_regime_tail(frame, side, tuple(columns), n_states, penalty)
                for n_states, penalty in jump_grid
            ])
            for columns in feature_groups
        ]
        return tuple(banks)

    if method == "jump_regime_boosted_tail":
        # V207 is the supervised second stage from the jump-model papers:
        # latent persistent states are generated on Development, then a
        # shallow gradient classifier forecasts the state from current spot
        # features.  The classifier never sees Validation/Test outcomes.
        model_grid = (
            (2, 2, 0.02), (2, 2, 0.05), (2, 3, 0.02), (2, 3, 0.05),
            (3, 2, 0.02), (3, 2, 0.05), (3, 3, 0.02), (3, 3, 0.05),
        )
        feature_groups = (
            (
                "ret_1_rank252", "ret_5_rank252", "ret_20_rank252",
                "trend_efficiency_20_rank252", "momentum_curvature_rank252",
                "range_position_20_rank252",
            ),
            (
                "true_range_pct_rank252", "vol_5_rank252", "vol_20_rank252",
                "downside_vol_20_rank252", "jump_intensity_20_rank252",
                "risk_expansion_rank",
            ),
            (
                "gap_rank252", "intraday_ret_rank252", "close_location_rank252",
                "upper_shadow_share_rank252", "lower_shadow_share_rank252",
                "gap_intraday_divergence_rank",
            ),
            (
                "amount_z_20_rank252", "volume_ratio_20_rank252", "amihud_rank252",
                "state_transition_rank", "compression_release_rank",
                "tail_uncertainty_rank",
            ),
        )
        banks = [
            np.column_stack([
                _static_jump_regime_boost(frame, side, tuple(columns), n_states, leaf_nodes, learning_rate)
                for n_states, leaf_nodes, learning_rate in model_grid
            ])
            for columns in feature_groups
        ]
        return tuple(banks)

    if method == "bayesian_es_range":
        b0 = np.column_stack([_direction(signed_oo, "up", h) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        b1 = np.column_stack([_bounded(_rolling_mean(np.maximum(signed_oo, 0), h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        b2 = np.column_stack([_bounded(_rolling_quantile(np.maximum(signed_oo, 0), h, q), span=2 * h) for h, q in zip((10, 20, 30, 40, 60, 90, 120, 180), (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.9))])
        b3 = np.column_stack([_bounded(_rolling_mean(np.maximum(signed_oo, 0) * (1 + tr), h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        return b0, b1, b2, b3

    if method == "loss_based_bayesian_seq_var":
        loss = np.maximum(-signed_oo, 0)
        posterior = []
        for h in (2, 4, 8, 16, 32, 64, 128, 256):
            alpha = _ewm((signed_oo > 0).astype(float), h)
            beta = _ewm((signed_oo <= 0).astype(float), h)
            posterior.append(alpha / (alpha + beta + 1e-8))
        b0 = np.column_stack(posterior)
        b1 = np.column_stack([_bounded(_ewm(loss, h), span=4 * h) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        b2 = np.column_stack([_direction(_ewm(signed_ret * np.maximum(np.abs(signed_ret), 1e-8), h), "up", 4 * h) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        b3 = np.column_stack([_bounded(_ewm(np.abs(gap) + tr, h), span=4 * h) for h in (4, 8, 16, 32, 64, 96, 128, 192)])
        return b0, b1, b2, b3

    if method == "quantile_spectral_overnight_intraday":
        b0 = np.column_stack([_direction(_ewm(signed_oo, h), "up", 4 * h) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        b1 = np.column_stack([_bounded(_spectral_cross(gap, intra, w, p), span=2 * w) for w, p in zip((32, 48, 64, 96, 128, 160, 192, 256), (4, 6, 8, 10, 12, 16, 20, 24))])
        b2 = np.column_stack([_bounded(_rolling_mean(signed_oo * np.sign(gap * intra), h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        b3 = np.column_stack([_bounded(_rolling_std(gap - intra, h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        return b0, b1, b2, b3

    if method == "nonexchangeable_conformal_tail":
        b0 = np.column_stack([_conformal_tail(oo, w, side) for w in (32, 48, 64, 96, 128, 160, 192, 256)])
        b1 = np.column_stack([_bounded(_rolling_quantile(np.maximum(signed_oo, 0), w, q), span=2 * w) for w, q in zip((32, 48, 64, 96, 128, 160, 192, 256), (0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88))])
        b2 = np.column_stack([_direction(_ewm(signed_ret, h), "up", 4 * h) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        b3 = np.column_stack([_bounded(_rolling_std(risk, h), span=2 * h) for h in (10, 20, 40, 80, 120, 160, 200, 256)])
        return b0, b1, b2, b3

    if method == "midas_lasso_tail":
        b0 = np.column_stack([_direction(_midas(signed_oo, w, c), "up", 4 * w) for w, c in zip((8, 16, 24, 32, 48, 64, 96, 128), (0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4))])
        b1 = np.column_stack([_bounded(_midas(np.abs(oo), w, c), span=4 * w) for w, c in zip((8, 16, 24, 32, 48, 64, 96, 128), (0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4))])
        b2 = np.column_stack([_direction(_midas(signed_ret, w, c), "up", 4 * w) for w, c in zip((8, 16, 24, 32, 48, 64, 96, 128), (4, 3, 2, 1.5, 1, 0.7, 0.5, 0.3))])
        b3 = np.column_stack([_bounded(_midas(np.abs(gap) + tr + 0.1 * np.abs(amount), w, c), span=4 * w) for w, c in zip((8, 16, 24, 32, 48, 64, 96, 128), (0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4))])
        return b0, b1, b2, b3

    if method == "wavelet_multiresolution_shock":
        b0 = np.column_stack([_direction(_haar_detail(signed_oo, s, l), "up", 4 * l) for s, l in ((2, 8), (3, 12), (4, 16), (5, 20), (8, 32), (12, 48), (16, 64), (24, 96))])
        b1 = np.column_stack([_bounded(np.abs(_haar_detail(ret, s, l)), span=4 * l) for s, l in ((2, 8), (3, 12), (4, 16), (5, 20), (8, 32), (12, 48), (16, 64), (24, 96))])
        b2 = np.column_stack([_direction(_haar_detail(signed_ret, s, l), "up", 4 * l) for s, l in ((2, 8), (3, 12), (4, 16), (5, 20), (8, 32), (12, 48), (16, 64), (24, 96))])
        b3 = np.column_stack([_bounded(_rolling_mean(np.abs(_haar_detail(ret, s, l)) + tr, l), span=4 * l) for s, l in ((2, 8), (3, 12), (4, 16), (5, 20), (8, 32), (12, 48), (16, 64), (24, 96))])
        return b0, b1, b2, b3

    if method == "knn_analog_tail":
        b0 = np.column_stack([_knn_analog(frame, side, w, k, sc) for w, k, sc in ((128, 8, 0.10), (192, 8, 0.20), (256, 12, 0.10), (384, 12, 0.20), (512, 16, 0.10), (768, 16, 0.20), (1024, 24, 0.10), (1536, 24, 0.20))])
        b1 = np.column_stack([_direction(_ewm(signed_oo, h), "up", 4 * h) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        b2 = np.column_stack([_bounded(_rolling_std(oo, h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        b3 = np.column_stack([_bounded(_rolling_mean(np.abs(gap) + tr, h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        return b0, b1, b2, b3

    if method == "local_ridge_functional_forecast":
        b0 = np.column_stack([_local_ridge(frame, side, w, r) for w, r in ((64, 0.1), (96, 0.2), (128, 0.5), (192, 1.0), (256, 2.0), (384, 4.0), (512, 8.0), (768, 16.0))])
        b1 = np.column_stack([_direction(_ewm(signed_oo, h), "up", 4 * h) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        b2 = np.column_stack([_bounded(_rolling_std(oo, h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        b3 = np.column_stack([_bounded(_rolling_mean(np.abs(gap) + tr + 0.2 * np.abs(amount), h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        return b0, b1, b2, b3

    if method == "res_caviar_overnight":
        b0 = np.column_stack([
            _bounded(_res_caviar_path(signed_ret, signed_oo, tr, a, b, lev), span=4 * h)
            for h, a, b, lev in (
                (8, 0.08, 0.78, 0.10), (12, 0.10, 0.80, 0.15),
                (16, 0.12, 0.82, 0.20), (24, 0.14, 0.84, 0.25),
                (32, 0.16, 0.86, 0.30), (48, 0.18, 0.88, 0.35),
                (64, 0.20, 0.90, 0.40), (96, 0.22, 0.92, 0.45),
            )
        ])
        b1 = np.column_stack([_direction(_ewm(signed_oo, h), "up", 4 * h) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        b2 = np.column_stack([_bounded(_rolling_mean(np.maximum(signed_ret, 0.0) + np.abs(gap), h), span=3 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        b3 = np.column_stack([_bounded(_rolling_mean(np.maximum(signed_ret, 0.0) * (1.0 + tr), h), span=3 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        return b0, b1, b2, b3

    if method == "realized_sv_skewt":
        b0 = np.column_stack([
            _bounded(_realized_sv_skewt_score(ret, tr, signed_oo, vs, ss, shape, side), span=3 * vs)
            for vs, ss, shape in (
                (8, 8, 0.10), (12, 8, 0.20), (16, 12, 0.30), (24, 16, 0.40),
                (32, 20, 0.50), (48, 24, 0.65), (64, 32, 0.80), (96, 48, 1.00),
            )
        ])
        b1 = np.column_stack([_direction(_ewm(signed_ret, h), "up", 4 * h) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        b2 = np.column_stack([_bounded(_rolling_std(ret, h) + 0.5 * _rolling_std(tr, h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        b3 = np.column_stack([_bounded(_rolling_mean(np.maximum(signed_oo, 0.0) + np.abs(gap), h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        return b0, b1, b2, b3

    if method == "volatility_forecast_reconciliation":
        b0 = np.column_stack([
            _bounded(_reconciled_volatility(ret, gap, intra, tr, h, tw, cw), span=4 * h)
            for h, tw, cw in (
                (5, 0.20, 0.00), (8, 0.30, 0.05), (13, 0.40, 0.10), (21, 0.50, 0.15),
                (34, 0.60, 0.20), (55, 0.70, 0.25), (89, 0.80, 0.30), (144, 0.90, 0.35),
            )
        ])
        b1 = np.column_stack([_direction(_ewm(signed_ret, h), "up", 4 * h) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        b2 = np.column_stack([_bounded(_rolling_std(gap - intra, h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        b3 = np.column_stack([_bounded(_rolling_mean(np.abs(gap * intra) + 0.25 * tr * tr, h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        return b0, b1, b2, b3

    if method == "amre_multicandle_spot_vol":
        b0 = np.column_stack([
            _amre_multicandle_score(frame, side, w, rw, bw)
            for w, rw, bw in (
                (2, 0.90, 0.10), (3, 0.80, 0.15), (5, 0.70, 0.20), (8, 0.60, 0.25),
                (13, 0.50, 0.30), (21, 0.40, 0.35), (34, 0.30, 0.40), (55, 0.20, 0.45),
            )
        ])
        b1 = np.column_stack([_direction(_ewm(signed_ret, h), "up", 4 * h) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        b2 = np.column_stack([_bounded(_rolling_mean(np.abs(gap) + np.abs(intra), h), span=2 * h) for h in (3, 5, 8, 13, 21, 34, 55, 89)])
        b3 = np.column_stack([_bounded(_rolling_mean(np.abs(amount), h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        return b0, b1, b2, b3

    if method == "bct_ar_context_tree":
        b0 = np.column_stack([
            _context_tree_score(ret, side, depth, cut, decay)
            for depth, cut, decay in (
                (1, 0.50, 0.55), (1, 0.75, 0.65), (2, 0.50, 0.55), (2, 0.75, 0.65),
                (3, 0.50, 0.55), (3, 0.75, 0.65), (4, 0.50, 0.55), (4, 0.75, 0.65),
            )
        ])
        b1 = np.column_stack([_direction(_ewm(signed_ret, h), "up", 4 * h) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        b2 = np.column_stack([_bounded(_rolling_std(ret, h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        b3 = np.column_stack([_bounded(_rolling_mean(np.maximum(signed_ret, 0.0) + np.abs(gap), h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        return b0, b1, b2, b3

    if method == "factor_overnight_garch_ito":
        b0 = np.column_stack([
            _factor_overnight_garch(gap, intra, signed_ret, a, b, cw, side)
            for a, b, cw in (
                (0.05, 0.90, 0.00), (0.07, 0.88, 0.10), (0.09, 0.86, 0.20), (0.11, 0.84, 0.30),
                (0.13, 0.82, 0.40), (0.15, 0.80, 0.50), (0.18, 0.76, 0.60), (0.22, 0.72, 0.70),
            )
        ])
        b1 = np.column_stack([_direction(_ewm(signed_oo, h), "up", 4 * h) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        b2 = np.column_stack([_bounded(_rolling_std(gap, h) + _rolling_std(intra, h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        b3 = np.column_stack([_bounded(_rolling_mean(np.abs(gap * intra) + tr, h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        return b0, b1, b2, b3

    if method == "directional_change_intrinsic_time":
        b0 = np.column_stack([
            _directional_change_path(frame, side, theta, memory, contrarian=False)
            for theta, memory in (
                (0.003, 8), (0.005, 12), (0.008, 16), (0.012, 24),
                (0.018, 32), (0.025, 48), (0.035, 64), (0.050, 96),
            )
        ])
        b1 = np.column_stack([
            _directional_change_path(frame, side, theta, memory, contrarian=False)
            for theta, memory in (
                (0.004, 4), (0.006, 8), (0.010, 12), (0.015, 20),
                (0.022, 28), (0.030, 40), (0.042, 56), (0.060, 80),
            )
        ])
        b2 = np.column_stack([_bounded(_rolling_mean(np.maximum(signed_ret, 0.0) * (1.0 + tr), h), span=3 * h) for h in (4, 8, 12, 20, 32, 48, 72, 120)])
        b3 = np.column_stack([_bounded(_rolling_mean(np.abs(gap) + np.abs(intra), h), span=3 * h) for h in (4, 8, 12, 20, 32, 48, 72, 120)])
        return b0, b1, b2, b3

    if method == "volume_clock_toxicity":
        b0 = np.column_stack([
            _volume_clock_toxicity(frame, side, window, bucket, shock)
            for window, bucket, shock in (
                (8, 0.50, 0.20), (12, 0.75, 0.25), (16, 1.00, 0.30), (24, 1.25, 0.35),
                (32, 1.50, 0.40), (48, 2.00, 0.45), (64, 2.50, 0.50), (96, 3.00, 0.55),
            )
        ])
        b1 = np.column_stack([_bounded(_rolling_mean(np.abs(_raw(frame, "close_location", 0.5) - 0.5) * (1.0 + np.abs(amount)), h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        b2 = np.column_stack([_bounded(_rolling_std(signed_ret, h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        b3 = np.column_stack([_bounded(_rolling_mean(np.abs(signed_ret) / (np.abs(_raw(frame, "volume_ratio_20")) + 1.0), h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        return b0, b1, b2, b3

    if method == "constrained_ohlc_state_regression":
        b0 = np.column_stack([
            _ohlc_state_regression(frame, side, window, lag, ridge)
            for window, lag, ridge in (
                (64, 1, 0.10), (96, 1, 0.25), (128, 1, 0.50), (192, 2, 0.75),
                (256, 2, 1.00), (384, 3, 2.00), (512, 3, 4.00), (768, 4, 8.00),
            )
        ])
        b1 = np.column_stack([_direction(_ewm(signed_oo, h), "up", 4 * h) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        b2 = np.column_stack([_bounded(_rolling_std(_raw(frame, "range_pct"), h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        b3 = np.column_stack([_bounded(_rolling_mean(np.abs(_raw(frame, "close_location", 0.5) - 0.5) + np.abs(gap), h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        return b0, b1, b2, b3

    if method == "symbolic_grammar_ohlcv":
        b0 = np.column_stack([_symbolic_grammar_score(frame, side, k, scale) for k, scale in enumerate((8, 12, 16, 24, 32, 48, 64, 96))])
        b1 = np.column_stack([_symbolic_grammar_score(frame, side, (k + 2) % 8, scale) for k, scale in enumerate((10, 15, 20, 30, 40, 60, 80, 120))])
        b2 = np.column_stack([_symbolic_grammar_score(frame, side, (k + 4) % 8, scale) for k, scale in enumerate((12, 18, 24, 36, 48, 72, 96, 144))])
        b3 = np.column_stack([_symbolic_grammar_score(frame, side, (k + 6) % 8, scale) for k, scale in enumerate((16, 24, 32, 48, 64, 96, 128, 192))])
        return b0, b1, b2, b3

    if method == "order_imbalance_inventory":
        b0 = np.column_stack([_order_imbalance_inventory(frame, side, h, rev, imp) for h, rev, imp in (
            (3, 0.25, 0.05), (5, 0.35, 0.08), (8, 0.45, 0.10), (13, 0.55, 0.12),
            (21, 0.65, 0.15), (34, 0.75, 0.18), (55, 0.85, 0.22), (89, 0.95, 0.26),
        )])
        b1 = np.column_stack([_direction(_rolling_mean(signed_ret, h), "up", 2 * h) for h in (3, 5, 8, 13, 21, 34, 55, 89)])
        b2 = np.column_stack([_bounded(_rolling_std(_raw(frame, "body_pct"), h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        b3 = np.column_stack([_bounded(_rolling_mean(np.abs(signed_ret) * (1.0 + np.abs(amount)), h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        return b0, b1, b2, b3

    if method == "dc_overshoot_hazard":
        b0 = np.column_stack([
            _dc_overshoot_hazard(frame, side, theta, memory, hazard)
            for theta, memory, hazard in (
                (0.004, 8, 0.25), (0.006, 12, 0.30), (0.010, 16, 0.35), (0.015, 24, 0.40),
                (0.022, 32, 0.45), (0.030, 48, 0.50), (0.042, 64, 0.55), (0.060, 96, 0.60),
            )
        ])
        b1 = np.column_stack([_directional_change_path(frame, side, theta, memory, contrarian=True) for theta, memory in (
            (0.004, 8), (0.006, 12), (0.010, 16), (0.015, 24), (0.022, 32), (0.030, 48), (0.042, 64), (0.060, 96)
        )])
        b2 = np.column_stack([_bounded(_rolling_mean(np.abs(signed_ret) + tr, h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        b3 = np.column_stack([_bounded(_rolling_std(gap - intra, h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        return b0, b1, b2, b3

    if method == "overnight_jump_reversal":
        b0 = np.column_stack([
            _overnight_jump_reversal(frame, side, window, cut, memory, vw)
            for window, cut, memory, vw in (
                (8, 1.5, 4, 0.05), (12, 1.8, 6, 0.08), (16, 2.0, 8, 0.10), (24, 2.2, 12, 0.12),
                (32, 2.5, 16, 0.15), (48, 2.8, 24, 0.18), (64, 3.0, 32, 0.22), (96, 3.5, 48, 0.26),
            )
        ])
        b1 = np.column_stack([_direction(_ewm(signed_oo, h), "up", 4 * h) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        b2 = np.column_stack([_bounded(_rolling_mean(np.abs(gap) * (1.0 + np.abs(intra)), h), span=2 * h) for h in (3, 5, 8, 13, 21, 34, 55, 89)])
        b3 = np.column_stack([_bounded(_rolling_mean(np.abs(signed_oo) + tr, h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        return b0, b1, b2, b3

    if method == "overnight_daytime_tugwar":
        b0 = np.column_stack([_overnight_daytime_tugwar(frame, side, w, cut, asym) for w, cut, asym in (
            (10, 0.50, 0.80), (15, 0.60, 0.75), (20, 0.70, 0.70), (30, 0.80, 0.65),
            (40, 0.90, 0.60), (60, 1.00, 0.55), (90, 1.20, 0.50), (120, 1.50, 0.45),
        )])
        b1 = np.column_stack([_bounded(_rolling_mean(np.maximum(signed_oo, 0.0), h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        b2 = np.column_stack([_bounded(_rolling_mean(np.maximum(-signed_oo, 0.0), h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        b3 = np.column_stack([_bounded(_rolling_std(gap - intra, h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        return b0, b1, b2, b3

    if method == "cross_quantilogram_gap_intraday":
        b0 = np.column_stack([_cross_quantilogram_gap_intraday(frame, side, w, qg, qi, lag) for w, qg, qi, lag in (
            (32, 0.10, 0.90, 1), (48, 0.15, 0.85, 1), (64, 0.20, 0.80, 1), (96, 0.25, 0.75, 1),
            (128, 0.30, 0.70, 2), (160, 0.35, 0.65, 2), (192, 0.40, 0.60, 3), (256, 0.45, 0.55, 3),
        )])
        b1 = np.column_stack([_direction(_ewm(signed_oo, h), "up", 4 * h) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        b2 = np.column_stack([_bounded(_rolling_std(gap, h) + _rolling_std(intra, h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        b3 = np.column_stack([_bounded(_rolling_mean(np.abs(gap * intra) + 0.25 * tr, h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        return b0, b1, b2, b3

    if method == "opening_reversal_liquidity":
        b0 = np.column_stack([_opening_reversal_liquidity(frame, side, w, cut, memory, vw) for w, cut, memory, vw in (
            (5, 0.15, 4, 0.05), (8, 0.20, 6, 0.08), (13, 0.25, 8, 0.10), (21, 0.30, 12, 0.12),
            (34, 0.35, 16, 0.15), (55, 0.40, 24, 0.18), (89, 0.45, 32, 0.22), (120, 0.50, 48, 0.26),
        )])
        b1 = np.column_stack([_direction(_ewm(signed_oo, h), "up", 4 * h) for h in (2, 4, 8, 16, 32, 64, 128, 256)])
        b2 = np.column_stack([_bounded(_rolling_mean(np.abs(gap) * (1.0 + np.abs(amount)), h), span=2 * h) for h in (3, 5, 8, 13, 21, 34, 55, 89)])
        b3 = np.column_stack([_bounded(_rolling_mean(np.abs(intra) + tr, h), span=2 * h) for h in (5, 10, 20, 40, 60, 90, 120, 180)])
        return b0, b1, b2, b3

    if method == "har_semirange_leverage":
        # One call exposes four independent causal banks, each with eight
        # scale/shape choices.  The memory and leverage arguments are
        # pre-registered construction settings, not fitted on Test.
        b0, b1, b2, b3 = _har_semirange_leverage(frame, side, 22, 0.55, 0.15)
        return b0, b1, b2, b3

    if method == "two_tail_pot_hawkes":
        grids = (
            (8, 1.4, 0.25), (12, 1.6, 0.35), (16, 1.8, 0.45), (24, 2.0, 0.55),
            (32, 2.2, 0.70), (48, 2.5, 0.85), (64, 2.8, 1.00), (96, 3.2, 1.20),
        )
        banks = [_two_tail_pot_hawkes(frame, side, m, a, w) for m, a, w in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "conditional_duration_pot":
        grids = (
            (8, 0.80, 0.35), (12, 0.82, 0.45), (16, 0.84, 0.55), (24, 0.86, 0.65),
            (32, 0.88, 0.72), (48, 0.90, 0.78), (64, 0.92, 0.84), (96, 0.94, 0.90),
        )
        banks = [_conditional_duration_pot(frame, side, m, q, p) for m, q, p in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "gas_var_es":
        grids = (
            (0.03, 0.05, 1.10), (0.05, 0.07, 1.15), (0.07, 0.10, 1.20), (0.10, 0.12, 1.25),
            (0.14, 0.15, 1.30), (0.18, 0.18, 1.40), (0.24, 0.20, 1.55), (0.32, 0.25, 1.75),
        )
        banks = [_gas_var_es(frame, side, eta, alpha, ratio) for eta, alpha, ratio in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "restricted_quantile_scale":
        grids = (
            (8, 0.05, 0.95), (13, 0.07, 0.93), (22, 0.10, 0.90), (33, 0.12, 0.88),
            (44, 0.15, 0.85), (66, 0.18, 0.82), (99, 0.20, 0.80), (132, 0.25, 0.75),
        )
        banks = [_restricted_quantile_scale(frame, side, w, lo, hi) for w, lo, hi in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "heavy_range_volume_leverage":
        grids = (
            (3, 0.20, 0.15), (5, 0.28, 0.20), (8, 0.36, 0.25), (13, 0.44, 0.30),
            (22, 0.52, 0.35), (33, 0.60, 0.40), (44, 0.68, 0.48), (66, 0.76, 0.56),
        )
        banks = [_heavy_range_volume_leverage(frame, side, m, mw, lev) for m, mw, lev in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "threshold_quantile_autoregression":
        grids = (
            (8, 0.25, 0.25), (13, 0.30, 0.35), (22, 0.35, 0.45), (33, 0.40, 0.55),
            (44, 0.50, 0.65), (66, 0.60, 0.72), (99, 0.70, 0.80), (132, 0.75, 0.88),
        )
        banks = [_threshold_quantile_autoregression(frame, side, w, q, p) for w, q, p in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "volume_conditioned_reversal":
        grids = (
            (3, 0.20, 0.35), (5, 0.25, 0.45), (8, 0.30, 0.55), (13, 0.35, 0.65),
            (21, 0.40, 0.75), (34, 0.45, 0.85), (55, 0.50, 0.95), (89, 0.55, 1.10),
        )
        banks = [_volume_conditioned_reversal(frame, side, w, cut, rw) for w, cut, rw in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "panic_momentum_crash":
        grids = (
            (3, 1.05, 0.45), (5, 1.10, 0.55), (8, 1.15, 0.65), (13, 1.20, 0.75),
            (21, 1.25, 0.85), (34, 1.30, 0.95), (55, 1.40, 1.05), (89, 1.50, 1.20),
        )
        banks = [_panic_momentum_crash(frame, side, w, cut, rw) for w, cut, rw in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "liquidity_imbalance_absorption":
        grids = (
            (3, 0.35, 0.45), (5, 0.40, 0.55), (8, 0.45, 0.65), (13, 0.50, 0.75),
            (21, 0.55, 0.85), (34, 0.60, 0.95), (55, 0.65, 1.05), (89, 0.70, 1.20),
        )
        banks = [_liquidity_imbalance_absorption(frame, side, w, sat, rw) for w, sat, rw in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "large_move_volume_followup":
        grids = (
            (3, 0.80, 0.25), (5, 0.82, 0.35), (8, 0.84, 0.45), (13, 0.86, 0.55),
            (21, 0.88, 0.65), (34, 0.90, 0.75), (55, 0.92, 0.85), (89, 0.94, 0.95),
        )
        banks = [_large_move_volume_followup(frame, side, w, q, cw) for w, q, cw in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "state_conditional_momentum_adaptation":
        grids = (
            (8, 0.10, 1.05, 0.20), (12, 0.12, 1.10, 0.25), (16, 0.15, 1.15, 0.30), (24, 0.18, 1.20, 0.35),
            (32, 0.20, 1.25, 0.40), (48, 0.22, 1.30, 0.45), (64, 0.25, 1.40, 0.50), (96, 0.30, 1.50, 0.60),
        )
        banks = [_state_conditional_momentum_adaptation(frame, side, w, sc, cc, ad) for w, sc, cc, ad in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "volume_visibility_premium":
        grids = (
            (8, 0.05, 3, 0.35), (12, 0.08, 4, 0.45), (16, 0.10, 5, 0.55), (24, 0.12, 6, 0.65),
            (32, 0.15, 8, 0.75), (48, 0.18, 10, 0.85), (64, 0.22, 13, 0.95), (96, 0.25, 16, 1.05),
        )
        banks = [_volume_visibility_premium(frame, side, w, cut, mem, cont) for w, cut, mem, cont in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "volume_autocorrelation_inventory":
        grids = (
            (8, 0.20, 0.35, 0.35), (12, 0.25, 0.45, 0.45), (16, 0.30, 0.55, 0.55), (24, 0.35, 0.65, 0.65),
            (32, 0.40, 0.75, 0.75), (48, 0.45, 0.85, 0.85), (64, 0.50, 0.95, 0.95), (96, 0.55, 1.05, 1.05),
        )
        banks = [_volume_autocorrelation_inventory(frame, side, w, cut, rw, cw) for w, cut, rw, cw in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "delayed_liquidity_shock":
        grids = (
            (8, 0.80, 3, 0.35), (12, 0.85, 4, 0.45), (16, 0.90, 5, 0.55), (24, 0.95, 6, 0.65),
            (32, 1.00, 8, 0.75), (48, 1.05, 10, 0.85), (64, 1.10, 13, 0.95), (96, 1.15, 16, 1.05),
        )
        banks = [_delayed_liquidity_shock(frame, side, w, cut, mem, rw) for w, cut, mem, rw in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "lagged_week_reversal":
        grids = (
            (4, 8, 0.20, 0.25), (5, 8, 0.25, 0.35), (6, 12, 0.30, 0.45), (7, 12, 0.35, 0.55),
            (8, 16, 0.40, 0.65), (9, 21, 0.45, 0.75), (10, 34, 0.50, 0.85), (12, 55, 0.55, 0.95),
        )
        banks = [_lagged_week_reversal(frame, side, lag, w, cut, conf) for lag, w, cut, conf in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "extreme_event_decay":
        grids = (
            (8, 0.80, 3, 0.25), (12, 0.82, 4, 0.35), (16, 0.84, 5, 0.45), (24, 0.86, 6, 0.55),
            (32, 0.88, 8, 0.65), (48, 0.90, 10, 0.75), (64, 0.92, 13, 0.85), (96, 0.94, 16, 0.95),
        )
        banks = [_extreme_event_decay(frame, side, w, q, mem, cw) for w, q, mem, cw in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "negative_tail_transition_hazard":
        grids = (
            (16, 0.70, 0.25), (24, 0.80, 0.35), (32, 0.90, 0.45), (48, 1.00, 0.55),
            (64, 1.10, 0.65), (96, 1.25, 0.75), (128, 1.40, 0.90), (192, 1.60, 1.05),
        )
        banks = [_negative_tail_transition_hazard(frame, side, w, cut, age) for w, cut, age in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "crash_rebound_contrarian_switch":
        grids = (
            (8, 0.80, 0.20), (12, 0.82, 0.30), (16, 0.84, 0.40), (24, 0.86, 0.50),
            (32, 0.88, 0.60), (48, 0.90, 0.70), (64, 0.92, 0.80), (96, 0.94, 0.90),
        )
        banks = [_crash_rebound_contrarian_switch(frame, side, w, q, sw) for w, q, sw in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "anchor_barrier_distance":
        grids = (
            (20, 0.005, 0.25), (32, 0.007, 0.35), (60, 0.010, 0.45), (90, 0.013, 0.55),
            (120, 0.016, 0.65), (180, 0.020, 0.75), (252, 0.025, 0.85), (384, 0.030, 0.95),
        )
        banks = [_anchor_barrier_distance(frame, side, w, cut, mix) for w, cut, mix in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "variance_ratio_state":
        grids = (
            (32, 2, 0.10, 0.35), (48, 3, 0.12, 0.45), (64, 4, 0.15, 0.55), (96, 5, 0.18, 0.65),
            (128, 6, 0.20, 0.75), (160, 8, 0.22, 0.85), (192, 10, 0.25, 0.95), (256, 13, 0.28, 1.05),
        )
        banks = [_variance_ratio_state(frame, side, w, q, cut, mw) for w, q, cut, mw in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "support_break_retest_failure":
        grids = (
            (20, 0.005, 0.25), (32, 0.007, 0.35), (60, 0.010, 0.45), (90, 0.013, 0.55),
            (120, 0.016, 0.65), (180, 0.020, 0.75), (252, 0.025, 0.85), (384, 0.030, 0.95),
        )
        banks = [_support_break_retest_failure(frame, side, w, cut, decay) for w, cut, decay in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "tail_asymmetry_leverage":
        grids = (
            (16, 0.05, 0.20), (24, 0.07, 0.30), (32, 0.09, 0.40), (48, 0.11, 0.50),
            (64, 0.13, 0.60), (96, 0.15, 0.70), (128, 0.18, 0.80), (192, 0.20, 0.90),
        )
        banks = [_tail_asymmetry_leverage(frame, side, w, q, lw) for w, q, lw in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "liquidity_conditioned_reversal":
        grids = (
            (8, 0.70, 0.35, 0.25), (12, 0.75, 0.45, 0.35), (16, 0.80, 0.55, 0.45), (24, 0.82, 0.65, 0.55),
            (32, 0.84, 0.75, 0.65), (48, 0.86, 0.85, 0.75), (64, 0.88, 0.95, 0.85), (96, 0.90, 1.05, 0.95),
        )
        banks = [_liquidity_conditioned_reversal(frame, side, w, q, vw, lw) for w, q, vw, lw in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "overnight_decline_cycle_hazard":
        grids = (
            (8, 0.35, 0.25), (12, 0.45, 0.35), (16, 0.55, 0.45), (24, 0.65, 0.55),
            (32, 0.75, 0.65), (48, 0.85, 0.75), (64, 0.95, 0.85), (96, 1.05, 0.95),
        )
        banks = [_overnight_decline_cycle_hazard(frame, side, w, cut, persist) for w, cut, persist in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "square_root_impact_inventory":
        grids = (
            (8, 0.35, 0.35, 3), (12, 0.40, 0.45, 4), (16, 0.45, 0.55, 5), (24, 0.50, 0.65, 6),
            (32, 0.55, 0.75, 8), (48, 0.60, 0.85, 10), (64, 0.65, 0.95, 13), (96, 0.70, 1.05, 16),
        )
        banks = [_square_root_impact_inventory(frame, side, w, cut, aw, mem) for w, cut, aw, mem in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "intraday_pressure_state_switch":
        grids = (
            (8, 0.20, 0.45, 3), (12, 0.30, 0.50, 4), (16, 0.40, 0.55, 5), (24, 0.50, 0.60, 6),
            (32, 0.60, 0.65, 8), (48, 0.70, 0.70, 10), (64, 0.80, 0.75, 13), (96, 0.90, 0.80, 16),
        )
        banks = [_intraday_pressure_state_switch(frame, side, w, pc, sc, mem) for w, pc, sc, mem in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "causal_rank_consensus_tail":
        grids = (
            (32, 0.70, 0.35, 3), (48, 0.80, 0.45, 4), (64, 0.90, 0.55, 5), (96, 1.00, 0.65, 6),
            (128, 1.10, 0.75, 8), (160, 1.20, 0.85, 10), (192, 1.30, 0.95, 13), (256, 1.40, 1.05, 16),
        )
        banks = [_causal_rank_consensus_tail(frame, side, w, power, cut, mem) for w, power, cut, mem in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "online_expectile_tail_link":
        grids = (
            (8, 0.65, 0.08, 0.20), (12, 0.70, 0.10, 0.35), (16, 0.75, 0.12, 0.50), (24, 0.80, 0.15, 0.75),
            (32, 0.84, 0.18, 1.00), (48, 0.88, 0.22, 1.50), (64, 0.92, 0.28, 2.00), (96, 0.95, 0.35, 3.00),
        )
        banks = [_online_expectile_tail_link(frame, side, w, tau, lr, ridge) for w, tau, lr, ridge in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "local_expectile_state":
        grids = (
            (24, 0.65, 0.12, 1), (32, 0.70, 0.16, 1), (48, 0.75, 0.22, 2), (64, 0.80, 0.28, 2),
            (96, 0.84, 0.36, 2), (128, 0.88, 0.46, 3), (192, 0.92, 0.60, 3), (256, 0.95, 0.78, 4),
        )
        banks = [_local_expectile_state(frame, side, w, tau, scale, iters) for w, tau, scale, iters in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "range_volume_location_tail":
        grids = (
            (16, 0.58, 0.35, 3), (24, 0.62, 0.45, 4), (32, 0.66, 0.55, 5), (48, 0.70, 0.65, 6),
            (64, 0.74, 0.75, 8), (96, 0.78, 0.85, 10), (128, 0.82, 0.95, 13), (192, 0.86, 1.05, 16),
        )
        banks = [_range_volume_location_tail(frame, side, w, cut, weight, mem) for w, cut, weight, mem in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "caviar_spot_tail_link":
        grids = (
            (16, 0.80, 0.03, 0.25), (24, 0.82, 0.04, 0.35), (32, 0.84, 0.05, 0.45), (48, 0.86, 0.06, 0.55),
            (64, 0.88, 0.08, 0.65), (96, 0.90, 0.10, 0.75), (128, 0.92, 0.14, 0.90), (192, 0.95, 0.18, 1.05),
        )
        banks = [_caviar_spot_tail_link(frame, side, w, tau, lr, sw) for w, tau, lr, sw in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "bayesian_state_tail":
        grids = (
            (16, 0.58, 1.0, 8), (24, 0.62, 2.0, 12), (32, 0.66, 4.0, 16), (48, 0.70, 8.0, 24),
            (64, 0.74, 12.0, 32), (96, 0.78, 20.0, 48), (128, 0.82, 32.0, 64), (192, 0.86, 48.0, 96),
        )
        banks = [_bayesian_state_tail(frame, side, w, cut, prior, mem) for w, cut, prior, mem in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "online_boa_spot_experts":
        grids = (
            (16, 0.10, 0.00, 0.50), (24, 0.18, 0.03, 0.65), (32, 0.28, 0.06, 0.80), (48, 0.40, 0.10, 0.95),
            (64, 0.55, 0.15, 1.10), (96, 0.75, 0.20, 1.30), (128, 1.00, 0.27, 1.60), (192, 1.35, 0.35, 2.00),
        )
        banks = [_online_boa_spot_experts(frame, side, w, learning, share, temperature) for w, learning, share, temperature in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "hawkes_spot_jump_intensity":
        grids = (
            (16, 0.08, 0.55, 0.70), (24, 0.11, 0.70, 0.85), (32, 0.15, 0.85, 1.00), (48, 0.20, 1.00, 1.15),
            (64, 0.26, 1.15, 1.30), (96, 0.33, 1.35, 1.50), (128, 0.41, 1.60, 1.75), (192, 0.50, 1.90, 2.00),
        )
        banks = [_hawkes_spot_jump_intensity(frame, side, w, decay, threshold, mark_power) for w, decay, threshold, mark_power in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "bayesian_online_spot_changepoint":
        grids = (
            (8, 0.08, 0.60, 8), (12, 0.10, 0.75, 12), (16, 0.12, 0.90, 16), (24, 0.15, 1.05, 24),
            (32, 0.18, 1.20, 32), (48, 0.22, 1.45, 48), (64, 0.28, 1.75, 72), (96, 0.35, 2.10, 96),
        )
        banks = [_bayesian_online_spot_changepoint(frame, side, m, hz, cut, mem) for m, hz, cut, mem in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "kalman_spot_dynamic_tail":
        grids = (
            (0.945, 0.20, 0.45, 0.00), (0.955, 0.35, 0.55, 0.10), (0.965, 0.55, 0.70, 0.20), (0.972, 0.80, 0.85, 0.30),
            (0.980, 1.10, 1.00, 0.40), (0.987, 1.50, 1.25, 0.55), (0.993, 2.10, 1.55, 0.70), (0.997, 3.00, 2.00, 0.90),
        )
        banks = [_kalman_spot_dynamic_tail(frame, side, d, prior, noise, mix) for d, prior, noise, mix in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "causal_quantile_partition_tail":
        grids = (
            (64, 2, 0.35, 0.50), (96, 3, 0.45, 0.75), (128, 4, 0.55, 1.00), (192, 5, 0.65, 1.35),
            (256, 6, 0.80, 1.75), (384, 7, 0.95, 2.25), (512, 8, 1.15, 3.00), (768, 10, 1.40, 4.00),
        )
        banks = [_causal_quantile_partition_tail(frame, side, w, b, rec, prior) for w, b, rec, prior in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "ordinal_open_transition_tail":
        grids = (
            (64, 3, 0.35, 0.50), (96, 3, 0.45, 0.75), (128, 4, 0.55, 1.00), (192, 4, 0.65, 1.35),
            (256, 5, 0.80, 1.75), (384, 6, 0.95, 2.25), (512, 7, 1.15, 3.00), (768, 8, 1.40, 4.00),
        )
        banks = [_ordinal_open_transition_tail(frame, side, w, b, rec, prior) for w, b, rec, prior in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "online_bayesian_logistic_tail":
        grids = (
            (0.945, 0.12, 0.08, 0.00), (0.955, 0.20, 0.14, 0.10), (0.965, 0.32, 0.22, 0.20), (0.972, 0.48, 0.32, 0.30),
            (0.980, 0.68, 0.48, 0.40), (0.987, 0.88, 0.70, 0.55), (0.993, 1.10, 1.00, 0.70), (0.997, 1.35, 1.50, 0.90),
        )
        banks = [_online_bayesian_logistic_tail(frame, side, d, lr, rg, mix) for d, lr, rg, mix in grids]
        return tuple(np.column_stack([x[j][:, i] for i, x in enumerate(banks)]) for j in range(4))

    if method == "quantile_transfer_entropy_tail":
        # Quantile-transfer-entropy banks: activity, range, gap/intraday
        # pressure, and signed price direction are deliberately separate
        # drivers.  The grid varies state resolution, memory, causal lag and
        # prior mass; it is not a free-form feature search.
        volume = np.log1p(np.maximum(_raw(frame, "volume"), 0.0))
        amount_level = np.log1p(np.maximum(_raw(frame, "amount"), 0.0))
        activity = 0.55 * volume + 0.45 * amount_level
        range_source = np.abs(tr) + 0.35 * np.abs(_raw(frame, "ret_1"))
        gap_source = np.abs(gap) + 0.70 * np.abs(intra)
        direction_source = signed_ret
        grids = (
            (2, 8, 1, 0.25), (3, 12, 1, 0.50), (4, 16, 2, 0.75), (5, 24, 2, 1.00),
            (6, 32, 3, 1.50), (7, 48, 3, 2.00), (8, 64, 4, 3.00), (8, 96, 5, 5.00),
        )
        sources = (activity, range_source, gap_source, direction_source)
        banks = [
            np.column_stack([
                _quantile_transfer_entropy_tail(frame, side, source, bins, memory, lag, alpha)
                for bins, memory, lag, alpha in grids
            ])
            for source in sources
        ]
        return tuple(banks)

    if method == "drawdown_speed_failure":
        grids = (
            (12, 0.35, 0.25, 0.20), (16, 0.45, 0.35, 0.30), (22, 0.55, 0.45, 0.40), (32, 0.65, 0.55, 0.50),
            (48, 0.75, 0.65, 0.60), (64, 0.85, 0.75, 0.70), (96, 1.00, 0.90, 0.80), (144, 1.20, 1.05, 0.95),
        )
        components = [_drawdown_speed_components(frame, side, w, p, aw, vw) for w, p, aw, vw in grids]
        return tuple(np.column_stack([component[j] for component in components]) for j in range(4))

    if method == "kernel_candle_support_tail":
        # A compact, systematic chart-pattern geometry proxy inspired by the
        # kernel pattern-recognition treatment of Lo, Mamaysky and Wang.  It
        # uses no hand-labelled pattern names: all inputs are causal ranks of
        # candle shape, support/resistance distance, range compression and
        # activity, with the open-direction persistence bank kept separate.
        def rank_value(name: str, inverse: bool = False) -> np.ndarray:
            value = np.nan_to_num(_raw(frame, f"{name}_rank252", 0.5), nan=0.5)
            return 1.0 - value if inverse else value

        b0 = np.column_stack([
            _direction(_rolling_mean(signed_oo, max(2, h)), "up", 4 * h)
            for h in (2, 4, 8, 16, 32, 64, 128, 256)
        ])
        if side == "down":
            b1 = np.column_stack([
                rank_value("lower_shadow_share") * w + rank_value("support_break") * (1.0 - w)
                for w in np.linspace(0.20, 0.90, 8)
            ])
            b2 = np.column_stack([
                rank_value("true_range_pct", True) * w + rank_value("range_position_20", True) * (1.0 - w)
                for w in np.linspace(0.20, 0.90, 8)
            ])
            b3 = np.column_stack([
                rank_value("amount_ratio_5", True) * w + rank_value("volume_ratio_20", True) * (1.0 - w)
                for w in np.linspace(0.20, 0.90, 8)
            ])
        else:
            b1 = np.column_stack([
                rank_value("resistance_break") * w + rank_value("close_location") * (1.0 - w)
                for w in np.linspace(0.20, 0.90, 8)
            ])
            b2 = np.column_stack([
                rank_value("true_range_pct") * w + rank_value("amount_z_20") * (1.0 - w)
                for w in np.linspace(0.20, 0.90, 8)
            ])
            b3 = np.column_stack([
                rank_value("volume_ratio_20") * w + rank_value("range_position_20") * (1.0 - w)
                for w in np.linspace(0.20, 0.90, 8)
            ])
        return b0, b1, b2, b3

    raise KeyError(f"unknown paper method: {method}")


def score_matrix(frame: pd.DataFrame, meta: dict[str, Any], side: str) -> tuple[np.ndarray, pd.DataFrame]:
    banks = _banks(frame, str(meta["core_logic_name"]), side)
    if len(banks) != 4 or any(bank.shape != (len(frame), 8) for bank in banks):
        raise AssertionError(f"paper primitive banks must be (n,8): {[bank.shape for bank in banks]}")
    scores: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    index = 0
    for a0 in range(8):
        for a1 in range(8):
            for a2 in range(8):
                for a3 in range(8):
                    score = 0.34 * banks[0][:, a0] + 0.28 * banks[1][:, a1] + 0.22 * banks[2][:, a2] + 0.16 * banks[3][:, a3]
                    scores.append(np.clip(score, 0, 1))
                    rows.append({
                        "base_candidate_id": f"base_{index + 1:04d}",
                        "base_index": index,
                        "axis_1": a0,
                        "axis_2": a1,
                        "axis_3": a2,
                        "axis_4": a3,
                    })
                    index += 1
    matrix = np.column_stack(scores)
    metadata = pd.DataFrame(rows)
    if matrix.shape != (len(frame), 4096):
        raise AssertionError(f"paper score matrix mismatch: {matrix.shape}")
    return matrix, metadata
