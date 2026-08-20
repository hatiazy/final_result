"""Causal executable adaptations for the V51--V90 paper candidates.

The research protocol treats a paper as a single core transformation, not as a
claim that a daily OHLCV implementation is identical to the paper's original
data.  This module therefore keeps every adaptation deliberately small,
one-sided, and auditable.  It returns an up-oriented signed score; the public
dispatcher multiplies by ``side`` so a larger value always supports that side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from .design_registry import SCORE_VARIANTS_BY_VERSION
from .logic_features import _base, _safe_div
from .logic_registry import LOGIC_BY_VERSION


DEV_END = pd.Timestamp("2022-12-31")
SELECTION_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
SEED = 15452026


def _finite(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)[np.isfinite(values)]


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    center = np.nanmedian(values, axis=0)
    scale = np.nanmedian(np.abs(values - center), axis=0) * 1.4826
    scale = np.where(np.isfinite(scale) & (scale > 1e-10), scale, 1.0)
    return (values - center) / scale


def _feature_frame(base: dict[str, pd.Series]) -> pd.DataFrame:
    """Low-dimensional causal spot-only channels used by reserve methods."""

    ret = base["ret"]
    frame = pd.DataFrame(
        {
            "ret": ret,
            "gap": base["gap"],
            "intraday": base["intraday"],
            "range": base["range"],
            "volume": np.log1p(base["volume"]),
            "volume_change": np.log1p(base["volume"]).diff(),
            "amount_change": np.log1p(base["amount"]).diff(),
            "open_ret": base["open_ret"],
            "location": base["location"],
        },
        index=base["close"].index,
    )
    return frame.replace([np.inf, -np.inf], np.nan)


def _last_window(frame: pd.DataFrame, i: int, window: int) -> np.ndarray | None:
    if i + 1 < window:
        return None
    values = frame.iloc[i - window + 1 : i + 1].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    return values


def _previous_two_windows(frame: pd.DataFrame, i: int, window: int) -> tuple[np.ndarray, np.ndarray] | None:
    if i + 1 < 2 * window:
        return None
    previous = frame.iloc[i - 2 * window + 1 : i - window + 1].to_numpy(dtype=float)
    recent = frame.iloc[i - window + 1 : i + 1].to_numpy(dtype=float)
    if not np.isfinite(previous).all() or not np.isfinite(recent).all():
        return None
    return previous, recent


def _pairwise_distances(left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    right = left if right is None else right
    return np.sqrt(np.maximum(0.0, ((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=2)))


def _subsample(values: np.ndarray, max_rows: int = 32) -> np.ndarray:
    if len(values) <= max_rows:
        return values
    take = np.linspace(0, len(values) - 1, max_rows).round().astype(int)
    return values[take]


def _mmd_score(previous: np.ndarray, recent: np.ndarray, bandwidth_multiplier: float) -> float:
    previous, recent = _subsample(previous), _subsample(recent)
    pooled = np.vstack([previous, recent])
    distances = _pairwise_distances(pooled)
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    bandwidth = max(1e-6, bandwidth * bandwidth_multiplier)
    kernel = lambda d: np.exp(-(d**2) / (2.0 * bandwidth**2))
    k_pp = kernel(_pairwise_distances(previous)).mean()
    k_rr = kernel(_pairwise_distances(recent)).mean()
    k_pr = kernel(_pairwise_distances(previous, recent)).mean()
    direction = np.sign(recent[:, 0].mean() - previous[:, 0].mean())
    return float(direction * max(0.0, k_pp + k_rr - 2.0 * k_pr))


def _energy_score(previous: np.ndarray, recent: np.ndarray, exponent: float) -> float:
    previous, recent = _subsample(previous), _subsample(recent)
    within_p = _pairwise_distances(previous)
    within_r = _pairwise_distances(recent)
    cross = _pairwise_distances(previous, recent)
    energy = 2.0 * np.power(cross, exponent).mean() - np.power(within_p, exponent).mean() - np.power(within_r, exponent).mean()
    direction = np.sign(recent[:, 0].mean() - previous[:, 0].mean())
    return float(direction * max(0.0, energy))


def _sliced_wasserstein(previous: np.ndarray, recent: np.ndarray, projection_count: int) -> float:
    previous, recent = _subsample(previous), _subsample(recent)
    rng = np.random.default_rng(SEED + projection_count)
    directions = rng.normal(size=(projection_count, previous.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(1e-12)
    distances = []
    signed = []
    for direction in directions:
        left = np.sort(previous @ direction)
        right = np.sort(recent @ direction)
        q = np.linspace(0.0, 1.0, min(len(left), len(right)))
        distances.append(float(np.mean(np.quantile(right, q) - np.quantile(left, q))))
        signed.append(float(np.mean(right) - np.mean(left)))
    magnitude = float(np.mean(np.abs(distances)))
    direction = np.sign(np.mean(signed))
    return direction * magnitude


def _rulsif_score(previous: np.ndarray, recent: np.ndarray, alpha: float) -> float:
    previous, recent = _subsample(previous, 20), _subsample(recent, 20)
    centers = previous[:: max(1, len(previous) // 10)]
    distances = _pairwise_distances(recent, centers)
    scale = np.median(_pairwise_distances(previous))
    scale = max(float(scale), 1e-6)
    kernels = np.exp(-(distances**2) / (2.0 * scale**2))
    numerator = kernels.mean(axis=1)
    denominator = alpha + (1.0 - alpha) * numerator.mean()
    ratio = np.clip(numerator / max(denominator, 1e-8), 1e-4, 1e4)
    direction = np.sign(recent[:, 0].mean() - previous[:, 0].mean())
    return float(direction * np.log(ratio.mean()))


def _graph_scan_score(previous: np.ndarray, recent: np.ndarray, k: int) -> float:
    previous, recent = _subsample(previous, 24), _subsample(recent, 24)
    all_values = np.vstack([previous, recent])
    distances = _pairwise_distances(all_values)
    np.fill_diagonal(distances, np.inf)
    labels = np.r_[np.zeros(len(previous), dtype=int), np.ones(len(recent), dtype=int)]
    nearest = np.argpartition(distances, min(k, len(all_values) - 2), axis=1)[:, :k]
    cross = np.mean(labels[nearest] != labels[:, None])
    direction = np.sign(recent[:, 0].mean() - previous[:, 0].mean())
    return float(direction * cross)


def _ordinal_entropy(values: np.ndarray, m: int, delay: int) -> tuple[float, float]:
    patterns: list[tuple[int, ...]] = []
    for end in range((m - 1) * delay, len(values)):
        block = values[end - (m - 1) * delay : end + 1 : delay]
        if len(block) != m or not np.isfinite(block).all():
            continue
        patterns.append(tuple(np.argsort(block, kind="mergesort")))
    if not patterns:
        return np.nan, np.nan
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum() / math.log(math.factorial(m)))
    ups = sum(pattern[-1] > pattern[0] for pattern in patterns) / len(patterns)
    return entropy, float(2.0 * ups - 1.0)


def _delay_embedding(values: np.ndarray, m: int, delay: int) -> np.ndarray:
    needed = (m - 1) * delay + 1
    if len(values) < needed:
        return np.empty((0, m))
    return np.vstack([values[i - (m - 1) * delay : i + 1 : delay] for i in range(needed - 1, len(values))])


def _rqa_score(values: np.ndarray, m: int, delay: int) -> float:
    points = _delay_embedding(values, m, delay)
    if len(points) < 12:
        return np.nan
    points = _subsample(points, 60)
    distances = _pairwise_distances(points)
    threshold = np.quantile(distances[np.triu_indices_from(distances, 1)], 0.10)
    recurrence = distances <= max(threshold, 1e-8)
    diagonal_runs = []
    for offset in range(-len(points) + 2, len(points) - 1):
        diagonal = np.diag(recurrence, k=offset)
        run = 0
        for value in diagonal:
            run = run + 1 if value else 0
            if run >= 2:
                diagonal_runs.append(run)
    determinism = float(sum(diagonal_runs) / max(1, recurrence.sum()))
    direction = np.sign(np.mean(np.diff(values[-min(10, len(values)) :])))
    return float(direction * determinism)


def _visibility_endpoint(path: np.ndarray) -> float:
    if len(path) < 4:
        return np.nan
    endpoint = path[-1]
    degree = 0
    slopes = []
    for j in range(len(path) - 1):
        slope = (endpoint - path[j]) / (len(path) - 1 - j)
        slopes.append(slope)
    running = -np.inf
    for slope in reversed(slopes):
        if slope >= running:
            degree += 1
            running = slope
    return float(degree / len(path))


def _matrix_profile_score(values: np.ndarray, length: int, neighbors: int) -> float:
    if len(values) < 2 * length + 3:
        return np.nan
    current = values[-length:]
    current = (current - current.mean()) / max(current.std(ddof=0), 1e-8)
    candidates: list[tuple[float, float]] = []
    end_max = len(values) - length - 1
    # A bounded historical library keeps the Matrix Profile adaptation causal
    # while preventing an O(T^2) scan for every daily endpoint.
    start = max(length - 1, end_max - 80)
    for end in range(start, end_max + 1, max(1, length)):
        if end + 1 >= len(values) - length + 1:
            break
        path = values[end - length + 1 : end + 1]
        if not np.isfinite(path).all():
            continue
        path = (path - path.mean()) / max(path.std(ddof=0), 1e-8)
        distance = float(np.sqrt(np.mean((path - current) ** 2)))
        candidates.append((distance, float(values[end + 1] - values[end])))
    if not candidates:
        return np.nan
    candidates.sort(key=lambda row: row[0])
    chosen = candidates[:neighbors]
    weights = np.array([1.0 / (1e-6 + row[0]) for row in chosen])
    return float(np.average([row[1] for row in chosen], weights=weights))


def _sax_word(values: np.ndarray, alphabet: int, segments: int = 4) -> tuple[int, ...] | None:
    if len(values) < segments:
        return None
    values = values[-segments * max(1, len(values) // segments) :]
    pieces = np.array_split(values, segments)
    means = np.array([piece.mean() for piece in pieces])
    z = (means - means.mean()) / max(means.std(ddof=0), 1e-8)
    bins = np.linspace(-1.5, 1.5, alphabet - 1)
    return tuple(np.digitize(z, bins).tolist())


def _sax_scores(log_close: np.ndarray, returns: np.ndarray, window: int, alphabet: int) -> np.ndarray:
    """One-pass causal SAX lookup table (historical words only)."""

    output = np.full(len(log_close), np.nan, dtype=float)
    words: list[tuple[int, ...] | None] = [None] * len(log_close)
    for i in range(window - 1, len(log_close)):
        words[i] = _sax_word(log_close[i - window + 1 : i + 1], alphabet)
    history: dict[tuple[int, ...], list[float]] = {}
    for i in range(window - 1, len(log_close)):
        previous_end = i - 1
        if previous_end >= window - 1 and words[previous_end] is not None and np.isfinite(returns[i]):
            history.setdefault(words[previous_end], []).append(float(returns[i]))
        current = words[i]
        if current is not None and len(history.get(current, [])) >= 3:
            output[i] = float(np.mean(history[current]))
    return output


def _transfer_entropy(values: pd.DataFrame, target_name: str, driver_name: str, lag: int, window: int) -> float:
    if len(values) < window:
        return np.nan
    frame = values.iloc[-window:].copy()
    target = pd.qcut(frame[target_name].rank(method="first"), 3, labels=False).to_numpy()
    driver = pd.qcut(frame[driver_name].rank(method="first"), 3, labels=False).to_numpy()
    rows: list[tuple[int, int, int]] = []
    for t in range(max(1, lag), len(frame)):
        rows.append((int(target[t]), int(target[t - 1]), int(driver[t - lag])))
    if len(rows) < 20:
        return np.nan
    array = np.asarray(rows, dtype=int)
    # Conditional mutual information I(Y_t; X_{t-lag}|Y_{t-1}).
    result = 0.0
    for y_prev in range(3):
        subset = array[array[:, 1] == y_prev]
        if not len(subset):
            continue
        p_y = np.bincount(subset[:, 0], minlength=3) / len(subset)
        p_x = np.bincount(subset[:, 2], minlength=3) / len(subset)
        p_yx = np.zeros((3, 3), dtype=float)
        for y, x in subset[:, [0, 2]]:
            p_yx[y, x] += 1.0
        p_yx /= len(subset)
        for y in range(3):
            for x in range(3):
                if p_yx[y, x] > 0 and p_y[y] > 0 and p_x[x] > 0:
                    result += (len(subset) / len(array)) * p_yx[y, x] * np.log(p_yx[y, x] / (p_y[y] * p_x[x]))
    direction = np.sign(frame.loc[frame.index[-min(10, len(frame)) :], target_name].mean())
    return float(direction * result)


def _directional_change(values: np.ndarray, threshold: float) -> float:
    if len(values) < 3:
        return np.nan
    direction = 1
    extreme = values[0]
    overshoot = 0.0
    event_age = 0
    for value in values[1:]:
        event_age += 1
        if direction > 0:
            extreme = max(extreme, value)
            if value < extreme - threshold:
                direction = -1
                overshoot = extreme - value
                extreme = value
                event_age = 0
        else:
            extreme = min(extreme, value)
            if value > extreme + threshold:
                direction = 1
                overshoot = value - extreme
                extreme = value
                event_age = 0
    return float(direction * (1.0 + overshoot / max(threshold, 1e-8)) / math.sqrt(1.0 + event_age))


def _hawkes_score(returns: np.ndarray, threshold: float, half_life: float) -> float:
    if len(returns) < 4:
        return np.nan
    sigma = np.nanstd(returns[:-1])
    if not np.isfinite(sigma) or sigma <= 1e-10:
        return 0.0
    up = (returns > threshold * sigma).astype(float)
    down = (returns < -threshold * sigma).astype(float)
    decay = math.exp(-math.log(2.0) / max(half_life, 1e-6))
    up_intensity = down_intensity = 0.0
    for up_event, down_event in zip(up, down):
        up_intensity = decay * up_intensity + up_event
        down_intensity = decay * down_intensity + down_event
    return float(up_intensity - down_intensity)


def _l1_trend_score(values: np.ndarray, lambda_scale: float) -> float:
    if len(values) < 8:
        return np.nan
    first = np.diff(values)
    second = np.diff(first)
    threshold = lambda_scale * np.nanstd(second)
    shrunk = np.sign(second) * np.maximum(np.abs(second) - threshold, 0.0)
    filtered = np.r_[first[0], first[0] + np.cumsum(shrunk)]
    return float(np.tanh(filtered[-1] / max(np.nanstd(first), 1e-8)))


def _ssa_score(values: np.ndarray, rank: int) -> float:
    if len(values) < 20:
        return np.nan
    length = max(5, min(len(values) // 2, 30))
    hankel = np.column_stack([values[i : i + length] for i in range(len(values) - length + 1)])
    u, singular, _ = np.linalg.svd(hankel - hankel.mean(axis=1, keepdims=True), full_matrices=False)
    keep = max(1, min(rank, len(singular)))
    reconstructed = (u[:, :keep] * singular[:keep]) @ np.ones((keep, hankel.shape[1]))
    trend = reconstructed[-1].mean()
    return float((trend - reconstructed[0].mean()) / max(np.nanstd(values), 1e-8))


def _wavelet_score(values: np.ndarray, wavelet: str, level: int) -> float:
    if len(values) < 16:
        return np.nan
    # A trailing Haar-like multiresolution proxy.  Filter names only alter the
    # fixed scale weights, so no centered/future padding can enter the score.
    weights = {"haar": 1.0, "db2": 1.1, "db4": 1.2, "sym4": 0.9}
    current = values.copy()
    low = 0.0
    high_energy = 0.0
    for number in range(level):
        if len(current) < 4:
            break
        even, odd = current[-(len(current) // 2) * 2 :: 2], current[-(len(current) // 2) * 2 + 1 :: 2]
        detail = (even - odd) / math.sqrt(2.0)
        current = (even + odd) / math.sqrt(2.0)
        low = float(current[-1] - current[0]) if len(current) else low
        high_energy += float(np.mean(detail**2))
    return float(weights.get(wavelet, 1.0) * low / max(math.sqrt(high_energy), 1e-8))


def _dmd_score(values: np.ndarray, delay_depth: int, rank: int) -> float:
    if len(values) < delay_depth + 12:
        return np.nan
    x = np.vstack([values[t - delay_depth : t] for t in range(delay_depth, len(values))]).T
    if x.shape[1] < 4:
        return np.nan
    left, right = x[:, :-1], x[:, 1:]
    u, singular, vt = np.linalg.svd(left, full_matrices=False)
    keep = max(1, min(rank, len(singular)))
    right_vectors = vt[:keep].T
    operator = right @ right_vectors @ np.diag(1.0 / np.maximum(singular[:keep], 1e-8)) @ u[:, :keep].T
    prediction = operator @ x[:, -1]
    return float(prediction[0])


def _signature_score(frame: np.ndarray, order: int) -> float:
    if len(frame) < 3:
        return np.nan
    increments = np.diff(_standardize(frame), axis=0)
    first = increments.sum(axis=0)
    score = float(first[0])
    if order >= 2:
        score += float((increments[:-1, :, None] * increments[1:, None, :]).sum()) * 0.1
    if order >= 3:
        score += float(np.mean(np.cumsum(increments, axis=0) ** 3)) * 0.02
    if order >= 4:
        score += float(np.mean(np.cumsum(increments, axis=0) ** 4)) * 0.005 * np.sign(first[0])
    return score


def _topology_score(values: np.ndarray, m: int, delay: int) -> float:
    points = _delay_embedding(values, m, delay)
    if len(points) < 8:
        return np.nan
    points = _subsample(points, 40)
    distances = _pairwise_distances(points)
    upper = distances[np.triu_indices_from(distances, 1)]
    if not len(upper):
        return np.nan
    # 0D persistence proxy: total MST length and the longest birth/death scale.
    order = np.argsort(upper)
    scale = np.quantile(upper, 0.75) + 1e-8
    topology = float(np.mean(np.minimum(upper, scale)) / scale)
    return float(np.sign(np.mean(np.diff(values[-min(12, len(values)) :]))) * topology)


def _edm_score(values: np.ndarray, embedding: int, neighbour_multiplier: int) -> float:
    points = _delay_embedding(values, embedding, 1)
    if len(points) < embedding + 3:
        return np.nan
    current = points[-1]
    library = points[:-2]
    distances = np.linalg.norm(library - current, axis=1)
    count = max(embedding + 1, min(len(library), neighbour_multiplier * (embedding + 1)))
    nearest = np.argsort(distances)[:count]
    next_values = values[embedding : len(values) - 1]
    next_values = next_values[-len(library) :]
    weights = np.exp(-distances[nearest] / max(np.median(distances[nearest]), 1e-8))
    return float(np.average(next_values[nearest], weights=weights))


def _duration_score(values: np.ndarray, state: np.ndarray, state_count: int, max_duration: int) -> float:
    if len(values) < 20:
        return np.nan
    signs = np.sign(values)
    current_sign = signs[-1]
    duration = 1
    while duration < len(signs) and signs[-duration - 1] == current_sign:
        duration += 1
    duration = min(duration, max_duration)
    # Explicit duration hazard: long same-sign runs are down-weighted by their
    # empirical continuation probability, unlike a memoryless HMM filter.
    continuation = 1.0 - duration / max_duration
    return float(current_sign * (0.5 + 0.5 * continuation) * (1.0 + 0.05 * (state_count - 2)))


def _annual_predictions(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    side: int,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> pd.Series:
    """Fit only on labels available before each evaluation year.

    For 2025 onward the training cut is permanently 2024-12-31, so Test labels
    can never enter a model fit even when several Test years are present.
    """

    dates = features.index
    target = side * panel.set_index("formation_date")["o2o_h1"].reindex(dates)
    zero = panel.set_index("formation_date")["state"].reindex(dates).eq(0)
    output = pd.Series(np.nan, index=dates, dtype=float)
    for year in sorted(set(dates.year)):
        start = pd.Timestamp(f"{year}-01-01")
        eval_mask = dates.year == year
        cutoff = SELECTION_END if start >= TEST_START else start - pd.Timedelta(days=1)
        train_mask = (dates < start) & (dates <= cutoff) & zero & target.notna()
        if int(train_mask.sum()) < 40:
            continue
        x_train = features.loc[train_mask].to_numpy(dtype=float)
        y_train = target.loc[train_mask].to_numpy(dtype=float)
        x_eval = features.loc[eval_mask].to_numpy(dtype=float)
        medians = np.nanmedian(x_train, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        x_train = np.where(np.isfinite(x_train), x_train, medians)
        x_eval = np.where(np.isfinite(x_eval), x_eval, medians)
        try:
            output.loc[eval_mask] = fit_predict(x_train, y_train, x_eval)
        except Exception:
            # A legal numerical failure is represented as missing candidates;
            # the runner records the failure count rather than silently changing
            # the method class.
            continue
    return output


def _model_features(base: dict[str, pd.Series], panel: pd.DataFrame) -> pd.DataFrame:
    ret = base["ret"]
    log_close = base["log_close"]
    frame = pd.DataFrame(
        {
            "ret1": ret,
            "ret3": log_close.diff(3),
            "ret5": log_close.diff(5),
            "ret10": log_close.diff(10),
            "gap": base["gap"],
            "intraday": base["intraday"],
            "range": base["range"],
            "volume_change": np.log1p(base["volume"]).diff(),
            "vol10": ret.rolling(10, min_periods=5).std(ddof=0),
            "vol40": ret.rolling(40, min_periods=20).std(ddof=0),
        },
        index=base["close"].index,
    )
    return frame.replace([np.inf, -np.inf], np.nan).reindex(pd.DatetimeIndex(panel["formation_date"]))


def _ridge_fit(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray, alpha: float) -> np.ndarray:
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=alpha)
    model.fit(x, y)
    return model.predict(eval_x)


def _supervised_variant(version: str, config: dict[str, Any], base: dict[str, pd.Series], panel: pd.DataFrame, side: int) -> pd.Series:
    method = LOGIC_BY_VERSION[version].method_key
    features = _model_features(base, panel)
    # Keep a compact lag matrix for the autoregressive versions.
    ret = base["open_ret"]
    lag_frame = pd.DataFrame({f"lag_{lag}": ret.shift(lag) for lag in range(1, 6)}, index=ret.index)
    if method in {"setar", "star", "qar"}:
        features = lag_frame.reindex(features.index)

    if method == "gam":
        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import SplineTransformer

            basis = int(config["basis_dimension"])
            transformer = SplineTransformer(n_knots=basis, degree=3, include_bias=False)
            return Ridge(alpha=float(config["smoothing_penalty"])).fit(transformer.fit_transform(x), y).predict(transformer.transform(eval_x))
    elif method == "quantile_regression":
        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.linear_model import QuantileRegressor

            model = QuantileRegressor(quantile=float(config["tau"]), alpha=float(config["l2"]), solver="highs")
            model.fit(x, y)
            return model.predict(eval_x)
    elif method == "expectile":
        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            alpha = float(config["ridge"])
            weights = np.ones(len(y))
            beta = np.zeros(x.shape[1])
            for _ in range(20):
                design = np.column_stack([np.ones(len(x)), x])
                weighted = design * np.sqrt(weights)[:, None]
                target = y * np.sqrt(weights)
                gram = weighted.T @ weighted + alpha * np.eye(design.shape[1])
                beta = np.linalg.solve(gram, weighted.T @ target)
                residual = y - design @ beta
                tau = float(config["tau"])
                weights = np.where(residual >= 0.0, tau, 1.0 - tau)
            return np.column_stack([np.ones(len(eval_x)), eval_x]) @ beta
    elif method == "elastic_net":
        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.linear_model import ElasticNet

            return ElasticNet(alpha=float(config["alpha"]), l1_ratio=float(config["l1_ratio"]), max_iter=2000, random_state=SEED).fit(x, y).predict(eval_x)
    elif method == "gaussian_process":
        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, RationalQuadratic, WhiteKernel

            kernel_name = config["kernel"]
            if kernel_name == "rbf":
                core = RBF(length_scale=1.0)
            elif kernel_name == "matern15":
                core = Matern(length_scale=1.0, nu=1.5)
            elif kernel_name == "matern25":
                core = Matern(length_scale=1.0, nu=2.5)
            else:
                core = RationalQuadratic(length_scale=1.0, alpha=1.0)
            n = min(len(x), 350)
            return GaussianProcessRegressor(kernel=ConstantKernel(1.0) * core + WhiteKernel(noise_level=float(config["noise"])), normalize_y=True, optimizer=None, random_state=SEED).fit(x[-n:], y[-n:]).predict(eval_x)
    elif method == "svr":
        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.svm import SVR

            gamma = float(config["gamma_multiplier"]) / max(x.shape[1], 1)
            return SVR(C=float(config["C"]), gamma=gamma, epsilon=0.1).fit(x, y).predict(eval_x)
    elif method == "random_forest":
        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(n_estimators=80, max_features="sqrt", max_depth=int(config["max_depth"]), min_samples_leaf=int(config["min_samples_leaf"]), random_state=SEED, n_jobs=1).fit(x, y).predict(eval_x)
    elif method == "rda":
        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            classes = np.array([0, 1])
            labels = (y > 0).astype(int)
            means = []
            covariances = []
            global_cov = np.cov(x, rowvar=False) + 1e-5 * np.eye(x.shape[1])
            shrink = float(config["covariance_shrinkage"])
            for cls in classes:
                subset = x[labels == cls]
                if len(subset) < 3:
                    subset = x
                means.append(subset.mean(axis=0))
                cov = np.cov(subset, rowvar=False) if len(subset) > 2 else global_cov
                cov = np.atleast_2d(cov) + 1e-5 * np.eye(x.shape[1])
                pooled = float(config["class_pooling"]) * global_cov + (1.0 - float(config["class_pooling"])) * cov
                covariances.append(shrink * np.diag(np.diag(pooled)) + (1.0 - shrink) * pooled)
            scores = []
            for cls in classes:
                inv = np.linalg.pinv(covariances[cls])
                delta = eval_x - means[cls]
                scores.append(-0.5 * np.einsum("ij,jk,ik->i", delta, inv, delta) - 0.5 * np.log(max(np.linalg.det(covariances[cls]), 1e-12)) + np.log(max(np.mean(labels == cls), 1e-4)))
            probability = 1.0 / (1.0 + np.exp(np.clip(scores[0] - scores[1], -50.0, 50.0)))
            return 2.0 * probability - 1.0
    elif method == "setar":
        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.linear_model import Ridge

            lag = int(config["ar_lag"])
            threshold = np.quantile(x[:, 0], float(config["threshold_quantile"]))
            pred = np.zeros(len(eval_x))
            for is_low in (True, False):
                regime = x[:, 0] <= threshold if is_low else x[:, 0] > threshold
                model = Ridge(alpha=1.0).fit(x[regime, :lag], y[regime]) if regime.sum() >= 10 else Ridge(alpha=1.0).fit(x[:, :lag], y)
                mask = eval_x[:, 0] <= threshold if is_low else eval_x[:, 0] > threshold
                pred[mask] = model.predict(eval_x[mask, :lag])
            return pred
    elif method == "star":
        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.linear_model import Ridge

            lag = int(config["ar_lag"])
            threshold = np.median(x[:, 0])
            smooth = float(config["smoothness"])
            low = Ridge(alpha=1.0).fit(x[:, :lag], y)
            high = Ridge(alpha=1.0).fit(x[:, :lag], y)
            weight = 1.0 / (1.0 + np.exp(-smooth * (eval_x[:, 0] - threshold) / max(np.std(x[:, 0]), 1e-8)))
            return (1.0 - weight) * low.predict(eval_x[:, :lag]) + weight * high.predict(eval_x[:, :lag])
    elif method == "qar":
        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.linear_model import QuantileRegressor

            model = QuantileRegressor(quantile=float(config["tau"]), alpha=0.01, solver="highs")
            model.fit(x[:, : int(config["ar_lag"])], y)
            return model.predict(eval_x[:, : int(config["ar_lag"])])
    elif method == "split_conformal":
        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.linear_model import Ridge

            n = max(20, int(len(x) * 0.7))
            model = Ridge(alpha=float(config["ridge"])).fit(x[:n], y[:n])
            calibration = np.abs(y[n:] - model.predict(x[n:]))
            radius = np.quantile(calibration, 1.0 - float(config["miscoverage"])) if len(calibration) else np.nanstd(y)
            return model.predict(eval_x) - radius
    elif method == "sparse_varx":
        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.linear_model import Ridge

            return Ridge(alpha=float(config["structured_penalty"])).fit(x, y).predict(eval_x)
    elif method == "echo_state":
        # Build the fixed reservoir once per candidate. Rebuilding a 200x200
        # recurrent matrix and its eigendecomposition for every annual fit
        # made the causal implementation needlessly slow while changing no
        # model information.
        size = int(config["reservoir_size"])
        rng = np.random.default_rng(SEED + size)
        weights = rng.normal(0.0, 1.0, (size, features.shape[1])) / math.sqrt(features.shape[1])
        recurrent = rng.normal(0.0, 1.0, (size, size))
        eig = max(np.max(np.abs(np.linalg.eigvals(recurrent))), 1e-8)
        recurrent = recurrent * (float(config["spectral_radius"]) / eig)

        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.linear_model import Ridge

            def states(values: np.ndarray) -> np.ndarray:
                state = np.zeros(size)
                out = []
                for row in values:
                    state = np.tanh(weights @ row + recurrent @ state)
                    out.append(state.copy())
                return np.asarray(out)
            train_state = states(x)
            eval_state = states(eval_x)
            return Ridge(alpha=10.0).fit(train_state, y).predict(eval_state)
    else:  # projection pursuit
        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            rng = np.random.default_rng(SEED + int(config.get("ridge_function_count", 1)))
            count = int(config["ridge_function_count"])
            span = float(config["smoother_span"])
            predictions = np.zeros(len(eval_x))
            for _ in range(count):
                projection = rng.normal(size=x.shape[1])
                projection /= np.linalg.norm(projection).clip(1e-8)
                train_p = x @ projection
                eval_p = eval_x @ projection
                bandwidth = max(np.std(train_p) * span, 1e-6)
                for i, value in enumerate(eval_p):
                    weights = np.exp(-0.5 * ((train_p - value) / bandwidth) ** 2)
                    predictions[i] += np.average(y, weights=weights)
            return predictions / max(count, 1)
    return _annual_predictions(features, panel, side, fit_predict)


def _unsupervised_scores(version: str, base: dict[str, pd.Series], panel: pd.DataFrame) -> pd.DataFrame:
    method = LOGIC_BY_VERSION[version].method_key
    frame = _feature_frame(base)
    ret = base["ret"].to_numpy(dtype=float)
    log_close = base["log_close"].to_numpy(dtype=float)
    open_ret = base["open_ret"].to_numpy(dtype=float)
    index = frame.index
    output = pd.DataFrame(index=index)
    configs = SCORE_VARIANTS_BY_VERSION[version]
    for number, config in enumerate(configs):
        values = np.full(len(index), np.nan, dtype=float)
        for i in range(len(index)):
            try:
                if method in {"kernel_mmd", "energy_distance", "sliced_wasserstein", "rulsif", "graph_scan"}:
                    windows = _previous_two_windows(frame, i, int(config["window"]))
                    if windows is None:
                        continue
                    previous, recent = windows
                    if method == "kernel_mmd":
                        values[i] = _mmd_score(previous, recent, float(config["bandwidth_multiplier"]))
                    elif method == "energy_distance":
                        values[i] = _energy_score(previous, recent, float(config["distance_exponent"]))
                    elif method == "sliced_wasserstein":
                        values[i] = _sliced_wasserstein(previous, recent, int(config["projection_count"]))
                    elif method == "rulsif":
                        values[i] = _rulsif_score(previous, recent, float(config["relative_alpha"]))
                    else:
                        values[i] = _graph_scan_score(previous, recent, int(config["knn_k"]))
                elif method == "permutation_entropy":
                    window = max(60, 10 * int(config["embedding_m"]) * int(config["delay"]))
                    if i + 1 < window:
                        continue
                    entropy, bias = _ordinal_entropy(ret[i - window + 1 : i + 1], int(config["embedding_m"]), int(config["delay"]))
                    values[i] = bias * (1.0 - entropy)
                elif method == "rqa":
                    window = max(80, 10 * int(config["embedding_m"]) * int(config["delay"]))
                    if i + 1 >= window:
                        values[i] = _rqa_score(ret[i - window + 1 : i + 1], int(config["embedding_m"]), int(config["delay"]))
                elif method == "visibility_graph":
                    window = min(int(config["window"]), 80)
                    if i + 1 >= window:
                        path = log_close[i - window + 1 : i + 1]
                        degree = _visibility_endpoint(path)
                        reverse = _visibility_endpoint(-path)
                        statistic = config["statistic"]
                        values[i] = degree if statistic == "end_degree" else degree - reverse if statistic == "degree_asymmetry" else np.sign(path[-1] - path[0]) * abs(degree - reverse) if statistic == "kl_irreversibility" else degree * np.sign(path[-1] - path[0])
                elif method == "matrix_profile":
                    values[i] = _matrix_profile_score(log_close[: i + 1], int(config["subsequence_length"]), int(config["neighbor_count"]))
                elif method == "sax":
                    window = int(config["window"])
                    if i == 0:
                        # The complete column is filled once below; this branch
                        # avoids repeating the one-pass table for every date.
                        values = _sax_scores(log_close, ret, window, int(config["alphabet_size"]))
                        break
                elif method == "transfer_entropy":
                    # The entropy window must end at the current formation row.
                    # The previous implementation repeatedly passed the full
                    # frame, making all dates equal to the last-window value
                    # (and therefore both non-causal and degenerate).
                    window = 120
                    if i + 1 >= window:
                        causal_frame = frame.iloc[i - window + 1 : i + 1]
                        values[i] = _transfer_entropy(causal_frame, "ret", str(config["driver"]), int(config["driver_lag"]), window)
                elif method == "directional_change":
                    window = int(config["volatility_window"])
                    if i + 1 >= window:
                        sigma = np.nanstd(ret[i - window + 1 : i])
                        values[i] = _directional_change(log_close[: i + 1], float(config["event_threshold_sigma"]) * max(sigma, 1e-8))
                elif method == "hawkes":
                    window = 120
                    if i + 1 >= window:
                        values[i] = _hawkes_score(ret[i - window + 1 : i + 1], float(config["event_threshold_sigma"]), float(config["half_life"]))
                elif method == "l1_trend_filter":
                    window = int(config["window"])
                    if i + 1 >= window:
                        values[i] = _l1_trend_score(log_close[i - window + 1 : i + 1], float(config["lambda_scale"]))
                elif method == "ssa":
                    window = min(int(config["window"]), 80)
                    if i + 1 >= window:
                        values[i] = _ssa_score(log_close[i - window + 1 : i + 1], int(config["rank"]))
                elif method == "wavelet":
                    window = 2 ** (int(config["level"]) + 4)
                    if i + 1 >= window:
                        values[i] = _wavelet_score(log_close[i - window + 1 : i + 1], str(config["wavelet"]), int(config["level"]))
                elif method == "dmd":
                    window = 60
                    if i + 1 >= window:
                        values[i] = _dmd_score(ret[i - window + 1 : i + 1], int(config["delay_depth"]), int(config["rank"]))
                elif method == "path_signature":
                    window = int(config["path_length"])
                    if i + 1 >= window:
                        values[i] = _signature_score(frame.iloc[i - window + 1 : i + 1, [0, 4, 3]].to_numpy(dtype=float), int(config["signature_order"]))
                elif method == "persistent_homology":
                    window = max(30, 8 * int(config["embedding_m"]) * int(config["delay"]))
                    if i + 1 >= window:
                        values[i] = _topology_score(ret[i - window + 1 : i + 1], int(config["embedding_m"]), int(config["delay"]))
                elif method == "edm_simplex":
                    window = 80
                    if i + 1 >= window:
                        values[i] = _edm_score(ret[i - window + 1 : i + 1], int(config["embedding_e"]), int(config["neighbor_multiplier"]))
                elif method == "hsmm":
                    window = min(240, max(40, int(config["max_duration"])))
                    if i + 1 >= window:
                        values[i] = _duration_score(ret[i - window + 1 : i + 1], panel["state"].to_numpy(dtype=float), int(config["state_count"]), int(config["max_duration"]))
                elif method == "fine_gray":
                    # This branch is replaced below by a panel-indexed score.
                    continue
                elif method == "aalen":
                    continue
                elif method == "gas":
                    window = 120
                    if i + 1 >= window:
                        persistence = float(config["persistence"])
                        score = 0.0
                        location = 0.0
                        scale = max(np.nanstd(ret[i - window + 1 : i]), 1e-6)
                        for value in ret[i - window + 1 : i + 1]:
                            z = (value - location) / scale
                            score = persistence * score + (1.0 - persistence) * z
                            location = persistence * location + (1.0 - persistence) * value
                        values[i] = score
                elif method == "conformal_martingale":
                    window = int(config["calibration_window"])
                    if i + 1 >= window:
                        z = np.abs(ret[i - window + 1 : i + 1] - np.nanmean(ret[i - window + 1 : i + 1])) / max(np.nanstd(ret[i - window + 1 : i + 1]), 1e-8)
                        p = 1.0 / (1.0 + np.argsort(np.argsort(z))[-1])
                        epsilon = 0.7 if config["betting_epsilon"] == "mixture" else float(config["betting_epsilon"])
                        values[i] = np.sign(ret[i]) * np.log(max(1e-8, 1.0 + epsilon * (0.5 - p)))
                else:
                    continue
            except (FloatingPointError, ValueError, IndexError, np.linalg.LinAlgError):
                values[i] = np.nan
        output[f"score_{number:02d}"] = values

    if method in {"fine_gray", "aalen"}:
        output = _risk_scores(version, panel)
    return output.reindex(pd.DatetimeIndex(panel["formation_date"]))


def _risk_scores(version: str, panel: pd.DataFrame) -> pd.DataFrame:
    method = LOGIC_BY_VERSION[version].method_key
    dates = pd.DatetimeIndex(panel["formation_date"])
    data = panel.copy().set_index("formation_date").sort_index()
    zero = data["state"].eq(0)
    event_side = data["next_frozen_state"]
    age = data["state_age"].astype(float)
    momentum = np.log(data["close"].astype(float)).diff(5)
    outputs = pd.DataFrame(index=dates)
    for number, config in enumerate(SCORE_VARIANTS_BY_VERSION[version]):
        values = np.full(len(data), np.nan, dtype=float)
        horizon = int(config["horizon"])
        for i, date in enumerate(data.index):
            if date < pd.Timestamp("2018-01-01") or not zero.iloc[i]:
                continue
            # The frozen Test begins at 2025-01-01.  Once that boundary is
            # reached, risk-rate fits remain permanently cut at the frozen
            # selection end; prior Test labels must not enter later Test
            # scores or any downstream audit.
            train_cutoff = min(date, SELECTION_END)
            historical = data.index < date
            historical &= data.index <= train_cutoff
            train = data.loc[historical & zero]
            if len(train) < 40:
                continue
            if method == "fine_gray":
                at_age = train["state_age"].astype(int).clip(upper=80)
                target = train["next_frozen_state"].eq(-1)
                # Cause-specific cumulative incidence through the requested
                # horizon, with the opposite side retained as a competing event.
                same = train.loc[at_age <= at_age.iloc[-1] + horizon]
                event = same["next_frozen_state"].eq(1)
                competing = same["next_frozen_state"].eq(-1)
                values[i] = float((event.mean() - 0.5 * competing.mean()) * (1.0 - math.exp(-at_age.iloc[-1] / max(horizon, 1))))
            else:
                x = pd.DataFrame({"const": 1.0, "age": train["state_age"].astype(float), "momentum": np.log(train["close"].astype(float)).diff(5).fillna(0.0), "vol": train["o2o_h1"].rolling(10, min_periods=2).std().fillna(0.0)})
                y = train["next_frozen_state"].eq(1).astype(float).to_numpy()
                x_values = x.to_numpy(dtype=float)
                penalty = float(config["ridge"])
                beta = np.linalg.solve(x_values.T @ x_values + penalty * np.eye(x_values.shape[1]), x_values.T @ y)
                current = np.array([1.0, float(age.iloc[i]), float(momentum.iloc[i] if np.isfinite(momentum.iloc[i]) else 0.0), float(np.nanstd(data["o2o_h1"].iloc[max(0, i - 10) : i + 1]))])
                values[i] = float(current @ beta)
        outputs[f"score_{number:02d}"] = values
    return outputs


def compute_reserve_scores(version: str, spot: pd.DataFrame, research_panel: pd.DataFrame, side: int) -> pd.DataFrame:
    """Compute V51--V90 scores; larger values support ``side``.

    V91--V120 live in a separate module so their paper-derived transformations
    cannot accidentally fall through to an older reserve implementation.
    """

    if version not in LOGIC_BY_VERSION or int(version[1:]) < 51:
        raise ValueError("compute_reserve_scores accepts V51-V90 only")
    if side not in (-1, 1):
        raise ValueError("side must be -1 or 1")
    if int(version[1:]) >= 91:
        from .advanced_features import compute_advanced_scores

        return compute_advanced_scores(version, spot, research_panel, side)
    base = _base(spot)
    method = LOGIC_BY_VERSION[version].method_key
    if method in {"gam", "quantile_regression", "expectile", "elastic_net", "gaussian_process", "svr", "random_forest", "rda", "setar", "star", "qar", "split_conformal", "sparse_varx", "echo_state", "projection_pursuit"}:
        output = pd.DataFrame(index=pd.DatetimeIndex(research_panel["formation_date"]))
        for number, config in enumerate(SCORE_VARIANTS_BY_VERSION[version]):
            # Compute each variant separately to preserve the exact parameter
            # record in the result and independent per-version freeze.
            output[f"score_{number:02d}"] = _supervised_variant(version, config, base, research_panel, side).to_numpy()
        return output
    output = _unsupervised_scores(version, base, research_panel)
    return output * float(side)
