"""Executable causal adaptations for the V91--V120 research reserve.

The implementations here are intentionally compact, deterministic adaptations
to one daily spot series.  They preserve the mathematical core named in the
registry, use only observations available at formation date ``t``, and expose
16 preregistered score variants per version.  A method is never silently
replaced by a generic momentum score: numerical failures become NaN and are
reported by the existing runner as a non-computable candidate.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from .design_registry import SCORE_VARIANTS_BY_VERSION
from .logic_features import _base
from .logic_registry import LOGIC_BY_VERSION


SEED = 15452026
SELECTION_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")


def _safe_scale(values: np.ndarray, fallback: float = 1.0) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float(fallback)
    scale = float(np.nanmedian(np.abs(finite - np.nanmedian(finite))) * 1.4826)
    if not np.isfinite(scale) or scale <= 1e-10:
        scale = float(np.nanstd(finite))
    return float(scale if np.isfinite(scale) and scale > 1e-10 else fallback)


def _z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    center = np.nanmedian(values)
    scale = _safe_scale(values)
    return (values - center) / scale


def _window(values: np.ndarray, i: int, width: int) -> np.ndarray | None:
    if i + 1 < width:
        return None
    out = np.asarray(values[i - width + 1 : i + 1], dtype=float)
    return out if np.isfinite(out).all() else None


def _ret_frame(base: dict[str, pd.Series]) -> pd.DataFrame:
    ret = base["ret"]
    log_close = base["log_close"]
    volume = np.log1p(base["volume"])
    frame = pd.DataFrame(
        {
            "ret1": ret,
            "ret2": log_close.diff(2),
            "ret3": log_close.diff(3),
            "ret5": log_close.diff(5),
            "ret10": log_close.diff(10),
            "gap": base["gap"],
            "intraday": base["intraday"],
            "range": base["range"],
            "vol5": ret.rolling(5, min_periods=3).std(ddof=0),
            "vol20": ret.rolling(20, min_periods=8).std(ddof=0),
            "vol60": ret.rolling(60, min_periods=20).std(ddof=0),
            "volume_change": volume.diff(),
            "amount_change": np.log1p(base["amount"]).diff(),
            "location": base["location"],
        },
        index=base["close"].index,
    )
    return frame.replace([np.inf, -np.inf], np.nan)


def _garch_path(ret: np.ndarray, omega_scale: float, persistence: float, mode: str, gamma: float = 0.0, power: float = 1.0) -> np.ndarray:
    """One-pass conditional scale and signed innovation path."""

    ret = np.asarray(ret, dtype=float)
    initial = max(float(np.nanvar(ret[: min(120, len(ret))])), 1e-8)
    omega = max(1e-10, (1.0 - persistence) * omega_scale * initial)
    variance = initial
    log_variance = math.log(initial)
    output = np.full(len(ret), np.nan, dtype=float)
    for i, value in enumerate(ret):
        if not np.isfinite(value):
            continue
        sigma = math.sqrt(max(variance, 1e-12))
        innovation = value / sigma
        if mode == "garch":
            variance = omega + persistence * variance + (1.0 - persistence) * value * value
            output[i] = math.tanh(innovation) * (1.0 + 0.15 * math.tanh((variance / initial) - 1.0))
        elif mode == "egarch":
            centered_abs = abs(innovation) - math.sqrt(2.0 / math.pi)
            log_variance = math.log(omega) + persistence * log_variance + centered_abs + gamma * innovation
            variance = float(np.clip(math.exp(np.clip(log_variance, -30.0, 30.0)), 1e-12, 1e6))
            output[i] = math.tanh(innovation + gamma * abs(innovation) * np.sign(innovation))
        elif mode == "gjr_garch":
            leverage = gamma * value * value if value < 0.0 else 0.0
            variance = omega + persistence * variance + (1.0 - persistence) * value * value + leverage
            output[i] = math.tanh(innovation) * (1.0 + 0.15 * (1.0 if value < 0 else -1.0) * math.tanh(variance / initial - 1.0))
        else:  # APARCH
            scale = max(variance, 1e-12) ** (0.5 * power)
            innovation_power = (abs(value) - gamma * value) ** power
            variance = max(1e-12, omega + persistence * variance + (1.0 - persistence) * innovation_power)
            output[i] = np.sign(value) * math.tanh(abs(value) ** power / max(scale, 1e-8))
    return output


def _figarch_score(ret: np.ndarray, fractional_d: float, memory_length: int) -> np.ndarray:
    ret = np.asarray(ret, dtype=float)
    output = np.full(len(ret), np.nan, dtype=float)
    weights = np.arange(1, memory_length + 1, dtype=float) ** (fractional_d - 1.0)
    weights /= weights.sum()
    squared = np.where(np.isfinite(ret), ret * ret, 0.0)
    for i in range(len(ret)):
        if not np.isfinite(ret[i]) or i < 5:
            continue
        left = max(0, i - memory_length + 1)
        w = weights[-(i - left + 1) :]
        past = squared[left : i + 1]
        variance = float(np.dot(w, past) / max(w.sum(), 1e-12))
        output[i] = math.tanh(ret[i] / math.sqrt(max(variance, 1e-12)))
    return output


def _har_score(ret: np.ndarray, range_values: np.ndarray, weekly_weight: float, monthly_weight: float) -> np.ndarray:
    realized = np.maximum(ret * ret + 0.25 * range_values * range_values, 0.0)
    daily = pd.Series(realized).rolling(1, min_periods=1).mean().to_numpy()
    weekly = pd.Series(realized).rolling(5, min_periods=3).mean().to_numpy()
    monthly = pd.Series(realized).rolling(22, min_periods=10).mean().to_numpy()
    forecast = daily + weekly_weight * weekly + monthly_weight * monthly
    baseline = pd.Series(realized).rolling(60, min_periods=20).median().to_numpy()
    pressure = np.log((forecast + 1e-12) / (baseline + 1e-12))
    # Use a trailing robust scale.  The previous full-panel _z(ret) allowed
    # Test-era observations to change every earlier HAR score.
    ret_series = pd.Series(ret, dtype=float)
    center = ret_series.rolling(120, min_periods=20).median()
    scale = ret_series.rolling(120, min_periods=20).apply(_safe_scale, raw=True)
    causal_z = ((ret_series - center) / scale.replace(0.0, np.nan)).to_numpy(dtype=float)
    return np.tanh(causal_z * (1.0 + 0.25 * np.tanh(pressure)))


def _caviar_score(ret: np.ndarray, tau: float, persistence: float) -> np.ndarray:
    ret = np.asarray(ret, dtype=float)
    scale = max(_safe_scale(ret[: min(120, len(ret))]), 1e-5)
    q = float(np.nanquantile(ret[: min(120, len(ret))], tau)) if np.isfinite(ret[: min(120, len(ret))]).any() else 0.0
    out = np.full(len(ret), np.nan, dtype=float)
    for i, value in enumerate(ret):
        if not np.isfinite(value):
            continue
        q = persistence * q + (1.0 - persistence) * (q + scale * (tau - float(value <= q)))
        out[i] = math.tanh((value - q) / max(scale, 1e-8))
        scale = 0.995 * scale + 0.005 * abs(value - q)
    return out


def _novas_score(ret: np.ndarray, volatility_window: int, shock_clip: float) -> np.ndarray:
    ret = np.asarray(ret, dtype=float)
    rolling = pd.Series(ret).rolling(volatility_window, min_periods=max(3, volatility_window // 2)).std(ddof=0).to_numpy()
    normalized = ret / np.maximum(rolling, 1e-8)
    return np.tanh(np.clip(normalized, -shock_clip, shock_clip) / max(shock_clip, 1e-8))


def _evt_score(ret: np.ndarray, threshold_quantile: float, tail_window: int) -> np.ndarray:
    ret = np.asarray(ret, dtype=float)
    out = np.full(len(ret), np.nan, dtype=float)
    for i in range(len(ret)):
        values = _window(ret, i, tail_window)
        if values is None or len(values) < 20:
            continue
        threshold = float(np.quantile(np.abs(values[:-1]), threshold_quantile))
        upper = values[values > threshold] - threshold
        lower = -values[values < -threshold] - threshold
        upper_mean = float(np.mean(upper)) if len(upper) else 0.0
        lower_mean = float(np.mean(lower)) if len(lower) else 0.0
        current = values[-1]
        out[i] = math.tanh(np.sign(current) * (upper_mean - lower_mean) / max(_safe_scale(values), 1e-8))
    return out


def _hill_index(values: np.ndarray, fraction: float) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 8:
        return np.nan
    k = max(3, min(len(values) // 2, int(len(values) * fraction)))
    tail = values[-k:]
    threshold = max(float(tail[0]), 1e-12)
    return float(k / max(np.sum(np.log(tail / threshold)), 1e-8))


def _hill_score(ret: np.ndarray, fraction: float, window: int) -> np.ndarray:
    out = np.full(len(ret), np.nan, dtype=float)
    for i in range(len(ret)):
        values = _window(ret, i, window)
        if values is None or len(values) < 20:
            continue
        positive = _hill_index(values[values > 0], fraction)
        negative = _hill_index(-values[values < 0], fraction)
        if not np.isfinite(positive) or not np.isfinite(negative):
            continue
        direction = np.sign(np.mean(values[-min(5, len(values)) :]))
        out[i] = float(direction * np.tanh((negative - positive) / max(positive + negative, 1e-8)))
    return out


def _embedding(values: np.ndarray, embedding: int) -> np.ndarray:
    if len(values) < embedding:
        return np.empty((0, embedding))
    return np.vstack([values[i - embedding + 1 : i + 1] for i in range(embedding - 1, len(values))])


def _smap_score(ret: np.ndarray, embedding: int, theta: float) -> np.ndarray:
    out = np.full(len(ret), np.nan, dtype=float)
    if len(ret) < embedding + 30:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(ret, embedding)
    for i in range(embedding + 25, len(ret) - 1):
        current = windows[i - embedding + 1]
        # Endpoints through i-2 have a known next return and are the only legal
        # library rows.  Vectorization keeps the S-map distinct without an
        # accidental O(T^2) Python inner loop.
        left = max(0, i - embedding - 180)
        right = i - embedding
        x = windows[left:right]
        y = ret[left + embedding : right + embedding]
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < embedding + 3:
            continue
        distances = np.linalg.norm(x - current[None, :], axis=1)
        scale = max(float(np.median(distances)), 1e-8)
        weights = np.exp(-theta * distances / scale)
        design = np.column_stack([np.ones(len(x)), x])
        ridge = 1e-4 * np.eye(design.shape[1])
        beta = np.linalg.solve(design.T @ (weights[:, None] * design) + ridge, design.T @ (weights * y))
        out[i] = float(np.tanh(beta[0] + current @ beta[1:]))
    return out


def _analog_score(ret: np.ndarray, embedding: int, neighbors: int) -> np.ndarray:
    out = np.full(len(ret), np.nan, dtype=float)
    if len(ret) < embedding + 30:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(ret, embedding)
    for i in range(embedding + 25, len(ret) - 1):
        current = windows[i - embedding + 1]
        current = (current - current.mean()) / max(current.std(ddof=0), 1e-8)
        left = max(0, i - embedding - 240)
        right = i - embedding
        rows = windows[left:right]
        targets = ret[left + embedding : right + embedding]
        valid = np.isfinite(rows).all(axis=1) & np.isfinite(targets)
        rows, targets = rows[valid], targets[valid]
        if len(rows) < 3:
            continue
        rows = (rows - rows.mean(axis=1, keepdims=True)) / np.maximum(rows.std(axis=1, keepdims=True), 1e-8)
        distances = np.linalg.norm(rows - current[None, :], axis=1)
        chosen = np.argsort(distances)[: min(neighbors, len(distances))]
        weights = 1.0 / (1e-6 + distances[chosen])
        out[i] = float(np.tanh(np.average(targets[chosen], weights=weights) / max(_safe_scale(ret[max(0, i - 60) : i + 1]), 1e-8)))
    return out


def _shapelet_score(ret: np.ndarray, length: int, count: int) -> np.ndarray:
    """Causal shapelet transform: choose historical low-distance prototypes."""

    out = np.full(len(ret), np.nan, dtype=float)
    if len(ret) < 3 * length + 5:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(ret, length)
    for i in range(3 * length + 5, len(ret) - 1):
        current = windows[i - length + 1]
        current = (current - current.mean()) / max(current.std(ddof=0), 1e-8)
        left = max(0, i - length - 360)
        right = i - length
        shapes = windows[left:right]
        targets = ret[left + length : right + length]
        valid = np.isfinite(shapes).all(axis=1) & np.isfinite(targets)
        shapes, targets = shapes[valid], targets[valid]
        if not len(shapes):
            continue
        shapes = (shapes - shapes.mean(axis=1, keepdims=True)) / np.maximum(shapes.std(axis=1, keepdims=True), 1e-8)
        distances = np.sqrt(np.mean((shapes - current[None, :]) ** 2, axis=1))
        chosen = np.argsort(distances)[: min(count, len(distances))]
        weights = 1.0 / (1e-6 + distances[chosen])
        out[i] = float(np.tanh(np.average(targets[chosen], weights=weights) / max(_safe_scale(ret[max(0, i - 60) : i + 1]), 1e-8)))
    return out


def _sample_entropy(values: np.ndarray, m: int, tolerance_scale: float) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < m + 4 or not np.isfinite(values).all():
        return np.nan
    tol = tolerance_scale * _safe_scale(values)
    counts = []
    for order in (m, m + 1):
        templates = np.vstack([values[i : i + order] for i in range(len(values) - order + 1)])
        # The exact pair count is quadratic in the short window.  Subsampling
        # at most 64 templates preserves the finite-sample entropy ordering but
        # makes the 16-variant grid practical for a multi-version run.
        if len(templates) > 64:
            take = np.linspace(0, len(templates) - 1, 64).round().astype(int)
            templates = templates[take]
        distance_matrix = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
        upper = distance_matrix[np.triu_indices(len(templates), k=1)]
        count = int(np.sum(upper <= tol))
        total = int(len(upper))
        counts.append(count / max(total, 1))
    # Finite-sample bias correction keeps rare high-order patterns measurable
    # instead of turning an entire parameter axis into an all-NaN degenerate
    # rule.  The correction is internal to Sample Entropy, not a fallback
    # momentum signal.
    counts = [max(float(value), 1.0 / (len(values) * len(values))) for value in counts]
    return float(-math.log(counts[1] / counts[0]))


def _entropy_score(ret: np.ndarray, m: int, tolerance_scale: float, multiscale: bool, max_scale: int = 2) -> np.ndarray:
    out = np.full(len(ret), np.nan, dtype=float)
    width = 60 if not multiscale else 20 * max_scale
    # Entropy is intentionally refreshed every few observations and carried
    # forward until the next refresh.  This is a causal holding of the last
    # computed complexity state (not a centered interpolation), and prevents a
    # quadratic pair-count from dominating the 30-version reserve run.
    refresh_step = 5
    for i in range(width, len(ret), refresh_step):
        values = ret[i - width + 1 : i + 1]
        if multiscale:
            entropies = []
            for scale in range(1, max_scale + 1):
                coarse = values[: len(values) // scale * scale].reshape(-1, scale).mean(axis=1)
                entropy = _sample_entropy(coarse, m, tolerance_scale)
                if np.isfinite(entropy):
                    entropies.append(entropy)
            entropy = float(np.mean(entropies)) if entropies else np.nan
        else:
            entropy = _sample_entropy(values, m, tolerance_scale)
        direction = np.sign(np.mean(values[-min(8, len(values)) :]))
        if np.isfinite(entropy):
            out[i : min(len(ret), i + refresh_step)] = float(direction * (1.0 / (1.0 + entropy)))
    return out


def _lz_complexity(values: np.ndarray, alphabet_size: int) -> float:
    if len(values) < 8:
        return np.nan
    cuts = np.quantile(values, np.linspace(0.0, 1.0, alphabet_size + 1)[1:-1])
    symbols = np.digitize(values, cuts).tolist()
    phrases: set[tuple[int, ...]] = set()
    complexity = 0
    i = 0
    while i < len(symbols):
        length = 1
        while i + length <= len(symbols) and tuple(symbols[i : i + length]) in phrases:
            length += 1
        phrase = tuple(symbols[i : i + length])
        phrases.add(phrase)
        complexity += 1
        i += length
    normalization = len(values) / max(math.log(max(len(values), 2), alphabet_size), 1.0)
    return float(complexity / normalization)


def _lz_score(ret: np.ndarray, alphabet_size: int, window: int) -> np.ndarray:
    out = np.full(len(ret), np.nan, dtype=float)
    for i in range(len(ret)):
        values = _window(ret, i, window)
        if values is None:
            continue
        complexity = _lz_complexity(values, alphabet_size)
        if np.isfinite(complexity):
            out[i] = np.sign(np.mean(values[-min(8, len(values)) :])) * (1.0 - min(complexity, 1.0))
    return out


def _higuchi(values: np.ndarray, k_max: int) -> float:
    n = len(values)
    lengths = []
    scales = []
    for k in range(1, min(k_max, n // 2) + 1):
        lk = []
        for m in range(k):
            indices = np.arange(m, n, k)
            if len(indices) < 2:
                continue
            distance = np.abs(np.diff(values[indices])).sum() * (n - 1) / (len(indices) * k)
            lk.append(distance)
        if lk and np.mean(lk) > 0:
            lengths.append(np.mean(lk))
            scales.append(k)
    if len(lengths) < 3:
        return np.nan
    slope = np.polyfit(np.log(scales), np.log(lengths), 1)[0]
    return float(-slope)


def _higuchi_score(ret: np.ndarray, k_max: int, window: int) -> np.ndarray:
    out = np.full(len(ret), np.nan, dtype=float)
    for i in range(len(ret)):
        values = _window(ret, i, window)
        if values is None:
            continue
        fd = _higuchi(values, k_max)
        if np.isfinite(fd):
            out[i] = np.sign(np.mean(values[-min(8, len(values)) :])) * np.tanh(2.0 - fd)
    return out


def _hilbert_score(ret: np.ndarray, window: int, amplitude_weight: float) -> np.ndarray:
    try:
        from scipy.signal import hilbert
    except Exception:
        return np.full(len(ret), np.nan, dtype=float)
    out = np.full(len(ret), np.nan, dtype=float)
    for i in range(len(ret)):
        values = _window(ret, i, window)
        if values is None or np.nanstd(values) <= 1e-10:
            continue
        centered = values - np.polyval(np.polyfit(np.arange(len(values)), values, 1), np.arange(len(values)))
        analytic = hilbert(centered)
        phase = np.unwrap(np.angle(analytic))
        velocity = phase[-1] - phase[-2]
        amplitude = abs(analytic[-1]) / max(np.nanstd(centered), 1e-8)
        out[i] = float(np.tanh(velocity) * (1.0 + amplitude_weight * np.tanh(amplitude - 1.0)))
    return out


def _emd_score(ret: np.ndarray, sift_passes: int, window: int) -> np.ndarray:
    out = np.full(len(ret), np.nan, dtype=float)
    for i in range(len(ret)):
        values = _window(ret, i, window)
        if values is None:
            continue
        residual = values.copy()
        for _ in range(sift_passes):
            span = max(2, window // (4 + 2 * _))
            smooth = pd.Series(residual).rolling(span, min_periods=1, center=False).mean().to_numpy()
            residual = residual - smooth + smooth[-1]
        trend = pd.Series(values).rolling(max(3, window // 4), min_periods=2).mean().to_numpy()
        trend_start = trend[np.flatnonzero(np.isfinite(trend))[0]] if np.isfinite(trend).any() else trend[-1]
        out[i] = float(np.tanh((trend[-1] - trend_start) / max(_safe_scale(values), 1e-8) + (residual[-1] - residual[0]) / max(_safe_scale(residual), 1e-8)))
    return out


def _vmd_score(ret: np.ndarray, mode_count: int, bandwidth: float, window: int) -> np.ndarray:
    out = np.full(len(ret), np.nan, dtype=float)
    for i in range(len(ret)):
        values = _window(ret, i, window)
        if values is None or np.nanstd(values) <= 1e-10:
            continue
        centered = values - values.mean()
        spectrum = np.fft.rfft(centered)
        frequencies = np.fft.rfftfreq(len(centered))
        modes = []
        for mode in range(1, mode_count + 1):
            center = mode / (mode_count + 1) * 0.5
            mask = np.exp(-0.5 * ((frequencies - center) / max(bandwidth / (10.0 * mode_count), 1e-4)) ** 2)
            filtered = np.fft.irfft(spectrum * mask, n=len(centered))
            modes.append(filtered)
        low = modes[0] if modes else centered
        high_energy = np.mean(np.asarray(modes[1:]) ** 2) if len(modes) > 1 else 0.0
        out[i] = float(np.tanh((low[-1] - low[0]) / max(_safe_scale(values), 1e-8)) / (1.0 + bandwidth * high_energy))
    return out


def _rocket_matrix(ret: np.ndarray, kernel_count: int, max_length: int, seed_offset: int) -> np.ndarray:
    """Causal PPV/max random convolution features for every endpoint."""

    n = len(ret)
    rng = np.random.default_rng(SEED + seed_offset + kernel_count + max_length)
    lengths = rng.integers(3, max(4, max_length + 1), size=kernel_count)
    weights = []
    biases = rng.normal(0.0, 0.5, size=kernel_count)
    for length in lengths:
        kernel = rng.normal(size=int(length))
        kernel -= kernel.mean()
        kernel /= max(np.linalg.norm(kernel), 1e-8)
        weights.append(kernel)
    output = np.full((n, 2 * kernel_count), np.nan, dtype=float)
    # Vectorize across endpoints.  Each convolution uses only the prefix ending
    # at the endpoint, and rolling PPV/max summaries are causal by construction.
    clean = np.where(np.isfinite(ret), ret, 0.0)
    for k, (kernel, bias) in enumerate(zip(weights, biases)):
        conv = np.convolve(clean, kernel[::-1], mode="valid") + bias
        ppv = pd.Series(conv > 0.0).rolling(max_length, min_periods=max(3, max_length // 2)).mean().to_numpy()
        maximum = pd.Series(conv).rolling(max_length, min_periods=max(3, max_length // 2)).max().to_numpy()
        start = len(kernel) - 1
        end = min(n, start + len(conv))
        output[start:end, 2 * k] = ppv[: end - start]
        output[start:end, 2 * k + 1] = maximum[: end - start]
    return output


def _model_features(base: dict[str, pd.Series], panel: pd.DataFrame) -> pd.DataFrame:
    return _ret_frame(base).reindex(pd.DatetimeIndex(panel["formation_date"]))


def _annual_predictions(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    side: int,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    target_kind: str = "return",
) -> pd.Series:
    dates = pd.DatetimeIndex(features.index)
    panel_indexed = panel.copy().set_index("formation_date").reindex(dates)
    if target_kind == "hazard":
        target = panel_indexed["next_frozen_state"].eq(side).astype(float)
    else:
        target = side * pd.to_numeric(panel_indexed["o2o_h1"], errors="coerce")
    zero = panel_indexed["state"].eq(0)
    output = pd.Series(np.nan, index=dates, dtype=float)
    for year in sorted(set(dates.year)):
        start = pd.Timestamp(f"{year}-01-01")
        eval_mask = dates.year == year
        cutoff = SELECTION_END if start >= TEST_START else start - pd.Timedelta(days=1)
        train_mask = (dates < start) & (dates <= cutoff) & zero.to_numpy() & target.notna().to_numpy()
        if int(train_mask.sum()) < 60:
            continue
        x_train = features.loc[train_mask].to_numpy(dtype=float)
        y_train = target.to_numpy(dtype=float)[train_mask]
        x_eval = features.loc[eval_mask].to_numpy(dtype=float)
        medians = np.nanmedian(x_train, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        scales = np.nanstd(x_train, axis=0)
        scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0)
        x_train = np.where(np.isfinite(x_train), x_train, medians)
        x_eval = np.where(np.isfinite(x_eval), x_eval, medians)
        x_train = (x_train - medians) / scales
        x_eval = (x_eval - medians) / scales
        try:
            prediction = fit_predict(x_train, y_train, x_eval)
            output.loc[eval_mask] = np.asarray(prediction, dtype=float)
        except Exception:
            continue
    return output


def _supervised(version: str, config: dict[str, Any], base: dict[str, pd.Series], panel: pd.DataFrame, side: int) -> pd.Series:
    method = LOGIC_BY_VERSION[version].method_key
    features = _model_features(base, panel)
    ret = base["ret"].reindex(features.index)

    if method == "rocket":
        count = int(config["kernel_count"])
        max_length = int(config["max_kernel_length"])
        matrix = _rocket_matrix(ret.to_numpy(dtype=float), count, max_length, int(config["reserve_variant"]))
        features = pd.DataFrame(matrix, index=features.index)

        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.linear_model import Ridge

            return Ridge(alpha=10.0).fit(x, y).predict(eval_x)
    elif method == "elm":
        hidden = int(config["hidden_units"])
        activation = str(config["activation"])
        rng = np.random.default_rng(SEED + hidden + int(config["reserve_variant"]))
        weights = rng.normal(0.0, 1.0 / math.sqrt(features.shape[1]), (features.shape[1], hidden))
        bias = rng.normal(0.0, 0.25, hidden)

        def transform(x: np.ndarray) -> np.ndarray:
            values = x @ weights + bias
            if activation == "relu":
                return np.maximum(values, 0.0)
            if activation == "sin":
                return np.sin(values)
            if activation == "sigmoid":
                return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))
            return np.tanh(values)

        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.linear_model import Ridge

            return Ridge(alpha=1.0).fit(transform(x), y).predict(transform(eval_x))
    elif method == "rff_ridge":
        count = int(config["feature_count"])
        rng = np.random.default_rng(SEED + count + int(config["reserve_variant"]))
        weights = rng.normal(0.0, 1.0 / max(float(config["bandwidth"]), 1e-6), (features.shape[1], count))
        phases = rng.uniform(0.0, 2.0 * math.pi, count)

        def transform(x: np.ndarray) -> np.ndarray:
            return math.sqrt(2.0 / count) * np.cos(x @ weights + phases)

        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.linear_model import Ridge

            return Ridge(alpha=1.0).fit(transform(x), y).predict(transform(eval_x))
    elif method == "pls":
        mode = str(config["scale_mode"])

        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.cross_decomposition import PLSRegression

            if mode == "none":
                x_fit, x_eval = x, eval_x
            elif mode == "robust":
                center = np.median(x, axis=0)
                scale = np.median(np.abs(x - center), axis=0) * 1.4826
                scale = np.where(scale > 1e-8, scale, 1.0)
                x_fit, x_eval = (x - center) / scale, (eval_x - center) / scale
            elif mode == "unit":
                scale = np.sqrt(np.mean(x * x, axis=0))
                scale = np.where(scale > 1e-8, scale, 1.0)
                x_fit, x_eval = x / scale, eval_x / scale
            else:
                x_fit, x_eval = x, eval_x
            model = PLSRegression(n_components=max(1, min(int(config["components"]), x_fit.shape[1], len(x_fit) - 1)), scale=False)
            model.fit(x_fit, y)
            return model.predict(x_eval).ravel()
    elif method == "huber_regression":
        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.linear_model import HuberRegressor

            return HuberRegressor(epsilon=float(config["epsilon"]), alpha=float(config["alpha"]), max_iter=300).fit(x, y).predict(eval_x)
    elif method == "ransac_trend":
        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.linear_model import LinearRegression, RANSACRegressor

            kwargs = {
                "min_samples": max(2, int(len(x) * float(config["min_samples_fraction"]))),
                "residual_threshold": float(config["residual_threshold"]),
                "random_state": SEED,
            }
            try:
                model = RANSACRegressor(estimator=LinearRegression(), **kwargs)
            except TypeError:
                model = RANSACRegressor(base_estimator=LinearRegression(), **kwargs)
            return model.fit(x, y).predict(eval_x)
    elif method == "logistic_hazard":
        weight_mode = str(config["class_weight"])
        if weight_mode == "balanced":
            class_weight: dict[int, float] | str | None = "balanced"
        elif weight_mode == "up_weighted":
            class_weight = {0: 1.0, 1: 1.5}
        elif weight_mode == "down_weighted":
            class_weight = {0: 1.5, 1: 1.0}
        else:
            class_weight = None

        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.linear_model import LogisticRegression

            labels = (y > 0.5).astype(int)
            if len(np.unique(labels)) < 2:
                return np.full(len(eval_x), float(labels[0]) if len(labels) else 0.5)
            model = LogisticRegression(C=float(config["C"]), class_weight=class_weight, max_iter=500, solver="lbfgs")
            return model.fit(x, labels).predict_proba(eval_x)[:, 1] - 0.5

        return _annual_predictions(features, panel, side, fit_predict, target_kind="hazard")
    elif method == "qda":
        prior_mode = str(config["class_prior"])

        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

            labels = (y > 0).astype(int)
            if len(np.unique(labels)) < 2:
                return np.full(len(eval_x), float(labels[0]) - 0.5 if len(labels) else 0.0)
            prior = None
            if prior_mode == "uniform":
                prior = [0.5, 0.5]
            elif prior_mode == "up_prior":
                prior = [0.35, 0.65]
            elif prior_mode == "down_prior":
                prior = [0.65, 0.35]
            model = QuadraticDiscriminantAnalysis(reg_param=float(config["reg_param"]), priors=prior)
            return model.fit(x, labels).predict_proba(eval_x)[:, 1] - 0.5
    elif method == "naive_bayes":
        prior_mode = str(config["prior_mode"])

        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.naive_bayes import GaussianNB

            labels = (y > 0).astype(int)
            if len(np.unique(labels)) < 2:
                return np.full(len(eval_x), float(labels[0]) - 0.5 if len(labels) else 0.0)
            priors = None
            if prior_mode == "uniform":
                priors = [0.5, 0.5]
            elif prior_mode == "up_prior":
                priors = [0.35, 0.65]
            elif prior_mode == "down_prior":
                priors = [0.65, 0.35]
            return GaussianNB(var_smoothing=float(config["var_smoothing"]), priors=priors).fit(x, labels).predict_proba(eval_x)[:, 1] - 0.5
    else:  # ExtraTrees is intentionally not RandomForest: split randomization differs.
        def fit_predict(x: np.ndarray, y: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
            from sklearn.ensemble import ExtraTreesRegressor

            return ExtraTreesRegressor(
                n_estimators=80,
                max_depth=int(config["max_depth"]),
                min_samples_leaf=int(config["min_samples_leaf"]),
                max_features=1.0,
                bootstrap=False,
                random_state=SEED,
                n_jobs=1,
            ).fit(x, y).predict(eval_x)
    return _annual_predictions(features, panel, side, fit_predict)


def _unsupervised(version: str, base: dict[str, pd.Series], panel: pd.DataFrame) -> pd.DataFrame:
    method = LOGIC_BY_VERSION[version].method_key
    ret = base["ret"].to_numpy(dtype=float)
    range_values = base["range"].to_numpy(dtype=float)
    close = base["log_close"].to_numpy(dtype=float)
    output = pd.DataFrame(index=pd.DatetimeIndex(panel["formation_date"]))
    configs = SCORE_VARIANTS_BY_VERSION[version]
    for number, config in enumerate(configs):
        try:
            if method == "garch":
                values = _garch_path(ret, float(config["omega_scale"]), float(config["persistence"]), "garch")
            elif method == "egarch":
                values = _garch_path(ret, 0.05, float(config["persistence"]), "egarch", float(config["leverage"]))
            elif method == "gjr_garch":
                values = _garch_path(ret, 0.05, float(config["persistence"]), "gjr_garch", float(config["threshold_gamma"]))
            elif method == "aparch":
                values = _garch_path(ret, 0.05, 0.92, "aparch", float(config["asymmetry"]), float(config["power"]))
            elif method == "figarch":
                values = _figarch_score(ret, float(config["fractional_d"]), int(config["memory_length"]))
            elif method == "har_rv":
                values = _har_score(ret, range_values, float(config["weekly_weight"]), float(config["monthly_weight"]))
            elif method == "caviar":
                values = _caviar_score(ret, float(config["tau"]), float(config["persistence"]))
            elif method == "novas":
                values = _novas_score(ret, int(config["volatility_window"]), float(config["shock_clip"]))
            elif method == "evt_pot":
                values = _evt_score(ret, float(config["threshold_quantile"]), int(config["tail_window"]))
            elif method == "hill_tail":
                values = _hill_score(ret, float(config["tail_fraction"]), int(config["window"]))
            elif method == "smap":
                values = _smap_score(ret, int(config["embedding"]), float(config["theta"]))
            elif method == "analog_knn":
                values = _analog_score(ret, int(config["embedding"]), int(config["neighbor_count"]))
            elif method == "shapelet":
                values = _shapelet_score(ret, int(config["shape_length"]), int(config["shape_count"]))
            elif method == "sample_entropy":
                values = _entropy_score(ret, int(config["embedding_m"]), float(config["tolerance_scale"]), False)
            elif method == "multiscale_entropy":
                values = _entropy_score(ret, int(config["embedding_m"]), 0.25, True, int(config["max_scale"]))
            elif method == "lz_complexity":
                values = _lz_score(ret, int(config["alphabet_size"]), int(config["window"]))
            elif method == "higuchi_fd":
                values = _higuchi_score(ret, int(config["k_max"]), int(config["window"]))
            elif method == "hilbert_phase":
                values = _hilbert_score(ret, int(config["window"]), float(config["amplitude_weight"]))
            elif method == "emd_residual":
                values = _emd_score(ret, int(config["sift_passes"]), int(config["window"]))
            elif method == "vmd":
                values = _vmd_score(ret, int(config["mode_count"]), float(config["bandwidth"]), 120)
            else:
                raise ValueError(f"{method} is not an unsupervised V91-V120 method")
            output[f"score_{number:02d}"] = values[: len(output)]
        except Exception:
            output[f"score_{number:02d}"] = np.nan
    return output


SUPERVISED_METHODS = {"rocket", "elm", "rff_ridge", "pls", "huber_regression", "ransac_trend", "logistic_hazard", "qda", "naive_bayes", "extra_trees"}


def compute_advanced_scores(version: str, spot: pd.DataFrame, research_panel: pd.DataFrame, side: int) -> pd.DataFrame:
    """Return 16 causal V91--V120 scores; larger values support ``side``."""

    if version not in LOGIC_BY_VERSION or not 91 <= int(version[1:]) <= 120:
        raise ValueError("compute_advanced_scores accepts V91-V120 only")
    base = _base(spot)
    method = LOGIC_BY_VERSION[version].method_key
    if method in SUPERVISED_METHODS:
        output = pd.DataFrame(index=pd.DatetimeIndex(research_panel["formation_date"]))
        for number, config in enumerate(SCORE_VARIANTS_BY_VERSION[version]):
            output[f"score_{number:02d}"] = _supervised(version, config, base, research_panel, side).to_numpy()
        return output
    return _unsupervised(version, base, research_panel) * float(side)
