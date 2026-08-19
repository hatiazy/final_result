from __future__ import annotations

import math

import numpy as np
import pandas as pd


EPS = 1e-12


def rolling_zscore(values: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    minp = min_periods or max(5, window // 2)
    prior = values.shift(1)
    mean = prior.rolling(window, min_periods=minp).mean()
    std = prior.rolling(window, min_periods=minp).std(ddof=0).replace(0.0, np.nan)
    return values.sub(mean).div(std)


def rolling_slope(values: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    xc = x - x.mean()
    denom = float(np.square(xc).sum())

    def calculate(raw: np.ndarray) -> float:
        if not np.isfinite(raw).all():
            return np.nan
        return float(np.dot(raw - raw.mean(), xc) / denom)

    return values.rolling(window, min_periods=window).apply(calculate, raw=True)


def rolling_curvature(values: pd.Series, window: int) -> pd.Series:
    x = np.linspace(-1.0, 1.0, window)

    def calculate(raw: np.ndarray) -> float:
        if not np.isfinite(raw).all():
            return np.nan
        return float(np.polyfit(x, raw, 2)[0])

    return values.rolling(window, min_periods=window).apply(calculate, raw=True)


def true_range(panel: pd.DataFrame) -> pd.Series:
    prior_close = panel["close"].shift(1)
    return pd.concat(
        [
            panel["high"].sub(panel["low"]),
            panel["high"].sub(prior_close).abs(),
            panel["low"].sub(prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def rsi(close: pd.Series, window: int) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    loss = change.mul(-1).clip(lower=0).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = gain.div(loss.replace(0.0, np.nan))
    return 100.0 - 100.0 / (1.0 + rs)


def stochastic(panel: pd.DataFrame, window: int, smooth: int) -> tuple[pd.Series, pd.Series]:
    low = panel["low"].rolling(window, min_periods=window).min()
    high = panel["high"].rolling(window, min_periods=window).max()
    k = panel["close"].sub(low).div(high.sub(low).replace(0.0, np.nan))
    d = k.rolling(smooth, min_periods=smooth).mean()
    return k, d


def one_sided_cusum(values: pd.Series, drift: float, scale_window: int) -> pd.Series:
    z = rolling_zscore(values, scale_window).clip(-8.0, 8.0).to_numpy(float)
    result = np.full(len(z), np.nan)
    level = 0.0
    for pos, value in enumerate(z):
        if not np.isfinite(value):
            continue
        level = max(0.0, level + value - drift)
        result[pos] = level
    return pd.Series(result, index=values.index)


def two_sided_cusum(values: pd.Series, drift: float, scale_window: int) -> pd.Series:
    z = rolling_zscore(values, scale_window).clip(-8.0, 8.0).to_numpy(float)
    result = np.full(len(z), np.nan)
    positive = 0.0
    negative = 0.0
    for pos, value in enumerate(z):
        if not np.isfinite(value):
            continue
        positive = max(0.0, positive + value - drift)
        negative = max(0.0, negative - value - drift)
        result[pos] = positive - negative
    return pd.Series(result, index=values.index)


def page_hinkley(values: pd.Series, delta: float, alpha: float) -> pd.Series:
    raw = values.to_numpy(float)
    result = np.full(len(raw), np.nan)
    running_mean = 0.0
    cumulative = 0.0
    minimum = 0.0
    count = 0
    for pos, value in enumerate(raw):
        if not np.isfinite(value):
            continue
        count += 1
        running_mean += (value - running_mean) / count
        cumulative = alpha * cumulative + value - running_mean - delta
        minimum = min(minimum, cumulative)
        result[pos] = cumulative - minimum
    return pd.Series(result, index=values.index)


def bocpd_directional_score(
    values: pd.Series,
    hazard: float,
    prior_strength: float,
    max_run: int = 100,
) -> pd.Series:
    """Known-variance Gaussian BOCPD run-length recursion with a causal variance.

    The output is short-run posterior mass multiplied by a positive innovation.
    It therefore keeps Adams–MacKay run-length inference while orienting the
    score to the predeclared exit direction.
    """
    x = values.to_numpy(float)
    result = np.full(len(x), np.nan)
    finite = x[np.isfinite(x)]
    prior_mean = float(finite[0]) if len(finite) else 0.0
    run_prob = np.array([1.0])
    means = np.array([prior_mean])
    kappas = np.array([prior_strength], dtype=float)
    causal_mean = prior_mean
    causal_m2 = 0.0
    causal_n = 0

    for pos, value in enumerate(x):
        if not np.isfinite(value):
            continue
        variance = causal_m2 / max(1, causal_n - 1) if causal_n >= 3 else 0.01
        variance = max(variance, 1e-5)
        pred_var = variance * (1.0 + 1.0 / kappas)
        density = np.exp(-0.5 * np.square(value - means) / pred_var) / np.sqrt(
            2.0 * math.pi * pred_var
        )
        change = float(np.sum(run_prob * hazard * density))
        growth = run_prob * (1.0 - hazard) * density
        updated = np.concatenate(([change], growth))
        total = updated.sum()
        if not np.isfinite(total) or total <= EPS:
            updated = np.zeros_like(updated)
            updated[0] = 1.0
        else:
            updated /= total

        new_means = np.concatenate(
            (
                [(prior_strength * prior_mean + value) / (prior_strength + 1.0)],
                (kappas * means + value) / (kappas + 1.0),
            )
        )
        new_kappas = np.concatenate(([prior_strength + 1.0], kappas + 1.0))
        if len(updated) > max_run + 1:
            updated = updated[: max_run + 1]
            updated[-1] += max(0.0, 1.0 - updated.sum())
            updated /= updated.sum()
            new_means = new_means[: max_run + 1]
            new_kappas = new_kappas[: max_run + 1]

        short_mass = float(updated[: min(4, len(updated))].sum())
        innovation = (value - causal_mean) / math.sqrt(variance)
        result[pos] = short_mass * max(0.0, innovation)
        run_prob, means, kappas = updated, new_means, new_kappas

        causal_n += 1
        difference = value - causal_mean
        causal_mean += difference / causal_n
        causal_m2 += difference * (value - causal_mean)
    return pd.Series(result, index=values.index)


def penalized_last_break(
    values: pd.Series,
    window: int,
    min_segment: int,
    penalty: float,
) -> pd.Series:
    raw = values.to_numpy(float)
    result = np.full(len(raw), np.nan)
    for end in range(window - 1, len(raw)):
        sample = raw[end - window + 1 : end + 1]
        if not np.isfinite(sample).all():
            continue
        base_sse = float(np.square(sample - sample.mean()).sum()) + EPS
        best = -np.inf
        for split in range(min_segment, window - min_segment + 1):
            left, right = sample[:split], sample[split:]
            within = float(np.square(left - left.mean()).sum() + np.square(right - right.mean()).sum())
            gain = (base_sse - within) / base_sse - penalty * math.log(window) / window
            signed = float(right.mean() - left.mean())
            best = max(best, max(0.0, signed) * max(0.0, gain))
        result[end] = best
    return pd.Series(result, index=values.index)


def slope_break(values: pd.Series, window: int, min_segment: int) -> pd.Series:
    raw = values.to_numpy(float)
    result = np.full(len(raw), np.nan)
    for end in range(window - 1, len(raw)):
        sample = raw[end - window + 1 : end + 1]
        if not np.isfinite(sample).all():
            continue
        best = -np.inf
        for split in range(min_segment, window - min_segment + 1):
            left, right = sample[:split], sample[split:]
            left_slope = np.polyfit(np.arange(len(left)), left, 1)[0]
            right_slope = np.polyfit(np.arange(len(right)), right, 1)[0]
            difference = float(right_slope - left_slope)
            if difference <= 0:
                continue
            fitted = np.r_[np.polyval(np.polyfit(np.arange(len(left)), left, 1), np.arange(len(left))),
                           np.polyval(np.polyfit(np.arange(len(right)), right, 1), np.arange(len(right)))]
            base_fit = np.polyval(np.polyfit(np.arange(window), sample, 1), np.arange(window))
            gain = np.square(sample - base_fit).sum() - np.square(sample - fitted).sum()
            best = max(best, difference * max(0.0, float(gain)))
        result[end] = max(0.0, best) if np.isfinite(best) else 0.0
    return pd.Series(result, index=values.index)


def _median_pair_slope(sample: np.ndarray) -> float:
    slopes = []
    for right in range(1, len(sample)):
        for left in range(right):
            slopes.append((sample[right] - sample[left]) / (right - left))
    return float(np.median(slopes)) if slopes else np.nan


def theil_sen_break(values: pd.Series, window: int) -> pd.Series:
    raw = values.to_numpy(float)
    result = np.full(len(raw), np.nan)
    split = window // 2
    for end in range(window - 1, len(raw)):
        sample = raw[end - window + 1 : end + 1]
        if not np.isfinite(sample).all():
            continue
        result[end] = _median_pair_slope(sample[split:]) - _median_pair_slope(sample[:split])
    return pd.Series(result, index=values.index)


def chow_break_score(values: pd.Series, window: int, split_share: float) -> pd.Series:
    raw = values.to_numpy(float)
    result = np.full(len(raw), np.nan)
    split = int(round(window * split_share))
    split = max(4, min(window - 4, split))
    x_all = np.arange(window, dtype=float)
    for end in range(window - 1, len(raw)):
        y = raw[end - window + 1 : end + 1]
        if not np.isfinite(y).all():
            continue
        fit_all = np.polyval(np.polyfit(x_all, y, 1), x_all)
        x1, x2 = np.arange(split, dtype=float), np.arange(window - split, dtype=float)
        fit1 = np.polyval(np.polyfit(x1, y[:split], 1), x1)
        fit2 = np.polyval(np.polyfit(x2, y[split:], 1), x2)
        sse_r = float(np.square(y - fit_all).sum())
        sse_u = float(np.square(y[:split] - fit1).sum() + np.square(y[split:] - fit2).sum())
        f_stat = max(0.0, (sse_r - sse_u) / 2.0) / max(EPS, sse_u / max(1, window - 4))
        slope_change = float(np.polyfit(x2, y[split:], 1)[0] - np.polyfit(x1, y[:split], 1)[0])
        result[end] = max(0.0, slope_change) * f_stat
    return pd.Series(result, index=values.index)


def kalman_level_innovation(values: pd.Series, process_var: float, obs_var: float) -> pd.Series:
    raw = values.to_numpy(float)
    result = np.full(len(raw), np.nan)
    level = np.nan
    variance = 1.0
    for pos, observation in enumerate(raw):
        if not np.isfinite(observation):
            continue
        if not np.isfinite(level):
            level = observation
            result[pos] = 0.0
            continue
        predicted_var = variance + process_var
        innovation = observation - level
        result[pos] = innovation / math.sqrt(max(EPS, predicted_var + obs_var))
        gain = predicted_var / (predicted_var + obs_var)
        level += gain * innovation
        variance = (1.0 - gain) * predicted_var
    return pd.Series(result, index=values.index)


def kalman_trend(values: pd.Series, level_var: float, trend_var: float, obs_var: float) -> pd.Series:
    raw = values.to_numpy(float)
    result = np.full(len(raw), np.nan)
    state = np.zeros(2)
    covariance = np.eye(2)
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    observation_matrix = np.array([[1.0, 0.0]])
    process = np.diag([level_var, trend_var])
    initialized = False
    for pos, observation in enumerate(raw):
        if not np.isfinite(observation):
            continue
        if not initialized:
            state[0] = observation
            initialized = True
            result[pos] = 0.0
            continue
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process
        innovation = observation - float(observation_matrix @ state)
        variance = float(observation_matrix @ covariance @ observation_matrix.T + obs_var)
        gain = covariance @ observation_matrix.T / variance
        state = state + gain[:, 0] * innovation
        covariance = (np.eye(2) - gain @ observation_matrix) @ covariance
        result[pos] = state[1]
    return pd.Series(result, index=values.index)


def sign_entropy(returns: pd.Series, window: int) -> pd.Series:
    positive = returns.gt(0).rolling(window, min_periods=window).mean().clip(EPS, 1.0 - EPS)
    return -(positive * np.log(positive) + (1.0 - positive) * np.log(1.0 - positive)) / math.log(2.0)


def directional_run_length(oriented_returns: pd.Series) -> pd.Series:
    sign = oriented_returns.gt(0).astype(int)
    group = sign.ne(sign.shift()).cumsum()
    run = sign.groupby(group).cumcount().add(1)
    return run.where(sign.eq(1), 0).astype(float)

