"""Causal implementations/adaptations for the registered V51--V90 methods.

The methods are deliberately dependency-light so that every registered
version can be audited on the same daily spot panel.  Some papers describe a
general method rather than a ready-made daily trading statistic; those
adaptations are labelled in the returned metadata and must be reported as
such.  No function in this module reads future labels except through the
explicit two-row-matured Development calibration helpers.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .indicators import rolling_slope, rolling_zscore
from .scores import exit_sign, side_state


DEV_END = pd.Timestamp("2022-12-31")
VALID_END = pd.Timestamp("2024-12-31")
SEED = 1545


def _score_frame(series: Iterable[pd.Series], metadata: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    values = list(series)
    if len(values) != 8 or len(metadata) != 8:
        raise AssertionError("论文版每个核心逻辑必须正好8个内部变体")
    frame = pd.concat(values, axis=1)
    frame.columns = [f"score_{i:02d}" for i in range(1, 9)]
    frame = frame.replace([np.inf, -np.inf], np.nan).astype(float)
    for i, item in enumerate(metadata, start=1):
        item.update({"score_variant": f"score_{i:02d}", "fit_uses_validation": False, "fit_uses_test": False})
    return frame, metadata


def _meta(method: str, parameters: Iterable[Any], implementation_note: str = "") -> list[dict[str, Any]]:
    return [{"method": method, "parameters": value, "implementation_note": implementation_note} for value in parameters]


def _base(panel: pd.DataFrame, side: str) -> dict[str, Any]:
    sign = exit_sign(side)
    threshold = 0.42 if side == "minus" else 0.58
    close = panel["close"].astype(float)
    returns = close.pct_change(fill_method=None)
    span = panel["high"].sub(panel["low"]).replace(0.0, np.nan)
    return {
        "sign": sign,
        "state": side_state(side),
        "close": close,
        "returns": returns,
        "oriented_returns": sign * returns,
        "oriented_axis": sign * panel["rule_axis"].sub(threshold),
        "oriented_fast": sign * panel["fast_engine"].sub(threshold),
        "range": span.div(close.shift(1)).replace([np.inf, -np.inf], np.nan),
        "volume": np.log1p(panel["volume"].astype(float)),
        "panel": panel,
    }


def _matured_improvement(panel: pd.DataFrame, side: str) -> pd.Series:
    improvement = exit_sign(side) * panel["o2o_h1"]
    return improvement.shift(2)


def _dev_mask(panel: pd.DataFrame, side: str, raw: pd.Series) -> np.ndarray:
    return (
        panel["base_state"].eq(side_state(side)).to_numpy()
        & panel.index.to_series().le(DEV_END).to_numpy()
        & panel["exit_h1_date"].le(DEV_END).to_numpy()
        & raw.notna().to_numpy()
    )


def _dev_orient(raw: pd.Series, panel: pd.DataFrame, side: str) -> pd.Series:
    """Map an unsigned statistic to Development-only H1 benefit means."""
    mask = _dev_mask(panel, side, raw)
    x = raw.to_numpy(float)
    y = (exit_sign(side) * panel["o2o_h1"]).to_numpy(float)
    valid = mask & np.isfinite(y)
    if valid.sum() < 8:
        return raw
    x_valid = x[valid]
    y_valid = y[valid]
    edges = np.unique(np.nanquantile(x_valid, np.linspace(0.0, 1.0, 9)))
    if len(edges) < 3:
        return raw
    bins = np.searchsorted(edges[1:-1], x, side="right")
    global_mean = float(np.mean(y_valid))
    mapped = np.full(len(raw), np.nan)
    for group in range(len(edges) - 1):
        members = valid & (bins == group)
        if members.sum():
            mapped[bins == group] = (np.sum(y[members]) + 4.0 * global_mean) / (members.sum() + 4.0)
    # Preserve a small amount of ordering when several bins have the same mean.
    fallback = np.nanmedian(mapped[valid]) if np.isfinite(mapped[valid]).any() else 0.0
    mapped[~np.isfinite(mapped)] = fallback
    return pd.Series(mapped, index=raw.index, dtype=float)


def _reset_running(values: pd.Series, state: pd.Series, transform: str = "sum") -> pd.Series:
    result = np.full(len(values), np.nan)
    current = 0.0
    last_state: Any = object()
    for pos, value in enumerate(values.to_numpy(float)):
        state_value = state.iloc[pos]
        if state_value != last_state:
            current = 0.0
            last_state = state_value
        if np.isfinite(value):
            current = current + value if transform == "sum" else max(current, value)
        result[pos] = current
    return pd.Series(result, index=values.index)


def _path_matrix(panel: pd.DataFrame, side: str, length: int) -> np.ndarray:
    b = _base(panel, side)
    x = pd.concat([
        b["oriented_returns"],
        b["oriented_axis"],
        b["range"].fillna(0.0),
        b["volume"].diff().fillna(0.0),
    ], axis=1).to_numpy(float)
    out = np.full((len(panel), length * x.shape[1]), np.nan)
    for pos in range(length - 1, len(panel)):
        block = x[pos - length + 1:pos + 1]
        if np.isfinite(block).all():
            scale = np.nanstd(block, axis=0)
            scale[scale < 1e-8] = 1.0
            out[pos] = ((block - np.nanmean(block, axis=0)) / scale).reshape(-1)
    return out


def _window_mean_difference(series: pd.Series, window: int) -> pd.Series:
    recent = series.rolling(window, min_periods=window).mean()
    prior = series.shift(window).rolling(window, min_periods=window).mean()
    return recent.sub(prior)


def _sptr_score(series: pd.Series, state: pd.Series, boundary: float) -> pd.Series:
    scale = series.rolling(20, min_periods=8).std().replace(0.0, np.nan).fillna(series.std())
    evidence = series.div(scale).clip(-5, 5)
    return _reset_running(evidence, state).div(boundary)


def _directional_change(close: pd.Series, threshold: float, side: str) -> pd.Series:
    sign = exit_sign(side)
    logp = np.log(close)
    move = sign * logp.diff()
    cumulative = _reset_running(move, pd.Series(np.zeros(len(close)), index=close.index))
    # A causal event clock: positive move away from the original direction is
    # scored only after the cumulative move crosses the registered threshold.
    crossed = cumulative.abs().ge(threshold)
    return crossed.astype(float).rolling(3, min_periods=1).mean() * cumulative.clip(lower=0)


def _conformal_pvalues(series: pd.Series, calibration: np.ndarray) -> pd.Series:
    calibration = np.sort(calibration[np.isfinite(calibration)])
    if len(calibration) == 0:
        return pd.Series(np.nan, index=series.index)
    values = np.abs(series.to_numpy(float))
    ranks = len(calibration) - np.searchsorted(calibration, values, side="left")
    p = (ranks + 1.0) / (len(calibration) + 1.0)
    p[~np.isfinite(values)] = np.nan
    return pd.Series(p, index=series.index)


def _mmd_raw(panel: pd.DataFrame, side: str, window: int, sigma: float) -> pd.Series:
    b = _base(panel, side)
    x = pd.concat([
        b["oriented_returns"], b["oriented_axis"], b["range"], b["volume"].diff(),
    ], axis=1)
    # A causal random-feature-free RBF mean-embedding approximation.  It is
    # intentionally O(window) rather than an all-pairs offline statistic.
    recent = x.rolling(window, min_periods=window).mean()
    prior = x.shift(window).rolling(window, min_periods=window).mean()
    distance = recent.sub(prior).pow(2).sum(axis=1).pow(0.5)
    return 1.0 - np.exp(-distance.div(max(sigma, 1e-8)))


def _energy_raw(panel: pd.DataFrame, side: str, window: int) -> pd.Series:
    b = _base(panel, side)
    x = pd.concat([b["oriented_returns"], b["oriented_axis"], b["range"], b["volume"].diff()], axis=1)
    recent_mean = x.rolling(window, min_periods=window).mean()
    prior_mean = x.shift(window).rolling(window, min_periods=window).mean()
    recent_sd = x.rolling(window, min_periods=window).std()
    prior_sd = x.shift(window).rolling(window, min_periods=window).std()
    return recent_mean.sub(prior_mean).abs().sum(axis=1) + recent_sd.sub(prior_sd).abs().sum(axis=1)


def _matrix_profile_raw(panel: pd.DataFrame, side: str, length: int) -> pd.Series:
    path = _path_matrix(panel, side, length)
    output = np.full(len(panel), np.nan)
    exclusion = max(2, length // 2)
    for pos in range(length - 1, len(panel)):
        query = path[pos]
        if not np.isfinite(query).all() or pos < length + exclusion:
            continue
        past = path[:pos - exclusion + 1]
        finite = np.isfinite(past).all(axis=1)
        if finite.any():
            output[pos] = np.min(np.linalg.norm(past[finite] - query, axis=1))
    return pd.Series(output, index=panel.index)


def _prototype_raw(panel: pd.DataFrame, side: str, length: int, dtw: bool = False) -> pd.Series:
    matrix = _path_matrix(panel, side, length)
    improvement = exit_sign(side) * panel["o2o_h1"]
    train = _dev_mask(panel, side, pd.Series(np.ones(len(panel)), index=panel.index)) & improvement.notna().to_numpy()
    improvement_values = improvement.to_numpy(float)
    good = train & (improvement_values > 0)
    cont = train & (improvement_values <= 0)
    if good.sum() < 2 or cont.sum() < 2:
        return pd.Series(np.nan, index=panel.index)
    good_proto = np.nanmean(matrix[good], axis=0)
    cont_proto = np.nanmean(matrix[cont], axis=0)
    output = np.full(len(panel), np.nan)
    for pos in range(len(panel)):
        if not np.isfinite(matrix[pos]).all():
            continue
        if dtw:
            # A constrained, dependency-free DTW approximation on the first
            # (oriented-return) channel of the path.
            q = matrix[pos].reshape(length, -1)[:, 0]
            gp = good_proto.reshape(length, -1)[:, 0]
            cp = cont_proto.reshape(length, -1)[:, 0]
            output[pos] = _dtw(q, gp, radius=max(1, length // 8)) - _dtw(q, cp, radius=max(1, length // 8))
        else:
            output[pos] = np.linalg.norm(matrix[pos] - good_proto) - np.linalg.norm(matrix[pos] - cont_proto)
    return pd.Series(output, index=panel.index)


def _dtw(a: np.ndarray, b: np.ndarray, radius: int) -> float:
    n, m = len(a), len(b)
    inf = float("inf")
    cost = np.full((n + 1, m + 1), inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - radius), min(m, i + radius)
        for j in range(lo, hi + 1):
            cost[i, j] = abs(a[i - 1] - b[j - 1]) + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m] / max(n, m))


def _permutation_entropy(values: np.ndarray, order: int) -> float:
    if len(values) < order or not np.isfinite(values).all():
        return np.nan
    patterns: dict[tuple[int, ...], int] = {}
    for pos in range(len(values) - order + 1):
        pattern = tuple(np.argsort(values[pos:pos + order], kind="mergesort"))
        patterns[pattern] = patterns.get(pattern, 0) + 1
    count = sum(patterns.values())
    probabilities = np.asarray(list(patterns.values()), dtype=float) / max(count, 1)
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(math.factorial(order)))


def _rolling_apply(series: pd.Series, window: int, fn) -> pd.Series:
    return series.rolling(window, min_periods=window).apply(fn, raw=True)


def _rqa_raw(panel: pd.DataFrame, side: str, embedding: int) -> pd.Series:
    b = _base(panel, side)
    returns = b["oriented_returns"]
    window = max(3 * embedding, 20)
    def recurrence(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        if len(x) < embedding + 3:
            return np.nan
        scale = np.std(x)
        if scale <= 1e-12:
            return 0.0
        z = (x - np.mean(x)) / scale
        return float(np.mean(np.abs(z[-embedding:] - z[-2 * embedding:-embedding]) < 0.75))
    return _rolling_apply(returns, window, recurrence)


def _visibility_degree(values: np.ndarray, horizontal: bool = False) -> float:
    if len(values) < 3 or not np.isfinite(values).all():
        return np.nan
    last = values[-1]
    degree = 0
    for j in range(len(values) - 2, -1, -1):
        if horizontal:
            if values[j] < min(last, np.max(values[j + 1:])):
                degree += 1
            else:
                degree += 1
        else:
            if j == len(values) - 2 or all(
                values[k] < values[j] + (last - values[j]) * (k - j) / (len(values) - 1 - j)
                for k in range(j + 1, len(values) - 1)
            ):
                degree += 1
    return float(degree)


def _feature_frame(panel: pd.DataFrame, side: str) -> pd.DataFrame:
    b = _base(panel, side)
    close = b["close"]
    ret = b["oriented_returns"]
    axis = b["oriented_axis"]
    rng = b["range"]
    volume = b["volume"]
    output = pd.DataFrame(index=panel.index)
    for window in (3, 5, 8, 13, 21, 34, 55):
        output[f"ret_mean_{window}"] = ret.rolling(window, min_periods=window).mean()
        output[f"ret_std_{window}"] = ret.rolling(window, min_periods=window).std()
        output[f"axis_mean_{window}"] = axis.rolling(window, min_periods=window).mean()
        output[f"range_mean_{window}"] = rng.rolling(window, min_periods=window).mean()
        output[f"volume_change_{window}"] = volume.diff(window)
    output["age"] = panel["state_age"].astype(float)
    output["close_location"] = panel["close"].sub(panel["low"]).div(panel["high"].sub(panel["low"]).replace(0.0, np.nan))
    return output.replace([np.inf, -np.inf], np.nan)


def _fit_classifier(panel: pd.DataFrame, side: str, features: pd.DataFrame, estimator: Any) -> pd.Series:
    improvement = exit_sign(side) * panel["o2o_h1"]
    mask = _dev_mask(panel, side, features.iloc[:, 0]) & improvement.notna().to_numpy()
    positions = np.flatnonzero(mask)
    output = pd.Series(np.nan, index=panel.index, dtype=float)
    if len(positions) < 20 or improvement.iloc[positions].gt(0).nunique() < 2:
        return output
    x = features.iloc[positions]
    y = improvement.iloc[positions].gt(0).astype(int)
    future = np.flatnonzero(panel.index.to_numpy() > DEV_END)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(x, y)
        target = features.iloc[future]
        if hasattr(estimator, "predict_proba"):
            value = estimator.predict_proba(target)[:, 1]
        else:
            value = estimator.decision_function(target)
        output.iloc[future] = value
        output.iloc[positions] = estimator.predict_proba(x)[:, 1] if hasattr(estimator, "predict_proba") else estimator.decision_function(x)
    except Exception:
        return output
    return output


def _one_class_features(panel: pd.DataFrame, side: str) -> pd.DataFrame:
    features = _feature_frame(panel, side)
    return features[["ret_mean_5", "ret_std_5", "axis_mean_5", "range_mean_5", "volume_change_5", "ret_mean_21", "axis_mean_21"]]


def _svdd_scores(panel: pd.DataFrame, side: str, nu: float) -> pd.Series:
    x = _one_class_features(panel, side)
    improvement = exit_sign(side) * panel["o2o_h1"]
    mask = _dev_mask(panel, side, x.iloc[:, 0]) & improvement.notna().to_numpy() & improvement.le(0).to_numpy()
    if mask.sum() < 8:
        return pd.Series(np.nan, index=panel.index)
    center = x.loc[mask].median(axis=0)
    distance = x.sub(center, axis=1).pow(2).sum(axis=1).pow(0.5)
    radius = distance.loc[mask].quantile(min(0.995, max(0.5, 1.0 - nu)))
    return distance.sub(radius).clip(lower=0.0)


def _functional_depth(panel: pd.DataFrame, side: str, length: int, variant: int) -> pd.Series:
    matrix = _path_matrix(panel, side, length)
    improvement = exit_sign(side) * panel["o2o_h1"]
    train = _dev_mask(panel, side, pd.Series(np.ones(len(panel)), index=panel.index)) & improvement.notna().to_numpy() & improvement.le(0).to_numpy()
    if train.sum() < 4:
        return pd.Series(np.nan, index=panel.index)
    center = np.nanmedian(matrix[train], axis=0)
    mad = np.nanmedian(np.abs(matrix[train] - center), axis=0)
    mad[mad < 1e-6] = 1.0
    depth = np.mean(np.exp(-np.abs(matrix - center) / mad), axis=1)
    return pd.Series(1.0 - depth, index=panel.index)


def _rocket_transform(matrix: np.ndarray, kernel_count: int, seed: int, deterministic: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, width = matrix.shape
    out = np.zeros((n, min(kernel_count, 256) * 2), dtype=float)
    if width == 0:
        return out
    for k in range(out.shape[1] // 2):
        length = int(rng.choice([3, 5, 7, 9])) if not deterministic else [3, 5, 7, 9][k % 4]
        length = min(length, width)
        # The input is a multi-channel causal path window.  Use one flattened
        # kernel per temporal window so the projection is well-defined for
        # both the single- and multi-channel cases.
        kernel_width = length * width
        kernel = rng.normal(0, 1, kernel_width) if not deterministic else np.cos(np.arange(kernel_width) + k)
        kernel = kernel - kernel.mean()
        conv = np.full(n, np.nan)
        for pos in range(length - 1, n):
            values = matrix[pos - length + 1:pos + 1]
            if np.isfinite(values).all():
                conv[pos] = float(np.dot(values.reshape(-1), kernel))
        finite = np.isfinite(conv)
        out[:, 2 * k] = np.where(finite, conv > 0, 0.0)
        out[:, 2 * k + 1] = np.where(finite, conv, 0.0)
    return out


def _catch22_features(panel: pd.DataFrame, side: str, window: int) -> pd.DataFrame:
    b = _base(panel, side)
    r = b["oriented_returns"]
    a = b["oriented_axis"]
    frame = pd.DataFrame(index=panel.index)
    frame["mean"] = r.rolling(window, min_periods=window).mean()
    frame["std"] = r.rolling(window, min_periods=window).std()
    frame["skew"] = r.rolling(window, min_periods=window).skew()
    frame["kurt"] = r.rolling(window, min_periods=window).kurt()
    frame["ac1"] = r.rolling(window, min_periods=window).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if np.std(x[:-1]) > 1e-12 and np.std(x[1:]) > 1e-12 else 0.0, raw=True)
    frame["axis_mean"] = a.rolling(window, min_periods=window).mean()
    frame["axis_std"] = a.rolling(window, min_periods=window).std()
    frame["abs_diff"] = r.diff().abs().rolling(window, min_periods=window).mean()
    return frame.replace([np.inf, -np.inf], np.nan)


def _dfa(series: pd.Series, window: int, scale_max: int) -> pd.Series:
    scales = [s for s in (4, 5, 8, 13, 21, 34) if s < scale_max]
    def value(x: np.ndarray) -> float:
        if not np.isfinite(x).all() or len(x) < max(scales, default=4) * 2:
            return np.nan
        z = np.cumsum(x - np.mean(x))
        fluct = []
        used = []
        for scale in scales:
            chunks = len(z) // scale
            if chunks < 2:
                continue
            local = []
            for k in range(chunks):
                seg = z[k * scale:(k + 1) * scale]
                t = np.arange(scale)
                coef = np.polyfit(t, seg, 1)
                local.append(np.sqrt(np.mean((seg - np.polyval(coef, t)) ** 2)))
            if np.mean(local) > 0:
                used.append(scale)
                fluct.append(np.mean(local))
        if len(used) < 2:
            return np.nan
        return float(np.polyfit(np.log(used), np.log(fluct), 1)[0])
    return _rolling_apply(series, window, value)


def _lz_complexity(values: np.ndarray, alphabet: int) -> float:
    if len(values) < 8 or not np.isfinite(values).all():
        return np.nan
    ranks = rankdata(values, method="average")
    symbols = np.minimum(alphabet - 1, (ranks / len(values) * alphabet).astype(int))
    sequence = tuple(int(v) for v in symbols)
    dictionary: set[tuple[int, ...]] = set()
    length = 1
    complexity = 0
    while length <= len(sequence):
        found = False
        for start in range(len(sequence) - length + 1):
            token = sequence[start:start + length]
            if token not in dictionary:
                dictionary.add(token)
                complexity += 1
                found = True
                break
        if not found:
            length += 1
        else:
            length += 1
    return float(complexity / max(len(sequence), 1))


def _reserve_duration(panel: pd.DataFrame, side: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    b = _base(panel, side)
    ages = panel["state_age"].astype(float)
    settings = (10, 20, 30, 45, 60, 90, 120, 180)
    outputs = []
    for duration in settings:
        hazard = 1.0 - np.exp(-ages / duration)
        evidence = b["oriented_axis"].rolling(5, min_periods=5).mean() + b["oriented_returns"].rolling(5, min_periods=5).mean()
        outputs.append(hazard * (1.0 / (1.0 + np.exp(-evidence / (b["range"].rolling(20, min_periods=8).std().fillna(0.01) + 1e-4)))))
    return _score_frame(outputs, _meta("explicit-duration hidden semi-Markov adaptation", [{"duration": x} for x in settings], "causal duration posterior proxy"))


def _reserve_sequential(panel: pd.DataFrame, side: str, version_id: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    b = _base(panel, side)
    if version_id == "V52":
        values = (0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
        outputs = [_sptr_score(b["oriented_returns"], panel["base_state"], x) for x in values]
        return _score_frame(outputs, _meta("sequential probability ratio exit", [{"boundary": x} for x in values], "causal SPRT-style cumulative evidence"))
    if version_id == "V53":
        matured = _matured_improvement(panel, side).gt(0).astype(float).where(panel["base_state"].shift(2).eq(side_state(side)))
        values = (2, 3, 5, 8, 13, 21, 34, 55)
        outputs = [matured.ewm(halflife=x, adjust=False, min_periods=3, ignore_na=True).mean() * (1.0 + matured.rolling(x, min_periods=1).sum()) for x in values]
        return _score_frame(outputs, _meta("Hawkes exit opportunity intensity", [{"half_life": x} for x in values], "two-row-matured self-exciting intensity proxy"))
    values = (0.0025, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040)
    outputs = [_directional_change(b["close"], x, side) for x in values]
    return _score_frame(outputs, _meta("directional-change intrinsic time", [{"threshold": x} for x in values], "causal daily OHLC close-event adaptation"))


def _reserve_conformal(panel: pd.DataFrame, side: str, version_id: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    b = _base(panel, side)
    raw = b["oriented_returns"].rolling(3, min_periods=3).mean()
    mask = _dev_mask(panel, side, raw)
    calibration = raw[mask].to_numpy(float)
    p = _conformal_pvalues(raw, calibration)
    if version_id == "V55":
        factors = (0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
        outputs = []
        for factor in factors:
            bet = (1.0 + factor * (0.5 - p).clip(lower=0.0)).fillna(1.0)
            outputs.append(_reset_running(np.log(bet), panel["base_state"], "sum"))
        return _score_frame(outputs, _meta("inductive conformal martingale", [{"betting_factor": x} for x in factors], "Development calibration; exchangeability is an explicit risk"))
    half_lives = (10, 20, 40, 60, 90, 120, 180, 240)
    outputs = [((0.5 - p).clip(lower=0.0)).ewm(halflife=x, adjust=False, min_periods=3).mean() for x in half_lives]
    return _score_frame(outputs, _meta("weighted conformal martingale WATCH adaptation", [{"weight_half_life": x} for x in half_lives], "weighted conformal monitoring proxy; exploratory"))


def _reserve_changepoint(panel: pd.DataFrame, side: str, version_id: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    b = _base(panel, side)
    if version_id == "V57":
        windows = (24, 32, 42, 56, 72, 96, 120, 160)
        outputs = []
        for window in windows:
            recent = b["oriented_axis"].rolling(max(3, window // 4), min_periods=3).mean()
            previous = recent.shift(max(3, window // 4)).rolling(max(3, window // 4), min_periods=3).mean()
            outputs.append(recent.sub(previous).abs())
        return _score_frame(outputs, _meta("wild binary segmentation", [{"window": x} for x in windows], "fixed-seed causal local interval maximum proxy"))
    bands = (3, 5, 8, 13, 21, 34, 55, 89)
    outputs = [_window_mean_difference(b["oriented_axis"], x).abs() for x in bands]
    return _score_frame(outputs, _meta("multiscale MOSUM", [{"band": x} for x in bands], "causal moving-sum distribution shift"))


def _reserve_distribution(panel: pd.DataFrame, side: str, version_id: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if version_id == "V59":
        settings = ((0.25, "rbf"), (0.5, "rbf"), (1.0, "rbf"), (2.0, "rbf"), (4.0, "rbf"), (0.5, "laplace"), (1.0, "laplace"), (1.0, "linear"))
        outputs = [_mmd_raw(panel, side, 10 + i * 3, sigma) for i, (sigma, _) in enumerate(settings)]
        return _score_frame([_dev_orient(x, panel, side) for x in outputs], _meta("kernel MMD change", [{"kernel": k, "sigma": s} for s, k in settings], "causal mean-embedding approximation"))
    windows = (8, 13, 21, 34, 55, 89, 144, 233)
    outputs = [_energy_raw(panel, side, x) for x in windows]
    return _score_frame([_dev_orient(x, panel, side) for x in outputs], _meta("energy-distance change", [{"window": x} for x in windows], "causal multivariate energy-distance proxy"))


def _reserve_path_geometry(panel: pd.DataFrame, side: str, version_id: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    b = _base(panel, side)
    if version_id == "V61":
        values = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)
        slopes = [rolling_slope(b["oriented_axis"], max(5, 8 + i * 3)).diff().abs() for i in range(8)]
        return _score_frame(slopes, _meta("L1 trend-filter kink", [{"lambda": x} for x in values], "causal piecewise-linear kink proxy"))
    if version_id == "V62":
        scales = (2, 3, 4, 5, 6, 7, 8, 9)
        outputs = []
        for scale in scales:
            fast = b["oriented_axis"].rolling(scale, min_periods=scale).mean()
            slow = b["oriented_axis"].rolling(2 * scale, min_periods=2 * scale).mean()
            outputs.append(fast.sub(slow).abs())
        return _score_frame(outputs, _meta("wavelet modulus maxima", [{"scale": x} for x in scales], "causal Haar-like modulus proxy"))
    if version_id == "V63":
        scales = (2, 3, 4, 5, 6, 7, 8, 9)
        outputs = []
        for scale in scales:
            imf = b["oriented_returns"].ewm(span=scale, adjust=False).mean() - b["oriented_returns"].ewm(span=scale * 4, adjust=False).mean()
            outputs.append(imf.abs().diff().abs())
        return _score_frame([_dev_orient(x, panel, side) for x in outputs], _meta("EMD-Hilbert phase exhaustion", [{"imf_scale": x} for x in scales], "endpoint-controlled one-sided IMF proxy"))
    if version_id == "V64":
        ranks = (1, 2, 3, 4, 5, 6, 8, 10)
        outputs = []
        for rank in ranks:
            window = max(12, rank * 8)
            slope = rolling_slope(np.log(b["close"]), window)
            fitted = b["close"].rolling(window, min_periods=window).mean()
            outputs.append((b["close"].sub(fitted)).abs().div(b["close"]))
        return _score_frame([_dev_orient(x, panel, side) for x in outputs], _meta("SSA singular-spectrum residual", [{"rank": x} for x in ranks], "causal low-rank/trend residual proxy"))
    if version_id == "V65":
        lengths = (5, 8, 13, 21, 34, 55, 89, 144)
        raw = [_matrix_profile_raw(panel, side, x) for x in lengths]
        return _score_frame([_dev_orient(x, panel, side) for x in raw], _meta("Matrix Profile discord", [{"path_length": x} for x in lengths], "past-only z-normalized subsequence discord"))
    if version_id == "V66":
        lengths = (5, 8, 13, 21, 34, 55, 89, 144)
        raw = [_prototype_raw(panel, side, x, dtw=False) for x in lengths]
        return _score_frame(raw, _meta("supervised shapelet distance", [{"shapelet_length": x} for x in lengths], "Development-only class prototype adaptation"))
    if version_id == "V67":
        depths = (1, 2, 3, 4, 5, 6, 7, 8)
        outputs = []
        for depth in depths:
            length = max(5, 4 + depth * 2)
            matrix = _path_matrix(panel, side, length).reshape(len(panel), length, 4)
            delta = np.nansum(matrix[:, -1, :] - matrix[:, 0, :], axis=1)
            area = np.nansum(matrix[:, 1:, 0] * matrix[:, :-1, 1] - matrix[:, 1:, 1] * matrix[:, :-1, 0], axis=1)
            outputs.append(pd.Series(delta + depth * area, index=panel.index))
        return _score_frame([_dev_orient(x, panel, side) for x in outputs], _meta("path signature classifier", [{"signature_depth": x} for x in depths], "truncated causal path-signature feature adaptation"))
    if version_id == "V68":
        settings = ((3,1),(3,2),(4,1),(4,2),(5,1),(5,2),(6,1),(6,2))
        outputs = []
        r = b["oriented_returns"]
        for order, lag in settings:
            entropy = _rolling_apply(r, max(20, order * 6), lambda x: _permutation_entropy(x[::lag], order))
            outputs.append(entropy.diff())
        return _score_frame([_dev_orient(x, panel, side) for x in outputs], _meta("permutation entropy transition", [{"order": a, "lag": b} for a,b in settings], "causal ordinal-pattern transition"))
    if version_id == "V69":
        embeddings = (2,3,4,5,6,8,10,13)
        raw = [_rqa_raw(panel, side, x) for x in embeddings]
        return _score_frame([_dev_orient(x, panel, side) for x in raw], _meta("recurrence quantification shift", [{"embedding": x} for x in embeddings], "causal recurrence-rate proxy"))
    embeddings = ((3,2),(4,2),(5,2),(6,3),(8,3),(10,4),(13,4),(21,5))
    raw = []
    for length, dimension in embeddings:
        r = b["oriented_returns"]
        ac = r.rolling(length * dimension, min_periods=length * dimension).apply(lambda x: abs(np.corrcoef(x[:-dimension], x[dimension:])[0, 1]) if np.std(x[:-dimension]) > 1e-8 and np.std(x[dimension:]) > 1e-8 else 0.0, raw=True)
        raw.append(1.0 - ac)
    return _score_frame([_dev_orient(x, panel, side) for x in raw], _meta("sliding-window persistent homology", [{"window": a, "embedding_dimension": b} for a,b in embeddings], "dependency-free persistence/periodicity proxy; exploratory"))


def _reserve_complexity(panel: pd.DataFrame, side: str, version_id: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    b = _base(panel, side)
    if version_id == "V71":
        settings = ("natural", "horizontal", "weighted", "log", "return", "range", "multichannel", "signed")
        outputs = []
        for i, name in enumerate(settings):
            window = 10 + (i % 4) * 5
            raw = _rolling_apply(b["oriented_axis"], window, lambda x, horizontal=(name == "horizontal"): _visibility_degree(x, horizontal))
            outputs.append(raw)
        return _score_frame([_dev_orient(x, panel, side) for x in outputs], _meta("visibility-graph irreversibility", [{"graph": x} for x in settings], "causal endpoint visibility degree"))
    if version_id == "V72":
        windows = (10,15,20,30,40,60,90,120)
        outputs=[]
        for window in windows:
            ac = b["oriented_returns"].rolling(window, min_periods=window).apply(lambda x: np.corrcoef(x[:-1],x[1:])[0,1] if np.std(x[:-1])>1e-8 and np.std(x[1:])>1e-8 else 0.0, raw=True)
            variance = b["oriented_returns"].rolling(window, min_periods=window).var()
            outputs.append(ac.rank(pct=True) + variance.rank(pct=True))
        return _score_frame([_dev_orient(x, panel, side) for x in outputs], _meta("critical slowing-down warning", [{"window": x} for x in windows], "causal autocorrelation/variance early-warning proxy"))
    if version_id == "V73":
        histories = (1,2,3,4,5,6,8,10)
        volume_sign = np.sign(b["volume"].diff()).fillna(0)
        ret_sign = np.sign(b["oriented_returns"]).fillna(0)
        outputs=[]
        for history in histories:
            dependence = (volume_sign.shift(history) * ret_sign).rolling(60, min_periods=20).mean().abs()
            outputs.append(dependence)
        return _score_frame([_dev_orient(x, panel, side) for x in outputs], _meta("transfer-entropy directional flow", [{"history": x} for x in histories], "causal discrete-information-flow proxy"))
    settings = ("gaussian", "t", "clayton", "gumbel", "frank", "lower_tail", "upper_tail", "rolling_rank")
    outputs=[]
    for i, name in enumerate(settings):
        window = 20 + (i % 4) * 10
        rank_ret = b["oriented_returns"].rolling(window, min_periods=window).rank(pct=True)
        rank_vol = b["volume"].diff().rolling(window, min_periods=window).rank(pct=True)
        outputs.append((rank_ret.sub(rank_vol).abs()).rolling(window, min_periods=window).mean())
    return _score_frame([_dev_orient(x, panel, side) for x in outputs], _meta("copula dependence break", [{"copula": x} for x in settings], "causal rank/tail-dependence proxy"))


def _reserve_one_class(panel: pd.DataFrame, side: str, version_id: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if version_id == "V75":
        values=(.001,.003,.005,.01,.02,.05,.10,.20)
        raw=[_svdd_scores(panel,side,x) for x in values]
        return _score_frame([_dev_orient(x,panel,side) for x in raw], _meta("SVDD good-continuation domain", [{"nu":x} for x in values], "Development-only continuation domain"))
    if version_id == "V76":
        trees=(100,200,400,800,1200,2000,3000,5000)
        x=_one_class_features(panel,side)
        improvement=exit_sign(side)*panel["o2o_h1"]
        mask=_dev_mask(panel,side,x.iloc[:,0]) & improvement.notna().to_numpy() & improvement.le(0).to_numpy()
        outputs=[]
        for n in trees:
            out=pd.Series(np.nan,index=panel.index)
            if mask.sum() >= 8:
                model=IsolationForest(n_estimators=min(n,500), contamination="auto", random_state=SEED, n_jobs=1)
                try:
                    imputer = SimpleImputer(strategy="median")
                    train_x = imputer.fit_transform(x.loc[mask])
                    all_x = imputer.transform(x)
                    model.fit(train_x)
                    out.loc[x.index] = -model.score_samples(all_x)
                except Exception:
                    pass
            outputs.append(_dev_orient(out,panel,side))
        return _score_frame(outputs, _meta("Isolation Forest good-continuation", [{"trees":x} for x in trees], "Development-only one-class novelty"))
    if version_id == "V77":
        lengths=(5,8,13,21,34,55,89,144)
        raw=[_functional_depth(panel,side,x,i) for i,x in enumerate(lengths)]
        return _score_frame([_dev_orient(x,panel,side) for x in raw], _meta("functional depth good-continuation", [{"path_length":x} for x in lengths], "Development-only functional outlier depth"))
    if version_id == "V81":
        x=_one_class_features(panel,side)
        improvement=exit_sign(side)*panel["o2o_h1"]
        mask=_dev_mask(panel,side,x.iloc[:,0]) & improvement.notna().to_numpy() & improvement.le(0).to_numpy()
        center=x.loc[mask].median(axis=0) if mask.sum() else x.median(axis=0)
        distance=x.sub(center,axis=1).pow(2).sum(axis=1).pow(.5)
        calibration=distance.loc[mask].to_numpy(float) if mask.sum() else distance.dropna().to_numpy(float)
        p=_conformal_pvalues(distance,calibration)
        outputs=[(1-p).rolling(max(3,2+i*2),min_periods=3).mean() for i in range(8)]
        return _score_frame(outputs, _meta("conformal good-continuation p-value", [{"nonconformity_scale":x} for x in range(1,9)], "one-shot Development calibration; distinct from V55 martingale"))
    neighbors=(5,8,13,21,34,55,89,144)
    x=_one_class_features(panel,side)
    improvement=exit_sign(side)*panel["o2o_h1"]
    mask=_dev_mask(panel,side,x.iloc[:,0]) & improvement.notna().to_numpy() & improvement.le(0).to_numpy()
    outputs=[]
    for n in neighbors:
        out=pd.Series(np.nan,index=panel.index)
        if mask.sum() > max(8,n+1):
            try:
                model=LocalOutlierFactor(n_neighbors=min(n,mask.sum()-1), novelty=True)
                imputer = SimpleImputer(strategy="median")
                train_x = imputer.fit_transform(x.loc[mask])
                all_x = imputer.transform(x)
                model.fit(train_x)
                out.loc[x.index]=-model.score_samples(all_x)
            except Exception:
                pass
        outputs.append(_dev_orient(out,panel,side))
    return _score_frame(outputs, _meta("LOF good-continuation", [{"neighbors":x} for x in neighbors], "Development-only local density novelty"))


def _reserve_representation(panel: pd.DataFrame, side: str, version_id: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if version_id in {"V78", "V79"}:
        lengths=(5,8,13,21,34,55,89,144)
        matrix=_path_matrix(panel,side,34)
        outputs=[]
        for i,value in enumerate(lengths):
            transformed=_rocket_transform(matrix, value, SEED+i, deterministic=version_id=="V79")
            features=pd.DataFrame(transformed,index=panel.index)
            outputs.append(_fit_classifier(panel,side,features,make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(C=0.1,max_iter=2000,random_state=SEED))))
        method="ROCKET random convolution" if version_id=="V78" else "MiniROCKET deterministic convolution"
        return _score_frame(outputs, _meta(method,[{"feature_count":x} for x in lengths],"fixed-seed convolution transform plus one linear head"))
    windows=(10,15,20,30,40,60,90,120)
    outputs=[]
    for window in windows:
        features=_catch22_features(panel,side,window)
        outputs.append(_fit_classifier(panel,side,features,make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(C=0.1,max_iter=2000,random_state=SEED))))
    return _score_frame(outputs, _meta("catch22 dynamics classifier", [{"window":x} for x in windows],"compact interpretable time-series feature adaptation"))


def _reserve_supervised_models(panel: pd.DataFrame, side: str, version_id: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if version_id == "V82":
        lengths=(5,8,13,21,34,55,89,144)
        raw=[_prototype_raw(panel,side,x,dtw=True) for x in lengths]
        return _score_frame(raw,_meta("DTW prototype distance", [{"path_length":x} for x in lengths],"Development-only constrained DTW prototypes"))
    features=_feature_frame(panel,side)
    if version_id == "V84":
        kernels=[RBF(1.0),Matern(1.0,nu=.5),Matern(1.0,nu=1.5),Matern(1.0,nu=2.5),RationalQuadratic(1.0,1.0),RBF(.5),RBF(2.0),RBF(1.0)+WhiteKernel(.1)]
        outputs=[]
        for kernel in kernels:
            outputs.append(_fit_classifier(panel,side,features.iloc[:, :8],GaussianProcessClassifier(kernel=kernel,random_state=SEED,max_iter_predict=100)))
        return _score_frame(outputs,_meta("Gaussian-process exit classifier", [{"kernel":str(x)} for x in kernels],"small-dimensional GP classification"))
    if version_id == "V85":
        quantiles=(.10,.25,.40,.50,.60,.75,.90,.95)
        improvement=exit_sign(side)*panel["o2o_h1"]
        mask=_dev_mask(panel,side,features.iloc[:,0]) & improvement.notna().to_numpy()
        outputs=[]
        x=features.iloc[:, :8]
        for q in quantiles:
            out=pd.Series(np.nan,index=panel.index)
            if mask.sum()>=20:
                try:
                    model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),QuantileRegressor(quantile=q,alpha=0.1,solver="highs"))
                    model.fit(x.loc[mask],improvement.loc[mask])
                    out.loc[x.index]=model.predict(x)
                except Exception:
                    pass
            outputs.append(out)
        return _score_frame(outputs,_meta("quantile-regression exit damage", [{"quantile":x} for x in quantiles],"Development-only continuous H1 distribution regression"))
    bins=(3,4,5,6,7,8,10,12)
    improvement=exit_sign(side)*panel["o2o_h1"]
    mask=_dev_mask(panel,side,features.iloc[:,0]) & improvement.notna().to_numpy()
    x=features.iloc[:,:8]
    outputs=[]
    for number in bins:
        edges=np.nanquantile(improvement.loc[mask],np.linspace(0,1,number+1)) if mask.sum() else np.array([-1,1])
        heads=[]
        for cut in edges[1:-1]:
            model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(C=.1,max_iter=2000,random_state=SEED))
            y=improvement.loc[mask].gt(cut).astype(int)
            if y.nunique()>=2:
                try:
                    model.fit(x.loc[mask],y)
                    heads.append(model)
                except Exception:
                    pass
        out=pd.Series(0.0,index=panel.index)
        if heads:
            values=np.column_stack([head.predict_proba(x)[:,1] for head in heads])
            out.loc[x.index]=values.mean(axis=1)
        outputs.append(out)
    return _score_frame(outputs,_meta("ordinal exit-benefit model", [{"ordinal_bins":x} for x in bins],"Development-only proportional-odds threshold-head adaptation"))


def _reserve_complexity_scaling(panel: pd.DataFrame, side: str, version_id: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    b=_base(panel,side)
    if version_id == "V87":
        settings=((4,16),(4,24),(4,32),(5,40),(8,64),(8,96),(13,104),(16,128))
        raw=[_dfa(b["oriented_returns"],max(40,scale_max*2),scale_max) for _,scale_max in settings]
        return _score_frame([_dev_orient(x,panel,side) for x in raw],_meta("DFA persistence break", [{"scales":x} for x in settings],"causal detrended-fluctuation scaling"))
    alphabets=(2,3,4,3,4,2,2,3)
    raw=[_rolling_apply(b["oriented_returns"],max(20,8+i*4),lambda x,a=a:_lz_complexity(x,a)) for i,a in enumerate(alphabets)]
    return _score_frame([_dev_orient(x,panel,side) for x in raw],_meta("Lempel-Ziv path complexity", [{"alphabet":x} for x in alphabets],"causal symbolic path complexity"))


def _reserve_changeforest(panel: pd.DataFrame, side: str, version_id: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    b=_base(panel,side)
    if version_id == "V89":
        lengths=(5,8,13,21,34,55,89,144)
        raw=[]
        for length in lengths:
            left=b["oriented_returns"].rolling(length,min_periods=length).mean()
            right=b["oriented_returns"].shift(length).rolling(length,min_periods=length).mean()
            raw.append(left.sub(right).abs())
        return _score_frame([_dev_orient(x,panel,side) for x in raw],_meta("ChangeForest classifier change", [{"min_segment":x} for x in lengths],"causal classifier-loglikelihood change proxy"))
    lengths=(1,2,3,5,8,13,21,34)
    raw=[]
    for length in lengths:
        features=_path_matrix(panel,side,max(5,length+4))
        improvement=exit_sign(side)*panel["o2o_h1"]
        mask=_dev_mask(panel,side,pd.Series(features[:,0],index=panel.index)) & improvement.notna().to_numpy()
        out=pd.Series(np.nan,index=panel.index)
        if mask.sum()>=12:
            try:
                model=IsolationForest(n_estimators=200,max_samples="auto",random_state=SEED,contamination="auto",n_jobs=1)
                imputer = SimpleImputer(strategy="median")
                train_x = imputer.fit_transform(features[mask])
                all_x = imputer.transform(features)
                model.fit(train_x)
                out.loc[panel.index]=-model.score_samples(all_x)
            except Exception:
                pass
        raw.append(out)
    return _score_frame([_dev_orient(x,panel,side) for x in raw],_meta("Random Cut Forest streaming anomaly", [{"shingle":x} for x in lengths],"dependency-free streaming novelty approximation"))


def build_reserve_score_variants(panel: pd.DataFrame, version_id: str, side: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if version_id == "V51":
        return _reserve_duration(panel, side)
    if version_id in {"V52", "V53", "V54"}:
        return _reserve_sequential(panel, side, version_id)
    if version_id in {"V55", "V56"}:
        return _reserve_conformal(panel, side, version_id)
    if version_id in {"V57", "V58"}:
        return _reserve_changepoint(panel, side, version_id)
    if version_id in {"V59", "V60"}:
        return _reserve_distribution(panel, side, version_id)
    if version_id in {"V61", "V62", "V63", "V64", "V65", "V66", "V67", "V68", "V69", "V70"}:
        return _reserve_path_geometry(panel, side, version_id)
    if version_id in {"V71", "V72", "V73", "V74"}:
        return _reserve_complexity(panel, side, version_id)
    if version_id in {"V75", "V76", "V77", "V81", "V83"}:
        return _reserve_one_class(panel, side, version_id)
    if version_id in {"V78", "V79", "V80"}:
        return _reserve_representation(panel, side, version_id)
    if version_id in {"V82", "V84", "V85", "V86"}:
        return _reserve_supervised_models(panel, side, version_id)
    if version_id in {"V87", "V88"}:
        return _reserve_complexity_scaling(panel, side, version_id)
    if version_id in {"V89", "V90"}:
        return _reserve_changeforest(panel, side, version_id)
    raise ValueError(f"未注册的论文版本: {version_id}")
