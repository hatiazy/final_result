from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd

from .design_registry import SCORE_VARIANTS_BY_VERSION
from .logic_registry import LOGIC_BY_VERSION


def _safe_div(a: pd.Series, b: pd.Series | float) -> pd.Series:
    if isinstance(b, pd.Series):
        b = b.where(b.abs() > 1e-12)
    elif abs(float(b)) <= 1e-12:
        b = np.nan
    return a / b


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    minimum = min(window, max(3, window // 2))
    mean = series.rolling(window, min_periods=minimum).mean()
    std = series.rolling(window, min_periods=minimum).std(ddof=0)
    return _safe_div(series - mean, std)


def _causal_percentile(series: pd.Series) -> pd.Series:
    """Percentile of each value against observations available up to that day."""

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        history = values[: i + 1]
        history = np.sort(history[np.isfinite(history)])
        if len(history):
            output[i] = float(np.searchsorted(history, value, side="right") / len(history))
    return pd.Series(output, index=series.index)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    centered = x - x.mean()
    denom = float(np.dot(centered, centered))
    return series.rolling(window, min_periods=window).apply(
        lambda y: float(np.dot(centered, y)) / denom if np.isfinite(y).all() else np.nan,
        raw=True,
    )


def _rolling_rs_hurst(values: np.ndarray) -> float:
    if not np.isfinite(values).all() or len(values) < 8:
        return np.nan
    centered = values - values.mean()
    scale = centered.std(ddof=0)
    if scale <= 1e-12:
        return 0.5
    span = np.cumsum(centered)
    rs = (span.max() - span.min()) / scale
    if rs <= 1e-12:
        return 0.5
    return float(np.log(rs) / np.log(len(values)))


def _cusum_score(z: np.ndarray, allowance: float) -> np.ndarray:
    output = np.full(len(z), np.nan, dtype=float)
    positive = 0.0
    negative = 0.0
    for i, value in enumerate(z):
        if not np.isfinite(value):
            continue
        positive = max(0.0, positive + value - allowance)
        negative = min(0.0, negative + value + allowance)
        output[i] = positive + negative
    return output


def _page_hinkley_score(values: np.ndarray, delta: float, decay: float) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=float)
    mean = 0.0
    weight = 0.0
    pos = 0.0
    neg = 0.0
    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        weight = decay * weight + 1.0
        mean = mean + (value - mean) / weight
        residual = value - mean
        pos = max(0.0, decay * pos + residual - delta)
        neg = min(0.0, decay * neg + residual + delta)
        output[i] = pos + neg
    return output


def _kalman(values: np.ndarray, q_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    level = np.full(len(values), np.nan, dtype=float)
    innovation = np.full(len(values), np.nan, dtype=float)
    state = np.nan
    covariance = 1.0
    obs_var = 1.0
    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        if not np.isfinite(state):
            state = value
            level[i] = state
            innovation[i] = 0.0
            continue
        pred_cov = covariance + q_ratio * obs_var
        residual = value - state
        innovation[i] = residual / math.sqrt(max(pred_cov + obs_var, 1e-12))
        gain = pred_cov / (pred_cov + obs_var)
        state = state + gain * residual
        covariance = (1.0 - gain) * pred_cov
        level[i] = state
    return level, innovation


def _dtw_distance(path: np.ndarray, template: np.ndarray) -> float:
    n, m = len(path), len(template)
    previous = np.full(m + 1, np.inf)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = np.full(m + 1, np.inf)
        for j in range(1, m + 1):
            cost = abs(path[i - 1] - template[j - 1])
            current[j] = cost + min(current[j - 1], previous[j], previous[j - 1])
        previous = current
    return float(previous[m])


def _dtw_direction(values: np.ndarray) -> float:
    if not np.isfinite(values).all() or len(values) < 4:
        return np.nan
    take = np.linspace(0, len(values) - 1, min(12, len(values))).round().astype(int)
    path = values[take]
    scale = np.std(path)
    if scale <= 1e-12:
        return 0.0
    path = (path - path[0]) / scale
    up = np.linspace(0.0, 1.0, len(path))
    down = -up
    return (_dtw_distance(path, down) - _dtw_distance(path, up)) / len(path)


def _hamilton_filter(returns: np.ndarray, persistence: float, scale: float) -> np.ndarray:
    output = np.full(len(returns), np.nan, dtype=float)
    probability_up = 0.5
    variance = max(float(np.nanvar(returns)), 1e-8)
    sigma = math.sqrt(variance)
    mean_size = scale * sigma
    for i, value in enumerate(returns):
        if not np.isfinite(value):
            continue
        prior_up = persistence * probability_up + (1.0 - persistence) * (1.0 - probability_up)
        up_like = math.exp(-0.5 * ((value - mean_size) / sigma) ** 2)
        down_like = math.exp(-0.5 * ((value + mean_size) / sigma) ** 2)
        denominator = prior_up * up_like + (1.0 - prior_up) * down_like
        probability_up = prior_up * up_like / max(denominator, 1e-300)
        output[i] = 2.0 * probability_up - 1.0
        variance = 0.99 * variance + 0.01 * value * value
        sigma = math.sqrt(max(variance, 1e-8))
        mean_size = scale * sigma
    return output


def _base(spot: pd.DataFrame) -> dict[str, pd.Series]:
    table = spot.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    open_ = table["open"].astype(float)
    high = table["high"].astype(float)
    low = table["low"].astype(float)
    close = table["close"].astype(float)
    volume = table["volume"].astype(float)
    amount = table["amount"].astype(float)
    log_close = np.log(close)
    log_open = np.log(open_)
    ret = log_close.diff()
    open_ret = log_open.diff()
    intraday = np.log(close / open_)
    gap = np.log(open_ / close.shift(1))
    log_range = np.log(high / low)
    location = ((close - low) / (high - low).replace(0.0, np.nan) - 0.5) * 2.0
    return {
        "table": table,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "amount": amount, "log_close": log_close,
        "log_open": log_open, "ret": ret, "open_ret": open_ret,
        "intraday": intraday, "gap": gap, "range": log_range,
        "location": location,
    }


def _tree_features(base: dict[str, pd.Series]) -> pd.DataFrame:
    result: dict[str, pd.Series] = {}
    ret = base["ret"]
    log_close = base["log_close"]
    for window in (2, 3, 5, 10, 20, 40, 60, 120):
        result[f"ret_{window}"] = log_close.diff(window)
        result[f"vol_{window}"] = ret.rolling(window, min_periods=max(2, window // 2)).std(ddof=0)
        result[f"range_{window}"] = base["range"].rolling(window, min_periods=max(2, window // 2)).mean()
        result[f"volume_z_{window}"] = _rolling_z(np.log1p(base["volume"]), window)
        result[f"location_{window}"] = base["location"].rolling(window, min_periods=max(2, window // 2)).mean()
    result["intraday"] = base["intraday"]
    result["gap"] = base["gap"]
    result["location"] = base["location"]
    result["amount_change"] = np.log1p(base["amount"]).diff()
    return pd.DataFrame(result).replace([np.inf, -np.inf], np.nan)


def _gradient_boosting_scores(
    spot: pd.DataFrame,
    research_panel: pd.DataFrame,
    side: int,
) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingRegressor

    base = _base(spot)
    features = _tree_features(base).reindex(pd.DatetimeIndex(research_panel["formation_date"]))
    target = side * pd.Series(
        research_panel["o2o_h1"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(research_panel["formation_date"]),
    )
    zero = pd.Series(research_panel["state"].eq(0).to_numpy(), index=features.index)
    development_end = pd.Timestamp("2022-12-31")
    configurations = SCORE_VARIANTS_BY_VERSION["V50"]
    output = pd.DataFrame(index=features.index)
    years = sorted(set(features.index.year))
    for number, config in enumerate(configurations):
        prediction = pd.Series(np.nan, index=features.index, dtype=float)
        for year in years:
            eval_mask = features.index.year == year
            if year <= 2018:
                continue
            train_mask = (
                (features.index.year < year)
                & (features.index <= development_end)
                & zero.to_numpy()
                & target.notna().to_numpy()
            )
            if int(train_mask.sum()) < 100:
                continue
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=float(config["learning_rate"]),
                max_iter=int(config["max_iter"]),
                max_leaf_nodes=int(config["max_leaf_nodes"]),
                max_depth=int(config["max_depth"]),
                min_samples_leaf=int(config["min_samples_leaf"]),
                l2_regularization=float(config["l2_regularization"]),
                random_state=int(config["random_state"]),
            )
            model.fit(features.loc[train_mask], target.loc[train_mask])
            prediction.loc[eval_mask] = model.predict(features.loc[eval_mask])
        output[f"score_{number:02d}"] = prediction
    return output


def _zero_hazard_scores(research_panel: pd.DataFrame, side: int) -> pd.DataFrame:
    panel = research_panel.copy()
    panel["formation_date"] = pd.to_datetime(panel["formation_date"], errors="raise")
    panel = panel.set_index("formation_date").sort_index()
    log_close = np.log(panel["close"].astype(float))
    development = panel.index <= pd.Timestamp("2022-12-31")
    event = panel["state"].eq(0) & panel["next_frozen_state"].eq(side)
    configurations = SCORE_VARIANTS_BY_VERSION["V48"]
    output = pd.DataFrame(index=panel.index)
    for number, config in enumerate(configurations):
        window = int(config["momentum_window"])
        age_width = int(config["age_bin_width"])
        smoothing = float(config["beta_smoothing"])
        momentum = log_close.diff(window)
        dev_momentum = momentum.loc[development & panel["state"].eq(0)].dropna()
        edges = np.unique(dev_momentum.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_numpy())
        if len(edges) < 3:
            momentum_bin = pd.Series(0, index=panel.index)
        else:
            edges[0], edges[-1] = -np.inf, np.inf
            momentum_bin = pd.cut(momentum, bins=edges, labels=False, include_lowest=True).fillna(-1).astype(int)
        age_bin = ((panel["state_age"].astype(int) - 1) // age_width).clip(upper=20)
        train = pd.DataFrame({"age": age_bin, "mom": momentum_bin, "event": event.fillna(False).astype(int)})
        train = train.loc[development & panel["state"].eq(0)]
        grouped = train.groupby(["age", "mom"])["event"].agg(["sum", "count"])
        rates = (grouped["sum"] + smoothing) / (grouped["count"] + 2.0 * smoothing)
        overall = float((train["event"].sum() + smoothing) / (len(train) + 2.0 * smoothing))
        score = pd.Series(
            [rates.get((int(a), int(m)), overall) for a, m in zip(age_bin, momentum_bin)],
            index=panel.index,
            dtype=float,
        )
        # Directional momentum remains a covariate inside the same discrete-hazard logic.
        # The rank is an online normalization.  A full-panel rank would let
        # later/Test observations alter an earlier score.
        score = score + 0.01 * side * _causal_percentile(momentum).fillna(0.5)
        output[f"score_{number:02d}"] = score
    return output


def compute_logic_scores(
    version: str,
    spot: pd.DataFrame,
    research_panel: pd.DataFrame,
    side: int,
) -> pd.DataFrame:
    """Return 16 causal scores; larger always means stronger support for `side`."""

    spec = LOGIC_BY_VERSION[version]
    if side not in (-1, 1):
        raise ValueError("side must be -1 or 1")
    if int(version[1:]) >= 51:
        from .reserve_features import compute_reserve_scores

        return compute_reserve_scores(version, spot, research_panel, side)
    if spec.method_key == "gradient_boosting":
        return _gradient_boosting_scores(spot, research_panel, side)
    if spec.method_key == "zero_hazard":
        return _zero_hazard_scores(research_panel, side)

    b = _base(spot)
    index = b["close"].index
    raw = pd.DataFrame(index=index)
    method = spec.method_key
    ret, log_close = b["ret"], b["log_close"]

    configurations = SCORE_VARIANTS_BY_VERSION[version]
    for number, config in enumerate(configurations):
        window = int(config["window"])
        minimum = max(2, window // 2)
        vol = ret.rolling(window, min_periods=minimum).std(ddof=0)
        momentum = log_close.diff(window) / max(window, 1)

        if method == "close_momentum":
            score = _safe_div(log_close.diff(window), vol * math.sqrt(window))
        elif method == "open_momentum":
            score = _safe_div(b["log_open"].diff(window), b["open_ret"].rolling(window, min_periods=minimum).std(ddof=0) * math.sqrt(window))
        elif method == "intraday_persistence":
            score = _safe_div(b["intraday"].rolling(window, min_periods=minimum).mean(), b["intraday"].rolling(window, min_periods=minimum).std(ddof=0))
        elif method == "gap_persistence":
            score = _safe_div(b["gap"].rolling(window, min_periods=minimum).mean(), b["gap"].rolling(window, min_periods=minimum).std(ddof=0))
        elif method == "close_location":
            score = _safe_div(b["location"].rolling(window, min_periods=minimum).mean(), b["location"].rolling(window, min_periods=minimum).std(ddof=0))
        elif method == "sma_distance":
            score = _safe_div(log_close - log_close.rolling(window, min_periods=minimum).mean(), vol)
        elif method == "ema_crossover":
            fast = max(2, window // 3)
            score = _safe_div(log_close.ewm(span=fast, adjust=False, min_periods=fast).mean() - log_close.ewm(span=window, adjust=False, min_periods=window).mean(), vol)
        elif method == "donchian_breakout":
            upper = b["high"].shift(1).rolling(window, min_periods=minimum).max()
            lower = b["low"].shift(1).rolling(window, min_periods=minimum).min()
            score = _safe_div(b["close"] - (upper + lower) / 2.0, upper - lower)
        elif method == "bollinger_z":
            score = _rolling_z(log_close, window)
        elif method == "ols_slope":
            slope = _rolling_slope(log_close, window)
            score = _safe_div(slope * math.sqrt(window), vol)
        elif method == "robust_median_trend":
            lags = sorted(set([max(1, window // 4), max(2, window // 2), window]))
            score = pd.concat([log_close.diff(lag) / lag for lag in lags], axis=1).median(axis=1)
            score = _safe_div(score, vol)
        elif method == "momentum_acceleration":
            short = max(2, window // 3)
            score = _safe_div(log_close.diff(short) / short - log_close.diff(window) / window, vol)
        elif method == "rsi":
            gain = ret.clip(lower=0).rolling(window, min_periods=minimum).mean()
            loss = (-ret.clip(upper=0)).rolling(window, min_periods=minimum).mean()
            score = _safe_div(gain - loss, gain + loss)
        elif method == "stochastic":
            upper = b["high"].rolling(window, min_periods=minimum).max()
            lower = b["low"].rolling(window, min_periods=minimum).min()
            score = 2.0 * _safe_div(b["close"] - lower, upper - lower) - 1.0
        elif method == "macd":
            fast = max(2, window // 3)
            slow = max(fast + 1, window)
            macd = log_close.ewm(span=fast, adjust=False).mean() - log_close.ewm(span=slow, adjust=False).mean()
            signal_span = max(2, int(math.sqrt(window)))
            score = _safe_div(macd - macd.ewm(span=signal_span, adjust=False).mean(), vol)
        elif method == "cusum":
            baseline = ret.rolling(window, min_periods=minimum).mean().shift(1)
            scale = ret.rolling(window, min_periods=minimum).std(ddof=0).shift(1)
            z = _safe_div(ret - baseline, scale)
            score = pd.Series(_cusum_score(z.to_numpy(dtype=float), float(config["allowance"])), index=index)
        elif method == "page_hinkley":
            z = _safe_div(ret, ret.rolling(window, min_periods=minimum).std(ddof=0).shift(1))
            score = pd.Series(
                _page_hinkley_score(
                    z.to_numpy(dtype=float),
                    float(config["delta"]),
                    float(config["decay"]),
                ),
                index=index,
            )
        elif method == "ewma_control":
            alpha = 2.0 / (window + 1.0)
            mean = ret.ewm(alpha=alpha, adjust=False, min_periods=minimum).mean()
            variance = (ret - mean.shift(1)).pow(2).ewm(alpha=alpha, adjust=False, min_periods=minimum).mean()
            score = _safe_div(mean, np.sqrt(variance))
        elif method in {"kalman_slope", "kalman_innovation"}:
            q_ratio = float(config["process_to_observation_variance_ratio"])
            level, innovation = _kalman(log_close.to_numpy(dtype=float), q_ratio)
            if method == "kalman_slope":
                score = _safe_div(pd.Series(level, index=index).diff(), vol)
            else:
                score = pd.Series(innovation, index=index)
        elif method == "bocpd_proxy":
            half = max(2, window // 2)
            recent = ret.rolling(half, min_periods=half).mean()
            previous = ret.shift(half).rolling(half, min_periods=half).mean()
            pooled = ret.rolling(2 * half, min_periods=2 * half).std(ddof=0)
            delta = _safe_div(recent - previous, pooled / math.sqrt(half))
            score = np.sign(delta) * np.log1p(delta.pow(2))
        elif method == "pelt_proxy":
            half = max(2, window // 2)
            recent = ret.rolling(half, min_periods=half).mean()
            previous = ret.shift(half).rolling(half, min_periods=half).mean()
            full_var = ret.rolling(2 * half, min_periods=2 * half).var(ddof=0)
            split_var = 0.5 * (ret.rolling(half, min_periods=half).var(ddof=0) + ret.shift(half).rolling(half, min_periods=half).var(ddof=0))
            improvement = ((full_var - split_var) / full_var.replace(0.0, np.nan)).clip(lower=0.0)
            score = np.sign(recent - previous) * np.sqrt(improvement)
        elif method == "variance_ratio":
            lag = int(config["aggregation_lag"])
            kret = log_close.diff(lag)
            vr = _safe_div(kret.rolling(window, min_periods=minimum).var(ddof=0), lag * ret.rolling(window, min_periods=minimum).var(ddof=0))
            score = ret.rolling(lag, min_periods=lag).sum() * (2.0 * vr - 1.0)
        elif method == "autocorrelation":
            lag = int(config["lag"])
            autocorr = ret.rolling(window, min_periods=minimum).corr(ret.shift(lag))
            score = autocorr * ret.rolling(lag, min_periods=lag).sum()
        elif method == "sign_runs":
            signs = np.sign(ret).replace(0.0, np.nan)
            imbalance = signs.rolling(window, min_periods=minimum).mean()
            persistence = signs.eq(signs.shift()).rolling(window, min_periods=minimum).mean() * 2.0 - 1.0
            score = imbalance * (1.0 + persistence)
        elif method == "hurst_rs":
            hurst = ret.rolling(window, min_periods=max(8, window)).apply(_rolling_rs_hurst, raw=True)
            score = np.sign(log_close.diff(max(2, window // 4))) * (hurst - 0.5)
        elif method == "volatility_regime":
            short = max(2, window // 4)
            short_vol = ret.rolling(short, min_periods=short).std(ddof=0)
            efficiency = _safe_div(log_close.diff(window).abs(), ret.abs().rolling(window, min_periods=minimum).sum())
            score = _safe_div(momentum, vol) * efficiency * _safe_div(vol, short_vol)
        elif method == "parkinson":
            variance = b["range"].pow(2).rolling(window, min_periods=minimum).mean() / (4.0 * math.log(2.0))
            score = _safe_div(momentum, np.sqrt(variance))
        elif method == "garman_klass":
            oc = np.log(b["close"] / b["open"])
            variance_daily = 0.5 * b["range"].pow(2) - (2.0 * math.log(2.0) - 1.0) * oc.pow(2)
            variance = variance_daily.clip(lower=0.0).rolling(window, min_periods=minimum).mean()
            score = _safe_div(momentum, np.sqrt(variance))
        elif method == "rogers_satchell":
            ho = np.log(b["high"] / b["open"])
            hc = np.log(b["high"] / b["close"])
            lo = np.log(b["low"] / b["open"])
            lc = np.log(b["low"] / b["close"])
            variance = (ho * hc + lo * lc).clip(lower=0.0).rolling(window, min_periods=minimum).mean()
            score = _safe_div(momentum, np.sqrt(variance))
        elif method == "volume_price_corr":
            volume_change = np.log1p(b["volume"]).diff()
            corr = ret.rolling(window, min_periods=minimum).corr(volume_change)
            score = _safe_div(momentum, vol) * corr
        elif method == "obv_slope":
            obv = (np.sign(ret).fillna(0.0) * b["volume"]).cumsum()
            score = _safe_div(_rolling_slope(obv, window), b["volume"].rolling(window, min_periods=minimum).mean())
        elif method == "money_flow":
            typical = (b["high"] + b["low"] + b["close"]) / 3.0
            flow = typical * b["volume"]
            positive = flow.where(typical.diff() > 0, 0.0).rolling(window, min_periods=minimum).sum()
            negative = flow.where(typical.diff() < 0, 0.0).rolling(window, min_periods=minimum).sum()
            score = _safe_div(positive - negative, positive + negative)
        elif method == "chaikin_ad":
            multiplier = _safe_div(2.0 * b["close"] - b["high"] - b["low"], b["high"] - b["low"])
            ad = (multiplier.fillna(0.0) * b["volume"]).cumsum()
            score = _safe_div(_rolling_slope(ad, window), b["volume"].rolling(window, min_periods=minimum).mean())
        elif method == "vwap_proxy":
            typical = (b["high"] + b["low"] + b["close"]) / 3.0
            weighted = (typical * b["volume"]).rolling(window, min_periods=minimum).sum()
            volume_sum = b["volume"].rolling(window, min_periods=minimum).sum()
            proxy = _safe_div(weighted, volume_sum)
            score = _safe_div(np.log(b["close"] / proxy), vol)
        elif method == "turnover_surprise":
            surprise = _rolling_z(np.log1p(b["amount"]), window)
            score = surprise * np.sign(log_close.diff(max(1, window // 4)))
        elif method == "signed_volume":
            signed = np.sign(ret).fillna(0.0) * b["volume"]
            score = _safe_div(signed.rolling(window, min_periods=minimum).sum(), b["volume"].rolling(window, min_periods=minimum).sum())
        elif method == "volume_breakout":
            volume_z = _rolling_z(np.log1p(b["volume"]), window)
            score = volume_z * np.sign(log_close.diff(max(1, window // 4)))
        elif method == "sign_entropy":
            probability = ret.gt(0).rolling(window, min_periods=minimum).mean().clip(1e-6, 1.0 - 1e-6)
            entropy = -(probability * np.log2(probability) + (1.0 - probability) * np.log2(1.0 - probability))
            score = (2.0 * probability - 1.0) * (1.0 - entropy)
        elif method == "path_efficiency":
            score = _safe_div(log_close.diff(window), ret.abs().rolling(window, min_periods=minimum).sum())
        elif method == "drawdown_recovery":
            peak = b["close"].rolling(window, min_periods=minimum).max()
            drawdown = np.log(b["close"] / peak)
            recovery = drawdown.diff(max(1, window // 4))
            score = _safe_div(recovery + drawdown / window, vol)
        elif method == "peak_distance":
            peak = b["close"].rolling(window, min_periods=minimum).max()
            score = _safe_div(np.log(b["close"] / peak), vol)
        elif method == "trough_distance":
            trough = b["close"].rolling(window, min_periods=minimum).min()
            score = _safe_div(np.log(b["close"] / trough), vol)
        elif method == "ulcer_direction":
            peak = b["close"].rolling(window, min_periods=minimum).max()
            drawdown = np.log(b["close"] / peak)
            ulcer = np.sqrt(drawdown.pow(2).rolling(window, min_periods=minimum).mean())
            score = _safe_div(momentum, ulcer)
        elif method == "excursion_balance":
            favorable_up = np.log(b["high"] / b["open"])
            adverse_up = np.log(b["open"] / b["low"])
            score = _safe_div((favorable_up - adverse_up).rolling(window, min_periods=minimum).mean(), b["range"].rolling(window, min_periods=minimum).mean())
        elif method == "kernel_path":
            bandwidth = max(1.0, window / float(config["bandwidth_divisor"]))
            weights = np.exp(-0.5 * ((np.arange(window) - (window - 1)) / bandwidth) ** 2)
            x = np.arange(window, dtype=float)
            center = np.average(x, weights=weights)
            denominator = float(np.sum(weights * (x - center) ** 2))
            score = log_close.rolling(window, min_periods=window).apply(
                lambda y: float(np.sum(weights * (x - center) * y)) / denominator,
                raw=True,
            )
            score = _safe_div(score, vol)
        elif method == "dtw_template":
            score = log_close.rolling(window, min_periods=window).apply(_dtw_direction, raw=True)
        elif method == "hamilton_filter":
            persistence = float(config["persistence"])
            scale = float(config["state_mean_scale"])
            score = pd.Series(_hamilton_filter(ret.to_numpy(dtype=float), persistence, scale), index=index)
        else:
            raise KeyError(f"Unsupported method: {method}")

        raw[f"score_{number:02d}"] = pd.Series(score, index=index).replace([np.inf, -np.inf], np.nan)

    aligned = raw.reindex(pd.DatetimeIndex(research_panel["formation_date"]))
    # Every downstream candidate treats larger values as stronger evidence for its own side.
    return aligned * float(side)
