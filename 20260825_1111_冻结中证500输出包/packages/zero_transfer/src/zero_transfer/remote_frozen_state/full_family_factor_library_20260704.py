from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    params: Mapping[str, int | float] = field(default_factory=dict)
    risk_direction: int = 1
    transform: str = "raw"
    transform_window: int | None = None
    transform_min_periods: int | None = None
    transform_stats_lag: int = 1
    transform_clip: float | None = None


def _causal_price_index(frame: pd.DataFrame) -> pd.Series:
    """由 adjusted_ret 递推研究价格指数，避免后复权终点锚定影响。"""

    ret = pd.to_numeric(frame["adjusted_ret"], errors="coerce").fillna(0.0)
    return (1.0 + ret).cumprod()


def _momentum(frame: pd.DataFrame, window: int) -> pd.Series:
    price = _causal_price_index(frame)
    return price.pct_change(window, fill_method=None)


def _vol_adjusted_momentum(
    frame: pd.DataFrame,
    window: int,
    vol_window: int,
) -> pd.Series:
    mom = _momentum(frame, window)
    vol = frame["adjusted_ret"].rolling(vol_window, min_periods=vol_window).std(ddof=0)
    return mom / (vol * np.sqrt(window)).replace(0.0, np.nan)


def _realized_vol(frame: pd.DataFrame, window: int) -> pd.Series:
    return frame["adjusted_ret"].rolling(window, min_periods=window).std(ddof=0)


def _downside_semivol(frame: pd.DataFrame, window: int) -> pd.Series:
    downside_sq = frame["adjusted_ret"].clip(upper=0).pow(2)
    return downside_sq.rolling(window, min_periods=window).mean().pow(0.5)


def _double_ma(frame: pd.DataFrame, short_window: int, long_window: int) -> pd.Series:
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")
    price = _causal_price_index(frame)
    short_ma = price.rolling(short_window, min_periods=short_window).mean()
    long_ma = price.rolling(long_window, min_periods=long_window).mean()
    return short_ma / long_ma - 1.0


def _ma_slope(frame: pd.DataFrame, ma_window: int, slope_window: int) -> pd.Series:
    if slope_window > ma_window:
        raise ValueError("slope_window must be <= ma_window")
    price = _causal_price_index(frame)
    ma = price.rolling(ma_window, min_periods=ma_window).mean()
    return ma / ma.shift(slope_window) - 1.0


def _macd(
    frame: pd.DataFrame,
    fast_window: int,
    slow_window: int,
    signal_window: int,
) -> pd.Series:
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window")
    price = _causal_price_index(frame)
    fast = price.ewm(span=fast_window, adjust=False, min_periods=fast_window).mean()
    slow = price.ewm(span=slow_window, adjust=False, min_periods=slow_window).mean()
    diff = fast - slow
    dea = diff.ewm(span=signal_window, adjust=False, min_periods=signal_window).mean()
    return (diff - dea) / price.replace(0.0, np.nan)


def _breakout_strength(frame: pd.DataFrame, window: int) -> pd.Series:
    price = _causal_price_index(frame)
    rolling_high = price.rolling(window, min_periods=window).max()
    rolling_low = price.rolling(window, min_periods=window).min()
    center = (rolling_high + rolling_low) / 2.0
    width = (rolling_high - rolling_low).replace(0.0, np.nan)
    return (price - center) / width


def _efficiency_ratio(frame: pd.DataFrame, window: int) -> pd.Series:
    price = _causal_price_index(frame)
    net_move = price.diff(window)
    path_move = price.diff().abs().rolling(window, min_periods=window).sum()
    return net_move / path_move.replace(0.0, np.nan)


def _trend_r2(frame: pd.DataFrame, window: int) -> pd.Series:
    price = _causal_price_index(frame)
    time_index = pd.Series(np.arange(len(price), dtype=float), index=price.index)
    corr = price.rolling(window, min_periods=window).corr(time_index)
    return np.sign(price.diff(window)) * corr.pow(2)


def _worst_k_loss(frame: pd.DataFrame, window: int, k: int) -> pd.Series:
    if not 1 <= k <= window:
        raise ValueError("k must satisfy 1 <= k <= window")

    def worst_mean(values: np.ndarray) -> float:
        return float(np.partition(values, k - 1)[:k].mean())

    return frame["adjusted_ret"].rolling(window, min_periods=window).apply(
        worst_mean, raw=True
    )


def _return_skew(frame: pd.DataFrame, window: int) -> pd.Series:
    return frame["adjusted_ret"].rolling(window, min_periods=window).skew()


def _drawdown(frame: pd.DataFrame, window: int) -> pd.Series:
    price = _causal_price_index(frame)
    prior_high = price.rolling(window, min_periods=window).max()
    return price / prior_high - 1.0


def _drawdown_speed(frame: pd.DataFrame, window: int, delta: int) -> pd.Series:
    drawdown = _drawdown(frame, window)
    return drawdown - drawdown.shift(delta)


def _rolling_max_drawdown(price: pd.Series, window: int) -> pd.Series:
    def max_drawdown(values: np.ndarray) -> float:
        peaks = np.maximum.accumulate(values)
        dd = values / peaks - 1.0
        return float(np.nanmin(dd))

    return price.rolling(window, min_periods=window).apply(max_drawdown, raw=True)


def _calmar_momentum(frame: pd.DataFrame, window: int) -> pd.Series:
    price = _causal_price_index(frame)
    mom = price.pct_change(window, fill_method=None)
    max_dd = _rolling_max_drawdown(price, window).abs()
    return mom / max_dd.replace(0.0, np.nan)


def _ulcer_momentum(frame: pd.DataFrame, window: int) -> pd.Series:
    price = _causal_price_index(frame)
    rolling_high = price.rolling(window, min_periods=window).max()
    drawdown = price / rolling_high - 1.0
    ulcer = drawdown.pow(2).rolling(window, min_periods=window).mean().pow(0.5)
    mom = price.pct_change(window, fill_method=None)
    return mom / ulcer.replace(0.0, np.nan)


def _gain_to_pain(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = frame["adjusted_ret"]
    net_return = ret.rolling(window, min_periods=window).sum()
    pain = ret.clip(upper=0).abs().rolling(window, min_periods=window).sum()
    return net_return / pain.replace(0.0, np.nan)


def _omega_ratio(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = frame["adjusted_ret"]
    upside = ret.clip(lower=0).rolling(window, min_periods=window).sum()
    downside = ret.clip(upper=0).abs().rolling(window, min_periods=window).sum()
    return np.log((upside + 1e-12) / (downside + 1e-12))


def _downside_vol_momentum(frame: pd.DataFrame, window: int) -> pd.Series:
    mom = _momentum(frame, window)
    downside = _downside_semivol(frame, window)
    return mom / (downside * np.sqrt(window)).replace(0.0, np.nan)


def _convexity_momentum(
    frame: pd.DataFrame,
    window: int,
    vol_window: int,
    mom_weight: float,
) -> pd.Series:
    price = _causal_price_index(frame)
    ret = frame["adjusted_ret"]
    vol = ret.rolling(vol_window, min_periods=vol_window).std(ddof=0)
    start = price.shift(window)
    mom = price / start - 1.0
    midpoint = (start + price) / 2.0
    avg_price = price.rolling(window, min_periods=window).mean()
    convexity = (midpoint - avg_price) / price.replace(0.0, np.nan)
    mom_scaled = mom / (vol * np.sqrt(window)).replace(0.0, np.nan)
    convexity_scaled = convexity / vol.replace(0.0, np.nan)
    return float(mom_weight) * mom_scaled + (1.0 - float(mom_weight)) * convexity_scaled


def _frog_in_pan(frame: pd.DataFrame, trend_window: int, freq_window: int) -> pd.Series:
    price = _causal_price_index(frame)
    ret = frame["adjusted_ret"].fillna(0.0)
    n_pos = (ret > 0).astype(float).rolling(freq_window, min_periods=freq_window).sum()
    n_neg = (ret < 0).astype(float).rolling(freq_window, min_periods=freq_window).sum()
    freq_score = (n_pos - n_neg) / freq_window
    trend_sign = np.sign(price.pct_change(trend_window, fill_method=None))
    # 只有长趋势与涨跌频率一致时保留信号，并维持原趋势方向。
    consistency = (trend_sign * freq_score).clip(lower=0.0)
    return trend_sign * consistency


def _up_down_frequency(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = frame["adjusted_ret"].fillna(0.0)
    n_pos = (ret > 0).astype(float).rolling(window, min_periods=window).sum()
    n_neg = (ret < 0).astype(float).rolling(window, min_periods=window).sum()
    return (n_pos - n_neg) / window


def _aroon_oscillator(frame: pd.DataFrame, window: int) -> pd.Series:
    price = _causal_price_index(frame)

    def oscillator(values: np.ndarray) -> float:
        periods = len(values) - 1
        # Aroon使用“距最近一次高/低点”的天数；反向搜索可正确处理并列极值。
        since_high = int(np.argmax(values[::-1]))
        since_low = int(np.argmin(values[::-1]))
        return float((since_low - since_high) / periods)

    return price.rolling(window + 1, min_periods=window + 1).apply(
        oscillator, raw=True
    )


def _donchian_position(frame: pd.DataFrame, window: int) -> pd.Series:
    price = _causal_price_index(frame)
    prior = price.shift(1)
    high = prior.rolling(window, min_periods=window).max()
    low = prior.rolling(window, min_periods=window).min()
    return (price - low) / (high - low).replace(0.0, np.nan) - 0.5


def _distance_from_low(frame: pd.DataFrame, window: int) -> pd.Series:
    price = _causal_price_index(frame)
    low = price.shift(1).rolling(window, min_periods=window).min()
    return price / low.replace(0.0, np.nan) - 1.0


def _atr_breakout_strength(
    frame: pd.DataFrame,
    channel_window: int,
    atr_window: int,
    atr_weight: float,
) -> pd.Series:
    high = pd.to_numeric(frame["adjusted_high"], errors="coerce")
    low = pd.to_numeric(frame["adjusted_low"], errors="coerce")
    close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(atr_window, min_periods=atr_window).mean()
    prior = close.shift(1)
    channel_high = prior.rolling(
        channel_window, min_periods=channel_window
    ).max()
    channel_low = prior.rolling(
        channel_window, min_periods=channel_window
    ).min()
    center = (channel_high + channel_low) / 2.0
    scale = (channel_high - channel_low) / 2.0 + float(atr_weight) * atr
    return (close - center) / scale.replace(0.0, np.nan)


def _trix(frame: pd.DataFrame, window: int, signal_window: int) -> pd.Series:
    price = _causal_price_index(frame)
    ema1 = price.ewm(span=window, adjust=False, min_periods=window).mean()
    ema2 = ema1.ewm(span=window, adjust=False, min_periods=window).mean()
    ema3 = ema2.ewm(span=window, adjust=False, min_periods=window).mean()
    trix = ema3.pct_change(fill_method=None)
    signal = trix.ewm(
        span=signal_window, adjust=False, min_periods=signal_window
    ).mean()
    return trix - signal


def _stable_momentum(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = frame["adjusted_ret"]
    mean_ret = ret.rolling(window, min_periods=window).mean()
    mean_abs_ret = ret.abs().rolling(window, min_periods=window).mean()
    return mean_ret / mean_abs_ret.replace(0.0, np.nan)


def _recovery_slope_vol_adjusted(
    frame: pd.DataFrame,
    window: int,
    vol_window: int,
) -> pd.Series:
    price = _causal_price_index(frame)

    def recovery_slope(values: np.ndarray) -> float:
        low_position = int(np.argmin(values))
        days_since_low = len(values) - 1 - low_position
        recovery = values[-1] / values[low_position] - 1.0
        return float(recovery / (days_since_low + 1))

    slope = price.rolling(window, min_periods=window).apply(
        recovery_slope, raw=True
    )
    vol = _realized_vol(frame, vol_window)
    return slope / vol.replace(0.0, np.nan)


def _left_tail_loss_ratio(frame: pd.DataFrame, window: int, k: int) -> pd.Series:
    if not 1 <= k <= window:
        raise ValueError("k must satisfy 1 <= k <= window")

    def ratio(values: np.ndarray) -> float:
        losses = np.clip(-values, 0.0, None)
        total = losses.sum()
        if total <= 0:
            return 0.0
        return float(np.partition(losses, len(losses) - k)[-k:].sum() / total)

    return frame["adjusted_ret"].rolling(window, min_periods=window).apply(
        ratio, raw=True
    )


def _trend_convexity_proxy(
    frame: pd.DataFrame,
    trend_window: int,
    vol_window: int,
    vol_penalty: float,
) -> pd.Series:
    mom = _momentum(frame, trend_window)
    vol = _realized_vol(frame, vol_window) * np.sqrt(trend_window)
    magnitude = (mom.abs() - float(vol_penalty) * vol).clip(lower=0.0)
    return np.sign(mom) * magnitude


def _intraday_purity(frame: pd.DataFrame, window: int) -> pd.Series:
    open_price = pd.to_numeric(frame["adjusted_open"], errors="coerce")
    close_price = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    intraday_ret = close_price / open_price.replace(0.0, np.nan) - 1.0
    return intraday_ret.rolling(window, min_periods=window).sum()


def _volume_shock(frame: pd.DataFrame, window: int) -> pd.Series:
    volume = frame["volume"]
    baseline = volume.rolling(window, min_periods=window).median()
    return volume / baseline - 1.0


def _oi_change(frame: pd.DataFrame, window: int) -> pd.Series:
    return frame["oi"].pct_change(window, fill_method=None)


def _range_expansion(frame: pd.DataFrame, window: int) -> pd.Series:
    # 使用后复权OHLC，避免主力换月造成虚假的振幅扩张。
    width = pd.to_numeric(frame["adjusted_high"], errors="coerce") - pd.to_numeric(
        frame["adjusted_low"], errors="coerce"
    )
    denominator = pd.to_numeric(
        frame["adjusted_close"], errors="coerce"
    ).shift(1).abs()
    daily_range = width / denominator
    baseline = daily_range.shift(1).rolling(window, min_periods=window).median()
    return daily_range / baseline - 1.0


def _volume_oi_pressure(frame: pd.DataFrame, window: int) -> pd.Series:
    volume = pd.to_numeric(frame["volume"], errors="coerce").replace(0.0, np.nan)
    log_volume = np.log(volume)
    history = log_volume.shift(1)
    mean = history.rolling(window, min_periods=max(5, window // 2)).mean()
    std = history.rolling(window, min_periods=max(5, window // 2)).std(ddof=0)
    vol_z = (log_volume - mean) / std.replace(0.0, np.nan)
    oi_chg = _oi_change(frame, window)
    return vol_z * np.sign(oi_chg)


def _volume_adjusted_momentum(
    frame: pd.DataFrame,
    momentum_window: int,
    volume_window: int,
) -> pd.Series:
    volume = pd.to_numeric(frame["volume"], errors="coerce").replace(0.0, np.nan)
    baseline = volume.shift(1).rolling(
        volume_window, min_periods=volume_window
    ).median()
    relative_volume = volume / baseline.replace(0.0, np.nan)
    return _momentum(frame, momentum_window) * relative_volume


def _volume_return_confirm(
    frame: pd.DataFrame,
    volume_window: int,
    return_window: int,
) -> pd.Series:
    volume = pd.to_numeric(frame["volume"], errors="coerce").replace(0.0, np.nan)
    log_volume = np.log(volume)
    history = log_volume.shift(1)
    mean = history.rolling(volume_window, min_periods=volume_window).mean()
    std = history.rolling(volume_window, min_periods=volume_window).std(ddof=0)
    volume_z = (log_volume - mean) / std.replace(0.0, np.nan)
    return volume_z * _momentum(frame, return_window)


def _price_volume_correlation(frame: pd.DataFrame, window: int) -> pd.Series:
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    return frame["adjusted_ret"].rolling(
        window, min_periods=window
    ).corr(volume.pct_change(fill_method=None))


def _oi_pressure_down(
    frame: pd.DataFrame,
    oi_window: int,
    return_window: int,
) -> pd.Series:
    oi_increase = _oi_change(frame, oi_window).clip(lower=0.0)
    negative_trend = (-_momentum(frame, return_window)).clip(lower=0.0)
    return oi_increase * negative_trend


def _range_expansion_down(frame: pd.DataFrame, window: int) -> pd.Series:
    high = pd.to_numeric(frame["adjusted_high"], errors="coerce")
    low = pd.to_numeric(frame["adjusted_low"], errors="coerce")
    close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    width = (high - low).replace(0.0, np.nan)
    daily_range = width / close.shift(1).abs().replace(0.0, np.nan)
    baseline = daily_range.shift(1).rolling(window, min_periods=window).median()
    expansion = (daily_range / baseline.replace(0.0, np.nan) - 1.0).clip(lower=0.0)
    close_near_low = ((high - close) / width).clip(0.0, 1.0)
    return expansion * close_near_low


def _multi_speed_tsmom_vote(
    frame: pd.DataFrame,
    fast_window: int,
    medium_window: int,
    slow_window: int,
) -> pd.Series:
    votes = pd.concat(
        [
            np.sign(_momentum(frame, fast_window)),
            np.sign(_momentum(frame, medium_window)),
            np.sign(_momentum(frame, slow_window)),
        ],
        axis=1,
    )
    return votes.mean(axis=1)


def _high_vol_negative_trend(
    frame: pd.DataFrame,
    trend_window: int,
    short_vol_window: int,
    long_vol_window: int,
) -> pd.Series:
    short_vol = _realized_vol(frame, short_vol_window)
    long_vol = frame["adjusted_ret"].shift(1).rolling(
        long_vol_window, min_periods=long_vol_window
    ).std(ddof=0)
    vol_expansion = (short_vol / long_vol.replace(0.0, np.nan) - 1.0).clip(
        lower=0.0
    )
    negative_trend = (-_momentum(frame, trend_window)).clip(lower=0.0)
    return vol_expansion * negative_trend


def _drawdown_acceleration_down(
    frame: pd.DataFrame,
    drawdown_window: int,
    delta: int,
    trend_window: int,
) -> pd.Series:
    worsening = (-_drawdown_speed(frame, drawdown_window, delta)).clip(lower=0.0)
    negative_trend = (-_momentum(frame, trend_window)).clip(lower=0.0)
    return worsening * negative_trend


def _gap_return(frame: pd.DataFrame, window: int) -> pd.Series:
    open_price = pd.to_numeric(frame["adjusted_open"], errors="coerce")
    prev_close = pd.to_numeric(frame["adjusted_close"], errors="coerce").shift(1)
    gap = open_price / prev_close.replace(0.0, np.nan) - 1.0
    return gap.rolling(window, min_periods=window).mean()


def _intraday_return(frame: pd.DataFrame, window: int) -> pd.Series:
    open_price = pd.to_numeric(frame["adjusted_open"], errors="coerce")
    close_price = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    intraday = close_price / open_price.replace(0.0, np.nan) - 1.0
    return intraday.rolling(window, min_periods=window).mean()


def _gap_fill_failure(frame: pd.DataFrame, window: int) -> pd.Series:
    open_price = pd.to_numeric(frame["adjusted_open"], errors="coerce")
    close_price = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    prev_close = close_price.shift(1)
    gap_down = (prev_close - open_price).clip(lower=0.0) / prev_close.abs().replace(0.0, np.nan)
    recovered = (close_price - open_price).clip(lower=0.0) / prev_close.abs().replace(0.0, np.nan)
    failure = (gap_down - recovered).clip(lower=0.0)
    return failure.rolling(window, min_periods=window).mean()


def _close_location(frame: pd.DataFrame, window: int) -> pd.Series:
    high = pd.to_numeric(frame["adjusted_high"], errors="coerce")
    low = pd.to_numeric(frame["adjusted_low"], errors="coerce")
    close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    location = (close - low) / (high - low).replace(0.0, np.nan) - 0.5
    return location.rolling(window, min_periods=window).mean()


def _upper_shadow_pressure(frame: pd.DataFrame, window: int) -> pd.Series:
    open_price = pd.to_numeric(frame["adjusted_open"], errors="coerce")
    close_price = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    high = pd.to_numeric(frame["adjusted_high"], errors="coerce")
    low = pd.to_numeric(frame["adjusted_low"], errors="coerce")
    upper = high - pd.concat([open_price, close_price], axis=1).max(axis=1)
    body_direction = np.sign(close_price / open_price.replace(0.0, np.nan) - 1.0)
    pressure = upper / (high - low).replace(0.0, np.nan) * body_direction.clip(lower=0.0)
    return pressure.rolling(window, min_periods=window).mean()


def _lower_shadow_support(frame: pd.DataFrame, window: int) -> pd.Series:
    open_price = pd.to_numeric(frame["adjusted_open"], errors="coerce")
    close_price = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    high = pd.to_numeric(frame["adjusted_high"], errors="coerce")
    low = pd.to_numeric(frame["adjusted_low"], errors="coerce")
    lower = pd.concat([open_price, close_price], axis=1).min(axis=1) - low
    support = lower / (high - low).replace(0.0, np.nan)
    return support.rolling(window, min_periods=window).mean()


def _vol_ratio(frame: pd.DataFrame, short_window: int, long_window: int) -> pd.Series:
    short = _realized_vol(frame, short_window)
    long = frame["adjusted_ret"].shift(1).rolling(long_window, min_periods=long_window).std(ddof=0)
    return short / long.replace(0.0, np.nan) - 1.0


def _downside_vol_ratio(frame: pd.DataFrame, short_window: int, long_window: int) -> pd.Series:
    short = _downside_semivol(frame, short_window)
    downside = frame["adjusted_ret"].clip(upper=0.0).pow(2)
    long = downside.shift(1).rolling(long_window, min_periods=long_window).mean().pow(0.5)
    return short / long.replace(0.0, np.nan) - 1.0


def _vol_of_vol(frame: pd.DataFrame, vol_window: int, smooth_window: int) -> pd.Series:
    vol = _realized_vol(frame, vol_window)
    return vol.rolling(smooth_window, min_periods=smooth_window).std(ddof=0)


def _return_jump_count(frame: pd.DataFrame, window: int, multiplier: float) -> pd.Series:
    ret = pd.to_numeric(frame["adjusted_ret"], errors="coerce")
    baseline = ret.shift(1).rolling(window, min_periods=window).std(ddof=0)
    jump = ret.abs().gt(float(multiplier) * baseline.replace(0.0, np.nan)).astype(float)
    return jump.rolling(window, min_periods=window).mean()


def _range_vol_ratio(frame: pd.DataFrame, short_window: int, long_window: int) -> pd.Series:
    high = pd.to_numeric(frame["adjusted_high"], errors="coerce")
    low = pd.to_numeric(frame["adjusted_low"], errors="coerce")
    close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    daily_range = (high - low) / close.shift(1).abs().replace(0.0, np.nan)
    short = daily_range.rolling(short_window, min_periods=short_window).mean()
    long = daily_range.shift(1).rolling(long_window, min_periods=long_window).mean()
    return short / long.replace(0.0, np.nan) - 1.0


def _choppiness_index(frame: pd.DataFrame, window: int) -> pd.Series:
    high = pd.to_numeric(frame["adjusted_high"], errors="coerce")
    low = pd.to_numeric(frame["adjusted_low"], errors="coerce")
    close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    tr_sum = true_range.rolling(window, min_periods=window).sum()
    range_width = high.rolling(window, min_periods=window).max() - low.rolling(window, min_periods=window).min()
    return np.log(tr_sum / range_width.replace(0.0, np.nan)) / np.log(float(window))


def _drawdown_recovery_gap(frame: pd.DataFrame, window: int) -> pd.Series:
    price = _causal_price_index(frame)
    rolling_high = price.rolling(window, min_periods=window).max()
    rolling_low = price.rolling(window, min_periods=window).min()
    drawdown = 1.0 - price / rolling_high.replace(0.0, np.nan)
    recovery = price / rolling_low.replace(0.0, np.nan) - 1.0
    return (drawdown - recovery).clip(lower=0.0)


def _loss_streak_intensity(frame: pd.DataFrame, window: int) -> pd.Series:
    losses = pd.to_numeric(frame["adjusted_ret"], errors="coerce").lt(0).astype(float)

    def max_streak(values: np.ndarray) -> float:
        best = current = 0
        for value in values:
            if value > 0:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return float(best / len(values))

    return losses.rolling(window, min_periods=window).apply(max_streak, raw=True)


def _gain_concentration(frame: pd.DataFrame, window: int, k: int) -> pd.Series:
    if not 1 <= k <= window:
        raise ValueError("k must satisfy 1 <= k <= window")

    def concentration(values: np.ndarray) -> float:
        gains = np.clip(values, 0.0, None)
        total = gains.sum()
        if total <= 0:
            return 0.0
        return float(np.partition(gains, len(gains) - k)[-k:].sum() / total)

    return frame["adjusted_ret"].rolling(window, min_periods=window).apply(
        concentration, raw=True
    )


def _trend_entropy(frame: pd.DataFrame, window: int) -> pd.Series:
    signs = pd.to_numeric(frame["adjusted_ret"], errors="coerce").pipe(np.sign)

    def entropy(values: np.ndarray) -> float:
        clean = values[~np.isnan(values)]
        if len(clean) == 0:
            return np.nan
        probs = np.array([(clean > 0).mean(), (clean < 0).mean(), (clean == 0).mean()])
        probs = probs[probs > 0]
        return float(-(probs * np.log(probs)).sum() / np.log(3.0))

    return signs.rolling(window, min_periods=window).apply(entropy, raw=True)


def _volume_trend(frame: pd.DataFrame, short_window: int, long_window: int) -> pd.Series:
    volume = pd.to_numeric(frame["volume"], errors="coerce").replace(0.0, np.nan)
    short = volume.rolling(short_window, min_periods=short_window).mean()
    long = volume.shift(1).rolling(long_window, min_periods=long_window).mean()
    return short / long.replace(0.0, np.nan) - 1.0


def _turnover_trend(frame: pd.DataFrame, short_window: int, long_window: int) -> pd.Series:
    turnover = pd.to_numeric(frame["total_turnover"], errors="coerce").replace(0.0, np.nan)
    short = turnover.rolling(short_window, min_periods=short_window).mean()
    long = turnover.shift(1).rolling(long_window, min_periods=long_window).mean()
    return short / long.replace(0.0, np.nan) - 1.0


def _volume_dry_up(frame: pd.DataFrame, short_window: int, long_window: int) -> pd.Series:
    return -_volume_trend(frame, short_window, long_window)


def _turnover_volatility(frame: pd.DataFrame, window: int) -> pd.Series:
    turnover = pd.to_numeric(frame["total_turnover"], errors="coerce").replace(0.0, np.nan)
    log_turnover = np.log(turnover)
    return log_turnover.diff().rolling(window, min_periods=window).std(ddof=0)


def _liquidity_price_impact(frame: pd.DataFrame, window: int) -> pd.Series:
    amount = pd.to_numeric(frame["total_turnover"], errors="coerce").replace(0.0, np.nan)
    ret_abs = pd.to_numeric(frame["adjusted_ret"], errors="coerce").abs()
    impact = ret_abs / np.log(amount).replace(0.0, np.nan)
    return impact.rolling(window, min_periods=window).mean()


def _volume_confirmed_momentum(
    frame: pd.DataFrame,
    momentum_window: int,
    volume_window: int,
) -> pd.Series:
    volume_expansion = _volume_trend(frame, max(3, volume_window // 3), volume_window).clip(lower=0.0)
    return _momentum(frame, momentum_window) * (1.0 + volume_expansion)


def _price_volume_divergence(
    frame: pd.DataFrame,
    momentum_window: int,
    volume_window: int,
) -> pd.Series:
    price_trend = _momentum(frame, momentum_window)
    volume_trend = _volume_trend(frame, max(3, volume_window // 3), volume_window)
    return price_trend - volume_trend


def _down_volume_pressure(
    frame: pd.DataFrame,
    return_window: int,
    volume_window: int,
) -> pd.Series:
    negative_trend = (-_momentum(frame, return_window)).clip(lower=0.0)
    volume_expansion = _volume_trend(frame, max(3, volume_window // 3), volume_window).clip(lower=0.0)
    return negative_trend * (1.0 + volume_expansion)


def _up_volume_exhaustion(
    frame: pd.DataFrame,
    return_window: int,
    volume_window: int,
) -> pd.Series:
    positive_trend = _momentum(frame, return_window).clip(lower=0.0)
    volume_expansion = _volume_trend(frame, max(3, volume_window // 3), volume_window).clip(lower=0.0)
    return positive_trend * volume_expansion


def _turnover_breakout_confirm(
    frame: pd.DataFrame,
    breakout_window: int,
    turnover_window: int,
) -> pd.Series:
    breakout = _breakout_strength(frame, breakout_window)
    turnover_expansion = _turnover_trend(frame, max(3, turnover_window // 3), turnover_window).clip(lower=0.0)
    return breakout * (1.0 + turnover_expansion)


def _short_term_reversal(frame: pd.DataFrame, lookback: int) -> pd.Series:
    return -_momentum(frame, lookback)


def _rsi_pressure(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = pd.to_numeric(frame["adjusted_ret"], errors="coerce")
    gain = ret.clip(lower=0.0).rolling(window, min_periods=window).mean()
    loss = (-ret.clip(upper=0.0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return (rsi - 50.0) / 50.0


def _stochastic_position(frame: pd.DataFrame, window: int) -> pd.Series:
    close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    high = pd.to_numeric(frame["adjusted_high"], errors="coerce").rolling(window, min_periods=window).max()
    low = pd.to_numeric(frame["adjusted_low"], errors="coerce").rolling(window, min_periods=window).min()
    return (close - low) / (high - low).replace(0.0, np.nan) - 0.5


def _distance_from_ma(frame: pd.DataFrame, ma_window: int) -> pd.Series:
    price = _causal_price_index(frame)
    ma = price.rolling(ma_window, min_periods=ma_window).mean()
    return price / ma.replace(0.0, np.nan) - 1.0


def _overheat_reversal_pressure(
    frame: pd.DataFrame,
    momentum_window: int,
    reversal_window: int,
) -> pd.Series:
    overheat = _momentum(frame, momentum_window).clip(lower=0.0)
    reversal = (-_momentum(frame, reversal_window)).clip(lower=0.0)
    return overheat * (1.0 + reversal)


BUILDERS: dict[str, Callable[..., pd.Series]] = {
    "momentum": _momentum,
    "vol_adjusted_momentum": _vol_adjusted_momentum,
    "double_ma": _double_ma,
    "ma_slope": _ma_slope,
    "macd": _macd,
    "breakout_strength": _breakout_strength,
    "efficiency_ratio": _efficiency_ratio,
    "trend_r2": _trend_r2,
    "realized_vol": _realized_vol,
    "downside_semivol": _downside_semivol,
    "worst_k_loss": _worst_k_loss,
    "return_skew": _return_skew,
    "drawdown": _drawdown,
    "drawdown_speed": _drawdown_speed,
    "calmar_momentum": _calmar_momentum,
    "ulcer_momentum": _ulcer_momentum,
    "gain_to_pain": _gain_to_pain,
    "omega_ratio": _omega_ratio,
    "downside_vol_momentum": _downside_vol_momentum,
    "convexity_momentum": _convexity_momentum,
    "frog_in_pan": _frog_in_pan,
    "up_down_frequency": _up_down_frequency,
    "aroon_oscillator": _aroon_oscillator,
    "donchian_position": _donchian_position,
    "distance_from_low": _distance_from_low,
    "atr_breakout_strength": _atr_breakout_strength,
    "trix": _trix,
    "stable_momentum": _stable_momentum,
    "recovery_slope_vol_adjusted": _recovery_slope_vol_adjusted,
    "left_tail_loss_ratio": _left_tail_loss_ratio,
    "trend_convexity_proxy": _trend_convexity_proxy,
    "intraday_purity": _intraday_purity,
    "volume_shock": _volume_shock,
    "oi_change": _oi_change,
    "range_expansion": _range_expansion,
    "volume_oi_pressure": _volume_oi_pressure,
    "volume_adjusted_momentum": _volume_adjusted_momentum,
    "volume_return_confirm": _volume_return_confirm,
    "price_volume_correlation": _price_volume_correlation,
    "oi_pressure_down": _oi_pressure_down,
    "range_expansion_down": _range_expansion_down,
    "multi_speed_tsmom_vote": _multi_speed_tsmom_vote,
    "high_vol_negative_trend": _high_vol_negative_trend,
    "drawdown_acceleration_down": _drawdown_acceleration_down,
    "gap_return": _gap_return,
    "intraday_return": _intraday_return,
    "gap_fill_failure": _gap_fill_failure,
    "close_location": _close_location,
    "upper_shadow_pressure": _upper_shadow_pressure,
    "lower_shadow_support": _lower_shadow_support,
    "vol_ratio": _vol_ratio,
    "downside_vol_ratio": _downside_vol_ratio,
    "vol_of_vol": _vol_of_vol,
    "return_jump_count": _return_jump_count,
    "range_vol_ratio": _range_vol_ratio,
    "choppiness_index": _choppiness_index,
    "drawdown_recovery_gap": _drawdown_recovery_gap,
    "loss_streak_intensity": _loss_streak_intensity,
    "gain_concentration": _gain_concentration,
    "trend_entropy": _trend_entropy,
    "volume_trend": _volume_trend,
    "turnover_trend": _turnover_trend,
    "volume_dry_up": _volume_dry_up,
    "turnover_volatility": _turnover_volatility,
    "liquidity_price_impact": _liquidity_price_impact,
    "volume_confirmed_momentum": _volume_confirmed_momentum,
    "price_volume_divergence": _price_volume_divergence,
    "down_volume_pressure": _down_volume_pressure,
    "up_volume_exhaustion": _up_volume_exhaustion,
    "turnover_breakout_confirm": _turnover_breakout_confirm,
    "short_term_reversal": _short_term_reversal,
    "rsi_pressure": _rsi_pressure,
    "stochastic_position": _stochastic_position,
    "distance_from_ma": _distance_from_ma,
    "overheat_reversal_pressure": _overheat_reversal_pressure,
}

REQUIRED_COLUMNS = {
    "momentum": {"adjusted_ret"},
    "vol_adjusted_momentum": {"adjusted_ret"},
    "double_ma": {"adjusted_ret"},
    "ma_slope": {"adjusted_ret"},
    "macd": {"adjusted_ret"},
    "breakout_strength": {"adjusted_ret"},
    "efficiency_ratio": {"adjusted_ret"},
    "trend_r2": {"adjusted_ret"},
    "realized_vol": {"adjusted_ret"},
    "downside_semivol": {"adjusted_ret"},
    "worst_k_loss": {"adjusted_ret"},
    "return_skew": {"adjusted_ret"},
    "drawdown": {"adjusted_ret"},
    "drawdown_speed": {"adjusted_ret"},
    "calmar_momentum": {"adjusted_ret"},
    "ulcer_momentum": {"adjusted_ret"},
    "gain_to_pain": {"adjusted_ret"},
    "omega_ratio": {"adjusted_ret"},
    "downside_vol_momentum": {"adjusted_ret"},
    "convexity_momentum": {"adjusted_ret"},
    "frog_in_pan": {"adjusted_ret"},
    "up_down_frequency": {"adjusted_ret"},
    "aroon_oscillator": {"adjusted_ret"},
    "donchian_position": {"adjusted_ret"},
    "distance_from_low": {"adjusted_ret"},
    "atr_breakout_strength": {
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
    },
    "trix": {"adjusted_ret"},
    "stable_momentum": {"adjusted_ret"},
    "recovery_slope_vol_adjusted": {"adjusted_ret"},
    "left_tail_loss_ratio": {"adjusted_ret"},
    "trend_convexity_proxy": {"adjusted_ret"},
    "intraday_purity": {"adjusted_open", "adjusted_close"},
    "volume_shock": {"volume"},
    "oi_change": {"oi"},
    "range_expansion": {"adjusted_high", "adjusted_low", "adjusted_close"},
    "volume_oi_pressure": {"volume", "oi"},
    "volume_adjusted_momentum": {"volume", "adjusted_ret"},
    "volume_return_confirm": {"volume", "adjusted_ret"},
    "price_volume_correlation": {"volume", "adjusted_ret"},
    "oi_pressure_down": {"oi", "adjusted_ret"},
    "range_expansion_down": {
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
    },
    "multi_speed_tsmom_vote": {"adjusted_ret"},
    "high_vol_negative_trend": {"adjusted_ret"},
    "drawdown_acceleration_down": {"adjusted_ret"},
    "gap_return": {"adjusted_open", "adjusted_close"},
    "intraday_return": {"adjusted_open", "adjusted_close"},
    "gap_fill_failure": {"adjusted_open", "adjusted_close"},
    "close_location": {"adjusted_high", "adjusted_low", "adjusted_close"},
    "upper_shadow_pressure": {"adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"},
    "lower_shadow_support": {"adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"},
    "vol_ratio": {"adjusted_ret"},
    "downside_vol_ratio": {"adjusted_ret"},
    "vol_of_vol": {"adjusted_ret"},
    "return_jump_count": {"adjusted_ret"},
    "range_vol_ratio": {"adjusted_high", "adjusted_low", "adjusted_close"},
    "choppiness_index": {"adjusted_high", "adjusted_low", "adjusted_close"},
    "drawdown_recovery_gap": {"adjusted_ret"},
    "loss_streak_intensity": {"adjusted_ret"},
    "gain_concentration": {"adjusted_ret"},
    "trend_entropy": {"adjusted_ret"},
    "volume_trend": {"volume"},
    "turnover_trend": {"total_turnover"},
    "volume_dry_up": {"volume"},
    "turnover_volatility": {"total_turnover"},
    "liquidity_price_impact": {"adjusted_ret", "total_turnover"},
    "volume_confirmed_momentum": {"volume", "adjusted_ret"},
    "price_volume_divergence": {"volume", "adjusted_ret"},
    "down_volume_pressure": {"volume", "adjusted_ret"},
    "up_volume_exhaustion": {"volume", "adjusted_ret"},
    "turnover_breakout_confirm": {"total_turnover", "adjusted_ret"},
    "short_term_reversal": {"adjusted_ret"},
    "rsi_pressure": {"adjusted_ret"},
    "stochastic_position": {"adjusted_high", "adjusted_low", "adjusted_close"},
    "distance_from_ma": {"adjusted_ret"},
    "overheat_reversal_pressure": {"adjusted_ret"},
}


def rolling_zscore(
    values: pd.Series,
    *,
    window: int,
    min_periods: int | None = None,
    stats_lag: int = 1,
    clip: float | None = None,
) -> pd.Series:
    """使用历史滚动均值/标准差做标准化，不读取未来样本。

    默认 stats_lag=1，表示 t 日特征使用截至 t-1 日的分布衡量异常程度；
    这比把 t 日自身放入均值和标准差更保守。
    """

    if window < 2:
        raise ValueError("rolling z-score window must be at least 2")
    if stats_lag < 0:
        raise ValueError("stats_lag cannot be negative")
    min_p = min_periods or max(20, window // 2)
    history = values.shift(stats_lag)
    mean = history.rolling(window, min_periods=min_p).mean()
    std = history.rolling(window, min_periods=min_p).std(ddof=0)
    transformed = (values - mean) / std.replace(0.0, np.nan)
    if clip is not None:
        transformed = transformed.clip(-abs(clip), abs(clip))
    return transformed


def expand_feature_specs_with_transforms(
    raw_specs: list[FeatureSpec],
) -> list[FeatureSpec]:
    """按 config.py 将每个原始特征扩展为 raw 与多个滚动 z-score 版本。"""

    from .factor_config import ROLLING_STANDARDIZATION_GRID

    grid = ROLLING_STANDARDIZATION_GRID
    expanded = list(raw_specs) if grid["include_raw"] else []
    for spec in raw_specs:
        for window in grid["zscore_windows"]:
            expanded.append(
                FeatureSpec(
                    name=spec.name,
                    params=spec.params,
                    risk_direction=spec.risk_direction,
                    transform="rolling_zscore",
                    transform_window=int(window),
                    transform_min_periods=max(
                        20, int(window * float(grid["min_period_ratio"]))
                    ),
                    transform_stats_lag=int(grid["stats_lag"]),
                    transform_clip=float(grid["clip"]),
                )
            )
    return expanded


def _column_name(spec: FeatureSpec) -> str:
    suffix = "__".join(f"{key}_{value}" for key, value in sorted(spec.params.items()))
    base = spec.name if not suffix else f"{spec.name}__{suffix}"
    if spec.transform == "raw":
        return base
    return f"{base}__{spec.transform}_{spec.transform_window}"


def default_feature_specs(
    *,
    batch_name: str = "B01_trend_direction",
    include_transforms: bool = True,
) -> list[FeatureSpec]:
    """读取指定研究批次；默认同时生成 raw 与滚动标准化版本。"""

    from .factor_config import first_round_feature_specs

    raw_specs = first_round_feature_specs(batch_name=batch_name)
    return (
        expand_feature_specs_with_transforms(raw_specs)
        if include_transforms
        else raw_specs
    )


def build_feature_matrix(
    data: pd.DataFrame,
    specs: list[FeatureSpec],
    *,
    progress_every: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每个“因子+参数”生成一列；所有计算只能使用 t 日及以前数据。

    progress_every 仅控制进度打印，不影响计算结果。设为 100 表示每完成 100 个
    特征版本打印一次进度，适合数千列批次。
    """

    required_base = {"symbol", "trade_date"}
    missing_base = required_base.difference(data.columns)
    if missing_base:
        raise ValueError(f"Missing base columns: {sorted(missing_base)}")

    pieces = []
    metadata = []
    from .factor_config import FEATURE_BATCH_MEMBERSHIP, FEATURE_SOURCE_MAP

    for symbol, raw in data.groupby("symbol", sort=False):
        frame = raw.sort_values("trade_date", kind="stable").reset_index(drop=True)
        feature_columns: dict[str, pd.Series] = {}
        # 每个品种单独 rolling，避免 IC/IM 在边界处相互污染。
        for spec_index, spec in enumerate(specs, start=1):
            if spec.name not in BUILDERS:
                raise KeyError(f"Unknown feature builder: {spec.name}")
            missing = REQUIRED_COLUMNS[spec.name].difference(frame.columns)
            if missing:
                raise ValueError(f"{symbol}/{spec.name}: missing {sorted(missing)}")
            column = _column_name(spec)
            values = BUILDERS[spec.name](frame, **dict(spec.params))
            if spec.transform == "rolling_zscore":
                values = rolling_zscore(
                    values,
                    window=int(spec.transform_window),
                    min_periods=spec.transform_min_periods,
                    stats_lag=spec.transform_stats_lag,
                    clip=spec.transform_clip,
                )
            elif spec.transform != "raw":
                raise ValueError(f"Unknown feature transform: {spec.transform}")
            feature_columns[column] = values.replace([np.inf, -np.inf], np.nan)
            metadata.append(
                {
                    "symbol": symbol,
                    "feature": column,
                    "family": spec.name,
                    "source": FEATURE_SOURCE_MAP.get(spec.name, "未登记"),
                    "batches": ",".join(FEATURE_BATCH_MEMBERSHIP.get(spec.name, ())),
                    "params": dict(spec.params),
                    "risk_direction": int(np.sign(spec.risk_direction) or 1),
                    "transform": spec.transform,
                    "transform_window": spec.transform_window,
                    "transform_min_periods": spec.transform_min_periods,
                    "transform_stats_lag": spec.transform_stats_lag,
                    "transform_clip": spec.transform_clip,
                }
            )
            if progress_every and spec_index % progress_every == 0:
                print(f"[{symbol}] 已完成 {spec_index}/{len(specs)} 个特征版本")
        # 一次性拼接数百列，避免逐列插入造成 DataFrame 内存碎片。
        result = pd.concat(
            [
                frame[["symbol", "trade_date"]].copy(),
                pd.DataFrame(feature_columns, index=frame.index),
            ],
            axis=1,
        )
        pieces.append(result)
    matrix = pd.concat(pieces, ignore_index=True)
    return matrix, pd.DataFrame(metadata).drop_duplicates(
        subset=["symbol", "feature"]
    )
