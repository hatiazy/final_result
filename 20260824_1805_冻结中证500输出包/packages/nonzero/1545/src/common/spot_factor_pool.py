from __future__ import annotations

import bisect
from collections.abc import Callable

import numpy as np
import pandas as pd


FACTOR_ENGINE_COLUMNS = (
    "fp_trend_axis",
    "fp_robust_axis",
    "fp_structure_axis",
    "fp_repair_axis",
    "fp_flow_axis",
    "fp_down_pressure",
)


def describe_factor_pool() -> pd.DataFrame:
    """Return a compact, screenshot-friendly map of the independent engines."""
    return pd.DataFrame(
        [
            {
                "engine": "trend",
                "component_count": 4,
                "representative_inputs": "20日累计收益、EMA斜率、回归斜率、中位价斜率",
                "economic_role": "慢方向主轴",
            },
            {
                "engine": "robust_return",
                "component_count": 4,
                "representative_inputs": "5/10日收益中位数、截尾均值、5/10/20日符号共识",
                "economic_role": "降低单日反转干扰",
            },
            {
                "engine": "price_structure",
                "component_count": 4,
                "representative_inputs": "高低点结构、突破计数、通道位置、前区间突破",
                "economic_role": "确认趋势结构是否成立",
            },
            {
                "engine": "repair_acceptance",
                "component_count": 5,
                "representative_inputs": "日内实体、收盘位置、跳空承接、两日延续、影线接受",
                "economic_role": "识别修复与反弹否决",
            },
            {
                "engine": "directional_flow",
                "component_count": 9,
                "representative_inputs": "成交量/成交额方向份额、量权收益、活跃度惊喜、下跌压力反向值",
                "economic_role": "量价方向确认",
            },
            {
                "engine": "down_pressure",
                "component_count": 3,
                "representative_inputs": "5/10日下跌成交参与、5对20日压力加速度",
                "economic_role": "只辅助确认已转弱方向",
            },
        ]
    )


def causal_rolling_percentile(
    values: pd.Series,
    window: int = 504,
    min_periods: int = 252,
) -> pd.Series:
    """Rank today's value only against observations available before today."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(numeric), np.nan, dtype=float)
    history: list[float] = []
    for position, current in enumerate(numeric):
        if np.isfinite(current) and len(history) >= int(min_periods):
            result[position] = bisect.bisect_right(history, float(current)) / len(
                history
            )
        if position >= int(window):
            expired = numeric[position - int(window)]
            if np.isfinite(expired):
                index = bisect.bisect_left(history, float(expired))
                if index >= len(history) or history[index] != float(expired):
                    raise AssertionError("causal factor percentile window is inconsistent")
                history.pop(index)
        if np.isfinite(current):
            bisect.insort_right(history, float(current))
    return pd.Series(result, index=values.index, dtype=float)


def _price(frame: pd.DataFrame) -> pd.Series:
    return frame["close"].astype(float)


def _returns(frame: pd.DataFrame) -> pd.Series:
    return _price(frame).pct_change(fill_method=None)


def _cumulative_return(frame: pd.DataFrame, window: int) -> pd.Series:
    return _price(frame).pct_change(window, fill_method=None)


def _ema_slope(frame: pd.DataFrame, span: int, lag: int) -> pd.Series:
    ema = _price(frame).ewm(span=span, adjust=False, min_periods=span).mean()
    return ema.pct_change(lag, fill_method=None)


def _regression_direction(frame: pd.DataFrame, window: int) -> pd.Series:
    log_price = np.log(_price(frame).where(_price(frame).gt(0)))
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.square(x_centered).sum())

    def slope(values: np.ndarray) -> float:
        if not np.isfinite(values).all():
            return np.nan
        return float(np.dot(values - values.mean(), x_centered) / denominator)

    return log_price.rolling(window, min_periods=window).apply(slope, raw=True)


def _median_price_slope(frame: pd.DataFrame, window: int, lag: int) -> pd.Series:
    median = _price(frame).rolling(window, min_periods=window).median()
    return median.pct_change(lag, fill_method=None)


def _median_return(frame: pd.DataFrame, window: int) -> pd.Series:
    return _returns(frame).rolling(window, min_periods=window).median()


def _trimmed_mean_return(frame: pd.DataFrame, window: int) -> pd.Series:
    def trimmed(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if len(values) < window:
            return np.nan
        ordered = np.sort(values)
        return float(ordered[1:-1].mean()) if len(ordered) > 2 else float(ordered.mean())

    return _returns(frame).rolling(window, min_periods=window).apply(trimmed, raw=True)


def _sign_return_consensus(
    frame: pd.DataFrame,
    short: int,
    medium: int,
    long: int,
) -> pd.Series:
    price = _price(frame)
    moves = pd.concat(
        [
            price.pct_change(short, fill_method=None),
            price.pct_change(medium, fill_method=None),
            price.pct_change(long, fill_method=None),
        ],
        axis=1,
    )
    return np.sign(moves).mean(axis=1)


def _higher_high_lower_low_balance(frame: pd.DataFrame, window: int) -> pd.Series:
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    higher_high = high.gt(high.shift(1))
    higher_low = low.gt(low.shift(1))
    structure = (
        higher_high.astype(float)
        + higher_low.astype(float)
        - (~higher_high).astype(float)
        - (~higher_low).astype(float)
    ) / 2.0
    structure = structure.where(high.shift(1).notna() & low.shift(1).notna())
    return structure.rolling(window, min_periods=window).mean()


def _directional_breakout_count(
    frame: pd.DataFrame,
    lookback: int,
    window: int,
) -> pd.Series:
    close = _price(frame)
    prior_high = close.shift(1).rolling(lookback, min_periods=lookback).max()
    prior_low = close.shift(1).rolling(lookback, min_periods=lookback).min()
    signal = close.gt(prior_high).astype(float) - close.lt(prior_low).astype(float)
    return signal.rolling(window, min_periods=window).mean()


def _rolling_channel_position(frame: pd.DataFrame, window: int) -> pd.Series:
    price = _price(frame)
    low = price.rolling(window, min_periods=window).min()
    high = price.rolling(window, min_periods=window).max()
    return 2.0 * (price - low) / (high - low).replace(0.0, np.nan) - 1.0


def _prior_range_break(frame: pd.DataFrame, window: int) -> pd.Series:
    prior_high = frame["high"].shift(1).rolling(window, min_periods=window).max()
    prior_low = frame["low"].shift(1).rolling(window, min_periods=window).min()
    width = (prior_high - prior_low).replace(0.0, np.nan)
    return ((_price(frame) - (prior_high + prior_low) / 2.0) / width).clip(-3.0, 3.0)


def _intraday_body_direction(frame: pd.DataFrame, window: int) -> pd.Series:
    span = (frame["high"] - frame["low"]).replace(0.0, np.nan)
    body = (frame["close"] - frame["open"]) / span
    return body.rolling(window, min_periods=window).mean()


def _intraday_bar_location(frame: pd.DataFrame, window: int) -> pd.Series:
    span = (frame["high"] - frame["low"]).replace(0.0, np.nan)
    location = (2.0 * frame["close"] - frame["high"] - frame["low"]) / span
    return location.rolling(window, min_periods=window).mean()


def _intraday_gap_follow(frame: pd.DataFrame, window: int) -> pd.Series:
    prior_close = frame["close"].shift(1)
    scale = prior_close.abs().replace(0.0, np.nan)
    gap = (frame["open"] - prior_close) / scale
    body = (frame["close"] - frame["open"]) / scale
    return (0.5 * gap + 0.5 * body).rolling(window, min_periods=window).mean()


def _two_day_body_continuation(frame: pd.DataFrame, window: int) -> pd.Series:
    body = np.sign(frame["close"] - frame["open"])
    continuation = body.where(body.eq(body.shift(1)), 0.0)
    return continuation.rolling(window, min_periods=window).mean()


def _shadow_reversal_acceptance(frame: pd.DataFrame, window: int) -> pd.Series:
    open_ = frame["open"].astype(float)
    close = frame["close"].astype(float)
    span = (frame["high"] - frame["low"]).replace(0.0, np.nan)
    upper = (frame["high"] - pd.concat([open_, close], axis=1).max(axis=1)) / span
    lower = (pd.concat([open_, close], axis=1).min(axis=1) - frame["low"]) / span
    return (lower - upper).rolling(window, min_periods=window).mean()


def _signed_weight_share(
    frame: pd.DataFrame,
    window: int,
    weight_column: str,
) -> pd.Series:
    weight = frame[weight_column].astype(float).clip(lower=0.0)
    signed = np.sign(_returns(frame)).fillna(0.0) * weight
    denominator = weight.rolling(window, min_periods=window).sum()
    return signed.rolling(window, min_periods=window).sum() / denominator.replace(
        0.0, np.nan
    )


def _signed_weighted_return(
    frame: pd.DataFrame,
    window: int,
    baseline: int,
    weight_column: str,
) -> pd.Series:
    weight = frame[weight_column].astype(float).clip(lower=0.0)
    scale = weight.shift(1).rolling(baseline, min_periods=baseline).median()
    relative = weight / scale.replace(0.0, np.nan)
    flow = _returns(frame) * relative.clip(0.0, 4.0)
    return flow.rolling(window, min_periods=window).mean()


def _weight_surprise_direction(
    frame: pd.DataFrame,
    window: int,
    baseline: int,
    weight_column: str,
) -> pd.Series:
    weight = frame[weight_column].astype(float).clip(lower=0.0)
    reference = weight.shift(1).rolling(baseline, min_periods=baseline).median()
    surprise = weight / reference.replace(0.0, np.nan) - 1.0
    return (np.sign(_returns(frame)) * surprise.clip(-3.0, 3.0)).rolling(
        window, min_periods=window
    ).mean()


def _downside_turnover_participation(
    frame: pd.DataFrame,
    window: int,
) -> pd.Series:
    turnover = frame["total_turnover"].astype(float).clip(lower=0.0)
    down = turnover.where(_returns(frame).lt(0.0), 0.0)
    return down.rolling(window, min_periods=window).sum() / turnover.rolling(
        window, min_periods=window
    ).sum().replace(0.0, np.nan)


def _downside_turnover_acceleration(
    frame: pd.DataFrame,
    short: int,
    baseline: int,
) -> pd.Series:
    participation = _downside_turnover_participation(frame, short)
    reference = participation.shift(1).rolling(
        baseline, min_periods=baseline
    ).mean()
    return (participation - reference).clip(-1.0, 1.0)


def _rank_components(
    frame: pd.DataFrame,
    specifications: dict[str, Callable[[pd.DataFrame], pd.Series]],
    *,
    percentile_window: int,
    percentile_min_periods: int,
) -> pd.DataFrame:
    ranked = {}
    for name, builder in specifications.items():
        raw = builder(frame).replace([np.inf, -np.inf], np.nan)
        ranked[name] = causal_rolling_percentile(
            raw,
            window=percentile_window,
            min_periods=percentile_min_periods,
        )
    return pd.DataFrame(ranked, index=frame.index)


def build_factor_pool(
    spot: pd.DataFrame,
    *,
    percentile_window: int = 504,
    percentile_min_periods: int = 252,
) -> pd.DataFrame:
    """Build an independent causal OHLCV/turnover direction-factor library.

    Every component is an OHLCV/turnover direction factor. Volatility and
    future returns are deliberately excluded. A high component percentile
    always means more positive direction except ``fp_down_pressure``.
    """
    required = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "total_turnover",
    }
    missing = sorted(required.difference(spot.columns))
    if missing:
        raise ValueError(f"factor-pool spot data is missing columns: {missing}")
    frame = spot.sort_index().astype(
        {
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
            "total_turnover": float,
        }
    )
    if frame[["open", "high", "low", "close"]].le(0).any().any():
        raise ValueError("factor-pool OHLC values must be positive")

    trend = _rank_components(
        frame,
        {
            "factor_trend_cumulative_20": lambda x: _cumulative_return(x, 20),
            "factor_trend_ema_slope_20_5": lambda x: _ema_slope(x, 20, 5),
            "factor_trend_regression_20": lambda x: _regression_direction(x, 20),
            "factor_trend_median_price_15_5": lambda x: _median_price_slope(
                x, 15, 5
            ),
        },
        percentile_window=percentile_window,
        percentile_min_periods=percentile_min_periods,
    )
    robust = _rank_components(
        frame,
        {
            "factor_robust_median_return_5": lambda x: _median_return(x, 5),
            "factor_robust_median_return_10": lambda x: _median_return(x, 10),
            "factor_robust_trimmed_return_10": lambda x: _trimmed_mean_return(x, 10),
            "factor_robust_sign_consensus_5_10_20": lambda x: _sign_return_consensus(
                x, 5, 10, 20
            ),
        },
        percentile_window=percentile_window,
        percentile_min_periods=percentile_min_periods,
    )
    structure = _rank_components(
        frame,
        {
            "factor_structure_hhll_10": lambda x: _higher_high_lower_low_balance(
                x, 10
            ),
            "factor_structure_breakout_10_5": lambda x: _directional_breakout_count(
                x, 10, 5
            ),
            "factor_structure_channel_20": lambda x: _rolling_channel_position(x, 20),
            "factor_structure_prior_range_10": lambda x: _prior_range_break(x, 10),
        },
        percentile_window=percentile_window,
        percentile_min_periods=percentile_min_periods,
    )
    repair = _rank_components(
        frame,
        {
            "factor_repair_body_5": lambda x: _intraday_body_direction(x, 5),
            "factor_repair_location_5": lambda x: _intraday_bar_location(x, 5),
            "factor_repair_gap_follow_5": lambda x: _intraday_gap_follow(x, 5),
            "factor_repair_two_day_5": lambda x: _two_day_body_continuation(x, 5),
            "factor_repair_shadow_5": lambda x: _shadow_reversal_acceptance(x, 5),
        },
        percentile_window=percentile_window,
        percentile_min_periods=percentile_min_periods,
    )
    flow = _rank_components(
        frame,
        {
            "factor_flow_signed_turnover_share_10": lambda x: (
                _signed_weight_share(x, 10, "total_turnover")
            ),
            "factor_flow_turnover_weighted_return_10_20": lambda x: (
                _signed_weighted_return(x, 10, 20, "total_turnover")
            ),
            "factor_flow_turnover_surprise_5_20": lambda x: (
                _weight_surprise_direction(x, 5, 20, "total_turnover")
            ),
            "factor_flow_signed_volume_share_10": lambda x: (
                _signed_weight_share(x, 10, "volume")
            ),
            "factor_flow_volume_weighted_return_10_20": lambda x: (
                _signed_weighted_return(x, 10, 20, "volume")
            ),
            "factor_flow_volume_surprise_5_20": lambda x: (
                _weight_surprise_direction(x, 5, 20, "volume")
            ),
        },
        percentile_window=percentile_window,
        percentile_min_periods=percentile_min_periods,
    )
    pressure = _rank_components(
        frame,
        {
            "factor_pressure_participation_5": lambda x: (
                _downside_turnover_participation(x, 5)
            ),
            "factor_pressure_participation_10": lambda x: (
                _downside_turnover_participation(x, 10)
            ),
            "factor_pressure_acceleration_5_20": lambda x: (
                _downside_turnover_acceleration(x, 5, 20)
            ),
        },
        percentile_window=percentile_window,
        percentile_min_periods=percentile_min_periods,
    )

    output = pd.concat([trend, robust, structure, repair, flow, pressure], axis=1)
    output["fp_trend_axis"] = trend.mean(axis=1)
    output["fp_robust_axis"] = robust.mean(axis=1)
    output["fp_structure_axis"] = structure.mean(axis=1)
    output["fp_repair_axis"] = repair.mean(axis=1)
    output["fp_down_pressure"] = pressure.mean(axis=1)
    output["fp_flow_axis"] = pd.concat(
        [flow, pressure.rsub(1.0).add_prefix("positive_")], axis=1
    ).mean(axis=1)
    output["fp_factor_consensus"] = output[
        [
            "fp_trend_axis",
            "fp_robust_axis",
            "fp_structure_axis",
            "fp_repair_axis",
            "fp_flow_axis",
        ]
    ].mean(axis=1)
    output["fp_weak_engine_count"] = (
        output[
            [
                "fp_trend_axis",
                "fp_robust_axis",
                "fp_structure_axis",
                "fp_flow_axis",
            ]
        ]
        .le(0.45)
        .sum(axis=1)
        .astype(float)
    )
    output["fp_strong_engine_count"] = (
        output[
            [
                "fp_trend_axis",
                "fp_robust_axis",
                "fp_structure_axis",
                "fp_flow_axis",
            ]
        ]
        .ge(0.60)
        .sum(axis=1)
        .astype(float)
    )
    output["fp_trend_change_1d"] = output["fp_trend_axis"].diff()
    output["fp_trend_change_3d"] = output["fp_trend_axis"].diff(3)
    output["fp_consensus_change_1d"] = output["fp_factor_consensus"].diff()
    output["fp_consensus_change_3d"] = output["fp_factor_consensus"].diff(3)
    output["fp_repair_change_1d"] = output["fp_repair_axis"].diff()
    output["fp_flow_change_1d"] = output["fp_flow_axis"].diff()
    output["fp_pressure_change_1d"] = output["fp_down_pressure"].diff()
    return output.replace([np.inf, -np.inf], np.nan)
