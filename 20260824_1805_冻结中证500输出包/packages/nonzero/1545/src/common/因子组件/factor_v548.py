"""Independent v5.9 spot-daily structure-direction candidate batch."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_BASE_PATH = Path(__file__).resolve().with_name("factor_v55.py")
_SPEC = importlib.util.spec_from_file_location("v59_base_factor", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)

FeatureSpec = _BASE.FeatureSpec
rolling_zscore = _BASE.rolling_zscore
BUILDERS = dict(_BASE.BUILDERS)


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _ret(frame: pd.DataFrame) -> pd.Series:
    return _num(frame, "adjusted_ret")


def _price(frame: pd.DataFrame) -> pd.Series:
    return (1.0 + _ret(frame).fillna(0.0)).cumprod()


# All trend families are signed price direction only.
def _higher_high_lower_low_balance(frame: pd.DataFrame, window: int) -> pd.Series:
    high = _num(frame, "adjusted_high")
    low = _num(frame, "adjusted_low")
    structure = high.diff().gt(0.0).astype(float) - low.diff().lt(0.0).astype(float)
    return structure.rolling(window, min_periods=window).mean()


def _directional_breakout_count(frame: pd.DataFrame, lookback: int, window: int) -> pd.Series:
    close = _num(frame, "adjusted_close")
    prior_high = close.shift(1).rolling(lookback, min_periods=lookback).max()
    prior_low = close.shift(1).rolling(lookback, min_periods=lookback).min()
    signal = close.gt(prior_high).astype(float) - close.lt(prior_low).astype(float)
    return signal.rolling(window, min_periods=window).mean()


def _ema_slope_direction(frame: pd.DataFrame, span: int, lag: int) -> pd.Series:
    price = _price(frame)
    ema = price.ewm(span=span, adjust=False, min_periods=span).mean()
    return ema.pct_change(lag, fill_method=None)


def _median_price_slope_direction(frame: pd.DataFrame, window: int, lag: int) -> pd.Series:
    median = _price(frame).rolling(window, min_periods=window).median()
    return median.pct_change(lag, fill_method=None)


def _short_return_direction(frame: pd.DataFrame, window: int) -> pd.Series:
    """Pure short-horizon close-to-close direction; no volatility scaling."""
    return _price(frame).pct_change(window, fill_method=None)


def _median_return_direction(frame: pd.DataFrame, window: int) -> pd.Series:
    """Robust pure direction: median daily return over the recent window."""
    return _ret(frame).rolling(window, min_periods=window).median()


def _trimmed_mean_return_direction(frame: pd.DataFrame, window: int) -> pd.Series:
    """Robust pure direction after removing the largest and smallest return."""
    def _trimmed(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        if len(finite) < 3:
            return np.nan
        ordered = np.sort(finite)
        return float(ordered[1:-1].mean())
    return _ret(frame).rolling(window, min_periods=window).apply(_trimmed, raw=True)


def _short_direction_run(frame: pd.DataFrame, window: int) -> pd.Series:
    """Pure direction persistence: signed fraction of up versus down days."""
    return np.sign(_ret(frame)).rolling(window, min_periods=window).mean()


def _sign_return_consensus(frame: pd.DataFrame, short: int, medium: int, long: int) -> pd.Series:
    """Unweighted direction vote across short, medium and long returns."""
    price = _price(frame)
    moves = pd.concat([
        price.pct_change(short, fill_method=None),
        price.pct_change(medium, fill_method=None),
        price.pct_change(long, fill_method=None),
    ], axis=1)
    return np.sign(moves).mean(axis=1)


def _intraday_body_direction(frame: pd.DataFrame, window: int) -> pd.Series:
    """Short-window signed candle body, normalized only by that day's range."""
    open_, high, low, close = (
        _num(frame, x) for x in ["adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"]
    )
    span = (high - low).replace(0.0, np.nan)
    return ((close - open_) / span).rolling(window, min_periods=window).mean()


def _intraday_bar_location(frame: pd.DataFrame, window: int) -> pd.Series:
    """Short-window close location inside the current bar's range."""
    high, low, close = (_num(frame, x) for x in ["adjusted_high", "adjusted_low", "adjusted_close"])
    span = (high - low).replace(0.0, np.nan)
    return ((2.0 * close - high - low) / span).rolling(window, min_periods=window).mean()


def _intraday_gap_follow(frame: pd.DataFrame, window: int) -> pd.Series:
    """Short-window gap plus same-day follow-through direction."""
    open_, close = (_num(frame, x) for x in ["adjusted_open", "adjusted_close"])
    previous_close = close.shift(1)
    gap = (open_ - previous_close) / previous_close.replace(0.0, np.nan)
    body = (close - open_) / previous_close.replace(0.0, np.nan)
    return (0.5 * gap + 0.5 * body).rolling(window, min_periods=window).mean()


def _intraday_body_sign_consensus(frame: pd.DataFrame, window: int) -> pd.Series:
    """Same-day buyer/seller direction, ignoring bar-size magnitude."""
    open_, close = (_num(frame, x) for x in ["adjusted_open", "adjusted_close"])
    return np.sign(close - open_).rolling(window, min_periods=window).mean()


def _intraday_gap_body_agreement(frame: pd.DataFrame, window: int) -> pd.Series:
    """Whether overnight gap and same-day body carry the same direction."""
    open_, close = (_num(frame, x) for x in ["adjusted_open", "adjusted_close"])
    previous_close = close.shift(1)
    gap_sign = np.sign(open_ - previous_close)
    body_sign = np.sign(close - open_)
    return (0.5 * gap_sign + 0.5 * body_sign).rolling(window, min_periods=window).mean()


def _intraday_close_prev_sign(frame: pd.DataFrame, window: int) -> pd.Series:
    """Close-to-previous-close acceptance direction over a short window."""
    close = _num(frame, "adjusted_close")
    return np.sign(close.diff()).rolling(window, min_periods=window).mean()


def _rolling_channel_position_direction(frame: pd.DataFrame, window: int) -> pd.Series:
    price = _price(frame)
    high = price.shift(1).rolling(window, min_periods=window).max()
    low = price.shift(1).rolling(window, min_periods=window).min()
    return 2.0 * (price - low) / (high - low).replace(0.0, np.nan) - 1.0


# Intraday families measure buyer/seller acceptance in the current/past bars.
def _prior_range_break_direction(frame: pd.DataFrame, window: int) -> pd.Series:
    high, low, close = (_num(frame, x) for x in ["adjusted_high", "adjusted_low", "adjusted_close"])
    previous_close = close.shift(1)
    prior_high = high.shift(1)
    prior_low = low.shift(1)
    scale = (prior_high - prior_low).replace(0.0, np.nan)
    value = (close - previous_close) / scale
    return value.clip(-3.0, 3.0).rolling(window, min_periods=window).mean()


def _two_day_body_continuation(frame: pd.DataFrame, window: int) -> pd.Series:
    open_, high, low, close = (
        _num(frame, x) for x in ["adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"]
    )
    body = (close - open_) / (high - low).replace(0.0, np.nan)
    continuation = 0.65 * body + 0.35 * body.shift(1)
    return continuation.rolling(window, min_periods=window).mean()


def _close_auction_balance(frame: pd.DataFrame, window: int) -> pd.Series:
    open_, high, low, close = (
        _num(frame, x) for x in ["adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"]
    )
    span = (high - low).replace(0.0, np.nan)
    location = (2.0 * close - high - low) / span
    body = (close - open_) / span
    return (0.6 * location + 0.4 * body).rolling(window, min_periods=window).mean()


def _shadow_reversal_acceptance(frame: pd.DataFrame, window: int) -> pd.Series:
    open_, high, low, close = (
        _num(frame, x) for x in ["adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"]
    )
    span = (high - low).replace(0.0, np.nan)
    upper = (high - pd.concat([open_, close], axis=1).max(axis=1)) / span
    lower = (pd.concat([open_, close], axis=1).min(axis=1) - low) / span
    return (lower - upper).rolling(window, min_periods=window).mean()


# Downside volume participation differs from return-impact constructs: it asks
# whether turnover is concentrated on down days, regardless of the size of a
# single return.  Acceleration then compares short-run participation to its
# own past baseline.  Both are spot-daily and are mapped to the low side of
# the existing volume-price state by a negative context direction.
def _downside_turnover_participation(frame: pd.DataFrame, window: int) -> pd.Series:
    turnover = _num(frame, "total_turnover")
    down_turnover = turnover.where(_ret(frame).lt(0.0), 0.0)
    return down_turnover.rolling(window, min_periods=window).sum() / turnover.rolling(
        window, min_periods=window
    ).sum().replace(0.0, np.nan)


def _downside_turnover_participation_acceleration(frame: pd.DataFrame, short: int, baseline: int) -> pd.Series:
    participation = _downside_turnover_participation(frame, short)
    reference = participation.shift(1).rolling(baseline, min_periods=baseline).mean()
    return (participation - reference).clip(-1.0, 1.0)


BUILDERS.update({
    "higher_high_lower_low_balance_v59": _higher_high_lower_low_balance,
    "directional_breakout_count_v59": _directional_breakout_count,
    "ema_slope_direction_v59": _ema_slope_direction,
    "median_price_slope_direction_v59": _median_price_slope_direction,
    "short_return_direction_v513": _short_return_direction,
    "median_return_direction_v542": _median_return_direction,
    "trimmed_mean_return_direction_v542": _trimmed_mean_return_direction,
    "short_direction_run_v521": _short_direction_run,
    "sign_return_consensus_v513": _sign_return_consensus,
    "intraday_body_direction_v514": _intraday_body_direction,
    "intraday_bar_location_v514": _intraday_bar_location,
    "intraday_gap_follow_v514": _intraday_gap_follow,
    "intraday_body_sign_consensus_v524": _intraday_body_sign_consensus,
    "intraday_gap_body_agreement_v524": _intraday_gap_body_agreement,
    "intraday_close_prev_sign_v524": _intraday_close_prev_sign,
    "rolling_channel_position_direction_v59": _rolling_channel_position_direction,
    "prior_range_break_direction_v59": _prior_range_break_direction,
    "two_day_body_continuation_v59": _two_day_body_continuation,
    "close_auction_balance_v59": _close_auction_balance,
    "shadow_reversal_acceptance_v59": _shadow_reversal_acceptance,
    "downside_turnover_participation_v541": _downside_turnover_participation,
    "downside_turnover_participation_acceleration_v541": _downside_turnover_participation_acceleration,
})
