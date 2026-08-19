"""Compact spot-daily factor builders for the v5.8 weak-panel refresh.

Trend builders are pure signed direction. Intraday builders use only the current
and past OHLC bars. Volume-price builders combine signed price moves with volume
or turnover and never use futures-only fields.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_BASE_PATH = Path(__file__).resolve().with_name("factor_v55.py")
_SPEC = importlib.util.spec_from_file_location("v58_base_factor", _BASE_PATH)
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


def _relative(series: pd.Series, window: int) -> pd.Series:
    base = series.shift(1).rolling(window, min_periods=window).median()
    return series / base.replace(0.0, np.nan)


# Pure direction: no volatility, drawdown, efficiency, volume or quality gate.
def _cumulative_return_direction(frame: pd.DataFrame, window: int) -> pd.Series:
    return _price(frame).pct_change(window, fill_method=None)


def _ema_distance_direction(frame: pd.DataFrame, span: int) -> pd.Series:
    price = _price(frame)
    ema = price.ewm(span=span, adjust=False, min_periods=span).mean()
    return price / ema.replace(0.0, np.nan) - 1.0


def _median_return_direction(frame: pd.DataFrame, window: int) -> pd.Series:
    return _ret(frame).rolling(window, min_periods=window).median()


def _regression_direction(frame: pd.DataFrame, window: int) -> pd.Series:
    log_price = np.log(_price(frame).replace(0.0, np.nan))
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denom = float(np.square(x).sum())
    return log_price.rolling(window, min_periods=window).apply(
        lambda y: float(np.dot(x, y - np.mean(y)) / denom) * window,
        raw=True,
    )


# Intraday acceptance/dominance using current and past OHLC bars only.
def _body_tail_balance(frame: pd.DataFrame, window: int) -> pd.Series:
    open_, high, low, close = (
        _num(frame, x) for x in ["adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"]
    )
    span = (high - low).replace(0.0, np.nan)
    upper = (high - pd.concat([open_, close], axis=1).max(axis=1)) / span
    lower = (pd.concat([open_, close], axis=1).min(axis=1) - low) / span
    body = (close - open_) / span
    return (body + lower - upper).rolling(window, min_periods=window).mean()


def _prior_range_close_direction(frame: pd.DataFrame, window: int) -> pd.Series:
    high, low, close = (_num(frame, x) for x in ["adjusted_high", "adjusted_low", "adjusted_close"])
    prior_high = high.shift(1)
    prior_low = low.shift(1)
    value = (2.0 * close - prior_high - prior_low) / (prior_high - prior_low).replace(0.0, np.nan)
    return value.clip(-3.0, 3.0).rolling(window, min_periods=window).mean()


def _gap_rejection_direction(frame: pd.DataFrame, window: int) -> pd.Series:
    open_, high, low, close = (
        _num(frame, x) for x in ["adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"]
    )
    previous = close.shift(1)
    gap = open_ / previous.replace(0.0, np.nan) - 1.0
    intraday = close / open_.replace(0.0, np.nan) - 1.0
    span = ((high - low) / previous.replace(0.0, np.nan)).replace(0.0, np.nan)
    # A rejected up-gap is bearish; a rejected down-gap is bullish.
    accepted = np.sign(gap) * np.minimum(gap.abs(), intraday.abs())
    rejected = np.sign(intraday) * np.minimum(gap.abs(), intraday.abs())
    value = pd.Series(np.where(np.sign(gap).eq(np.sign(intraday)), accepted, rejected), index=frame.index)
    return (value / span).clip(-3.0, 3.0).rolling(window, min_periods=window).mean()


def _range_expansion_body_direction(frame: pd.DataFrame, window: int, baseline: int) -> pd.Series:
    open_, high, low, close = (
        _num(frame, x) for x in ["adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"]
    )
    span = (high - low).replace(0.0, np.nan)
    body = (close - open_) / span
    relative_range = _relative(span / close.shift(1).replace(0.0, np.nan), baseline).clip(0.0, 4.0)
    return (body * relative_range).rolling(window, min_periods=window).mean()


# Signed volume-price states: high bullish confirmation, low bearish pressure.
def _signed_turnover_share(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = _ret(frame)
    turnover = _num(frame, "total_turnover")
    signed = np.sign(ret) * turnover
    return signed.rolling(window, min_periods=window).sum() / turnover.rolling(
        window, min_periods=window
    ).sum().replace(0.0, np.nan)


def _signed_volume_weighted_return(frame: pd.DataFrame, window: int, baseline: int) -> pd.Series:
    ret = _ret(frame)
    relative_volume = _relative(_num(frame, "volume"), baseline).clip(0.0, 4.0)
    flow = ret * relative_volume
    scale = (ret.abs() * relative_volume).rolling(window, min_periods=window).sum()
    return flow.rolling(window, min_periods=window).sum() / scale.replace(0.0, np.nan)


def _turnover_surprise_direction(frame: pd.DataFrame, window: int, baseline: int) -> pd.Series:
    ret = _ret(frame)
    surprise = np.log(_relative(_num(frame, "total_turnover"), baseline).clip(0.25, 4.0))
    value = np.sign(ret) * surprise.abs()
    return value.rolling(window, min_periods=window).mean()


def _downside_turnover_acceleration(frame: pd.DataFrame, window: int, baseline: int) -> pd.Series:
    ret = _ret(frame)
    relative_turnover = _relative(_num(frame, "total_turnover"), baseline).clip(0.0, 4.0)
    pressure = (-ret).clip(lower=0.0) * relative_turnover
    return pressure.rolling(window, min_periods=window).mean()


BUILDERS.update({
    "cumulative_return_direction_v58": _cumulative_return_direction,
    "ema_distance_direction_v58": _ema_distance_direction,
    "median_return_direction_v58": _median_return_direction,
    "regression_direction_v58": _regression_direction,
    "body_tail_balance_v58": _body_tail_balance,
    "prior_range_close_direction_v58": _prior_range_close_direction,
    "gap_rejection_direction_v58": _gap_rejection_direction,
    "range_expansion_body_direction_v58": _range_expansion_body_direction,
    "signed_turnover_share_v58": _signed_turnover_share,
    "signed_volume_weighted_return_v58": _signed_volume_weighted_return,
    "turnover_surprise_direction_v58": _turnover_surprise_direction,
    "downside_turnover_acceleration_v58": _downside_turnover_acceleration,
})
