"""v5.1 factor library: v5 families plus bounded weak-state challengers."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_V5_PATH = Path(__file__).resolve().with_name("factor_v5.py")
_SPEC = importlib.util.spec_from_file_location("v51_base_factor", _V5_PATH)
_BASE = importlib.util.module_from_spec(_SPEC); sys.modules[_SPEC.name] = _BASE; _SPEC.loader.exec_module(_BASE)
FeatureSpec = _BASE.FeatureSpec
rolling_zscore = _BASE.rolling_zscore
BUILDERS = dict(_BASE.BUILDERS)


def _ret(frame): return pd.to_numeric(frame["adjusted_ret"], errors="coerce")
def _price(frame): return (1.0 + _ret(frame).fillna(0.0)).cumprod()


def _trend_sign_stability(frame: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    price = _price(frame)
    fast_ret = price.pct_change(fast, fill_method=None)
    slow_ret = price.pct_change(slow, fill_method=None)
    agreement = np.sign(fast_ret) * (np.sign(fast_ret) == np.sign(slow_ret))
    return agreement * np.sqrt(fast_ret.abs() * slow_ret.abs())


def _directional_breakout_balance(frame: pd.DataFrame, window: int, smooth: int) -> pd.Series:
    price = _price(frame)
    high = price.shift(1).rolling(window, min_periods=window).max()
    low = price.shift(1).rolling(window, min_periods=window).min()
    event = price.gt(high).astype(float) - price.lt(low).astype(float)
    return event.rolling(smooth, min_periods=smooth).sum() / float(smooth)


def _drawdown_resilient_trend(frame: pd.DataFrame, window: int) -> pd.Series:
    price = _price(frame); momentum = price.pct_change(window, fill_method=None)
    peak = price.rolling(window, min_periods=window).max(); drawdown = (price / peak - 1.0).abs()
    return momentum / (1.0 + 5.0 * drawdown)


def _signed_relative_volume_flow(frame: pd.DataFrame, window: int, volume_window: int) -> pd.Series:
    ret = _ret(frame); volume = pd.to_numeric(frame["volume"], errors="coerce")
    rel = volume / volume.shift(1).rolling(volume_window, min_periods=volume_window).median().replace(0.0, np.nan)
    num = (ret * rel.clip(0.0, 5.0)).rolling(window, min_periods=window).sum()
    den = ret.abs().rolling(window, min_periods=window).sum().replace(0.0, np.nan)
    return num / den


def _downside_turnover_pressure(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = _ret(frame); turnover = pd.to_numeric(frame["total_turnover"], errors="coerce")
    rel = turnover / turnover.shift(1).rolling(window, min_periods=window).median().replace(0.0, np.nan)
    pressure = (-ret).clip(lower=0.0) * rel.clip(0.0, 5.0)
    return pressure.rolling(window, min_periods=window).sum()


def _price_volume_trend_agreement(frame: pd.DataFrame, price_window: int, volume_window: int) -> pd.Series:
    price_move = _price(frame).pct_change(price_window, fill_method=None)
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    volume_move = volume / volume.shift(volume_window).replace(0.0, np.nan) - 1.0
    return np.sign(price_move) * np.sqrt(price_move.abs() * volume_move.abs().clip(upper=5.0)) * (np.sign(price_move) == np.sign(volume_move))


def _intraday_follow_through_efficiency(frame: pd.DataFrame, window: int) -> pd.Series:
    open_ = pd.to_numeric(frame["adjusted_open"], errors="coerce")
    close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    high = pd.to_numeric(frame["adjusted_high"], errors="coerce")
    low = pd.to_numeric(frame["adjusted_low"], errors="coerce")
    body = (close - open_) / open_.replace(0.0, np.nan)
    span = (high - low).abs() / open_.replace(0.0, np.nan)
    return (body / span.replace(0.0, np.nan)).rolling(window, min_periods=window).mean()


def _gap_close_alignment(frame: pd.DataFrame, window: int) -> pd.Series:
    open_ = pd.to_numeric(frame["adjusted_open"], errors="coerce")
    close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    gap = open_ / close.shift(1) - 1.0; intraday = close / open_ - 1.0
    aligned = np.sign(gap) * intraday
    return aligned.rolling(window, min_periods=window).mean()


def _downside_realized_share(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = _ret(frame); down = ret.pow(2).where(ret.lt(0), 0.0).rolling(window, min_periods=window).sum()
    total = ret.pow(2).rolling(window, min_periods=window).sum().replace(0.0, np.nan)
    return down / total


def _drawdown_path_efficiency(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = _ret(frame); net_down = (-_price(frame).pct_change(window, fill_method=None)).clip(lower=0.0)
    negative_path = (-ret.clip(upper=0.0)).rolling(window, min_periods=window).sum().replace(0.0, np.nan)
    return net_down / negative_path


def _failed_high_close_pressure(frame: pd.DataFrame, background: int, trigger: int) -> pd.Series:
    high = pd.to_numeric(frame["adjusted_high"], errors="coerce")
    close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    open_ = pd.to_numeric(frame["adjusted_open"], errors="coerce")
    prior_high = high.shift(1).rolling(background, min_periods=background).max()
    at_high = (high / prior_high.replace(0.0, np.nan)).clip(0.0, 1.2)
    weakness = ((high - close) / (high - pd.to_numeric(frame["adjusted_low"], errors="coerce")).replace(0.0, np.nan)).clip(0.0, 1.0)
    red = ((open_ - close) / open_.replace(0.0, np.nan)).clip(lower=0.0)
    return at_high * (weakness.rolling(trigger, min_periods=trigger).mean() + red.rolling(trigger, min_periods=trigger).sum())


def _volume_exhaustion_rollover(frame: pd.DataFrame, background: int, trigger: int) -> pd.Series:
    price = _price(frame); volume = pd.to_numeric(frame["volume"], errors="coerce")
    rise = price.shift(trigger).pct_change(background, fill_method=None).clip(lower=0.0)
    rollover = (-price.pct_change(trigger, fill_method=None)).clip(lower=0.0)
    dry = (1.0 - volume / volume.shift(1).rolling(background, min_periods=background).mean().replace(0.0, np.nan)).clip(lower=0.0)
    return np.sqrt(rise * rollover) * (1.0 + dry)


BUILDERS.update({
    "trend_sign_stability": _trend_sign_stability,
    "directional_breakout_balance": _directional_breakout_balance,
    "drawdown_resilient_trend": _drawdown_resilient_trend,
    "signed_relative_volume_flow": _signed_relative_volume_flow,
    "downside_turnover_pressure": _downside_turnover_pressure,
    "price_volume_trend_agreement": _price_volume_trend_agreement,
    "intraday_follow_through_efficiency": _intraday_follow_through_efficiency,
    "gap_close_alignment": _gap_close_alignment,
    "downside_realized_share": _downside_realized_share,
    "drawdown_path_efficiency": _drawdown_path_efficiency,
    "failed_high_close_pressure": _failed_high_close_pressure,
    "volume_exhaustion_rollover": _volume_exhaustion_rollover,
})
