"""Additional economically targeted families for the stable nine-state baseline."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


_BASE_PATH = Path(__file__).resolve().parents[1] / "full_family_factor_library_20260704.py"
_SPEC = importlib.util.spec_from_file_location("stable_base_factor_library", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)

FeatureSpec = _BASE.FeatureSpec
rolling_zscore = _BASE.rolling_zscore
BUILDERS = dict(_BASE.BUILDERS)


def _price(frame: pd.DataFrame) -> pd.Series:
    ret = pd.to_numeric(frame["adjusted_ret"], errors="coerce").fillna(0.0)
    return (1.0 + ret).cumprod()


def _multi_horizon_direction_consensus(frame: pd.DataFrame, short: int, medium: int, long: int) -> pd.Series:
    price = _price(frame)
    parts = [np.sign(price.pct_change(w, fill_method=None)) for w in (short, medium, long)]
    return pd.concat(parts, axis=1).mean(axis=1)


def _signed_trend_efficiency(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = pd.to_numeric(frame["adjusted_ret"], errors="coerce")
    net = _price(frame).pct_change(window, fill_method=None)
    path = ret.abs().rolling(window, min_periods=window).sum().replace(0.0, np.nan)
    return net / path


def _breakout_persistence(frame: pd.DataFrame, window: int, persistence: int) -> pd.Series:
    price = _price(frame)
    high = price.shift(1).rolling(window, min_periods=window).max()
    low = price.shift(1).rolling(window, min_periods=window).min()
    event = price.gt(high).astype(float) - price.lt(low).astype(float)
    return event.rolling(persistence, min_periods=persistence).mean()


def _signed_volume_surprise(frame: pd.DataFrame, price_window: int, volume_window: int) -> pd.Series:
    direction = np.sign(_price(frame).pct_change(price_window, fill_method=None))
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    baseline = volume.shift(1).rolling(volume_window, min_periods=volume_window).median()
    surprise = (volume / baseline.replace(0.0, np.nan) - 1.0).clip(-3.0, 3.0)
    return direction * surprise


def _volume_direction_asymmetry(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = pd.to_numeric(frame["adjusted_ret"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    up = volume.where(ret.gt(0), 0.0).rolling(window, min_periods=window).sum()
    down = volume.where(ret.lt(0), 0.0).rolling(window, min_periods=window).sum()
    return (up - down) / (up + down).replace(0.0, np.nan)


def _obv_slope_normalized(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = pd.to_numeric(frame["adjusted_ret"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    signed = np.sign(ret) * volume
    return signed.rolling(window, min_periods=window).sum() / volume.rolling(window, min_periods=window).sum().replace(0.0, np.nan)


def _turnover_impact_direction(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = pd.to_numeric(frame["adjusted_ret"], errors="coerce")
    turnover = pd.to_numeric(frame["total_turnover"], errors="coerce").abs()
    scale = turnover.shift(1).rolling(window, min_periods=window).median().replace(0.0, np.nan)
    impact = ret * (scale / turnover.replace(0.0, np.nan)).clip(0.0, 10.0)
    return impact.rolling(window, min_periods=window).sum()


def _gap_follow_through(frame: pd.DataFrame, window: int) -> pd.Series:
    open_ = pd.to_numeric(frame["adjusted_open"], errors="coerce")
    close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    gap = open_ / close.shift(1) - 1.0
    intraday = close / open_ - 1.0
    follow = np.sign(gap) * intraday
    return (gap + np.sign(gap) * follow.abs()).rolling(window, min_periods=window).mean()


def _gap_recovery_balance(frame: pd.DataFrame, window: int) -> pd.Series:
    open_ = pd.to_numeric(frame["adjusted_open"], errors="coerce")
    close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    gap = open_ / close.shift(1) - 1.0
    intraday = close / open_ - 1.0
    return (gap + intraday).rolling(window, min_periods=window).sum()


def _downside_jump_share(frame: pd.DataFrame, window: int, multiplier: float) -> pd.Series:
    ret = pd.to_numeric(frame["adjusted_ret"], errors="coerce")
    vol = ret.shift(1).rolling(window, min_periods=window).std(ddof=0)
    jump = ret.lt(-float(multiplier) * vol)
    downside = ret.abs().where(jump, 0.0).rolling(window, min_periods=window).sum()
    total = ret.abs().rolling(window, min_periods=window).sum().replace(0.0, np.nan)
    return downside / total


def _liquidity_impact_pressure(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = pd.to_numeric(frame["adjusted_ret"], errors="coerce").abs()
    turnover = pd.to_numeric(frame["total_turnover"], errors="coerce").abs().replace(0.0, np.nan)
    normalized_turnover = turnover / turnover.shift(1).rolling(window, min_periods=window).median()
    return (ret / normalized_turnover.replace(0.0, np.nan)).rolling(window, min_periods=window).mean()


def _failed_breakout_pressure(frame: pd.DataFrame, background: int, trigger: int) -> pd.Series:
    price = _price(frame)
    prior_high = price.shift(1).rolling(background, min_periods=background).max()
    near_high = (price / prior_high.replace(0.0, np.nan)).clip(0.0, 1.1)
    weakness = (-pd.to_numeric(frame["adjusted_ret"], errors="coerce").rolling(trigger, min_periods=trigger).sum()).clip(lower=0.0)
    return near_high * weakness


def _momentum_rollover_pressure(frame: pd.DataFrame, background: int, trigger: int) -> pd.Series:
    price = _price(frame)
    prior_rise = price.shift(trigger).pct_change(background, fill_method=None).clip(lower=0.0)
    recent = (-price.pct_change(trigger, fill_method=None)).clip(lower=0.0)
    return np.sqrt(prior_rise * recent)


BUILDERS.update({
    "multi_horizon_direction_consensus": _multi_horizon_direction_consensus,
    "signed_trend_efficiency": _signed_trend_efficiency,
    "breakout_persistence": _breakout_persistence,
    "signed_volume_surprise": _signed_volume_surprise,
    "volume_direction_asymmetry": _volume_direction_asymmetry,
    "obv_slope_normalized": _obv_slope_normalized,
    "turnover_impact_direction": _turnover_impact_direction,
    "gap_follow_through": _gap_follow_through,
    "gap_recovery_balance": _gap_recovery_balance,
    "downside_jump_share": _downside_jump_share,
    "liquidity_impact_pressure": _liquidity_impact_pressure,
    "failed_breakout_pressure": _failed_breakout_pressure,
    "momentum_rollover_pressure": _momentum_rollover_pressure,
})
