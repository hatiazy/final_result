"""Targeted weak-state factor families for the v5.3 local research package."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_BASE_PATH = Path(__file__).resolve().with_name("factor_v51.py")
_SPEC = importlib.util.spec_from_file_location("v53_base_factor", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)

FeatureSpec = _BASE.FeatureSpec
rolling_zscore = _BASE.rolling_zscore
BUILDERS = dict(_BASE.BUILDERS)


def _ret(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame["adjusted_ret"], errors="coerce")


def _price(frame: pd.DataFrame) -> pd.Series:
    return (1.0 + _ret(frame).fillna(0.0)).cumprod()


def _short_medium_direction_quality(frame: pd.DataFrame, short: int, medium: int) -> pd.Series:
    ret = _ret(frame); price = _price(frame)
    short_move = price.pct_change(short, fill_method=None)
    medium_move = price.pct_change(medium, fill_method=None)
    path = ret.abs().rolling(medium, min_periods=medium).sum().replace(0.0, np.nan)
    quality = medium_move.abs() / path
    agreement = (np.sign(short_move) == np.sign(medium_move)).astype(float)
    return np.sign(medium_move) * np.sqrt(short_move.abs() * medium_move.abs()) * quality * agreement


def _slope_efficiency_blend(frame: pd.DataFrame, window: int, quality_window: int) -> pd.Series:
    price = _price(frame); ret = _ret(frame)
    direction = price.pct_change(window, fill_method=None)
    path = ret.abs().rolling(quality_window, min_periods=quality_window).sum().replace(0.0, np.nan)
    efficiency = price.pct_change(quality_window, fill_method=None).abs() / path
    return direction * efficiency.clip(0.0, 1.0)


def _volume_weighted_return_efficiency(frame: pd.DataFrame, price_window: int, volume_window: int) -> pd.Series:
    ret = _ret(frame); volume = pd.to_numeric(frame["volume"], errors="coerce")
    relative = volume / volume.shift(1).rolling(volume_window, min_periods=volume_window).median().replace(0.0, np.nan)
    weighted = (ret * relative.clip(0.0, 4.0)).rolling(price_window, min_periods=price_window).sum()
    scale = (ret.abs() * relative.clip(0.0, 4.0)).rolling(price_window, min_periods=price_window).sum().replace(0.0, np.nan)
    return weighted / scale


def _downside_volume_impact_ratio(frame: pd.DataFrame, window: int, volume_window: int) -> pd.Series:
    ret = _ret(frame); volume = pd.to_numeric(frame["volume"], errors="coerce")
    relative = volume / volume.shift(1).rolling(volume_window, min_periods=volume_window).median().replace(0.0, np.nan)
    down = ((-ret).clip(lower=0.0) * relative.clip(0.0, 4.0)).rolling(window, min_periods=window).sum()
    up = (ret.clip(lower=0.0) * relative.clip(0.0, 4.0)).rolling(window, min_periods=window).sum()
    return down / (up + down).replace(0.0, np.nan)


def _volume_confirmation_persistence(frame: pd.DataFrame, price_window: int, volume_window: int) -> pd.Series:
    ret = _ret(frame); price_move = _price(frame).pct_change(price_window, fill_method=None)
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    relative = volume / volume.shift(1).rolling(volume_window, min_periods=volume_window).median().replace(0.0, np.nan)
    signed_days = np.sign(ret) * relative.clip(0.0, 4.0)
    persistence = signed_days.rolling(price_window, min_periods=price_window).mean()
    return np.sign(price_move) * persistence.abs() * (np.sign(price_move) == np.sign(persistence))


def _high_momentum_rollover_gate(frame: pd.DataFrame, background: int, trigger: int) -> pd.Series:
    price = _price(frame)
    rise = price.shift(trigger).pct_change(background, fill_method=None).clip(lower=0.0)
    prior_high = price.shift(1).rolling(background, min_periods=background).max()
    high_position = (price / prior_high.replace(0.0, np.nan)).clip(0.0, 1.05)
    rollover = (-price.pct_change(trigger, fill_method=None)).clip(lower=0.0)
    return np.sqrt(rise * rollover) * high_position


def _failed_breakout_close_gate(frame: pd.DataFrame, background: int, trigger: int) -> pd.Series:
    high = pd.to_numeric(frame["adjusted_high"], errors="coerce")
    low = pd.to_numeric(frame["adjusted_low"], errors="coerce")
    close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    price = _price(frame)
    prior_high = high.shift(1).rolling(background, min_periods=background).max()
    attempted = (high / prior_high.replace(0.0, np.nan) - 1.0).clip(lower=0.0)
    rejected = ((prior_high - close) / prior_high.replace(0.0, np.nan)).clip(lower=0.0)
    close_weakness = ((high - close) / (high - low).replace(0.0, np.nan)).clip(0.0, 1.0)
    rise = price.shift(trigger).pct_change(background, fill_method=None).clip(lower=0.0)
    return np.sqrt(rise * (attempted + rejected)) * close_weakness.rolling(trigger, min_periods=trigger).mean()


def _high_volume_divergence_gate(frame: pd.DataFrame, background: int, trigger: int) -> pd.Series:
    price = _price(frame); volume = pd.to_numeric(frame["volume"], errors="coerce")
    rise = price.shift(trigger).pct_change(background, fill_method=None).clip(lower=0.0)
    recent_price = price.pct_change(trigger, fill_method=None)
    volume_slow = volume.shift(trigger).rolling(background, min_periods=background).mean()
    volume_recent = volume.rolling(trigger, min_periods=trigger).mean()
    dry = (1.0 - volume_recent / volume_slow.replace(0.0, np.nan)).clip(lower=0.0)
    stall = (-recent_price).clip(lower=0.0) + recent_price.abs().rolling(trigger, min_periods=trigger).mean() * 0.25
    return np.sqrt(rise * stall.clip(lower=0.0)) * (1.0 + dry.clip(upper=2.0))


def _recovery_failure_pressure(frame: pd.DataFrame, background: int, recovery: int) -> pd.Series:
    price = _price(frame)
    peak = price.shift(recovery).rolling(background, min_periods=background).max()
    drawdown = (1.0 - price.shift(recovery) / peak.replace(0.0, np.nan)).clip(lower=0.0)
    rebound = price.pct_change(recovery, fill_method=None).clip(lower=0.0)
    renewed_loss = (-price.pct_change(recovery, fill_method=None)).clip(lower=0.0)
    return drawdown * (1.0 + renewed_loss) / (1.0 + rebound)


def _downside_path_acceleration(frame: pd.DataFrame, short: int, long: int) -> pd.Series:
    price = _price(frame)
    short_down = (-price.pct_change(short, fill_method=None)).clip(lower=0.0) / np.sqrt(short)
    long_down = (-price.pct_change(long, fill_method=None)).clip(lower=0.0) / np.sqrt(long)
    return (short_down - long_down).clip(lower=0.0)


def _loss_cluster_acceleration(frame: pd.DataFrame, short: int, long: int) -> pd.Series:
    loss = _ret(frame).lt(0).astype(float)
    short_rate = loss.rolling(short, min_periods=short).mean()
    long_rate = loss.rolling(long, min_periods=long).mean()
    return (short_rate - long_rate).clip(lower=0.0) * short_rate


def _close_location_persistence(frame: pd.DataFrame, window: int) -> pd.Series:
    high = pd.to_numeric(frame["adjusted_high"], errors="coerce")
    low = pd.to_numeric(frame["adjusted_low"], errors="coerce")
    close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    location = (2.0 * close - high - low) / (high - low).replace(0.0, np.nan)
    return location.rolling(window, min_periods=window).mean()


def _overnight_intraday_alignment(frame: pd.DataFrame, window: int) -> pd.Series:
    open_ = pd.to_numeric(frame["adjusted_open"], errors="coerce")
    close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    gap = open_ / close.shift(1) - 1.0
    intraday = close / open_.replace(0.0, np.nan) - 1.0
    aligned = np.sign(gap) * np.sqrt(gap.abs() * intraday.abs()) * (np.sign(gap) == np.sign(intraday))
    return aligned.rolling(window, min_periods=window).mean()


def _rejection_support_balance(frame: pd.DataFrame, window: int) -> pd.Series:
    high = pd.to_numeric(frame["adjusted_high"], errors="coerce")
    low = pd.to_numeric(frame["adjusted_low"], errors="coerce")
    open_ = pd.to_numeric(frame["adjusted_open"], errors="coerce")
    close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    span = (high - low).replace(0.0, np.nan)
    upper = (high - pd.concat([open_, close], axis=1).max(axis=1)) / span
    lower = (pd.concat([open_, close], axis=1).min(axis=1) - low) / span
    return (lower - upper).rolling(window, min_periods=window).mean()


BUILDERS.update({
    "short_medium_direction_quality": _short_medium_direction_quality,
    "slope_efficiency_blend": _slope_efficiency_blend,
    "volume_weighted_return_efficiency": _volume_weighted_return_efficiency,
    "downside_volume_impact_ratio": _downside_volume_impact_ratio,
    "volume_confirmation_persistence": _volume_confirmation_persistence,
    "high_momentum_rollover_gate": _high_momentum_rollover_gate,
    "failed_breakout_close_gate": _failed_breakout_close_gate,
    "high_volume_divergence_gate": _high_volume_divergence_gate,
    "recovery_failure_pressure": _recovery_failure_pressure,
    "downside_path_acceleration": _downside_path_acceleration,
    "loss_cluster_acceleration": _loss_cluster_acceleration,
    "close_location_persistence": _close_location_persistence,
    "overnight_intraday_alignment": _overnight_intraday_alignment,
    "rejection_support_balance": _rejection_support_balance,
})
