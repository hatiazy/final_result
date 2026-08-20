"""Economically constrained spot-daily factors for the v5.5 weak-panel refresh."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_BASE_PATH = Path(__file__).resolve().with_name("factor_v53.py")
_SPEC = importlib.util.spec_from_file_location("v55_base_factor", _BASE_PATH)
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
    baseline = series.shift(1).rolling(window, min_periods=window).median()
    return series / baseline.replace(0.0, np.nan)


# Pure direction: no volatility, drawdown, efficiency or quality gate enters these families.
def _multi_scale_direction_vote(frame: pd.DataFrame, short: int, medium: int, long: int) -> pd.Series:
    price = _price(frame)
    moves = pd.concat([
        price.pct_change(short, fill_method=None),
        price.pct_change(medium, fill_method=None),
        price.pct_change(long, fill_method=None),
    ], axis=1)
    return np.sign(moves).mean(axis=1)


def _ema_stack_direction(frame: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    price = _price(frame)
    fast_ema = price.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = price.ewm(span=slow, adjust=False, min_periods=slow).mean()
    return fast_ema / slow_ema.replace(0.0, np.nan) - 1.0


def _channel_direction_balance(frame: pd.DataFrame, window: int) -> pd.Series:
    price = _price(frame)
    high = price.shift(1).rolling(window, min_periods=window).max()
    low = price.shift(1).rolling(window, min_periods=window).min()
    return 2.0 * (price - low) / (high - low).replace(0.0, np.nan) - 1.0


def _direction_run_balance(frame: pd.DataFrame, window: int) -> pd.Series:
    return np.sign(_ret(frame)).rolling(window, min_periods=window).mean()


# Volume-price: high means bullish confirmation; low means bearish volume pressure.
def _signed_turnover_impulse(frame: pd.DataFrame, window: int, baseline: int) -> pd.Series:
    ret = _ret(frame)
    turnover = _num(frame, "total_turnover")
    rel = _relative(turnover, baseline).clip(0.0, 4.0)
    flow = np.sign(ret) * ret.abs() * rel
    scale = (ret.abs() * rel).rolling(window, min_periods=window).sum().replace(0.0, np.nan)
    return flow.rolling(window, min_periods=window).sum() / scale


def _up_down_turnover_balance(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = _ret(frame)
    turnover = _num(frame, "total_turnover")
    up = turnover.where(ret.gt(0.0), 0.0).rolling(window, min_periods=window).sum()
    down = turnover.where(ret.lt(0.0), 0.0).rolling(window, min_periods=window).sum()
    return (up - down) / (up + down).replace(0.0, np.nan)


def _close_location_turnover_flow(frame: pd.DataFrame, window: int, baseline: int) -> pd.Series:
    high, low, close = (_num(frame, x) for x in ["adjusted_high", "adjusted_low", "adjusted_close"])
    location = (2.0 * close - high - low) / (high - low).replace(0.0, np.nan)
    rel = _relative(_num(frame, "total_turnover"), baseline).clip(0.0, 4.0)
    return (location * rel).rolling(window, min_periods=window).mean()


def _signed_volume_impulse_decay(frame: pd.DataFrame, short: int, long: int) -> pd.Series:
    ret = _ret(frame)
    volume = _num(frame, "volume")
    rel = _relative(volume, long).clip(0.0, 4.0)
    flow = np.sign(ret) * rel
    return flow.rolling(short, min_periods=short).mean() - flow.rolling(long, min_periods=long).mean()


# Path fragility: all families describe damaged paths, not generic volatility.
def _underwater_duration_pressure(frame: pd.DataFrame, window: int) -> pd.Series:
    price = _price(frame)
    peak = price.rolling(window, min_periods=window).max()
    drawdown = (1.0 - price / peak.replace(0.0, np.nan)).clip(lower=0.0)
    underwater = drawdown.gt(0.002).astype(float).rolling(window, min_periods=window).mean()
    return drawdown * underwater


def _new_low_range_stress(frame: pd.DataFrame, window: int, range_window: int) -> pd.Series:
    high, low, close = (_num(frame, x) for x in ["adjusted_high", "adjusted_low", "adjusted_close"])
    span = (high - low).replace(0.0, np.nan)
    close_weak = ((high - close) / span).clip(0.0, 1.0)
    rel_range = _relative(span / close.shift(1).replace(0.0, np.nan), range_window).clip(0.0, 4.0)
    rolling_low = close.shift(1).rolling(window, min_periods=window).min()
    near_low = (1.0 - close / rolling_low.replace(0.0, np.nan)).clip(lower=0.0) + close.le(rolling_low).astype(float)
    return (close_weak * rel_range * near_low).rolling(max(2, window // 4), min_periods=max(2, window // 4)).mean()


def _drawdown_repair_asymmetry(frame: pd.DataFrame, background: int, trigger: int) -> pd.Series:
    price = _price(frame)
    peak = price.shift(trigger).rolling(background, min_periods=background).max()
    prior_dd = (1.0 - price.shift(trigger) / peak.replace(0.0, np.nan)).clip(lower=0.0)
    recent = price.pct_change(trigger, fill_method=None)
    renewed_loss = (-recent).clip(lower=0.0)
    repair = recent.clip(lower=0.0)
    return prior_dd * (1.0 + renewed_loss) / (1.0 + 2.0 * repair)


# Intraday acceptance: signed states remain continuous on every available daily bar.
def _gap_acceptance_direction(frame: pd.DataFrame, window: int) -> pd.Series:
    open_, close = (_num(frame, x) for x in ["adjusted_open", "adjusted_close"])
    gap = open_ / close.shift(1).replace(0.0, np.nan) - 1.0
    intraday = close / open_.replace(0.0, np.nan) - 1.0
    same = np.sign(gap).eq(np.sign(intraday))
    accepted = np.sign(gap) * np.sqrt(gap.abs() * intraday.abs())
    rejected = np.sign(intraday) * np.sqrt(gap.abs() * intraday.abs())
    value = pd.Series(np.where(same, accepted, rejected), index=frame.index)
    return value.rolling(window, min_periods=window).mean()


def _range_body_direction(frame: pd.DataFrame, window: int, baseline: int) -> pd.Series:
    open_, high, low, close = (_num(frame, x) for x in ["adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"])
    span = (high - low).replace(0.0, np.nan)
    body = (close - open_) / span
    rel_range = _relative(span / close.shift(1).replace(0.0, np.nan), baseline).clip(0.0, 4.0)
    return (body * rel_range).rolling(window, min_periods=window).mean()


def _close_location_range_flow(frame: pd.DataFrame, window: int, baseline: int) -> pd.Series:
    high, low, close = (_num(frame, x) for x in ["adjusted_high", "adjusted_low", "adjusted_close"])
    span = (high - low).replace(0.0, np.nan)
    location = (2.0 * close - high - low) / span
    rel_range = _relative(span / close.shift(1).replace(0.0, np.nan), baseline).clip(0.0, 4.0)
    return (location * rel_range).rolling(window, min_periods=window).mean()


def _gap_close_location_balance(frame: pd.DataFrame, window: int) -> pd.Series:
    open_, high, low, close = (_num(frame, x) for x in ["adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"])
    gap = open_ / close.shift(1).replace(0.0, np.nan) - 1.0
    location = (2.0 * close - high - low) / (high - low).replace(0.0, np.nan)
    direction = np.sign(gap) * gap.abs() + location * gap.abs()
    return direction.rolling(window, min_periods=window).mean()


# Reversal pressure: each family already requires a high/rising background and a weakening trigger.
def _high_position_rollover(frame: pd.DataFrame, background: int, trigger: int) -> pd.Series:
    price = _price(frame)
    high = price.shift(trigger).rolling(background, min_periods=background).max()
    low = price.shift(trigger).rolling(background, min_periods=background).min()
    position = ((price.shift(trigger) - low) / (high - low).replace(0.0, np.nan)).clip(0.0, 1.0)
    rise = price.shift(trigger).pct_change(background, fill_method=None).clip(lower=0.0)
    rollover = (-price.pct_change(trigger, fill_method=None)).clip(lower=0.0)
    return position * np.sqrt(rise * rollover)


def _high_position_rejection(frame: pd.DataFrame, background: int, trigger: int) -> pd.Series:
    price = _price(frame)
    high, low, open_, close = (_num(frame, x) for x in ["adjusted_high", "adjusted_low", "adjusted_open", "adjusted_close"])
    prior_high = high.shift(1).rolling(background, min_periods=background).max()
    position = (close.shift(1) / prior_high.replace(0.0, np.nan)).clip(0.0, 1.05)
    rise = price.shift(trigger).pct_change(background, fill_method=None).clip(lower=0.0)
    upper = (high - pd.concat([open_, close], axis=1).max(axis=1)) / (high - low).replace(0.0, np.nan)
    weak_close = ((open_ - close) / (high - low).replace(0.0, np.nan)).clip(lower=0.0)
    trigger_value = (upper + weak_close).rolling(trigger, min_periods=trigger).mean()
    return position * np.sqrt(rise * trigger_value.clip(lower=0.0))


def _failed_new_high_pressure(frame: pd.DataFrame, background: int, trigger: int) -> pd.Series:
    price = _price(frame)
    high, close = (_num(frame, x) for x in ["adjusted_high", "adjusted_close"])
    prior_high = high.shift(1).rolling(background, min_periods=background).max()
    attempted = (high / prior_high.replace(0.0, np.nan) - 1.0).clip(lower=0.0)
    failed = ((prior_high - close) / prior_high.replace(0.0, np.nan)).clip(lower=0.0)
    rise = price.shift(trigger).pct_change(background, fill_method=None).clip(lower=0.0)
    return np.sqrt(rise * (attempted + failed).rolling(trigger, min_periods=trigger).mean().clip(lower=0.0))


def _overheat_turnover_stall(frame: pd.DataFrame, background: int, trigger: int, turnover_window: int) -> pd.Series:
    price = _price(frame)
    rise = price.shift(trigger).pct_change(background, fill_method=None).clip(lower=0.0)
    recent = price.pct_change(trigger, fill_method=None)
    stall = (-recent).clip(lower=0.0) + recent.abs().mul(0.25)
    rel_turnover = _relative(_num(frame, "total_turnover"), turnover_window).clip(0.0, 4.0)
    return np.sqrt(rise * stall.clip(lower=0.0)) * rel_turnover.rolling(trigger, min_periods=trigger).mean()


BUILDERS.update({
    "multi_scale_direction_vote_v55": _multi_scale_direction_vote,
    "ema_stack_direction_v55": _ema_stack_direction,
    "channel_direction_balance_v55": _channel_direction_balance,
    "direction_run_balance_v55": _direction_run_balance,
    "signed_turnover_impulse_v55": _signed_turnover_impulse,
    "up_down_turnover_balance_v55": _up_down_turnover_balance,
    "close_location_turnover_flow_v55": _close_location_turnover_flow,
    "signed_volume_impulse_decay_v55": _signed_volume_impulse_decay,
    "underwater_duration_pressure_v55": _underwater_duration_pressure,
    "new_low_range_stress_v55": _new_low_range_stress,
    "drawdown_repair_asymmetry_v55": _drawdown_repair_asymmetry,
    "gap_acceptance_direction_v55": _gap_acceptance_direction,
    "range_body_direction_v55": _range_body_direction,
    "close_location_range_flow_v55": _close_location_range_flow,
    "gap_close_location_balance_v55": _gap_close_location_balance,
    "high_position_rollover_v55": _high_position_rollover,
    "high_position_rejection_v55": _high_position_rejection,
    "failed_new_high_pressure_v55": _failed_new_high_pressure,
    "overheat_turnover_stall_v55": _overheat_turnover_stall,
})
