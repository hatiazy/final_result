"""Merged, self-contained v5.51 spot-daily construction library.

The production direction, intraday, acceptance and volume builders are kept
as versioned components below ``库函数/因子组件``. Production uses their
explicit union and never resolves code from an external research workspace.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


COMPONENTS = Path(__file__).resolve().parent / "因子组件"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_V548 = _load(
    "company_v548_factor",
    COMPONENTS / "factor_v548.py",
)
_V58 = _load(
    "company_v58_factor",
    COMPONENTS / "factor_v58.py",
)

FeatureSpec = _V548.FeatureSpec
rolling_zscore = _V548.rolling_zscore
_ALL_BUILDERS = {**_V58.BUILDERS, **_V548.BUILDERS}

# Only these spot/OHLCV families are reachable from the frozen eight-state
# recipes. Keeping the whitelist at the public dispatch boundary prevents a
# future caller from accidentally selecting an inherited OI/futures builder.
SPOT_STATE_FAMILIES = frozenset({
    "recovery_slope_vol_adjusted", "donchian_position", "stochastic_position",
    "distance_from_ma", "momentum", "price_volume_divergence",
    "high_position_rejection_v55",
    "left_tail_loss_ratio", "range_vol_ratio", "worst_k_loss", "high_vol_negative_trend",
    "higher_high_lower_low_balance_v59", "median_price_slope_direction_v59", "short_return_direction_v513",
    "choppiness_index", "drawdown_acceleration_down", "drawdown_recovery_gap",
    "gap_close_alignment", "intraday_return", "gap_return",
    "rsi_pressure", "gain_concentration", "up_volume_exhaustion",
    "price_volume_trend_agreement", "down_volume_pressure", "volume_return_confirm",
})
BUILDERS = {name: _ALL_BUILDERS[name] for name in SPOT_STATE_FAMILIES if name in _ALL_BUILDERS}
if SPOT_STATE_FAMILIES.difference(BUILDERS):
    raise ImportError(f"spot state builder whitelist missing families: {sorted(SPOT_STATE_FAMILIES.difference(BUILDERS))}")
