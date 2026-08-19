from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .specs import VERSION_SPECS


RAW_FIELDS = {"open", "high", "low", "close", "volume", "amount", "prev_close"}
LABEL_FIELDS = {
    "entry_date", "label_exit_date", "future_open_to_open_return_1d",
    "future_close_to_close_return_1d", "max_feature_date",
}


def _last_rank(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if not len(finite) or not np.isfinite(values[-1]):
        return np.nan
    # Average-rank handling of ties avoids assigning every zero-valued sparse
    # event to the top of its rolling distribution.
    target = values[-1]
    less = np.sum(finite < target)
    equal = np.sum(finite == target)
    return float((less + 0.5 * equal) / len(finite))


def _rolling_rank(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    return series.astype(float).rolling(window, min_periods=min_periods).apply(_last_rank, raw=True)


def _run_length(mask: pd.Series) -> pd.Series:
    arr = mask.fillna(False).to_numpy(dtype=bool)
    out = np.zeros(len(arr), dtype=float)
    count = 0
    for i, flag in enumerate(arr):
        count = count + 1 if flag else 0
        out[i] = count
    return pd.Series(out, index=mask.index)


def _safe_log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.log(a.astype(float).clip(lower=1e-12) / b.astype(float).clip(lower=1e-12))


def _rank_sources(out: pd.DataFrame, names: Iterable[str]) -> None:
    for name in dict.fromkeys(names):
        if name not in out:
            raise KeyError(f"rank source was not constructed: {name}")
        out[f"{name}_rank252"] = _rolling_rank(out[name])


def build_causal_features(raw: pd.DataFrame) -> pd.DataFrame:
    missing = sorted({"date", *RAW_FIELDS} - set(raw.columns))
    if missing:
        raise ValueError(f"missing canonical spot fields: {missing}")
    df = raw.copy().sort_values("date").reset_index(drop=True)
    if df.date.duplicated().any() or not df.date.is_monotonic_increasing:
        raise ValueError("date must be unique and increasing")
    out = df[["date", *sorted(RAW_FIELDS)]].copy()
    o, h, l, c, pc = (df[x].astype(float) for x in ("open", "high", "low", "close", "prev_close"))
    v, amount = df.volume.astype(float), df.amount.astype(float)
    span = (h - l).replace(0, np.nan)

    out["ret_1"] = c / pc - 1
    out["gap"] = o / pc - 1
    out["intraday_ret"] = c / o - 1
    out["range_pct"] = (h - l) / pc
    out["body_pct"] = (c - o) / pc
    out["abs_body_pct"] = out.body_pct.abs()
    out["close_location"] = (c - l) / span
    out["upper_shadow_share"] = (h - np.maximum(o, c)) / span
    out["lower_shadow_share"] = (np.minimum(o, c) - l) / span
    out["true_range_pct"] = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1) / pc

    oo = o.pct_change()
    out["oo_ret_1"] = oo
    out["oo_ret_5"] = (1 + oo).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1
    out["oo_ret_20"] = (1 + oo).rolling(20, min_periods=20).apply(np.prod, raw=True) - 1
    out["oo_vol_5"] = oo.rolling(5, min_periods=5).std(ddof=0)
    out["oo_vol_20"] = oo.rolling(20, min_periods=20).std(ddof=0)
    out["oo_down_share_20"] = (oo < 0).astype(float).rolling(20, min_periods=20).mean()
    out["oo_up_share_20"] = (oo > 0).astype(float).rolling(20, min_periods=20).mean()

    amount_log = np.log1p(amount.clip(lower=0))
    for window in (3, 5, 10, 20, 60, 120):
        out[f"ret_{window}"] = c.pct_change(window)
        out[f"mean_ret_{window}"] = out.ret_1.rolling(window, min_periods=window).mean()
        out[f"vol_{window}"] = out.ret_1.rolling(window, min_periods=window).std(ddof=0)
        out[f"drawdown_{window}"] = c / c.rolling(window, min_periods=window).max() - 1
        rolling_min = c.rolling(window, min_periods=window).min()
        rolling_max = c.rolling(window, min_periods=window).max()
        out[f"range_position_{window}"] = (c - rolling_min) / (rolling_max - rolling_min).replace(0, np.nan)
        out[f"volume_ratio_{window}"] = v / v.rolling(window, min_periods=window).median() - 1
        out[f"amount_ratio_{window}"] = amount / amount.rolling(window, min_periods=window).median() - 1
        denom = amount_log.rolling(window, min_periods=window).std(ddof=0)
        out[f"amount_z_{window}"] = (amount_log - amount_log.rolling(window, min_periods=window).mean()) / denom
        out[f"negative_share_{window}"] = (out.ret_1 < 0).astype(float).rolling(window, min_periods=window).mean()
        path = out.ret_1.abs().rolling(window, min_periods=window).sum()
        out[f"trend_efficiency_{window}"] = out[f"ret_{window}"].abs() / path.replace(0, np.nan)

    out["drawdown_change_5"] = out.drawdown_60 - out.drawdown_60.shift(5)
    out["momentum_curvature"] = out.ret_5 - 0.25 * out.ret_20
    out["vol_term_spread"] = out.vol_5 / out.vol_20.replace(0, np.nan) - 1

    log_hl = _safe_log_ratio(h, l)
    log_co = _safe_log_ratio(c, o)
    log_ho = _safe_log_ratio(h, o)
    log_hc = _safe_log_ratio(h, c)
    log_lo = _safe_log_ratio(l, o)
    log_lc = _safe_log_ratio(l, c)
    log_oc = _safe_log_ratio(o, pc)
    parkinson_var = log_hl.pow(2) / (4 * math.log(2))
    gk_var = (0.5 * log_hl.pow(2) - (2 * math.log(2) - 1) * log_co.pow(2)).clip(lower=0)
    rs_var = (log_ho * log_hc + log_lo * log_lc).clip(lower=0)
    for window in (10, 20):
        out[f"parkinson_vol_{window}"] = parkinson_var.rolling(window, min_periods=window).mean().clip(lower=0).pow(0.5)
        out[f"gk_vol_{window}"] = gk_var.rolling(window, min_periods=window).mean().clip(lower=0).pow(0.5)
        out[f"rs_vol_{window}"] = rs_var.rolling(window, min_periods=window).mean().clip(lower=0).pow(0.5)
        overnight_var = log_oc.rolling(window, min_periods=window).var(ddof=1)
        close_var = log_co.rolling(window, min_periods=window).var(ddof=1)
        k = 0.34 / (1.34 + (window + 1) / max(window - 1, 1))
        yz_var = overnight_var + k * close_var + (1 - k) * rs_var.rolling(window, min_periods=window).mean()
        out[f"yz_vol_{window}"] = yz_var.clip(lower=0).pow(0.5)

    squared = out.ret_1.pow(2)
    out["downside_vol_20"] = out.ret_1.clip(upper=0).pow(2).rolling(20, min_periods=20).mean().pow(0.5)
    out["upside_vol_20"] = out.ret_1.clip(lower=0).pow(2).rolling(20, min_periods=20).mean().pow(0.5)
    out["skew_20"] = out.ret_1.rolling(20, min_periods=20).skew()
    out["kurt_20"] = out.ret_1.rolling(20, min_periods=20).kurt()

    out["down_liquidity_impact"] = (-out.ret_1).clip(lower=0) * (1 + out.amount_z_20.clip(lower=0))
    out["up_liquidity_impact"] = out.ret_1.clip(lower=0) * (1 + out.amount_z_20.clip(lower=0))
    out["loss_cluster_5"] = (-out.ret_5).clip(lower=0) * out.negative_share_10
    out["gain_cluster_5"] = out.ret_5.clip(lower=0) * (1 - out.negative_share_10)
    out["amihud"] = out.ret_1.abs() / (amount / 1e8 + 1e-12)

    out["down_run"] = _run_length(out.ret_1 < 0)
    out["up_run"] = _run_length(out.ret_1 > 0)
    jump_cut = out.ret_1.abs().shift(1).rolling(252, min_periods=60).quantile(0.90)
    out["jump_intensity_20"] = (out.ret_1.abs() / jump_cut.replace(0, np.nan)).clip(upper=10).rolling(20, min_periods=20).mean()
    out["down_jump_share_20"] = ((out.ret_1 < -jump_cut).astype(float)).rolling(20, min_periods=20).mean()
    out["up_jump_share_20"] = ((out.ret_1 > jump_cut).astype(float)).rolling(20, min_periods=20).mean()

    prior_low = l.shift(1).rolling(20, min_periods=20).min()
    prior_high = h.shift(1).rolling(20, min_periods=20).max()
    out["support_break"] = ((prior_low - c) / prior_low).clip(lower=0)
    out["resistance_break"] = ((c - prior_high) / prior_high).clip(lower=0)

    delta = c.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    out["rsi_14"] = 100 - 100 / (1 + rs)

    # Path-dependent kernels. All EWM values at t use values no later than t.
    out["pdv_trend_fast"] = out.ret_1.ewm(halflife=5, adjust=False, min_periods=20).mean()
    out["pdv_trend_slow"] = out.ret_1.ewm(halflife=60, adjust=False, min_periods=120).mean()
    out["pdv_activity_fast"] = squared.ewm(halflife=5, adjust=False, min_periods=20).mean().pow(0.5)
    out["pdv_activity_slow"] = squared.ewm(halflife=60, adjust=False, min_periods=120).mean().pow(0.5)
    negative_shock = (-out.ret_1).clip(lower=0)
    positive_shock = out.ret_1.clip(lower=0)
    out["pdv_negative_shock_fast"] = negative_shock.ewm(halflife=5, adjust=False, min_periods=20).mean()
    out["pdv_negative_shock_slow"] = negative_shock.ewm(halflife=60, adjust=False, min_periods=120).mean()
    out["pdv_positive_shock_fast"] = positive_shock.ewm(halflife=5, adjust=False, min_periods=20).mean()
    out["pdv_positive_shock_slow"] = positive_shock.ewm(halflife=60, adjust=False, min_periods=120).mean()
    for halflife in (3, 5, 10, 20, 60):
        out[f"ewm_ret_hl{halflife}"] = out.ret_1.ewm(halflife=halflife, adjust=False, min_periods=max(10, 2 * halflife)).mean()
        out[f"ewm_absret_hl{halflife}"] = out.ret_1.abs().ewm(halflife=halflife, adjust=False, min_periods=max(10, 2 * halflife)).mean()
        out[f"ewm_sqret_hl{halflife}"] = squared.ewm(halflife=halflife, adjust=False, min_periods=max(10, 2 * halflife)).mean().pow(0.5)

    rank_sources = (
        "ret_1", "ret_3", "ret_5", "ret_20", "gap", "intraday_ret", "body_pct", "abs_body_pct",
        "close_location", "upper_shadow_share", "lower_shadow_share", "true_range_pct",
        "oo_ret_1", "oo_ret_5", "oo_ret_20", "oo_vol_5", "oo_vol_20", "oo_down_share_20", "oo_up_share_20",
        "vol_5", "vol_20", "vol_60", "vol_term_spread", "drawdown_60", "drawdown_change_5",
        "range_position_20", "range_position_60", "trend_efficiency_20", "momentum_curvature",
        "volume_ratio_20", "amount_ratio_5", "amount_ratio_20", "amount_z_20",
        "negative_share_10", "negative_share_20", "down_liquidity_impact", "up_liquidity_impact",
        "loss_cluster_5", "gain_cluster_5", "amihud", "down_run", "up_run", "jump_intensity_20",
        "down_jump_share_20", "up_jump_share_20", "support_break", "resistance_break", "rsi_14",
        "parkinson_vol_10", "parkinson_vol_20", "gk_vol_20", "rs_vol_20", "yz_vol_20",
        "downside_vol_20", "upside_vol_20", "skew_20", "kurt_20",
        "pdv_trend_fast", "pdv_trend_slow", "pdv_activity_fast", "pdv_activity_slow",
        "pdv_negative_shock_fast", "pdv_negative_shock_slow",
        "pdv_positive_shock_fast", "pdv_positive_shock_slow",
        "ewm_ret_hl3", "ewm_ret_hl10", "ewm_ret_hl20", "ewm_absret_hl3", "ewm_absret_hl20",
        "ewm_sqret_hl5", "ewm_sqret_hl60",
    )
    _rank_sources(out, rank_sources)

    out["vol_regime_causal"] = out.vol_20_rank252
    out["trend_regime_causal"] = out.ret_20_rank252
    out["drawdown_regime_causal"] = out.drawdown_60_rank252
    out["risk_expansion_rank"] = (
        out.vol_5_rank252 + out.true_range_pct_rank252 + out.abs_body_pct_rank252
    ) / 3
    out["tail_uncertainty_rank"] = (
        out.vol_5_rank252 + out.true_range_pct_rank252 + out.abs_body_pct_rank252 + out.amihud_rank252
    ) / 4
    out["compression_release_rank"] = (
        (1 - out.vol_20_rank252) + out.vol_5_rank252 + out.true_range_pct_rank252
    ) / 3
    out["gap_intraday_divergence_rank"] = (out.gap_rank252 - out.intraday_ret_rank252).abs()
    transition_raw = (
        (out.vol_regime_causal - out.vol_regime_causal.shift(5)).abs()
        + (out.trend_regime_causal - out.trend_regime_causal.shift(5)).abs()
    )
    out["state_transition_rank"] = _rolling_rank(transition_raw)

    # Forward fields are constructed only after all feature columns exist.
    out["entry_date"] = out.date.shift(-1)
    out["label_exit_date"] = out.date.shift(-2)
    out["future_open_to_open_return_1d"] = o.shift(-2) / o.shift(-1) - 1
    out["future_close_to_close_return_1d"] = c.shift(-1) / c - 1
    out["max_feature_date"] = out.date
    numeric = out.select_dtypes(include=[np.number]).columns
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    _validate_feature_contract(out)
    return out


def _component_column(token: str) -> str:
    return token.split(":", 1)[-1]


def required_component_columns() -> set[str]:
    return {
        _component_column(token)
        for spec in VERSION_SPECS.values()
        for token in (*spec.down_components, *spec.up_components)
    }


def _validate_feature_contract(frame: pd.DataFrame) -> None:
    missing = sorted(required_component_columns() - set(frame.columns))
    if missing:
        raise AssertionError(f"version registry references missing causal features: {missing}")
    forbidden = [c for c in frame.columns if c in {"liquidity_pressure", "risk_strength"}]
    if forbidden:
        raise AssertionError(f"forbidden fields constructed: {forbidden}")
    if not frame.max_feature_date.equals(frame.date):
        raise AssertionError("max_feature_date must equal formation date for every row")
    expected_o2o = frame.open.shift(-2) / frame.open.shift(-1) - 1
    expected_c2c = frame.close.shift(-1) / frame.close - 1
    if not np.allclose(frame.future_open_to_open_return_1d, expected_o2o, equal_nan=True):
        raise AssertionError("O2O alignment failure")
    if not np.allclose(frame.future_close_to_close_return_1d, expected_c2c, equal_nan=True):
        raise AssertionError("C2C alignment failure")


def feature_lineage_table() -> pd.DataFrame:
    rows = []
    for column in sorted(required_component_columns()):
        rows.append({
            "field": column,
            "source": "中证500现货日频 OHLCV/成交额",
            "construction": "仅使用 formation_date 当日及以前的 rolling/ewm/rank/价格路径派生",
            "involves_futures": False,
            "allowed_candidate": True,
        })
    return pd.DataFrame(rows)
