from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .indicators import (
    bocpd_directional_score,
    chow_break_score,
    directional_run_length,
    kalman_level_innovation,
    kalman_trend,
    one_sided_cusum,
    page_hinkley,
    penalized_last_break,
    rolling_curvature,
    rolling_slope,
    rolling_zscore,
    rsi,
    sign_entropy,
    slope_break,
    stochastic,
    theil_sen_break,
    true_range,
    two_sided_cusum,
)
from .registry import BY_ID, SCORE_VARIANTS


def side_state(side: str) -> int:
    if side == "minus":
        return -1
    if side == "plus":
        return 1
    raise ValueError(f"side must be minus/plus, got {side!r}")


def exit_sign(side: str) -> int:
    return -side_state(side)


def _score_frame(series: Iterable[pd.Series], metadata: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    values = list(series)
    if len(values) != SCORE_VARIANTS or len(metadata) != SCORE_VARIANTS:
        raise AssertionError("每版必须正好包含8个逻辑内部评分变体")
    frame = pd.concat(values, axis=1)
    frame.columns = [f"score_{number:02d}" for number in range(1, SCORE_VARIANTS + 1)]
    frame = frame.replace([np.inf, -np.inf], np.nan).astype(float)
    for number, item in enumerate(metadata, start=1):
        item["score_variant"] = f"score_{number:02d}"
    return frame, metadata


def _meta(parameters: Iterable[Any], name: str) -> list[dict[str, Any]]:
    return [{"method": name, "parameters": value} for value in parameters]


def _known_benefit(panel: pd.DataFrame, side: str) -> pd.Series:
    state = side_state(side)
    improvement = exit_sign(side) * panel["o2o_h1"]
    known = improvement.gt(0).where(panel["base_state"].eq(state)).shift(2)
    return known


def _causal_grouped_rate(
    panel: pd.DataFrame,
    side: str,
    keys: pd.Series,
    prior_strength: float,
) -> pd.Series:
    """Development online rates; post-Development values use frozen Dev counts."""
    state = side_state(side)
    event = (exit_sign(side) * panel["o2o_h1"]).gt(0)
    valid_label = panel["o2o_h1"].notna() & panel["base_state"].eq(state)
    index = panel.index
    counts: dict[Any, int] = {}
    successes: dict[Any, float] = {}
    global_count = 0
    global_success = 0.0
    result = np.full(len(panel), np.nan)
    key_values = keys.to_numpy(object)

    for pos, date in enumerate(index):
        # At close t, the t-2 formation H1 outcome is already known at today's open.
        reveal = pos - 2
        if reveal >= 0 and index[reveal] <= pd.Timestamp("2022-12-31") and valid_label.iloc[reveal]:
            key = key_values[reveal]
            if pd.notna(key):
                outcome = float(event.iloc[reveal])
                counts[key] = counts.get(key, 0) + 1
                successes[key] = successes.get(key, 0.0) + outcome
                global_count += 1
                global_success += outcome
        key = key_values[pos]
        if pd.isna(key):
            continue
        prior_mean = (global_success + 1.0) / (global_count + 2.0)
        result[pos] = (successes.get(key, 0.0) + prior_strength * prior_mean) / (
            counts.get(key, 0) + prior_strength
        )
    return pd.Series(result, index=index)


def _duration_scores(panel: pd.DataFrame, version_id: str, side: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    age = panel["state_age"].astype(float)
    if version_id == "V01":
        shapes = (0.55, 0.72, 0.90, 1.10, 1.30, 1.55, 1.85, 2.20)
        scale = 6.0
        scores = [shape / scale * np.power(age.div(scale).clip(lower=1e-4), shape - 1.0) for shape in shapes]
        return _score_frame(scores, _meta([{"shape": x, "scale": scale} for x in shapes], "Weibull state-age hazard"))

    if version_id == "V02":
        schemes = [
            ((1, 2, 3, 5, 8, 13, 21), 2.0),
            ((1, 3, 5, 8, 13, 21), 4.0),
            ((1, 2, 4, 7, 11, 16), 6.0),
            ((1, 3, 6, 10, 15), 8.0),
            ((1, 2, 3, 4, 5, 8, 12), 10.0),
            ((1, 4, 8, 12, 20), 12.0),
            ((1, 2, 5, 10, 20), 16.0),
            ((1, 5, 10, 15, 25), 20.0),
        ]
        outputs = []
        details = []
        for boundaries, prior in schemes:
            key = pd.cut(age, bins=[0, *boundaries[1:], np.inf], labels=False, right=True)
            outputs.append(_causal_grouped_rate(panel, side, key, prior))
            details.append({"age_boundaries": boundaries, "beta_prior_strength": prior})
        return _score_frame(outputs, _meta(details, "grouped discrete-time empirical hazard"))

    if version_id == "V03":
        axis = panel["rule_axis"].astype(float)
        oriented_margin = exit_sign(side) * axis.sub(0.42 if side == "minus" else 0.58)
        settings = [
            ((0, 2, 4, 7, 12, np.inf), (-np.inf, -0.15, -0.05, 0.05, 0.15, np.inf), 3.0),
            ((0, 3, 6, 10, 16, np.inf), (-np.inf, -0.10, 0.0, 0.10, np.inf), 4.0),
            ((0, 1, 3, 5, 8, 13, np.inf), (-np.inf, -0.20, -0.08, 0.02, 0.12, np.inf), 5.0),
            ((0, 4, 8, 15, np.inf), (-np.inf, -0.12, -0.04, 0.04, 0.12, np.inf), 6.0),
            ((0, 2, 5, 9, 14, np.inf), (-np.inf, -0.18, -0.06, 0.06, 0.18, np.inf), 8.0),
            ((0, 3, 7, 12, 20, np.inf), (-np.inf, -0.08, 0.0, 0.08, np.inf), 10.0),
            ((0, 5, 10, 20, np.inf), (-np.inf, -0.20, -0.10, 0.0, 0.10, 0.20, np.inf), 12.0),
            ((0, 2, 4, 8, 16, np.inf), (-np.inf, -0.05, 0.05, np.inf), 16.0),
        ]
        outputs = []
        details = []
        for age_bins, margin_bins, prior in settings:
            age_key = pd.cut(age, bins=list(age_bins), labels=False, include_lowest=True)
            margin_key = pd.cut(oriented_margin, bins=list(margin_bins), labels=False, include_lowest=True)
            key = age_key.astype("Int64").astype(str) + "|" + margin_key.astype("Int64").astype(str)
            outputs.append(_causal_grouped_rate(panel, side, key, prior))
            details.append({"age_bins": age_bins, "margin_bins": margin_bins, "prior_strength": prior})
        return _score_frame(outputs, _meta(details, "age × rule-margin grouped hazard"))

    if version_id == "V04":
        known = _known_benefit(panel, side)
        matching = panel["base_state"].shift(2).eq(side_state(side))
        settings = ((0.65, 4), (0.75, 5), (0.82, 6), (0.87, 8), (0.90, 10), (0.93, 12), (0.95, 16), (0.97, 20))
        outputs = []
        for decay, cap in settings:
            run = np.zeros(len(panel), dtype=float)
            level = 0.0
            for pos in range(len(panel)):
                if panel["base_state"].iloc[pos] != side_state(side):
                    level = 0.0
                elif matching.iloc[pos] and pd.notna(known.iloc[pos]):
                    level = 0.0 if bool(known.iloc[pos]) else min(float(cap), decay * level + 1.0)
                else:
                    level = min(float(cap), decay * level + 1.0)
                run[pos] = level
            outputs.append(pd.Series(run, index=panel.index))
        return _score_frame(outputs, _meta([{"decay": a, "cap": b} for a, b in settings], "causal survival residual"))

    if version_id == "V05":
        improvement = exit_sign(side) * panel["o2o_h1"]
        revealed = improvement.where(panel["base_state"].eq(side_state(side))).shift(2)
        half_lives = (2, 3, 5, 8, 13, 21, 34, 55)
        outputs = [revealed.ewm(halflife=value, adjust=False, min_periods=3, ignore_na=True).mean() for value in half_lives]
        return _score_frame(outputs, _meta([{"half_life": x} for x in half_lives], "online revealed-exit intensity"))
    raise AssertionError(version_id)


def _rule_scores(panel: pd.DataFrame, version_id: str, side: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    sign = exit_sign(side)
    axis = panel["rule_axis"]
    if version_id == "V06":
        settings = ((1, 1), (2, 1), (3, 1), (5, 1), (8, 1), (2, 3), (3, 5), (5, 8))
        scores = [sign * axis.diff(lag).rolling(smooth, min_periods=smooth).mean() / lag for lag, smooth in settings]
        return _score_frame(scores, _meta([{"lag": x, "smooth": y} for x, y in settings], "rule-axis first difference"))
    if version_id == "V07":
        settings = ((1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 3), (5, 3))
        scores = [sign * axis.diff(lag).diff(lag).rolling(smooth, min_periods=smooth).mean() for lag, smooth in settings]
        return _score_frame(scores, _meta([{"lag": x, "smooth": y} for x, y in settings], "rule-axis second difference"))
    if version_id == "V08":
        windows = (5, 7, 9, 11, 14, 18, 24, 32)
        scores = [sign * rolling_curvature(axis, window) for window in windows]
        return _score_frame(scores, _meta([{"window": x} for x in windows], "rule-axis local curvature"))
    if version_id == "V09":
        spread = sign * (panel["fast_engine"] - panel["slow_engine"])
        settings = ((1, 1), (2, 1), (3, 1), (5, 1), (1, 3), (2, 3), (3, 5), (5, 8))
        scores = [spread.diff(lag).rolling(smooth, min_periods=smooth).mean() + 0.25 * spread for lag, smooth in settings]
        return _score_frame(scores, _meta([{"spread_lag": x, "smooth": y} for x, y in settings], "slow-fast engine crossover"))
    if version_id == "V10":
        threshold = 0.42 if side == "minus" else 0.58
        sources = (
            ("rule_axis", 1.00), ("rule_axis_continuous", 1.00), ("rule_axis_band", 1.00),
            ("fast_engine", 1.00), ("slow_engine", 1.00), ("rule_axis", 0.85),
            ("fast_engine", 0.85), ("slow_engine", 0.85),
        )
        scores = [sign * (panel[column] - threshold) * weight + (1.0 - weight) * sign * panel[column].diff(2) for column, weight in sources]
        return _score_frame(scores, _meta([{"source": x, "level_weight": y, "continue_threshold": threshold} for x, y in sources], "continuation-margin erosion"))
    raise AssertionError(version_id)


def _changepoint_scores(panel: pd.DataFrame, version_id: str, side: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    sign = exit_sign(side)
    oriented_axis = sign * panel["rule_axis"]
    oriented_change = oriented_axis.diff()
    oriented_log_price = sign * np.log(panel["close"])
    if version_id == "V11":
        settings = ((0.00, 20), (0.10, 20), (0.25, 20), (0.50, 20), (0.10, 40), (0.25, 40), (0.50, 40), (0.75, 60))
        return _score_frame([one_sided_cusum(oriented_change, a, b) for a, b in settings], _meta([{"drift": a, "scale_window": b} for a, b in settings], "one-sided Page CUSUM"))
    if version_id == "V12":
        settings = ((0.00, 20), (0.10, 20), (0.25, 20), (0.50, 20), (0.10, 40), (0.25, 40), (0.50, 40), (0.75, 60))
        return _score_frame([two_sided_cusum(oriented_change, a, b) for a, b in settings], _meta([{"drift": a, "scale_window": b} for a, b in settings], "two-sided CUSUM asymmetry"))
    if version_id == "V13":
        settings = ((0.000, 0.90), (0.002, 0.90), (0.005, 0.90), (0.010, 0.90), (0.000, 0.97), (0.002, 0.97), (0.005, 0.97), (0.010, 0.97))
        return _score_frame([page_hinkley(oriented_axis, a, b) for a, b in settings], _meta([{"delta": a, "memory": b} for a, b in settings], "Page-Hinkley"))
    if version_id == "V14":
        settings = ((2, 10, 20), (3, 12, 20), (3, 20, 40), (5, 20, 40), (5, 30, 60), (8, 30, 60), (8, 50, 100), (13, 55, 120))
        scores = []
        for fast, slow, scale in settings:
            difference = oriented_axis.ewm(span=fast, adjust=False).mean() - oriented_axis.ewm(span=slow, adjust=False).mean()
            scores.append(difference.div(oriented_axis.diff().rolling(scale, min_periods=scale // 2).std().replace(0.0, np.nan)))
        return _score_frame(scores, _meta([{"fast_span": a, "slow_span": b, "scale_window": c} for a, b, c in settings], "EWMA standardized innovation"))
    if version_id == "V15":
        settings = ((1/15, 0.5), (1/20, 1.0), (1/30, 1.0), (1/40, 2.0), (1/60, 2.0), (1/80, 4.0), (1/120, 4.0), (1/180, 8.0))
        return _score_frame([bocpd_directional_score(oriented_axis, a, b) for a, b in settings], _meta([{"hazard": a, "prior_strength": b} for a, b in settings], "Bayesian online changepoint run-length posterior"))
    if version_id == "V16":
        settings = ((16, 4, 0.25), (20, 4, 0.50), (24, 5, 0.75), (30, 6, 1.00), (36, 7, 1.25), (42, 8, 1.50), (50, 10, 1.75), (60, 12, 2.00))
        return _score_frame([penalized_last_break(oriented_axis, a, b, c) for a, b, c in settings], _meta([{"window": a, "min_segment": b, "penalty": c} for a, b, c in settings], "penalized last-break cost gain"))
    if version_id == "V17":
        settings = ((16, 4), (20, 4), (24, 5), (30, 6), (36, 7), (42, 8), (50, 10), (60, 12))
        return _score_frame([slope_break(oriented_log_price, a, b) for a, b in settings], _meta([{"window": a, "min_segment": b} for a, b in settings], "binary-segmentation slope break"))
    if version_id == "V18":
        windows = (10, 12, 14, 16, 20, 24, 30, 36)
        return _score_frame([theil_sen_break(oriented_log_price, x) for x in windows], _meta([{"window": x} for x in windows], "Theil-Sen slope break"))
    if version_id == "V19":
        settings = ((16, .40), (20, .40), (24, .40), (30, .40), (20, .50), (30, .50), (40, .50), (50, .60))
        return _score_frame([chow_break_score(oriented_log_price, a, b) for a, b in settings], _meta([{"window": a, "split_share": b} for a, b in settings], "rolling Chow structural break"))
    raise AssertionError(version_id)


def _state_space_scores(panel: pd.DataFrame, version_id: str, side: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    sign = exit_sign(side)
    if version_id == "V20":
        oriented = sign * panel["rule_axis"]
        settings = ((1e-5, .0025), (5e-5, .0025), (1e-4, .0025), (5e-4, .0025), (1e-4, .005), (5e-4, .005), (1e-3, .01), (5e-3, .02))
        return _score_frame([kalman_level_innovation(oriented, a, b) for a, b in settings], _meta([{"process_var": a, "obs_var": b} for a, b in settings], "Kalman local-level innovation"))
    if version_id == "V21":
        oriented = sign * np.log(panel["close"])
        settings = ((1e-6,1e-7,1e-4),(1e-6,1e-6,1e-4),(1e-5,1e-6,1e-4),(1e-5,1e-5,1e-4),(1e-5,1e-6,5e-4),(1e-4,1e-5,5e-4),(1e-4,1e-4,1e-3),(1e-3,1e-4,2e-3))
        return _score_frame([kalman_trend(oriented, a, b, c) for a, b, c in settings], _meta([{"level_var": a, "trend_var": b, "obs_var": c} for a, b, c in settings], "Kalman local-linear trend"))
    raise AssertionError(version_id)


def _path_scores(panel: pd.DataFrame, version_id: str, side: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    sign = exit_sign(side)
    state = side_state(side)
    close = panel["close"].astype(float)
    log_close = np.log(close)
    returns = close.pct_change(fill_method=None)
    oriented_return = sign * returns
    oriented_log = sign * log_close
    if version_id == "V22":
        windows = (5, 8, 10, 13, 20, 30, 40, 60)
        scores = []
        for window in windows:
            low = oriented_log.rolling(window, min_periods=window).min()
            high = oriented_log.rolling(window, min_periods=window).max()
            scores.append(oriented_log.sub(low).div(high.sub(low).replace(0.0, np.nan)))
        return _score_frame(scores, _meta([{"window": x} for x in windows], "rolling-extreme repair"))
    if version_id == "V23":
        windows = (8, 10, 13, 16, 20, 30, 40, 60)
        scores = []
        for window in windows:
            def recovery(raw: np.ndarray) -> float:
                if not np.isfinite(raw).all(): return np.nan
                trough = int(np.argmin(raw))
                span = float(raw.max() - raw.min())
                if trough == len(raw) - 1 or span <= 1e-12: return 0.0
                return float((raw[-1] - raw[trough]) / span * (len(raw) - 1 - trough) / len(raw))
            scores.append(oriented_log.rolling(window, min_periods=window).apply(recovery, raw=True))
        return _score_frame(scores, _meta([{"window": x} for x in windows], "ordered drawdown recovery ratio"))
    if version_id == "V24":
        settings = ((5,1),(8,1),(10,1),(13,1),(20,1),(30,1),(10,2),(20,3))
        atr = true_range(panel).rolling(14, min_periods=7).mean().replace(0.0, np.nan)
        scores=[]
        for lookback, smooth in settings:
            prior_high = panel["high"].shift(2).rolling(lookback, min_periods=lookback).max()
            prior_low = panel["low"].shift(2).rolling(lookback, min_periods=lookback).min()
            if side == "minus":
                breach = prior_low.sub(close.shift(1)).clip(lower=0)
                reentry = close.sub(prior_low).clip(lower=0)
            else:
                breach = close.shift(1).sub(prior_high).clip(lower=0)
                reentry = prior_high.sub(close).clip(lower=0)
            raw = breach.add(reentry).div(atr)
            scores.append(raw.rolling(smooth, min_periods=smooth).mean())
        return _score_frame(scores, _meta([{"lookback": a, "smooth": b} for a,b in settings], "failed prior-range breakout"))
    if version_id == "V25":
        prior_close = close.shift(1)
        gap = panel["open"].div(prior_close).sub(1)
        body = panel["close"].div(panel["open"]).sub(1)
        settings=((.2,.8,1),(.4,.6,1),(.6,.4,1),(.8,.2,1),(.3,.7,2),(.5,.5,2),(.7,.3,3),(.5,.5,5))
        scores=[sign*(a*gap+b*body).rolling(c,min_periods=c).mean() for a,b,c in settings]
        return _score_frame(scores,_meta([{"gap_weight":a,"body_weight":b,"smooth":c} for a,b,c in settings],"gap-body reversal"))
    if version_id == "V26":
        span=panel["high"].sub(panel["low"]).replace(0.0,np.nan)
        lower=pd.concat([panel["open"],panel["close"]],axis=1).min(axis=1).sub(panel["low"]).div(span)
        upper=panel["high"].sub(pd.concat([panel["open"],panel["close"]],axis=1).max(axis=1)).div(span)
        rejection=lower if side=="minus" else upper
        settings=((1,0),(2,0),(3,0),(5,0),(2,1),(3,1),(5,1),(8,2))
        scores=[rejection.rolling(a,min_periods=a).mean()+b*rejection.diff() for a,b in settings]
        return _score_frame(scores,_meta([{"window":a,"change_weight":b} for a,b in settings],"wick rejection"))
    if version_id == "V27":
        span=panel["high"].sub(panel["low"]).replace(0.0,np.nan)
        clv=(panel["close"].sub(panel["low"])).div(span)
        oriented=clv if side=="minus" else 1.0-clv
        settings=((1,0),(2,0),(3,0),(5,0),(8,0),(3,1),(5,1),(8,2))
        scores=[oriented.rolling(a,min_periods=a).mean()+b*oriented.diff() for a,b in settings]
        return _score_frame(scores,_meta([{"window":a,"change_weight":b} for a,b in settings],"close-location repair"))
    if version_id == "V28":
        log_volume=np.log1p(panel["volume"])
        settings=((3,5),(5,8),(5,13),(8,13),(10,20),(13,21),(20,30),(20,60))
        scores=[]
        for price_window,volume_window in settings:
            original_momentum=state*close.pct_change(price_window,fill_method=None)
            volume_decay=-log_volume.diff(volume_window)/volume_window
            scores.append(original_momentum.clip(lower=0)*rolling_zscore(volume_decay,60).clip(lower=0))
        return _score_frame(scores,_meta([{"price_window":a,"volume_window":b} for a,b in settings],"price-volume divergence"))
    if version_id == "V29":
        obv=(np.sign(returns).fillna(0)*panel["volume"]).cumsum()
        windows=(3,5,8,10,13,20,30,40)
        scores=[sign*rolling_slope(obv,w).div(panel["volume"].rolling(w,min_periods=w).mean().replace(0.0,np.nan)) for w in windows]
        return _score_frame(scores,_meta([{"window":x} for x in windows],"OBV slope reversal"))
    if version_id == "V30":
        directed=np.sign(oriented_return).fillna(0)*panel["volume"]
        windows=(3,5,8,10,13,20,30,40)
        scores=[directed.rolling(w,min_periods=w).sum().div(panel["volume"].rolling(w,min_periods=w).sum().replace(0.0,np.nan)) for w in windows]
        return _score_frame(scores,_meta([{"window":x} for x in windows],"signed-volume imbalance"))
    if version_id == "V31":
        settings=((5,1),(7,1),(9,1),(14,1),(14,2),(21,2),(21,3),(28,3))
        scores=[]
        for window,lag in settings:
            value=rsi(close,window)
            change=sign*value.diff(lag)
            extremity=(50-value.shift(lag)).clip(lower=0)/50 if side=="minus" else (value.shift(lag)-50).clip(lower=0)/50
            scores.append(change*extremity)
        return _score_frame(scores,_meta([{"rsi_window":a,"lag":b} for a,b in settings],"RSI failure swing"))
    if version_id == "V32":
        settings=((5,2),(7,2),(9,3),(14,3),(14,5),(21,3),(21,5),(28,5))
        scores=[]
        for window,smooth in settings:
            k,d=stochastic(panel,window,smooth)
            scores.append(sign*(k-d))
        return _score_frame(scores,_meta([{"window":a,"smooth":b} for a,b in settings],"stochastic oscillator reversal"))
    if version_id == "V33":
        settings=((3,8,3),(5,13,3),(5,20,5),(8,21,5),(8,26,9),(12,26,9),(12,35,9),(16,40,12))
        scores=[]
        for fast,slow,signal in settings:
            macd=close.ewm(span=fast,adjust=False).mean()-close.ewm(span=slow,adjust=False).mean()
            histogram=macd-macd.ewm(span=signal,adjust=False).mean()
            scores.append(sign*histogram.diff().div(close))
        return _score_frame(scores,_meta([{"fast":a,"slow":b,"signal":c} for a,b,c in settings],"MACD histogram rollover"))
    if version_id == "V34":
        settings=((10,1.0),(10,1.5),(15,1.5),(20,1.5),(20,2.0),(30,2.0),(40,2.0),(60,2.5))
        scores=[]
        for window,width in settings:
            mean=close.shift(1).rolling(window,min_periods=window).mean()
            std=close.shift(1).rolling(window,min_periods=window).std(ddof=0).replace(0.0,np.nan)
            z=close.sub(mean).div(std)
            prior=z.shift(1)
            if side=="minus": score=(prior.lt(-width).astype(float)*(z+width).clip(lower=0))+0.1*sign*z.diff()
            else: score=(prior.gt(width).astype(float)*(width-z).clip(lower=0))+0.1*sign*z.diff()
            scores.append(score)
        return _score_frame(scores,_meta([{"window":a,"band_width":b} for a,b in settings],"Bollinger re-entry"))
    if version_id == "V35":
        atr=true_range(panel)
        settings=((1,5),(1,10),(2,10),(3,10),(3,14),(5,14),(5,20),(8,20))
        scores=[sign*close.diff(lag).div(atr.rolling(window,min_periods=window).mean().replace(0.0,np.nan)) for lag,window in settings]
        return _score_frame(scores,_meta([{"return_lag":a,"atr_window":b} for a,b in settings],"ATR-normalized reversal"))
    if version_id == "V36":
        atr=true_range(panel)
        settings=((8,8,1.0),(10,10,1.0),(13,10,1.5),(20,14,1.5),(20,14,2.0),(30,20,1.5),(40,20,2.0),(60,30,2.0))
        scores=[]
        for ema_span,atr_window,width in settings:
            center=close.ewm(span=ema_span,adjust=False).mean()
            scale=atr.rolling(atr_window,min_periods=atr_window).mean().replace(0.0,np.nan)
            z=close.sub(center).div(scale)
            prior=z.shift(1)
            if side=="minus": score=prior.lt(-width).astype(float)*(z+width).clip(lower=0)
            else: score=prior.gt(width).astype(float)*(width-z).clip(lower=0)
            scores.append(score)
        return _score_frame(scores,_meta([{"ema_span":a,"atr_window":b,"width":c} for a,b,c in settings],"Keltner channel failure"))
    if version_id == "V37":
        settings=((5,1),(8,1),(10,2),(13,2),(20,3),(30,3),(40,5),(60,5))
        scores=[]
        for window,lag in settings:
            net=close.diff(window).abs()
            path=close.diff().abs().rolling(window,min_periods=window).sum().replace(0.0,np.nan)
            efficiency=net.div(path)
            collapse=efficiency.shift(lag).sub(efficiency)
            scores.append(collapse+0.25*sign*close.pct_change(lag,fill_method=None))
        return _score_frame(scores,_meta([{"window":a,"lag":b} for a,b in settings],"efficiency-ratio collapse"))
    if version_id == "V38":
        windows=(5,7,9,11,14,18,24,32)
        return _score_frame([sign*rolling_curvature(log_close,w) for w in windows],_meta([{"window":w} for w in windows],"log-price path curvature"))
    if version_id == "V39":
        settings=((5,1),(8,1),(10,1),(13,2),(20,2),(30,3),(40,5),(60,5))
        scores=[]
        for window,lag in settings:
            entropy=sign_entropy(returns,window)
            scores.append(entropy+entropy.diff(lag))
        return _score_frame(scores,_meta([{"window":a,"lag":b} for a,b in settings],"return-sign entropy rise"))
    if version_id == "V40":
        original_run=directional_run_length(-oriented_return)
        settings=((1,1),(2,1),(3,1),(5,1),(8,1),(3,2),(5,2),(8,3))
        scores=[]
        for min_run,smooth in settings:
            raw=original_run.shift(1).where(original_run.shift(1).ge(min_run),0.0)*oriented_return.clip(lower=0)
            scores.append(raw.rolling(smooth,min_periods=smooth).mean())
        return _score_frame(scores,_meta([{"minimum_prior_run":a,"smooth":b} for a,b in settings],"directional-run exhaustion"))
    raise AssertionError(version_id)


def build_rule_score_variants(panel: pd.DataFrame, version_id: str, side: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    spec = BY_ID[version_id]
    if spec.group == "duration":
        return _duration_scores(panel, version_id, side)
    if spec.group == "rule_evidence":
        return _rule_scores(panel, version_id, side)
    if spec.group == "changepoint":
        return _changepoint_scores(panel, version_id, side)
    if spec.group == "state_space":
        return _state_space_scores(panel, version_id, side)
    if spec.group in {"spot_path", "spot_volume"}:
        return _path_scores(panel, version_id, side)
    raise ValueError(f"{version_id} is a model version and must use common.models")

