"""Frozen eight-state spot-only reconstruction used by the company package.

The state recipes are generated into ``spot_eight_state_config.json`` when the
upload package is built.  The JSON contains only the frozen feature families,
parameters, transforms and composition rules; it contains no daily values,
scores, thresholds learned from Test, or local paths.

The implementation deliberately accepts one spot OHLCV/turnover table and
reconstructs the eight non-futures states causally.  The liquidity/futures
state is not an input and is not synthesized as a ninth state.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FULL_START = pd.Timestamp("2018-01-01")
FULL_END = pd.Timestamp("2024-12-31")
ROLLING_WINDOW = 504
ROLLING_MIN_PERIODS = 252

POSITION = "位置修复状态"
REVERSAL = "反转压力状态"
TAIL = "尾部压力/波动切换状态"
TREND = "趋势状态"
PATH = "路径脆弱状态"
INTRADAY = "跳空/日内承接状态"
HEAT = "过热状态"
VOLUME_PRICE = "量价状态"

EIGHT_STATE_COLUMNS = [
    POSITION,
    INTRADAY,
    TREND,
    VOLUME_PRICE,
    PATH,
    TAIL,
    REVERSAL,
    HEAT,
]

_CONFIG_NAME = "spot_eight_state_config.json"


def _config_path() -> Path:
    return Path(__file__).with_name(_CONFIG_NAME)


def load_frozen_recipes() -> list[dict[str, Any]]:
    path = _config_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"公司包缺少八状态冻结配置: {path.name}; 请使用完整上传包构建器生成"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    recipes = payload.get("recipes")
    if not isinstance(recipes, list) or len(recipes) != 8:
        raise ValueError("八状态冻结配置必须包含恰好八个状态方案")
    names = [str(item.get("state")) for item in recipes]
    if set(names) != set(EIGHT_STATE_COLUMNS):
        raise ValueError(f"八状态配置不完整: {names}")
    return recipes


def _causal_rolling_percentile(values: pd.Series) -> pd.Series:
    """Rank each value against the previous 504 observations only."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(numeric)
    output = np.full(len(numeric), np.nan, dtype=float)
    history: list[float] = []
    for position, current in enumerate(numeric):
        if valid[position] and len(history) >= ROLLING_MIN_PERIODS:
            output[position] = bisect.bisect_right(history, float(current)) / len(history)
        if position >= ROLLING_WINDOW:
            expired = numeric[position - ROLLING_WINDOW]
            if np.isfinite(expired):
                index = bisect.bisect_left(history, float(expired))
                if index >= len(history) or history[index] != float(expired):
                    raise AssertionError("八状态滚动分位窗口不一致")
                history.pop(index)
        if valid[position]:
            bisect.insort_right(history, float(current))
    return pd.Series(output, index=values.index, dtype=float)


def _anchor_percentile(values: pd.Series, dates: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").reset_index(drop=True)
    date_values = pd.to_datetime(dates).reset_index(drop=True)
    train = numeric.loc[
        date_values.between(start, end) & numeric.notna()
    ].sort_values().to_numpy(dtype=float)
    output = np.full(len(numeric), np.nan, dtype=float)
    current = numeric.to_numpy(dtype=float)
    valid = np.isfinite(current)
    if len(train):
        output[valid] = np.searchsorted(train, current[valid], side="right") / len(train)
    return pd.Series(output, index=values.index, dtype=float)


def _rolling_with_causal_warmup(raw: pd.Series) -> pd.Series:
    values = pd.to_numeric(raw, errors="coerce").reset_index(drop=True)
    output = _causal_rolling_percentile(values)
    missing = np.flatnonzero(output.isna().to_numpy() & values.notna().to_numpy())
    for position in missing:
        history = values.iloc[max(0, position - ROLLING_WINDOW + 1): position + 1].dropna()
        if len(history):
            output.iloc[position] = float(
                (history.le(values.iloc[position]).sum() - 0.5) / len(history)
            )
    return output.clip(0.0, 1.0).set_axis(raw.index)


def _groupby_mean(values: list[pd.Series]) -> pd.Series:
    """Match pandas groupby(mean) accumulation used by the frozen run.

    The reference first writes the representative rows long and then applies
    a grouped mean.  ``DataFrame.mean(axis=1)`` can differ by one ULP from
    that reduction; because the next rolling percentile is rank based, that
    otherwise invisible difference can move a tied observation by one rank.
    Building the small long table and reducing it with ``groupby.mean``
    reproduces that exact accumulation.  This matters because the next
    rolling percentile is rank based and can move a tied observation by one
    rank after a one-ULP change.
    """
    if not values:
        return pd.Series(dtype=float)
    matrix = pd.concat(
        [pd.to_numeric(value, errors="coerce").rename(str(pos)) for pos, value in enumerate(values)],
        axis=1,
    )
    valid = matrix.notna().all(axis=1)
    # This is the same reduction as the reference long-form
    # ``groupby([trade_date, family]).mean()``.
    reduced = matrix.stack(dropna=False).groupby(level=0, sort=False).mean()
    reduced = reduced.reindex(matrix.index)
    reduced.loc[~valid] = np.nan
    return reduced.astype(float)


def _runtime_frame(spot: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume", "total_turnover"}
    missing = required.difference(spot.columns)
    if missing:
        raise ValueError(f"现货输入缺少八状态字段: {sorted(missing)}")
    work = spot.copy()
    if "date" in work.columns:
        work["trade_date"] = pd.to_datetime(work.pop("date"), errors="raise").dt.normalize()
    elif isinstance(work.index, pd.DatetimeIndex):
        work["trade_date"] = pd.DatetimeIndex(work.index).normalize()
    elif "trade_date" in work.columns:
        work["trade_date"] = pd.to_datetime(work["trade_date"], errors="raise").dt.normalize()
    else:
        raise ValueError("现货输入必须有 date/trade_date 或 DatetimeIndex")
    work = work.sort_values("trade_date", kind="stable").drop_duplicates("trade_date", keep="last")
    for column in ["open", "high", "low", "close", "volume", "total_turnover"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if "prev_close" in work.columns:
        previous = pd.to_numeric(work["prev_close"], errors="coerce")
    else:
        previous = work["close"].shift(1)
    work["adjusted_ret"] = work["close"].div(previous.replace(0.0, np.nan)).sub(1.0)
    for column in ["open", "high", "low", "close"]:
        work[f"adjusted_{column}"] = work[column]
    return work.reset_index(drop=True)


def _feature_value_cache(frame: pd.DataFrame, recipes: list[dict[str, Any]], factor: Any) -> dict[tuple, tuple[pd.Series, pd.Series]]:
    cache: dict[tuple, tuple[pd.Series, pd.Series]] = {}
    for recipe in recipes:
        for item in recipe["features"]:
            key = (
                item["family"],
                json.dumps(item["params"], ensure_ascii=False, sort_keys=True),
                item["transform"],
                item.get("window"),
                item.get("min_periods"),
                item.get("lag", 1),
                item.get("clip", 5.0),
                item["direction"],
            )
            if key in cache:
                continue
            params = dict(item["params"])
            raw = factor.BUILDERS[item["family"]](frame, **params)
            raw = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan)
            if item["transform"] == "raw":
                transformed = raw
            elif item["transform"] == "rolling_zscore":
                transformed = factor.rolling_zscore(
                    raw,
                    window=int(item["window"]),
                    min_periods=int(item["min_periods"]),
                    stats_lag=int(item.get("lag", 1)),
                    clip=float(item.get("clip", 5.0)),
                ).replace([np.inf, -np.inf], np.nan)
                # This is the frozen production warm-up rule: after the first
                # valid transformed value, a later zero-variance gap is neutral.
                if transformed.notna().any():
                    first = transformed.first_valid_index()
                    transformed.loc[first:] = transformed.loc[first:].fillna(0.0)
            else:
                raise ValueError(f"不支持的八状态变换: {item['transform']}")
            oriented = transformed * float(item["direction"])
            dates = frame["trade_date"]
            fixed = _anchor_percentile(oriented, dates, FULL_START, FULL_END)
            rolling = _causal_rolling_percentile(oriented.reset_index(drop=True))
            rolling.index = frame.index
            cache[key] = (fixed, rolling)
    return cache


def _compose_state(wide: pd.DataFrame, state: str, variant: str, families: list[str]) -> pd.Series:
    if state == REVERSAL and str(variant).startswith("strict_reversal"):
        background = wide[["distance_from_ma", "momentum"]].mean(axis=1, skipna=False)
        trigger_families = [
            family for family in families
            if family not in {"distance_from_ma", "momentum"}
        ]
        if not trigger_families:
            raise ValueError("严格反转配方缺少衰竭触发 family")
        trigger = wide[trigger_families].apply(
            lambda row: row.nlargest(min(2, len(trigger_families))).mean(),
            axis=1,
        )
        return np.sqrt(
            ((background - 0.50) / 0.50).clip(0.0, 1.0)
            * ((trigger - 0.50) / 0.50).clip(0.0, 1.0)
        )
    if state == VOLUME_PRICE:
        bullish = wide[["price_volume_trend_agreement", "volume_return_confirm"]].mean(
            axis=1, skipna=False
        )
        bearish = wide[["down_volume_pressure"]].mean(axis=1, skipna=False)
        return pd.concat([bullish, bearish], axis=1).mean(axis=1, skipna=False)
    return wide[families].mean(axis=1, skipna=False)


def compute_eight_states(spot: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recompute the frozen eight state values from spot data only."""
    recipes = load_frozen_recipes()
    frame = _runtime_frame(spot)
    # Imported lazily so the research source can be syntax-checked without
    # carrying the company factor component directory in the research tree.
    from .spot_factor_library import rolling_zscore as _unused_rolling_zscore  # noqa: F401
    from . import spot_factor_library as factor

    cache = _feature_value_cache(frame, recipes, factor)
    output: dict[str, pd.Series] = {}
    raw_outputs: dict[str, pd.Series] = {}
    dates = pd.DatetimeIndex(frame["trade_date"])
    keep = dates >= FULL_START
    for recipe in recipes:
        state = str(recipe["state"])
        families = [str(name) for name in recipe["families"]]
        family_values: dict[str, list[pd.Series]] = {name: [] for name in families}
        for item in recipe["features"]:
            key = (
                item["family"],
                json.dumps(item["params"], ensure_ascii=False, sort_keys=True),
                item["transform"],
                item.get("window"),
                item.get("min_periods"),
                item.get("lag", 1),
                item.get("clip", 5.0),
                item["direction"],
            )
            family_values.setdefault(str(item["family"]), []).append(cache[key][1])
        wide = pd.concat(
            {family: _groupby_mean(values) for family, values in family_values.items()},
            axis=1,
        )
        wide.columns = families
        raw_state = _compose_state(wide, state, str(recipe["variant"]), families)
        # The production freeze computes each feature's rolling percentile on
        # the full spot history, but the *state-level* causal warm-up starts
        # at FULL_START (the first date retained in family_daily).  Applying
        # the state warm-up to pre-2018 rows changes the first state values and
        # then shifts every subsequent 504-row window.
        state_raw = raw_state.loc[keep].reset_index(drop=True)
        state_value = _rolling_with_causal_warmup(state_raw)
        state_value.index = dates[keep]
        state_raw.index = dates[keep]
        if state == REVERSAL:
            # Frozen strict-reversal semantics: an inactive raw gate is an
            # exact zero, not the .5 single-observation warm-up percentile.
            state_value = state_value.mask(state_raw.eq(0.0), 0.0)
        raw_outputs[state] = state_raw
        output[state] = state_value

    values = pd.DataFrame(output).sort_index()
    if values.empty or values.isna().any().any():
        missing = int(values.isna().sum().sum())
        raise ValueError(f"八状态重算出现缺失连续值: {missing}")
    bands = pd.DataFrame(index=values.index)
    for state in EIGHT_STATE_COLUMNS:
        if state in {TREND, VOLUME_PRICE, REVERSAL}:
            # v5.51's confirmation/event states use four ordered levels.
            bins, labels = [-np.inf, 0.50, 0.70, 0.85, np.inf], [1, 2, 3, 4]
        elif state in {PATH, TAIL}:
            # Pressure states use the shared low/high/extreme three-level
            # interpretation.
            bins, labels = [-np.inf, 0.60, 0.80, np.inf], [1, 2, 3]
        else:
            bins, labels = [-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf], [1, 2, 3, 4, 5]
        bands[f"band_{state}"] = pd.cut(
            values[state], bins, labels=labels, right=False
        ).astype("int8")
    panel = values.join(bands)
    manifest = {
        "state_count": len(EIGHT_STATE_COLUMNS),
        "state_names": list(EIGHT_STATE_COLUMNS),
        "excluded_state": "成交流动性状态（期货/持仓量相关；不读取、不重算）",
        "recipe_version": "v5.51_IC_eight_spot_state_frozen_recipes",
        "recipe_config": _CONFIG_NAME,
        "rows": int(len(panel)),
        "date_min": str(panel.index.min().date()),
        "date_max": str(panel.index.max().date()),
        "missing_continuous_values": int(values.isna().sum().sum()),
        "continuous_range_ok": bool(((values >= 0.0) & (values <= 1.0)).all().all()),
        "causal_rolling_window": ROLLING_WINDOW,
        "causal_rolling_min_periods": ROLLING_MIN_PERIODS,
        "full_train_anchor": "2018-01-01..2024-12-31",
    }
    if not manifest["continuous_range_ok"]:
        raise AssertionError("八状态连续值超出[0,1]")
    return panel, manifest


def _band_rank(frame: pd.DataFrame, state: str, maxima: dict[str, int]) -> pd.Series:
    return frame[f"band_{state}"].astype(float).sub(1.0).div(max(1, int(maxima[state]) - 1))


def build_economic_features_eight(frame: pd.DataFrame, maxima: dict[str, int]) -> pd.DataFrame:
    """Build the 1545 economic panel with PATH/TAIL as the only risk states."""
    positive = [POSITION, INTRADAY, TREND, VOLUME_PRICE]
    risks = [PATH, TAIL]
    cv = {state: frame[f"cv_{state}"].astype(float) for state in [*positive, *risks, REVERSAL, HEAT]}
    br = {state: _band_rank(frame, state, maxima) for state in [*positive, *risks, REVERSAL, HEAT]}
    trend = cv[TREND]
    volume = cv[VOLUME_PRICE]
    position = cv[POSITION]
    intraday = cv[INTRADAY]
    slow = (trend + volume) / 2.0
    fast = (position + intraday) / 2.0
    dual = (slow + fast) / 2.0
    slow_band = (br[TREND] + br[VOLUME_PRICE]) / 2.0
    fast_band = (br[POSITION] + br[INTRADAY]) / 2.0
    dual_band = (slow_band + fast_band) / 2.0
    risk_pressure = pd.concat([cv[state] for state in risks], axis=1).mean(axis=1)
    risk_band = pd.concat([br[state] for state in risks], axis=1).mean(axis=1)
    output = pd.DataFrame(index=frame.index)
    for state, value in cv.items():
        output[f"cv_{state}"] = value
    for state, value in br.items():
        output[f"br_{state}"] = value
        if state in positive:
            output[f"br_{state}_change_1d"] = value.diff()
    output["repair"] = fast
    output["trend_volume"] = (trend + volume) / 2.0
    output["direction_core"] = dual
    output["core_risk"] = risk_pressure
    output["repair_band"] = fast_band
    output["trend_volume_band"] = slow_band
    output["direction_band"] = dual_band
    output["core_risk_band"] = risk_band
    output["positive_count"] = pd.concat([br[state] for state in positive], axis=1).ge(2 / 3).sum(axis=1)
    output["weak_positive_count"] = pd.concat([br[state] for state in positive], axis=1).le(1 / 3).sum(axis=1)
    output["risk_count"] = pd.concat([br[state] for state in risks], axis=1).ge(2 / 3).sum(axis=1)
    output["heat_reversal"] = (cv[HEAT] + cv[REVERSAL]) / 2.0
    output["tail_liquidity"] = (cv[PATH] + cv[TAIL]) / 2.0
    output["direction_change_1d"] = dual.diff()
    output["direction_change_2d"] = dual.diff(2)
    output["direction_change_3d"] = dual.diff(3)
    output["direction_acceleration_1d"] = output["direction_change_1d"].diff()
    output["repair_change_1d"] = fast.diff()
    output["repair_change_2d"] = fast.diff(2)
    output["trend_volume_change_1d"] = output["trend_volume"].diff()
    output["trend_volume_change_2d"] = output["trend_volume"].diff(2)
    output["trend_band_weak"] = br[TREND].le(1 / 3)
    output["volume_price_band_weak"] = br[VOLUME_PRICE].le(1 / 3)
    output["direction_band_weak_count"] = pd.concat([br[state] for state in positive], axis=1).le(1 / 3).sum(axis=1)
    output["positive_count_change_1d"] = output["positive_count"].diff()
    output["positive_count_change_2d"] = output["positive_count"].diff(2)
    for label, cut in (("35", 0.35), ("45", 0.45)):
        weak = pd.concat([cv[state] for state in positive], axis=1).le(cut).sum(axis=1)
        output[f"positive_low{label}_count"] = weak
        output[f"positive_low{label}_count_change_1d"] = weak.diff()
    output["position_change_1d"] = position.diff()
    output["intraday_change_1d"] = intraday.diff()
    output["trend_change_1d"] = trend.diff()
    output["volume_price_change_1d"] = volume.diff()
    output["risk_change_1d"] = risk_pressure.diff()
    output["risk_change_2d"] = risk_pressure.diff(2)
    output["risk_persistence_2d"] = risk_pressure.rolling(2, min_periods=2).mean()
    output["slow_engine_continuous"] = slow
    output["fast_engine_continuous"] = fast
    output["slow_engine_band"] = slow_band
    output["fast_engine_band"] = fast_band
    output["dual_axis_continuous"] = dual
    output["dual_axis_band"] = dual_band
    output["factor_slow_direction"] = (
        0.45 * pd.to_numeric(frame.get("fp_trend_axis"), errors="coerce")
        + 0.30 * pd.to_numeric(frame.get("fp_robust_axis"), errors="coerce")
        + 0.15 * pd.to_numeric(frame.get("fp_structure_axis"), errors="coerce")
        + 0.10 * pd.to_numeric(frame.get("fp_flow_axis"), errors="coerce")
    )
    output["risk_pressure"] = risk_pressure
    output["risk_pressure_band"] = risk_band
    output["risk_high_count"] = output["risk_count"].astype(float)
    output["direction_low_count"] = pd.concat([cv[state] for state in positive], axis=1).le(1 / 3).sum(axis=1).astype(float)
    output["direction_high_count"] = pd.concat([cv[state] for state in positive], axis=1).ge(2 / 3).sum(axis=1).astype(float)
    output["dual_axis_change_1d"] = dual.diff()
    output["dual_axis_change_2d"] = dual.diff(2)
    output["dual_axis_change_3d"] = dual.diff(3)
    output["dual_axis_acceleration_1d"] = output["dual_axis_change_1d"].diff()
    output["slow_engine_change_1d"] = slow.diff()
    output["fast_engine_change_1d"] = fast.diff()
    output["risk_pressure_change_1d"] = risk_pressure.diff()
    output["risk_pressure_change_2d"] = risk_pressure.diff(2)
    output["repair_change_1d"] = fast.diff()
    output["repair_change_2d"] = fast.diff(2)
    output["rebound_veto"] = (
        (output["repair_change_1d"].gt(0) & output["dual_axis_change_1d"].gt(0))
        | (output["dual_axis_acceleration_1d"].gt(0) & output["dual_axis_change_1d"].gt(0))
    )
    output["heat_reversal_pressure"] = (cv[HEAT] + cv[REVERSAL]) / 2.0
    for column in frame.columns:
        if column.startswith("fp_"):
            output[column] = pd.to_numeric(frame[column], errors="coerce")
    return output


def _ordered_three_state_machine(
    down_entry: pd.Series,
    down_continue: pd.Series,
    strong_repair: pd.Series,
    ordinary_entry: pd.Series,
    ordinary_continue: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Frozen 1545 causal -1/0/1 state process for the baseline specification."""
    index = down_entry.index
    down_in = down_entry.fillna(False).to_numpy(bool)
    down_hold = down_continue.fillna(False).to_numpy(bool)
    repair = strong_repair.fillna(False).to_numpy(bool)
    one_in = ordinary_entry.fillna(False).to_numpy(bool)
    one_hold = ordinary_continue.fillna(False).to_numpy(bool)
    raw_values = np.zeros(len(index), dtype=np.int8)
    raw_values[one_in] = 1
    raw_values[down_in] = -1
    assigned = np.zeros(len(index), dtype=np.int8)
    pending = np.zeros(len(index), dtype=bool)
    if len(index) == 0:
        return (
            pd.Series(dtype="int8", index=index),
            pd.Series(dtype="int8", index=index),
            pd.Series(dtype=bool, index=index),
        )
    current = 0
    target: int | None = None
    bridge_target: int | None = None
    streak = 0
    dwell = 1
    assigned[0] = current
    for position in range(1, len(index)):
        if current == -1:
            if repair[position] or not down_hold[position]:
                if target == 0:
                    streak += 1
                else:
                    target, streak = 0, 1
                pending[position] = True
                if streak >= 2 and dwell >= 5:
                    current, target, streak, dwell = 0, None, 0, 1
                    bridge_target = 1 if one_in[position] else None
                    pending[position] = False
                else:
                    dwell += 1
            else:
                target, streak = None, 0
                dwell += 1
            assigned[position] = current
            continue
        if current == 1 and down_in[position] and dwell >= 5:
            current = 0
            target, streak, dwell = None, 0, 1
            bridge_target = -1
            assigned[position] = current
            continue
        if current == 1:
            if one_hold[position]:
                target, streak = None, 0
                dwell += 1
            else:
                if target == 0:
                    streak += 1
                else:
                    target, streak = 0, 1
                pending[position] = True
                if streak >= 2 and dwell >= 5:
                    current, target, streak, dwell = 0, None, 0, 1
                    bridge_target = -1 if down_in[position] else None
                    pending[position] = False
                else:
                    dwell += 1
            assigned[position] = current
            continue
        if down_in[position]:
            if target == -1:
                streak += 1
            else:
                target, streak = -1, 1
            pending[position] = True
            is_bridge = bridge_target == -1
            if streak >= (1 if is_bridge else 2) and dwell >= (1 if is_bridge else 2):
                current, target, streak, dwell = -1, None, 0, 1
                bridge_target = None
                pending[position] = False
            else:
                dwell += 1
        elif one_in[position]:
            if target == 1:
                streak += 1
            else:
                target, streak = 1, 1
            pending[position] = True
            is_bridge = bridge_target == 1
            if streak >= (1 if is_bridge else 2) and dwell >= (1 if is_bridge else 2):
                current, target, streak, dwell = 1, None, 0, 1
                bridge_target = None
                pending[position] = False
            else:
                dwell += 1
        else:
            target, streak = None, 0
            bridge_target = None
            dwell += 1
        assigned[position] = current
    return (
        pd.Series(assigned, index=index, dtype="int8"),
        pd.Series(raw_values, index=index, dtype="int8"),
        pd.Series(pending, index=index, dtype=bool),
    )


def assign_eight_base_state(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the frozen ERS-1545 baseline without a liquidity/futures state."""
    trend = features[f"cv_{TREND}"]
    volume = features[f"cv_{VOLUME_PRICE}"]
    position = features[f"cv_{POSITION}"]
    intraday = features[f"cv_{INTRADAY}"]
    slow = features["slow_engine_continuous"]
    fast = features["fast_engine_continuous"]
    continuous = 0.40 * trend + 0.30 * volume + 0.15 * position + 0.15 * intraday
    band = 0.40 * features[f"br_{TREND}"] + 0.30 * features[f"br_{VOLUME_PRICE}"] + 0.15 * features[f"br_{POSITION}"] + 0.15 * features[f"br_{INTRADAY}"]
    axis = 0.70 * continuous + 0.30 * band
    axis_change_1d = axis.diff()
    axis_change_2d = axis.diff(2)
    axis_change_3d = axis.diff(3)
    low_count = features["direction_low_count"]
    high_count = features["direction_high_count"]
    exit_veto = (features["heat_reversal_pressure"].ge(2 / 3) & axis_change_1d.lt(0)).fillna(False)
    long_positive_context = (
        (slow.ge(0.58) & fast.ge(0.58) & high_count.ge(2))
        | ((continuous.ge(0.70)) & axis_change_2d.ge(0))
    ).fillna(False)
    route_flag = slow.le(0.30) & fast.le(0.42) & low_count.ge(3) & axis_change_2d.le(0)
    rebound = features["rebound_veto"].fillna(False)
    down_entry = ((route_flag & axis.le(0.30)) & ~long_positive_context & ~rebound).fillna(False)
    down_continue = (axis.le(0.42) & low_count.ge(2) & axis_change_3d.le(0.02) & ~rebound).fillna(False)
    positive_route = trend.ge(0.70) & volume.ge(0.70) & position.ge(0.42) & intraday.ge(0.42)
    ordinary_entry = (positive_route & axis.ge(0.70) & ~exit_veto).fillna(False)
    ordinary_continue = (axis.ge(0.58) & ~((exit_veto) & axis_change_2d.lt(0))).fillna(False)
    strong_repair = (
        (features["repair_change_1d"].gt(0) & axis_change_1d.gt(0))
        | (features["dual_axis_acceleration_1d"].gt(0) & axis_change_1d.gt(0))
    ).fillna(False)
    assigned, raw, pending = _ordered_three_state_machine(
        down_entry, down_continue, strong_repair, ordinary_entry, ordinary_continue
    )
    output = pd.DataFrame(
        {
            "direction_level": assigned.eq(1).astype("int8"),
            "risk_flag": assigned.eq(-1).astype("int8"),
            "four_state": assigned,
            "raw_direction_level": raw.eq(1).astype("int8"),
            "raw_risk_flag": raw.eq(-1).astype("int8"),
            "direction_axis": axis,
            "direction_axis_continuous": continuous,
            "direction_axis_band": band,
            "slow_engine": slow,
            "fast_engine": fast,
            "risk_pressure": features["risk_pressure"],
            "risk_high_count": features["risk_high_count"],
            "downside_route_flag": route_flag.fillna(False),
            "downside_evidence": down_entry,
            "downside_continuation": down_continue,
            "positive_evidence": ordinary_entry,
            "positive_continuation": ordinary_continue,
            "rebound_veto": rebound,
            "heat_reversal_exit_veto": exit_veto,
            "long_positive_context": long_positive_context,
        },
        index=features.index,
    )
    return output, pending


__all__ = [
    "EIGHT_STATE_COLUMNS",
    "FULL_START",
    "compute_eight_states",
    "build_economic_features_eight",
    "assign_eight_base_state",
    "load_frozen_recipes",
]
