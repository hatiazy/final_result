"""Remote-aligned eight-state and three-state construction from spot only.

The state definitions in this module are the state-definition portion of the
20260813 remote freeze-alignment package. The reference notebooks and result
screenshots are kept under ``reference/remote_freeze_alignment_20260813``;
they are not runtime inputs. At runtime this module reads one raw spot table,
recomputes the frozen eight continuous states, then applies the matching
economic-role three-state machine::

    spot OHLCV/amount
        -> compute_eight_states (frozen recipes)
        -> build_economic_features_eight
        -> assign_eight_base_state (frozen -1/0/1 process)

No precomputed state file, nine-state file, futures/OI/basis field, or remote
comparison baseline is read.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .remote_frozen_state import (
    EIGHT_STATE_COLUMNS,
    FULL_END,
    FULL_START,
    ROLLING_MIN_PERIODS,
    ROLLING_WINDOW,
    assign_eight_base_state,
    build_economic_features_eight,
    compute_eight_states,
    load_frozen_recipes,
)


# Keep the public name used by the existing research panel builder, while
# using the exact Chinese state names from the remote freeze manifest.
EIGHT_STATE_NAMES = tuple(EIGHT_STATE_COLUMNS)

BANNED_TOKENS = (
    "liquidity_pressure",
    "risk_strength",
    "open_interest",
    "basis",
    "term_structure",
    "futures",
    "期货",
    "基差",
    "期限结构",
    "升贴水",
    "期现价差",
    "成交流动性",
)


@dataclass(frozen=True)
class SpotEightStateSpec:
    """Immutable summary of the remote freeze constants."""

    recipe_version: str = "v5.51_IC_eight_spot_state_frozen_recipes"
    full_start: str = "2018-01-01"
    full_end: str = "2024-12-31"
    rolling_window: int = ROLLING_WINDOW
    rolling_min_periods: int = ROLLING_MIN_PERIODS
    development_end: str = "2022-12-31"
    validation_end: str = "2024-12-31"
    state_count: int = 8
    excluded_state: str = "成交流动性状态"


STATE_SPEC = SpotEightStateSpec()
_CONFIG_PATH = Path(__file__).resolve().parent / "remote_frozen_state" / "spot_eight_state_config.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _date_column(frame: pd.DataFrame) -> pd.Series:
    if "date" in frame.columns:
        values = frame["date"]
    elif "formation_date" in frame.columns:
        values = frame["formation_date"]
    elif "trade_date" in frame.columns:
        values = frame["trade_date"]
    elif "trade_dt" in frame.columns:
        values = frame["trade_dt"].astype("string")
    else:
        raise ValueError("spot input must contain date, formation_date, trade_date, or trade_dt")
    text = values.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.to_datetime(values, errors="coerce")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    if parsed.isna().any():
        raise ValueError("spot input contains an unparseable trading date")
    return parsed.dt.normalize()


def _remote_input(spot: pd.DataFrame) -> pd.DataFrame:
    """Normalize the local spot schema to the remote engine's schema."""

    work = spot.copy()
    work["date"] = _date_column(work)
    if "amount" not in work.columns and "total_turnover" in work.columns:
        work["amount"] = work["total_turnover"]
    if "prev_close" not in work.columns and "preclose" in work.columns:
        work["prev_close"] = work["preclose"]
    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    missing = sorted(required.difference(work.columns))
    if missing:
        raise KeyError(f"spot input missing remote eight-state fields: {missing}")
    if work["date"].duplicated().any():
        raise ValueError("spot input contains duplicate trading dates")
    work = work.sort_values("date", kind="stable").reset_index(drop=True)
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    work[numeric] = work[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(work[numeric].to_numpy(dtype=float)).all():
        raise ValueError("spot input contains non-finite OHLCV/amount values")
    if (work[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("spot prices must be strictly positive")
    if (work[["volume", "amount"]] < 0).any().any():
        raise ValueError("spot volume/amount must be non-negative")
    result = work[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
    result = result.rename(columns={"amount": "total_turnover"})
    if "prev_close" in work.columns:
        result["prev_close"] = pd.to_numeric(work["prev_close"], errors="coerce").to_numpy()
    return result


def _state_age(state: pd.Series) -> pd.Series:
    values = pd.to_numeric(state, errors="raise").astype("int8")
    groups = values.ne(values.shift()).cumsum()
    return values.groupby(groups).cumcount().add(1).astype("int16")


def _maxima(frame: pd.DataFrame) -> dict[str, int]:
    development = frame.loc[frame.index <= pd.Timestamp("2022-12-31")]
    maxima: dict[str, int] = {}
    for name in EIGHT_STATE_NAMES:
        value = development[f"band_{name}"].max()
        if not np.isfinite(value):
            value = frame[f"band_{name}"].max()
        if not np.isfinite(value):
            raise ValueError(f"remote state band has no finite development value: {name}")
        maxima[name] = max(1, int(value))
    return maxima


def build_spot_eight_state_panel(
    spot: pd.DataFrame,
    spec: SpotEightStateSpec = STATE_SPEC,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recompute remote-aligned eight states and the matching three-state base.

    Rows before the remote frozen start (2018-01-01) are intentionally omitted
    from the research panel. The raw spot table can still retain them for
    warm-up and for actual trading-calendar alignment.
    """

    if spec != STATE_SPEC:
        raise ValueError("remote frozen state spec is immutable; custom state parameters are not accepted")
    remote_spot = _remote_input(spot)
    values, value_manifest = compute_eight_states(remote_spot)
    if values.empty:
        raise ValueError("remote frozen eight-state engine returned no rows")

    spot_indexed = remote_spot.set_index("date").sort_index()
    state_columns = list(EIGHT_STATE_NAMES)
    feature_frame = values[state_columns].add_prefix("cv_")
    feature_frame = feature_frame.join(values[[f"band_{name}" for name in state_columns]])
    feature_frame = feature_frame.join(
        spot_indexed[["open", "high", "low", "close", "volume", "total_turnover"]],
        how="inner",
    )
    feature_frame = feature_frame.sort_index()
    maxima = _maxima(feature_frame)
    economic = build_economic_features_eight(feature_frame, maxima)
    frozen, pending = assign_eight_base_state(economic)
    state = pd.to_numeric(frozen["four_state"], errors="raise").astype("int8")

    result = values.copy()
    result["direction_score"] = pd.to_numeric(frozen["direction_axis"], errors="coerce")
    result["direction_score_continuous"] = pd.to_numeric(frozen["direction_axis_continuous"], errors="coerce")
    result["direction_score_band"] = pd.to_numeric(frozen["direction_axis_band"], errors="coerce")
    result["slow_engine"] = pd.to_numeric(frozen["slow_engine"], errors="coerce")
    result["fast_engine"] = pd.to_numeric(frozen["fast_engine"], errors="coerce")
    result["risk_pressure"] = pd.to_numeric(frozen["risk_pressure"], errors="coerce")
    result["risk_high_count"] = pd.to_numeric(frozen["risk_high_count"], errors="coerce")
    for column in (
        "downside_route_flag",
        "downside_evidence",
        "downside_continuation",
        "positive_evidence",
        "positive_continuation",
        "rebound_veto",
        "heat_reversal_exit_veto",
        "long_positive_context",
    ):
        result[column] = frozen[column].astype(bool)
    result["raw_three_state"] = pd.to_numeric(frozen["four_state"], errors="raise").astype("int8")
    result["pending_transition"] = pending.astype(bool)
    result["state"] = state
    result["state_age"] = _state_age(state)
    result["state_switch_flag"] = state.ne(state.shift()).astype("int8")
    result["state_label"] = state.map({-1: "down", 0: "neutral", 1: "up"})
    result.insert(0, "formation_date", pd.DatetimeIndex(result.index))
    result = result.reset_index(drop=True)

    counts = {str(key): int(value) for key, value in state.value_counts().sort_index().items()}
    recipe_payload = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    audit = {
        "engine": "remote_frozen_eight_state_v5.51_with_ERS1545_three_state",
        "data_sources": ["spot OHLCV and amount only"],
        "state_names": list(EIGHT_STATE_NAMES),
        "state_spec": {
            **asdict(STATE_SPEC),
            "config_version": recipe_payload.get("config_version"),
            "recipe_count": len(load_frozen_recipes()),
            "excluded_state": recipe_payload.get("excluded_state"),
        },
        "rows": int(len(result)),
        "date_min": str(result["formation_date"].min().date()),
        "date_max": str(result["formation_date"].max().date()),
        "state_counts": counts,
        "state_values": sorted(int(value) for value in state.unique()),
        "eight_state_score_columns": list(EIGHT_STATE_NAMES),
        "continuous_non_spot_inputs": [],
        "forbidden_column_hits": [
            str(column)
            for column in result.columns
            if any(token in str(column).lower() for token in BANNED_TOKENS)
        ],
        "future_values_used_in_state": False,
        "calendar_day_or_bday_used": False,
        "state_generation": [
            "compute_eight_states",
            "build_economic_features_eight",
            "assign_eight_base_state",
        ],
        "remote_recipe_config": "src/zero_transfer/remote_frozen_state/spot_eight_state_config.json",
        "remote_recipe_sha256": _sha256(_CONFIG_PATH),
        "remote_value_manifest": value_manifest,
        "maxima_from_development": maxima,
        "external_state_file_read": False,
        "external_three_state_baseline_read": False,
        "non_spot_inputs_read": [],
    }
    if audit["forbidden_column_hits"]:
        raise AssertionError(f"forbidden field reached remote spot state panel: {audit['forbidden_column_hits']}")
    if not set(audit["state_values"]).issubset({-1, 0, 1}):
        raise AssertionError(f"invalid remote three-state values: {audit['state_values']}")
    if result[state_columns].isna().any().any():
        raise AssertionError("remote eight-state continuous values contain missing rows")
    return result, audit


__all__ = [
    "EIGHT_STATE_NAMES",
    "STATE_SPEC",
    "SpotEightStateSpec",
    "build_spot_eight_state_panel",
]
