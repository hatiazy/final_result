"""Load one raw spot file and construct the spot-native eight-state panel.

The upload contract deliberately has one input path only.  No precomputed
state, 1545 package, nine-state file, or sidecar is accepted.  The eight
continuous state channels and the derived three-state qualification regime are
computed inside the package from OHLCV and amount through the current row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zero_transfer.spot_eight_state import EIGHT_STATE_NAMES, build_spot_eight_state_panel


REQUIRED_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
}
OPTIONAL_COLUMNS = {
    "trade_dt", "index_code", "prev_close",
    # Canonical remote CSI500_SPOT_md_eod_raw_最终版 metadata/aliases.  These
    # are accepted only as raw-file metadata; the state engine never uses them.
    "crncy_code", "preclose", "change", "pctchange", "data_source", "month",
}
ALLOWED_COLUMNS = REQUIRED_COLUMNS | OPTIONAL_COLUMNS | {"date", "total_turnover"}
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
    "state",
    "state_age",
)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    raise ValueError("COMPANY_SPOT_PATH must be a Parquet, CSV, or TSV file")


def _date_column(frame: pd.DataFrame) -> pd.Series:
    if "date" in frame.columns:
        values = frame["date"]
    elif "trade_dt" in frame.columns:
        values = frame["trade_dt"].astype(str)
    else:
        raise ValueError("spot input must contain date or trade_dt")
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        # Eight-digit integer trading dates need an explicit format.
        parsed = pd.to_datetime(values.astype(str), format="%Y%m%d", errors="coerce")
    if parsed.isna().any():
        raise ValueError("spot panel contains an unparseable trading date")
    return parsed.dt.normalize()


def _state_age(state: pd.Series) -> pd.Series:
    values = pd.to_numeric(state, errors="raise").astype("int8")
    groups = values.ne(values.shift()).cumsum()
    return values.groupby(groups).cumcount().add(1).astype("int16")


def _future_values(values: np.ndarray, offsets: list[int]) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for offset in offsets:
        output = np.full(len(values), np.nan, dtype=float)
        if offset < len(values):
            output[:-offset] = values[offset:]
        result[offset] = output
    return result


def load_spot_panel(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read exactly one spot-panel file and construct model inputs.

    The returned ``spot`` contains only OHLCV/amount fields and ``panel`` adds
    the internally computed eight-state scores, qualification regime, and
    causal date-aligned labels.  No filesystem access other than ``path``
    occurs in this function.
    """

    source = Path(path).expanduser()
    if not source.is_absolute():
        raise ValueError(f"COMPANY_SPOT_PATH 必须使用绝对路径：{path}")
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"COMPANY_SPOT_PATH does not exist: {source}")
    raw = _read_table(source).copy()
    raw.columns = [str(column).strip() for column in raw.columns]
    original_columns = raw.columns.tolist()
    banned = [
        column
        for column in raw.columns
        if any(token in column.lower() for token in BANNED_TOKENS)
    ]
    if banned:
        raise ValueError(f"spot panel contains forbidden non-spot columns: {banned}")
    unknown = sorted(set(raw.columns) - ALLOWED_COLUMNS)
    if unknown:
        raise ValueError(f"spot panel contains columns outside the spot-only contract: {unknown}")
    if "amount" not in raw.columns and "total_turnover" in raw.columns:
        raw = raw.rename(columns={"total_turnover": "amount"})
    if "prev_close" not in raw.columns and "preclose" in raw.columns:
        raw = raw.rename(columns={"preclose": "prev_close"})
    date_alias_present = bool({"date", "trade_dt"} & set(raw.columns))
    missing = sorted(REQUIRED_COLUMNS - set(raw.columns))
    if not date_alias_present:
        missing.insert(0, "date (or formation_date/trade_dt)")
    if missing:
        raise ValueError("the single input must contain raw spot OHLCV and amount columns: " + ", ".join(missing))

    frame = raw.copy()
    frame["date"] = _date_column(frame)
    if frame["date"].duplicated().any():
        duplicate_dates = frame.loc[frame["date"].duplicated(keep=False), "date"].dt.strftime("%Y-%m-%d").unique().tolist()
        raise ValueError(f"spot panel has duplicate trading dates: {duplicate_dates[:5]}")
    frame = frame.sort_values("date").reset_index(drop=True)
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[["open", "high", "low", "close", "volume", "amount"]].to_numpy(dtype=float)).all():
        raise ValueError("spot OHLCV/amount contains non-finite values")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("spot prices must be strictly positive")
    full_frame = frame.copy()
    state_frame, state_audit = build_spot_eight_state_panel(frame)
    state_dates = pd.DatetimeIndex(state_frame["formation_date"])
    if not state_dates.isin(pd.DatetimeIndex(full_frame["date"])).all():
        raise AssertionError("state formation dates are not present in the raw spot calendar")
    state_mask = full_frame["date"].isin(state_dates)
    frame = full_frame.loc[state_mask].copy().reset_index(drop=True)
    if len(frame) != len(state_frame):
        raise AssertionError("state/raw spot date alignment changed the row count")
    frame["state"] = state_frame["state"].to_numpy(dtype="int8")
    frame["state_age"] = state_frame["state_age"].to_numpy(dtype="int16")
    for column in state_frame.columns:
        if column != "formation_date":
            frame[column] = state_frame[column].to_numpy()
    if "prev_close" not in frame.columns:
        frame["prev_close"] = frame["close"].shift(1)
    else:
        frame["prev_close"] = pd.to_numeric(frame["prev_close"], errors="coerce")
    if "trade_dt" not in frame.columns:
        frame["trade_dt"] = frame["date"].dt.strftime("%Y%m%d").astype(int)
    if "index_code" not in frame.columns:
        frame["index_code"] = "company_spot"

    full_dates = pd.DatetimeIndex(full_frame["date"])
    positions = full_dates.get_indexer(pd.DatetimeIndex(frame["date"]))
    if (positions < 0).any():
        raise AssertionError("state rows cannot be mapped to the raw spot calendar")
    full_opens = full_frame["open"].to_numpy(dtype=float)
    full_closes = full_frame["close"].to_numpy(dtype=float)
    future_open = _future_values(full_opens, [1, 2, 3, 4])
    future_close = _future_values(full_closes, [1])

    def _at(values: np.ndarray, offset: int) -> np.ndarray:
        output = np.full(len(positions), np.nan, dtype=float)
        valid = positions + offset < len(values)
        output[valid] = values[positions[valid] + offset]
        return output

    def _date_at(offset: int) -> np.ndarray:
        output = np.full(len(positions), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        valid = positions + offset < len(full_dates)
        output[valid] = full_dates.to_numpy()[positions[valid] + offset]
        return output

    dates = pd.DatetimeIndex(frame["date"])
    frame["formation_date"] = dates
    frame["effective_date"] = _date_at(1)
    frame["exit_date"] = _date_at(2)
    frame["h2_exit_date"] = _date_at(3)
    frame["h3_exit_date"] = _date_at(4)
    frame["o2o_h1"] = _at(future_open[2], 0) / _at(future_open[1], 0) - 1.0
    frame["o2o_h2"] = _at(future_open[3], 0) / _at(future_open[1], 0) - 1.0
    frame["o2o_h3"] = _at(future_open[4], 0) / _at(future_open[1], 0) - 1.0
    frame["c2c_h1"] = _at(future_close[1], 0) / frame["close"].to_numpy(dtype=float) - 1.0
    frame["next_state"] = frame["state"].shift(-1)
    frame["next2_state"] = frame["state"].shift(-2)
    frame["next3_state"] = frame["state"].shift(-3)
    # Compatibility aliases for the already generated research code.  These
    # are calculated from the same spot-native state, not read from a frozen
    # or external state source.
    frame["next_frozen_state"] = frame["next_state"]
    frame["next2_frozen_state"] = frame["next2_state"]
    frame["next3_frozen_state"] = frame["next3_state"]

    spot_columns = [
        "index_code", "trade_dt", "date", "open", "high", "low", "close",
        "volume", "prev_close", "amount",
    ]
    panel_columns = [
        "formation_date", *EIGHT_STATE_NAMES,
        *[f"band_{name}" for name in EIGHT_STATE_NAMES],
        "direction_score", "state", "state_label", "state_age", "state_switch_flag",
        "effective_date", "exit_date", "h2_exit_date", "h3_exit_date",
        "open", "high", "low", "close", "volume", "amount", "o2o_h1", "o2o_h2",
        "o2o_h3", "c2c_h1", "next_state", "next2_state", "next3_state",
        "next_frozen_state", "next2_frozen_state", "next3_frozen_state",
    ]
    spot = full_frame[spot_columns].copy().reset_index(drop=True)
    panel = frame[panel_columns].copy().reset_index(drop=True)
    audit = {
        "passed": True,
        "data_source_whitelist": ["COMPANY_SPOT_PATH"],
        "input_file_name": source.name,
        "input_rows": int(len(full_frame)),
        "state_panel_rows": int(len(frame)),
        "date_min": str(full_frame["date"].min().date()),
        "date_max": str(full_frame["date"].max().date()),
        "state_date_min": str(frame["date"].min().date()),
        "state_date_max": str(frame["date"].max().date()),
        "spot_columns_used": spot_columns,
        "raw_columns_seen": original_columns,
        "remote_spot_schema": "CSI500_SPOT_md_eod_raw_最终版",
        "state_engine": state_audit["engine"],
        "eight_state_names": list(EIGHT_STATE_NAMES),
        "eight_state_spec": state_audit["state_spec"],
        "state_values": state_audit["state_values"],
        "state_counts": state_audit["state_counts"],
        "state_calculated_from_spot": True,
        "future_values_used_in_state": bool(state_audit["future_values_used_in_state"]),
        "calendar_day_or_bday_used": bool(state_audit["calendar_day_or_bday_used"]),
        "formation": "t close",
        "effective": "next actual row in the supplied spot file (t+1)",
        "exit": "next actual row in the supplied spot file (t+2)",
        "primary_label": "O2O_H1=open[t+2]/open[t+1]-1",
        "observation_label": "C2C_H1=close[t+1]/close[t]-1",
        "external_state_file_read": False,
        "external_three_state_baseline_read": False,
        "non_spot_inputs_read": [],
        "test_used_for_selection": False,
    }
    return spot, panel, audit


__all__ = ["load_spot_panel", "REQUIRED_COLUMNS", "ALLOWED_COLUMNS"]
