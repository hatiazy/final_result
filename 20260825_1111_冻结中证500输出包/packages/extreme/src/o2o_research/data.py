from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_CANONICAL = ("date", "open", "high", "low", "close", "volume", "amount", "prev_close")
COLUMN_ALIASES = {
    "date": ("date", "trade_date", "trade_dt", "datetime"),
    "open": ("open", "open_price", "s_dq_open"),
    "high": ("high", "high_price", "s_dq_high"),
    "low": ("low", "low_price", "s_dq_low"),
    "close": ("close", "close_price", "s_dq_close"),
    "volume": ("volume", "vol", "s_dq_volume"),
    "amount": ("amount", "total_turnover", "turnover", "s_dq_amount"),
    "prev_close": ("prev_close", "preclose", "pre_close", "previous_close", "s_dq_preclose"),
}
FORBIDDEN_TOKENS = (
    "future", "futures", "open_interest", "openinterest", "basis", "term_structure",
    "carry", "premium_discount", "spot_future", "liquidity_pressure", "risk_strength",
    "升贴水", "期货", "持仓", "基差", "期限结构", "期现价差",
)
REMOTE_SPOT_GLOB = "/home/hzy/cta/IC数据更新*最终固化版/现货最终版/CSI500_SPOT_md_eod_raw*最终版.parquet"
DEFAULT_RESEARCH_INPUT = Path(REMOTE_SPOT_GLOB)
# Keep the input-history lineage used by the frozen V156/V189 scores.  This
# is not the Development start; training/selection periods are defined later
# from 2018 onward.  Extra pre-2007 rows from a live RQ query would alter
# causal rolling features and invalidate the frozen score audit.
FROZEN_INPUT_START = pd.Timestamp("2007-01-15")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_input_path(explicit: str | Path | None = None) -> Path:
    raw = str(explicit) if explicit else str(DEFAULT_RESEARCH_INPUT)
    if not Path(raw).expanduser().is_absolute():
        raise ValueError(f"spot input must be an absolute path: {raw}")
    matches = [Path(p).expanduser().resolve() for p in glob.glob(raw)]
    if not matches and not any(ch in raw for ch in "*?["):
        candidate = Path(raw).expanduser().resolve()
        matches = [candidate] if candidate.exists() else []
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise FileNotFoundError(f"spot input must resolve to exactly one file; pattern={raw!r}, matches={len(unique)}")
    path = unique[0]
    if not path.is_file() or path.suffix.lower() not in {".parquet", ".csv"}:
        raise ValueError(f"spot input must be one parquet/csv file: {path}")
    return path


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _field_map(columns: list[str]) -> dict[str, str]:
    lower = {str(c).strip().lower(): str(c) for c in columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in lower:
            mapping[canonical] = lower[canonical]
            continue
        hits = [lower[a] for a in aliases if a in lower]
        if len(hits) > 1:
            # Identical aliases are still ambiguous because silent precedence can hide stale fields.
            raise ValueError(f"multiple aliases found for {canonical}: {hits}")
        if hits:
            mapping[canonical] = hits[0]
    return mapping


def _date_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(values) or pd.api.types.is_float_dtype(values):
        text = values.round().astype("Int64").astype(str)
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.tz_localize(None).dt.normalize()


def source_whitelist() -> pd.DataFrame:
    rows = [
        ("date", "米筐/公司日频现货交易日", "原始字段", False, True),
        ("open", "中证500现货开盘价", "原始字段", False, True),
        ("high", "中证500现货最高价", "原始字段", False, True),
        ("low", "中证500现货最低价", "原始字段", False, True),
        ("close", "中证500现货收盘价", "原始字段", False, True),
        ("volume", "中证500现货成交量", "原始字段", False, True),
        ("amount", "中证500现货成交额", "原始字段", False, True),
        ("prev_close", "中证500现货前收盘价；缺失时仅用上一实际交易日 close 因果补齐", "原始/因果补齐", False, True),
        ("SEG_ERS_1545", "本轮 V01-V50 未使用", "未进入候选", False, False),
        ("九状态字段", "本轮 V01-V50 未使用", "未进入候选", None, False),
    ]
    return pd.DataFrame(rows, columns=["field", "source", "construction", "involves_futures", "allowed_candidate"])


def load_spot_input(explicit: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = resolve_input_path(explicit)
    raw = _read(path)
    forbidden = sorted(c for c in raw.columns if any(t in str(c).lower() for t in FORBIDDEN_TOKENS))
    if forbidden:
        raise ValueError(f"forbidden or ambiguous fields present in spot input: {forbidden}")
    mapping = _field_map([str(c) for c in raw.columns])
    missing = sorted(set(REQUIRED_CANONICAL) - set(mapping))
    if missing == ["prev_close"]:
        pass
    elif missing:
        raise ValueError(f"missing required spot fields: {missing}")
    out = pd.DataFrame(index=raw.index)
    for canonical in REQUIRED_CANONICAL:
        if canonical in mapping:
            out[canonical] = raw[mapping[canonical]]
    out["date"] = _date_series(out["date"])
    if out.date.isna().any():
        raise ValueError(f"unparseable dates: {int(out.date.isna().sum())}")
    out = out.sort_values("date").reset_index(drop=True)
    raw_rows = int(len(out))
    if out.date.duplicated().any():
        dup = out.loc[out.date.duplicated(keep=False), "date"].dt.strftime("%Y-%m-%d").tolist()[:10]
        raise ValueError(f"duplicate spot dates: {dup}")
    if not out.date.is_monotonic_increasing:
        raise ValueError("spot dates must be strictly increasing")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if "prev_close" not in out:
        out["prev_close"] = out.close.shift(1)
        prev_close_source = "causal_prior_close"
    else:
        out["prev_close"] = pd.to_numeric(out.prev_close, errors="coerce")
        prev_close_source = "input"
    out = out.loc[out["date"] >= FROZEN_INPUT_START].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"spot input has no rows on or after frozen history start {FROZEN_INPUT_START.date()}")
    required_numeric = ["open", "high", "low", "close", "volume", "amount"]
    bad_numeric = {c: int(out[c].isna().sum()) for c in required_numeric if out[c].isna().any()}
    if bad_numeric:
        raise ValueError(f"missing/non-numeric required values: {bad_numeric}")
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("spot prices must be strictly positive")
    if (out[["volume", "amount"]] < 0).any().any():
        raise ValueError("spot volume/amount cannot be negative")
    ohlc_bad = (out.high + 1e-10 < out[["open", "close", "low"]].max(axis=1)) | (
        out.low - 1e-10 > out[["open", "close", "high"]].min(axis=1)
    )
    if ohlc_bad.any():
        raise ValueError(f"OHLC ordering violations: {int(ohlc_bad.sum())}")
    instrument_columns = [c for c in raw.columns if str(c).lower() in {"index_code", "order_book_id", "symbol", "ticker"}]
    instruments = sorted({str(x) for c in instrument_columns for x in raw[c].dropna().unique()})
    if len(instruments) > 1:
        raise ValueError(f"input must contain one spot instrument, found {instruments}")
    audit = {
        "input_path": str(path),
        "input_sha256": _sha256(path),
        "rows": int(len(out)),
        "raw_rows_before_frozen_history_trim": raw_rows,
        "frozen_input_start": FROZEN_INPUT_START.strftime("%Y-%m-%d"),
        "date_start": out.date.min().strftime("%Y-%m-%d"),
        "date_end": out.date.max().strftime("%Y-%m-%d"),
        "field_mapping": mapping,
        "prev_close_source": prev_close_source,
        "instrument_values": instruments,
        "forbidden_fields_used": [],
        "uses_futures": False,
        "uses_1545": False,
        "uses_nine_state": False,
    }
    return out, audit


@dataclass
class FrozenRecord:
    version: str
    side: str
    candidate_id: str
    frozen_at_stage: str
    test_used_for_selection: bool = False


class TestVault:
    """Small guard that makes test access conditional on a concrete frozen record."""

    def __init__(self, frame: pd.DataFrame):
        self.__frame = frame.copy()
        self._opened = False

    def unlock(self, record: FrozenRecord) -> pd.DataFrame:
        if record.frozen_at_stage != "development_validation" or record.test_used_for_selection:
            raise RuntimeError("test cannot be opened before a valid development+validation freeze")
        self._opened = True
        return self.__frame.copy()

    @property
    def opened(self) -> bool:
        return self._opened


class TestVaultFactory:
    """Retain Test outcomes behind a fresh, freeze-gated vault per run.

    Only non-outcome metadata is exposed before a version/side has frozen its
    Development+Validation Top1.  This keeps notebook users from accidentally
    inspecting Test labels through the prepared research object.
    """

    def __init__(self, frame: pd.DataFrame):
        self.__frame = frame.copy()

    def issue(self) -> TestVault:
        return TestVault(self.__frame)

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "rows_including_unlabeled_latest": int(len(self.__frame)),
            "date_start": self.__frame.date.min().strftime("%Y-%m-%d"),
            "date_end": self.__frame.date.max().strftime("%Y-%m-%d"),
        }


def split_research_periods(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, TestVaultFactory, pd.DataFrame]:
    required = {"date", "label_exit_date", "future_open_to_open_return_1d"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"cannot split frame; missing {missing}")
    dev = frame.loc[
        frame.date.between("2018-01-01", "2022-12-31")
        & frame.label_exit_date.le(pd.Timestamp("2022-12-31"))
        & frame.future_open_to_open_return_1d.notna()
    ].copy()
    val = frame.loc[
        frame.date.between("2023-01-01", "2024-12-31")
        & frame.label_exit_date.le(pd.Timestamp("2024-12-31"))
        & frame.future_open_to_open_return_1d.notna()
    ].copy()
    test = frame.loc[frame.date.ge("2025-01-01")].copy()
    latest = frame.tail(10).copy()
    if dev.empty or val.empty or test.empty:
        raise ValueError(f"empty phase: development={len(dev)}, validation={len(val)}, test={len(test)}")
    return dev, val, TestVaultFactory(test), latest
