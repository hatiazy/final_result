"""Four-signal merge, audit, visualization, and report pipeline.

The two source projects each export an effective-date five-column view.  This
module keeps the source state columns auditable, creates a canonical nine-
column combined view, and calculates event-day, daily-state, segment, and
price-path diagnostics under the event-only policy used by both packages.

Run with the project Anaconda Python environment:

    /Users/hzy/anaconda3/bin/python3 02_HTML生成工具/analysis_pipeline.py

The notebook in the same directory calls :func:`run_all` and displays the
same tables and figures.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

try:
    from scipy.stats import beta as beta_dist
except Exception:  # pragma: no cover - optional in a minimal environment
    beta_dist = None


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "01_报告资料与基础上传包" / "数据源"
TABLE_DIR = ROOT / "01_报告资料与基础上传包" / "统计表"
REPORT_DIR = ROOT / "03_汇报版本" / "20260817_1628_03汇报_基础合并版"
FIG_DIR = REPORT_DIR / "图片"

REV_SIGNAL_PATH = DATA_DIR / "1545_notebook03_signal_rq_20260814.csv"
ZERO_SIGNAL_PATH = DATA_DIR / "长0_event_three_state_signal_rq_20260817.csv"
ZERO_PRICE_PATH = DATA_DIR / "long0_spot_rq_20260817.parquet"
REV_PRICE_PATH = DATA_DIR / "1545_spot_rq_20260814.parquet"

COMBINED_PATH = DATA_DIR / "combined_four_signals_nine_columns.csv"
MERGE_AUDIT_PATH = TABLE_DIR / "merge_audit.json"
MANIFEST_PATH = ROOT / "01_报告资料与基础上传包" / "analysis_manifest.json"

REPORT_FILENAMES = {
    "minus_exit": "负向退出信号详细分析.md",
    "plus_exit": "正向退出信号详细分析.md",
    "minus_entry": "零段向下信号详细分析.md",
    "plus_entry": "零段向上信号详细分析.md",
}

PHASES = {
    "Development": (pd.Timestamp("2018-01-01"), pd.Timestamp("2022-12-31")),
    "Validation": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
    "Test": (pd.Timestamp("2025-01-01"), pd.Timestamp("2100-01-01")),
}

STATE_LABELS = {-1: "-1", 0: "0", 1: "+1"}
SERIES_LABELS = {
    "base": "基础三状态",
    "reversal": "仅退出信号",
    "transfer": "仅零段转移",
    "combined": "四信号合并",
}


@dataclass(frozen=True)
class SignalSpec:
    key: str
    label: str
    source: str
    column: str
    expected_base: int
    final_series: str
    effect_sign: int
    action: str
    model: str


SIGNAL_SPECS = (
    SignalSpec(
        "minus_exit",
        "负向退出（-1→0）",
        "1545反转预测",
        "minus_exit_signal",
        -1,
        "reversal",
        +1,
        "退出负向持仓",
        "V55 / inductive conformal martingale",
    ),
    SignalSpec(
        "plus_exit",
        "正向退出（+1→0）",
        "1545反转预测",
        "plus_exit_signal",
        +1,
        "reversal",
        -1,
        "退出正向持仓",
        "V80 / catch22 dynamics classifier",
    ),
    SignalSpec(
        "minus_entry",
        "零段向下（0→-1）",
        "长0两侧转移预测",
        "minus_entry_signal",
        0,
        "transfer",
        -1,
        "从中性进入负向",
        "V38 / volume breakout",
    ),
    SignalSpec(
        "plus_entry",
        "零段向上（0→+1）",
        "长0两侧转移预测",
        "plus_entry_signal",
        0,
        "transfer",
        +1,
        "从中性进入正向",
        "V57 / recurrence quantification",
    ),
)


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _parse_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    text = series.astype(str)
    compact = text.str.fullmatch(r"\d{8}")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(
            text.loc[compact], format="%Y%m%d", errors="coerce"
        )
    parsed = parsed.dt.normalize()
    if parsed.isna().any():
        raise ValueError("Found unparseable date values")
    return parsed


def load_source_signals() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read both effective-date source exports and verify their common grid."""

    rev = pd.read_csv(REV_SIGNAL_PATH, encoding="utf-8-sig")
    zero = pd.read_csv(ZERO_SIGNAL_PATH, encoding="utf-8-sig")
    rev["date"] = _parse_date(rev["date"])
    zero["date"] = _parse_date(zero["date"])
    rev_cols = [
        "date",
        "three_state",
        "minus_exit_signal",
        "plus_exit_signal",
        "final_three_state",
    ]
    zero_cols = [
        "date",
        "three_state",
        "minus_entry_signal",
        "plus_entry_signal",
        "final_three_state",
    ]
    missing_rev = sorted(set(rev_cols) - set(rev.columns))
    missing_zero = sorted(set(zero_cols) - set(zero.columns))
    if missing_rev or missing_zero:
        raise ValueError(f"Missing source columns: reversal={missing_rev}, zero={missing_zero}")
    rev = rev.loc[:, rev_cols].copy()
    zero = zero.loc[:, zero_cols].copy()
    for frame in (rev, zero):
        for column in frame.columns[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int8")
    rev = rev.sort_values("date").reset_index(drop=True)
    zero = zero.sort_values("date").reset_index(drop=True)
    if rev["date"].duplicated().any() or zero["date"].duplicated().any():
        raise ValueError("Source signal grid contains duplicate dates")
    date_match = rev["date"].equals(zero["date"])
    state_match = rev["three_state"].equals(zero["three_state"])
    audit = {
        "reversal_source": str(REV_SIGNAL_PATH),
        "transfer_source": str(ZERO_SIGNAL_PATH),
        "reversal_rows": int(len(rev)),
        "transfer_rows": int(len(zero)),
        "date_match": bool(date_match),
        "base_three_state_match": bool(state_match),
        "reversal_date_min": rev["date"].min(),
        "reversal_date_max": rev["date"].max(),
        "transfer_date_min": zero["date"].min(),
        "transfer_date_max": zero["date"].max(),
        "reversal_base_counts": rev["three_state"].value_counts().sort_index().to_dict(),
        "transfer_base_counts": zero["three_state"].value_counts().sort_index().to_dict(),
        "reversal_signal_counts": rev[["minus_exit_signal", "plus_exit_signal"]]
        .sum()
        .to_dict(),
        "transfer_signal_counts": zero[["minus_entry_signal", "plus_entry_signal"]]
        .sum()
        .to_dict(),
    }
    if not date_match or not state_match:
        mismatch = rev.loc[
            (rev["date"] != zero["date"])
            | (rev["three_state"] != zero["three_state"])
        ]
        audit["mismatch_rows"] = mismatch.head(20).to_dict(orient="records")
        raise ValueError(f"Source grids do not align; audit={audit}")
    return rev, zero, audit


def apply_combined_state(
    base: pd.Series | np.ndarray,
    minus_exit: pd.Series | np.ndarray,
    plus_exit: pd.Series | np.ndarray,
    minus_entry: pd.Series | np.ndarray,
    plus_entry: pd.Series | np.ndarray,
) -> np.ndarray:
    """Apply all four events only on their own signal day.

    Exit events act only on matching active states; entry events act only on
    base-zero rows.  A simultaneous down/up entry signal remains neutral, as
    in the source transfer package.  The function does not carry a signal
    forward to later dates.
    """

    base_arr = np.asarray(base, dtype=np.int8)
    mx = np.asarray(minus_exit, dtype=bool)
    px = np.asarray(plus_exit, dtype=bool)
    me = np.asarray(minus_entry, dtype=bool)
    pe = np.asarray(plus_entry, dtype=bool)
    if len({len(base_arr), len(mx), len(px), len(me), len(pe)}) != 1:
        raise ValueError("Combined state inputs have different lengths")
    out = base_arr.copy()
    out[(base_arr == -1) & mx] = 0
    out[(base_arr == 1) & px] = 0
    conflict = (base_arr == 0) & me & pe
    out[(base_arr == 0) & me & ~conflict] = -1
    out[(base_arr == 0) & pe & ~conflict] = 1
    return out.astype("int8")


def build_combined_nine_columns(
    rev: pd.DataFrame, zero: pd.DataFrame, audit: dict[str, Any]
) -> pd.DataFrame:
    """Create the canonical nine-column combined output.

    The nine columns are: date, one canonical base state, two exits, the
    reversal-only state, two entries, the transfer-only state, and the final
    four-signal state.  The two source base-state columns are compared in the
    audit and collapsed to one canonical column because they are identical.
    """

    combined = pd.DataFrame(
        {
            "date": rev["date"],
            "three_state": rev["three_state"].astype("int8"),
            "minus_exit_signal": rev["minus_exit_signal"].astype("int8"),
            "plus_exit_signal": rev["plus_exit_signal"].astype("int8"),
            "reversal_final_three_state": rev["final_three_state"].astype("int8"),
            "minus_entry_signal": zero["minus_entry_signal"].astype("int8"),
            "plus_entry_signal": zero["plus_entry_signal"].astype("int8"),
            "transfer_final_three_state": zero["final_three_state"].astype("int8"),
        }
    )
    combined["combined_final_three_state"] = apply_combined_state(
        combined["three_state"],
        combined["minus_exit_signal"],
        combined["plus_exit_signal"],
        combined["minus_entry_signal"],
        combined["plus_entry_signal"],
    )
    columns = [
        "date",
        "three_state",
        "minus_exit_signal",
        "plus_exit_signal",
        "reversal_final_three_state",
        "minus_entry_signal",
        "plus_entry_signal",
        "transfer_final_three_state",
        "combined_final_three_state",
    ]
    combined = combined.loc[:, columns].copy()
    if not combined["date"].is_unique:
        raise AssertionError("Combined date grid is not unique")
    # Source final states should be reproduced exactly by the local formulas.
    expected_reversal = apply_combined_state(
        combined["three_state"],
        combined["minus_exit_signal"],
        combined["plus_exit_signal"],
        np.zeros(len(combined), dtype=np.int8),
        np.zeros(len(combined), dtype=np.int8),
    )
    expected_transfer = apply_combined_state(
        combined["three_state"],
        np.zeros(len(combined), dtype=np.int8),
        np.zeros(len(combined), dtype=np.int8),
        combined["minus_entry_signal"],
        combined["plus_entry_signal"],
    )
    audit.update(
        {
            "combined_rows": int(len(combined)),
            "combined_columns": list(combined.columns),
            "reversal_final_reproduction_match": bool(
                np.array_equal(expected_reversal, combined["reversal_final_three_state"])
            ),
            "transfer_final_reproduction_match": bool(
                np.array_equal(expected_transfer, combined["transfer_final_three_state"])
            ),
            "combined_final_counts": combined["combined_final_three_state"]
            .value_counts()
            .sort_index()
            .to_dict(),
            "entry_conflict_days": int(
                ((combined["minus_entry_signal"] == 1) & (combined["plus_entry_signal"] == 1)).sum()
            ),
        }
    )
    return combined


def load_price_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the full-history raw spot file and compare overlapping prices."""

    raw = pd.read_parquet(ZERO_PRICE_PATH).copy()
    if "date" in raw.columns:
        dates = _parse_date(raw["date"])
    elif "trade_dt" in raw.columns:
        dates = _parse_date(raw["trade_dt"])
    else:
        raise ValueError("Price file must contain date or trade_dt")
    raw["date"] = dates
    rename = {"preclose": "prev_close"}
    raw = raw.rename(columns=rename)
    if "prev_close" not in raw.columns:
        raw["prev_close"] = raw["close"].shift(1)
    keep = [c for c in ["date", "open", "high", "low", "close", "prev_close", "volume", "amount"] if c in raw.columns]
    price = raw.loc[:, keep].copy().sort_values("date").drop_duplicates("date", keep="last")
    for column in keep[1:]:
        price[column] = pd.to_numeric(price[column], errors="coerce")
    if price[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("Price file contains missing OHLC values")
    rev_raw = pd.read_parquet(REV_PRICE_PATH).copy()
    rev_raw["date"] = _parse_date(rev_raw["date"] if "date" in rev_raw else rev_raw["trade_dt"])
    overlap = price.merge(
        rev_raw[["date", "open", "high", "low", "close"]],
        on="date",
        how="inner",
        suffixes=("_full", "_rev"),
    )
    price_match = bool(
        len(overlap) > 0
        and np.isclose(
            overlap[["open_full", "high_full", "low_full", "close_full"]].to_numpy(),
            overlap[["open_rev", "high_rev", "low_rev", "close_rev"]].to_numpy(),
            equal_nan=False,
            rtol=0,
            atol=1e-7,
        ).all()
    )
    audit = {
        "price_source": str(ZERO_PRICE_PATH),
        "price_rows": int(len(price)),
        "price_date_min": price["date"].min(),
        "price_date_max": price["date"].max(),
        "overlap_with_reversal_price_rows": int(len(overlap)),
        "overlap_ohlc_exact_within_1e-7": price_match,
    }
    return price.reset_index(drop=True), audit


def make_analysis_panel(combined: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """Join output-grid states to prices and calculate forward open returns."""

    panel = combined.copy()
    panel = panel.merge(price, on="date", how="left", validate="one_to_one")
    price_idx = price.set_index("date").sort_index()
    for horizon in (1, 2, 3, 5, 10):
        future = price_idx["open"].shift(-horizon)
        panel[f"o2o_h{horizon}"] = panel["date"].map(future).div(panel["open"]).sub(1.0)
    # The raw close-to-close observation is useful for shape context but not
    # used as a replacement for the frozen O2O signal metric.
    panel["c2c_1d"] = panel["close"].div(panel["prev_close"]).sub(1.0)
    panel["price_available"] = panel["open"].notna()
    panel["o2o_h1_available"] = panel["o2o_h1"].notna()
    panel["phase"] = panel["date"].map(phase_for_date)
    return panel


def phase_for_date(value: pd.Timestamp) -> str:
    value = pd.Timestamp(value)
    for phase, (start, end) in PHASES.items():
        if start <= value <= end:
            return phase
    return "Outside"


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (np.nan, np.nan)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_mean_interval(values: Iterable[float], seed: int = 20260817, draws: int = 3000) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed + len(values))
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _bp(value: float | int | None) -> float:
    return float(value) * 10000 if value is not None and np.isfinite(value) else np.nan


def _pct(value: float | int | None) -> float:
    return float(value) * 100 if value is not None and np.isfinite(value) else np.nan


def add_segment_context(panel: pd.DataFrame, state_col: str, prefix: str) -> pd.DataFrame:
    out = panel.copy()
    state = out[state_col].astype(int).to_numpy()
    starts = np.r_[True, state[1:] != state[:-1]]
    seg_id = np.cumsum(starts) - 1
    out[f"{prefix}_segment_id"] = seg_id
    group_sizes = pd.Series(seg_id).value_counts().sort_index()
    out[f"{prefix}_segment_length"] = pd.Series(seg_id).map(group_sizes).to_numpy()
    positions = np.arange(len(out)) - np.r_[0, np.flatnonzero(starts)[1:]][seg_id]
    out[f"{prefix}_position"] = positions + 1
    out[f"{prefix}_position_pct"] = (positions + 1) / out[f"{prefix}_segment_length"].to_numpy()
    return out


def _segment_return(raw_returns: pd.Series) -> float:
    values = pd.to_numeric(raw_returns, errors="coerce").to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        return np.nan
    return float(np.prod(1.0 + values) - 1.0)


def build_segments(panel: pd.DataFrame, state_col: str, series: str) -> pd.DataFrame:
    """Build contiguous state segments and price-path shape fields."""

    values = panel[state_col].astype(int).to_numpy()
    starts = np.r_[True, values[1:] != values[:-1]]
    start_pos = np.flatnonzero(starts)
    end_pos = np.r_[start_pos[1:] - 1, len(values) - 1]
    rows: list[dict[str, Any]] = []
    for sid, (start, end) in enumerate(zip(start_pos, end_pos)):
        state = int(values[start])
        block = panel.iloc[start : end + 1]
        raw_ret = _segment_return(block["o2o_h1"])
        aligned = raw_ret * state if state in (-1, 1) and np.isfinite(raw_ret) else raw_ret
        # For path-shape diagnostics include the first open after the state
        # segment when available.  This is intentionally outside the segment:
        # it captures a reversal that starts immediately after the last state
        # day, matching the existing negative-segment audit convention.
        block_dates = pd.to_datetime(block["date"]).reset_index(drop=True)
        path_block = block
        if end + 1 < len(panel):
            next_row = panel.iloc[[end + 1]]
            if pd.notna(next_row["open"].iloc[0]):
                path_block = pd.concat([block, next_row], ignore_index=True)
        opens = pd.to_numeric(path_block["open"], errors="coerce").to_numpy(dtype=float)
        path_dates = pd.to_datetime(path_block["date"]).reset_index(drop=True)
        path_ok = len(opens) > 0 and np.isfinite(opens).all() and (opens > 0).all()
        if path_ok:
            norm = opens / opens[0] - 1.0
            min_i = int(np.argmin(norm))
            max_i = int(np.argmax(norm))
            min_rel = float(norm[min_i])
            max_rel = float(norm[max_i])
            if state == -1:
                tail = opens[min_i:]
                rebound = float(np.max(tail) / opens[min_i] - 1.0) if len(tail) else np.nan
                reversal_extreme_date = path_dates.iloc[min_i]
                adverse_reversal = rebound
                adverse_reversal_date_2 = _first_threshold_date(
                    path_dates.iloc[min_i:], tail / opens[min_i] - 1.0, 0.02
                )
                adverse_reversal_date_3 = _first_threshold_date(
                    path_dates.iloc[min_i:], tail / opens[min_i] - 1.0, 0.03
                )
            elif state == 1:
                tail = opens[max_i:]
                drawdown = float(np.min(tail) / opens[max_i] - 1.0) if len(tail) else np.nan
                rebound = -drawdown if np.isfinite(drawdown) else np.nan
                reversal_extreme_date = path_dates.iloc[max_i]
                adverse_reversal = rebound
                adverse_reversal_date_2 = _first_threshold_date(
                    path_dates.iloc[max_i:], -(tail / opens[max_i] - 1.0), 0.02
                )
                adverse_reversal_date_3 = _first_threshold_date(
                    path_dates.iloc[max_i:], -(tail / opens[max_i] - 1.0), 0.03
                )
            else:
                rebound = np.nan
                reversal_extreme_date = pd.NaT
                adverse_reversal = np.nan
                adverse_reversal_date_2 = pd.NaT
                adverse_reversal_date_3 = pd.NaT
            min_date = path_dates.iloc[min_i]
            max_date = path_dates.iloc[max_i]
            min_position = min_i + 1
            max_position = max_i + 1
        else:
            min_rel = max_rel = rebound = adverse_reversal = np.nan
            reversal_extreme_date = min_date = max_date = pd.NaT
            adverse_reversal_date_2 = adverse_reversal_date_3 = pd.NaT
            min_position = max_position = np.nan
        rows.append(
            {
                "series": series,
                "state": state,
                "segment_id": sid,
                "start_date": block_dates.iloc[0],
                "end_date": block_dates.iloc[-1],
                "length": int(len(block)),
                "start_pos": int(start),
                "end_pos": int(end),
                "raw_segment_return": raw_ret,
                "aligned_segment_return": aligned,
                "return_available": bool(np.isfinite(raw_ret)),
                "price_path_available": bool(path_ok),
                "min_relative_price": min_rel,
                "max_relative_price": max_rel,
                "min_date": min_date,
                "max_date": max_date,
                "min_position": min_position,
                "max_position": max_position,
                "favorable_extreme_date": reversal_extreme_date,
                "adverse_rebound_from_favorable_extreme": adverse_reversal,
                "adverse_reversal_date_2pct": adverse_reversal_date_2,
                "adverse_reversal_date_3pct": adverse_reversal_date_3,
            }
        )
    return pd.DataFrame(rows)


def _first_threshold_date(dates: pd.Series, values: np.ndarray, threshold: float) -> pd.Timestamp | pd.NaT:
    values = np.asarray(values, dtype=float)
    hit = np.flatnonzero(np.isfinite(values) & (values >= threshold))
    return pd.Timestamp(dates.iloc[int(hit[0])]) if len(hit) else pd.NaT


def summarize_segments(segments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (series, state), group in segments.groupby(["series", "state"], sort=True):
        lengths = group["length"].astype(float)
        active_returns = group.loc[
            group["return_available"] & group["state"].isin([-1, 1]), "aligned_segment_return"
        ].dropna()
        raw_returns = group.loc[
            group["return_available"] & group["state"].isin([-1, 1]), "raw_segment_return"
        ].dropna()
        rows.append(
            {
                "series": series,
                "series_label": SERIES_LABELS.get(series, series),
                "state": int(state),
                "state_label": STATE_LABELS.get(int(state), str(state)),
                "segments": int(len(group)),
                "total_days": int(lengths.sum()),
                "mean_days": float(lengths.mean()),
                "median_days": float(lengths.median()),
                "p25_days": float(lengths.quantile(0.25)),
                "p75_days": float(lengths.quantile(0.75)),
                "max_days": int(lengths.max()),
                "one_day_share_pct": _pct((lengths == 1).mean()),
                "le_3d_share_pct": _pct((lengths <= 3).mean()),
                "return_observations": int(len(active_returns)),
                "mean_aligned_segment_return_bp": _bp(active_returns.mean()),
                "median_aligned_segment_return_bp": _bp(active_returns.median()),
                "p05_aligned_segment_return_bp": _bp(active_returns.quantile(0.05)),
                "p95_aligned_segment_return_bp": _bp(active_returns.quantile(0.95)),
                "segment_win_rate_pct": _pct((active_returns > 0).mean()),
                "adverse_tail_gt_3pct": int((active_returns < -0.03).sum()),
                "adverse_tail_gt_5pct": int((active_returns < -0.05).sum()),
                "raw_adverse_segment_count": int(
                    (raw_returns > 0).sum() if state == -1 else (raw_returns < 0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_daily_states(panel: pd.DataFrame, series_to_state: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for series, state_col in series_to_state.items():
        for state in (-1, 0, 1):
            mask = (panel[state_col] == state) & panel["o2o_h1"].notna()
            raw = panel.loc[mask, "o2o_h1"].dropna()
            aligned = raw * state if state in (-1, 1) else raw
            rows.append(
                {
                    "series": series,
                    "series_label": SERIES_LABELS.get(series, series),
                    "state": state,
                    "state_label": STATE_LABELS[state],
                    "days": int(mask.sum()),
                    "mean_raw_h1_bp": _bp(raw.mean()),
                    "median_raw_h1_bp": _bp(raw.median()),
                    "p05_raw_h1_bp": _bp(raw.quantile(0.05)),
                    "p95_raw_h1_bp": _bp(raw.quantile(0.95)),
                    "mean_directional_h1_bp": _bp(aligned.mean()),
                    "median_directional_h1_bp": _bp(aligned.median()),
                    "directional_win_rate_pct": _pct((aligned > 0).mean()),
                    "directional_loss_gt_3pct": int((aligned < -0.03).sum()),
                }
            )
    return pd.DataFrame(rows)


def signal_run_context(panel: pd.DataFrame, spec: SignalSpec) -> tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    """Return valid-event flags and consecutive signal-run coordinates."""

    mask = panel[spec.column].eq(1) & panel["three_state"].eq(spec.expected_base)
    flags = mask.to_numpy(dtype=bool)
    run_id = np.full(len(flags), -1, dtype=int)
    run_position = np.full(len(flags), np.nan, dtype=float)
    run_length = np.full(len(flags), np.nan, dtype=float)
    if len(flags):
        starts = np.flatnonzero(np.r_[flags[0], flags[1:] & ~flags[:-1]])
        ends = np.flatnonzero(np.r_[flags[:-1] & ~flags[1:], flags[-1]])
        for current_id, (start, end) in enumerate(zip(starts, ends)):
            indices = np.arange(int(start), int(end) + 1)
            length = len(indices)
            run_id[indices] = current_id
            run_position[indices] = np.arange(1, length + 1)
            run_length[indices] = length
    return mask, run_id, run_position, run_length


def event_metrics(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return signal summary, phase summary, event rows, and forward curves."""

    summary_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    event_rows: list[pd.DataFrame] = []
    forward_rows: list[dict[str, Any]] = []
    context = add_segment_context(panel, "three_state", "base")
    for spec in SIGNAL_SPECS:
        eligible = context["three_state"].eq(spec.expected_base)
        valid_signal, signal_run_id, signal_run_position, signal_run_length = signal_run_context(context, spec)
        signal = context[spec.column].eq(1)
        invalid_signal = signal & ~eligible
        observed = valid_signal & context["o2o_h1"].notna()
        baseline = eligible & context["o2o_h1"].notna()
        raw = context.loc[observed, "o2o_h1"].astype(float)
        directional = spec.effect_sign * raw
        baseline_directional = spec.effect_sign * context.loc[baseline, "o2o_h1"].astype(float)
        ci = bootstrap_mean_interval(directional * 10000, seed=20260817 + len(summary_rows))
        win_lo, win_hi = wilson_interval(int((directional > 0).sum()), len(directional))
        row = {
            "signal_key": spec.key,
            "signal_label": spec.label,
            "source": spec.source,
            "model": spec.model,
            "action": spec.action,
            "expected_base_state": spec.expected_base,
            "event_days": int(valid_signal.sum()),
            "invalid_signal_days": int(invalid_signal.sum()),
            "eligible_base_days": int(eligible.sum()),
            "coverage_pct": _pct(valid_signal.sum() / eligible.sum()) if eligible.sum() else np.nan,
            "realized_event_days": int(observed.sum()),
            "raw_o2o_mean_bp": _bp(raw.mean()),
            "raw_o2o_median_bp": _bp(raw.median()),
            "directional_mean_bp": _bp(directional.mean()),
            "directional_median_bp": _bp(directional.median()),
            "directional_p05_bp": _bp(directional.quantile(0.05)),
            "directional_p95_bp": _bp(directional.quantile(0.95)),
            "directional_win_rate_pct": _pct((directional > 0).mean()),
            "directional_win_ci_low_pct": _pct(win_lo),
            "directional_win_ci_high_pct": _pct(win_hi),
            "mean_ci_low_bp": ci[0],
            "mean_ci_high_bp": ci[1],
            "eligible_directional_mean_bp": _bp(baseline_directional.mean()),
            "signal_lift_vs_eligible_bp": _bp(directional.mean() - baseline_directional.mean()),
            "eligible_directional_win_rate_pct": _pct((baseline_directional > 0).mean()),
        }
        summary_rows.append(row)

        for phase in ("Full", "Development", "Validation", "Test"):
            if phase == "Full":
                phase_mask = pd.Series(True, index=context.index)
            else:
                phase_mask = context["phase"].eq(phase)
            e = valid_signal & phase_mask
            o = e & context["o2o_h1"].notna()
            b = eligible & phase_mask & context["o2o_h1"].notna()
            r = context.loc[o, "o2o_h1"].astype(float)
            d = spec.effect_sign * r
            bd = spec.effect_sign * context.loc[b, "o2o_h1"].astype(float)
            phase_rows.append(
                {
                    "signal_key": spec.key,
                    "signal_label": spec.label,
                    "phase": phase,
                    "event_days": int(e.sum()),
                    "realized_event_days": int(o.sum()),
                    "eligible_base_days": int((eligible & phase_mask).sum()),
                    "coverage_pct": _pct(e.sum() / (eligible & phase_mask).sum())
                    if (eligible & phase_mask).sum()
                    else np.nan,
                    "directional_mean_bp": _bp(d.mean()),
                    "directional_median_bp": _bp(d.median()),
                    "directional_win_rate_pct": _pct((d > 0).mean()),
                    "eligible_directional_mean_bp": _bp(bd.mean()),
                    "lift_vs_eligible_bp": _bp(d.mean() - bd.mean()),
                    "eligible_directional_win_rate_pct": _pct((bd > 0).mean()),
                }
            )

        details = context.loc[valid_signal].copy()
        details["signal_key"] = spec.key
        details["signal_label"] = spec.label
        details["expected_base_state"] = spec.expected_base
        details["raw_o2o_h1_bp"] = details["o2o_h1"] * 10000
        details["directional_improvement_bp"] = details["o2o_h1"] * spec.effect_sign * 10000
        details["signal_position_pct"] = details["base_position_pct"] * 100
        details["days_from_segment_start"] = details["base_position"] - 1
        details["days_to_segment_end"] = details["base_segment_length"] - details["base_position"]
        valid_indices = valid_signal.to_numpy(dtype=bool)
        details["signal_run_id"] = signal_run_id[valid_indices]
        details["signal_run_position"] = signal_run_position[valid_indices]
        details["signal_run_length"] = signal_run_length[valid_indices]
        details["price_available"] = details["open"].notna()
        event_rows.append(
            details[
                [
                    "signal_key",
                    "signal_label",
                    "date",
                    "phase",
                    "three_state",
                    "base_segment_id",
                    "base_segment_length",
                    "base_position",
                    "signal_position_pct",
                    "days_from_segment_start",
                    "days_to_segment_end",
                    "signal_run_id",
                    "signal_run_position",
                    "signal_run_length",
                    "open",
                    "close",
                    "o2o_h1",
                    "o2o_h2",
                    "o2o_h3",
                    "o2o_h5",
                    "o2o_h10",
                    "raw_o2o_h1_bp",
                    "directional_improvement_bp",
                    "price_available",
                ]
            ]
        )
        for horizon in (1, 2, 3, 5, 10):
            hmask = valid_signal & context[f"o2o_h{horizon}"].notna()
            values = spec.effect_sign * context.loc[hmask, f"o2o_h{horizon}"].astype(float)
            eligible_values = spec.effect_sign * context.loc[
                eligible & context[f"o2o_h{horizon}"].notna(), f"o2o_h{horizon}"
            ].astype(float)
            forward_rows.append(
                {
                    "signal_key": spec.key,
                    "signal_label": spec.label,
                    "horizon_days": horizon,
                    "event_n": int(len(values)),
                    "event_mean_directional_bp": _bp(values.mean()),
                    "event_median_directional_bp": _bp(values.median()),
                    "event_win_rate_pct": _pct((values > 0).mean()),
                    "event_p25_bp": _bp(values.quantile(0.25)),
                    "event_p75_bp": _bp(values.quantile(0.75)),
                    "eligible_n": int(len(eligible_values)),
                    "eligible_mean_directional_bp": _bp(eligible_values.mean()),
                    "eligible_win_rate_pct": _pct((eligible_values > 0).mean()),
                }
            )
    events = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(phase_rows),
        events,
        pd.DataFrame(forward_rows),
    )


def signal_run_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in SIGNAL_SPECS:
        _, run_ids, _, run_lengths = signal_run_context(panel, spec)
        if len(run_ids) == 0:
            continue
        starts = np.flatnonzero(np.r_[run_ids[0] >= 0, (run_ids[1:] >= 0) & (run_ids[1:] != run_ids[:-1])])
        ends = np.flatnonzero(np.r_[((run_ids[:-1] >= 0) & (run_ids[:-1] != run_ids[1:])), run_ids[-1] >= 0])
        for current_id, (start, end) in enumerate(zip(starts, ends)):
            length = int(run_lengths[int(start)])
            rows.append(
                {
                    "signal_key": spec.key,
                    "signal_label": spec.label,
                    "run_id": int(current_id),
                    "start_date": panel.iloc[int(start)]["date"],
                    "end_date": panel.iloc[int(end)]["date"],
                    "run_length_trading_days": int(length),
                    "phase": panel.iloc[int(start)]["phase"],
                    "raw_h1_mean_bp": _bp(panel.iloc[int(start) : int(end) + 1]["o2o_h1"].mean()),
                    "directional_h1_mean_bp": _bp(
                        spec.effect_sign * panel.iloc[int(start) : int(end) + 1]["o2o_h1"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def event_year_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    """Break event-day effects into calendar years for stability checking."""

    rows: list[dict[str, Any]] = []
    context = add_segment_context(panel, "three_state", "base")
    years = sorted(context["date"].dt.year.dropna().astype(int).unique())
    for spec in SIGNAL_SPECS:
        eligible = context["three_state"].eq(spec.expected_base)
        valid_signal, _, _, _ = signal_run_context(context, spec)
        for year in years:
            year_mask = context["date"].dt.year.eq(year)
            eligible_h1 = eligible & year_mask & context["o2o_h1"].notna()
            observed = valid_signal & year_mask & context["o2o_h1"].notna()
            event_values = spec.effect_sign * context.loc[observed, "o2o_h1"].astype(float) * 10000
            baseline_values = spec.effect_sign * context.loc[eligible_h1, "o2o_h1"].astype(float) * 10000
            rows.append(
                {
                    "signal_key": spec.key,
                    "signal_label": spec.label,
                    "year": int(year),
                    "event_days": int((valid_signal & year_mask).sum()),
                    "realized_event_days": int(observed.sum()),
                    "eligible_base_days": int((eligible & year_mask).sum()),
                    "coverage_pct": _pct((valid_signal & year_mask).sum() / (eligible & year_mask).sum())
                    if (eligible & year_mask).sum()
                    else np.nan,
                    "directional_mean_bp": float(event_values.mean()) if len(event_values) else np.nan,
                    "directional_median_bp": float(event_values.median()) if len(event_values) else np.nan,
                    "directional_p05_bp": float(event_values.quantile(0.05)) if len(event_values) else np.nan,
                    "directional_p95_bp": float(event_values.quantile(0.95)) if len(event_values) else np.nan,
                    "directional_min_bp": float(event_values.min()) if len(event_values) else np.nan,
                    "directional_max_bp": float(event_values.max()) if len(event_values) else np.nan,
                    "directional_win_rate_pct": _pct((event_values > 0).mean()) if len(event_values) else np.nan,
                    "directional_loss_gt_3pct_pct": _pct((event_values < -300).mean()) if len(event_values) else np.nan,
                    "eligible_directional_mean_bp": float(baseline_values.mean()) if len(baseline_values) else np.nan,
                    "eligible_directional_win_rate_pct": _pct((baseline_values > 0).mean()) if len(baseline_values) else np.nan,
                    "lift_vs_eligible_bp": float(event_values.mean() - baseline_values.mean())
                    if len(event_values) and len(baseline_values)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def event_position_metrics(events: pd.DataFrame) -> pd.DataFrame:
    """Compare event outcomes by early, middle, and late segment location."""

    if events.empty:
        return pd.DataFrame()
    work = events.copy()
    work["position_bucket"] = pd.cut(
        work["signal_position_pct"],
        bins=[-np.inf, 33.333, 66.667, np.inf],
        labels=["前段（≤1/3）", "中段（1/3-2/3）", "后段（>2/3）"],
        include_lowest=True,
    )
    rows: list[dict[str, Any]] = []
    for spec in SIGNAL_SPECS:
        sub = work.loc[work.signal_key.eq(spec.key)].copy()
        for bucket in ["前段（≤1/3）", "中段（1/3-2/3）", "后段（>2/3）"]:
            group = sub.loc[sub["position_bucket"].astype(str).eq(bucket)]
            realized = group["directional_improvement_bp"].dropna().astype(float)
            rows.append(
                {
                    "signal_key": spec.key,
                    "signal_label": spec.label,
                    "position_bucket": bucket,
                    "event_days": int(len(group)),
                    "realized_event_days": int(len(realized)),
                    "mean_signal_position_pct": float(group["signal_position_pct"].mean()) if len(group) else np.nan,
                    "mean_days_from_segment_start": float(group["days_from_segment_start"].mean()) if len(group) else np.nan,
                    "mean_days_to_segment_end": float(group["days_to_segment_end"].mean()) if len(group) else np.nan,
                    "directional_mean_bp": float(realized.mean()) if len(realized) else np.nan,
                    "directional_median_bp": float(realized.median()) if len(realized) else np.nan,
                    "directional_p25_bp": float(realized.quantile(0.25)) if len(realized) else np.nan,
                    "directional_p75_bp": float(realized.quantile(0.75)) if len(realized) else np.nan,
                    "directional_win_rate_pct": _pct((realized > 0).mean()) if len(realized) else np.nan,
                    "directional_loss_gt_3pct_pct": _pct((realized < -300).mean()) if len(realized) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def event_cluster_metrics(events: pd.DataFrame) -> pd.DataFrame:
    """Compare first signal days with repeated days and use runs as units."""

    rows: list[dict[str, Any]] = []
    for spec in SIGNAL_SPECS:
        sub = events.loc[events.signal_key.eq(spec.key)].copy()
        observed = sub.loc[sub.directional_improvement_bp.notna()].copy()
        run_table = (
            sub.groupby("signal_run_id", as_index=False)
            .agg(
                run_length_trading_days=("signal_run_length", "first"),
                event_days=("date", "size"),
                run_mean_directional_bp=("directional_improvement_bp", "mean"),
            )
            if len(sub)
            else pd.DataFrame()
        )
        run_means = run_table["run_mean_directional_bp"].dropna().to_numpy(dtype=float) if len(run_table) else np.array([])
        run_ci = bootstrap_mean_interval(run_means, seed=20260901 + len(rows))
        first = observed.loc[observed.signal_run_position.eq(1), "directional_improvement_bp"].astype(float)
        repeat = observed.loc[observed.signal_run_position.gt(1), "directional_improvement_bp"].astype(float)
        rows.append(
            {
                "signal_key": spec.key,
                "signal_label": spec.label,
                "event_days": int(len(sub)),
                "run_count": int(len(run_table)),
                "mean_run_length_days": float(run_table["run_length_trading_days"].mean()) if len(run_table) else np.nan,
                "median_run_length_days": float(run_table["run_length_trading_days"].median()) if len(run_table) else np.nan,
                "max_run_length_days": int(run_table["run_length_trading_days"].max()) if len(run_table) else np.nan,
                "single_day_run_share_pct": _pct((run_table["run_length_trading_days"].eq(1)).mean()) if len(run_table) else np.nan,
                "run_mean_directional_bp": float(run_means.mean()) if len(run_means) else np.nan,
                "run_mean_ci_low_bp": run_ci[0],
                "run_mean_ci_high_bp": run_ci[1],
                "run_positive_share_pct": _pct((run_means > 0).mean()) if len(run_means) else np.nan,
                "first_event_days": int(len(first)),
                "first_event_mean_bp": float(first.mean()) if len(first) else np.nan,
                "first_event_win_rate_pct": _pct((first > 0).mean()) if len(first) else np.nan,
                "repeat_event_days": int(len(repeat)),
                "repeat_event_mean_bp": float(repeat.mean()) if len(repeat) else np.nan,
                "repeat_event_win_rate_pct": _pct((repeat > 0).mean()) if len(repeat) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_shape_audit(panel: pd.DataFrame, segments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add signal capture and path-shape fields to base segments."""

    base_segments = segments.loc[segments["series"].eq("base")].copy()
    event_map = {
        -1: ["minus_exit_signal"],
        0: ["minus_entry_signal", "plus_entry_signal"],
        1: ["plus_exit_signal"],
    }
    rows: list[dict[str, Any]] = []
    date_to_position = pd.Series(np.arange(len(panel), dtype=int), index=panel["date"])

    def trading_position(value: Any) -> int | None:
        if pd.isna(value):
            return None
        found = date_to_position.get(pd.Timestamp(value))
        return int(found) if found is not None and not pd.isna(found) else None

    for _, seg in base_segments.iterrows():
        block = panel.iloc[int(seg.start_pos) : int(seg.end_pos) + 1]
        cols = event_map[int(seg.state)]
        signal_mask = block[cols].sum(axis=1).gt(0)
        signal_dates = block.loc[signal_mask, "date"]
        if int(seg.state) == -1:
            threshold_date = seg.adverse_reversal_date_3pct
        elif int(seg.state) == 1:
            threshold_date = seg.adverse_reversal_date_3pct
        else:
            threshold_date = pd.NaT
        valid_threshold = pd.notna(threshold_date)
        captured = bool(
            valid_threshold
            and len(signal_dates)
            and signal_dates.min() <= pd.Timestamp(threshold_date)
        )
        first_signal = signal_dates.min() if len(signal_dates) else pd.NaT
        last_signal = signal_dates.max() if len(signal_dates) else pd.NaT
        first_signal_pos = (
            int((block["date"] <= first_signal).sum()) if pd.notna(first_signal) else np.nan
        )
        threshold_pos = (
            int((block["date"] <= threshold_date).sum()) if valid_threshold else np.nan
        )
        first_signal_idx = trading_position(first_signal)
        threshold_idx = trading_position(threshold_date)
        extreme_idx = trading_position(seg.favorable_extreme_date)
        end_idx = trading_position(seg.end_date)
        reversal_duration_trading_days = (
            threshold_idx - extreme_idx
            if threshold_idx is not None and extreme_idx is not None and threshold_idx >= extreme_idx
            else np.nan
        )
        signal_to_reversal_trading_days = (
            threshold_idx - first_signal_idx
            if captured and threshold_idx is not None and first_signal_idx is not None
            else np.nan
        )
        shape = "未形成明显路径反转"
        if int(seg.state) in (-1, 1):
            if np.isfinite(seg.adverse_rebound_from_favorable_extreme):
                r = float(seg.adverse_rebound_from_favorable_extreme)
                if r >= 0.05:
                    shape = "强路径反转（≥5%）"
                elif r >= 0.03:
                    shape = "明显路径反转（≥3%）"
                elif r >= 0.02:
                    shape = "中等路径反转（≥2%）"
                elif r > 0:
                    shape = "轻微修复"
        rows.append(
            {
                "state": int(seg.state),
                "start_date": seg.start_date,
                "end_date": seg.end_date,
                "length": int(seg.length),
                "raw_segment_return": seg.raw_segment_return,
                "aligned_segment_return": seg.aligned_segment_return,
                "adverse_rebound_pct": _pct(seg.adverse_rebound_from_favorable_extreme),
                "favorable_extreme_date": seg.favorable_extreme_date,
                "reversal_date_3pct": threshold_date,
                "shape_class": shape,
                "signal_count_in_segment": int(signal_mask.sum()),
                "first_signal_date": first_signal,
                "last_signal_date": last_signal,
                "first_signal_position": first_signal_pos,
                "reversal_position": threshold_pos,
                "signal_before_or_on_reversal": captured,
                "signal_lead_trading_days": (
                    signal_to_reversal_trading_days
                ),
                "signal_lead_calendar_days": (
                    int((pd.Timestamp(threshold_date) - pd.Timestamp(first_signal)).days)
                    if captured and pd.notna(first_signal) and valid_threshold
                    else np.nan
                ),
                "reversal_duration_trading_days_3pct": reversal_duration_trading_days,
                "days_after_first_signal_to_segment_end": (
                    int((pd.Timestamp(seg.end_date) - pd.Timestamp(first_signal)).days)
                    if pd.notna(first_signal)
                    else np.nan
                ),
                "trading_days_after_first_signal_to_segment_end": (
                    end_idx - first_signal_idx
                    if end_idx is not None and first_signal_idx is not None
                    else np.nan
                ),
            }
        )
    shape_df = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for state, group in shape_df.groupby("state", sort=True):
        with_path = group["reversal_date_3pct"].notna()
        captured = group.loc[with_path, "signal_before_or_on_reversal"]
        reversal_duration = group["reversal_duration_trading_days_3pct"].dropna()
        signal_lead = group["signal_lead_trading_days"].dropna()
        after_signal = group["trading_days_after_first_signal_to_segment_end"].dropna()
        summary_rows.append(
            {
                "state": int(state),
                "segments": int(len(group)),
                "mean_length": float(group["length"].mean()),
                "median_length": float(group["length"].median()),
                "net_adverse_segments": int((group["aligned_segment_return"] < 0).sum())
                if int(state) in (-1, 1)
                else np.nan,
                "path_reversal_ge_2pct": int((group["adverse_rebound_pct"] >= 2).sum()),
                "path_reversal_ge_3pct": int((group["adverse_rebound_pct"] >= 3).sum()),
                "path_reversal_ge_5pct": int((group["adverse_rebound_pct"] >= 5).sum()),
                "path_reversal_ge_3pct_share_pct": _pct((group["adverse_rebound_pct"] >= 3).mean()),
                "segments_with_signal": int((group["signal_count_in_segment"] > 0).sum()),
                "segments_with_3pct_reversal": int(with_path.sum()),
                "reversal_segments_captured": int(captured.sum()) if len(captured) else 0,
                "capture_rate_among_3pct_reversals_pct": _pct(captured.mean()) if len(captured) else np.nan,
                "mean_reversal_duration_trading_days_3pct": float(reversal_duration.mean()) if len(reversal_duration) else np.nan,
                "median_reversal_duration_trading_days_3pct": float(reversal_duration.median()) if len(reversal_duration) else np.nan,
                "mean_signal_lead_trading_days": float(signal_lead.mean()) if len(signal_lead) else np.nan,
                "median_signal_lead_trading_days": float(signal_lead.median()) if len(signal_lead) else np.nan,
                "mean_trading_days_after_first_signal_to_segment_end": float(after_signal.mean()) if len(after_signal) else np.nan,
                "mean_signal_position_pct": float(
                    group.loc[group["signal_count_in_segment"] > 0, "first_signal_position"].div(
                        group.loc[group["signal_count_in_segment"] > 0, "length"]
                    ).mean()
                    * 100
                )
                if (group["signal_count_in_segment"] > 0).any()
                else np.nan,
            }
        )
    return shape_df, pd.DataFrame(summary_rows)


def make_relevant_comparison(
    summaries: pd.DataFrame, daily: pd.DataFrame
) -> pd.DataFrame:
    """Compare base and each event-only series by state."""

    rows: list[dict[str, Any]] = []
    for state in (-1, 0, 1):
        base_seg = summaries.loc[(summaries.series == "base") & (summaries.state == state)].iloc[0]
        for series in ("reversal", "transfer", "combined"):
            cur = summaries.loc[(summaries.series == series) & (summaries.state == state)].iloc[0]
            rows.append(
                {
                    "state": state,
                    "state_label": STATE_LABELS[state],
                    "comparison": f"{SERIES_LABELS[series]} vs 基础",
                    "series": series,
                    "base_segments": base_seg.segments,
                    "new_segments": cur.segments,
                    "segment_count_delta": int(cur.segments - base_seg.segments),
                    "base_total_days": base_seg.total_days,
                    "new_total_days": cur.total_days,
                    "total_days_delta": int(cur.total_days - base_seg.total_days),
                    "base_mean_days": base_seg.mean_days,
                    "new_mean_days": cur.mean_days,
                    "mean_days_delta": cur.mean_days - base_seg.mean_days,
                    "base_one_day_share_pct": base_seg.one_day_share_pct,
                    "new_one_day_share_pct": cur.one_day_share_pct,
                    "one_day_share_delta_pct": cur.one_day_share_pct - base_seg.one_day_share_pct,
                    "base_mean_aligned_return_bp": base_seg.mean_aligned_segment_return_bp,
                    "new_mean_aligned_return_bp": cur.mean_aligned_segment_return_bp,
                    "mean_aligned_return_delta_bp": cur.mean_aligned_segment_return_bp
                    - base_seg.mean_aligned_segment_return_bp,
                    "base_win_rate_pct": base_seg.segment_win_rate_pct,
                    "new_win_rate_pct": cur.segment_win_rate_pct,
                    "win_rate_delta_pct": cur.segment_win_rate_pct - base_seg.segment_win_rate_pct,
                    "base_adverse_tail_gt_3pct": base_seg.adverse_tail_gt_3pct,
                    "new_adverse_tail_gt_3pct": cur.adverse_tail_gt_3pct,
                    "adverse_tail_delta": cur.adverse_tail_gt_3pct - base_seg.adverse_tail_gt_3pct,
                }
            )
    return pd.DataFrame(rows)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )


COLORS = {
    "minus_exit": "#b2182b",
    "plus_exit": "#2166ac",
    "minus_entry": "#ef8a62",
    "plus_entry": "#67a9cf",
    "base": "#666666",
    "reversal": "#b2182b",
    "transfer": "#2166ac",
    "combined": "#4d9221",
    "state_-1": "#d73027",
    "state_0": "#bdbdbd",
    "state_1": "#4575b4",
}


def save_fig(fig: plt.Figure, name: str) -> str:
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(ROOT))


def plot_signal_timeline(panel: pd.DataFrame, spec: SignalSpec) -> str:
    event = panel.loc[panel[spec.column].eq(1) & panel["three_state"].eq(spec.expected_base)]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]})
    ax = axes[0]
    price = panel.loc[panel["close"].notna()].copy()
    if len(price):
        ax.plot(price["date"], price["close"] / price["close"].iloc[0], color="#333333", lw=1.2, label="CSI500 close / first close")
    for state in (-1, 0, 1):
        mask = panel["three_state"].eq(state) & panel["close"].notna()
        if mask.any():
            ax.scatter(panel.loc[mask, "date"], panel.loc[mask, "close"] / price["close"].iloc[0], s=4, alpha=0.20, color=COLORS[f"state_{state}"], label=f"base state {state}")
    if len(event):
        ev = event.loc[event["close"].notna()]
        ax.scatter(ev["date"], ev["close"] / price["close"].iloc[0], s=26, color=COLORS[spec.key], edgecolor="white", linewidth=0.35, label=f"{spec.label} event")
    ax.set_ylabel("Normalized close")
    ax.set_title(f"{spec.label}: signal location on CSI500 price path")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax2 = axes[1]
    if len(event):
        x = event["date"]
        y = event["directional_improvement_bp"] if "directional_improvement_bp" in event else event["o2o_h1"] * spec.effect_sign * 10000
        colors = np.where(y >= 0, COLORS[spec.key], "#999999")
        ax2.axhline(0, color="#333333", lw=0.7)
        ax2.scatter(x, y, c=colors, s=22, alpha=0.85)
    ax2.set_ylabel("Signal-day improvement (bp)")
    ax2.set_xlabel("Effective date (signal acts on this day only)")
    ax2.xaxis.set_major_locator(mdates.YearLocator(1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return save_fig(fig, f"signal_{spec.key}_timeline.png")


def plot_signal_forward(forward: pd.DataFrame, spec: SignalSpec, panel: pd.DataFrame) -> str:
    sub = forward.loc[forward.signal_key.eq(spec.key)].sort_values("horizon_days")
    h = sub["horizon_days"].to_numpy()
    event_mean = sub["event_mean_directional_bp"].to_numpy(dtype=float)
    eligible_mean = sub["eligible_mean_directional_bp"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.plot(h, event_mean, marker="o", color=COLORS[spec.key], lw=2, label="signal events")
    ax.plot(h, eligible_mean, marker="s", color="#555555", ls="--", lw=1.5, label="all eligible base days")
    ax.axhline(0, color="#333333", lw=0.7)
    ax.set_xticks(h)
    ax.set_xlabel("Forward horizon (trading days)")
    ax.set_ylabel("Directional / exit improvement (bp)")
    ax.set_title("Forward effect vs eligible baseline")
    ax.legend(fontsize=8)
    ax2 = axes[1]
    sub2 = panel.loc[panel[spec.column].eq(1) & panel["three_state"].eq(spec.expected_base)].copy()
    if len(sub2):
        values = sub2["o2o_h1"] * spec.effect_sign * 10000
        ax2.hist(values.dropna(), bins=min(10, max(4, len(values) // 3)), color=COLORS[spec.key], alpha=0.75, edgecolor="white")
        ax2.axvline(values.mean(), color="#111111", ls="--", lw=1.2, label=f"mean {values.mean():.1f} bp")
    ax2.axvline(0, color="#333333", lw=0.7)
    ax2.set_xlabel("Signal-day directional effect (bp)")
    ax2.set_ylabel("Event count")
    ax2.set_title("Event-day outcome distribution")
    ax2.legend(fontsize=8)
    return save_fig(fig, f"signal_{spec.key}_forward.png")


def plot_signal_shape(
    spec: SignalSpec, segments: pd.DataFrame, panel: pd.DataFrame, comparison: pd.DataFrame
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    relevant_state = spec.expected_base
    base = segments.loc[(segments.series == "base") & (segments.state == relevant_state), "length"]
    final_col = "reversal" if spec.final_series == "reversal" else "transfer"
    final = segments.loc[(segments.series == final_col) & (segments.state == relevant_state), "length"]
    bins = np.arange(0.5, max(10, int(max(base.max() if len(base) else 1, final.max() if len(final) else 1))) + 1.5, 1)
    ax.hist(base, bins=bins, alpha=0.55, color="#777777", label="base segments")
    ax.hist(final, bins=bins, alpha=0.55, color=COLORS[spec.final_series], label=f"{SERIES_LABELS[spec.final_series]}")
    ax.set_xlabel(f"Segment length in base state {relevant_state} (trading days)")
    ax.set_ylabel("Segment count")
    ax.set_title("Segment length shape around this signal")
    ax.legend(fontsize=8)
    ax2 = axes[1]
    events = panel.loc[panel[spec.column].eq(1) & panel["three_state"].eq(relevant_state)]
    if len(events):
        position = events["base_position_pct"] * 100
        ax2.hist(position, bins=np.linspace(0, 100, 11), color=COLORS[spec.key], alpha=0.8, edgecolor="white")
        ax2.axvline(position.mean(), color="#111111", ls="--", lw=1.2, label=f"mean {position.mean():.1f}%")
    ax2.set_xlabel("Signal position inside the original state segment (%)")
    ax2.set_ylabel("Event count")
    ax2.set_title("Where does the signal occur?")
    ax2.legend(fontsize=8)
    return save_fig(fig, f"signal_{spec.key}_shape.png")


def plot_signal_effect_summary(event_summary: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    data = event_summary.copy()
    x = np.arange(len(data))
    y = data["directional_mean_bp"].to_numpy(dtype=float)
    low = y - data["mean_ci_low_bp"].to_numpy(dtype=float)
    high = data["mean_ci_high_bp"].to_numpy(dtype=float) - y
    bars = ax.bar(x, y, color=[COLORS[k] for k in data["signal_key"]], alpha=0.85)
    ax.errorbar(x, y, yerr=[low, high], fmt="none", ecolor="#222222", capsize=4, lw=1)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x, [str(v).replace("（", "\n（") for v in data["signal_label"]])
    ax.set_ylabel("Mean signal-day directional / exit improvement (bp)")
    ax.set_title("Four signal effects: event-day mean and bootstrap 95% interval")
    for rect, row in zip(bars, data.itertuples()):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{row.directional_mean_bp:.1f}", ha="center", va="bottom" if row.directional_mean_bp >= 0 else "top", fontsize=9)
    return save_fig(fig, "overall_signal_effect_summary.png")


def plot_annual_stability(year_summary: pd.DataFrame) -> str:
    """Show yearly event effect, baseline, and sample count for each signal."""

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    for ax, spec in zip(axes.flat, SIGNAL_SPECS):
        sub = year_summary.loc[year_summary.signal_key.eq(spec.key)].sort_values("year")
        if len(sub):
            x = sub["year"].to_numpy(dtype=int)
            y = sub["directional_mean_bp"].to_numpy(dtype=float)
            baseline = sub["eligible_directional_mean_bp"].to_numpy(dtype=float)
            ax.bar(x, y, color=COLORS[spec.key], alpha=0.80, width=0.65, label="事件均值")
            ax.plot(x, baseline, color="#333333", marker="o", ls="--", lw=1.2, label="条件基础均值")
            for year, value, count in zip(x, y, sub["realized_event_days"]):
                if np.isfinite(value):
                    ax.text(year, value, f"{value:.0f}\n(n={int(count)})", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
        ax.axhline(0, color="#333333", lw=0.7)
        ax.set_title(spec.label, fontsize=10)
        ax.set_ylabel("方向化效果 (bp)")
        ax.set_xlabel("年份")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("年度稳定性：事件均值与条件基础均值")
    return save_fig(fig, "overall_annual_signal_stability.png")


def plot_overall_state_counts(panel: pd.DataFrame) -> str:
    series = {"base": "three_state", "reversal": "reversal_final_three_state", "transfer": "transfer_final_three_state", "combined": "combined_final_three_state"}
    counts = pd.DataFrame({name: panel[col].value_counts().reindex([-1, 0, 1], fill_value=0) for name, col in series.items()}).T
    fig, ax = plt.subplots(figsize=(10.5, 5))
    bottom = np.zeros(len(counts))
    for state in (-1, 0, 1):
        values = counts[state].to_numpy()
        ax.bar(counts.index, values, bottom=bottom, label=f"state {state}", color=COLORS[f"state_{state}"])
        for i, (b, v) in enumerate(zip(bottom, values)):
            if v > 80:
                ax.text(i, b + v / 2, str(int(v)), ha="center", va="center", fontsize=8, color="white" if state != 0 else "#222222")
        bottom += values
    ax.set_ylabel("Days on the common effective-date grid")
    ax.set_title("State composition before and after event-only adjustments")
    ax.legend(ncol=3)
    return save_fig(fig, "overall_state_counts.png")


def plot_overall_segment_duration(summaries: pd.DataFrame) -> str:
    active = summaries.loc[summaries.state.isin([-1, 1])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for state, ax in zip((-1, 1), axes):
        sub = active.loc[active.state.eq(state)].copy()
        x = np.arange(len(sub))
        ax.bar(x, sub.mean_days, color=[COLORS.get(s, "#777777") for s in sub.series], alpha=0.85)
        ax.set_xticks(x, [SERIES_LABELS[s] for s in sub.series], rotation=20, ha="right")
        ax.set_ylabel("Mean segment length (days)")
        ax.set_title(f"State {state}: mean length")
        for i, row in enumerate(sub.itertuples()):
            ax.text(i, row.mean_days, f"{row.mean_days:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Event-only adjustments change segment shape; no signal is carried forward")
    return save_fig(fig, "overall_segment_duration.png")


def plot_overall_return_distribution(segments: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, state in zip(axes, (-1, 1)):
        sub = segments.loc[
            segments.state.eq(state) & segments.return_available & segments.series.isin(["base", "reversal", "transfer", "combined"])
        ].copy()
        groups = [sub.loc[sub.series.eq(s), "aligned_segment_return"].dropna().to_numpy() * 100 for s in ["base", "reversal", "transfer", "combined"]]
        ax.boxplot(groups, labels=["base", "exit", "entry", "all"], showfliers=False)
        ax.axhline(0, color="#333333", lw=0.7)
        ax.set_ylabel("Aligned segment return (%)")
        ax.set_title(f"State {state}: segment return distribution")
    fig.suptitle("Directional segment returns: median, IQR, and the visible tail")
    return save_fig(fig, "overall_segment_return_distribution.png")


def plot_overall_index_curve(panel: pd.DataFrame) -> str:
    price = panel.loc[panel.close.notna()].copy()
    fig, ax = plt.subplots(figsize=(14, 6))
    if len(price):
        base = price.close.iloc[0]
        ax.plot(price.date, price.close / base, color="#333333", lw=1.0, label="CSI500 close normalized")
    for state in (-1, 0, 1):
        mask = price.three_state.eq(state)
        if mask.any():
            ax.scatter(price.loc[mask, "date"], price.loc[mask, "close"] / price.close.iloc[0], s=2.5, alpha=0.20, color=COLORS[f"state_{state}"], label=f"base {state}")
    for spec in SIGNAL_SPECS:
        mask = price[spec.column].eq(1) & price.three_state.eq(spec.expected_base)
        if mask.any():
            marker = "v" if "minus" in spec.key else "^"
            ax.scatter(price.loc[mask, "date"], price.loc[mask, "close"] / price.close.iloc[0], s=22, marker=marker, color=COLORS[spec.key], edgecolor="white", linewidth=0.25, label=spec.label)
    ax.set_title("CSI500 price path with all four signal families")
    ax.set_ylabel("Normalized close")
    ax.set_xlabel("Effective date")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return save_fig(fig, "overall_index_curve_four_signals.png")


def plot_overall_shape_capture(shape_summary: pd.DataFrame, shape_df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    sub = shape_summary.loc[shape_summary.state.isin([-1, 1])].copy()
    labels = [f"state {int(s)}" for s in sub.state]
    x = np.arange(len(sub))
    axes[0].bar(x - 0.18, sub.path_reversal_ge_3pct, width=0.36, color="#999999", label="≥3% path reversals")
    axes[0].bar(x + 0.18, sub.reversal_segments_captured, width=0.36, color="#4d9221", label="captured on/before reversal")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Segment count")
    axes[0].set_title("Path reversal and signal capture")
    axes[0].legend(fontsize=8)
    for i, row in enumerate(sub.itertuples()):
        axes[0].text(i + 0.18, row.reversal_segments_captured, f"{row.capture_rate_among_3pct_reversals_pct:.0f}%", ha="center", va="bottom", fontsize=8)
    for state, color in [(-1, COLORS["minus_exit"]), (1, COLORS["plus_exit"])]:
        values = shape_df.loc[shape_df.state.eq(state), "adverse_rebound_pct"].dropna()
        axes[1].hist(values, bins=np.linspace(0, max(5, values.max() if len(values) else 5), 12), alpha=0.6, color=color, label=f"state {state}")
    axes[1].axvline(3, color="#333333", ls="--", lw=0.9)
    axes[1].set_xlabel("Adverse rebound/drop from favorable extreme (%)")
    axes[1].set_ylabel("Segment count")
    axes[1].set_title("Original active-segment path shape")
    axes[1].legend(fontsize=8)
    return save_fig(fig, "overall_shape_reversal_capture.png")


def _format_table(df: pd.DataFrame, columns: list[str], rename: dict[str, str] | None = None, rows: int | None = None) -> str:
    sub = df.loc[:, columns].copy()
    if rows is not None:
        sub = sub.head(rows)
    if rename:
        sub = sub.rename(columns=rename)
    for column in sub.columns:
        if pd.api.types.is_datetime64_any_dtype(sub[column]):
            sub[column] = sub[column].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_integer_dtype(sub[column]):
            use_comma = column not in {"年份", "year"}
            sub[column] = sub[column].map(
                lambda value: "—"
                if pd.isna(value)
                else (f"{int(value):,}" if use_comma else f"{int(value)}")
            )
        elif pd.api.types.is_float_dtype(sub[column]):
            sub[column] = sub[column].map(lambda value: "—" if pd.isna(value) else f"{float(value):,.2f}")
        else:
            sub[column] = sub[column].map(lambda value: "—" if pd.isna(value) else value)
    return sub.to_markdown(index=False)


def _fmt_num(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):,.{digits}f}"


def write_signal_report(
    spec: SignalSpec,
    summary: pd.DataFrame,
    phase: pd.DataFrame,
    year_summary: pd.DataFrame,
    position_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    events: pd.DataFrame,
    forward: pd.DataFrame,
    segments: pd.DataFrame,
    comparison: pd.DataFrame,
    run_df: pd.DataFrame,
    shape_summary: pd.DataFrame,
    shape_df: pd.DataFrame,
    figure_paths: dict[str, str],
) -> Path:
    row = summary.loc[summary.signal_key.eq(spec.key)].iloc[0]
    phases = phase.loc[phase.signal_key.eq(spec.key)].copy()
    annual = year_summary.loc[year_summary.signal_key.eq(spec.key)].copy()
    positions = position_summary.loc[position_summary.signal_key.eq(spec.key)].copy()
    cluster = cluster_summary.loc[cluster_summary.signal_key.eq(spec.key)].copy()
    ev = events.loc[events.signal_key.eq(spec.key)].copy()
    fw = forward.loc[forward.signal_key.eq(spec.key)].copy()
    runs = run_df.loc[run_df.signal_key.eq(spec.key)].copy()
    shape_row = shape_summary.loc[shape_summary.state.eq(spec.expected_base)].copy()
    shape_details = shape_df.loc[shape_df.state.eq(spec.expected_base)].copy()
    rel_cmp = comparison.loc[(comparison.state == spec.expected_base) & comparison.series.isin([spec.final_series, "combined"])]
    signal_position = ev["signal_position_pct"].dropna() if len(ev) else pd.Series(dtype=float)
    early = int((signal_position <= 33.333).sum())
    mid = int(((signal_position > 33.333) & (signal_position <= 66.667)).sum())
    late = int((signal_position > 66.667).sum())
    best_position = "—"
    worst_position = "—"
    verdict = "有较明确的正向证据" if row.directional_mean_bp > 0 and row.signal_lift_vs_eligible_bp > 0 else "证据偏弱或需要谨慎解读"
    if row.realized_event_days < 20:
        verdict += "；事件样本较少，不能仅凭均值下强结论"
    source_note = "负向/正向退出只作用于信号当天，后续日期恢复基础三状态。" if spec.final_series == "reversal" else "零段转移只在当天把基础 0 改成 -1/+1；后续 0 日不延续重标，同日双向冲突保持 0。"
    report = []
    report.append(f"# {spec.label}｜独立详细分析\n")
    report.append(
        "本报告只分析一个信号，数据来自最新本地重跑的最终上传包口径。所有收益使用 effective-date 网格上的 CSI500 下一交易日开盘到再下一交易日开盘（O2O H1），不把 C2C 当作冻结评价指标。"
    )
    report.append("\n## 1. 信号定义与结论\n")
    report.append(
        f"- 来源：{spec.source}；冻结规则：`{spec.model}`。\n- 作用：{spec.action}；要求信号发生时基础状态为 `{spec.expected_base}`。\n- 事件口径：{source_note}\n- 本信号的判断：**{verdict}**。"
    )
    report.append(
        f"\n全周期有效事件 {int(row.event_days)} 天，覆盖符合条件的基础状态 {int(row.eligible_base_days)} 天中的 {row.coverage_pct:.2f}%；可计算 H1 的事件 {int(row.realized_event_days)} 天。信号日方向化/退出改善均值 **{row.directional_mean_bp:.2f} bp**，bootstrap 95% 区间 **[{row.mean_ci_low_bp:.2f}, {row.mean_ci_high_bp:.2f}] bp**；相对所有符合条件基础日均值 {row.eligible_directional_mean_bp:.2f} bp，差值为 **{row.signal_lift_vs_eligible_bp:.2f} bp**。"
    )
    report.append("\n单位说明：**100 bp = 1%**，所以 57.35 bp 等于 0.5735%。退出信号的 bp 是‘避免原持仓方向不利波动’的方向化表达；入口信号的 bp 是‘进入目标方向后’的方向化 O2O，不等于已经实现的组合收益。")
    report.append("\n![信号在指数路径中的位置](图片/" + Path(figure_paths["timeline"]).name + ")\n")
    report.append("\n## 2. 全周期与分阶段事件质量\n")
    report.append(
        _format_table(
            phases,
            ["phase", "event_days", "realized_event_days", "eligible_base_days", "coverage_pct", "directional_mean_bp", "directional_median_bp", "directional_win_rate_pct", "eligible_directional_mean_bp", "lift_vs_eligible_bp"],
            {"phase": "阶段", "event_days": "事件日", "realized_event_days": "已实现H1", "eligible_base_days": "条件基础日", "coverage_pct": "覆盖率%", "directional_mean_bp": "方向化均值bp", "directional_median_bp": "中位数bp", "directional_win_rate_pct": "胜率%", "eligible_directional_mean_bp": "条件基础均值bp", "lift_vs_eligible_bp": "相对提升bp"},
        )
    )
    report.append(
        f"\n事件日方向化胜率为 **{row.directional_win_rate_pct:.2f}%**，Wilson 95% 区间为 **[{row.directional_win_ci_low_pct:.2f}%, {row.directional_win_ci_high_pct:.2f}%]**。这里的‘胜率’是信号日 H1 方向化收益为正，不是最终持仓段胜率；两者需要分开看。"
    )
    annual_observed = annual.loc[annual.realized_event_days > 0].copy()
    positive_years = int((annual_observed.directional_mean_bp > 0).sum()) if len(annual_observed) else 0
    negative_years = int((annual_observed.directional_mean_bp < 0).sum()) if len(annual_observed) else 0
    annual_range = (
        f"{annual_observed.directional_mean_bp.min():.2f} 至 {annual_observed.directional_mean_bp.max():.2f} bp"
        if len(annual_observed)
        else "—"
    )
    report.append("\n### 2.1 年度稳定性：平均值是否只由少数年份贡献\n")
    report.append(
        f"按自然年拆分后，有完整 H1 观测的年份中，年度事件均值为正 **{positive_years} 年**、为负 **{negative_years} 年**，年度均值范围为 **{annual_range}**。这比全周期一个均值更能检验信号是否跨行情阶段重复出现；但年度样本仍可能很小，不能把‘正号年份数’当成统计显著性。"
    )
    report.append(
        _format_table(
            annual,
            ["year", "event_days", "realized_event_days", "eligible_base_days", "coverage_pct", "directional_mean_bp", "directional_median_bp", "directional_p05_bp", "directional_p95_bp", "directional_win_rate_pct", "directional_loss_gt_3pct_pct", "eligible_directional_mean_bp", "lift_vs_eligible_bp"],
            {"year": "年份", "event_days": "事件日", "realized_event_days": "已实现H1", "eligible_base_days": "条件基础日", "coverage_pct": "覆盖率%", "directional_mean_bp": "事件均值bp", "directional_median_bp": "中位bp", "directional_p05_bp": "P05bp", "directional_p95_bp": "P95bp", "directional_win_rate_pct": "胜率%", "directional_loss_gt_3pct_pct": "不利>3%", "eligible_directional_mean_bp": "条件均值bp", "lift_vs_eligible_bp": "相对提升bp"},
        )
    )
    report.append("\n![前向效果与事件日分布](图片/" + Path(figure_paths["forward"]).name + ")\n")
    report.append("\n## 3. 反转/转移发生在原始段的哪里\n")
    report.append(
        f"本信号发生在原始基础状态 `{spec.expected_base}` 段内。按段内位置分成前 1/3、中 1/3、后 1/3：分别为 **{early}、{mid}、{late}** 个事件；有事件的平均首个信号位置为 **{signal_position.mean():.1f}%**。这回答了信号是偏早、偏中段还是偏尾部，但不等于信号已经在最低点/最高点之前预测成功。"
        if len(signal_position)
        else "本信号没有可用事件位置记录。"
    )
    if len(runs):
        report.append(
            "\n信号连续运行段：\n\n"
            + _format_table(
                pd.DataFrame(
                    {
                        "指标": ["连续信号段数", "平均连续长度", "中位连续长度", "最大连续长度", "单日信号段占比"],
                        "数值": [len(runs), runs.run_length_trading_days.mean(), runs.run_length_trading_days.median(), runs.run_length_trading_days.max(), (runs.run_length_trading_days == 1).mean() * 100],
                    }
                ),
                ["指标", "数值"],
            )
        )
    else:
        report.append("\n没有形成连续信号段。\n")
    if len(positions):
        observed_positions = positions.loc[positions.realized_event_days > 0].copy()
        if len(observed_positions):
            best_position = observed_positions.loc[observed_positions.directional_mean_bp.idxmax(), "position_bucket"]
            worst_position = observed_positions.loc[observed_positions.directional_mean_bp.idxmin(), "position_bucket"]
            report.append(
                f"\n按原始段位置看，已实现 H1 的分组中，均值最高的是 **{best_position}**，最低的是 **{worst_position}**。位置分组用于观察‘信号是提前、居中还是偏尾部’，不是事后择优规则。\n\n"
                + _format_table(
                    positions,
                    ["position_bucket", "event_days", "realized_event_days", "mean_signal_position_pct", "mean_days_from_segment_start", "mean_days_to_segment_end", "directional_mean_bp", "directional_median_bp", "directional_p25_bp", "directional_p75_bp", "directional_win_rate_pct", "directional_loss_gt_3pct_pct"],
                    {"position_bucket": "段内位置", "event_days": "事件日", "realized_event_days": "已实现H1", "mean_signal_position_pct": "平均位置%", "mean_days_from_segment_start": "距段首天数", "mean_days_to_segment_end": "距段尾天数", "directional_mean_bp": "均值bp", "directional_median_bp": "中位bp", "directional_p25_bp": "P25bp", "directional_p75_bp": "P75bp", "directional_win_rate_pct": "胜率%", "directional_loss_gt_3pct_pct": "不利>3%"},
                )
            )
    if len(cluster):
        cluster_row = cluster.iloc[0]
        report.append(
            "\n### 3.1 连续信号的首次日、重复日与信号簇\n"
            f"把连续出现的信号视为一个信号簇后，本信号共有 **{int(cluster_row.run_count)} 个簇**，平均每簇 **{cluster_row.mean_run_length_days:.2f} 天**，最长 **{int(cluster_row.max_run_length_days)} 天**；按簇均值计算的方向化效果为 **{cluster_row.run_mean_directional_bp:.2f} bp**，簇级 bootstrap 区间为 **[{cluster_row.run_mean_ci_low_bp:.2f}, {cluster_row.run_mean_ci_high_bp:.2f}] bp**。首次信号日均值 **{cluster_row.first_event_mean_bp:.2f} bp**，重复信号日均值 **{cluster_row.repeat_event_mean_bp:.2f} bp**。这能区分‘真正的第一天有信息’和‘同一行情持续几天导致事件日均值被重复计数’。\n\n"
            + _format_table(
                cluster,
                ["run_count", "mean_run_length_days", "median_run_length_days", "max_run_length_days", "single_day_run_share_pct", "run_mean_directional_bp", "run_mean_ci_low_bp", "run_mean_ci_high_bp", "run_positive_share_pct", "first_event_days", "first_event_mean_bp", "first_event_win_rate_pct", "repeat_event_days", "repeat_event_mean_bp", "repeat_event_win_rate_pct"],
                {"run_count": "信号簇数", "mean_run_length_days": "平均簇长", "median_run_length_days": "中位簇长", "max_run_length_days": "最大簇长", "single_day_run_share_pct": "单日簇占比%", "run_mean_directional_bp": "簇均值bp", "run_mean_ci_low_bp": "簇CI下限bp", "run_mean_ci_high_bp": "簇CI上限bp", "run_positive_share_pct": "正向簇占比%", "first_event_days": "首次日样本", "first_event_mean_bp": "首次日均值bp", "first_event_win_rate_pct": "首次日胜率%", "repeat_event_days": "重复日样本", "repeat_event_mean_bp": "重复日均值bp", "repeat_event_win_rate_pct": "重复日胜率%"},
            )
        )
    fwd1 = fw.loc[fw.horizon_days.eq(1), "event_mean_directional_bp"]
    fwd10 = fw.loc[fw.horizon_days.eq(10), "event_mean_directional_bp"]
    fwd1_value = float(fwd1.iloc[0]) if len(fwd1) and pd.notna(fwd1.iloc[0]) else np.nan
    fwd10_value = float(fwd10.iloc[0]) if len(fwd10) and pd.notna(fwd10.iloc[0]) else np.nan
    late_share = late / len(signal_position) * 100 if len(signal_position) else np.nan
    report.append("\n### 3.2 细致观察与评价\n")
    report.append(
        f"- **时序位置**：事件平均落在原始段 **{signal_position.mean():.1f}%** 的位置，后 1/3 占 **{late_share:.1f}%**；分组均值最高为 **{best_position}**，最低为 **{worst_position}**。因此，位置上更应把它理解为‘段内何时确认/应对’，而不是笼统地称为提前预测。\n\n"
        f"- **跨年份稳定性**：有 H1 观测的年份中，正均值 **{positive_years}/{positive_years + negative_years}**；年度均值范围 **{annual_range}**。若正均值主要由单个大事件年份贡献，整体均值的可迁移性就要打折。\n\n"
        f"- **前向路径**：信号日 H1 为 **{fwd1_value:.2f} bp**，H10 为 **{fwd10_value:.2f} bp**；这反映信号后的路径是否继续沿目标方向/继续出现可避免的反向波动，但由于 event-only 只改当天标签，不能直接理解为信号后持仓十天的策略收益。\n\n"
        f"- **重复计数风险**：本信号连续簇的簇均值为 **{_fmt_num(cluster.iloc[0].run_mean_directional_bp)} bp**，首次日均值 **{_fmt_num(cluster.iloc[0].first_event_mean_bp)} bp**，重复日均值 **{_fmt_num(cluster.iloc[0].repeat_event_mean_bp)} bp**。簇级结果比逐日均值更接近‘一次行情触发’，应与逐日结果并读。"
    )
    report.append("\n![原始段长度和信号位置](图片/" + Path(figure_paths["shape"]).name + ")\n")
    report.append("\n## 4. 与原始三状态相比，持仓段形状如何变化\n")
    report.append(
        "下表只看该信号对应的基础状态，并同时列出仅应用本信号所属组、以及四信号合并后的 event-only 结果。event-only 结果不是把信号向后延长的真实持仓路径；它的作用是当天改标签，下一天重新读取基础状态。"
    )
    if len(rel_cmp):
        report.append(
            _format_table(
                rel_cmp,
                ["comparison", "base_segments", "new_segments", "segment_count_delta", "base_total_days", "new_total_days", "base_mean_days", "new_mean_days", "base_one_day_share_pct", "new_one_day_share_pct", "base_mean_aligned_return_bp", "new_mean_aligned_return_bp", "base_win_rate_pct", "new_win_rate_pct", "base_adverse_tail_gt_3pct", "new_adverse_tail_gt_3pct"],
                {"comparison": "对比", "base_segments": "基础段数", "new_segments": "调整后段数", "segment_count_delta": "段数变化", "base_total_days": "基础天数", "new_total_days": "调整后天数", "base_mean_days": "基础均长", "new_mean_days": "调整后均长", "base_one_day_share_pct": "基础单日段%", "new_one_day_share_pct": "调整后单日段%", "base_mean_aligned_return_bp": "基础段均收益bp", "new_mean_aligned_return_bp": "调整后段均收益bp", "base_win_rate_pct": "基础胜率%", "new_win_rate_pct": "调整后胜率%", "base_adverse_tail_gt_3pct": "基础不利>3%段", "new_adverse_tail_gt_3pct": "调整后不利>3%段"},
            )
        )
    report.append(
        "\n解释：退出信号的改善重点应是减少与持仓方向相反的尾部，入口信号的改善重点应是让原始 0 段中新增的方向段有正的方向化收益。若段数上升、单日段上升，这是 event-only 规则的机械结果，不应误称为持仓稳定性改善。"
    )
    report.append("\n### 4.1 原始段形状与反转持续时间\n")
    if len(shape_row):
        report.append(
            _format_table(
                shape_row,
                ["state", "segments", "mean_length", "median_length", "path_reversal_ge_3pct", "path_reversal_ge_3pct_share_pct", "segments_with_signal", "segments_with_3pct_reversal", "reversal_segments_captured", "capture_rate_among_3pct_reversals_pct", "mean_reversal_duration_trading_days_3pct", "median_reversal_duration_trading_days_3pct", "mean_signal_lead_trading_days", "median_signal_lead_trading_days", "mean_signal_position_pct"],
                {"state": "基础状态", "segments": "原始段数", "mean_length": "平均段长", "median_length": "中位段长", "path_reversal_ge_3pct": "路径反转≥3%段数", "path_reversal_ge_3pct_share_pct": "路径反转≥3%占比%", "segments_with_signal": "含相关信号段", "segments_with_3pct_reversal": "有≥3%反转段", "reversal_segments_captured": "捕获段数", "capture_rate_among_3pct_reversals_pct": "捕获率%", "mean_reversal_duration_trading_days_3pct": "极值至3%反转均值交易日", "median_reversal_duration_trading_days_3pct": "极值至3%反转中位交易日", "mean_signal_lead_trading_days": "首信号领先均值交易日", "median_signal_lead_trading_days": "首信号领先中位交易日", "mean_signal_position_pct": "首信号平均位置%"},
            )
        )
        if spec.expected_base in (-1, 1):
            report.append(
                "\n这里的‘极值至 3% 反转交易日’是从该基础持仓段的有利极值开始，到价格路径累计出现 3% 不利反转的交易日数；‘首信号领先’是首个相关信号相对该阈值日提前的交易日数。它们描述信号与路径反转的时序关系，不是把信号收益直接等同于交易收益。"
            )
        else:
            report.append(
                "\n该信号发生在基础 0 段，0 段没有方向化的有利极值反转定义，因此这里重点看段长、信号位置和后续方向化前向收益，不把 0 段强行解释成持仓反转段。"
            )
    if len(shape_details):
        if spec.expected_base in (-1, 1):
            shape_top = shape_details.sort_values("adverse_rebound_pct", ascending=False).head(5)
            report.append("\n该基础状态中路径反转最明显的 5 个段：\n\n")
            report.append(
                _format_table(
                    shape_top,
                    ["start_date", "end_date", "length", "aligned_segment_return", "adverse_rebound_pct", "favorable_extreme_date", "reversal_date_3pct", "reversal_duration_trading_days_3pct", "signal_count_in_segment", "first_signal_date", "signal_before_or_on_reversal", "signal_lead_trading_days"],
                    {"start_date": "开始", "end_date": "结束", "length": "段长", "aligned_segment_return": "方向化段收益", "adverse_rebound_pct": "不利反弹/回撤%", "favorable_extreme_date": "有利极值日", "reversal_date_3pct": "≥3%反转日", "reversal_duration_trading_days_3pct": "极值至反转交易日", "signal_count_in_segment": "段内信号数", "first_signal_date": "首信号日", "signal_before_or_on_reversal": "是否捕获", "signal_lead_trading_days": "首信号领先交易日"},
                )
            )
    report.append("\n## 5. 事件日明细与前向分布\n")
    report.append(
        _format_table(
            fw,
            ["horizon_days", "event_n", "event_mean_directional_bp", "event_median_directional_bp", "event_win_rate_pct", "event_p25_bp", "event_p75_bp", "eligible_n", "eligible_mean_directional_bp", "eligible_win_rate_pct"],
            {"horizon_days": "前向天数", "event_n": "事件样本", "event_mean_directional_bp": "事件均值bp", "event_median_directional_bp": "事件中位bp", "event_win_rate_pct": "事件胜率%", "event_p25_bp": "事件P25bp", "event_p75_bp": "事件P75bp", "eligible_n": "条件样本", "eligible_mean_directional_bp": "条件均值bp", "eligible_win_rate_pct": "条件胜率%"},
        )
    )
    if len(ev):
        top = ev.sort_values("directional_improvement_bp", ascending=False).head(10).copy()
        report.append("\n表现最好的 10 个事件日：\n\n")
        report.append(
            _format_table(
                top,
                ["date", "phase", "three_state", "base_segment_length", "base_position", "signal_position_pct", "raw_o2o_h1_bp", "directional_improvement_bp"],
                {"date": "日期", "phase": "阶段", "three_state": "基础状态", "base_segment_length": "原始段长", "base_position": "段内位置", "signal_position_pct": "段内位置%", "raw_o2o_h1_bp": "原始O2O bp", "directional_improvement_bp": "方向化/退出改善bp"},
            )
        )
        report.append("\n注意：上述是描述性排序，不能把事后收益最大的事件当作事前可知的信息。完整事件表和所有四个信号的结果在 `../01_报告资料与基础上传包/统计表/` 中。\n")
    report.append("\n## 6. 最终判断\n")
    report.append(
        f"综合本信号的事件均值、相对条件基础日提升、不同阶段稳定性、前向 1/3/5/10 日路径以及原始段位置，本信号的证据强度判断为：**{verdict}**。最重要的限制是事件样本不是独立同分布的：连续信号日会集中在同一段行情，且最近一个有效期末日可能没有完整未来 O2O 观察。因此，本报告支持‘是否有描述性改善’的判断，不支持把历史均值解释成未来收益保证。"
    )
    path = REPORT_DIR / REPORT_FILENAMES[spec.key]
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return path


def write_overall_report(
    panel: pd.DataFrame,
    merge_audit: dict[str, Any],
    price_audit: dict[str, Any],
    event_summary: pd.DataFrame,
    phase_summary: pd.DataFrame,
    year_summary: pd.DataFrame,
    position_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    summaries: pd.DataFrame,
    daily: pd.DataFrame,
    comparison: pd.DataFrame,
    shape_summary: pd.DataFrame,
    shape_df: pd.DataFrame,
    forward: pd.DataFrame,
    figure_paths: dict[str, str],
) -> Path:
    lines: list[str] = []
    lines.append("# 四个反转信号合并总体分析\n")
    lines.append(
        "本报告是四个信号合并后的独立总报告，覆盖：上传包与远端结果一致性、9 列合并规则、四个信号逐个有效性、最终三状态的持仓段结构、收益分布、胜率和原始价格路径形状。分析严格按当前 event-only 口径：信号只改对应的一天，后续没有新信号的日期恢复基础三状态。"
    )
    lines.append("\n## 1. 数据和一致性结论\n")
    lines.append(
        f"- 合并输入：反转侧 `{Path(REV_SIGNAL_PATH).name}` 与零段侧 `{Path(ZERO_SIGNAL_PATH).name}`。\n- 两个五列源结果都为 **{merge_audit['reversal_rows']} 行**，日期范围 **{merge_audit['reversal_date_min'].date()} 至 {merge_audit['reversal_date_max'].date()}**；日期网格一致，基础 `three_state` 逐日一致。\n- 基础状态计数为 **-1={merge_audit['reversal_base_counts'].get(-1, merge_audit['reversal_base_counts'].get('-1'))}、0={merge_audit['reversal_base_counts'].get(0, merge_audit['reversal_base_counts'].get('0'))}、+1={merge_audit['reversal_base_counts'].get(1, merge_audit['reversal_base_counts'].get('1'))}**。\n- 价格使用长0项目的完整现货快照；与反转项目现货快照的重叠 OHLC 在 1e-7 容差内一致：`{price_audit['overlap_ohlc_exact_within_1e-7']}`。\n- 远端截图的关键数字已由本地项目内的结果对照文件和截图复核：2091/2091 行、日期至 2026-08-17、基础三状态 233/1452/406、逐日差异 0；零段侧远端与米筐本地核心检查 `all_core_checks_exact=true`。"
    )
    lines.append("\n这里的 2026-08-17 是 2026-08-14 形成日后的下一实际交易日展示行；该尾行没有完整未来价格，所以不把它当成已经实现的 H1 收益。\n")
    lines.append("\n## 2. 9 列合并文件与状态规则\n")
    lines.append(
        "已生成 `../01_报告资料与基础上传包/数据源/combined_four_signals_nine_columns.csv`。9 列为：`date`、`three_state`、两个退出信号、退出组 event-only 状态、两个零段转移信号、转移组 event-only 状态、四信号合并最终状态。两个源文件中的重复基础状态先逐日核对，再收敛为一个 canonical `three_state`；源文件的两个 `final_three_state` 分别保留为组内审计列。"
    )
    lines.append(
        "\n规则表：\n\n| 信号 | 仅在基础状态 | 当天动作 | 后续日期 |\n|---|---:|---|---|\n| 负向退出 | -1 | 改为 0 | 恢复基础状态 |\n| 正向退出 | +1 | 改为 0 | 恢复基础状态 |\n| 零段向下 | 0 | 改为 -1 | 恢复基础状态 |\n| 零段向上 | 0 | 改为 +1 | 恢复基础状态 |\n\n同一天 `minus_entry_signal=1` 和 `plus_entry_signal=1` 时保持 0；本次冲突日为 **" + str(merge_audit["entry_conflict_days"]) + " 天**。退出信号不会在基础 0 上生效，入口信号不会在基础 ±1 上生效。"
    )
    lines.append("\n![四个信号在指数曲线中的位置](图片/" + Path(figure_paths["index_curve"]).name + ")\n")
    lines.append("\n## 3. 四个信号逐个有效性比较\n")
    lines.append(
        _format_table(
            event_summary,
            ["signal_label", "model", "event_days", "eligible_base_days", "coverage_pct", "realized_event_days", "directional_mean_bp", "mean_ci_low_bp", "mean_ci_high_bp", "eligible_directional_mean_bp", "signal_lift_vs_eligible_bp", "directional_win_rate_pct", "eligible_directional_win_rate_pct"],
            {"signal_label": "信号", "model": "冻结模型", "event_days": "事件日", "eligible_base_days": "条件基础日", "coverage_pct": "覆盖率%", "realized_event_days": "已实现H1", "directional_mean_bp": "事件均值bp", "mean_ci_low_bp": "CI下限bp", "mean_ci_high_bp": "CI上限bp", "eligible_directional_mean_bp": "条件均值bp", "signal_lift_vs_eligible_bp": "相对提升bp", "directional_win_rate_pct": "事件胜率%", "eligible_directional_win_rate_pct": "条件胜率%"},
        )
    )
    lines.append("\n![四个信号事件日均值和置信区间](图片/" + Path(figure_paths["effect"]).name + ")\n")
    lines.append(
        "判断不能只看均值：负向/正向退出的提升符号是‘退出避免原持仓方向的反向行情’，零段向下/向上是进入目标方向后的方向化收益。事件日样本存在连续聚集，bootstrap 区间只作描述性不确定性参考。"
    )
    lines.append("\n### 3.1 年度稳定性与行情阶段差异\n")
    lines.append(
        "年度拆分用来回答：总体均值是否由某一轮行情集中贡献。下表同时给出事件均值、P05/P95、胜率、条件基础均值和相对提升；当某年事件样本很少时，应优先看方向是否一致和区间宽度，而不是只看点估计。"
    )
    lines.append(
        _format_table(
            year_summary,
            ["signal_label", "year", "event_days", "realized_event_days", "directional_mean_bp", "directional_median_bp", "directional_p05_bp", "directional_p95_bp", "directional_win_rate_pct", "eligible_directional_mean_bp", "lift_vs_eligible_bp"],
            {"signal_label": "信号", "year": "年份", "event_days": "事件日", "realized_event_days": "已实现H1", "directional_mean_bp": "事件均值bp", "directional_median_bp": "中位bp", "directional_p05_bp": "P05bp", "directional_p95_bp": "P95bp", "directional_win_rate_pct": "胜率%", "eligible_directional_mean_bp": "条件均值bp", "lift_vs_eligible_bp": "相对提升bp"},
        )
    )
    lines.append("\n![年度稳定性](图片/" + Path(figure_paths["annual"]).name + ")\n")
    lines.append("\n### 3.2 信号在原始持仓段中的位置\n")
    lines.append(
        _format_table(
            position_summary,
            ["signal_label", "position_bucket", "event_days", "realized_event_days", "mean_signal_position_pct", "mean_days_to_segment_end", "directional_mean_bp", "directional_median_bp", "directional_p25_bp", "directional_p75_bp", "directional_win_rate_pct", "directional_loss_gt_3pct_pct"],
            {"signal_label": "信号", "position_bucket": "段内位置", "event_days": "事件日", "realized_event_days": "已实现H1", "mean_signal_position_pct": "平均位置%", "mean_days_to_segment_end": "距段尾天数", "directional_mean_bp": "均值bp", "directional_median_bp": "中位bp", "directional_p25_bp": "P25bp", "directional_p75_bp": "P75bp", "directional_win_rate_pct": "胜率%", "directional_loss_gt_3pct_pct": "不利>3%"},
        )
    )
    lines.append("\n### 3.3 信号簇与重复信号日\n")
    lines.append(
        "连续信号日按同一信号簇处理后，簇级均值更接近‘一次行情触发’的视角。首次日与重复日的拆分则用来判断信号信息是否主要集中在第一天。"
    )
    lines.append(
        _format_table(
            cluster_summary,
            ["signal_label", "run_count", "mean_run_length_days", "max_run_length_days", "single_day_run_share_pct", "run_mean_directional_bp", "run_mean_ci_low_bp", "run_mean_ci_high_bp", "run_positive_share_pct", "first_event_mean_bp", "first_event_win_rate_pct", "repeat_event_mean_bp", "repeat_event_win_rate_pct"],
            {"signal_label": "信号", "run_count": "信号簇数", "mean_run_length_days": "平均簇长", "max_run_length_days": "最大簇长", "single_day_run_share_pct": "单日簇占比%", "run_mean_directional_bp": "簇均值bp", "run_mean_ci_low_bp": "簇CI下限bp", "run_mean_ci_high_bp": "簇CI上限bp", "run_positive_share_pct": "正向簇占比%", "first_event_mean_bp": "首次日均值bp", "first_event_win_rate_pct": "首次日胜率%", "repeat_event_mean_bp": "重复日均值bp", "repeat_event_win_rate_pct": "重复日胜率%"},
        )
    )
    lines.append("\n## 4. 基础、退出组、转移组和四信号合并后的状态持仓段\n")
    lines.append(
        _format_table(
            summaries,
            ["series_label", "state_label", "segments", "total_days", "mean_days", "median_days", "p25_days", "p75_days", "max_days", "one_day_share_pct", "le_3d_share_pct", "return_observations", "mean_aligned_segment_return_bp", "median_aligned_segment_return_bp", "p05_aligned_segment_return_bp", "p95_aligned_segment_return_bp", "segment_win_rate_pct", "adverse_tail_gt_3pct"],
            {"series_label": "系列", "state_label": "状态", "segments": "段数", "total_days": "天数", "mean_days": "均长", "median_days": "中位长", "p25_days": "P25长", "p75_days": "P75长", "max_days": "最大长", "one_day_share_pct": "单日段%", "le_3d_share_pct": "≤3日段%", "return_observations": "可计收益段", "mean_aligned_segment_return_bp": "段均方向收益bp", "median_aligned_segment_return_bp": "段中位方向收益bp", "p05_aligned_segment_return_bp": "P05bp", "p95_aligned_segment_return_bp": "P95bp", "segment_win_rate_pct": "段胜率%", "adverse_tail_gt_3pct": "不利>3%段"},
        )
    )
    lines.append("\n![状态构成变化](图片/" + Path(figure_paths["counts"]).name + ")\n")
    lines.append("\n![持仓段平均长度变化](图片/" + Path(figure_paths["duration"]).name + ")\n")
    lines.append("\n![持仓段收益分布](图片/" + Path(figure_paths["returns"]).name + ")\n")
    lines.append(
        "总体结构上，event-only 规则会机械地增加状态切换和短段，特别是零段转移会把原始 0 段中的事件日切出。因此，‘四信号合并后段数/单日段比例’不能单独当作持仓质量；需要结合事件日收益、原始段反转覆盖率和不利尾部一起判断。"
    )
    lines.append("\n## 5. 原始持仓段形状：反转发生在哪里、持续多久\n")
    lines.append(
        _format_table(
            shape_summary,
            ["state", "segments", "mean_length", "median_length", "net_adverse_segments", "path_reversal_ge_2pct", "path_reversal_ge_3pct", "path_reversal_ge_5pct", "path_reversal_ge_3pct_share_pct", "segments_with_signal", "segments_with_3pct_reversal", "reversal_segments_captured", "capture_rate_among_3pct_reversals_pct", "mean_reversal_duration_trading_days_3pct", "median_reversal_duration_trading_days_3pct", "mean_signal_lead_trading_days", "median_signal_lead_trading_days", "mean_trading_days_after_first_signal_to_segment_end", "mean_signal_position_pct"],
            {"state": "基础状态", "segments": "原始段数", "mean_length": "平均段长", "median_length": "中位段长", "net_adverse_segments": "净结果不利段", "path_reversal_ge_2pct": "路径反转≥2%", "path_reversal_ge_3pct": "路径反转≥3%", "path_reversal_ge_5pct": "路径反转≥5%", "path_reversal_ge_3pct_share_pct": "路径反转≥3%占比%", "segments_with_signal": "含相关信号段", "segments_with_3pct_reversal": "有≥3%反转段", "reversal_segments_captured": "信号捕获段", "capture_rate_among_3pct_reversals_pct": "捕获率%", "mean_reversal_duration_trading_days_3pct": "极值至3%反转均值交易日", "median_reversal_duration_trading_days_3pct": "极值至3%反转中位交易日", "mean_signal_lead_trading_days": "首信号领先均值交易日", "median_signal_lead_trading_days": "首信号领先中位交易日", "mean_trading_days_after_first_signal_to_segment_end": "首信号至段尾均值交易日", "mean_signal_position_pct": "首信号平均位置%"},
        )
    )
    lines.append("\n![原始段路径反转和信号捕获](图片/" + Path(figure_paths["shape"]).name + ")\n")
    for state in (-1, 1):
        group = shape_df.loc[shape_df.state.eq(state)].copy()
        if len(group):
            top = group.sort_values("adverse_rebound_pct", ascending=False).head(5)
            lines.append(f"\n基础状态 `{state}` 路径反转最明显的段：\n\n")
            lines.append(
                _format_table(
                    top,
                    ["start_date", "end_date", "length", "raw_segment_return", "aligned_segment_return", "adverse_rebound_pct", "favorable_extreme_date", "reversal_date_3pct", "reversal_duration_trading_days_3pct", "signal_count_in_segment", "first_signal_date", "signal_before_or_on_reversal", "signal_lead_trading_days"],
                    {"start_date": "开始", "end_date": "结束", "length": "段长", "raw_segment_return": "原始段收益", "aligned_segment_return": "方向化段收益", "adverse_rebound_pct": "不利反弹/回撤%", "favorable_extreme_date": "有利极值日", "reversal_date_3pct": "≥3%反转日", "reversal_duration_trading_days_3pct": "极值至反转交易日", "signal_count_in_segment": "段内相关信号数", "first_signal_date": "首信号日", "signal_before_or_on_reversal": "是否提前/当日捕获", "signal_lead_trading_days": "首信号领先交易日"},
                )
            )
    lines.append(
        "\n形状解释：对基础 -1 段，用段内最低点后的上涨衡量反向反弹；对基础 +1 段，用段内最高点后的下跌衡量反向回撤。≥3% 是描述性阈值，不是训练或冻结规则。‘捕获’只表示相关信号在该阈值反转日之前或当日出现，不表示信号一定发生在极值点之前，也不表示退出后的后续路径被永久删除。"
    )
    lines.append("\n## 6. 最终合并结果的收益与胜率\n")
    combined_daily = daily.loc[daily.series.eq("combined")].copy()
    lines.append(
        _format_table(
            combined_daily,
            ["state_label", "days", "mean_raw_h1_bp", "median_raw_h1_bp", "p05_raw_h1_bp", "p95_raw_h1_bp", "mean_directional_h1_bp", "median_directional_h1_bp", "directional_win_rate_pct", "directional_loss_gt_3pct"],
            {"state_label": "最终状态", "days": "可计H1天数", "mean_raw_h1_bp": "原始H1均值bp", "median_raw_h1_bp": "原始H1中位bp", "p05_raw_h1_bp": "P05bp", "p95_raw_h1_bp": "P95bp", "mean_directional_h1_bp": "方向化H1均值bp", "median_directional_h1_bp": "方向化H1中位bp", "directional_win_rate_pct": "方向化日胜率%", "directional_loss_gt_3pct": "方向化亏损>3%日"},
        )
    )
    lines.append(
        "\n最终结果应这样读：退出信号的有效性体现在信号日避免不利方向；零段入口信号的有效性体现在新增的 -1/+1 事件日是否有正方向化收益。最终 `combined_final_three_state` 是标签层的 event-only 结果，不能把它直接当作把持仓从信号日开始延续到整段的交易回测。"
    )
    lines.append("\n## 7. 可重复运行与文件位置\n")
    lines.append(
        "主运行文件是 `../02_HTML生成工具/四信号合并与综合分析.ipynb`；`../02_HTML生成工具/analysis_pipeline.py` 是可复用库函数。notebook 会重新读取 `../01_报告资料与基础上传包/数据源/` 下的两个五列结果和现货快照，重建 9 列文件、审计表、PNG 图和本报告，不依赖远端凭据。\n\n核心输出：\n\n- `../01_报告资料与基础上传包/数据源/combined_four_signals_nine_columns.csv`：9 列合并结果；\n- `../01_报告资料与基础上传包/统计表/`：事件日、年度稳定性、段内位置、信号簇、状态段、价格形状和合并审计表；\n- `图片/`：本报告和四个单信号报告使用的图片；\n- 五份中文 Markdown 报告均位于当前汇报文件夹。"
    )
    lines.append("\n## 8. 局限\n\n1. 事件日存在连续聚集，不是独立样本；bootstrap 仅作描述性区间。\n2. 2026-08-17 是展示尾行，缺少完整未来价格，收益统计会排除缺失的 O2O。\n3. Test 只作为冻结后的观察区间；报告不反向用 Test 选参数。\n4. 本报告分析的是指数状态标签和事件收益，不包含交易成本、滑点、资金容量或真实组合执行。\n")
    path = REPORT_DIR / "四个反转信号合并总体分析.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_all() -> dict[str, Any]:
    """Run the complete merge, audit, plotting, and report pipeline."""

    _ensure_dirs()
    configure_plot_style()
    rev, zero, merge_audit = load_source_signals()
    combined = build_combined_nine_columns(rev, zero, merge_audit)
    COMBINED_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(COMBINED_PATH, index=False, encoding="utf-8-sig")
    save_json(merge_audit, MERGE_AUDIT_PATH)

    price, price_audit = load_price_data()
    panel = make_analysis_panel(combined, price)
    series_to_state = {
        "base": "three_state",
        "reversal": "reversal_final_three_state",
        "transfer": "transfer_final_three_state",
        "combined": "combined_final_three_state",
    }
    segments = pd.concat(
        [build_segments(panel, col, series) for series, col in series_to_state.items()],
        ignore_index=True,
    )
    summaries = summarize_segments(segments)
    daily = summarize_daily_states(panel, series_to_state)
    event_summary, phase_summary, events, forward = event_metrics(panel)
    runs = signal_run_metrics(panel)
    year_summary = event_year_metrics(panel)
    position_summary = event_position_metrics(events)
    cluster_summary = event_cluster_metrics(events)
    shape_df, shape_summary = build_shape_audit(panel, segments)
    comparison = make_relevant_comparison(summaries, daily)

    tables = {
        "combined_signal": combined,
        "panel": panel,
        "segments": segments,
        "segment_summary": summaries,
        "daily_state_summary": daily,
        "event_summary": event_summary,
        "event_phase_summary": phase_summary,
        "event_year_summary": year_summary,
        "event_position_summary": position_summary,
        "event_cluster_summary": cluster_summary,
        "event_details": events,
        "event_forward_summary": forward,
        "signal_runs": runs,
        "shape_details": shape_df,
        "shape_summary": shape_summary,
        "base_vs_adjusted_comparison": comparison,
    }
    for name, table in tables.items():
        path = TABLE_DIR / f"{name}.csv"
        output = table.copy()
        for column in output.columns:
            if pd.api.types.is_datetime64_any_dtype(output[column]):
                output[column] = output[column].dt.strftime("%Y-%m-%d")
        output.to_csv(path, index=False, encoding="utf-8-sig")

    figure_paths: dict[str, str] = {}
    for spec in SIGNAL_SPECS:
        figure_paths[f"{spec.key}_timeline"] = plot_signal_timeline(panel, spec)
        figure_paths[f"{spec.key}_forward"] = plot_signal_forward(forward, spec, panel)
        figure_paths[f"{spec.key}_shape"] = plot_signal_shape(spec, segments, add_segment_context(panel, "three_state", "base"), comparison)
    figure_paths["effect"] = plot_signal_effect_summary(event_summary)
    figure_paths["annual"] = plot_annual_stability(year_summary)
    figure_paths["counts"] = plot_overall_state_counts(panel)
    figure_paths["duration"] = plot_overall_segment_duration(summaries)
    figure_paths["returns"] = plot_overall_return_distribution(segments)
    figure_paths["index_curve"] = plot_overall_index_curve(panel)
    figure_paths["shape"] = plot_overall_shape_capture(shape_summary, shape_df)

    reports: dict[str, str] = {}
    for spec in SIGNAL_SPECS:
        reports[spec.key] = str(
            write_signal_report(
                spec,
                event_summary,
                phase_summary,
                year_summary,
                position_summary,
                cluster_summary,
                events,
                forward,
                segments,
                comparison,
                runs,
                shape_summary,
                shape_df,
                {
                    "timeline": figure_paths[f"{spec.key}_timeline"],
                    "forward": figure_paths[f"{spec.key}_forward"],
                    "shape": figure_paths[f"{spec.key}_shape"],
                },
            )
        )
    reports["overall"] = str(
        write_overall_report(
            panel,
            merge_audit,
            price_audit,
            event_summary,
            phase_summary,
            year_summary,
            position_summary,
            cluster_summary,
            summaries,
            daily,
            comparison,
            shape_summary,
            shape_df,
            forward,
            {
                "effect": figure_paths["effect"],
                "annual": figure_paths["annual"],
                "counts": figure_paths["counts"],
                "duration": figure_paths["duration"],
                "returns": figure_paths["returns"],
                "index_curve": figure_paths["index_curve"],
                "shape": figure_paths["shape"],
            },
        )
    )
    manifest = {
        "run_date": "2026-08-17",
        "root": str(ROOT),
        "combined_file": str(COMBINED_PATH),
        "merge_audit": merge_audit,
        "price_audit": price_audit,
        "row_count": int(len(panel)),
        "date_min": panel.date.min(),
        "date_max": panel.date.max(),
        "price_available_rows": int(panel.price_available.sum()),
        "o2o_h1_available_rows": int(panel.o2o_h1_available.sum()),
        "figures": figure_paths,
        "reports": reports,
        "signal_summary": event_summary.to_dict(orient="records"),
        "event_year_summary": year_summary.to_dict(orient="records"),
        "event_position_summary": position_summary.to_dict(orient="records"),
        "event_cluster_summary": cluster_summary.to_dict(orient="records"),
        "segment_summary": summaries.to_dict(orient="records"),
        "shape_summary": shape_summary.to_dict(orient="records"),
    }
    save_json(manifest, MANIFEST_PATH)
    return {
        "combined": combined,
        "panel": panel,
        "segments": segments,
        "segment_summary": summaries,
        "daily_state_summary": daily,
        "event_summary": event_summary,
        "event_phase_summary": phase_summary,
        "event_year_summary": year_summary,
        "event_position_summary": position_summary,
        "event_cluster_summary": cluster_summary,
        "event_details": events,
        "event_forward_summary": forward,
        "signal_runs": runs,
        "shape_details": shape_df,
        "shape_summary": shape_summary,
        "comparison": comparison,
        "merge_audit": merge_audit,
        "price_audit": price_audit,
        "figures": figure_paths,
        "reports": reports,
        "manifest": manifest,
    }


if __name__ == "__main__":
    result = run_all()
    print("Combined file:", COMBINED_PATH)
    print("Rows:", len(result["combined"]))
    print(result["event_summary"].to_string(index=False))
    print("Reports:")
    for path in result["reports"].values():
        print(" -", path)
