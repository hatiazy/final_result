"""Pure-spot helpers used by the optional segment and parity notebooks.

The production freeze does not depend on this module.  The helpers only read
the event-export CSV created in ``COMPANY_OUTPUT_DIR`` and, when requested,
the same single raw spot file used by the production notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spot_panel import load_spot_panel


EVENT_SIGNAL_COLUMNS = (
    "date",
    "three_state",
    "minus_entry_signal",
    "plus_entry_signal",
    "final_three_state",
)

PHASES = {
    "Development": (pd.Timestamp("2018-01-01"), pd.Timestamp("2022-12-31")),
    "Validation": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
    "Test": (pd.Timestamp("2025-01-01"), pd.Timestamp("2100-01-01")),
}


def load_event_signal(path: str | Path) -> pd.DataFrame:
    """Load and validate the five-column event-state export."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"event signal CSV not found: {source}")
    frame = pd.read_csv(source)
    missing = sorted(set(EVENT_SIGNAL_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"event signal CSV missing columns: {missing}")
    frame = frame.loc[:, EVENT_SIGNAL_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame["date"].duplicated().any():
        raise ValueError("event signal CSV contains duplicate dates")
    for column in EVENT_SIGNAL_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    for column in ("three_state", "final_three_state"):
        if not frame[column].isin([-1, 0, 1]).all():
            raise ValueError(f"{column} contains a value outside -1/0/1")
    for column in ("minus_entry_signal", "plus_entry_signal"):
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"{column} contains a value outside 0/1")
    return frame.sort_values("date").reset_index(drop=True)


def attach_spot_execution_metrics(
    signal: pd.DataFrame,
    spot_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach close and O2O_H1 on the signal's actual execution date."""

    signal = signal.copy()
    signal["date"] = pd.to_datetime(signal["date"], errors="raise").dt.normalize()
    spot, panel, audit = load_spot_panel(spot_path)
    panel_dates = pd.to_datetime(panel["effective_date"], errors="coerce").dt.normalize()
    execution = pd.DataFrame(
        {
            "date": panel_dates,
            "o2o_h1": pd.to_numeric(panel["o2o_h1"], errors="coerce"),
        }
    ).dropna(subset=["date"])
    execution = execution.drop_duplicates("date", keep="last")
    prices = spot[["date", "open", "close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="raise").dt.normalize()
    prices = prices.sort_values("date").drop_duplicates("date", keep="last")
    prices["open_next"] = prices["open"].shift(-1)
    prices = prices.rename(
        columns={"open": "open_current", "close": "close_current"}
    )
    result = signal.merge(execution, on="date", how="left")
    result = result.merge(prices, on="date", how="left")
    result["o2o_h1"] = pd.to_numeric(result["o2o_h1"], errors="coerce")
    result["o2o_h1_bp"] = result["o2o_h1"] * 10000.0
    result["period"] = np.select(
        [
            result["date"].le(PHASES["Development"][1]),
            result["date"].le(PHASES["Validation"][1]),
        ],
        ["Development", "Validation"],
        default="Test",
    )
    return result, audit


def state_runs(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return contiguous runs for one state column."""

    work = frame[["date", column]].copy()
    work["run_id"] = work[column].ne(work[column].shift()).cumsum()
    rows: list[dict[str, Any]] = []
    for run_id, group in work.groupby("run_id", sort=True):
        rows.append(
            {
                "run_id": int(run_id),
                "state": int(group[column].iloc[0]),
                "start_date": group["date"].iloc[0].date().isoformat(),
                "end_date": group["date"].iloc[-1].date().isoformat(),
                "trading_days": int(len(group)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["run_id", "state", "start_date", "end_date", "trading_days"],
    )


def duration_summary(runs: pd.DataFrame, label: str) -> pd.DataFrame:
    """Summarize count, total, central tendency and fragmentation by state."""

    rows: list[dict[str, Any]] = []
    for state in (-1, 0, 1):
        values = pd.to_numeric(
            runs.loc[runs["state"].eq(state), "trading_days"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "series": label,
                "state": state,
                "segments": int(len(values)),
                "total_days": int(values.sum()),
                "mean_days": float(values.mean()) if len(values) else np.nan,
                "median_days": float(values.median()) if len(values) else np.nan,
                "p25_days": float(values.quantile(0.25)) if len(values) else np.nan,
                "p75_days": float(values.quantile(0.75)) if len(values) else np.nan,
                "one_day_share": float((values == 1).mean()) if len(values) else np.nan,
                "two_day_share": float((values <= 2).mean()) if len(values) else np.nan,
                "max_days": int(values.max()) if len(values) else 0,
                "min_days": int(values.min()) if len(values) else 0,
            }
        )
    return pd.DataFrame(rows)


def state_daily_return_summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Summarize daily raw and state-directional O2O_H1 returns."""

    rows: list[dict[str, Any]] = []
    returns = pd.to_numeric(frame["o2o_h1"], errors="coerce")
    for state in (-1, 0, 1):
        mask = frame[column].eq(state) & returns.notna()
        raw = returns.loc[mask].to_numpy(dtype=float)
        directional = raw * state if state else raw
        rows.append(
            {
                "state": state,
                "days": int(len(raw)),
                "mean_raw_h1_bp": float(np.mean(raw) * 10000) if len(raw) else np.nan,
                "median_raw_h1_bp": float(np.median(raw) * 10000) if len(raw) else np.nan,
                "p05_raw_h1_bp": float(np.quantile(raw, 0.05) * 10000) if len(raw) else np.nan,
                "p95_raw_h1_bp": float(np.quantile(raw, 0.95) * 10000) if len(raw) else np.nan,
                "mean_directional_h1_bp": float(np.mean(directional) * 10000) if len(raw) else np.nan,
                "median_directional_h1_bp": float(np.median(directional) * 10000) if len(raw) else np.nan,
                "directional_win_rate": float(np.mean(directional > 0)) if len(raw) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def segment_return_rows(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return one compounded O2O_H1 result for every complete state segment."""

    work = frame[["date", column, "o2o_h1"]].copy()
    work["run_id"] = work[column].ne(work[column].shift()).cumsum()
    rows: list[dict[str, Any]] = []
    for run_id, group in work.groupby("run_id", sort=True):
        state = int(group[column].iloc[0])
        values = pd.to_numeric(group["o2o_h1"], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(values):
            compounded = np.nan
        else:
            compounded = float(np.prod(1.0 + values) - 1.0)
        directional = compounded * state if np.isfinite(compounded) and state else compounded
        rows.append(
            {
                "run_id": int(run_id),
                "state": state,
                "start_date": group["date"].iloc[0].date().isoformat(),
                "end_date": group["date"].iloc[-1].date().isoformat(),
                "trading_days": int(len(group)),
                "segment_raw_return_bp": compounded * 10000 if np.isfinite(compounded) else np.nan,
                "segment_directional_return_bp": directional * 10000 if np.isfinite(directional) else np.nan,
                "segment_directional_win": bool(directional > 0) if np.isfinite(directional) and state else np.nan,
            }
        )
    return pd.DataFrame(rows)


def segment_return_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate compounded segment outcomes by state."""

    output: list[dict[str, Any]] = []
    for state in (-1, 0, 1):
        values = pd.to_numeric(
            rows.loc[rows["state"].eq(state), "segment_directional_return_bp"],
            errors="coerce",
        ).dropna()
        output.append(
            {
                "state": state,
                "segments": int(len(values)),
                "mean_segment_directional_bp": float(values.mean()) if len(values) else np.nan,
                "median_segment_directional_bp": float(values.median()) if len(values) else np.nan,
                "p05_segment_directional_bp": float(values.quantile(0.05)) if len(values) else np.nan,
                "p95_segment_directional_bp": float(values.quantile(0.95)) if len(values) else np.nan,
                "segment_directional_win_rate": float((values > 0).mean()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(output)


def state_count_table(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    """Return day count and share by state."""

    total = len(frame)
    return pd.DataFrame(
        {
            "series": label,
            "state": [-1, 0, 1],
            "days": [int(frame[column].eq(value).sum()) for value in (-1, 0, 1)],
        }
    ).assign(share_pct=lambda x: x["days"] / total * 100 if total else np.nan)


def compare_state_frames(
    generated: pd.DataFrame,
    baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align generated/base state dates and return summary, crosstab, counts, mismatches."""

    generated = generated.copy()
    baseline = baseline.copy()
    generated["date"] = pd.to_datetime(generated["date"], errors="raise").dt.normalize()
    baseline["date"] = pd.to_datetime(baseline["date"], errors="raise").dt.normalize()
    left = generated[["date", "three_state"]].rename(columns={"three_state": "generated_three_state"})
    right = baseline[["date", "three_state"]].rename(columns={"three_state": "baseline_three_state"})
    common = left.merge(right, on="date", how="inner")
    common["match"] = common["generated_three_state"].eq(common["baseline_three_state"])
    common["delta_generated_minus_baseline"] = (
        common["generated_three_state"] - common["baseline_three_state"]
    )
    mismatches = common.loc[~common["match"]].copy()
    summary = pd.DataFrame(
        [
            {
                "generated_rows": len(left),
                "baseline_rows": len(right),
                "common_dates": len(common),
                "matching_dates": int(common["match"].sum()),
                "mismatch_dates": int((~common["match"]).sum()),
                "match_rate": float(common["match"].mean()) if len(common) else np.nan,
                "generated_date_min": left["date"].min().date().isoformat() if len(left) else "",
                "generated_date_max": left["date"].max().date().isoformat() if len(left) else "",
                "baseline_date_min": right["date"].min().date().isoformat() if len(right) else "",
                "baseline_date_max": right["date"].max().date().isoformat() if len(right) else "",
            }
        ]
    )
    cross = pd.crosstab(
        common["generated_three_state"],
        common["baseline_three_state"],
        rownames=["generated_three_state"],
        colnames=["baseline_three_state"],
        dropna=False,
    )
    counts = pd.DataFrame(
        {
            "generated_days": left["generated_three_state"].value_counts().reindex([-1, 0, 1], fill_value=0),
            "baseline_days": right["baseline_three_state"].value_counts().reindex([-1, 0, 1], fill_value=0),
        }
    ).rename_axis("three_state").reset_index()
    counts["delta_generated_minus_baseline"] = counts["generated_days"] - counts["baseline_days"]
    return summary, cross, counts, mismatches


__all__ = [
    "EVENT_SIGNAL_COLUMNS",
    "PHASES",
    "attach_spot_execution_metrics",
    "compare_state_frames",
    "duration_summary",
    "load_event_signal",
    "segment_return_rows",
    "segment_return_summary",
    "state_count_table",
    "state_daily_return_summary",
    "state_runs",
]
