"""Post-freeze holding-segment and return-distribution audit.

This module is descriptive only. It never enters candidate admission or
company-side Top1 selection. It compares the original three-state series with
the event-day exit-adjusted series on the effective-date/open-price grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "date",
    "three_state",
    "minus_exit_signal",
    "plus_exit_signal",
    "final_three_state",
)
STATE_VALUES = (-1, 0, 1)


def _read_price(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path, encoding="utf-8-sig")
    else:
        raise ValueError(f"不支持的价格文件: {path}")
    if isinstance(raw.index, pd.DatetimeIndex):
        raw = raw.reset_index()
    date_column = next(
        (name for name in ("date", "trade_dt", "trade_date", "effective_date") if name in raw.columns),
        None,
    )
    if date_column is None or "open" not in raw.columns:
        raise ValueError("价格文件至少需要 date/trade_dt 和 open 列")
    frame = pd.DataFrame({
        "date": pd.to_datetime(raw[date_column], errors="coerce").dt.normalize(),
        "open": pd.to_numeric(raw["open"], errors="coerce"),
    }).dropna().drop_duplicates("date", keep="last").sort_values("date")
    if frame.empty:
        raise ValueError("价格文件没有有效日期和开盘价")
    return frame.set_index("date")


def _load_default_price() -> tuple[pd.DataFrame, str]:
    from common.data import _read_spot, load_paths

    _, paths = load_paths()
    spot = _read_spot(paths["spot"])
    return spot[["open"]].copy(), str(paths["spot"])


def _series_segments(frame: pd.DataFrame, state_column: str) -> pd.DataFrame:
    work = frame[["date", state_column]].copy()
    work["run_id"] = work[state_column].ne(work[state_column].shift()).cumsum()
    rows: list[dict[str, Any]] = []
    for run_id, group in work.groupby("run_id", sort=True):
        rows.append({
            "run_id": int(run_id),
            "state": int(group[state_column].iloc[0]),
            "start_date": group["date"].iloc[0],
            "end_date": group["date"].iloc[-1],
            "holding_days": int(len(group)),
        })
    return pd.DataFrame(rows)


def _add_segment_returns(segments: pd.DataFrame, dates: pd.DatetimeIndex, price: pd.DataFrame) -> pd.DataFrame:
    output = segments.copy()
    open_map = price["open"]
    # A segment's return is measured from its first execution-day open to the
    # next execution-day open after its last day. The final incomplete row is
    # kept in the holding-duration audit but has no return observation.
    next_date = pd.Series(dates[1:].to_numpy(), index=dates[:-1])
    raw_returns: list[float] = []
    aligned_returns: list[float] = []
    available: list[bool] = []
    for row in output.itertuples(index=False):
        state = int(row.state)
        start = pd.Timestamp(row.start_date)
        end = pd.Timestamp(row.end_date)
        after = next_date.get(end, pd.NaT)
        start_open = open_map.get(start, np.nan)
        after_open = open_map.get(after, np.nan) if pd.notna(after) else np.nan
        ok = bool(np.isfinite(start_open) and np.isfinite(after_open) and start_open > 0)
        if ok and state in (-1, 1):
            raw = float(after_open / start_open - 1.0)
            aligned = float((1.0 + state * raw) - 1.0)
        elif ok and state == 0:
            raw = 0.0
            aligned = 0.0
        else:
            raw = np.nan
            aligned = np.nan
        raw_returns.append(raw)
        aligned_returns.append(aligned)
        available.append(ok)
    output["raw_segment_return"] = raw_returns
    output["aligned_segment_return"] = aligned_returns
    output["return_available"] = available
    return output


def _daily_returns(frame: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    open_map = price["open"]
    dates = pd.DatetimeIndex(output["date"])
    next_dates = pd.Series(dates[1:].to_numpy(), index=dates[:-1])
    values: list[float] = []
    for date in dates:
        nxt = next_dates.get(date, pd.NaT)
        current = open_map.get(date, np.nan)
        following = open_map.get(nxt, np.nan) if pd.notna(nxt) else np.nan
        values.append(float(following / current - 1.0) if np.isfinite(current) and np.isfinite(following) and current > 0 else np.nan)
    output["raw_o2o_return"] = values
    output["aligned_o2o_return"] = output["raw_o2o_return"] * output["state_for_return"]
    return output


def _summary(segments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (series, state), group in segments.groupby(["series", "state"], sort=True):
        durations = group["holding_days"].astype(float)
        returns = group.loc[group["return_available"] & group["state"].isin([-1, 1]), "aligned_segment_return"].dropna()
        raw_returns = group.loc[group["return_available"] & group["state"].isin([-1, 1]), "raw_segment_return"].dropna()
        rows.append({
            "series": series,
            "state": int(state),
            "segments": int(len(group)),
            "total_holding_days": int(durations.sum()),
            "mean_holding_days": float(durations.mean()),
            "median_holding_days": float(durations.median()),
            "max_holding_days": int(durations.max()),
            "one_day_segments": int((durations == 1).sum()),
            "one_day_share_pct": float((durations == 1).mean() * 100),
            "short_le_3d_segments": int((durations <= 3).sum()),
            "short_le_3d_share_pct": float((durations <= 3).mean() * 100),
            "return_observations": int(len(returns)),
            "positive_return_segments": int((returns > 0).sum()),
            "positive_return_share_pct": float((returns > 0).mean() * 100) if len(returns) else np.nan,
            "return_gt_3pct_segments": int((returns > 0.03).sum()),
            "return_gt_5pct_segments": int((returns > 0.05).sum()),
            "mean_positive_segment_return_pct": float(returns.loc[returns > 0].mean() * 100) if (returns > 0).any() else np.nan,
            "segment_win_rate_pct": float((returns > 0).mean() * 100) if len(returns) else np.nan,
            "mean_aligned_segment_return_pct": float(returns.mean() * 100) if len(returns) else np.nan,
            "median_aligned_segment_return_pct": float(returns.median() * 100) if len(returns) else np.nan,
            "min_aligned_segment_return_pct": float(returns.min() * 100) if len(returns) else np.nan,
            "max_aligned_segment_return_pct": float(returns.max() * 100) if len(returns) else np.nan,
            "raw_positive_segment_count": int((raw_returns > 0).sum()),
            "raw_positive_segment_share_pct": float((raw_returns > 0).mean() * 100) if len(raw_returns) else np.nan,
            "raw_gt_3pct_segment_count": int((raw_returns > 0.03).sum()),
            "raw_gt_5pct_segment_count": int((raw_returns > 0.05).sum()),
            "mean_positive_raw_segment_return_pct": float(raw_returns.loc[raw_returns > 0].mean() * 100) if (raw_returns > 0).any() else np.nan,
            "raw_segment_win_rate_pct": float((raw_returns > 0).mean() * 100) if len(raw_returns) else np.nan,
            "mean_raw_segment_return_pct": float(raw_returns.mean() * 100) if len(raw_returns) else np.nan,
            "median_raw_segment_return_pct": float(raw_returns.median() * 100) if len(raw_returns) else np.nan,
            "min_raw_segment_return_pct": float(raw_returns.min() * 100) if len(raw_returns) else np.nan,
            "max_raw_segment_return_pct": float(raw_returns.max() * 100) if len(raw_returns) else np.nan,
        })
    return pd.DataFrame(rows)


def _distribution(segments: pd.DataFrame, level: str) -> pd.DataFrame:
    return _distribution_column(
        segments,
        level,
        "aligned_segment_return" if level == "segment" else "aligned_o2o_return",
    )


def _distribution_column(segments: pd.DataFrame, level: str, value_column: str) -> pd.DataFrame:
    if level == "segment":
        eligible = segments["return_available"] & segments["state"].isin([-1, 1])
    elif level == "daily":
        eligible = segments["return_available"] & segments["state"].isin([-1, 1])
    else:
        raise ValueError(level)
    rows: list[dict[str, Any]] = []
    for (series, state), group in segments.loc[eligible].groupby(["series", "state"], sort=True):
        values = group[value_column].dropna().to_numpy(float)
        if not len(values):
            continue
        quantiles = np.nanpercentile(values, [1, 5, 25, 50, 75, 95, 99]) * 100
        rows.append({
            "level": level,
            "series": series,
            "state": int(state),
            "n": int(len(values)),
            "mean_pct": float(np.mean(values) * 100),
            "std_pct": float(np.std(values, ddof=1) * 100) if len(values) > 1 else np.nan,
            "q01_pct": float(quantiles[0]),
            "q05_pct": float(quantiles[1]),
            "q25_pct": float(quantiles[2]),
            "q50_pct": float(quantiles[3]),
            "q75_pct": float(quantiles[4]),
            "q95_pct": float(quantiles[5]),
            "q99_pct": float(quantiles[6]),
            "win_rate_pct": float(np.mean(values > 0) * 100),
        })
    return pd.DataFrame(rows)


def audit_state_adjustment(signal: pd.DataFrame, price: pd.DataFrame | None = None, price_source: str | None = None) -> dict[str, Any]:
    """Return segment/duration/return-distribution audits for both series."""
    missing = [column for column in REQUIRED_COLUMNS if column not in signal.columns]
    if missing:
        raise ValueError(f"signal CSV 缺少列: {missing}")
    frame = signal[list(REQUIRED_COLUMNS)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for column in REQUIRED_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    if price is None:
        price, price_source = _load_default_price()
    price = price.copy()
    price.index = pd.to_datetime(price.index, errors="raise").normalize()
    price = price[~price.index.duplicated(keep="last")].sort_index()

    segment_frames: list[pd.DataFrame] = []
    for series, column in (("original", "three_state"), ("adjusted", "final_three_state")):
        segments = _series_segments(frame, column)
        segments["series"] = series
        segment_frames.append(_add_segment_returns(segments, pd.DatetimeIndex(frame["date"]), price))
    segments = pd.concat(segment_frames, ignore_index=True)

    # Daily returns are duplicated by series to make the distribution table
    # directly comparable with the segment table.
    daily_frames: list[pd.DataFrame] = []
    for series, column in (("original", "three_state"), ("adjusted", "final_three_state")):
        daily = frame[["date", column]].rename(columns={column: "state_for_return"})
        daily["series"] = series
        daily_frames.append(_daily_returns(daily, price))
    daily = pd.concat(daily_frames, ignore_index=True)
    daily["return_available"] = daily["raw_o2o_return"].notna()
    daily["state"] = daily["state_for_return"]

    segment_summary = _summary(segments)
    daily_summary = _summary(
        daily.assign(
            run_id=np.arange(len(daily)),
            start_date=daily["date"],
            end_date=daily["date"],
            holding_days=1,
            raw_segment_return=daily["raw_o2o_return"],
            aligned_segment_return=daily["aligned_o2o_return"],
        )
    ).rename(columns={"segments": "daily_observations"})
    segment_distribution = _distribution(segments, "segment")
    daily_distribution = _distribution(daily, "daily")
    raw_segment_distribution = _distribution_column(segments, "segment", "raw_segment_return")
    raw_daily_distribution = _distribution_column(daily, "daily", "raw_o2o_return")

    # Compact, screenshot-friendly comparison of the raw index tail.  For the
    # -1 state, a positive raw index return is the adverse (upward) tail.
    raw_tail_rows: list[dict[str, Any]] = []
    for state in (-1, 1):
        original = segment_summary.loc[
            segment_summary["series"].eq("original") & segment_summary["state"].eq(state)
        ].iloc[0]
        adjusted = segment_summary.loc[
            segment_summary["series"].eq("adjusted") & segment_summary["state"].eq(state)
        ].iloc[0]
        raw_orig = raw_segment_distribution.loc[
            raw_segment_distribution["series"].eq("original") & raw_segment_distribution["state"].eq(state)
        ].iloc[0]
        raw_adj = raw_segment_distribution.loc[
            raw_segment_distribution["series"].eq("adjusted") & raw_segment_distribution["state"].eq(state)
        ].iloc[0]
        raw_tail_rows.append({
            "state": state,
            "positive_raw_segments_original": int(original["raw_positive_segment_count"]),
            "positive_raw_segments_adjusted": int(adjusted["raw_positive_segment_count"]),
            "positive_raw_segments_delta": int(adjusted["raw_positive_segment_count"] - original["raw_positive_segment_count"]),
            "positive_raw_share_pct_original": float(original["raw_positive_segment_share_pct"]),
            "positive_raw_share_pct_adjusted": float(adjusted["raw_positive_segment_share_pct"]),
            "positive_raw_share_delta_pct": float(adjusted["raw_positive_segment_share_pct"] - original["raw_positive_segment_share_pct"]),
            "raw_gt_3pct_segments_original": int(original["raw_gt_3pct_segment_count"]),
            "raw_gt_3pct_segments_adjusted": int(adjusted["raw_gt_3pct_segment_count"]),
            "raw_gt_3pct_segments_delta": int(adjusted["raw_gt_3pct_segment_count"] - original["raw_gt_3pct_segment_count"]),
            "raw_gt_5pct_segments_original": int(original["raw_gt_5pct_segment_count"]),
            "raw_gt_5pct_segments_adjusted": int(adjusted["raw_gt_5pct_segment_count"]),
            "raw_gt_5pct_segments_delta": int(adjusted["raw_gt_5pct_segment_count"] - original["raw_gt_5pct_segment_count"]),
            "raw_mean_pct_original": float(original["mean_raw_segment_return_pct"]),
            "raw_mean_pct_adjusted": float(adjusted["mean_raw_segment_return_pct"]),
            "raw_mean_delta_pct": float(adjusted["mean_raw_segment_return_pct"] - original["mean_raw_segment_return_pct"]),
            "raw_q95_pct_original": float(raw_orig["q95_pct"]),
            "raw_q95_pct_adjusted": float(raw_adj["q95_pct"]),
            "raw_q95_delta_pct": float(raw_adj["q95_pct"] - raw_orig["q95_pct"]),
            "raw_max_pct_original": float(original["max_raw_segment_return_pct"]),
            "raw_max_pct_adjusted": float(adjusted["max_raw_segment_return_pct"]),
            "raw_max_delta_pct": float(adjusted["max_raw_segment_return_pct"] - original["max_raw_segment_return_pct"]),
        })
    raw_tail_comparison = pd.DataFrame(raw_tail_rows)

    # Side-by-side deltas make it easy to see whether exit adjustment shortened
    # positions and whether the aligned segment return distribution improved.
    comparison = segment_summary.pivot(index="state", columns="series")
    comparison.columns = [f"{metric}_{series}" for metric, series in comparison.columns]
    comparison = comparison.reset_index()
    for metric in ("total_holding_days", "mean_holding_days", "median_holding_days", "one_day_share_pct", "short_le_3d_share_pct", "positive_return_segments", "positive_return_share_pct", "return_gt_3pct_segments", "return_gt_5pct_segments", "mean_positive_segment_return_pct", "segment_win_rate_pct", "mean_aligned_segment_return_pct", "median_aligned_segment_return_pct", "raw_positive_segment_count", "raw_positive_segment_share_pct", "raw_gt_3pct_segment_count", "raw_gt_5pct_segment_count", "mean_positive_raw_segment_return_pct", "raw_segment_win_rate_pct", "mean_raw_segment_return_pct", "median_raw_segment_return_pct"):
        final_name = f"{metric}_adjusted"
        original_name = f"{metric}_original"
        if final_name in comparison and original_name in comparison:
            comparison[f"{metric}_delta_adjusted_minus_original"] = comparison[final_name] - comparison[original_name]

    original_daily = daily.loc[daily["series"].eq("original")]
    quality_rows: list[dict[str, Any]] = []
    for state in (-1, 1):
        original = segment_summary.loc[
            segment_summary["series"].eq("original") & segment_summary["state"].eq(state)
        ].iloc[0]
        adjusted = segment_summary.loc[
            segment_summary["series"].eq("adjusted") & segment_summary["state"].eq(state)
        ].iloc[0]
        quality_rows.append({
            "state": state,
            "segment_win_rate_improved": bool(adjusted["segment_win_rate_pct"] > original["segment_win_rate_pct"]),
            "positive_segment_count_reduced": bool(adjusted["positive_return_segments"] < original["positive_return_segments"]),
            "gt_3pct_positive_tail_reduced": bool(adjusted["return_gt_3pct_segments"] < original["return_gt_3pct_segments"]),
            "gt_5pct_positive_tail_reduced": bool(adjusted["return_gt_5pct_segments"] < original["return_gt_5pct_segments"]),
            "q95_positive_tail_reduced": bool(
                segment_distribution.loc[
                    segment_distribution["series"].eq("adjusted") & segment_distribution["state"].eq(state), "q95_pct"
                ].iloc[0]
                < segment_distribution.loc[
                    segment_distribution["series"].eq("original") & segment_distribution["state"].eq(state), "q95_pct"
                ].iloc[0]
            ),
            "max_positive_tail_reduced": bool(
                adjusted["max_aligned_segment_return_pct"] < original["max_aligned_segment_return_pct"]
            ),
            "negative_tail_loss_reduced": bool(
                segment_distribution.loc[
                    segment_distribution["series"].eq("adjusted") & segment_distribution["state"].eq(state), "q05_pct"
                ].iloc[0]
                > segment_distribution.loc[
                    segment_distribution["series"].eq("original") & segment_distribution["state"].eq(state), "q05_pct"
                ].iloc[0]
            ),
            "raw_positive_segment_count_reduced": bool(adjusted["raw_positive_segment_count"] < original["raw_positive_segment_count"]),
            "raw_gt_3pct_positive_tail_reduced": bool(adjusted["raw_gt_3pct_segment_count"] < original["raw_gt_3pct_segment_count"]),
            "raw_gt_5pct_positive_tail_reduced": bool(adjusted["raw_gt_5pct_segment_count"] < original["raw_gt_5pct_segment_count"]),
            "raw_q95_positive_tail_reduced": bool(
                raw_segment_distribution.loc[
                    raw_segment_distribution["series"].eq("adjusted") & raw_segment_distribution["state"].eq(state), "q95_pct"
                ].iloc[0]
                < raw_segment_distribution.loc[
                    raw_segment_distribution["series"].eq("original") & raw_segment_distribution["state"].eq(state), "q95_pct"
                ].iloc[0]
            ),
            "raw_max_positive_tail_reduced": bool(adjusted["max_raw_segment_return_pct"] < original["max_raw_segment_return_pct"]),
            "selection_used": False,
        })
    quality_flags = pd.DataFrame(quality_rows)

    return {
        "segments": segments,
        "daily": daily,
        "segment_summary": segment_summary,
        "daily_summary": daily_summary,
        "segment_distribution": segment_distribution,
        "daily_distribution": daily_distribution,
        "raw_segment_distribution": raw_segment_distribution,
        "raw_daily_distribution": raw_daily_distribution,
        "raw_tail_comparison": raw_tail_comparison,
        "comparison": comparison,
        "quality_flags": quality_flags,
        "price_source": price_source,
        "signal_date_min": str(frame["date"].min().date()),
        "signal_date_max": str(frame["date"].max().date()),
        "price_date_min": str(price.index.min().date()),
        "price_date_max": str(price.index.max().date()),
        # Report one signal-day series as the primary count.  The internal
        # table contains both original and adjusted copies, so its raw count
        # is exactly twice this value.
        "return_available_daily_rows": int(original_daily["return_available"].sum()),
        "return_missing_daily_rows": int((~original_daily["return_available"]).sum()),
        "return_available_daily_rows_all_series": int(daily["return_available"].sum()),
        "return_missing_daily_rows_all_series": int((~daily["return_available"]).sum()),
        "selection_used": False,
    }


__all__ = ["REQUIRED_COLUMNS", "audit_state_adjustment"]
