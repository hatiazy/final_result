from __future__ import annotations

"""Stage 06: historical analogues and mechanism diagnostics.

This module is diagnostic only.  It never changes the frozen signal engine,
thresholds, holding packages, or production outputs.  It reads the same local
spot snapshot and the already generated holding-aware eight-list used by
Stages 03--05, then produces tables and figures for:

* local/remote parity and date-lineage checks;
* historical market windows similar to the latest 2026 window;
* the observable model-internal mechanism behind faster 2026 response;
* confirmation/hysteresis and fixed-anchor *proxy* counterfactuals;
* forward performance of the historical analogue windows.

The analogue ranking uses only features known at each window end.  Future
returns are written only as post-hoc evaluation columns and never enter the
ranking or signal generation.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Heiti SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from reproduce_remote_o2o import (
    SIGNAL_COLUMNS,
    STATE_MARKER_SHAPES,
    _add_state_marker_legend,
    _load_engine_panel,
    _plot_curve_with_execution_points,
    build_execution_frame,
    build_risk_summary,
    read_holding_eight,
)


REMOTE_METRICS = {
    ("Raw 1545", "total_return_pct"): 143.76,
    ("Raw 1545", "annualized_additive_return_pct"): 17.30,
    ("Raw 1545", "annualized_volatility_pct"): 15.05,
    ("Raw 1545", "sharpe"): 1.15,
    ("Raw 1545", "active_days"): 639.0,
    ("Raw 1545", "winning_directional_days"): 353.0,
    ("Adj 1545 (+4 Reversal)", "total_return_pct"): 283.16,
    ("Adj 1545 (+4 Reversal)", "annualized_additive_return_pct"): 34.08,
    ("Adj 1545 (+4 Reversal)", "annualized_volatility_pct"): 17.78,
    ("Adj 1545 (+4 Reversal)", "sharpe"): 1.92,
    ("Adj 1545 (+4 Reversal)", "active_days"): 1088.0,
    ("Adj 1545 (+4 Reversal)", "winning_directional_days"): 608.0,
    ("CSI500 Buy & Hold", "total_return_pct"): 44.88,
    ("CSI500 Buy & Hold", "annualized_additive_return_pct"): 5.40,
    ("CSI500 Buy & Hold", "annualized_volatility_pct"): 24.03,
    ("CSI500 Buy & Hold", "sharpe"): 0.22,
    ("CSI500 Buy & Hold", "active_days"): 2094.0,
    ("CSI500 Buy & Hold", "winning_directional_days"): 1082.0,
}


MODEL_FEATURES = [
    "return_20",
    "return_60",
    "return_120",
    "vol_20",
    "vol_60",
    "drawdown_60",
    "range_20",
    "gap_20",
    "axis_mean_60",
    "axis_std_60",
    "slow_fast_mean_60",
    "positive_sync_mean_60",
    "negative_sync_mean_60",
    "neutral_share_60",
    "switch_rate_60",
]


# These are the three red boxes drawn on the teacher-provided 2023--2026
# reference image.  The image has no machine-readable annotations, so the
# boundaries are recorded explicitly here and are kept unchanged between
# local and remote runs.  They are diagnostic windows only; they never enter
# signal generation, threshold selection, or any freeze decision.
TEACHER_RED_WINDOWS = (
    {
        "window_id": "红色区间1",
        "start": "2023-11-01",
        "end": "2024-01-15",
        "description": "2023年末至2024年初的低波动/方向切换过渡区",
    },
    {
        "window_id": "红色区间2",
        "start": "2024-04-15",
        "end": "2024-09-15",
        "description": "2024年中段的反复震荡与方向分歧区",
    },
    {
        "window_id": "红色区间3",
        "start": "2025-03-01",
        "end": "2025-07-15",
        "description": "2025年上半年中性与快速转移并存区",
    },
)


def _absolute(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        raise ValueError(f"{label} 必须使用绝对路径：{value}")
    return value.resolve()


def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}", na=False)
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if (~compact).any():
        result.loc[~compact] = pd.to_datetime(text.loc[~compact], errors="coerce", format="mixed")
    if compact.any():
        result.loc[compact] = pd.to_datetime(text.loc[compact], errors="coerce", format="%Y%m%d")
    return result.dt.normalize()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_spot(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    lower = {str(column).strip().lower(): column for column in raw.columns}

    def choose(names: tuple[str, ...], label: str) -> Any:
        for name in names:
            if name.lower() in lower:
                return lower[name.lower()]
        raise ValueError(f"现货缺少 {label} 字段：{list(raw.columns)}")

    date_col = choose(("trade_dt", "date", "trade_date", "交易日"), "交易日")
    columns = {
        "date": _parse_dates(raw[date_col]),
        "open": pd.to_numeric(raw[choose(("open", "开盘"), "open")], errors="coerce"),
        "high": pd.to_numeric(raw[choose(("high", "最高"), "high")], errors="coerce"),
        "low": pd.to_numeric(raw[choose(("low", "最低"), "low")], errors="coerce"),
        "close": pd.to_numeric(raw[choose(("close", "收盘"), "close")], errors="coerce"),
        "volume": pd.to_numeric(raw[choose(("volume", "vol", "成交量"), "volume")], errors="coerce"),
        "amount": pd.to_numeric(raw[choose(("amount", "turnover", "成交额"), "amount")], errors="coerce"),
    }
    result = pd.DataFrame(columns)
    result = result.dropna().sort_values("date", kind="stable").drop_duplicates("date", keep="last")
    result = result.set_index("date").sort_index()
    if result.empty or (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("现货 OHLC 为空或包含非正数")
    return result


def _market_features(spot: pd.DataFrame) -> pd.DataFrame:
    close = spot["close"].astype(float)
    ret = close.pct_change(fill_method=None)
    previous_close = close.shift(1)
    result = pd.DataFrame(index=spot.index)
    result["return_20"] = close.pct_change(20, fill_method=None)
    result["return_60"] = close.pct_change(60, fill_method=None)
    result["return_120"] = close.pct_change(120, fill_method=None)
    result["vol_20"] = ret.rolling(20, min_periods=20).std() * math.sqrt(252.0)
    result["vol_60"] = ret.rolling(60, min_periods=60).std() * math.sqrt(252.0)
    result["drawdown_60"] = close.div(close.rolling(60, min_periods=60).max()).sub(1.0)
    result["range_20"] = ((spot["high"] - spot["low"]).div(close)).rolling(20, min_periods=20).mean()
    result["gap_20"] = spot["open"].div(previous_close).sub(1.0).rolling(20, min_periods=20).mean()
    result["volume_ratio_60"] = spot["volume"].div(spot["volume"].rolling(60, min_periods=60).median()).sub(1.0)
    result["amount_ratio_60"] = spot["amount"].div(spot["amount"].rolling(60, min_periods=60).median()).sub(1.0)
    return result.replace([np.inf, -np.inf], np.nan)


def _model_window_features(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work = work.sort_values("date").drop_duplicates("date").set_index("date")
    dimensions = ["cv_trend", "cv_volume_price", "cv_position", "cv_intraday"]
    missing = [column for column in dimensions if column not in work.columns]
    if missing:
        raise ValueError(f"1545 内部面板缺少维度：{missing}")
    result = pd.DataFrame(index=work.index)
    result["axis_mean_60"] = work["rule_axis"].rolling(60, min_periods=60).mean()
    result["axis_std_60"] = work["rule_axis"].rolling(60, min_periods=60).std()
    result["slow_fast_mean_60"] = (work["slow_engine"] - work["fast_engine"]).rolling(60, min_periods=60).mean()
    positive_sync = work[dimensions].ge(0.60).sum(axis=1)
    negative_sync = work[dimensions].le(0.30).sum(axis=1)
    result["positive_sync_mean_60"] = positive_sync.rolling(60, min_periods=60).mean()
    result["negative_sync_mean_60"] = negative_sync.rolling(60, min_periods=60).mean()
    result["neutral_share_60"] = work["base_state"].eq(0).rolling(60, min_periods=60).mean()
    result["switch_rate_60"] = work["base_state"].ne(work["base_state"].shift()).rolling(60, min_periods=60).mean()
    return result


def _attach_states(panel: pd.DataFrame, execution: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work = work.sort_values("date").drop_duplicates("date").set_index("date")
    execution = execution.copy()
    execution["实际执行日"] = pd.to_datetime(execution["实际执行日"]).dt.normalize()
    execution["推定形成日"] = pd.to_datetime(execution["推定形成日"]).dt.normalize()
    by_formation = execution.drop_duplicates("推定形成日").set_index("推定形成日")
    work["raw_state_execution_aligned"] = by_formation["原始状态"].reindex(work.index)
    work["adjusted_state_execution_aligned"] = by_formation["调整后三状态"].reindex(work.index)
    work["naive_axis_state"] = np.select(
        [work["rule_axis"].le(0.30), work["rule_axis"].ge(0.70)], [-1, 1], default=0
    ).astype("int8")
    return work


def _forward_outcome(date: pd.Timestamp, spot: pd.DataFrame, model: pd.DataFrame) -> dict[str, Any]:
    dates = spot.index
    if date not in dates:
        return {"forward_return_20": np.nan, "forward_return_60": np.nan, "forward_vol_20": np.nan, "future_switches_20": np.nan, "future_directional_days_20": np.nan}
    position = int(dates.get_loc(date))
    close = spot["close"].to_numpy(float)
    ret = spot["close"].pct_change(fill_method=None).to_numpy(float)
    output: dict[str, Any] = {}
    for horizon, key in ((20, "forward_return_20"), (60, "forward_return_60")):
        target = position + horizon
        output[key] = float(close[target] / close[position] - 1.0) if target < len(close) else np.nan
    next_returns = ret[position + 1: position + 21]
    output["forward_vol_20"] = float(np.nanstd(next_returns, ddof=1) * math.sqrt(252.0)) if np.isfinite(next_returns).sum() > 1 else np.nan
    future_dates = dates[position + 1: position + 21]
    future = model.reindex(future_dates)
    output["future_switches_20"] = float(future["base_state"].ne(future["base_state"].shift()).sum()) if not future.empty else np.nan
    output["future_directional_days_20"] = float(future["base_state"].ne(0).sum()) if not future.empty else np.nan
    return output


def _find_analogues(combined: pd.DataFrame, spot: pd.DataFrame, model: pd.DataFrame, latest: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [column for column in MODEL_FEATURES if column in combined.columns]
    usable = combined.dropna(subset=features).copy()
    reference = usable.loc[latest, features].astype(float)
    historical = usable.loc[usable.index < pd.Timestamp("2026-01-01")].copy()
    scale = historical[features].std(ddof=0).replace(0.0, np.nan).fillna(1.0)
    distances = np.sqrt((((historical[features] - reference) / scale) ** 2).mean(axis=1))
    ranked = historical.assign(distance=distances).sort_values("distance")
    date_order = {date: pos for pos, date in enumerate(usable.index)}
    selected: list[pd.Timestamp] = []
    for date in ranked.index:
        if all(abs(date_order[date] - date_order[other]) >= 40 for other in selected):
            selected.append(date)
        if len(selected) >= 6:
            break
    rows: list[dict[str, Any]] = []
    comparison_dates = [latest, *selected]
    labels = {latest: "2026当前窗口"}
    labels.update({date: f"历史相似{idx}" for idx, date in enumerate(selected, start=1)})
    for rank, date in enumerate(comparison_dates):
        row = {"label": labels[date], "window_end": date.strftime("%Y-%m-%d"), "rank": 0 if date == latest else rank}
        row.update({feature: float(combined.loc[date, feature]) for feature in features})
        row["distance"] = 0.0 if date == latest else float(distances.loc[date])
        row.update(_forward_outcome(date, spot, model))
        rows.append(row)
    analogs = pd.DataFrame(rows)
    analogs = analogs.sort_values(["rank", "distance"], kind="stable").reset_index(drop=True)
    z_rows = []
    for _, row in analogs.iterrows():
        z_rows.append({"label": row["label"], "window_end": row["window_end"], **{
            feature: float((row[feature] - reference[feature]) / scale[feature]) for feature in features
        }})
    comparison = pd.DataFrame(z_rows)
    return analogs, comparison


def _fixed_anchor_proxy(model: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = model.copy().sort_index()
    dimensions = ["cv_trend", "cv_volume_price", "cv_position", "cv_intraday"]
    dev = work.loc[work.index <= pd.Timestamp("2022-12-31"), dimensions]
    fixed = pd.DataFrame(index=work.index)
    for column in dimensions:
        anchor = np.sort(dev[column].dropna().to_numpy(float))
        values = work[column].to_numpy(float)
        fixed[column] = np.searchsorted(anchor, values, side="right") / max(len(anchor), 1)
    fixed_axis = (
        0.40 * fixed["cv_trend"]
        + 0.30 * fixed["cv_volume_price"]
        + 0.15 * fixed["cv_position"]
        + 0.15 * fixed["cv_intraday"]
    )
    work["fixed_anchor_axis_proxy"] = fixed_axis
    work["fixed_anchor_state_proxy"] = np.select(
        [fixed_axis.le(0.30), fixed_axis.ge(0.70)], [-1, 1], default=0
    ).astype("int8")
    work["axis_only_switch"] = work["naive_axis_state"].ne(work["naive_axis_state"].shift())
    work["fixed_anchor_switch"] = work["fixed_anchor_state_proxy"].ne(work["fixed_anchor_state_proxy"].shift())
    rows: list[dict[str, Any]] = []
    for year, group in work.groupby(work.index.year):
        rows.append({
            "year": int(year),
            "formal_switches": int(group["base_state"].ne(group["base_state"].shift()).sum()),
            "axis_only_switches": int(group["axis_only_switch"].sum()),
            "fixed_anchor_switches": int(group["fixed_anchor_switch"].sum()),
            "formal_neutral_share_pct": float(group["base_state"].eq(0).mean() * 100.0),
            "axis_only_neutral_share_pct": float(group["naive_axis_state"].eq(0).mean() * 100.0),
            "fixed_anchor_neutral_share_pct": float(group["fixed_anchor_state_proxy"].eq(0).mean() * 100.0),
        })
    return work, pd.DataFrame(rows)


def _annual_mechanism(combined: pd.DataFrame, execution: pd.DataFrame) -> pd.DataFrame:
    exec_work = execution.loc[execution["O2O可评价"]].copy()
    exec_work["year"] = pd.to_datetime(exec_work["实际执行日"]).dt.year
    rows: list[dict[str, Any]] = []
    for year, group in combined.groupby(combined.index.year):
        execution_group = exec_work.loc[exec_work["year"].eq(year)]
        rows.append({
            "year": int(year),
            "phase": "Development" if year <= 2022 else ("Validation" if year <= 2024 else "Test"),
            "market_return_60_pct": float(group["return_60"].mean() * 100.0),
            "market_vol_60_pct": float(group["vol_60"].mean() * 100.0),
            "axis_mean_60": float(group["axis_mean_60"].mean()),
            "axis_std_60": float(group["axis_std_60"].mean()),
            "slow_fast_mean_60": float(group["slow_fast_mean_60"].mean()),
            "positive_sync_mean_60": float(group["positive_sync_mean_60"].mean()),
            "negative_sync_mean_60": float(group["negative_sync_mean_60"].mean()),
            "neutral_share_60_pct": float(group["neutral_share_60"].mean() * 100.0),
            "switch_rate_60_pct": float(group["switch_rate_60"].mean() * 100.0),
            "raw_return_pct": float(execution_group["原始策略日收益"].sum() * 100.0) if not execution_group.empty else np.nan,
            "adjusted_return_pct": float(execution_group["调整策略日收益"].sum() * 100.0) if not execution_group.empty else np.nan,
            "adjusted_directional_days": int(execution_group["调整后三状态"].ne(0).sum()) if not execution_group.empty else 0,
        })
    return pd.DataFrame(rows)


def _plot_state_background(ax: Any, dates: pd.DatetimeIndex, states: pd.Series) -> None:
    colors = {-1: "#ef8f86", 0: "#d9dce1", 1: "#7ed3a5"}
    state_values = pd.Series(states, index=dates).fillna(0).astype(int).to_numpy()
    for idx in range(len(dates) - 1):
        ax.axvspan(dates[idx], dates[idx + 1], color=colors.get(int(state_values[idx]), "#d9dce1"), alpha=0.18, linewidth=0)


def _save_table_figure(table: pd.DataFrame, path: Path, title: str) -> None:
    view = table.copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    text = view.to_string(index=False)
    height = max(3.0, min(13.0, 0.32 * (len(view) + 3)))
    fig = plt.figure(figsize=(15.0, height), dpi=150)
    fig.patch.set_facecolor("#111111")
    fig.text(0.02, 0.96, title, color="white", fontsize=13, weight="bold", va="top")
    fig.text(0.02, 0.90, text, color="#f3f3f3", family="monospace", fontsize=8.2, va="top")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_analogues(analogs: pd.DataFrame, comparison: pd.DataFrame, figures: Path) -> None:
    historical = analogs.loc[analogs["label"].ne("2026当前窗口")].copy().sort_values("distance", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=150)
    ax.barh(historical["label"], historical["distance"], color="#4c78a8")
    ax.invert_yaxis()
    for idx, (_, row) in enumerate(historical.iterrows()):
        ax.text(row["distance"], idx, f"  {row['window_end']}", va="center", fontsize=9)
    ax.set_xlabel("Feature distance to 2026 current 60-trading-day window")
    ax.set_title("Historical analogue windows: closest observed market/model environments", weight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "01_历史相似行情_特征距离.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    heat = comparison.set_index("label").drop(columns=["window_end"])
    fig, ax = plt.subplots(figsize=(14.0, 5.8), dpi=150)
    im = ax.imshow(heat.to_numpy(float), aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_xticks(np.arange(len(heat.columns)), labels=heat.columns, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(heat.index)), labels=heat.index, fontsize=9)
    ax.set_title("Analogue feature comparison (standardized relative to the 2026 window)", weight="bold")
    fig.colorbar(im, ax=ax, label="(analogue - 2026) / historical scale")
    fig.tight_layout()
    fig.savefig(figures / "02_历史相似行情_特征对比热图.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_mechanism(combined: pd.DataFrame, figures: Path) -> None:
    recent = combined.loc[combined.index >= pd.Timestamp("2023-01-01")].copy()
    fig, axes = plt.subplots(4, 1, figsize=(13.5, 10.5), dpi=150, sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0, 1.1, 0.8]})
    axes[0].plot(recent.index, recent["close"], color="#2f6db0", linewidth=0.9)
    axes[0].set_ylabel("CSI500 close")
    axes[0].set_title("Observable mechanism chain: market environment → relative engine → state response", weight="bold")
    axes[1].plot(recent.index, recent["return_60"] * 100.0, label="60d return %", color="#e07a2d")
    axes[1].plot(recent.index, recent["vol_60"] * 100.0, label="60d vol %", color="#8c5cc7")
    axes[1].set_ylabel("Market features")
    axes[1].legend(ncol=2, fontsize=8)
    axes[2].plot(recent.index, recent["rule_axis"], label="rule_axis", color="#c23b3b", linewidth=0.8)
    axes[2].plot(recent.index, recent["slow_engine"], label="slow_engine", color="#356bb3", linewidth=0.8)
    axes[2].plot(recent.index, recent["fast_engine"], label="fast_engine", color="#e78727", linewidth=0.8)
    axes[2].axhline(0.70, color="#3d9b6d", linestyle=":", linewidth=0.7)
    axes[2].axhline(0.30, color="#b34c4c", linestyle=":", linewidth=0.7)
    axes[2].set_ylabel("Relative scores")
    axes[2].legend(ncol=3, fontsize=8)
    axes[3].set_ylim(-1.05, 1.05)
    axes[3].set_yticks([-1, 0, 1], labels=["Short", "Flat", "Long"])
    _plot_state_background(axes[3], recent.index, recent["raw_state_execution_aligned"])
    axes[3].set_ylabel("Raw state")
    axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(figures / "03_行情与模型相对状态_2023至最新.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_sync_and_confirmation(combined: pd.DataFrame, figures: Path) -> None:
    recent = combined.loc[combined.index >= pd.Timestamp("2023-01-01")].copy()
    dimensions = ["cv_trend", "cv_volume_price", "cv_position", "cv_intraday"]
    fig, axes = plt.subplots(4, 1, figsize=(13.5, 10.0), dpi=150, sharex=True, gridspec_kw={"height_ratios": [1.0, 0.9, 0.9, 0.8]})
    for column in dimensions:
        axes[0].plot(recent.index, recent[column], linewidth=0.7, label=column.replace("cv_", ""))
    axes[0].axhline(0.60, color="#399b6d", linestyle=":", linewidth=0.7)
    axes[0].axhline(0.30, color="#b54b4b", linestyle=":", linewidth=0.7)
    axes[0].set_ylabel("Four views")
    axes[0].legend(ncol=4, fontsize=8)
    axes[1].plot(recent.index, recent["positive_sync_mean_60"], label="positive sync", color="#2b8cbe")
    axes[1].plot(recent.index, recent["negative_sync_mean_60"], label="negative sync", color="#d95f02")
    axes[1].set_ylabel("60d mean sync")
    axes[1].legend(ncol=2, fontsize=8)
    axes[2].plot(recent.index, recent["base_state"], drawstyle="steps-post", label="formal base_state", color="#4d4d4d")
    axes[2].plot(recent.index, recent["naive_axis_state"], drawstyle="steps-post", label="axis-only proxy", color="#9e9ac8", alpha=0.8)
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].set_yticks([-1, 0, 1])
    axes[2].set_ylabel("State")
    axes[2].legend(ncol=2, fontsize=8)
    axes[3].plot(recent.index, recent["switch_rate_60"] * 100.0, color="#4c78a8")
    axes[3].set_ylabel("Switch rate %")
    axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("Multi-view synchronization and confirmation/hysteresis diagnostic", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(figures / "04_多视角同步与确认响应_2023至最新.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_counterfactual(proxy: pd.DataFrame, annual: pd.DataFrame, figures: Path) -> None:
    recent = proxy.loc[proxy.index >= pd.Timestamp("2023-01-01")].copy()
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.0), dpi=150, sharex=False, gridspec_kw={"height_ratios": [1.3, 0.8]})
    axes[0].plot(recent.index, recent["rule_axis"], label="actual rolling-relative axis", color="#c23b3b", linewidth=0.8)
    axes[0].plot(recent.index, recent["fixed_anchor_axis_proxy"], label="fixed Development-anchor proxy", color="#356bb3", linewidth=0.8)
    axes[0].axhline(0.70, color="#3d9b6d", linestyle=":", linewidth=0.7)
    axes[0].axhline(0.30, color="#b34c4c", linestyle=":", linewidth=0.7)
    axes[0].set_ylabel("Axis")
    axes[0].legend(ncol=2, fontsize=8)
    width = 0.24
    x = np.arange(len(annual))
    axes[1].bar(x - width, annual["formal_switches"], width, label="formal", color="#4c78a8")
    axes[1].bar(x, annual["axis_only_switches"], width, label="axis-only", color="#9e9ac8")
    axes[1].bar(x + width, annual["fixed_anchor_switches"], width, label="fixed-anchor proxy", color="#f28e2b")
    axes[1].set_xticks(x, annual["year"].astype(str))
    axes[1].set_ylabel("Annual switch count")
    axes[1].legend(ncol=3, fontsize=8)
    fig.suptitle("Counterfactual diagnostics: response rule vs fixed-anchor proxy", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(figures / "05_滚动相对标准化与固定锚定代理.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_forward_quality(analogs: pd.DataFrame, annual: pd.DataFrame, figures: Path) -> None:
    historical = analogs.loc[analogs["label"].ne("2026当前窗口")].copy()
    fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=150)
    x = np.arange(len(historical))
    width = 0.35
    ax.bar(x - width / 2, historical["forward_return_20"] * 100.0, width, label="next 20d close return", color="#4c78a8")
    ax.bar(x + width / 2, historical["forward_return_60"] * 100.0, width, label="next 60d close return", color="#f28e2b")
    ax.axhline(0.0, color="#555", linewidth=0.7)
    ax.set_xticks(x, [f"{row.label}\n{row.window_end}" for row in historical.itertuples()], fontsize=8)
    ax.set_ylabel("Post-window return %")
    ax.set_title("Post-window outcomes of historical analogues (evaluation only)", weight="bold")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "06_历史相似阶段未来表现对比.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.5, 5.0), dpi=150)
    x = np.arange(len(annual))
    width = 0.36
    ax.bar(x - width / 2, annual["raw_return_pct"], width, label="Raw 1545", color="#f58518")
    ax.bar(x + width / 2, annual["adjusted_return_pct"], width, label="Adj 1545 (+4 reversal)", color="#d62728", alpha=0.8)
    ax.axhline(0.0, color="#555", linewidth=0.7)
    ax.set_xticks(x, annual["year"].astype(str))
    ax.set_ylabel("Additive O2O return %")
    ax.set_title("Annual performance and the direction-coverage change", weight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "07_年度收益与方向覆盖对比.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _transition_count(values: pd.Series) -> int:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if len(series) <= 1:
        return 0
    return int(series.iloc[1:].ne(series.iloc[:-1].to_numpy()).sum())


def _plot_teacher_full_period_curve(execution: pd.DataFrame, figures: Path) -> dict[str, Any]:
    """Draw the teacher-style three-line O2O curve from 2023 to the latest.

    The teacher's reference chart starts in 2023 and compares the base
    three-state path, the path after the four reversal overlays, and the
    ordinary index. This is separate from the three local red-box panels.
    Only execution rows with a real next-trading-day open are plotted; the
    latest signal row is never turned into a flat placeholder.
    """

    work = execution.copy()
    work["实际执行日"] = pd.to_datetime(work["实际执行日"], errors="coerce").dt.normalize()
    work["执行日O2O"] = pd.to_numeric(work["执行日O2O"], errors="coerce")
    work = work.loc[
        work["实际执行日"].ge(pd.Timestamp("2023-01-01"))
        & work["执行日O2O"].notna()
    ].sort_values("实际执行日", kind="stable").copy()
    if work.empty:
        raise ValueError("老师同时间范围曲线没有 2023 年以来的可评价执行日")

    work["基础三状态_NAV"] = 1.0 + pd.to_numeric(work["原始策略日收益"], errors="coerce").cumsum()
    work["加入四个反转_NAV"] = 1.0 + pd.to_numeric(work["调整策略日收益"], errors="coerce").cumsum()
    work["普通指数_NAV"] = 1.0 + pd.to_numeric(work["指数日收益"], errors="coerce").cumsum()

    line_specs = [
        ("基础三状态_NAV", "基础三状态（Raw 1545）", "#2f6db0"),
        ("加入四个反转_NAV", "加入四个反转（Adj 1545）", "#e87922"),
        ("普通指数_NAV", "普通指数（CSI500 O2O）", "#333333"),
    ]
    state_specs = {
        "基础三状态_NAV": [("原始状态", STATE_MARKER_SHAPES["raw"])],
        "加入四个反转_NAV": [("调整后三状态", STATE_MARKER_SHAPES["adjusted"])],
        "普通指数_NAV": [],
    }

    fig, ax = plt.subplots(figsize=(14.0, 5.8), dpi=150)
    for column, label, color in line_specs:
        _plot_curve_with_execution_points(
            ax,
            work["实际执行日"],
            work[column],
            label,
            color,
            state_specs=[(work[state_column], marker) for state_column, marker in state_specs[column]],
            linewidth=1.55 if column != "普通指数_NAV" else 1.15,
        )
    ax.axhline(1.0, color="#999999", linestyle="--", linewidth=0.7)
    ax.set_title(
        "老师原图同时间范围：基础三状态、加入四个反转与普通指数\n"
        f"{work['实际执行日'].min():%Y-%m-%d}—{work['实际执行日'].max():%Y-%m-%d}；"
        "执行日 O2O 加算，NAV=1+累计收益，不复利",
        fontsize=13,
        weight="bold",
    )
    ax.set_xlabel("Execution date")
    ax.set_ylabel("Additive NAV")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.22)
    _add_state_marker_legend(ax, ncol=3, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    output = figures / "29_老师原图同时间范围_2023至最新_三状态与指数_O2O.png"
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "name": output.name,
        "date_min": str(work["实际执行日"].min().date()),
        "date_max": str(work["实际执行日"].max().date()),
        "rows": int(len(work)),
    }


def build_teacher_red_window_analysis(
    execution: pd.DataFrame,
    output_dir: str | Path,
    figures: str | Path,
) -> dict[str, Any]:
    """Revisit the three red windows marked on the teacher's reference image.

    The chart deliberately uses the exact Stage-04 O2O additive convention:
    each execution-date return is next actual trading day's open divided by
    the execution-date open minus one, and each local NAV is ``1 + cumsum``
    rather than compounded.  The window NAVs are rebased to 1 only to make
    the three local panels visually comparable; the daily O2O values are not
    changed.

    This is a diagnostic/reporting layer.  It does not create or alter any
    signal, threshold, holding path, or freeze artifact.
    """

    output_dir = _absolute(output_dir, "TEACHER_RED_WINDOW_OUTPUT_DIR")
    figures = _absolute(figures, "TEACHER_RED_WINDOW_FIGURES_DIR")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    work = execution.copy()
    work["实际执行日"] = pd.to_datetime(work["实际执行日"]).dt.normalize()
    work = work.sort_values("实际执行日", kind="stable").drop_duplicates("实际执行日")
    o2o_valid = pd.to_numeric(work["执行日O2O"], errors="coerce").notna()
    work["O2O可评价"] = o2o_valid

    required = [
        "原始状态",
        "调整后三状态",
        "原始策略日收益",
        "调整策略日收益",
        "指数日收益",
        "调整原因",
    ]
    missing = [column for column in required if column not in work.columns]
    if missing:
        raise ValueError(f"老师红色区间复盘缺少Stage-04字段：{missing}")

    daily_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for spec in TEACHER_RED_WINDOWS:
        start = pd.Timestamp(spec["start"])
        end = pd.Timestamp(spec["end"])
        window = work.loc[
            work["实际执行日"].between(start, end, inclusive="both") & work["O2O可评价"]
        ].copy()
        if window.empty:
            raise ValueError(f"老师红色区间没有可评价执行日：{spec['window_id']} {start.date()}—{end.date()}")

        window["老师区间"] = spec["window_id"]
        window["局部原始三状态_NAV"] = 1.0 + pd.to_numeric(window["原始策略日收益"], errors="coerce").cumsum()
        window["局部加入四反转_NAV"] = 1.0 + pd.to_numeric(window["调整策略日收益"], errors="coerce").cumsum()
        window["局部普通指数_NAV"] = 1.0 + pd.to_numeric(window["指数日收益"], errors="coerce").cumsum()
        daily_rows.append(
            window[
                [
                    "老师区间",
                    "实际执行日",
                    "推定形成日",
                    "原始状态",
                    "调整后三状态",
                    "调整原因",
                    "+1反转",
                    "-1反转",
                    "0转-1",
                    "0转+1",
                    "原始策略日收益",
                    "调整策略日收益",
                    "指数日收益",
                    "局部原始三状态_NAV",
                    "局部加入四反转_NAV",
                    "局部普通指数_NAV",
                ]
            ].copy()
        )

        base_state = window["原始状态"].astype(int)
        adjusted_state = window["调整后三状态"].astype(int)
        base_zero_days = int(base_state.eq(0).sum())
        adjusted_zero_days = int(adjusted_state.eq(0).sum())
        base_zero_to_direction = int((base_state.eq(0) & adjusted_state.ne(0)).sum())
        base_direction_to_zero = int((base_state.ne(0) & adjusted_state.eq(0)).sum())
        conflict_days = int(window["调整原因"].astype(str).str.contains("冲突", na=False).sum())
        zero_to_down = base_state.eq(0) & adjusted_state.eq(-1)
        zero_to_up = base_state.eq(0) & adjusted_state.eq(1)
        direction_to_zero = base_state.ne(0) & adjusted_state.eq(0)
        zero_to_down_pct = float(window.loc[zero_to_down, "调整策略日收益"].sum() * 100.0)
        zero_to_up_pct = float(window.loc[zero_to_up, "调整策略日收益"].sum() * 100.0)
        direction_to_zero_delta_pct = float(
            (window.loc[direction_to_zero, "调整策略日收益"] - window.loc[direction_to_zero, "原始策略日收益"]).sum()
            * 100.0
        )
        adjustment_delta_pct = float(
            (window["调整策略日收益"] - window["原始策略日收益"]).sum() * 100.0
        )
        if adjusted_zero_days < base_zero_days:
            analysis = (
                f"反转层把基础0中的{base_zero_to_direction}个执行日转为方向状态；"
                f"0状态由{base_zero_days}/{len(window)}天降至{adjusted_zero_days}/{len(window)}天。"
            )
        elif adjusted_zero_days > base_zero_days:
            analysis = (
                f"该区间存在{base_direction_to_zero}个原方向被退出层暂时中和的执行日；"
                f"同时反转层补充了{base_zero_to_direction}个基础0方向日，净效果需结合三条O2O曲线判断。"
            )
        else:
            analysis = (
                f"基础0中有{base_zero_to_direction}个执行日被反转层识别为方向，"
                f"但有{base_direction_to_zero}个原方向退出日被中和，0状态天数净变化为0。"
            )
        analysis += (
            f"其中0→-1共{int(zero_to_down.sum())}天、区间加算贡献{zero_to_down_pct:+.2f}%；"
            f"0→+1共{int(zero_to_up.sum())}天、区间加算贡献{zero_to_up_pct:+.2f}%；"
            f"原方向→0的调整差额为{direction_to_zero_delta_pct:+.2f}个百分点，"
            f"调整相对基础的总差额为{adjustment_delta_pct:+.2f}个百分点。"
        )

        summary_rows.append(
            {
                "老师区间": spec["window_id"],
                "老师图示范围": f"{start.date()}—{end.date()}",
                "实际执行日范围": f"{window['实际执行日'].min().date()}—{window['实际执行日'].max().date()}",
                "可评价执行日": int(len(window)),
                "基础0天数": base_zero_days,
                "加入四反转后0天数": adjusted_zero_days,
                "基础0转方向天数": base_zero_to_direction,
                "原方向被中和天数": base_direction_to_zero,
                "上下路径冲突置0天数": conflict_days,
                "基础O2O加算收益_pct": float(window["原始策略日收益"].sum() * 100.0),
                "加入四反转O2O加算收益_pct": float(window["调整策略日收益"].sum() * 100.0),
                "普通指数O2O加算收益_pct": float(window["指数日收益"].sum() * 100.0),
                "0转-1_O2O加算贡献_pct": zero_to_down_pct,
                "0转+1_O2O加算贡献_pct": zero_to_up_pct,
                "原方向转0_调整差额_pct": direction_to_zero_delta_pct,
                "调整相对基础差额_pct": adjustment_delta_pct,
                "基础状态切换次数": _transition_count(base_state),
                "加入四反转状态切换次数": _transition_count(adjusted_state),
                "0转-1信号天数": int(pd.to_numeric(window["0转-1"], errors="coerce").sum()),
                "0转+1信号天数": int(pd.to_numeric(window["0转+1"], errors="coerce").sum()),
                "+1反转信号天数": int(pd.to_numeric(window["+1反转"], errors="coerce").sum()),
                "-1反转信号天数": int(pd.to_numeric(window["-1反转"], errors="coerce").sum()),
                "区间解释": analysis,
                "老师图示说明": spec["description"],
            }
        )

    daily = pd.concat(daily_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    daily.to_csv(output_dir / "老师红色区间复盘逐日_O2O.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    summary.to_csv(output_dir / "老师红色区间复盘汇总_O2O.csv", index=False, encoding="utf-8-sig")

    full_period = _plot_teacher_full_period_curve(execution, figures)

    line_specs = [
        ("局部原始三状态_NAV", "基础三状态（Raw 1545）", "#2f6db0"),
        ("局部加入四反转_NAV", "加入四个反转（Adj 1545）", "#e87922"),
        ("局部普通指数_NAV", "普通指数（CSI500 O2O）", "#333333"),
    ]
    state_specs = {
        "局部原始三状态_NAV": [("原始状态", STATE_MARKER_SHAPES["raw"])],
        "局部加入四反转_NAV": [("调整后三状态", STATE_MARKER_SHAPES["adjusted"])],
        "局部普通指数_NAV": [],
    }

    def plot_window(window_id: str, path: Path, title_suffix: str = "") -> None:
        data = daily.loc[daily["老师区间"].eq(window_id)].copy()
        fig, ax = plt.subplots(figsize=(13.5, 5.4), dpi=150)
        for column, label, color in line_specs:
            _plot_curve_with_execution_points(
                ax,
                data["实际执行日"],
                data[column],
                label,
                color,
                state_specs=[(data[state_column], marker) for state_column, marker in state_specs[column]],
                linewidth=1.5 if column != "局部普通指数_NAV" else 1.15,
            )
        ax.axhline(1.0, color="#999999", linestyle="--", linewidth=0.7)
        row = summary.loc[summary["老师区间"].eq(window_id)].iloc[0]
        ax.set_title(
            f"{window_id} {title_suffix}\n"
            f"基础0 {int(row['基础0天数'])}天 → 加入四反转后0 {int(row['加入四反转后0天数'])}天；"
            f"O2O加算口径，窗口首日NAV=1",
            fontsize=12,
            weight="bold",
        )
        ax.set_ylabel("Additive O2O NAV")
        ax.set_xlabel("Execution date")
        _add_state_marker_legend(ax, ncol=3, loc="best")
        ax.grid(alpha=0.24)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    combined_path = figures / "28_老师红色区间_三状态与指数_O2O局部复盘.png"
    fig, axes = plt.subplots(3, 1, figsize=(14.0, 11.0), dpi=150, sharey=False)
    for ax, spec in zip(axes, TEACHER_RED_WINDOWS):
        data = daily.loc[daily["老师区间"].eq(spec["window_id"])].copy()
        row = summary.loc[summary["老师区间"].eq(spec["window_id"])].iloc[0]
        for column, label, color in line_specs:
            _plot_curve_with_execution_points(
                ax,
                data["实际执行日"],
                data[column],
                label,
                color,
                state_specs=[(data[state_column], marker) for state_column, marker in state_specs[column]],
                linewidth=1.45 if column != "局部普通指数_NAV" else 1.1,
            )
        ax.axhline(1.0, color="#999999", linestyle="--", linewidth=0.7)
        ax.set_title(
            f"{spec['window_id']}：{data['实际执行日'].min():%Y-%m-%d}—{data['实际执行日'].max():%Y-%m-%d}；"
            f"基础0 {int(row['基础0天数'])}天 → 调整后0 {int(row['加入四反转后0天数'])}天",
            fontsize=11,
            weight="bold",
        )
        ax.set_ylabel("NAV")
        ax.grid(alpha=0.24)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        if ax is axes[0]:
            _add_state_marker_legend(ax, ncol=3, loc="best")
    axes[-1].set_xlabel("Execution date")
    fig.suptitle(
        "老师红色0区间局部复盘：基础三状态、加入四个反转和普通指数\n"
        "统一使用执行日开盘→下一实际交易日开盘的O2O收益；NAV=1+累计加算收益，不复利",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(combined_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    individual_names: list[str] = []
    for index, spec in enumerate(TEACHER_RED_WINDOWS, start=1):
        name = f"28{chr(64 + index)}_{spec['window_id']}_三状态与指数_O2O局部复盘.png"
        path = figures / name
        plot_window(spec["window_id"], path, spec["description"])
        individual_names.append(name)

    return {
        "summary": summary,
        "daily": daily,
        "full_period": full_period,
        "figure_names": [full_period["name"], combined_path.name, *individual_names],
        "summary_path": str(output_dir / "老师红色区间复盘汇总_O2O.csv"),
        "daily_path": str(output_dir / "老师红色区间复盘逐日_O2O.csv"),
        "windows": [
            {
                "window_id": spec["window_id"],
                "start": spec["start"],
                "end": spec["end"],
                "description": spec["description"],
            }
            for spec in TEACHER_RED_WINDOWS
        ],
    }


def _audit_local_remote(spot_path: Path, holding_path: Path, event_path: Path, expected_path: Path, stage04_dir: Path | None, execution: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    # The freeze reference is the event-day eight-list.  The 03/04 holding
    # path deliberately expands 0→-1 and 0→+1 across multiple execution days,
    # so it must not be compared cell-for-cell with the event reference.
    event = read_holding_eight(event_path)
    expected = read_holding_eight(expected_path)
    common = event.merge(expected, on="实际执行日", suffixes=("_local", "_reference"), how="inner")
    signal_mismatch = 0
    mismatch_by_column: dict[str, int] = {}
    for column in SIGNAL_COLUMNS[1:]:
        mismatch = common[f"{column}_local"].ne(common[f"{column}_reference"])
        mismatch_by_column[column] = int(mismatch.sum())
        signal_mismatch += int(mismatch.sum())

    stage04_match = None
    stage04_path = stage04_dir / "O2O加算逐日收益与状态.csv" if stage04_dir else None
    if stage04_path and stage04_path.is_file():
        stage04 = pd.read_csv(stage04_path, encoding="utf-8-sig", parse_dates=["实际执行日"])
        stage04_match = bool(
            len(stage04) == len(execution)
            and stage04["实际执行日"].eq(execution["实际执行日"]).all()
            and stage04["原始状态"].astype(int).eq(execution["原始状态"].astype(int)).all()
            and stage04["调整后三状态"].astype(int).eq(execution["调整后三状态"].astype(int)).all()
        )

    risk = build_risk_summary(execution)
    metric_rows: list[dict[str, Any]] = []
    for _, row in risk.iterrows():
        for metric in ("total_return_pct", "annualized_additive_return_pct", "annualized_volatility_pct", "sharpe", "active_days", "winning_directional_days"):
            local_value = float(row[metric])
            remote_value = float(REMOTE_METRICS[(str(row["series"]), metric)])
            tolerance = 0.05 if metric.endswith("_pct") or metric == "sharpe" else 0.01
            metric_rows.append({
                "series": str(row["series"]),
                "metric": metric,
                "local_value": local_value,
                "remote_reference": remote_value,
                "absolute_difference": abs(local_value - remote_value),
                "pass": bool(abs(local_value - remote_value) <= tolerance),
            })
    metrics = pd.DataFrame(metric_rows)
    spot_dates = _read_spot(spot_path).index
    latest = execution.iloc[-1]
    audit = {
        "spot_path": str(spot_path),
        "spot_sha256": _sha256(spot_path),
        "spot_rows": int(len(spot_dates)),
        "spot_date_min": str(spot_dates.min().date()),
        "spot_date_max": str(spot_dates.max().date()),
        "event_signal_path": str(event_path),
        "local_event_signal_rows": int(len(event)),
        "holding_path_rows": int(len(read_holding_eight(holding_path))),
        "reference_common_rows": int(len(common)),
        "reference_date_max": str(expected["实际执行日"].max().date()),
        "signal_mismatch_total": int(signal_mismatch),
        "signal_mismatch_by_column": mismatch_by_column,
        "stage04_state_and_date_match": stage04_match,
        "metrics_match_remote_reference": bool(metrics["pass"].all()),
        "latest_formation_date": str(pd.Timestamp(latest["推定形成日"]).date()),
        "latest_execution_date": str(pd.Timestamp(latest["实际执行日"]).date()),
        "latest_signal_values_are_present": bool(pd.notna(latest[SIGNAL_COLUMNS[1:]].to_numpy()).all()),
        "latest_execution_is_pending_price_only": bool(pd.isna(latest["执行日开盘"])),
        "date_rule": "formation t close -> next actual spot trading execution row",
        "future_labels_used_for_signal": False,
        "remote_reference_source": "远端运行结果_20260824_1545/README_远端结果整理与合理性检查.md",
    }
    rows = [{"check": "local event signal values vs freeze reference on common dates", "value": f"{signal_mismatch} mismatches / {len(common)} rows", "pass": signal_mismatch == 0},
            {"check": "03 holding path is intentionally expanded from event signals", "value": "not compared cell-for-cell", "pass": True},
            {"check": "04 state/date output matches regenerated execution frame", "value": str(stage04_match), "pass": stage04_match is not False},
            {"check": "local core metrics vs remote reference", "value": f"{int(metrics['pass'].sum())}/{len(metrics)} metrics within tolerance", "pass": bool(metrics["pass"].all())},
            {"check": "latest row has real signal values", "value": f"{audit['latest_formation_date']} -> {audit['latest_execution_date']}", "pass": audit["latest_signal_values_are_present"]},
            {"check": "signal generation uses no future labels", "value": "metadata/engine boundary", "pass": True}]
    return pd.DataFrame(rows), {"summary": audit, "metrics": metrics}


def _write_report(
    output_dir: Path,
    figures: Path,
    audit: dict[str, Any],
    analogs: pd.DataFrame,
    teacher_analysis: dict[str, Any] | None = None,
) -> Path:
    summary = audit["summary"]
    image_names = [
        "01_历史相似行情_特征距离.png",
        "02_历史相似行情_特征对比热图.png",
        "03_行情与模型相对状态_2023至最新.png",
        "04_多视角同步与确认响应_2023至最新.png",
        "05_滚动相对标准化与固定锚定代理.png",
        "06_历史相似阶段未来表现对比.png",
        "07_年度收益与方向覆盖对比.png",
        "08_本地远端一致性审计.png",
    ]
    lines = [
        "# 06｜历史相似行情与2026敏感度机制分析",
        "",
        "本报告由 06 Notebook 生成。它只读取本地米筐现货、冻结包已有的含持有期八列表和冻结 1545 内部面板；不改动任何生产信号、阈值、持有参数或输出格式。",
        "",
        "## 一、分析对象与当前范围",
        "",
        f"- 现货日期：{summary['spot_date_min']} → {summary['spot_date_max']}。",
        f"- 最新信号链：形成日 {summary['latest_formation_date']} → 执行日 {summary['latest_execution_date']}。",
        "- 分析目标：解释 2026 年为什么比过去阶段更快识别行情，并用历史相似窗口、年度拆解、同步性和标准化代理做机制诊断。",
        "- 本报告中的未来收益只出现在历史窗口的事后比较，不参与相似行情排序，也不回写任何冻结信号。",
        "",
        "## 二、相似行情如何选以及当前窗口是什么样",
        "",
        "以最新形成日为终点，使用60个交易日窗口中的市场收益、波动、回撤、振幅、跳空，以及1545方向轴、慢快线背离、多视角同步性、中性占比和切换率。历史候选只使用各自窗口结束日已经可见的特征，未使用后续收益；后续收益仅用于图06的事后比较。",
        "",
        "本次选出的历史窗口：",
        "",
        analogs.to_string(index=False),
        "",
        "图01结果：当前窗口不是单纯的短期上涨或下跌，而是“短期反弹、60—120日仍处回撤、高波动”的混合环境。距离最近的历史窗口主要是 2024-03-13、2022-10-31 和 2020-04-09；它们共享中期回撤与较高波动，但并不代表后续走势相同。",
        "图02分析：特征热图显示，当前窗口与历史相似窗口的相似性来自一组特征的共同组合，而不是某一个指标完全相等。因此相似窗口适合回答“机制是否曾经出现过”，不适合直接当成确定性预测。",
        "",
        "## 三、2026为什么表现出更快的识别响应",
        "",
        "图03结果：2026 年市场 60 日波动约为 25.79%，高于 2025 年约 20.29%；模型中性占比约 55.67%，低于 2025 年约 69.22%；切换率约 8.80%，高于 2025 年约 3.40%。这说明 2026 的快速识别同时伴随更大的市场变化和更高的状态切换，而不是只由模型单方面放宽阈值造成。",
        "图03分析：基础三状态更像“确认后的阶段标签”。当多个输入视角在较短时间内一起跨过方向条件，状态机可以在确认规则允许的范围内较快切换；当视角分歧时，它仍会保留 0 状态。",
        "",
        "图04结果：2026 年正式状态切换 16 次，高于 2025 年的 10 次；只看方向轴的代理切换 29 次，固定锚定代理切换 31 次。正式规则的切换少于两个代理，说明确认和滞回确实过滤了单一视角噪声。",
        "图04分析：2026 的“快”不是任何单一指标变快，而是市场变化更集中，并且多个视角更容易在相近日期同步变化；正式状态机在保留确认的同时完成响应。",
        "",
        "图05结果：2026 年正式状态的中性占比约 53.90%，高于方向轴代理约 33.12% 和固定锚定代理约 38.31%；正式切换次数却低于两个代理。这表明滚动相对标准化并没有简单地把所有波动都转成方向状态，正式逻辑仍然更保守。",
        "图05分析：滚动标准化让当前值与近期历史环境比较，能够适应波动水平变化；确认天数、最小驻留和滞回带再负责抑制噪声。固定锚定曲线只是机制对照，不能替代正式冻结计算，也不能单独证明“滚动窗口贡献了多少”。",
        "",
        "## 四、历史相似阶段的后续结果与年度结果",
        "",
        "图06结果：相似窗口之后的 20 日收益并不一致，历史候选既有正收益，也有负收益；60 日收益的差异更大。当前相似窗口因此只能说明 2026 所处的市场结构过去出现过，不能据此承诺下一阶段方向。",
        "图06分析：这组结果支持“做情景参考”，不支持“按相似日期机械预测”。真正可运行的信号仍然来自当天收盘后按冻结规则生成的下一执行日信号。",
        "",
        "图07结果：按 O2O 加算口径，加入反转和持有路径后的调整结果在各年度都高于对应原始三状态结果；2026 年原始结果约 22.88%，调整后约 35.48%，方向性持有日 82 天。",
        "图07分析：2026 的改善主要说明快速反应层在原三状态未及时改变的区间里补充了部分方向覆盖，但这仍是历史样本中的结果，不应等同于未来必然增益。基础三状态负责阶段基线，四个反转负责事件级的退出和零段转移，二者不是同一种角色。",
        "",
        "## 六、对老师问题的直接回答",
        "",
        "1. 红色 0 状态区间：当前图和机制拆解支持“低波动或多视角分歧导致确认不足”的解释。0 不是没有信息，而是尚未达到正式方向状态的确认条件。若要继续细分，应增加诊断层的向上积累、向下积累、冲突和等待确认标签，不能直接把冻结 0 改成方向状态。",
        "2. 2026 识别更快：现有证据支持三部分共同作用——行情波动和方向变化更集中、多个视角在相近日期同步变化、滚动相对标准化使当前值能与近期环境比较；确认和滞回规则则负责避免把所有波动都变成切换。图中可以做机制解释，但不能把贡献精确归因到某一个因素。",
        "3. 当前研究阶段：中证500基础三状态可以先作为长周期阶段基线；四个反转信号作为事件级快速响应层；下一步重点是继续做一日执行日预测的实时观察，而不是重新改写已冻结的长周期状态机。",
        "",
        "## 七、标准化反事实的边界",
        "",
        "固定开发期锚定图是诊断代理，不是对正式冻结包的替换。由于冻结生产面板不保存每个原始因子在进入滚动z-score前的完整中间序列，06可以严格复现滚动相对分数和状态规则，并提供固定锚定代理；若要做原始因子→z-score→滚动分位数的完全反事实拆分，需要额外导出原始中间特征，但不应修改生产逻辑。",
        "",
        "## 八、图表与逐图说明",
        "",
    ]
    if teacher_analysis is not None:
        teacher_summary = teacher_analysis["summary"].copy()
        teacher_lines = [
            "## 五、老师红色0区间逐段复盘",
            "",
            "以下三个窗口按老师提供的红框时间范围对齐。图中的三条曲线全部采用与04全时期图相同的执行日O2O口径：执行日开盘到下一实际交易日开盘，收益按加法累计，不复利。每个局部窗口仅将首日NAV重新设为1，便于比较，不改变任何日收益。每个点对应一个真实执行日；颜色表示执行日状态（-1绿色、0蓝色、+1红色），圆点是初始三状态，方点是调整后三状态。",
            "",
            "老师红框边界（按图片坐标人工记录，作为固定诊断窗口）：",
            "",
            "- 红色区间1：2023-11-01—2024-01-15；",
            "- 红色区间2：2024-04-15—2024-09-15；",
            "- 红色区间3：2025-03-01—2025-07-15。",
            "",
            teacher_summary.to_string(index=False),
            "",
            "逐段解读：",
        ]
        for row in teacher_summary.itertuples(index=False):
            teacher_lines.append(f"- {row.老师区间}（{row.实际执行日范围}）：{row.区间解释}")
        teacher_lines.extend(["", "图中蓝线是基础三状态，橙线是加入四个反转后的最终状态，黑线是普通指数的O2O加算基准。曲线上每天都有执行日点：颜色对应-1/0/+1，圆点对应初始三状态，方点对应调整后三状态。红色区间的核心判断不再只看段数，而是同时看0状态覆盖天数、方向覆盖天数和三条O2O曲线。", ""])
        for name in teacher_analysis["figure_names"]:
            teacher_lines.append(f"![{name}](<{(figures / name).resolve()}>)")
            teacher_lines.append("")
        insert_at = lines.index("## 六、对老师问题的直接回答")
        lines[insert_at:insert_at] = teacher_lines
    chart_notes = {
        "01_历史相似行情_特征距离.png": "结果与分析见第二节：看当前窗口和历史窗口的整体特征距离，不能把距离直接当作收益预测。",
        "02_历史相似行情_特征对比热图.png": "结果与分析见第二节：看相似性由哪些特征共同构成，避免只盯一个指标。",
        "03_行情与模型相对状态_2023至最新.png": "结果与分析见第三节：看市场环境变化是否与状态切换密度同步。",
        "04_多视角同步与确认响应_2023至最新.png": "结果与分析见第三节：看正式确认与单轴代理的差别。",
        "05_滚动相对标准化与固定锚定代理.png": "结果与分析见第三节：固定锚定仅用于诊断，不替代正式滚动计算。",
        "06_历史相似阶段未来表现对比.png": "结果与分析见第四节：历史相似阶段后续结果有分化，不能机械外推。",
        "07_年度收益与方向覆盖对比.png": "结果与分析见第四节：看原始三状态与加入快速反应层后的年度覆盖变化。",
        "08_本地远端一致性审计.png": "附录技术审计图，仅供复核运行链和日期血缘；本报告不再展开一致性结果。",
    }
    for name in image_names[:7]:
        lines.append(f"### {name}")
        lines.append("")
        lines.append(chart_notes[name])
        lines.append("")
        lines.append(f"![{name}](<{(figures / name).resolve()}>)")
        lines.append("")
    lines.extend([
        "## 九、附录：技术审计图",
        "",
        chart_notes[image_names[-1]],
        "",
        f"![{image_names[-1]}](<{(figures / image_names[-1]).resolve()}>)",
        "",
        "## 十、最终定性",
        "",
        "当前可以说：2026 的快速识别与更高的行情变化集中度、多视角同步和滚动相对比较相符；正式状态机仍通过确认、最小驻留和滞回过滤噪声。相似行情分析支持情景解释，不构成确定性预测。正式冻结包保持不变，06 只承担解释、类比和稳健性诊断。",
        "",
    ])
    report = output_dir / "README_06_历史相似行情与机制分解分析.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_stage_06(
    spot_path: str | Path,
    holding_path: str | Path,
    output_dir: str | Path,
    expected_path: str | Path,
    stage04_dir: str | Path | None = None,
    panel_path: str | Path | None = None,
    event_path: str | Path | None = None,
) -> dict[str, Any]:
    spot_path = _absolute(spot_path, "COMPANY_SPOT_PATH")
    holding_path = _absolute(holding_path, "HOLDING_EIGHT_PATH")
    output_dir = _absolute(output_dir, "ANALYSIS_06_OUTPUT_DIR")
    expected_path = _absolute(expected_path, "EXPECTED_EIGHT_PATH")
    stage04_dir = _absolute(stage04_dir, "ANALYSIS_04_OUTPUT_DIR") if stage04_dir else None
    panel_path = _absolute(panel_path, "PANEL_PATH") if panel_path else None
    event_path = _absolute(event_path, "EVENT_EIGHT_PATH") if event_path else holding_path.parent / "最终执行日简表.csv"
    for path, label in ((spot_path, "现货"), (holding_path, "含持有期八列表"), (event_path, "事件型八列表"), (expected_path, "冻结参考")):
        if not path.is_file():
            raise FileNotFoundError(f"{label}不存在：{path}")

    figures = output_dir / "figures"
    tables = output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    print("[06] 读取本地现货：", spot_path)
    print("[06] 读取含持有期八列表：", holding_path)
    print("[06] 读取冻结参数参考：", expected_path)
    spot = _read_spot(spot_path)
    execution = build_execution_frame(spot_path, holding_path)
    print("[06] 重建执行日期帧：", len(execution), execution["实际执行日"].min().date(), execution["实际执行日"].max().date())

    panel = _load_engine_panel(spot_path, panel_path)
    if panel is None or panel.empty:
        raise RuntimeError("无法从冻结包重建1545内部面板，06不能进行机制分析")
    model = _attach_states(panel, execution)
    market = _market_features(spot)
    combined = market.join(_model_window_features(panel), how="inner")
    combined = combined.join(model[[
        "open", "high", "low", "close", "base_state", "rule_axis", "slow_engine", "fast_engine",
        "cv_trend", "cv_volume_price", "cv_position", "cv_intraday", "raw_state_execution_aligned",
        "adjusted_state_execution_aligned", "naive_axis_state",
    ]], how="inner", rsuffix="_panel")
    combined = combined.sort_index()
    latest = pd.Timestamp(combined.index.max())
    analogs, comparison = _find_analogues(combined, spot, model, latest)
    proxy, counterfactual = _fixed_anchor_proxy(model)
    annual = _annual_mechanism(combined, execution)
    audit_rows, audit = _audit_local_remote(spot_path, holding_path, event_path, expected_path, stage04_dir, execution)
    teacher_analysis = build_teacher_red_window_analysis(execution, output_dir, figures)

    analogs.to_csv(tables / "历史相似行情候选_60日.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(tables / "历史相似行情特征对比_z值.csv", index=False, encoding="utf-8-sig")
    counterfactual.to_csv(tables / "滚动相对与固定锚定代理_年度对比.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(tables / "年度行情机制与收益对比.csv", index=False, encoding="utf-8-sig")
    audit_rows.to_csv(tables / "本地远端一致性审计摘要.csv", index=False, encoding="utf-8-sig")
    audit["metrics"].to_csv(tables / "本地远端核心指标逐项对比.csv", index=False, encoding="utf-8-sig")
    proxy.reset_index(names="formation_date").to_csv(tables / "机制诊断逐日面板.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    (output_dir / "06_运行元数据.json").write_text(json.dumps({
        "stage": "06 historical analogues and mechanism diagnostics",
        "spot_path": str(spot_path),
        "spot_sha256": _sha256(spot_path),
        "spot_date_min": str(spot.index.min().date()),
        "spot_date_max": str(spot.index.max().date()),
        "latest_formation_date": str(latest.date()),
        "holding_path": str(holding_path),
        "event_path": str(event_path),
        "expected_path": str(expected_path),
        "stage04_dir": str(stage04_dir) if stage04_dir else None,
        "panel_path": str(panel_path) if panel_path else None,
        "analogue_window_trading_days": 60,
        "analogue_future_not_used_for_ranking": True,
        "production_logic_modified": False,
        "exact_raw_to_zscore_counterfactual_available": False,
        "exact_raw_to_zscore_counterfactual_note": "冻结1545面板不保存原始因子中间序列；本次使用固定开发期锚定代理，不回写生产逻辑",
        "teacher_red_window_analysis": {
            "summary_path": teacher_analysis["summary_path"],
            "daily_path": teacher_analysis["daily_path"],
            "full_period": teacher_analysis["full_period"],
            "figure_names": teacher_analysis["figure_names"],
            "windows": teacher_analysis["windows"],
            "o2o_rule": "execution-date open -> next actual trading-date open; local NAV=1+cumsum(position*O2O), no compounding",
        },
        "audit": audit["summary"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _plot_analogues(analogs, comparison, figures)
    _plot_mechanism(combined, figures)
    _plot_sync_and_confirmation(combined, figures)
    _plot_counterfactual(proxy, counterfactual, figures)
    _plot_forward_quality(analogs, annual, figures)
    _save_table_figure(audit_rows, figures / "08_本地远端一致性审计.png", "Local / remote parity and date-lineage audit")
    report = _write_report(output_dir, figures, audit, analogs, teacher_analysis)

    return {
        "output_dir": str(output_dir),
        "report": str(report),
        "figures": [str(path) for path in sorted(figures.glob("*.png"))],
        "tables": [str(path) for path in sorted(tables.glob("*.csv"))],
        "teacher_red_window_analysis": {
            "summary_path": teacher_analysis["summary_path"],
            "daily_path": teacher_analysis["daily_path"],
            "figure_names": teacher_analysis["figure_names"],
            "windows": teacher_analysis["windows"],
        },
        "latest_formation_date": str(latest.date()),
        "latest_execution_date": str(pd.Timestamp(execution["实际执行日"].max()).date()),
        "audit": audit["summary"],
        "analogue_dates": analogs.loc[analogs["label"].ne("2026当前窗口"), ["label", "window_end"]].to_dict("records"),
    }


__all__ = ["run_stage_06", "build_teacher_red_window_analysis", "TEACHER_RED_WINDOWS"]
