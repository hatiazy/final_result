from __future__ import annotations

import hashlib
import json
import os
import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .spot_eight_state import (
    EIGHT_STATE_COLUMNS,
    FULL_START,
    assign_eight_base_state,
    build_economic_features_eight,
    compute_eight_states,
)
from .spot_factor_pool import FACTOR_ENGINE_COLUMNS, build_factor_pool


DEV_END = pd.Timestamp("2022-12-31")
VALID_END = pd.Timestamp("2024-12-31")
# The frozen V55/V80 score lineage starts with the first trading row used by
# the 06 package.  This is an input-history boundary for causal rolling
# features, not a training-period boundary; Development still starts at
# 2018-01-01 below.
FROZEN_INPUT_START = pd.Timestamp("2007-01-15")
ALLOWED_EIGHT_STATES = tuple(EIGHT_STATE_COLUMNS)
DEFAULT_SPOT_PATTERN = "/home/hzy/cta/IC数据更新*最终固化版/现货最终版/CSI500_SPOT_md_eod_raw*最终版.parquet"
SEALED_BASE_EXCEPTION = "冻结1545只输出base_state；八状态预测器不读取期货/流动性状态"
FORBIDDEN_TOKENS = (
    "liquidity", "risk_strength", "risk_pressure", "futures", "open_interest", "oi_",
    "basis", "term_structure", "premium", "成交流动性", "期货", "持仓量", "基差",
    "期限结构", "升贴水", "期现价差",
)

POSITION, INTRADAY, TREND, VOLUME_PRICE, PATH, TAIL, REVERSAL, HEAT = EIGHT_STATE_COLUMNS

BASELINE_SPEC = {
    "family": "economic_role_ordered_three_state",
    "unified_id": "ERS_1545",
    "output_states": [-1, 0, 1],
    "axis_family": "trend_volume_led_axis",
    "representation": "continuous_band_70_30",
    "threshold_package": "strict",
    "negative_entry": 0.30,
    "negative_continue": 0.42,
    "positive_entry": 0.70,
    "positive_continue": 0.58,
    "downside_family": "persistent_dual_weakness",
    "positive_family": "trend_volume_positive",
    "neutral_band_package": "standard_neutral",
    "neutral_lower": 0.42,
    "neutral_upper": 0.58,
    "state_process_package": "five_day_hysteresis",
    "down_confirm_days": 2,
    "down_exit_confirm_days": 2,
    "minimum_downside_dwell": 5,
    "positive_confirm_days": 2,
    "positive_exit_confirm_days": 2,
    "minimum_positive_dwell": 5,
    "minimum_neutral_dwell": 2,
    "minimum_strong_dwell": 5,
    "shape_control_package": "baseline_shape_control",
    "tail_guard_mode": "disabled",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    raise ValueError(f"只支持 CSV/Parquet 现货输入: {path}")


def _resolve(raw: pd.DataFrame, configured: str, aliases: tuple[str, ...], role: str) -> str:
    if configured in raw.columns:
        return configured
    by_lower = {str(column).lower(): str(column) for column in raw.columns}
    found = next((by_lower[name.lower()] for name in aliases if name.lower() in by_lower), None)
    if found is None:
        raise ValueError(f"现货文件缺少 {role} 字段；候选={aliases}；实际={list(raw.columns)}")
    return found


def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}", na=False)
    dates = pd.to_datetime(values, errors="coerce")
    if compact.any():
        dates.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    return dates.dt.normalize()


def _read_spot(path: Path) -> pd.DataFrame:
    raw = _read(path)
    date_col = _resolve(raw, "trade_dt", ("trade_dt", "trade_date", "date", "交易日"), "交易日")
    open_col = _resolve(raw, "open", ("open", "开盘", "open_price"), "open")
    high_col = _resolve(raw, "high", ("high", "最高", "highestprice"), "high")
    low_col = _resolve(raw, "low", ("low", "最低", "lowestprice"), "low")
    close_col = _resolve(raw, "close", ("close", "收盘", "close_price"), "close")
    volume_col = _resolve(raw, "volume", ("volume", "vol", "成交量"), "volume")
    amount_col = _resolve(raw, "amount", ("amount", "turnover", "total_turnover", "amt", "成交额"), "amount")
    if "index_code" in raw.columns:
        codes = raw["index_code"].astype("string").str.upper().str.strip()
        preferred = codes.isin({"000905.SH", "000905.XSHG", "000905", "CSI500", "IC"})
        if preferred.any():
            raw = raw.loc[preferred].copy()
        elif codes.nunique(dropna=True) > 1:
            raise ValueError(
                "现货文件包含多个指数，但没有识别到 CSI500/000905 index_code；"
                f"实际值={sorted(codes.dropna().unique().tolist())}"
            )
    previous_col = _resolve(
        raw,
        "preclose",
        ("preclose", "prev_close", "previous_close", "pre_close", "昨收", "前收盘"),
        "preclose/昨收",
    ) if any(
        str(column).lower() in {"preclose", "prev_close", "previous_close", "pre_close", "昨收", "前收盘"}
        for column in raw.columns
    ) else None
    previous = raw[previous_col] if previous_col else None
    result = pd.DataFrame(
        {
            "date": _parse_dates(raw[date_col]),
            "open": pd.to_numeric(raw[open_col], errors="coerce"),
            "high": pd.to_numeric(raw[high_col], errors="coerce"),
            "low": pd.to_numeric(raw[low_col], errors="coerce"),
            "close": pd.to_numeric(raw[close_col], errors="coerce"),
            "volume": pd.to_numeric(raw[volume_col], errors="coerce"),
            "total_turnover": pd.to_numeric(raw[amount_col], errors="coerce"),
        }
    )
    result["prev_close"] = (
        pd.to_numeric(previous, errors="coerce").to_numpy()
        if previous is not None
        else result["close"].shift(1).to_numpy()
    )
    result = result.dropna().sort_values("date", kind="stable").drop_duplicates("date", keep="last")
    result = result.loc[result["date"] >= FROZEN_INPUT_START].copy()
    if result.empty or (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("现货没有有效且为正的 OHLC")
    if (result[["volume", "total_turnover"]] < 0).any().any():
        raise ValueError("现货成交量或成交额出现负数")
    return result.set_index("date").sort_index()


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_paths(start: Path | None = None) -> tuple[Path, dict[str, Path]]:
    """Resolve the single company input: one spot OHLCV/turnover file."""
    root = (start or _package_root()).resolve()
    env_name = "1545_SPOT_PATH"
    # The remote company runner has one sanctioned external dependency. An
    # environment override remains useful for local reproduction, but the
    # package works remotely with the supplied spot-only path by default.
    value = os.environ.get(env_name, DEFAULT_SPOT_PATTERN)
    expanded = os.path.expanduser(value)
    if not Path(expanded).is_absolute():
        raise ValueError(f"{env_name} 必须使用绝对路径：{value}")
    if any(token in expanded for token in "*?["):
        matches = [Path(item).resolve() for item in glob.glob(expanded) if Path(item).is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(
                "公司端现货通配路径必须唯一匹配一个文件；"
                f"pattern={value!r}, matches={[str(item) for item in matches]}"
            )
        path = matches[0]
    else:
        path = Path(expanded).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"公司端现货输入失效: {path}")
    return root, {"spot": path}


def _state_age(state: pd.Series) -> pd.Series:
    group = state.ne(state.shift()).cumsum()
    return state.groupby(group).cumcount().add(1).astype("int16")


def _distance_to_natural_switch(state: pd.Series) -> pd.Series:
    values = state.to_numpy()
    result = np.full(len(values), np.nan)
    next_change: int | None = None
    for position in range(len(values) - 2, -1, -1):
        if values[position + 1] != values[position]:
            next_change = position + 1
        if next_change is not None:
            result[position] = next_change - position
    return pd.Series(result, index=state.index, dtype=float)


def _actual_calendar_maps(index: pd.DatetimeIndex, spot: pd.DataFrame) -> dict[str, pd.Series]:
    calendar = pd.DatetimeIndex(spot.index).sort_values()
    positions = calendar.get_indexer(index)
    if (positions < 0).any():
        raise AssertionError("formation dates are not present in the spot trading calendar")

    def dates_at(offset: int) -> pd.Series:
        output = np.full(len(index), np.datetime64("NaT"), dtype="datetime64[ns]")
        valid = positions + offset < len(calendar)
        output[valid] = calendar.to_numpy()[positions[valid] + offset]
        return pd.Series(output, index=index)

    def values_at(column: str, offset: int) -> pd.Series:
        source = spot[column].to_numpy(float)
        output = np.full(len(index), np.nan)
        valid = positions + offset < len(calendar)
        output[valid] = source[positions[valid] + offset]
        return pd.Series(output, index=index)

    return {
        "effective_date": dates_at(1),
        "exit_h1_date": dates_at(2),
        "exit_h2_date": dates_at(3),
        "exit_h3_date": dates_at(4),
        "open_t1": values_at("open", 1),
        "open_t2": values_at("open", 2),
        "open_t3": values_at("open", 3),
        "open_t4": values_at("open", 4),
        "close_t1": values_at("close", 1),
    }


def _phase(index: pd.DatetimeIndex) -> pd.Series:
    values = np.where(index <= DEV_END, "Development", np.where(index <= VALID_END, "Validation", "Test"))
    return pd.Series(values, index=index, dtype="string")


def _lineage_manifest() -> dict[str, Any]:
    return {
        "input_sources": ["spot"],
        "allowed_state_fields": list(EIGHT_STATE_COLUMNS),
        "excluded_state_fields": ["成交流动性状态"],
        "allowed_lineage": {name: "spot" for name in EIGHT_STATE_COLUMNS},
        "external_state_file_read": False,
        "futures_file_read": False,
        "sealed_base_exception": SEALED_BASE_EXCEPTION,
    }


def build_panel(start: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    root, paths = load_paths(start)
    spot = _read_spot(paths["spot"])
    state_values, state_manifest = compute_eight_states(spot)
    factor_pool = build_factor_pool(spot)
    usable = factor_pool.dropna(subset=list(FACTOR_ENGINE_COLUMNS)).index
    common = state_values.index.intersection(spot.index).intersection(usable).sort_values()
    if len(common) < 100:
        raise ValueError(f"八状态/现货/冻结因子共同交易日不足: {len(common)}")
    state_frame = state_values.loc[common]
    frame = (
        state_frame[EIGHT_STATE_COLUMNS].add_prefix("cv_")
        .join(state_frame[[f"band_{name}" for name in EIGHT_STATE_COLUMNS]])
        .join(spot.loc[common, ["open", "high", "low", "close", "volume", "total_turnover"]])
        .join(factor_pool.loc[common])
        .sort_index()
    )
    maxima = {
        name: int(frame.loc[:DEV_END, f"band_{name}"].max())
        for name in EIGHT_STATE_COLUMNS
    }
    economic_features = build_economic_features_eight(frame, maxima)
    frozen_states, pending = assign_eight_base_state(economic_features)
    base_state = pd.to_numeric(frozen_states["four_state"], errors="raise").astype("int8")
    observed = sorted(base_state.unique().tolist())
    if not set(observed).issubset({-1, 0, 1}):
        raise AssertionError(f"冻结1545出现非法状态: {observed}")

    aliases = {
        "position": POSITION, "reversal": REVERSAL, "tail": TAIL, "trend": TREND,
        "path": PATH, "intraday": INTRADAY, "heat": HEAT, "volume_price": VOLUME_PRICE,
    }
    panel = frame[["open", "high", "low", "close", "volume", "total_turnover"]].copy()
    for alias, name in aliases.items():
        panel[f"cv_{alias}"] = frame[f"cv_{name}"].astype(float)
        panel[f"br_{alias}"] = frame[f"band_{name}"].astype(float).sub(1).div(max(1, maxima[name] - 1))

    continuous_axis = 0.40 * panel["cv_trend"] + 0.30 * panel["cv_volume_price"] + 0.15 * panel["cv_position"] + 0.15 * panel["cv_intraday"]
    band_axis = 0.40 * panel["br_trend"] + 0.30 * panel["br_volume_price"] + 0.15 * panel["br_position"] + 0.15 * panel["br_intraday"]
    panel["rule_axis_continuous"] = continuous_axis
    panel["rule_axis_band"] = band_axis
    panel["rule_axis"] = 0.70 * continuous_axis + 0.30 * band_axis
    panel["slow_engine"] = (panel["cv_trend"] + panel["cv_volume_price"]) / 2.0
    panel["fast_engine"] = (panel["cv_position"] + panel["cv_intraday"]) / 2.0
    panel["base_state"] = base_state
    panel["state_age"] = _state_age(base_state)
    panel["phase"] = _phase(panel.index)
    panel["distance_to_natural_switch"] = _distance_to_natural_switch(base_state)

    for name, values in _actual_calendar_maps(panel.index, spot).items():
        panel[name] = values
    panel["o2o_h1"] = panel["open_t2"].div(panel["open_t1"]).sub(1)
    panel["o2o_h2"] = panel["open_t3"].div(panel["open_t1"]).sub(1)
    panel["o2o_h3"] = panel["open_t4"].div(panel["open_t1"]).sub(1)
    panel["c2c_obs"] = panel["close_t1"].div(panel["close"]).sub(1)

    predictor_columns = [
        column for column in panel.columns
        if column.startswith(("cv_", "br_", "rule_axis", "slow_engine", "fast_engine"))
    ] + ["open", "high", "low", "close", "volume", "total_turnover", "state_age"]
    forbidden_hits = sorted({
        token for column in predictor_columns for token in FORBIDDEN_TOKENS
        if token.lower() in column.lower()
    })
    if forbidden_hits:
        raise AssertionError(f"反转预测特征命中禁用字段: {forbidden_hits}")

    manifest: dict[str, Any] = {
        "audit_passed": True,
        "workspace": str(root),
        "data_sources": ["spot"],
        "spot": {
            "configured_path": os.environ.get("1545_SPOT_PATH", DEFAULT_SPOT_PATTERN),
            "path": str(paths["spot"]),
            "sha256": _sha256(paths["spot"]),
            "rows": int(len(spot)),
            "date_min": str(spot.index.min().date()),
            "date_max": str(spot.index.max().date()),
            "frozen_input_start": FROZEN_INPUT_START.strftime("%Y-%m-%d"),
            "columns_used": ["index_code", "trade_dt", "preclose", "open", "high", "low", "close", "volume", "amount"],
            "columns_ignored": ["crncy_code", "change", "pctchange", "data_source", "month"],
        },
        "eight_state": state_manifest,
        "lineage": _lineage_manifest(),
        "frozen_1545": {
            "source": "spot_eight_state_recompute",
            "spec": BASELINE_SPEC,
            "rows": int(len(base_state)),
            "date_min": str(base_state.index.min().date()),
            "date_max": str(base_state.index.max().date()),
            "unique_values": observed,
            "counts": {str(k): int(v) for k, v in base_state.value_counts().sort_index().items()},
            "reference_parity": {"available": False, "note": "公司包不读取外部九状态/冻结结果文件"},
        },
        "alignment": {
            "formation": "t close",
            "effective": "next actual spot trading row open (t+1)",
            "exit_h1": "second actual spot trading row open (t+2)",
            "o2o_h1": "open[t+2]/open[t+1]-1",
            "c2c_observation_only": "close[t+1]/close[t]-1",
            "business_day_offset_used": False,
            "whole_signal_shifted_again": False,
            "last_formation_date": str(panel.index.max().date()),
            "last_known_effective_date": str(panel["effective_date"].dropna().max().date()),
            "latest_formation_effective_pending": bool(pd.isna(panel["effective_date"].iloc[-1])),
        },
        "splits": {
            "Development": "2018-01-01..2022-12-31",
            "Validation": "2023-01-01..2024-12-31",
            "Test": "2025-01-01..latest",
        },
        "predictor_columns": predictor_columns,
        "forbidden_tokens": list(FORBIDDEN_TOKENS),
        "forbidden_feature_hits": forbidden_hits,
        "external_nine_state_read": False,
        "external_futures_read": False,
        "test_used_for_selection": False,
        "pending_base_state_rows": int(pending.sum()),
    }
    return panel, manifest


def cache_panel(output_dir: Path, start: Path | None = None) -> tuple[Path, Path]:
    panel, manifest = build_panel(start)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = output_dir / "canonical_panel.parquet"
    manifest_path = output_dir / "input_audit.json"
    panel.reset_index(names="formation_date").to_parquet(panel_path, index=False)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return panel_path, manifest_path
