from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from legacy_layout import build_html


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "03_汇报版本" / "20260818_1024_05汇报_基础三状态非零退出大涨大跌版"
SOURCE_04 = ROOT / "03_汇报版本" / "20260817_1657_04汇报_非零退出与四信号版"
# Prefer the latest local pure-spot cache.  The file named 20260817 currently
# contains prices through 2026-08-14; its overlap with the audited V156/V189
# fixture is identical for every price and volume field.
SPOT_CANDIDATES = [
    REPORT / "数据源" / "米筐_中证500_现货_20260818.parquet",
    Path("/Users/hzy/Desktop/0817合并查看/01_报告资料与基础上传包/数据源/long0_spot_rq_20260817.parquet"),
    Path("/Users/hzy/Desktop/0817合并查看/01_报告资料与基础上传包/数据源/1545_spot_rq_20260814.parquet"),
    Path("/Users/hzy/Desktop/一日大涨大跌/research_inputs/CSI500_SPOT_000905_XSHG_20070115_20260807_audited_pure_spot_fixture.parquet"),
]
EXTREME_SRC = Path("/Users/hzy/Desktop/一日大涨大跌/00_最终交付/01_最终上传包/20260815_最终纯现货二分类上传包_v6/src")

warnings.filterwarnings("ignore")


def _json_value(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(k): _json_value(v) for k, v in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _phase(dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates)
    return pd.Series(
        np.select(
            [dates.dt.year.between(2018, 2022), dates.dt.year.between(2023, 2024), dates.dt.year.ge(2025)],
            ["Development", "Validation", "Test"],
            default="样本预热段",
        ),
        index=dates.index,
    )


def _phase_metrics(rows: pd.DataFrame, side: str, threshold: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for phase_name, part in rows.groupby("phase", sort=False):
        labeled = part[part["target_return"].notna()]
        predicted = labeled[labeled["predicted"] == 1]
        actual = labeled[labeled["actual_extreme"] == 1]
        true_positive = int((predicted["actual_extreme"] == 1).sum())
        false_positive = int(len(predicted) - true_positive)
        false_negative = int(len(actual) - true_positive)
        true_negative = int(len(labeled) - true_positive - false_positive - false_negative)
        precision = true_positive / len(predicted) if len(predicted) else None
        recall = true_positive / len(actual) if len(actual) else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        base_rate = len(actual) / len(labeled) if len(labeled) else None
        signed = predicted["signed_return_bp"]
        out.append(
            {
                "phase": phase_name,
                "n_labeled": int(len(labeled)),
                "n_signal": int(len(predicted)),
                "n_actual_extreme": int(len(actual)),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "coverage_pct": float(len(predicted) / len(labeled) * 100) if len(labeled) else None,
                "extreme_hits": true_positive,
                "precision_pct": float(precision * 100) if precision is not None else None,
                "recall_pct": float(recall * 100) if recall is not None else None,
                "f1_pct": float(f1 * 100) if f1 is not None else None,
                "accuracy_pct": float((true_positive + true_negative) / len(labeled) * 100) if len(labeled) else None,
                "base_rate_pct": float(base_rate * 100) if base_rate is not None else None,
                "lift": float(precision / base_rate) if precision is not None and base_rate else None,
                "direction_accuracy_pct": float(predicted["direction_correct"].mean() * 100) if len(predicted) else None,
                "signed_mean_bp": float(signed.mean()) if len(signed) else None,
                "threshold_pct": float(threshold * 100),
                "side": side,
            }
        )
    return out


def _axes(base_index: int) -> list[int]:
    remainder = int(base_index)
    values = []
    for divisor in (512, 64, 8, 1):
        values.append(remainder // divisor)
        remainder %= divisor
    return values


def _normalise_spot_source(path: Path) -> pd.DataFrame:
    """Read one local pure-spot file into the frozen model's schema."""
    frame = pd.read_parquet(path).copy()
    if "date" not in frame.columns:
        if "trade_dt" not in frame.columns:
            raise ValueError(f"现货输入缺少 date/trade_dt: {path}")
        frame["date"] = pd.to_datetime(frame["trade_dt"].astype(str), format="%Y%m%d", errors="raise")
    else:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if "prev_close" not in frame.columns and "preclose" in frame.columns:
        frame["prev_close"] = frame["preclose"]
    if "amount" not in frame.columns and "total_turnover" in frame.columns:
        frame["amount"] = frame["total_turnover"]
    required = ["date", "open", "high", "low", "close", "volume", "amount", "prev_close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"现货输入缺少字段 {missing}: {path}")
    frame = frame.loc[:, required].copy()
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if frame["date"].isna().any() or frame["open"].isna().all():
        raise ValueError(f"现货输入日期或开盘价无效: {path}")
    return frame


def _refresh_panel_prices(panel: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    """Overlay the newest pure-spot prices and recompute open-to-open returns."""
    panel = panel.copy()
    spot = spot.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    spot_dates = spot["date"].tolist()
    spot_by_date = spot.set_index("date")
    for column in ["open", "high", "low", "close", "prev_close", "volume", "amount"]:
        if column in spot_by_date.columns and column in panel.columns:
            mapped = panel["date"].map(spot_by_date[column])
            panel[column] = mapped.combine_first(panel[column])
    for horizon in [1, 2, 3, 5, 10]:
        future_open = spot["open"].shift(-horizon)
        returns = future_open / spot["open"] - 1
        available = spot["open"].notna() & future_open.notna()
        return_by_date = dict(zip(spot_dates, returns))
        available_by_date = dict(zip(spot_dates, available))
        mapped_return = panel["date"].map(return_by_date)
        mapped_available = panel["date"].map(available_by_date).fillna(False).astype(bool)
        panel[f"o2o_h{horizon}"] = mapped_return.combine_first(panel[f"o2o_h{horizon}"])
        panel[f"o2o_h{horizon}_available"] = mapped_available
    panel["price_available"] = panel["date"].isin(set(spot_dates)) & panel["open"].notna() & panel["close"].notna()
    return panel


def _resolve_spot_source() -> tuple[Path, Path, pd.DataFrame]:
    """Choose the newest available pure-spot input and materialise a local copy."""
    available: list[tuple[pd.Timestamp, pd.Timestamp, Path, pd.DataFrame]] = []
    errors: list[str] = []
    for candidate in SPOT_CANDIDATES:
        if not candidate.is_file():
            continue
        try:
            frame = _normalise_spot_source(candidate)
            available.append((frame["date"].max(), frame["date"].min(), candidate, frame))
        except Exception as exc:  # pragma: no cover - only reached for a bad optional candidate
            errors.append(f"{candidate}: {exc}")
    if not available:
        details = "；".join(errors) if errors else "没有候选文件"
        raise FileNotFoundError(f"没有可用的大涨大跌现货输入：{details}")
    # Keep the original frozen start date so adding the newer tail cannot
    # change the causal warm-up history.  The first candidate is preferred
    # when two files have the same last date because it is the full-history
    # cache used by the current report inputs.
    _, _, input_path, frame = max(available, key=lambda item: item[0])
    frame = frame.loc[frame["date"].ge(pd.Timestamp("2007-01-15"))].reset_index(drop=True)
    normalised_path = REPORT / "数据源" / "大涨大跌现货输入_标准化.parquet"
    normalised_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(normalised_path, index=False)
    return input_path, normalised_path, frame


def build_extreme_data() -> tuple[dict[str, Any], pd.DataFrame]:
    input_spot_path, spot_path, source_frame = _resolve_spot_source()
    sys.path.insert(0, str(EXTREME_SRC))
    from o2o_research.paper_engine import _banks, load_paper_metadata, score_matrix  # type: ignore
    from o2o_research.pipeline import prepare_research  # type: ignore

    prepared = prepare_research(spot_path)
    raw = source_frame.copy()
    # The model score is formed on the close of source date t, then executed
    # at the next trading day's open.  The report therefore uses execution
    # date t+1 as the displayed row date, while keeping the original
    # prediction-to-return pairing: Open[t+2] / Open[t+1] - 1.
    execution_date = raw["date"].shift(-1)
    target = raw["open"].shift(-2) / raw["open"].shift(-1) - 1.0
    execution_phase = _phase(execution_date)
    development_target = target.loc[execution_phase.eq("Development") & target.notna()]
    thresholds = {
        "q10": float(development_target.quantile(0.10)),
        "q90": float(development_target.quantile(0.90)),
    }
    metadata = load_paper_metadata(EXTREME_SRC / "paper_metadata.json")
    display_start_date = pd.Timestamp("2018-01-01")
    configs = {
        "down": {
            "side": "down",
            "version": "V156",
            "candidate_id": "base_0621_cov_0.075",
            "base_index": 620,
            "score_threshold": 0.6832148298881413,
            "logic_name": "隔夜—日内拉锯频率",
            "core_logic": "overnight_daytime_tugwar",
            "axes": _axes(620),
            "indicator_names": ["隔夜/日内反转频率", "目标侧正向隔夜强度", "目标侧负向隔夜强度", "隔夜—日内差异波动"],
        },
        "up": {
            "side": "up",
            "version": "V189",
            "candidate_id": "base_1839_cov_0.055",
            "base_index": 1838,
            "score_threshold": 0.8135114753699175,
            "logic_name": "区间—成交活动—收盘位置联合尾部依赖",
            "core_logic": "range_volume_location_tail",
            "axes": _axes(1838),
            "indicator_names": ["区间尾部强度", "平滑联合尾部强度", "边际尾部依赖", "三指标一致性"],
        },
    }

    all_rows: dict[str, pd.DataFrame] = {}
    display_rows: dict[str, pd.DataFrame] = {}
    for side, config in configs.items():
        scores, _ = score_matrix(prepared.frame, metadata[config["version"]], side)
        banks = _banks(prepared.frame, config["core_logic"], side)
        score = scores[:, config["base_index"]]
        component_arrays = [banks[i][:, config["axes"][i]] for i in range(4)]
        frame = raw[["open", "high", "low", "close", "volume", "amount", "prev_close"]].shift(-1).copy()
        frame["formation_date"] = raw["date"]
        frame["date"] = execution_date
        frame["target_date"] = raw["date"].shift(-2)
        frame["target_return"] = target
        frame["target_bp"] = target * 10000.0
        frame["phase"] = execution_phase
        frame["score"] = score
        frame["score_threshold"] = config["score_threshold"]
        frame["score_excess"] = score - config["score_threshold"]
        frame["predicted"] = (np.isfinite(score) & (score >= config["score_threshold"])).astype(int)
        frame["actual_extreme"] = (target <= thresholds["q10"] if side == "down" else target >= thresholds["q90"]).fillna(False).astype(int)
        frame["correct"] = (frame["predicted"] & frame["actual_extreme"]).astype(int)
        frame["direction_correct"] = (
            frame["predicted"].astype(bool)
            & (frame["target_return"].lt(0) if side == "down" else frame["target_return"].gt(0))
        ).fillna(False).astype(int)
        frame["signed_return_bp"] = ((-target if side == "down" else target) * 10000.0)
        for idx, values in enumerate(component_arrays, start=1):
            frame[f"indicator_{idx}"] = values
        frame["prediction_label"] = np.where(frame["predicted"].eq(1), "预测" + ("大跌" if side == "down" else "大涨"), "")
        all_rows[side] = frame
        display_rows[side] = frame.loc[frame["date"].ge(display_start_date)].copy()

    combined = display_rows["up"].loc[:, ["date", "target_date", "open", "close", "target_return", "target_bp", "phase"]].copy()
    for side in ("up", "down"):
        side_frame = display_rows[side]
        combined[f"{side}_score"] = side_frame["score"].to_numpy()
        combined[f"{side}_predicted"] = side_frame["predicted"].to_numpy()
        combined[f"{side}_actual_extreme"] = side_frame["actual_extreme"].to_numpy()
        combined[f"{side}_score_excess"] = side_frame["score_excess"].to_numpy()
    combined["conflict"] = (combined["up_predicted"].eq(1) & combined["down_predicted"].eq(1)).astype(int)
    combined["marker"] = np.select(
        [combined["conflict"].eq(1), combined["up_predicted"].eq(1), combined["down_predicted"].eq(1)],
        ["conflict", "up", "down"],
        default="none",
    )

    models_payload: dict[str, Any] = {}
    for side, config in configs.items():
        frame = display_rows[side]
        keep = [
            "date", "target_date", "open", "high", "low", "close", "volume", "amount", "prev_close", "phase",
            "target_return", "target_bp", "score", "score_threshold", "score_excess", "predicted",
            "actual_extreme", "correct", "direction_correct", "signed_return_bp",
            "indicator_1", "indicator_2", "indicator_3", "indicator_4", "prediction_label",
        ]
        overall = frame.copy()
        overall["phase"] = "2018+"
        models_payload[side] = {
            **config,
            "threshold_target": thresholds["q10"] if side == "down" else thresholds["q90"],
            "threshold_target_bp": (thresholds["q10"] if side == "down" else thresholds["q90"]) * 10000.0,
            "score_weights": [0.34, 0.28, 0.22, 0.16],
            "indicator_names": config["indicator_names"],
            "overall_metrics": _phase_metrics(overall, side, thresholds["q10"] if side == "down" else thresholds["q90"])[0],
            "phase_metrics": _phase_metrics(frame, side, thresholds["q10"] if side == "down" else thresholds["q90"]),
            "rows": _records(frame[keep]),
        }

    source_audit = {
        key: value
        for key, value in prepared.data_audit.items()
        if key not in {"o2o_formula", "formation_date_start", "formation_date_end"}
    }
    source_audit["display_o2o_formula"] = "open[t+1] / open[t] - 1"
    payload = {
        "source": str(input_spot_path),
        "normalised_source": str(spot_path),
        "source_audit": source_audit,
        "display_start_date": display_start_date.strftime("%Y-%m-%d"),
        "date_convention": "页面主日期为执行日 t；target_date 为下一实际交易日",
        "target_formula": "open[t+1] / open[t] - 1",
        "target_description": "执行日 t 的开盘到下一实际交易日 t+1 的开盘 O2O",
        "latest_spot_date": raw["date"].max().strftime("%Y-%m-%d"),
        "latest_complete_execution_date": execution_date.loc[target.notna()].max().strftime("%Y-%m-%d"),
        "data_status_note": f"本地现货最后日期为 {raw['date'].max().strftime('%Y-%m-%d')}；之后的执行日和 O2O 保留为空，等待对应实际开盘价。",
        "thresholds": {
            "q10": thresholds["q10"],
            "q90": thresholds["q90"],
            "q10_bp": thresholds["q10"] * 10000.0,
            "q90_bp": thresholds["q90"] * 10000.0,
            "fit_period": "Development 2018–2022",
        },
        "models": models_payload,
        "combined_rows": _records(combined),
        "conflict_rows": _records(combined.loc[combined["conflict"].eq(1)]),
    }
    return payload, raw


def build_payload() -> dict[str, Any]:
    panel = pd.read_csv(SOURCE_04 / "统计表" / "panel.csv")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["nonzero_final_three_state"] = panel["three_state"]
    panel.loc[(panel["three_state"] == -1) & (panel["minus_exit_signal"] == 1), "nonzero_final_three_state"] = 0
    panel.loc[(panel["three_state"] == 1) & (panel["plus_exit_signal"] == 1), "nonzero_final_three_state"] = 0
    panel["phase"] = panel["phase"].replace({"Development": "Development", "Validation": "Validation", "Test": "Test"})

    event_summary = pd.read_csv(SOURCE_04 / "统计表" / "event_summary.csv")
    event_phase = pd.read_csv(SOURCE_04 / "统计表" / "event_phase_summary.csv")
    event_year = pd.read_csv(SOURCE_04 / "统计表" / "event_year_summary.csv")
    event_details = pd.read_csv(SOURCE_04 / "统计表" / "event_details.csv")
    event_keys = ["minus_exit", "plus_exit"]
    event_summary = event_summary[event_summary.signal_key.isin(event_keys)].copy()
    event_phase = event_phase[event_phase.signal_key.isin(event_keys)].copy()
    event_year = event_year[event_year.signal_key.isin(event_keys)].copy()
    event_details = event_details[event_details.signal_key.isin(event_keys)].copy()

    summaries: dict[str, dict[str, Any]] = {}
    for key in event_keys:
        row = event_summary.loc[event_summary.signal_key.eq(key)].iloc[0].to_dict()
        phase_rows = event_phase.loc[event_phase.signal_key.eq(key)].copy()
        summaries[key] = {
            "signal_key": key,
            "signal_label": row["signal_label"],
            "version": "V55" if key == "minus_exit" else "V80",
            "candidate_id": "score_02_q68_a1_c2_cd0" if key == "minus_exit" else "score_02_q90_a5_c1_cd0",
            "expected_state": int(row["expected_base_state"]),
            "event_days": int(row["event_days"]),
            "eligible_base_days": int(row["eligible_base_days"]),
            "coverage_pct": float(row["coverage_pct"]),
            "directional_mean_bp": float(row["directional_mean_bp"]),
            "directional_median_bp": float(row["directional_median_bp"]),
            "directional_p05_bp": float(row["directional_p05_bp"]),
            "directional_p95_bp": float(row["directional_p95_bp"]),
            "directional_win_rate_pct": float(row["directional_win_rate_pct"]),
            "mean_ci_low_bp": float(row["mean_ci_low_bp"]),
            "mean_ci_high_bp": float(row["mean_ci_high_bp"]),
            "eligible_directional_mean_bp": float(row["eligible_directional_mean_bp"]),
            "signal_lift_vs_eligible_bp": float(row["signal_lift_vs_eligible_bp"]),
            "eligible_directional_win_rate_pct": float(row["eligible_directional_win_rate_pct"]),
            "phase_rows": _records(phase_rows),
        }

    extreme_payload, raw = build_extreme_data()
    panel = _refresh_panel_prices(panel, raw)
    state_counts = {str(int(k)): int(v) for k, v in panel["three_state"].value_counts().sort_index().items()}
    nonzero_counts = {str(int(k)): int(v) for k, v in panel["nonzero_final_three_state"].value_counts().sort_index().items()}
    combined = panel[["date", "three_state", "minus_exit_signal", "plus_exit_signal", "nonzero_final_three_state", "close", "o2o_h1", "phase"]].copy()
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

    raw_export = raw.copy()
    raw_export["date"] = raw_export["date"].dt.strftime("%Y-%m-%d")
    raw_export["next_execution_date"] = raw["date"].shift(-1).dt.strftime("%Y-%m-%d")
    raw_export.to_csv(REPORT / "数据源" / "大涨大跌现货曲线.csv", index=False, encoding="utf-8-sig")

    payload = {
        "meta": {
            "title": "基础三状态、非零退出与大涨大跌预测",
            "built_at": "2026-08-18",
            "scope": "本包只覆盖基础三状态、非零退出和大涨大跌；不含其他转移信号。",
            "state_source": "03_汇报版本/20260817_1657_04汇报_非零退出与四信号版/统计表/panel.csv",
            "extreme_source": extreme_payload["source"],
            "extreme_normalised_source": extreme_payload["normalised_source"],
            "latest_state_date": str(panel["date"].max().date()),
            "latest_price_date": str(pd.to_datetime(panel.loc[panel["price_available"], "date"]).max().date()),
            "latest_o2o_execution_date": str(pd.to_datetime(panel.loc[panel["o2o_h1_available"], "date"]).max().date()),
            "data_status_note": (
                f"米筐现货最后有价格日期为 {pd.to_datetime(panel.loc[panel['price_available'], 'date']).max().date()}；"
                f"当前完整 H1 O2O 最后可算执行日为 {pd.to_datetime(panel.loc[panel['o2o_h1_available'], 'date']).max().date()}；"
                "尚无下一实际交易日开盘价的执行日保留为空。"
            ),
        },
        "state": {
            "rows": _records(combined),
            "state_counts": state_counts,
            "nonzero_counts": nonzero_counts,
            "minus_exit_count": int(panel["minus_exit_signal"].sum()),
            "plus_exit_count": int(panel["plus_exit_signal"].sum()),
        },
        "exit": {
            "summaries": summaries,
            "details": _records(event_details),
            "year_rows": _records(event_year),
        },
        "extreme": extreme_payload,
    }
    payload["legacy_layout"] = _build_legacy_payload(panel)
    return payload


def _read_legacy_reference() -> tuple[str, dict[str, Any]]:
    """Read only the old layout and its non-code reference summaries."""
    path = SOURCE_04 / "三状态与反转信号_离线互动汇报.html"
    text = path.read_text(encoding="utf-8")
    match = re.search(r"window\.__REPORT_DATA__\s*=\s*(.*?);\s*</script>", text, flags=re.S)
    if not match:
        raise ValueError(f"旧版汇报 HTML 中没有找到内嵌数据: {path}")
    return text, json.loads(match.group(1))


def _build_legacy_payload(panel: pd.DataFrame) -> dict[str, Any]:
    """Shape current freeze data for the archived four-signal workspace."""
    _, reference = _read_legacy_reference()
    view = panel.copy()
    view["date"] = pd.to_datetime(view["date"]).dt.strftime("%Y-%m-%d")
    view = view.rename(
        columns={
            "three_state": "state",
            "minus_exit_signal": "minusExit",
            "plus_exit_signal": "plusExit",
            "o2o_h1": "h1",
            "o2o_h2": "h2",
            "o2o_h3": "h3",
            "o2o_h5": "h5",
            "o2o_h10": "h10",
        }
    )
    view["combined"] = view["state"]
    view.loc[(view["state"] == -1) & (view["minusExit"] == 1), "combined"] = 0
    view.loc[(view["state"] == 1) & (view["plusExit"] == 1), "combined"] = 0
    columns = ["date", "state", "minusExit", "plusExit", "combined", "open", "close", "h1", "h2", "h3", "h5", "h10", "phase"]
    view = view.loc[:, columns].copy()
    event_keys = {"minus_exit", "plus_exit"}
    return {
        "panel": _records(view),
        "eventSummary": [row for row in reference["eventSummary"] if row.get("key") in event_keys],
        "forwardSummary": [row for row in reference["forwardSummary"] if row.get("key") in event_keys],
        "eventPositions": [row for row in reference["eventPositions"] if row.get("key") in event_keys],
        "eventYears": [row for row in reference["eventYears"] if row.get("key") in event_keys],
        "clusterSummary": [row for row in reference["clusterSummary"] if row.get("key") in event_keys],
        "sourceFiles": [
            "01_非零状态退出详细讲稿.md",
            "统计表/panel.csv",
            "统计表/非零退出事件明细.csv",
            "统计表/非零退出汇总.csv",
        ],
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>基础三状态、非零退出与大涨大跌预测 · 离线互动汇报</title>
  <style>
    :root{--ink:#162033;--muted:#68758a;--line:#dbe3ee;--panel:#fff;--bg:#f4f7fb;--blue:#2563eb;--blue-soft:#e9f0ff;--red:#dc2626;--green:#159447;--gray:#6b7280;--amber:#d97706;--purple:#7c3aed;}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
    .wrap{max-width:1500px;margin:0 auto;padding:26px 28px 56px}.hero{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin-bottom:20px}.eyebrow{color:var(--blue);font-weight:800;letter-spacing:.08em;font-size:12px}.hero h1{margin:4px 0 8px;font-size:30px;line-height:1.2}.hero p{margin:0;color:var(--muted);max-width:900px}.stamp{white-space:nowrap;color:var(--muted);font-size:12px;text-align:right;padding-top:6px}
    .tabs{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 20px;position:sticky;top:0;z-index:10;background:rgba(244,247,251,.94);padding:10px 0;backdrop-filter:blur(8px)}.tab-btn{border:1px solid var(--line);background:#fff;color:#445168;border-radius:999px;padding:8px 14px;cursor:pointer;font-weight:700}.tab-btn:hover{border-color:#9db6e8}.tab-btn.active{background:var(--ink);color:#fff;border-color:var(--ink)}
    .page{display:none}.page.active{display:block}.section{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;margin:14px 0;box-shadow:0 8px 24px rgba(28,44,75,.04)}.section h2{margin:0 0 6px;font-size:20px}.section h3{margin:4px 0 8px;font-size:16px}.section-intro{color:var(--muted);margin:0 0 16px}.grid{display:grid;gap:14px}.grid-2{grid-template-columns:repeat(2,minmax(0,1fr))}.grid-3{grid-template-columns:repeat(3,minmax(0,1fr))}.grid-4{grid-template-columns:repeat(4,minmax(0,1fr))}.metric{border:1px solid var(--line);border-radius:13px;padding:13px 14px;background:linear-gradient(180deg,#fff,#fbfdff)}.metric .label{color:var(--muted);font-size:12px}.metric .value{font-size:24px;font-weight:800;margin-top:2px}.metric .sub{color:var(--muted);font-size:12px}.pill{display:inline-flex;align-items:center;gap:5px;border-radius:999px;padding:3px 8px;background:var(--blue-soft);color:#315db9;font-size:12px;font-weight:700}.pill.red{background:#fff0f0;color:#b42323}.pill.green{background:#ecfdf3;color:#117a3e}.pill.gray{background:#f0f1f3;color:#4b5563}.pill.amber{background:#fff7e7;color:#a35c05}
    .chart-wrap{border:1px solid var(--line);border-radius:14px;padding:10px;background:#fcfdff;overflow:hidden}.chart-toolbar{display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;margin:0 2px 6px}.toolbar-left,.toolbar-right{display:flex;align-items:center;gap:6px;flex-wrap:wrap}.mini-btn{border:1px solid var(--line);background:#fff;border-radius:8px;padding:5px 9px;color:#4a5568;cursor:pointer;font-size:12px}.mini-btn:hover{border-color:#91aee5}.chart-note{font-size:12px;color:var(--muted)}svg.chart{display:block;width:100%;height:360px;touch-action:none}.axis{stroke:#cad5e2;stroke-width:1}.gridline{stroke:#e8edf4;stroke-width:1}.line{fill:none;stroke:#334155;stroke-width:1.7}.line-soft{fill:none;stroke:#64748b;stroke-width:1.2;opacity:.72}.state-dot{stroke:#fff;stroke-width:1}.marker-up{fill:var(--red);stroke:#fff;stroke-width:1}.marker-down{fill:var(--green);stroke:#fff;stroke-width:1}.marker-conflict{fill:#7b818a;stroke:#fff;stroke-width:1}.actual-ring{fill:#fff;stroke:var(--amber);stroke-width:2}.actual-hit{fill:var(--amber);stroke:#fff;stroke-width:1}.actual-miss{fill:#fff;stroke:#f59e0b;stroke-width:1.5}.legend{display:flex;gap:14px;flex-wrap:wrap;color:#566176;font-size:12px;margin:4px 2px}.legend span{display:inline-flex;align-items:center;gap:5px}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%}.legend .red-dot{background:var(--red)}.legend .green-dot{background:var(--green)}.legend .gray-dot{background:#7b818a}.legend .amber-dot{background:var(--amber)}.legend .ring{background:#fff;border:2px solid var(--amber)}
    .tooltip{position:fixed;z-index:30;display:none;pointer-events:none;min-width:210px;max-width:330px;background:rgba(18,28,45,.96);color:#fff;border-radius:10px;padding:10px 12px;font-size:12px;box-shadow:0 12px 30px rgba(0,0,0,.2)}.tooltip b{font-size:13px}.tooltip .muted{color:#cbd5e1}.hover-readout{min-height:28px;margin:6px 2px 0;color:#4b5568;font-size:12px}
    table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:8px 9px;border-bottom:1px solid #e9edf3;text-align:left;vertical-align:top}th{color:#66738a;background:#f8fafc;position:sticky;top:0;z-index:1}tbody tr:hover{background:#fbfdff}.table-scroll{max-height:390px;overflow:auto;border:1px solid var(--line);border-radius:12px}.num{text-align:right;font-variant-numeric:tabular-nums}.good{color:#0f8a43;font-weight:700}.bad{color:#c52828;font-weight:700}.neutral{color:#667085}.select-row{display:flex;gap:10px;align-items:end;flex-wrap:wrap;margin-bottom:12px}.control label{display:block;color:var(--muted);font-size:12px;margin-bottom:3px}.control select,.control input{border:1px solid var(--line);border-radius:8px;background:#fff;padding:7px 9px;color:var(--ink);min-width:130px}.control input[type=date]{min-width:145px}.callout{border-left:4px solid var(--blue);background:#f3f7ff;padding:11px 13px;border-radius:0 10px 10px 0;color:#334155}.callout.red{border-left-color:var(--red);background:#fff6f6}.callout.green{border-left-color:var(--green);background:#f2fcf6}.callout.gray{border-left-color:#7b818a;background:#f6f7f8}.small{font-size:12px;color:var(--muted)}.empty{padding:22px;text-align:center;color:var(--muted);border:1px dashed var(--line);border-radius:12px}.formula{font-family:Georgia,"Times New Roman",serif;color:#1e293b;background:#f8fafc;border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin:8px 0}.footer{color:var(--muted);font-size:12px;margin-top:22px}.hide{display:none!important}
    @media(max-width:900px){.wrap{padding:18px 14px}.hero{display:block}.stamp{text-align:left;margin-top:8px}.grid-2,.grid-3,.grid-4{grid-template-columns:1fr}.hero h1{font-size:24px}svg.chart{height:300px}}
  </style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <div><div class="eyebrow">05 汇报包 · OFFLINE INTERACTIVE REPORT</div><h1>基础三状态、非零退出与大涨大跌预测</h1><p>主展示文件：先看交互页面，再用两份 Markdown 补充构造过程、公式和最终冻结口径。页面内数据已内嵌，断网也可以打开。</p></div>
    <div class="stamp">构建日期：2026-08-18<br>数据口径：本地冻结结果</div>
  </header>
  <nav class="tabs" aria-label="报告页面">
    <button class="tab-btn active" data-page="base">1 · 基础三状态</button>
    <button class="tab-btn" data-page="combined-exit">2 · 加入两个非零退出</button>
    <button class="tab-btn" data-page="minus-exit">3 · 负向退出独立</button>
    <button class="tab-btn" data-page="plus-exit">4 · 正向退出独立</button>
    <button class="tab-btn" data-page="extreme-combined">5 · 大涨大跌合并</button>
    <button class="tab-btn" data-page="extreme-up">6 · 大涨独立</button>
    <button class="tab-btn" data-page="extreme-down">7 · 大跌独立</button>
  </nav>

  <main>
    <section class="page active" id="page-base">
      <div class="section"><h2>基础三状态</h2><p class="section-intro">先固定原始状态序列，作为后面两类信号的共同基准。这里不做额外改动，只展示状态数量、价格曲线和逐日状态。</p>
        <div id="base-metrics" class="grid grid-4"></div>
      </div>
      <div class="section"><h2>指数价格曲线与基础状态</h2><div id="base-chart"></div><div class="small">曲线可拖动查看局部区间；悬停可以读取日期、收盘价、基础状态和当日 O2O。</div></div>
      <div class="section"><h2>逐日基础状态</h2><div id="base-table" class="table-scroll"></div></div>
    </section>

    <section class="page" id="page-combined-exit">
      <div class="section"><h2>加入两个非零退出</h2><p class="section-intro">模型在前一形成日收盘后计算，并在下一实际交易日开盘执行；只在执行日改变状态：负向基础状态遇到负向退出改为 0，正向基础状态遇到正向退出改为 0；后续执行日重新读取基础三状态。</p>
        <div id="combined-exit-metrics" class="grid grid-4"></div>
      </div>
      <div class="section"><h2>基础状态与加入退出后的对照</h2><div id="combined-exit-chart"></div><div class="legend"><span><i class="red-dot"></i>负向退出</span><span><i class="green-dot"></i>正向退出</span><span><i class="amber-dot"></i>退出后进入状态 0</span></div></div>
      <div class="section"><h2>两类退出的核心结果</h2><div id="combined-exit-table"></div></div>
    </section>

    <section class="page" id="page-minus-exit">
      <div class="section"><h2>负向退出独立分析 · V55</h2><p class="section-intro">目标是识别基础状态 -1 中适合退出到状态 0 的执行日；主指标为方向化 O2O H1。</p><div id="minus-metrics" class="grid grid-4"></div></div>
      <div class="section"><h2>信号位置</h2><div id="minus-chart"></div></div>
      <div class="section"><h2>信号逐日明细</h2><div id="minus-table" class="table-scroll"></div></div>
    </section>

    <section class="page" id="page-plus-exit">
      <div class="section"><h2>正向退出独立分析 · V80</h2><p class="section-intro">目标是识别基础状态 +1 中适合退出到状态 0 的执行日；主指标为方向化 O2O H1。</p><div id="plus-metrics" class="grid grid-4"></div></div>
      <div class="section"><h2>信号位置</h2><div id="plus-chart"></div></div>
      <div class="section"><h2>信号逐日明细</h2><div id="plus-table" class="table-scroll"></div></div>
    </section>

    <section class="page" id="page-extreme-combined">
      <div class="section"><h2>大涨 / 大跌独立建模 · 合并观察</h2><p class="section-intro">大涨和大跌是两个独立信号，不是互为补集。红色向上三角表示预测大涨，绿色向下三角表示预测大跌；同日双预测用灰色菱形表示。</p><div id="extreme-threshold-metrics" class="grid grid-4"></div></div>
      <div class="section"><h2>1 · 全览或拖动查看指数价格曲线</h2><div id="extreme-combined-chart"></div><div class="legend"><span><i class="red-dot"></i>预测大涨</span><span><i class="green-dot"></i>预测大跌</span><span><i class="gray-dot"></i>大涨/大跌冲突</span></div></div>
      <div class="section"><h2>2 · 灰色冲突日与信号指标</h2><p class="section-intro">只列出同日同时触发两侧预测的日期；每条信号同时保留得分、冻结阈值和超过阈值的幅度。</p><div id="conflict-table"></div></div>
      <div class="section"><h2>3 · 选择侧别和时间段，核对预测与实际阈值事件</h2><div class="select-row"><div class="control"><label for="combined-side">查看侧别</label><select id="combined-side"><option value="up">大涨</option><option value="down">大跌</option></select></div><div class="control"><label for="combined-start">开始日期</label><input id="combined-start" type="date"></div><div class="control"><label for="combined-end">结束日期</label><input id="combined-end" type="date"></div></div><div id="combined-period-chart"></div><div id="combined-period-table" class="table-scroll"></div></div>
    </section>

    <section class="page" id="page-extreme-up">
      <div class="section"><h2>大涨预测独立分析 · V189</h2><p class="section-intro">单独查看大涨侧预测、阈值命中和方向化 O2O。该页去掉冲突日板块，保留单侧指标和逐日对照。</p><div id="up-metrics" class="grid grid-4"></div></div>
      <div class="section"><h2>指数曲线与大涨预测</h2><div id="up-chart"></div><div class="legend"><span><i class="red-dot"></i>预测大涨</span><span><i class="ring"></i>实际达到大涨阈值</span></div></div>
      <div class="section"><h2>大涨逐日预测明细</h2><div id="up-table" class="table-scroll"></div></div>
    </section>

    <section class="page" id="page-extreme-down">
      <div class="section"><h2>大跌预测独立分析 · V156</h2><p class="section-intro">单独查看大跌侧预测、阈值命中和方向化 O2O。该页去掉冲突日板块，保留单侧指标和逐日对照。</p><div id="down-metrics" class="grid grid-4"></div></div>
      <div class="section"><h2>指数曲线与大跌预测</h2><div id="down-chart"></div><div class="legend"><span><i class="green-dot"></i>预测大跌</span><span><i class="ring"></i>实际达到大跌阈值</span></div></div>
      <div class="section"><h2>大跌逐日预测明细</h2><div id="down-table" class="table-scroll"></div></div>
    </section>
  </main>
  <div class="footer">提示：页面是离线单文件，所有数据已嵌入。完整的构造流程、冻结参数、公式和讲解顺序见同目录两份 Markdown。</div>
</div>
<div id="tooltip" class="tooltip"></div>
<script>
const DATA = __PAYLOAD__;
const charts = {};
const $ = (id) => document.getElementById(id);
const fmt = (v, digits=2) => (v === null || v === undefined || Number.isNaN(Number(v))) ? '—' : Number(v).toFixed(digits);
const pct = (v, digits=2) => (v === null || v === undefined || Number.isNaN(Number(v))) ? '—' : `${Number(v).toFixed(digits)}%`;
const bp = (v, digits=2) => (v === null || v === undefined || Number.isNaN(Number(v))) ? '—' : `${Number(v).toFixed(digits)} bp`;
const stateLabel = (v) => ({'-1':'-1 · 负向','0':'0 · 中性','1':'+1 · 正向'})[String(v)] || String(v);
const html = (s) => String(s ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));

function metricCard(label, value, sub='') { return `<div class="metric"><div class="label">${html(label)}</div><div class="value">${html(value)}</div><div class="sub">${html(sub)}</div></div>`; }
function table(headers, rows) { if (!rows.length) return '<div class="empty">暂无记录</div>'; return `<table><thead><tr>${headers.map(h=>`<th>${html(h.label)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${headers.map(h=>`<td class="${h.num?'num':''}">${h.render? h.render(row):html(row[h.key])}</td>`).join('')}</tr>`).join('')}</tbody></table>`; }
function phaseTag(v) { const cls = v==='Test'?'red':(v==='Validation'?'amber':''); return `<span class="pill ${cls}">${html(v)}</span>`; }
function showTooltip(evt, content) { const tip=$('tooltip'); tip.innerHTML=content; tip.style.display='block'; let x=evt.clientX+14,y=evt.clientY+14; if(x+340>window.innerWidth)x=evt.clientX-350; if(y+210>window.innerHeight)y=evt.clientY-220; tip.style.left=`${Math.max(8,x)}px`; tip.style.top=`${Math.max(8,y)}px`; }
function hideTooltip(){ $('tooltip').style.display='none'; }
function dateMs(s){ return new Date(`${s}T00:00:00`).getTime(); }
function shortDate(s){ return String(s).slice(0,10); }

function makeChart(containerId, rows, cfg={}) {
  const el=$(containerId); if(!el) return;
  if(!rows.length){el.innerHTML='<div class="chart-wrap"><div class="empty">当前日期范围没有可展示的数据，请调整开始日期和结束日期。</div></div>';return;}
  const key=cfg.key||containerId; if(!charts[key]) charts[key]={start:0,end:rows.length-1};
  const state=charts[key]; state.start=Math.max(0,Math.min(state.start,Math.max(0,rows.length-2))); state.end=Math.max(state.start+1,Math.min(state.end,rows.length-1));
  const shown=rows.slice(state.start,state.end+1); const W=1160,H=360,M={l:58,r:22,t:18,b:34}; const iw=W-M.l-M.r,ih=H-M.t-M.b;
  const values=shown.map(r=>Number(r[cfg.valueKey||'close'])).filter(Number.isFinite); const lo=Math.min(...values),hi=Math.max(...values); const pad=(hi-lo||1)*.08; const y0=lo-pad,y1=hi+pad;
  const x=(i)=>M.l+(i/(Math.max(1,shown.length-1)))*iw; const y=(v)=>M.t+(1-(Number(v)-y0)/(y1-y0))*ih;
  let path=''; shown.forEach((r,i)=>{ const v=Number(r[cfg.valueKey||'close']); if(!Number.isFinite(v))return; path+=(path?' L':'M')+`${x(i).toFixed(2)},${y(v).toFixed(2)}`; });
  const ticks=[0,.25,.5,.75,1].map(t=>{const yy=M.t+t*ih;const val=y1-t*(y1-y0);return `<line class="gridline" x1="${M.l}" x2="${W-M.r}" y1="${yy}" y2="${yy}"/><text x="${M.l-8}" y="${yy+4}" text-anchor="end" fill="#7b8799" font-size="11">${fmt(val,0)}</text>`}).join('');
  const markers=(cfg.markers||[]).map(m=>{const index=rows.findIndex(r=>r.date===m.date); if(index<state.start||index>state.end||index<0)return '';const px=x(index-state.start),py=y(rows[index][cfg.valueKey||'close']); return m.shape==='up'?`<path class="marker-up" d="M ${px} ${py-9} L ${px-8} ${py+6} L ${px+8} ${py+6} Z" data-i="${index}"/>`:m.shape==='down'?`<path class="marker-down" d="M ${px} ${py+9} L ${px-8} ${py-6} L ${px+8} ${py-6} Z" data-i="${index}"/>`:m.shape==='diamond'?`<path class="marker-conflict" d="M ${px} ${py-9} L ${px+9} ${py} L ${px} ${py+9} L ${px-9} ${py} Z" data-i="${index}"/>`:m.shape==='hit'?`<circle class="actual-hit" cx="${px}" cy="${py}" r="5" data-i="${index}"/>`:m.shape==='actual'?`<circle class="actual-ring" cx="${px}" cy="${py}" r="6" data-i="${index}"/>`:''; }).join('');
  const svg=`<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><g>${ticks}<line class="axis" x1="${M.l}" x2="${W-M.r}" y1="${H-M.b}" y2="${H-M.b}"/><path class="line" d="${path}"/>${markers}<text x="${M.l}" y="${H-10}" fill="#7b8799" font-size="11">${html(shortDate(shown[0].date))}</text><text x="${W-M.r}" y="${H-10}" text-anchor="end" fill="#7b8799" font-size="11">${html(shortDate(shown[shown.length-1].date))}</text><rect x="${M.l}" y="${M.t}" width="${iw}" height="${ih}" fill="transparent" data-overlay="1"/></g></svg>`;
  const toolbar=`<div class="chart-toolbar"><div class="toolbar-left"><button class="mini-btn" data-act="full">全览</button><button class="mini-btn" data-act="recent">最近3年</button><button class="mini-btn" data-act="reset">重置</button></div><div class="chart-note">拖动曲线可平移；悬停读取该日信息</div></div><div class="legend">${cfg.legend||'<span><i class="gray-dot"></i>指数价格</span>'}</div><div class="chart-svg">${svg}</div><div class="hover-readout" id="readout-${key}">将鼠标移到曲线上查看详细信息</div>`;
  el.innerHTML=`<div class="chart-wrap">${toolbar}</div>`;
  el.querySelectorAll('[data-act]').forEach(btn=>btn.addEventListener('click',()=>{const a=btn.dataset.act;if(a==='full'||a==='reset'){state.start=0;state.end=rows.length-1}else if(a==='recent'){const min=dateMs('2023-01-01');let i=rows.findIndex(r=>dateMs(r.date)>=min);if(i<0)i=Math.max(0,rows.length-700);state.start=i;state.end=rows.length-1}makeChart(containerId,rows,cfg)}));
  const svgEl=el.querySelector('svg'); const overlay=el.querySelector('[data-overlay]'); let drag=null;
  const nearest=(evt)=>{const rect=svgEl.getBoundingClientRect(); const px=(evt.clientX-rect.left)/rect.width*W; const ratio=(px-M.l)/iw; const idx=Math.max(0,Math.min(shown.length-1,Math.round(ratio*(shown.length-1)))); return {idx,global:state.start+idx,row:rows[state.start+idx],px};};
  overlay.addEventListener('pointermove',evt=>{const n=nearest(evt);const r=n.row;if(!r)return;const extras=cfg.tooltip?r=>cfg.tooltip(r):`<b>${html(r.date)}</b><br><span class="muted">收盘价</span> ${fmt(r.close)}<br><span class="muted">状态</span> ${html(stateLabel(r.three_state??''))}`;showTooltip(evt,typeof extras==='function'?extras(r):extras);const rd=$(`readout-${key}`);if(rd)rd.innerHTML=cfg.readout?cfg.readout(r):`当前：${html(r.date)} · 收盘 ${fmt(r.close)} · ${html(stateLabel(r.three_state??''))}`});
  overlay.addEventListener('pointerleave',hideTooltip); overlay.addEventListener('pointerdown',evt=>{drag={x:evt.clientX,start:state.start,end:state.end};overlay.setPointerCapture(evt.pointerId)}); overlay.addEventListener('pointerup',evt=>{if(!drag)return;const dx=evt.clientX-drag.x;const shift=Math.round(-dx/iw*shown.length);const width=drag.end-drag.start;state.start=Math.max(0,Math.min(rows.length-width-1,drag.start+shift));state.end=state.start+width;drag=null;makeChart(containerId,rows,cfg)});
}

function renderBase(){
  const s=DATA.state; $('base-metrics').innerHTML=[metricCard('基础状态 -1',s.state_counts['-1']+' 天','原始基础状态'),metricCard('基础状态 0',s.state_counts['0']+' 天','原始基础状态'),metricCard('基础状态 +1',s.state_counts['1']+' 天','原始基础状态'),metricCard('样本日期',s.rows[0].date+' → '+s.rows[s.rows.length-1].date,'共 '+s.rows.length+' 个执行日')].join('');
  makeChart('base-chart',s.rows,{key:'base',legend:'<span><i class="gray-dot"></i>指数收盘曲线</span>',tooltip:r=>`<b>${html(r.date)}</b><br><span class="muted">收盘</span> ${fmt(r.close)}<br><span class="muted">基础状态</span> ${html(stateLabel(r.three_state))}<br><span class="muted">O2O H1</span> ${bp(r.o2o_h1*10000)}`,readout:r=>`当前：${r.date} · 基础状态 ${stateLabel(r.three_state)} · O2O H1 ${bp(r.o2o_h1*10000)}`});
  const rows=s.rows.slice().reverse().slice(0,220); $('base-table').innerHTML=table([{label:'日期',key:'date'},{label:'基础状态',key:'three_state',render:r=>stateLabel(r.three_state)},{label:'收盘',key:'close',num:true,render:r=>fmt(r.close)},{label:'O2O H1',key:'o2o_h1',num:true,render:r=>bp(r.o2o_h1*10000)},{label:'阶段',key:'phase',render:r=>phaseTag(r.phase)}],rows);
}

function exitMetricHtml(key){const m=DATA.exit.summaries[key];return [metricCard('事件数',m.event_days+' 天','可用基础状态 '+m.eligible_base_days+' 天'),metricCard('覆盖率',pct(m.coverage_pct),'在对应基础状态内'),metricCard('方向化均值',bp(m.directional_mean_bp),'方向化 O2O H1'),metricCard('均值区间',`[${fmt(m.mean_ci_low_bp)}, ${fmt(m.mean_ci_high_bp)}] bp`,'描述性重抽样区间')].join('')}
function exitMarkers(key){return DATA.exit.details.filter(r=>r.signal_key===key).map(r=>({date:r.date,shape:key==='minus_exit'?'down':'up'}));}
function renderCombinedExit(){
  const s=DATA.state; $('combined-exit-metrics').innerHTML=[metricCard('退出信号合计',(s.minus_exit_count+s.plus_exit_count)+' 天','两侧独立相加'),metricCard('负向退出',s.minus_exit_count+' 天','V55'),metricCard('正向退出',s.plus_exit_count+' 天','V80'),metricCard('退出后状态 0',String(s.rows.filter(r=>Number(r.three_state)!==0&&Number(r.nonzero_final_three_state)===0).length)+' 天','仅执行日改变')].join('');
  const markers=[...exitMarkers('minus_exit'),...exitMarkers('plus_exit')];makeChart('combined-exit-chart',s.rows,{key:'combined-exit',markers,legend:'<span><i class="red-dot"></i>负向退出</span><span><i class="green-dot"></i>正向退出</span>',tooltip:r=>`<b>${html(r.date)}</b><br><span class="muted">基础状态</span> ${html(stateLabel(r.three_state))}<br><span class="muted">退出后</span> ${html(stateLabel(r.nonzero_final_three_state))}<br><span class="muted">信号</span> ${Number(r.minus_exit_signal)?'负向退出 ':''}${Number(r.plus_exit_signal)?'正向退出':''}`});
  $('combined-exit-table').innerHTML=table([{label:'信号',key:'signal_label'},{label:'版本',key:'version'},{label:'事件数',key:'event_days',num:true},{label:'覆盖率',key:'coverage_pct',num:true,render:r=>pct(r.coverage_pct)},{label:'方向化均值',key:'directional_mean_bp',num:true,render:r=>bp(r.directional_mean_bp)},{label:'均值区间',key:'ci',render:r=>`[${fmt(r.mean_ci_low_bp)}, ${fmt(r.mean_ci_high_bp)}] bp`},{label:'相对基础提升',key:'signal_lift_vs_eligible_bp',num:true,render:r=>bp(r.signal_lift_vs_eligible_bp)}],Object.values(DATA.exit.summaries).map(m=>({...m,ci:''})));
}

function renderExitPage(key,prefix){const m=DATA.exit.summaries[key];$(prefix+'-metrics').innerHTML=[metricCard('版本',m.version,m.candidate_id),metricCard('事件数',m.event_days+' 天','有效基础状态 '+m.eligible_base_days+' 天'),metricCard('方向化均值',bp(m.directional_mean_bp),'方向化 O2O H1'),metricCard('方向化胜率',pct(m.directional_win_rate_pct),'描述性区间 ['+fmt(m.mean_ci_low_bp)+', '+fmt(m.mean_ci_high_bp)+'] bp')].join('');const rows=DATA.state.rows;makeChart(prefix+'-chart',rows,{key:prefix,markers:exitMarkers(key),legend:key==='minus_exit'?'<span><i class="green-dot"></i>负向退出信号</span>':'<span><i class="red-dot"></i>正向退出信号</span>',tooltip:r=>`<b>${html(r.date)}</b><br><span class="muted">基础状态</span> ${html(stateLabel(r.three_state))}<br><span class="muted">信号</span> ${key==='minus_exit'?Number(r.minus_exit_signal)?'负向退出':'无':Number(r.plus_exit_signal)?'正向退出':'无'}<br><span class="muted">O2O H1</span> ${bp(r.o2o_h1*10000)}`});const details=DATA.exit.details.filter(r=>r.signal_key===key).slice().reverse();$(prefix+'-table').innerHTML=table([{label:'日期',key:'date'},{label:'阶段',key:'phase',render:r=>phaseTag(r.phase)},{label:'基础状态',key:'three_state',render:r=>stateLabel(r.three_state)},{label:'段内位置',key:'signal_position_pct',num:true,render:r=>pct(r.signal_position_pct)},{label:'H1 原始',key:'raw_o2o_h1_bp',num:true,render:r=>bp(r.raw_o2o_h1_bp)},{label:'H1 方向化',key:'directional_improvement_bp',num:true,render:r=>bp(r.directional_improvement_bp)},{label:'H5',key:'o2o_h5',num:true,render:r=>bp(r.o2o_h5*10000)}],details)}

function renderExtremeThresholds(){const t=DATA.extreme.thresholds;const up=DATA.extreme.models.up,down=DATA.extreme.models.down;$('extreme-threshold-metrics').innerHTML=[metricCard('大跌实际阈值',pct(t.q10*100),'q10 · '+bp(t.q10_bp)),metricCard('大涨实际阈值',pct(t.q90*100),'q90 · '+bp(t.q90_bp)),metricCard('大跌预测阈值',fmt(down.score_threshold,4),'V156 · '+down.candidate_id),metricCard('大涨预测阈值',fmt(up.score_threshold,4),'V189 · '+up.candidate_id)].join('')}
function combinedExtremeTooltip(r){const up=DATA.extreme.models.up,down=DATA.extreme.models.down;return `<b>${html(r.date)}</b><br><span class="muted">收盘</span> ${fmt(r.close)}<br><span class="muted">大涨得分 / 阈值</span> ${fmt(r.up_score,4)} / ${fmt(up.score_threshold,4)}<br><span class="muted">大跌得分 / 阈值</span> ${fmt(r.down_score,4)} / ${fmt(down.score_threshold,4)}<br><span class="muted">预测</span> ${Number(r.up_predicted)&&Number(r.down_predicted)?'冲突':Number(r.up_predicted)?'大涨':Number(r.down_predicted)?'大跌':'无'}<br><span class="muted">实际 O2O</span> ${bp(r.target_bp)}`}
function renderExtremeCombinedChart(){const rows=DATA.extreme.combined_rows;const markers=rows.filter(r=>r.marker!=='none').map(r=>({date:r.date,shape:r.marker==='conflict'?'diamond':r.marker==='up'?'up':'down'}));makeChart('extreme-combined-chart',rows,{key:'extreme-combined',markers,legend:'<span><i class="red-dot"></i>预测大涨</span><span><i class="green-dot"></i>预测大跌</span><span><i class="gray-dot"></i>同日冲突</span>',tooltip:r=>combinedExtremeTooltip(r),readout:r=>`${r.date} · ${Number(r.up_predicted)&&Number(r.down_predicted)?'冲突':Number(r.up_predicted)?'预测大涨':Number(r.down_predicted)?'预测大跌':'无预测'} · 大涨 ${fmt(r.up_score,4)} · 大跌 ${fmt(r.down_score,4)} · 实际 O2O ${bp(r.target_bp)}`})}
function extremeMarkers(side){const model=DATA.extreme.models[side];const rows=model.rows;return rows.filter(r=>Number(r.predicted)).map(r=>({date:r.date,shape:side==='up'?'up':'down'})).concat(rows.filter(r=>Number(r.actual_extreme)).map(r=>({date:r.date,shape:'actual'})));}
function extremeTooltip(side,r){const model=DATA.extreme.models[side];const names=model.indicator_names;return `<b>${html(r.date)}</b><br><span class="muted">预测</span> ${Number(r.predicted)?(side==='up'?'大涨':'大跌'):'无'}<br><span class="muted">得分 / 阈值</span> ${fmt(r.score,4)} / ${fmt(r.score_threshold,4)}<br><span class="muted">超过阈值</span> ${fmt(r.score_excess,4)}<br><span class="muted">实际 O2O</span> ${bp(r.target_bp)}<br><span class="muted">实际阈值事件</span> ${Number(r.actual_extreme)?'是':'否'}<br><span class="muted">指标</span> ${names.map((n,i)=>html(n)+' '+fmt(r['indicator_'+(i+1)],3)).join(' · ')}`}
function extremeReadout(side,r){return `${r.date} · ${side==='up'?'大涨':'大跌'}得分 ${fmt(r.score,4)} · 阈值 ${fmt(r.score_threshold,4)} · 实际 O2O ${bp(r.target_bp)} · ${Number(r.predicted)?(Number(r.correct)?'命中':'预测未命中'):'未预测'}`}
function renderExtremeMetrics(side,prefix){const m=DATA.extreme.models[side], test=m.phase_metrics.find(x=>x.phase==='Test'),t=DATA.extreme.thresholds;const target=side==='up'?t.q90_bp:t.q10_bp;$(prefix+'-metrics').innerHTML=[metricCard('冻结版本',m.version,m.candidate_id),metricCard('实际阈值',pct(target/100),'Development '+DATA.extreme.thresholds.fit_period),metricCard('Test 预测数',test?test.n_signal+' 天':'—',test?('命中 '+test.extreme_hits+' 天'):'无'),metricCard('Test 方向化均值',test?bp(test.signed_mean_bp):'—',test?('方向准确率 '+pct(test.direction_accuracy_pct)):'')].join('')}
function renderExtremePage(side){const prefix=side==='up'?'up':'down';const m=DATA.extreme.models[side];renderExtremeMetrics(side,prefix);makeChart(prefix+'-chart',m.rows,{key:prefix,markers:extremeMarkers(side),legend:side==='up'?'<span><i class="red-dot"></i>预测大涨</span><span><i class="ring"></i>实际达到 q90</span>':'<span><i class="green-dot"></i>预测大跌</span><span><i class="ring"></i>实际达到 q10</span>',tooltip:r=>extremeTooltip(side,r),readout:r=>extremeReadout(side,r)});const sig=m.rows.filter(r=>Number(r.predicted)).slice().reverse();$(prefix+'-table').innerHTML=table([{label:'日期',key:'date'},{label:'阶段',key:'phase',render:r=>phaseTag(r.phase)},{label:'得分',key:'score',num:true,render:r=>fmt(r.score,4)},{label:'超阈值',key:'score_excess',num:true,render:r=>fmt(r.score_excess,4)},{label:'实际 O2O',key:'target_bp',num:true,render:r=>bp(r.target_bp)},{label:'实际阈值事件',key:'actual_extreme',render:r=>Number(r.actual_extreme)?'<span class="good">是</span>':'否'},{label:'结果',key:'correct',render:r=>r.target_return===null?'<span class="neutral">待实现</span>':Number(r.correct)?'<span class="good">预测正确</span>':'<span class="bad">预测错误</span>'}],sig)}
function renderConflict(){const rows=DATA.extreme.conflict_rows; if(!rows.length){$('conflict-table').innerHTML='<div class="empty">当前数据中没有同日同时触发大涨和大跌预测，因此没有灰色菱形冲突日。</div>';return;}const up=DATA.extreme.models.up,down=DATA.extreme.models.down;const map={};up.rows.forEach(r=>map[r.date]={...r,up_score:r.score,up_excess:r.score_excess});down.rows.forEach(r=>map[r.date]={...map[r.date],down_score:r.score,down_excess:r.score_excess});$('conflict-table').innerHTML=table([{label:'日期',key:'date'},{label:'实际 O2O',key:'target_bp',num:true,render:r=>bp(r.target_bp)},{label:'大涨得分 / 超阈值',key:'up_score',num:true,render:r=>`${fmt(map[r.date]?.up_score,4)} / ${fmt(map[r.date]?.up_excess,4)}`},{label:'大跌得分 / 超阈值',key:'down_score',num:true,render:r=>`${fmt(map[r.date]?.down_score,4)} / ${fmt(map[r.date]?.down_excess,4)}`},{label:'阶段',key:'phase',render:r=>phaseTag(r.phase)}],rows)}
function renderCombinedPeriod(){const side=$('combined-side').value;const all=DATA.extreme.models[side].rows;const start=$('combined-start').value||all[0].date,end=$('combined-end').value||all[all.length-1].date;const rows=all.filter(r=>r.date>=start&&r.date<=end);const markers=[];rows.forEach(r=>{if(Number(r.actual_extreme))markers.push({date:r.date,shape:Number(r.correct)?'hit':'actual'});if(Number(r.predicted))markers.push({date:r.date,shape:side==='up'?'up':'down'});});makeChart('combined-period-chart',rows,{key:'combined-period-'+side,markers,legend:side==='up'?'<span><i class="red-dot"></i>预测大涨</span><span><i class="ring"></i>实际 q90</span>':'<span><i class="green-dot"></i>预测大跌</span><span><i class="ring"></i>实际 q10</span>',tooltip:r=>extremeTooltip(side,r),readout:r=>extremeReadout(side,r)});const filtered=rows.filter(r=>Number(r.predicted)||Number(r.actual_extreme)).slice().reverse();$('combined-period-table').innerHTML=table([{label:'日期',key:'date'},{label:'阶段',key:'phase',render:r=>phaseTag(r.phase)},{label:'得分',key:'score',num:true,render:r=>fmt(r.score,4)},{label:'阈值',key:'score_threshold',num:true,render:r=>fmt(r.score_threshold,4)},{label:'得分超出',key:'score_excess',num:true,render:r=>fmt(r.score_excess,4)},{label:'实际 O2O',key:'target_bp',num:true,render:r=>bp(r.target_bp)},{label:'分类',key:'class',render:r=>{if(Number(r.predicted)&&Number(r.actual_extreme))return '<span class="good">TP · 预测对</span>';if(Number(r.predicted))return '<span class="bad">FP · 预测错</span>';return '<span class="neutral">FN · 漏报</span>'}}],filtered.map(r=>({...r,class:''})))}
function setDefaultDates(){const all=DATA.extreme.models.up.rows;$('combined-start').value=all[0].date;$('combined-end').value=all[all.length-1].date;$('combined-side').addEventListener('change',renderCombinedPeriod);$('combined-start').addEventListener('change',renderCombinedPeriod);$('combined-end').addEventListener('change',renderCombinedPeriod)}
function renderPage(id){if(id==='base')renderBase();if(id==='combined-exit')renderCombinedExit();if(id==='minus-exit')renderExitPage('minus_exit','minus');if(id==='plus-exit')renderExitPage('plus_exit','plus');if(id==='extreme-combined'){renderExtremeThresholds();renderExtremeCombinedChart();renderConflict();renderCombinedPeriod()}if(id==='extreme-up')renderExtremePage('up');if(id==='extreme-down')renderExtremePage('down')}
document.querySelectorAll('.tab-btn').forEach(btn=>btn.addEventListener('click',()=>{document.querySelectorAll('.tab-btn').forEach(x=>x.classList.toggle('active',x===btn));document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id==='page-'+btn.dataset.page));renderPage(btn.dataset.page)}));
setDefaultDates();renderPage('base');
</script>
</body>
</html>'''


def write_report(payload: dict[str, Any]) -> None:
    (REPORT / "数据源").mkdir(parents=True, exist_ok=True)
    (REPORT / "统计表").mkdir(parents=True, exist_ok=True)
    (REPORT / "图片").mkdir(parents=True, exist_ok=True)
    (REPORT / "数据源" / "基础三状态与非零退出.csv").write_text(
        pd.DataFrame(payload["state"]["rows"]).to_csv(index=False), encoding="utf-8"
    )
    pd.DataFrame(payload["exit"]["details"]).to_csv(REPORT / "统计表" / "非零退出事件明细.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(list(payload["exit"]["summaries"].values())).drop(columns=["phase_rows"], errors="ignore").to_csv(REPORT / "统计表" / "非零退出汇总.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([r for m in payload["exit"]["summaries"].values() for r in m["phase_rows"]]).to_csv(REPORT / "统计表" / "非零退出阶段汇总.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(payload["exit"]["year_rows"]).to_csv(REPORT / "统计表" / "非零退出年度汇总.csv", index=False, encoding="utf-8-sig")
    for side, model in payload["extreme"]["models"].items():
        pd.DataFrame(model["rows"]).to_csv(REPORT / "统计表" / f"{side}_大涨大跌预测明细.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(model["phase_metrics"]).to_csv(REPORT / "统计表" / f"{side}_大涨大跌阶段汇总.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(payload["extreme"]["combined_rows"]).to_csv(REPORT / "统计表" / "大涨大跌合并逐日结果.csv", index=False, encoding="utf-8-sig")
    export_payload = {key: value for key, value in payload.items() if key != "legacy_layout"}
    (REPORT / "数据源" / "report_payload.json").write_text(json.dumps(export_payload, ensure_ascii=False, indent=2, default=_json_value) + "\n", encoding="utf-8")
    html = build_html(payload, SOURCE_04 / "三状态与反转信号_离线互动汇报.html")
    (REPORT / "基础三状态_非零退出_大涨大跌_离线互动汇报.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    payload = build_payload()
    write_report(payload)
    print(json.dumps({
        "report": str(REPORT),
        "html": str(REPORT / "基础三状态_非零退出_大涨大跌_离线互动汇报.html"),
        "thresholds": payload["extreme"]["thresholds"],
        "conflicts": len(payload["extreme"]["conflict_rows"]),
        "exit_summaries": {k: {x: v for x, v in m.items() if x not in {"phase_rows"}} for k, m in payload["exit"]["summaries"].items()},
    }, ensure_ascii=False, indent=2))
