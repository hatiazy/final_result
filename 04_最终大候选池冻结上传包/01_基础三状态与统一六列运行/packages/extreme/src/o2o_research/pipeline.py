from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .candidates import candidate_parameter_table, raw_candidate_count, score_matrix
from .data import FrozenRecord, TestVault, TestVaultFactory, load_spot_input, source_whitelist, split_research_periods
from .features import build_causal_features, feature_lineage_table
from .literature import LITERATURE
from .metrics import (
    annual_metrics,
    block_bootstrap_metrics,
    bootstrap_summary,
    error_diagnostics,
    evaluate_frozen_candidate,
    label_thresholds,
    score_bins,
    score_candidate_pool,
    select_top1,
)
from .specs import VersionSpec, get_version_spec


@dataclass
class PreparedResearch:
    frame: pd.DataFrame
    development: pd.DataFrame
    validation: pd.DataFrame
    _test_factory: TestVaultFactory = field(repr=False)
    latest: pd.DataFrame
    data_audit: dict[str, Any]

    def new_test_vault(self) -> TestVault:
        return self._test_factory.issue()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if pd.isna(value):
        return None
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def prepare_research(input_path: str | Path | None = None) -> PreparedResearch:
    raw, data_audit = load_spot_input(input_path)
    full_frame = build_causal_features(raw)
    development, validation, test_factory, _ = split_research_periods(full_frame)
    test_audit = test_factory.audit_metadata()
    # Test feature values may be scored after freezing, but Test outcomes are
    # masked in every object exposed by prepare_research.  The original outcomes
    # exist only inside TestVaultFactory and require a valid FrozenRecord.
    frame = full_frame.copy()
    test_mask = frame.date.ge("2025-01-01")
    frame.loc[test_mask, ["future_open_to_open_return_1d", "future_close_to_close_return_1d"]] = np.nan
    latest = frame.tail(10).copy()
    data_audit = {
        **data_audit,
        "formation_date_start": frame.date.min().strftime("%Y-%m-%d"),
        "formation_date_end": frame.date.max().strftime("%Y-%m-%d"),
        "development_rows": int(len(development)),
        "development_start": development.date.min().strftime("%Y-%m-%d"),
        "development_end": development.date.max().strftime("%Y-%m-%d"),
        "validation_rows": int(len(validation)),
        "validation_start": validation.date.min().strftime("%Y-%m-%d"),
        "validation_end": validation.date.max().strftime("%Y-%m-%d"),
        "test_rows_including_unlabeled_latest": test_audit["rows_including_unlabeled_latest"],
        "test_start": test_audit["date_start"],
        "test_end": test_audit["date_end"],
        "o2o_formula": "open[t+2] / open[t+1] - 1",
        "c2c_formula_observation_only": "close[t+1] / close[t] - 1",
        "effective_date_rule": "next actual row/trading day",
        "test_used_for_selection": False,
    }
    return PreparedResearch(frame, development, validation, test_factory, latest, data_audit)


def _phase_from_date(date: pd.Series) -> np.ndarray:
    return np.select(
        [date.dt.year.between(2018, 2022), date.dt.year.between(2023, 2024), date.dt.year.ge(2025)],
        ["development", "validation", "test"],
        default="other",
    )


def _research_pass(metrics: dict[str, Any], phase: str) -> bool:
    def finite(name: str, default: float) -> float:
        try:
            value = float(metrics.get(name, default))
        except (TypeError, ValueError):
            return default
        return value if np.isfinite(value) else default

    min_signal = 20 if phase != "test" else 15
    return bool(
        int(metrics.get("n_signal", 0)) >= min_signal
        and finite("precision_lift", 0.0) >= 1.5
        and finite("signed_mean_o2o", 0.0) > 0
        and finite("rank_ic", 0.0) > 0
        and finite("reverse_extreme_rate", 1.0) <= 0.20
    )


def _frozen_prediction_frame(
    frame: pd.DataFrame,
    score: np.ndarray,
    active: np.ndarray,
    side: str,
    threshold: float,
) -> pd.DataFrame:
    cols = [
        "date", "entry_date", "label_exit_date", "max_feature_date",
        "future_open_to_open_return_1d", "future_close_to_close_return_1d",
    ]
    out = frame[cols].copy()
    out["phase"] = _phase_from_date(out.date)
    out["side"] = side
    out["score"] = score
    out["score_threshold_fitted_development"] = threshold
    out["alert"] = np.asarray(active, bool).astype(int)
    out["test_used_for_selection"] = False
    return out.loc[out.date.ge("2018-01-01")].reset_index(drop=True)


def run_version_side(
    prepared: PreparedResearch,
    version: str | VersionSpec,
    side: str,
    output_dir: str | Path,
    bootstrap_draws: int = 500,
) -> dict[str, Any]:
    spec = version if isinstance(version, VersionSpec) else get_version_spec(version)
    if side not in {"down", "up"}:
        raise ValueError("side must be down or up")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "data_audit.json", prepared.data_audit)
    source_whitelist().to_csv(output_dir / "source_whitelist.csv", index=False)
    feature_lineage_table().to_csv(output_dir / "feature_lineage.csv", index=False)

    thresholds = label_thresholds(prepared.development)
    parameters = candidate_parameter_table(spec, side)
    parameters.to_csv(output_dir / "candidate_parameters.csv", index=False)
    full_scores, base_metadata = score_matrix(prepared.frame, spec, side)
    base_metadata.to_csv(output_dir / "base_candidate_parameters.csv", index=False)
    dev_scores = full_scores[prepared.development.index.to_numpy(), :]
    val_scores = full_scores[prepared.validation.index.to_numpy(), :]
    candidates = score_candidate_pool(
        prepared.development,
        prepared.validation,
        dev_scores,
        val_scores,
        parameters,
        side,
        thresholds,
    )
    candidates.to_csv(output_dir / "candidate_metrics_development_validation.csv", index=False)
    candidates.loc[candidates.is_unique_signal].head(20).to_csv(output_dir / "top20_development_validation.csv", index=False)
    frozen = select_top1(candidates)
    base_j = int(str(frozen.base_candidate_id).split("_")[-1]) - 1
    freeze_record = FrozenRecord(
        version=spec.version,
        side=side,
        candidate_id=str(frozen.candidate_id),
        frozen_at_stage="development_validation",
        test_used_for_selection=False,
    )
    freeze_payload = {
        **freeze_record.__dict__,
        "frozen_first_before_test": True,
        "test_gate_used_for_enablement": False,
        "label_thresholds_fitted_development": thresholds,
        "candidate_parameters": {
            key: frozen[key] for key in (
                "base_candidate_id", "coverage_config", "aggregator", "component_cut",
                "w1", "w2", "w3", "w4", "component_power", "agreement_weight",
                "down_components", "up_components", "score_threshold_fitted_development",
            )
        },
        "joint_score_formula_version": "o2o_joint_v1_fixed_before_V01",
        "joint_score": frozen.joint_score,
        "candidate_count_raw": raw_candidate_count(),
        "candidate_count_unique_signal": int(candidates.is_unique_signal.sum()),
        "duplicate_signal_ratio": float(1 - candidates.is_unique_signal.mean()),
        "test_metrics_present_at_freeze_time": False,
    }
    # This file is written before the TestVault can be opened.
    write_json(output_dir / "FROZEN_TOP1_BEFORE_TEST.json", freeze_payload)

    vault = prepared.new_test_vault()
    test = vault.unlock(freeze_record)
    if not vault.opened:
        raise AssertionError("test vault did not record a valid freeze")
    dev_score = dev_scores[:, base_j]
    val_score = val_scores[:, base_j]
    test_score = full_scores[test.index.to_numpy(), base_j]
    dev_metrics, dev_active = evaluate_frozen_candidate(prepared.development, dev_score, frozen, side, thresholds)
    val_metrics, val_active = evaluate_frozen_candidate(prepared.validation, val_score, frozen, side, thresholds)
    test_metrics, test_active = evaluate_frozen_candidate(test, test_score, frozen, side, thresholds)

    three_phase = pd.DataFrame([
        {"phase": "development", **dev_metrics},
        {"phase": "validation", **val_metrics},
        {"phase": "test_frozen_observation_only", **test_metrics},
    ])
    three_phase.to_csv(output_dir / "frozen_top1_three_phase_metrics.csv", index=False)

    phase_bins = []
    for name, frame, score in (
        ("development", prepared.development, dev_score),
        ("validation", prepared.validation, val_score),
        ("test_frozen_observation_only", test, test_score),
    ):
        for bin_count in (5, 10):
            bins = score_bins(frame, score, side, bins=bin_count)
            if not bins.empty:
                bins.insert(0, "group_count", bin_count)
                bins.insert(0, "phase", name)
                phase_bins.append(bins)
    group_metrics = pd.concat(phase_bins, ignore_index=True)
    group_metrics.to_csv(output_dir / "score_group_metrics_5_and_10.csv", index=False)
    group_metrics.loc[group_metrics.group_count.eq(10)].to_csv(output_dir / "score_decile_metrics.csv", index=False)

    threshold = float(frozen.score_threshold_fitted_development)
    score_all = full_scores[:, base_j]
    active_all = score_all >= threshold
    observed_frame = prepared.frame.copy()
    outcome_columns = ["future_open_to_open_return_1d", "future_close_to_close_return_1d"]
    observed_frame.loc[test.index, outcome_columns] = test[outcome_columns]
    research_frame = observed_frame.loc[
        observed_frame.date.ge("2018-01-01") & observed_frame.future_open_to_open_return_1d.notna()
    ].copy()
    annual = annual_metrics(
        research_frame,
        score_all[research_frame.index.to_numpy()],
        active_all[research_frame.index.to_numpy()],
        side,
        thresholds,
    )
    annual.to_csv(output_dir / "annual_metrics.csv", index=False)

    bootstrap_parts = []
    bootstrap_summaries = {}
    phase_payloads = (
        ("development", prepared.development, dev_active),
        ("validation", prepared.validation, val_active),
        ("test_frozen_observation_only", test, test_active),
    )
    version_number = int(spec.version[1:])
    for phase_number, (name, frame, active) in enumerate(phase_payloads, start=1):
        samples = block_bootstrap_metrics(
            frame, active, side, thresholds,
            seed=20260812 + version_number * 100 + phase_number * 10 + (0 if side == "down" else 1),
            draws=bootstrap_draws,
        )
        samples.insert(0, "phase", name)
        bootstrap_parts.append(samples)
        bootstrap_summaries[name] = bootstrap_summary(samples)
    pd.concat(bootstrap_parts, ignore_index=True).to_csv(output_dir / "bootstrap_samples.csv", index=False)
    write_json(output_dir / "bootstrap_summary.json", bootstrap_summaries)

    errors = error_diagnostics(test, test_score, test_active, side, thresholds)
    errors.to_csv(output_dir / "test_error_diagnostics.csv", index=False)
    if not errors.empty:
        numeric = errors.select_dtypes(include=[np.number]).columns.tolist()
        errors.groupby("error_type")[numeric].agg(["count", "mean", "median"]).to_csv(output_dir / "test_error_scenario_summary.csv")
    else:
        pd.DataFrame(columns=["error_type"]).to_csv(output_dir / "test_error_scenario_summary.csv", index=False)

    predictions = _frozen_prediction_frame(observed_frame, score_all, active_all, side, threshold)
    predictions.to_csv(output_dir / "frozen_top1_daily_predictions.csv", index=False)
    predictions.tail(10).to_csv(output_dir / "latest_10_scores_and_alerts.csv", index=False)

    label_rows = []
    for phase_name, phase_frame in (
        ("development", prepared.development),
        ("validation", prepared.validation),
        ("test_frozen_observation_only", test),
    ):
        y = phase_frame.future_open_to_open_return_1d.astype(float)
        valid = y.notna()
        label_rows.append({
            "phase": phase_name,
            "n_rows": int(len(phase_frame)),
            "n_labeled": int(valid.sum()),
            "down_label_n": int((y[valid] <= thresholds["q10"]).sum()),
            "down_label_ratio": float((y[valid] <= thresholds["q10"]).mean()) if valid.any() else np.nan,
            "up_label_n": int((y[valid] >= thresholds["q90"]).sum()),
            "up_label_ratio": float((y[valid] >= thresholds["q90"]).mean()) if valid.any() else np.nan,
            "q10_fitted_development": thresholds["q10"],
            "q90_fitted_development": thresholds["q90"],
            "test_used_for_selection": False,
        })
    pd.DataFrame(label_rows).to_csv(output_dir / "phase_sample_and_label_audit.csv", index=False)

    pretest_pass = _research_pass(dev_metrics, "development") and _research_pass(val_metrics, "validation")
    frozen_test_pass = _research_pass(test_metrics, "test")
    latest_row = predictions.iloc[-1]
    latest_labeled = predictions.loc[predictions.future_open_to_open_return_1d.notna()].iloc[-1]
    summary = {
        "version": spec.version,
        "side": side,
        "core_logic_name": spec.core_logic_name,
        "title_zh": spec.title_zh,
        "hypothesis": spec.hypothesis,
        "aggregator": spec.aggregator,
        "candidate_count_raw": raw_candidate_count(),
        "candidate_count_unique_signal": int(candidates.is_unique_signal.sum()),
        "logic_family_count": 1,
        "model_family_count": 1,
        "parameter_dimensions": ["four_component_weights", "component_power", "agreement_weight", "coverage"],
        "duplicate_signal_ratio": float(1 - candidates.is_unique_signal.mean()),
        "label_thresholds_development": thresholds,
        "frozen_candidate_id": str(frozen.candidate_id),
        "frozen_base_candidate_id": str(frozen.base_candidate_id),
        "frozen_score_threshold": threshold,
        "development": dev_metrics,
        "validation": val_metrics,
        "test_frozen_observation_only": test_metrics,
        "pretest_research_pass": pretest_pass,
        "frozen_test_diagnostic_pass": frozen_test_pass,
        "formal_pass_after_frozen_test": bool(pretest_pass and frozen_test_pass),
        "test_used_for_selection": False,
        "c2c_used_for_selection": False,
        "uses_1545": spec.uses_1545,
        "uses_nine_state": spec.uses_nine_state,
        "literature": [LITERATURE[key] for key in spec.literature_keys],
        "latest_formation_date": latest_row.date,
        "latest_effective_date": latest_row.entry_date,
        "latest_exit_date": latest_row.label_exit_date,
        "latest_fully_labeled_formation_date": latest_labeled.date,
        "latest_fully_labeled_effective_date": latest_labeled.entry_date,
        "latest_fully_labeled_exit_date": latest_labeled.label_exit_date,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _pct(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(value) else f"{value:.2%}"


def write_version_readme(version_dir: str | Path, spec: VersionSpec, summaries: dict[str, dict[str, Any]]) -> None:
    version_dir = Path(version_dir)
    lines = [
        f"# {spec.version} 大涨大跌预测｜{spec.title_zh}",
        "",
        f"- `core_logic_name`: `{spec.core_logic_name}`",
        f"- 核心假设：{spec.hypothesis}",
        f"- 聚合器：`{spec.aggregator}`；本大版本不混入其他模型或经济逻辑。",
        "- 主标签：`open[t+2] / open[t+1] - 1`；C2C 只作观察。",
        "- 输入：纯中证500现货日频 OHLCV/成交额；1545 与九状态字段均未进入本版候选。",
        "- 每侧原始候选 3,072；阈值仅由 Development 拟合，Development＋Validation 联合冻结 Top1，Test 仅冻结后展示。",
        "",
        "## 文献方法与任务改写",
        "",
    ]
    for key in spec.literature_keys:
        item = LITERATURE[key]
        lines.append(f"- [{item['title']}]({item['url']})（{item['authors']}, {item['year']}）：{item['use']}")
    lines += [
        "",
        "保留：可由 t 日及以前现货数据定义的路径、波动、流动性或状态思想。舍弃：期权、期货、基差、持仓、测试年份开关，以及论文原样本上的胜出参数。",
        "",
        "## 冻结结果",
        "",
        "| 侧别 | 原始/去重候选 | 冻结 Top1 | Dev Rank IC / precision | Val Rank IC / precision | Test Rank IC / precision | 正式结论 |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for side in ("down", "up"):
        s = summaries[side]
        d, v, t = s["development"], s["validation"], s["test_frozen_observation_only"]
        verdict = "通过" if s["formal_pass_after_frozen_test"] else "未正式通过，保留最佳可计算 Top1"
        lines.append(
            f"| {'大跌' if side == 'down' else '大涨'} | {s['candidate_count_raw']}/{s['candidate_count_unique_signal']} | "
            f"`{s['frozen_candidate_id']}` | {_pct(d['rank_ic'])} / {_pct(d['precision'])} | "
            f"{_pct(v['rank_ic'])} / {_pct(v['precision'])} | {_pct(t['rank_ic'])} / {_pct(t['precision'])} | {verdict} |"
        )
    lines += [
        "",
        "## 文件",
        "",
        "- `01_O2O大跌预测.ipynb`、`02_O2O大涨预测.ipynb` 可分别独立 Run All。",
        "- `results/down/`、`results/up/` 保存候选、冻结、三阶段、分层、年度、Bootstrap、错误场景和最新十日输出。",
        "- `FROZEN_TOP1_BEFORE_TEST.json` 的落盘先于 TestVault 解锁；`test_used_for_selection=false`。",
        "",
        "## 边界",
        "",
        "Test 区间在历史项目中已被观察，本版仅保证同版代码严格隔离，不能声称从未见过。未纳入成本、滑点、冲击、容量和仓位约束。",
        "",
    ]
    (version_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_version(
    prepared: PreparedResearch,
    version: str,
    version_dir: str | Path,
    bootstrap_draws: int = 500,
) -> dict[str, dict[str, Any]]:
    spec = get_version_spec(version)
    version_dir = Path(version_dir)
    summaries = {
        "down": run_version_side(prepared, spec, "down", version_dir / "results" / "down", bootstrap_draws),
        "up": run_version_side(prepared, spec, "up", version_dir / "results" / "up", bootstrap_draws),
    }
    write_version_readme(version_dir, spec, summaries)
    return summaries
