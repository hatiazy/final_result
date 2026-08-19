from __future__ import annotations

"""Strict binary extreme-layer runner.

This module is deliberately separate from the four-state experiments.  It
only evaluates one binary extreme side at a time (``down`` or ``up``), using
the spot-derived score surface supplied by a construction version.  A
candidate is selected only after Development+Validation have been frozen;
Test is opened through :class:`TestVault` afterwards and is observational.

The existing engines grew different eligibility/fallback details over time.
This engine is the common, stricter protocol used for the current binary
enhancement pass: at least 20 alerts in both Development and Validation,
exact Development+Validation signal deduplication, and no fallback when the
strict pool is empty.
"""

import gc
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import reserve_engine as base
from .data import FrozenRecord, source_whitelist
from .features import feature_lineage_table
from .pipeline import PreparedResearch, write_json


COVERAGE_GRID: tuple[float, ...] = (
    0.025, 0.035, 0.045, 0.055, 0.065, 0.075,
    0.085, 0.095, 0.105, 0.115, 0.130, 0.150,
)
# A 2.5%--4.5% alert rate produces only a handful of Test observations on
# the current 2018+ split.  Such a row can look accurate by chance, so the
# strict DV protocol excludes it before any Test unlock.  The floor is
# deliberately moderate rather than a high-coverage objective: precision
# remains the ranking objective within the admissible 5.5%--15% grid.
MIN_COVERAGE_CONFIG = 0.055


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if pd.isna(value):
        return None
    raise TypeError(type(value).__name__)


def _normalise_base_meta(base_meta: pd.DataFrame) -> pd.DataFrame:
    """Give all score providers one common base-index contract."""
    out = base_meta.copy().reset_index(drop=True)
    if "base_index" not in out:
        out.insert(1 if "base_candidate_id" in out else 0, "base_index", np.arange(len(out), dtype=int))
    out["base_index"] = pd.to_numeric(out["base_index"], errors="raise").astype(int)
    if "base_candidate_id" not in out:
        out.insert(0, "base_candidate_id", [f"base_{i + 1:04d}" for i in range(len(out))])
    if out.base_candidate_id.duplicated().any():
        raise AssertionError("base candidate ids must be unique")
    expected = np.arange(len(out), dtype=int)
    if not np.array_equal(np.sort(out.base_index.to_numpy()), expected):
        # A provider may expose a one-based/other ordering.  Re-index it
        # explicitly; the score matrix is always in provider row order.
        out["base_index"] = expected
    return out


def candidate_parameter_table(
    base_meta: pd.DataFrame,
    side: str,
    meta: dict[str, Any],
) -> pd.DataFrame:
    base_meta = _normalise_base_meta(base_meta)
    rows: list[dict[str, Any]] = []
    for record in base_meta.to_dict(orient="records"):
        for coverage in COVERAGE_GRID:
            row = dict(record)
            base_id = str(record["base_candidate_id"])
            row.update({
                "candidate_id": f"{base_id}_cov_{coverage:.3f}",
                "coverage_config": float(coverage),
                "side": side,
                "reserve_id": str(meta.get("reserve_id", meta.get("version", ""))),
                "core_logic_name": str(meta.get("core_logic_name", "")),
                "title_zh": str(meta.get("title_zh", "")),
                "candidate_schema": str(meta.get("candidate_schema", "provider")),
            })
            rows.append(row)
    result = pd.DataFrame(rows)
    expected = len(base_meta) * len(COVERAGE_GRID)
    if len(result) != expected or result.candidate_id.nunique() != expected:
        raise AssertionError(f"candidate table mismatch: {len(result)} != {expected}")
    return result


def _strict_pass(metrics: dict[str, Any], phase: str) -> bool:
    minimum = 15 if phase == "test" else 20

    def finite(name: str, default: float) -> float:
        try:
            value = float(metrics.get(name, default))
        except (TypeError, ValueError):
            return default
        return value if np.isfinite(value) else default

    return bool(
        int(metrics.get("n_signal", 0)) >= minimum
        and finite("precision_lift", 0.0) >= 1.5
        and finite("signed_mean_o2o", 0.0) > 0
        and finite("rank_ic", 0.0) > 0
        and finite("reverse_extreme_rate", 1.0) <= 0.20
    )


def _score_candidate_pool(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    development_scores: np.ndarray,
    validation_scores: np.ndarray,
    parameters: pd.DataFrame,
    side: str,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Score every base×coverage candidate with strict DV-only metadata."""
    if development_scores.shape[1] != parameters.base_index.nunique():
        raise AssertionError(
            f"score/base mismatch: {development_scores.shape[1]} != {parameters.base_index.nunique()}"
        )
    y_scale = float(development.future_open_to_open_return_1d.std(ddof=0))
    rows: list[dict[str, Any]] = []
    # Keeping the metrics implementation shared with the legacy engines makes
    # old and new result files directly comparable.
    for parameter in parameters.itertuples(index=False):
        j = int(parameter.base_index)
        ds = np.asarray(development_scores[:, j], float)
        vs = np.asarray(validation_scores[:, j], float)
        threshold = float(np.nanquantile(ds, 1.0 - float(parameter.coverage_config)))
        da = np.isfinite(ds) & (ds >= threshold)
        va = np.isfinite(vs) & (vs >= threshold)
        dm = base._phase_metrics(development, ds, da, side, thresholds)
        vm = base._phase_metrics(validation, vs, va, side, thresholds)
        row = parameter._asdict()
        row.update({f"dev_{key}": value for key, value in dm.items()})
        row.update({f"val_{key}": value for key, value in vm.items()})
        row["score_threshold_fitted_development"] = threshold
        row["joint_score"] = base._joint_score(dm, vm, y_scale)
        row["signal_hash_dev_validation"] = base._signal_hash(da, va)
        # Keep the sample-count diagnostic separate from the actual freeze
        # gate.  A candidate is eligible for Test unlock only when both DV
        # phases pass the full strict rule (sample count, lift, signed O2O,
        # positive RankIC and reverse-extreme cap).
        row["eligible_min_samples"] = bool(dm["n_signal"] >= 20 and vm["n_signal"] >= 20)
        row["eligible_coverage_floor"] = bool(
            float(parameter.coverage_config) >= MIN_COVERAGE_CONFIG
        )
        row["eligible_strict_dv"] = bool(
            row["eligible_coverage_floor"]
            and row["eligible_min_samples"]
            and _strict_pass(dm, "development")
            and _strict_pass(vm, "validation")
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    # A signal is represented once.  If several parameter rows generate the
    # same DV alert vector, prefer an eligible row first, then joint score.
    result = result.sort_values(
        ["signal_hash_dev_validation", "eligible_strict_dv", "eligible_min_samples", "joint_score", "candidate_id"],
        ascending=[True, False, False, False, True], kind="mergesort",
    ).reset_index(drop=True)
    result["signal_duplicate_rank"] = result.groupby("signal_hash_dev_validation").cumcount()
    result["is_unique_signal"] = result.signal_duplicate_rank.eq(0)
    leader = result.loc[
        result.is_unique_signal,
        ["signal_hash_dev_validation", "candidate_id"],
    ].rename(columns={"candidate_id": "signal_duplicate_of"})
    result = result.merge(leader, on="signal_hash_dev_validation", how="left")
    result = result.sort_values(
        ["is_unique_signal", "eligible_strict_dv", "eligible_min_samples", "joint_score", "val_precision", "candidate_id"],
        ascending=[False, False, False, False, False, True], kind="mergesort",
    ).reset_index(drop=True)
    result["selection_rank"] = np.arange(1, len(result) + 1)
    return result


def _phase_audit(prepared: PreparedResearch, test: pd.DataFrame, thresholds: dict[str, float], output_dir: Path) -> None:
    rows = []
    for name, frame in (
        ("development", prepared.development),
        ("validation", prepared.validation),
        ("test_frozen_observation_only", test),
    ):
        y = frame.future_open_to_open_return_1d.astype(float)
        valid = y.notna()
        rows.append({
            "phase": name,
            "n_rows": int(len(frame)),
            "n_labeled": int(valid.sum()),
            "down_label_n": int((y[valid] <= thresholds["q10"]).sum()),
            "down_label_ratio": float((y[valid] <= thresholds["q10"]).mean()) if valid.any() else np.nan,
            "up_label_n": int((y[valid] >= thresholds["q90"]).sum()),
            "up_label_ratio": float((y[valid] >= thresholds["q90"]).mean()) if valid.any() else np.nan,
            "q10_fitted_development": thresholds["q10"],
            "q90_fitted_development": thresholds["q90"],
            "test_used_for_selection": False,
        })
    pd.DataFrame(rows).to_csv(output_dir / "phase_sample_and_label_audit.csv", index=False)


def _write_error_summary(errors: pd.DataFrame, output_dir: Path) -> None:
    if errors.empty:
        pd.DataFrame(columns=["error_type"]).to_csv(output_dir / "test_error_scenario_summary.csv", index=False)
        return
    numeric = errors.select_dtypes(include=[np.number]).columns.tolist()
    errors.groupby("error_type")[numeric].agg(["count", "mean", "median"]).to_csv(
        output_dir / "test_error_scenario_summary.csv"
    )


def _flatten_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {key: metric.get(key) for key in (
        "n", "n_signal", "n_event", "coverage", "precision", "recall",
        "event_prevalence", "precision_lift", "mean_o2o", "median_o2o",
        "signed_mean_o2o", "signed_median_o2o", "return_spread",
        "reverse_extreme_rate", "rank_ic", "pearson",
    )}


def run_extreme_side(
    prepared: PreparedResearch,
    version: str,
    meta: dict[str, Any],
    side: str,
    output_dir: str | Path,
    full_scores: np.ndarray,
    base_metadata: pd.DataFrame,
    bootstrap_draws: int = 0,
) -> dict[str, Any]:
    if side not in {"down", "up"}:
        raise ValueError("side must be down or up")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if full_scores.shape[0] != len(prepared.frame):
        raise AssertionError("score matrix row count does not match prepared frame")

    write_json(output_dir / "data_audit.json", prepared.data_audit)
    source_whitelist().to_csv(output_dir / "source_whitelist.csv", index=False)
    feature_lineage_table().to_csv(output_dir / "feature_lineage.csv", index=False)
    thresholds = base.label_thresholds(prepared.development)
    parameters = candidate_parameter_table(base_metadata, side, meta)
    parameters.to_csv(output_dir / "candidate_parameters.csv", index=False)
    base_metadata.to_csv(output_dir / "base_candidate_parameters.csv", index=False)
    score_surface_fingerprint = hashlib.sha256(np.round(full_scores, 8).tobytes()).hexdigest()

    dev_scores = full_scores[prepared.development.index.to_numpy(), :]
    val_scores = full_scores[prepared.validation.index.to_numpy(), :]
    candidates = _score_candidate_pool(
        prepared.development, prepared.validation, dev_scores, val_scores,
        parameters, side, thresholds,
    )
    candidates.to_csv(output_dir / "candidate_metrics_development_validation.csv", index=False)
    candidates.loc[candidates.is_unique_signal].head(20).to_csv(
        output_dir / "top20_development_validation.csv", index=False
    )

    pool = candidates.loc[candidates.is_unique_signal & candidates.eligible_strict_dv].copy()
    if pool.empty:
        reason = (
            "no unique Development+Validation candidate passing the strict DV gate "
            f"and coverage floor >= {MIN_COVERAGE_CONFIG:.3f}"
        )
        blocked = {
            "version": version,
            "side": side,
            "core_logic_name": meta.get("core_logic_name", ""),
            "title_zh": meta.get("title_zh", ""),
            "candidate_count_raw": int(len(parameters)),
            "candidate_count_unique_signal": int(candidates.is_unique_signal.sum()),
            "duplicate_signal_ratio": float(1 - candidates.is_unique_signal.mean()),
            "score_surface_fingerprint": score_surface_fingerprint,
            "frozen": False,
            "freeze_block_reason": reason,
            "pretest_research_pass": False,
            "frozen_test_diagnostic_pass": False,
            "formal_pass_after_frozen_test": False,
            "test_used_for_selection": False,
            "cross_version_test_comparison_before_freeze": False,
            "test_feedback_to_later_versions": False,
            "uses_1545": False,
            "uses_nine_state": False,
        }
        write_json(output_dir / "FREEZE_BLOCKED_BEFORE_TEST.json", blocked)
        write_json(output_dir / "summary.json", blocked)
        return blocked

    frozen = pool.sort_values(
        ["joint_score", "val_precision", "candidate_id"],
        ascending=[False, False, True], kind="mergesort",
    ).iloc[0]
    base_index = int(frozen.base_index)
    record = FrozenRecord(
        version=version,
        side=side,
        candidate_id=str(frozen.candidate_id),
        frozen_at_stage="development_validation",
        test_used_for_selection=False,
    )
    freeze_payload = {
        **record.__dict__,
        "frozen_first_before_test": True,
        "test_metrics_present_at_freeze_time": False,
        "cross_version_test_comparison_before_freeze": False,
        "test_feedback_to_later_versions": False,
        "label_thresholds_fitted_development": thresholds,
        "candidate_parameters": {
            key: frozen[key] for key in (
                "base_candidate_id", "base_index", "axis_1", "axis_2", "axis_3", "axis_4",
                "coverage_config", "score_threshold_fitted_development",
            ) if key in frozen.index
        },
        "coverage_floor_config": MIN_COVERAGE_CONFIG,
        "coverage_floor_is_pre_registered": True,
        "core_logic_name": meta.get("core_logic_name", ""),
        "candidate_count_raw": int(len(parameters)),
        "candidate_count_unique_signal": int(candidates.is_unique_signal.sum()),
        "duplicate_signal_ratio": float(1 - candidates.is_unique_signal.mean()),
        "frozen_signal_hash_dev_validation": str(frozen.signal_hash_dev_validation),
        "score_surface_fingerprint": score_surface_fingerprint,
    }
    # This write must precede TestVault.unlock.
    write_json(output_dir / "FROZEN_TOP1_BEFORE_TEST.json", freeze_payload)

    test = prepared.new_test_vault().unlock(record)
    dev_score = dev_scores[:, base_index]
    val_score = val_scores[:, base_index]
    test_score = full_scores[test.index.to_numpy(), base_index]
    dev_metrics, dev_active = base.evaluate_frozen_candidate(
        prepared.development, dev_score, frozen, side, thresholds
    )
    val_metrics, val_active = base.evaluate_frozen_candidate(
        prepared.validation, val_score, frozen, side, thresholds
    )
    test_metrics, test_active = base.evaluate_frozen_candidate(
        test, test_score, frozen, side, thresholds
    )
    pd.DataFrame([
        {"phase": "development", **dev_metrics},
        {"phase": "validation", **val_metrics},
        {"phase": "test_frozen_observation_only", **test_metrics},
    ]).to_csv(output_dir / "frozen_top1_three_phase_metrics.csv", index=False)

    bins: list[pd.DataFrame] = []
    for name, phase_frame, phase_score in (
        ("development", prepared.development, dev_score),
        ("validation", prepared.validation, val_score),
        ("test_frozen_observation_only", test, test_score),
    ):
        for count in (5, 10):
            table = base.score_bins(phase_frame, phase_score, side, bins=count)
            if not table.empty:
                table.insert(0, "group_count", count)
                table.insert(0, "phase", name)
                bins.append(table)
    (pd.concat(bins, ignore_index=True) if bins else pd.DataFrame()).to_csv(
        output_dir / "score_group_metrics_5_and_10.csv", index=False
    )

    observed = prepared.frame.copy()
    outcomes = ["future_open_to_open_return_1d", "future_close_to_close_return_1d"]
    observed.loc[test.index, outcomes] = test[outcomes]
    research = observed.loc[
        observed.date.ge("2018-01-01") & observed.future_open_to_open_return_1d.notna()
    ].copy()
    score_all = np.asarray(full_scores[:, base_index], float)
    threshold = float(frozen.score_threshold_fitted_development)
    active_all = np.isfinite(score_all) & (score_all >= threshold)
    full_metrics = base._phase_metrics(
        research,
        score_all[research.index.to_numpy()],
        active_all[research.index.to_numpy()],
        side,
        thresholds,
    )
    base.annual_metrics(
        research,
        score_all[research.index.to_numpy()],
        active_all[research.index.to_numpy()],
        side,
        thresholds,
    ).to_csv(output_dir / "annual_metrics.csv", index=False)

    boot_parts: list[pd.DataFrame] = []
    boot_summary: dict[str, Any] = {}
    for phase_number, (name, phase_frame, phase_active) in enumerate((
        ("development", prepared.development, dev_active),
        ("validation", prepared.validation, val_active),
        ("test_frozen_observation_only", test, test_active),
    ), start=1):
        samples = base.block_bootstrap_metrics(
            phase_frame,
            phase_active,
            side,
            thresholds,
            seed=20260814 + int(version[1:]) * 100 + phase_number * 10 + (0 if side == "down" else 1),
            draws=int(bootstrap_draws),
        )
        samples.insert(0, "phase", name)
        boot_parts.append(samples)
        boot_summary[name] = base.bootstrap_summary(samples)
    (pd.concat(boot_parts, ignore_index=True) if boot_parts else pd.DataFrame()).to_csv(
        output_dir / "bootstrap_samples.csv", index=False
    )
    write_json(output_dir / "bootstrap_summary.json", boot_summary)

    errors = base.error_diagnostics(test, test_score, test_active, side, thresholds)
    errors.to_csv(output_dir / "test_error_diagnostics.csv", index=False)
    _write_error_summary(errors, output_dir)
    predictions = base._prediction_frame(observed, score_all, active_all, side, threshold)
    predictions.to_csv(output_dir / "frozen_top1_daily_predictions.csv", index=False)
    predictions.tail(10).to_csv(output_dir / "latest_10_scores_and_alerts.csv", index=False)
    _phase_audit(prepared, test, thresholds, output_dir)

    dv_pass = _strict_pass(dev_metrics, "development") and _strict_pass(val_metrics, "validation")
    test_pass = _strict_pass(test_metrics, "test")
    labeled = predictions.loc[predictions.future_open_to_open_return_1d.notna()]
    latest_labeled = labeled.iloc[-1] if not labeled.empty else predictions.iloc[-1]
    summary = {
        "version": version,
        "side": side,
        "core_logic_name": meta.get("core_logic_name", ""),
        "title_zh": meta.get("title_zh", ""),
        "hypothesis": meta.get("hypothesis", ""),
        "paper_title": meta.get("paper_title", ""),
        "authors": meta.get("authors", ""),
        "year": meta.get("year"),
        "source_url": meta.get("source_url", ""),
        "candidate_count_raw": int(len(parameters)),
        "candidate_count_unique_signal": int(candidates.is_unique_signal.sum()),
        "eligible_unique_candidate_count": int(len(pool)),
        "duplicate_signal_ratio": float(1 - candidates.is_unique_signal.mean()),
        "score_surface_fingerprint": score_surface_fingerprint,
        "frozen": True,
        "frozen_candidate_id": str(frozen.candidate_id),
        "frozen_base_candidate_id": str(frozen.base_candidate_id),
        "frozen_score_threshold": threshold,
        "frozen_signal_hash_dev_validation": str(frozen.signal_hash_dev_validation),
        "label_thresholds_development": thresholds,
        "full_cycle": full_metrics,
        "development": dev_metrics,
        "validation": val_metrics,
        "test_frozen_observation_only": test_metrics,
        "pretest_research_pass": dv_pass,
        "frozen_test_diagnostic_pass": test_pass,
        "formal_pass_after_frozen_test": bool(dv_pass and test_pass),
        "test_used_for_selection": False,
        "cross_version_test_comparison_before_freeze": False,
        "test_feedback_to_later_versions": False,
        "uses_1545": False,
        "uses_nine_state": False,
        "pure_spot_only": True,
        "development_start": "2018-01-01",
        "bootstrap_draws": int(bootstrap_draws),
        "latest_formation_date": predictions.iloc[-1].date,
        "latest_effective_date": predictions.iloc[-1].entry_date,
        "latest_exit_date": predictions.iloc[-1].label_exit_date,
        "latest_fully_labeled_formation_date": latest_labeled.date,
        "latest_fully_labeled_effective_date": latest_labeled.entry_date,
        "latest_fully_labeled_exit_date": latest_labeled.label_exit_date,
    }
    write_json(output_dir / "summary.json", summary)
    del candidates, dev_scores, val_scores, full_scores
    gc.collect()
    return summary


def run_extreme_version(
    prepared: PreparedResearch,
    version: str,
    meta: dict[str, Any],
    version_dir: str | Path,
    score_provider: Any,
    bootstrap_draws: int = 0,
) -> dict[str, dict[str, Any]]:
    """Build the two independent sides from one version's spot score provider."""
    version_dir = Path(version_dir)
    up_scores, down_scores, base_metadata = score_provider()
    base_metadata = _normalise_base_meta(base_metadata)
    summaries: dict[str, dict[str, Any]] = {}
    for side, scores in (("down", down_scores), ("up", up_scores)):
        summaries[side] = run_extreme_side(
            prepared, version, meta, side, version_dir / "results" / side,
            scores, base_metadata, bootstrap_draws,
        )
    lines = [
        f"# {version}｜{meta.get('title_zh', meta.get('core_logic_name', ''))}",
        "",
        f"核心逻辑：{meta.get('core_logic_name', '')}",
        "输入：纯现货日频；大跌和大涨两侧独立候选、独立冻结。",
        "冻结顺序：Development+Validation → FROZEN_TOP1_BEFORE_TEST.json → Test 观察。",
        "",
        "|方向|候选数/去重|冻结候选|Dev precision lift|Val precision lift|Test precision lift（冻结后）|正式通过|",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for side, label in (("down", "大跌"), ("up", "大涨")):
        s = summaries[side]
        if not s.get("frozen"):
            lines.append(f"|{label}|{s.get('candidate_count_raw', 0)}/{s.get('candidate_count_unique_signal', 0)}|未冻结|—|—|—|否（{s.get('freeze_block_reason', '')}）|")
        else:
            lines.append(
                f"|{label}|{s['candidate_count_raw']}/{s['candidate_count_unique_signal']}|`{s['frozen_candidate_id']}`|"
                f"{s['development']['precision_lift']:.3f}|{s['validation']['precision_lift']:.3f}|"
                f"{s['test_frozen_observation_only']['precision_lift']:.3f}|"
                f"{'是' if s['formal_pass_after_frozen_test'] else '否'}|"
            )
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries
