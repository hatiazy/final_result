from __future__ import annotations

"""Strict company-side runner for the pure-spot binary candidate package.

The package contains the historical candidate-pool helpers for provenance,
but the upload/runtime entry point below is a reproduction path: it reads
the already-frozen V156/V189 parameters and computes only those scores. Test
is unlocked only after the frozen parameters are applied and is never used
for reselection.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from o2o_research.candidates import candidate_parameter_table as spot_parameter_table
from o2o_research.candidates import score_matrix as spot_score_matrix
from o2o_research.data import FrozenRecord
from o2o_research.extreme_engine import MIN_COVERAGE_CONFIG
from o2o_research.extreme_engine import _score_candidate_pool as paper_score_pool
from o2o_research.extreme_engine import candidate_parameter_table as paper_parameter_table
from o2o_research.metrics import (
    _phase_metrics,
    annual_metrics,
    evaluate_frozen_candidate,
    label_thresholds,
    score_candidate_pool as spot_score_pool,
    score_bins,
)
from o2o_research.paper_engine import load_paper_metadata, score_matrix as paper_score_matrix
from o2o_research.paper_engine import _banks as paper_banks
from o2o_research.pipeline import prepare_research, write_json
from o2o_research.specs import get_version_spec
from pool_registry import POOL_RULE_ID, POOL_SELECTION_RULE, POOL_VERSION_SIDES
try:
    from pool_registry import POOL_SELECTION_RULE_SIDES
except ImportError:  # Backward-compatible import for pre-side-rule packages.
    POOL_SELECTION_RULE_SIDES = {}


PACKAGE_SRC = Path(__file__).resolve().parent
_metadata_path = PACKAGE_SRC / "paper_metadata.json"
if not _metadata_path.is_file():
    raise FileNotFoundError("bundled paper metadata not found")
PAPER_METADATA = load_paper_metadata(_metadata_path)

# The package uses exact DV hashes first, then a deterministic near-duplicate
# audit on the candidate rows that can affect the frozen ranking.  These are
# deliberately fixed here rather than tuned after looking at Test.
NEAR_DUPLICATE_JACCARD = 0.80
NEAR_DUPLICATE_OVERLAP_OF_SMALLER = 0.90
NEAR_DUPLICATE_AUDIT_TOP_K = 2048
MIN_TEST_SIGNAL_AUDIT = 20
# Secondary directional sanity gate.  It is evaluated only on the two
# pre-Test freeze periods; Test remains observation-only.  The comparison is
# against the phase's unconditional same-side sign prevalence so a bullish
# or bearish base rate is not mistaken for signal skill.
MIN_DIRECTION_ACCURACY = 0.50
MIN_DIRECTION_EXCESS_OVER_BASELINE = 0.03

# These are the already-frozen production/report parameters. The remote
# reproduction path must not rerun candidate selection or rebuild V158/V168/
# V211. Only the selected base score and its frozen threshold are evaluated.
FROZEN_EXTREME_PARAMETERS: dict[str, dict[str, Any]] = {
    "down": {
        "version": "V156",
        "base_candidate_id": "base_0621",
        "base_index": 620,
        "coverage_config": 0.075,
        "score_threshold_fitted_development": 0.6832148298881413,
    },
    "up": {
        "version": "V189",
        "base_candidate_id": "base_1839",
        "base_index": 1838,
        "coverage_config": 0.055,
        "score_threshold_fitted_development": 0.8135114753699175,
    },
}


def _version_number(version: str) -> int:
    return int(str(version)[1:])


def _base_index(row: pd.Series) -> int:
    if "score_column_index" in row and pd.notna(row["score_column_index"]):
        return int(row["score_column_index"])
    if "base_index" in row and pd.notna(row["base_index"]):
        return int(row["base_index"])
    return int(str(row["base_candidate_id"]).split("_")[-1]) - 1


def _frozen_paper_score(frame: pd.DataFrame, meta: dict[str, Any], side: str, base_index: int) -> np.ndarray:
    """Compute one frozen paper base score, not a 4,096-column grid."""
    if not 0 <= int(base_index) < 4096:
        raise ValueError(f"paper base_index out of range: {base_index}")
    banks = paper_banks(frame, str(meta["core_logic_name"]), side)
    if len(banks) != 4 or any(bank.shape != (len(frame), 8) for bank in banks):
        raise AssertionError(f"paper primitive banks must be (n,8): {[bank.shape for bank in banks]}")
    number = int(base_index)
    axes = (number // 512, (number // 64) % 8, (number // 8) % 8, number % 8)
    score = (
        0.34 * banks[0][:, axes[0]]
        + 0.28 * banks[1][:, axes[1]]
        + 0.22 * banks[2][:, axes[2]]
        + 0.16 * banks[3][:, axes[3]]
    )
    return np.clip(score, 0, 1)[:, None]


def _candidate_mask_int(
    prepared: Any,
    row: pd.Series,
    score_cache: dict[str, np.ndarray],
) -> tuple[int, int]:
    """Encode one DV alert vector compactly for fast overlap checks."""
    version = str(row["version"])
    base_j = _base_index(row)
    scores = score_cache[version]
    threshold = float(row["score_threshold_fitted_development"])
    di = prepared.development.index.to_numpy()
    vi = prepared.validation.index.to_numpy()
    ds = scores[di, base_j]
    vs = scores[vi, base_j]
    active = np.concatenate([
        np.isfinite(ds) & (ds >= threshold),
        np.isfinite(vs) & (vs >= threshold),
    ])
    packed = np.packbits(active.astype(np.uint8), bitorder="little")
    value = int.from_bytes(packed.tobytes(), byteorder="little", signed=False)
    return value, int(active.sum())


def _near_duplicate_stats(left: tuple[int, int], right: tuple[int, int]) -> tuple[float, float]:
    a, na = left
    b, nb = right
    intersection = (a & b).bit_count()
    union = (a | b).bit_count()
    smaller = min(na, nb)
    return (
        float(intersection / union) if union else 1.0,
        float(intersection / smaller) if smaller else 1.0,
    )


def _near_duplicate_gate(
    ranked: pd.DataFrame,
    prepared: Any,
    score_cache: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop near-duplicate rows from the deterministic ranking prefix.

    Exact DV signal hashes already remove identical alert vectors.  This second
    gate checks a fixed prefix of the strict-DV ranking with the registered
    Jaccard/overlap thresholds.  The prefix is intentionally large enough to
    cover all rows that can be displayed or selected by the package; rows past
    it remain reserve diagnostics rather than being silently promoted over a
    checked row.
    """
    result = ranked.copy().reset_index(drop=True)
    for column, default in (
        ("near_duplicate_of", ""),
        ("near_duplicate_jaccard", np.nan),
        ("near_duplicate_overlap_of_smaller", np.nan),
        ("near_duplicate_checked", False),
    ):
        result[column] = default
    if result.empty:
        return result, {
            "jaccard_threshold": NEAR_DUPLICATE_JACCARD,
            "overlap_of_smaller_threshold": NEAR_DUPLICATE_OVERLAP_OF_SMALLER,
            "audit_top_k": NEAR_DUPLICATE_AUDIT_TOP_K,
            "audited_rows": 0,
            "filtered_rows": 0,
            "pairs": [],
        }

    limit = min(len(result), NEAR_DUPLICATE_AUDIT_TOP_K)
    kept: list[tuple[str, tuple[int, int]]] = []
    drop_indices: list[int] = []
    pairs: list[dict[str, Any]] = []
    for i in range(limit):
        row = result.iloc[i]
        key = str(row.get("candidate_key", f"{row['version']}:{row['candidate_id']}"))
        mask = _candidate_mask_int(prepared, row, score_cache)
        result.at[i, "near_duplicate_checked"] = True
        duplicate = None
        for prior_key, prior_mask in kept:
            jaccard, overlap = _near_duplicate_stats(mask, prior_mask)
            if jaccard >= NEAR_DUPLICATE_JACCARD and overlap >= NEAR_DUPLICATE_OVERLAP_OF_SMALLER:
                duplicate = (prior_key, jaccard, overlap)
                break
        if duplicate is None:
            kept.append((key, mask))
        else:
            prior_key, jaccard, overlap = duplicate
            result.at[i, "near_duplicate_of"] = prior_key
            result.at[i, "near_duplicate_jaccard"] = jaccard
            result.at[i, "near_duplicate_overlap_of_smaller"] = overlap
            drop_indices.append(i)
            pairs.append({
                "candidate_key": key,
                "near_duplicate_of": prior_key,
                "jaccard": jaccard,
                "overlap_of_smaller": overlap,
            })
    if drop_indices:
        result = result.drop(index=drop_indices).reset_index(drop=True)
    return result, {
        "jaccard_threshold": NEAR_DUPLICATE_JACCARD,
        "overlap_of_smaller_threshold": NEAR_DUPLICATE_OVERLAP_OF_SMALLER,
        "audit_top_k": NEAR_DUPLICATE_AUDIT_TOP_K,
        "audited_rows": limit,
        "filtered_rows": len(drop_indices),
        "pairs": pairs[:100],
    }


def _strict_dv_frame(metrics: pd.DataFrame) -> pd.Series:
    """Require both Development and Validation to pass before Test unlock."""
    def finite(name: str, default: float) -> pd.Series:
        return pd.to_numeric(metrics[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    direction_gate = _direction_sanity_frame(metrics)
    return (
        metrics.dev_n_signal.ge(20)
        & metrics.val_n_signal.ge(20)
        & pd.to_numeric(metrics.coverage_config, errors="coerce").ge(MIN_COVERAGE_CONFIG)
        & finite("dev_precision_lift", 0.0).ge(1.5)
        & finite("val_precision_lift", 0.0).ge(1.5)
        & finite("dev_signed_mean_o2o", 0.0).gt(0.0)
        & finite("val_signed_mean_o2o", 0.0).gt(0.0)
        & finite("dev_rank_ic", 0.0).gt(0.0)
        & finite("val_rank_ic", 0.0).gt(0.0)
        & finite("dev_reverse_extreme_rate", 1.0).le(0.20)
        & finite("val_reverse_extreme_rate", 1.0).le(0.20)
        & direction_gate
    )


def _direction_sanity_frame(metrics: pd.DataFrame) -> pd.Series:
    """Require pointwise directional skill over each DV phase's sign base rate."""
    def finite(name: str, default: float) -> pd.Series:
        return pd.to_numeric(metrics[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)

    dev_acc = finite("dev_direction_accuracy", 0.0)
    val_acc = finite("val_direction_accuracy", 0.0)
    dev_base = finite("dev_same_side_sign_prevalence", 1.0)
    val_base = finite("val_same_side_sign_prevalence", 1.0)
    dev_required = np.maximum(MIN_DIRECTION_ACCURACY, dev_base + MIN_DIRECTION_EXCESS_OVER_BASELINE)
    val_required = np.maximum(MIN_DIRECTION_ACCURACY, val_base + MIN_DIRECTION_EXCESS_OVER_BASELINE)
    return dev_acc.ge(dev_required) & val_acc.ge(val_required)


def _selection_rule(side: str) -> str:
    return str(POOL_SELECTION_RULE_SIDES.get(side, POOL_SELECTION_RULE))


def _pool_sort_columns(side: str) -> tuple[list[str], list[bool]]:
    """Return the pre-registered cross-version DV ranking rule."""
    rule = _selection_rule(side)
    if rule == "precision_stability_v1":
        # Accuracy is primary, but a candidate must also clear a stronger
        # two-period stability gate (both Dev and Val lift >= 2.0).  This is
        # fixed before Test and prevents a one-slice Validation spike from
        # displacing a candidate with positive evidence in both periods.
        return (
            ["eligible_precision_stable", "eligible_min_samples", "val_precision", "dev_precision", "val_precision_lift", "joint_score", "version", "candidate_id"],
            [False, False, False, False, False, False, True, True],
        )
    if rule == "validation_precision_priority_v1":
        # Accuracy is the stated objective.  Strict Development/Validation
        # gates remain mandatory; validation precision is only the primary
        # ordering after those gates, never a Test-derived quantity.
        return (
            ["eligible_strict_dv", "eligible_min_samples", "val_precision", "dev_precision", "val_precision_lift", "joint_score", "version", "candidate_id"],
            [False, False, False, False, False, False, True, True],
        )
    if rule == "precision_floor_priority_v1":
        # A balanced accuracy objective: both Development and Validation
        # precision must be high.  This avoids selecting a one-slice spike.
        return (
            ["eligible_strict_dv", "eligible_min_samples", "precision_floor", "val_precision", "dev_precision", "val_precision_lift", "joint_score", "version", "candidate_id"],
            [False, False, False, False, False, False, False, True, True],
        )
    return (
        ["eligible_strict_dv", "eligible_min_samples", "joint_score", "val_precision", "version", "candidate_id"],
        [False, False, False, False, True, True],
    )


def _score_candidate_version(
    prepared: Any,
    version: str,
    side: str,
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, np.ndarray, int, str]:
    number = _version_number(version)
    if number <= 50:
        spec = get_version_spec(version)
        parameters = spot_parameter_table(spec, side)
        full_scores, _ = spot_score_matrix(prepared.frame, spec, side)
        dev_scores = full_scores[prepared.development.index.to_numpy(), :]
        val_scores = full_scores[prepared.validation.index.to_numpy(), :]
        metrics = spot_score_pool(prepared.development, prepared.validation, dev_scores, val_scores, parameters, side, thresholds)
        logic_name = spec.core_logic_name
        # The old spot helper used a 20/15 eligibility rule.  The new package
        # deliberately tightens it to 20/20 without changing the raw grid.
        metrics["eligible_min_samples"] = (metrics.dev_n_signal >= 20) & (metrics.val_n_signal >= 20)
    elif number >= 135:
        meta = PAPER_METADATA[version]
        full_scores, base_meta = paper_score_matrix(prepared.frame, meta, side)
        parameters = paper_parameter_table(base_meta, side, meta)
        dev_scores = full_scores[prepared.development.index.to_numpy(), :]
        val_scores = full_scores[prepared.validation.index.to_numpy(), :]
        metrics = paper_score_pool(prepared.development, prepared.validation, dev_scores, val_scores, parameters, side, thresholds)
        logic_name = meta["core_logic_name"]
    else:
        raise ValueError(f"version {version} is not in the strict package registry")
    metrics = metrics.copy()
    metrics.insert(0, "candidate_key", [f"{version}:{side}:{x}" for x in metrics.candidate_id])
    metrics["version"] = version
    metrics["side"] = side
    metrics["core_logic_name"] = logic_name
    metrics["pool_rule_id"] = POOL_RULE_ID
    metrics["pool_membership_uses_test"] = False
    metrics["eligible_direction_sanity"] = _direction_sanity_frame(metrics)
    metrics["eligible_strict_dv"] = _strict_dv_frame(metrics)
    metrics["eligible_precision_stable"] = (
        metrics["eligible_strict_dv"]
        & pd.to_numeric(metrics["dev_precision_lift"], errors="coerce").ge(2.0)
        & pd.to_numeric(metrics["val_precision_lift"], errors="coerce").ge(2.0)
    )
    metrics["precision_floor"] = np.minimum(
        pd.to_numeric(metrics["dev_precision"], errors="coerce").fillna(0.0),
        pd.to_numeric(metrics["val_precision"], errors="coerce").fillna(0.0),
    )
    return metrics, full_scores, int(len(parameters)), logic_name


def _build_side_pool(
    prepared: Any,
    side: str,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    versions = tuple(POOL_VERSION_SIDES[side])
    rule = _selection_rule(side)
    thresholds = label_thresholds(prepared.development)
    frames: list[pd.DataFrame] = []
    score_cache: dict[str, np.ndarray] = {}
    raw_count = 0
    for position, version in enumerate(versions, start=1):
        if progress:
            progress(f"候选池 {position}/{len(versions)}：重建 {version} 全部候选")
        metrics, scores, raw_n, _ = _score_candidate_version(prepared, version, side, thresholds)
        frames.append(metrics)
        score_cache[version] = scores
        raw_count += raw_n
        if progress:
            progress(f"候选池 {position}/{len(versions)}：{version} 完成，候选数={raw_n}")
    all_metrics = pd.concat(frames, ignore_index=True)
    sort_columns, sort_ascending = _pool_sort_columns(side)
    all_metrics = all_metrics.sort_values(sort_columns, ascending=sort_ascending, kind="mergesort").reset_index(drop=True)
    all_metrics["signal_provenance_count"] = all_metrics.groupby("signal_hash_dev_validation")["candidate_key"].transform("size")
    all_metrics["cross_version_signal_provenance_count"] = all_metrics.groupby("signal_hash_dev_validation")["version"].transform("nunique")
    all_metrics["cross_version_signal_duplicate"] = all_metrics.cross_version_signal_provenance_count.gt(1)
    unique = all_metrics.drop_duplicates("signal_hash_dev_validation", keep="first").reset_index(drop=True)
    eligibility_column = "eligible_precision_stable" if rule == "precision_stability_v1" else "eligible_strict_dv"
    ranked = unique.loc[unique[eligibility_column].astype(bool)].copy()
    if rule in {"validation_precision_priority_v1", "precision_stability_v1"}:
        ranked = ranked.sort_values(
            ["val_precision", "dev_precision", "val_precision_lift", "joint_score", "version", "candidate_id"],
            ascending=[False, False, False, False, True, True], kind="mergesort",
        ).reset_index(drop=True)
    elif rule == "precision_floor_priority_v1":
        ranked = ranked.sort_values(
            ["precision_floor", "val_precision", "dev_precision", "val_precision_lift", "joint_score", "version", "candidate_id"],
            ascending=[False, False, False, False, False, True, True], kind="mergesort",
        ).reset_index(drop=True)
    else:
        ranked = ranked.sort_values(
            ["joint_score", "val_precision", "version", "candidate_id"],
            ascending=[False, False, True, True], kind="mergesort",
        ).reset_index(drop=True)
    ranked, near_duplicate_audit = _near_duplicate_gate(ranked, prepared, score_cache)
    if ranked.empty:
        return {
            "versions": versions,
            "raw_candidate_count": raw_count,
            "unique_signal_rows": len(unique),
            "ranked": ranked,
            "top1": None,
            "score_cache": score_cache,
            "thresholds": thresholds,
            "logic_top1_candidates": [],
            "near_duplicate_audit": near_duplicate_audit,
        }

    logic_top1: list[dict[str, Any]] = []
    for (version, logic_name), version_metrics in all_metrics.groupby(["version", "core_logic_name"], sort=True):
        version_unique = version_metrics.drop_duplicates("signal_hash_dev_validation", keep="first")
        eligible = version_unique.loc[version_unique[eligibility_column].astype(bool)]
        if eligible.empty:
            continue
        if rule in {"validation_precision_priority_v1", "precision_stability_v1"}:
            chosen = eligible.sort_values(["val_precision", "dev_precision", "val_precision_lift", "joint_score", "candidate_id"], ascending=[False, False, False, False, True], kind="mergesort").iloc[0]
        elif rule == "precision_floor_priority_v1":
            chosen = eligible.sort_values(["precision_floor", "val_precision", "dev_precision", "val_precision_lift", "joint_score", "candidate_id"], ascending=[False, False, False, False, False, True], kind="mergesort").iloc[0]
        else:
            chosen = eligible.sort_values(["joint_score", "val_precision", "candidate_id"], ascending=[False, False, True], kind="mergesort").iloc[0]
        family = all_metrics.loc[all_metrics.signal_hash_dev_validation.eq(chosen.signal_hash_dev_validation)]
        record = chosen.to_dict()
        record.update({
            "logic_selection_rank": 1,
            "cross_version_duplicate_versions": sorted({str(x) for x in family.version}),
            "cross_version_signal_duplicate": len(set(family.version)) > 1,
            "global_ranked_pool_member": str(chosen.candidate_key) in set(ranked.candidate_key.astype(str)),
        })
        logic_top1.append(record)
    return {
        "versions": versions,
        "selection_rule": rule,
        "raw_candidate_count": raw_count,
        "unique_signal_rows": len(unique),
        "ranked": ranked,
        "top1": ranked.iloc[0],
        "score_cache": score_cache,
        "thresholds": thresholds,
        "logic_top1_candidates": logic_top1,
        "near_duplicate_audit": near_duplicate_audit,
    }


def _phase_result(
    prepared: Any,
    side: str,
    top1: pd.Series,
    score_cache: dict[str, np.ndarray],
    thresholds: dict[str, float],
    freeze_artifact: Path,
) -> dict[str, Any]:
    version = str(top1.version)
    record = FrozenRecord(version, side, str(top1.candidate_id), "development_validation", False)
    freeze_payload = {
        **record.__dict__,
        "frozen_first_before_test": True,
        "test_metrics_present_at_freeze_time": False,
        "test_used_for_selection": False,
        "selection_data": "company Development+Validation only",
        "threshold_fit": "company Development only",
        "direction_sanity_gate": {
            "min_accuracy": MIN_DIRECTION_ACCURACY,
            "min_excess_over_same_side_baseline": MIN_DIRECTION_EXCESS_OVER_BASELINE,
            "evaluated_on": "Development and Validation only",
            "test_used_for_selection": False,
        },
        "candidate_parameters": {
            str(key): value
            for key, value in top1.to_dict().items()
            if str(key) in {
                "candidate_id", "base_candidate_id", "base_index", "coverage_config",
                "score_threshold_fitted_development", "core_logic_name", "version", "side",
                "signal_hash_dev_validation", "joint_score",
            }
        },
    }
    # This must succeed before TestVault.unlock.  A read-only or invalid output
    # location therefore fails closed instead of silently reducing the freeze
    # to an in-memory flag.
    write_json(freeze_artifact, freeze_payload)
    test = prepared.new_test_vault().unlock(record)
    base_j = _base_index(top1)
    scores = score_cache[version]
    dev_score = scores[prepared.development.index.to_numpy(), base_j]
    val_score = scores[prepared.validation.index.to_numpy(), base_j]
    test_score = scores[test.index.to_numpy(), base_j]
    dev, dev_active = evaluate_frozen_candidate(prepared.development, dev_score, top1, side, thresholds)
    val, val_active = evaluate_frozen_candidate(prepared.validation, val_score, top1, side, thresholds)
    test_metrics, test_active = evaluate_frozen_candidate(test, test_score, top1, side, thresholds)
    observed = prepared.frame.copy()
    observed.loc[test.index, ["future_open_to_open_return_1d", "future_close_to_close_return_1d"]] = test[["future_open_to_open_return_1d", "future_close_to_close_return_1d"]]
    research = observed.loc[observed.date.ge("2018-01-01") & observed.future_open_to_open_return_1d.notna()].copy()
    full_score = scores[:, base_j]
    full_active = full_score >= float(top1.score_threshold_fitted_development)
    full = _phase_metrics(research, full_score[research.index.to_numpy()], full_active[research.index.to_numpy()], side, thresholds)
    annual = annual_metrics(research, full_score[research.index.to_numpy()], full_active[research.index.to_numpy()], side, thresholds)
    research_score = full_score[research.index.to_numpy()]
    research_active = full_active[research.index.to_numpy()]
    target = research.future_open_to_open_return_1d.astype(float).to_numpy()
    actual_event = target <= float(thresholds["q10"]) if side == "down" else target >= float(thresholds["q90"])
    phase = np.select(
        [research.date.dt.year.between(2018, 2022), research.date.dt.year.between(2023, 2024), research.date.dt.year.ge(2025)],
        ["development", "validation", "test"],
        default="other",
    )
    plot = research.loc[:, ["date", "close", "future_open_to_open_return_1d"]].copy()
    plot["date"] = plot.date.dt.strftime("%Y-%m-%d")
    plot["phase"] = phase
    plot["score"] = research_score
    plot["predicted"] = research_active
    plot["actual_extreme"] = actual_event
    plot["correct"] = research_active & actual_event
    # Keep ordinary direction hits visible separately from q10/q90 extreme
    # hits.  The latter remains the historical `correct` field used for
    # extreme-event precision; this field answers whether O2O merely had the
    # predicted sign.
    plot["direction_correct"] = research_active & ((target > 0) if side == "up" else (target < 0))
    plot["direction_neutral"] = research_active & (target == 0)
    plot["prediction_side"] = np.where(research_active, side, "none")
    plot["o2o_bp"] = target * 10000.0
    plot["signed_o2o_bp"] = (target if side == "up" else -target) * 10000.0
    plot["marker_color"] = np.where(~research_active, "none", np.where(actual_event, "correct", "incorrect"))
    del dev_active, val_active, test_active
    return {
        "full_cycle": full,
        "development": dev,
        "validation": val,
        "test_frozen_observation_only": test_metrics,
        "coverage_floor_config": MIN_COVERAGE_CONFIG,
        "test_sample_size_audit": {
            "minimum_n_signal": MIN_TEST_SIGNAL_AUDIT,
            "n_signal": int(test_metrics.get("n_signal", 0)),
            "pass": bool(int(test_metrics.get("n_signal", 0)) >= MIN_TEST_SIGNAL_AUDIT),
            "test_used_for_selection": False,
        },
        "annual_metrics": annual.to_dict(orient="records"),
        "plot_data": plot.to_dict(orient="records"),
        "test_used_for_selection": False,
        "frozen_first_before_test": True,
        "freeze_artifact_path": str(freeze_artifact),
    }


def _run_candidate_pool_company_side(side: str, input_path: str | Path | None = None, progress: bool = True) -> dict[str, Any]:
    if side not in {"down", "up"}:
        raise ValueError("side must be down or up")

    def emit(message: str) -> None:
        if progress:
            print(f"[O2O {side} {datetime.now():%H:%M:%S}] {message}", flush=True)

    emit("1/6 读取唯一现货并构造因果特征")
    prepared = prepare_research(input_path)
    thresholds = label_thresholds(prepared.development)
    emit(f"1/6 完成：rows={len(prepared.frame)}，Development={len(prepared.development)}（起点固定 2018-01-01），Validation={len(prepared.validation)}")
    emit("2/6 在公司 Development＋Validation 重建候选池（Test 尚未打开）")
    pool = _build_side_pool(prepared, side, emit if progress else None)
    if pool["top1"] is None:
        return {"status": "NO_COMPUTABLE_CANDIDATE", "side": side, "pool_rule_id": POOL_RULE_ID, "test_used_for_selection": False, "input_audit": prepared.data_audit, "pool": {"versions": pool["versions"], "raw_candidate_count": pool["raw_candidate_count"], "unique_signal_rows": pool["unique_signal_rows"]}}
    top1 = pool["top1"]
    emit(f"3/6 DV 冻结 Top1：{top1.version}:{top1.candidate_id}；现在才打开 Test")
    runtime_root = PACKAGE_SRC.parent / "runtime_outputs"
    global_freeze_artifact = runtime_root / f"FROZEN_TOP1_BEFORE_TEST_{side}.json"
    periods = _phase_result(prepared, side, top1, pool["score_cache"], thresholds, global_freeze_artifact)
    logic_observations: list[dict[str, Any]] = []
    emit(f"4/6 逐逻辑 Top1 冻结后观察：{len(pool['logic_top1_candidates'])} 个")
    for candidate in pool["logic_top1_candidates"]:
        logic = pd.Series(candidate)
        safe_version = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(logic.get("version", "version")))
        safe_candidate = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(logic.get("candidate_id", "candidate")))
        logic_freeze_artifact = runtime_root / "logic_freezes" / f"FROZEN_{side}_{safe_version}_{safe_candidate}.json"
        candidate_periods = _phase_result(prepared, side, logic, pool["score_cache"], thresholds, logic_freeze_artifact)
        logic_observations.append({key: candidate[key] for key in ("version", "core_logic_name", "candidate_id", "coverage_config", "joint_score", "cross_version_duplicate_versions", "global_ranked_pool_member") if key in candidate} | {"test_used_for_selection": False, **{key: candidate_periods[key] for key in ("full_cycle", "development", "validation", "test_frozen_observation_only")}})
    emit("5/6 全周期与分周期结果已生成")
    freeze = top1.to_dict()
    freeze.update({"pool_rule_id": POOL_RULE_ID, "side": side, "selection_rule": _selection_rule(side), "selection_data": "company Development+Validation only", "threshold_fit": "company Development only", "test_used_for_selection": False, "freeze_artifact_path": str(global_freeze_artifact), "coverage_floor_config": MIN_COVERAGE_CONFIG, "coverage_floor_is_pre_registered": True, "direction_sanity_gate": {"min_accuracy": MIN_DIRECTION_ACCURACY, "min_excess_over_same_side_baseline": MIN_DIRECTION_EXCESS_OVER_BASELINE, "evaluated_on": "Development and Validation only", "test_used_for_selection": False}})
    return {
        "status": "COMPLETED",
        "side": side,
        "pool_rule_id": POOL_RULE_ID,
        "selection_rule": _selection_rule(side),
        "test_used_for_selection": False,
        "input_audit": prepared.data_audit,
        "pool": {
            "versions": list(pool["versions"]),
            "raw_candidate_count": pool["raw_candidate_count"],
            "unique_signal_rows": pool["unique_signal_rows"],
            "ranked_rows": len(pool["ranked"]),
            "near_duplicate_audit": pool.get("near_duplicate_audit", {}),
            "logic_diagnostics": [{"version": x["version"], "core_logic_name": x["core_logic_name"], "candidate_id": x["candidate_id"], "global_ranked_pool_member": x["global_ranked_pool_member"]} for x in logic_observations],
        },
        "top20": pool["ranked"].head(20).to_dict(orient="records"),
        "freeze": freeze,
        "periods": periods,
        "logic_top1": logic_observations,
    }


def run_company_side(side: str, input_path: str | Path | None = None, progress: bool = True) -> dict[str, Any]:
    """Reproduce one side from its already-frozen parameters only."""
    if side not in FROZEN_EXTREME_PARAMETERS:
        raise ValueError("side must be down or up")

    def emit(message: str) -> None:
        if progress:
            print(f"[O2O {side} {datetime.now():%H:%M:%S}] {message}", flush=True)

    frozen = dict(FROZEN_EXTREME_PARAMETERS[side])
    version = str(frozen["version"])
    meta = PAPER_METADATA[version]
    candidate_id = f"{frozen['base_candidate_id']}_cov_{float(frozen['coverage_config']):.3f}"
    emit("1/4 读取唯一现货并构造因果特征")
    prepared = prepare_research(input_path)
    thresholds = label_thresholds(prepared.development)
    emit(f"1/4 完成：rows={len(prepared.frame)}，Development={len(prepared.development)}，Validation={len(prepared.validation)}")
    emit(f"2/4 只计算已冻结参数：{version}:{candidate_id}；不重建候选池")
    score = _frozen_paper_score(prepared.frame, meta, side, int(frozen["base_index"]))
    top1 = pd.Series({
        **frozen,
        "candidate_id": candidate_id,
        "version": version,
        "side": side,
        "core_logic_name": meta["core_logic_name"],
        "score_column_index": 0,
        "selection_rule": "previously_frozen_parameters_v1",
        "test_used_for_selection": False,
    })
    runtime_root = PACKAGE_SRC.parent / "runtime_outputs"
    freeze_artifact = runtime_root / f"FROZEN_FINAL_ONLY_{side}.json"
    emit("3/4 用固定阈值生成冻结预测，之后才读取 Test")
    periods = _phase_result(prepared, side, top1, {version: score}, thresholds, freeze_artifact)
    freeze = top1.to_dict()
    freeze.update({
        "frozen_first_before_test": True,
        "selection_data": "previously_frozen_parameters; no remote reselection",
        "threshold_fit": "previously_frozen Development threshold",
        "test_used_for_selection": False,
        "freeze_artifact_path": str(freeze_artifact),
    })
    emit("4/4 冻结参数逐日结果已生成")
    return {
        "status": "COMPLETED",
        "side": side,
        "pool_rule_id": "frozen_parameters_reproduction_v1",
        "selection_rule": "previously_frozen_parameters_v1",
        "test_used_for_selection": False,
        "input_audit": prepared.data_audit,
        "pool": {
            "mode": "frozen_parameters_only",
            "versions": [version],
            "raw_candidate_count": 1,
            "candidate_id": candidate_id,
            "candidate_grid_rebuilt": False,
            "reselection_performed": False,
        },
        "top20": [freeze],
        "freeze": freeze,
        "periods": periods,
        "logic_top1": [],
    }


__all__ = ["run_company_side", "POOL_RULE_ID"]
