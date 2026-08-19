"""Company-side runner for the DV-frozen cross-version full grids."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from common.candidates import CandidateSpec, build_candidate_signal, freeze_candidates
from common.data import DEV_END, VALID_END, build_panel
from common.diagnostics import post_freeze_diagnostics
from common.models import build_model_score_variants
from common.registry import BY_ID
from common.reserve_registry import RESERVE_BY_ID
from common.reserve_scores import build_reserve_score_variants
from common.scores import build_rule_score_variants, exit_sign, side_state
from pool_registry import POOL_RULE_ID, POOL_VERSION_SIDES


VERSION_SPECS = {**BY_ID, **RESERVE_BY_ID}


def _score_variants(panel: pd.DataFrame, version_id: str, side: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    spec = VERSION_SPECS[version_id]
    if version_id in BY_ID:
        if spec.group == "model":
            return build_model_score_variants(panel, version_id, side)
        return build_rule_score_variants(panel, version_id, side)
    return build_reserve_score_variants(panel, version_id, side)


def _period_mask(panel: pd.DataFrame, period: str) -> np.ndarray:
    index = pd.DatetimeIndex(panel.index)
    known = panel["exit_h1_date"].notna().to_numpy()
    if period in ("Full cycle", "Full", "全周期"):
        return known
    if period == "Development":
        return (index <= DEV_END) & known & panel["exit_h1_date"].le(DEV_END).to_numpy()
    if period == "Validation":
        return (index > DEV_END) & (index <= VALID_END) & known & panel["exit_h1_date"].le(VALID_END).to_numpy()
    if period == "Test":
        return (index > VALID_END) & known
    raise ValueError(period)


def _metrics(panel: pd.DataFrame, signal: pd.Series, side: str, period: str) -> dict[str, Any]:
    mask = _period_mask(panel, period)
    improvement = (exit_sign(side) * panel["o2o_h1"]).to_numpy(float)
    triggered = signal.to_numpy(bool) & mask & np.isfinite(improvement)
    eligible = panel["base_state"].eq(side_state(side)).to_numpy() & mask
    values = improvement[triggered]
    h3 = (exit_sign(side) * panel["o2o_h3"]).to_numpy(float)
    c2c = (exit_sign(side) * panel["c2c_obs"]).to_numpy(float)
    h3_values = h3[triggered & np.isfinite(h3)]
    c2c_values = c2c[triggered & np.isfinite(c2c)]
    return {
        "period": period,
        "n": int(triggered.sum()),
        "eligible_n": int(eligible.sum()),
        "coverage": float(triggered.sum() / eligible.sum()) if eligible.sum() else np.nan,
        "h1_improvement_bp": float(values.mean() * 10_000) if len(values) else np.nan,
        "h1_median_bp": float(np.median(values) * 10_000) if len(values) else np.nan,
        "h1_win_rate": float(np.mean(values > 0)) if len(values) else np.nan,
        "h3_improvement_bp": float(h3_values.mean() * 10_000) if len(h3_values) else np.nan,
        "c2c_observation_bp": float(c2c_values.mean() * 10_000) if len(c2c_values) else np.nan,
        "test_used_for_selection": False,
    }


def _year_metrics(panel: pd.DataFrame, signal: pd.Series, side: str) -> list[dict[str, Any]]:
    mask = _period_mask(panel, "Test")
    years = pd.DatetimeIndex(panel.index).year
    improvement = (exit_sign(side) * panel["o2o_h1"]).to_numpy(float)
    output: list[dict[str, Any]] = []
    for year in sorted(set(years[mask])):
        triggered = signal.to_numpy(bool) & mask & (years == year) & np.isfinite(improvement)
        values = improvement[triggered]
        output.append({
            "period": "Test",
            "year": int(year),
            "n": int(triggered.sum()),
            "mean_h1_improvement_bp": float(values.mean() * 10_000) if len(values) else np.nan,
            "win_rate": float(np.mean(values > 0)) if len(values) else np.nan,
        })
    return output


def _candidate_spec(row: pd.Series) -> CandidateSpec:
    return CandidateSpec(
        score_variant=str(row["score_variant"]),
        threshold_quantile=float(row["threshold_quantile"]),
        threshold_value=float(row["threshold_value"]),
        min_state_age=int(row["min_state_age"]),
        confirm_days=int(row["confirm_days"]),
        cooldown_days=int(row["cooldown_days"]),
    )


def _signal_hash(signal: np.ndarray) -> str:
    """Hash a binary signal with the same packing convention as candidates.py."""
    packed = np.packbits(np.asarray(signal, dtype=np.uint8), bitorder="little")
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _dv_age_baseline_hashes(panel: pd.DataFrame, dv_panel: pd.DataFrame, side: str) -> dict[int, str]:
    """Return DV-only hashes for the simple state-age baselines.

    This is a fixed Development+Validation guard used only to stop a version's
    Top1 from being a pure ``base_state + state_age`` switch.  It does not
    remove any rows from an admitted version's 2,592-row grid.
    """
    base = panel["base_state"].eq(side_state(side)).to_numpy()
    dv_mask = np.asarray(panel.index <= VALID_END, dtype=bool)
    return {
        age: _signal_hash((base & panel["state_age"].ge(age).to_numpy())[dv_mask])
        for age in (1, 3, 5, 8)
    }


def _version_dv_top1(
    frozen: dict[str, Any],
    baseline_hashes: dict[int, str],
) -> dict[str, Any] | None:
    """Freeze one version's Top1 using its own DV formal rule only."""
    metrics = frozen["unique_metrics"].copy()
    if metrics.empty:
        return None
    metrics["age_only_baseline"] = [
        baseline_hashes.get(int(age)) == digest
        for age, digest in zip(metrics["min_state_age"], metrics["signal_hash_dev_validation"])
    ]
    eligible = metrics.loc[
        metrics["formal_pass"].astype(bool)
        & ~metrics["age_only_baseline"].astype(bool)
        & metrics["joint_score"].notna()
    ].copy()
    if eligible.empty:
        return None
    eligible = eligible.sort_values(
        ["joint_score", "pooled_n", "candidate_id"],
        ascending=[False, False, True],
    )
    return eligible.iloc[0].to_dict()


def _version_top1_observation(
    panel: pd.DataFrame,
    scores: pd.DataFrame,
    top1: dict[str, Any] | None,
    version_id: str,
    side: str,
    spec: Any,
) -> dict[str, Any] | None:
    """Evaluate one version's own DV-frozen Top1 after it is frozen.

    This table is descriptive only.  It is deliberately computed after the
    version-local DV Top1 exists and is never used by the cross-version pool
    ranking or by the final company-side Top1.
    """
    if top1 is None:
        return None
    signal = pd.Series(
        build_candidate_signal(
            panel,
            scores[str(top1["score_variant"])],
            _candidate_spec(pd.Series(top1)),
            side,
        ),
        index=panel.index,
        name="exit_to_zero",
    )
    periods = {
        period: _metrics(panel, signal, side, period)
        for period in ("Full cycle", "Development", "Validation", "Test")
    }
    row: dict[str, Any] = {
        "version": version_id,
        "side": side,
        "core_logic_name": spec.core_logic_name,
        "logic_title_cn": getattr(spec, "title_cn", spec.core_logic_name),
        "logic_family": getattr(spec, "group", getattr(spec, "family", "")),
        "literature": getattr(spec, "literature", ""),
        "candidate_id": top1["candidate_id"],
        "score_variant": top1["score_variant"],
        "threshold_quantile": top1["threshold_quantile"],
        "threshold_value": top1["threshold_value"],
        "min_state_age": top1["min_state_age"],
        "confirm_days": top1["confirm_days"],
        "cooldown_days": top1["cooldown_days"],
        "joint_score": top1["joint_score"],
        "signal_hash_dev_validation": top1["signal_hash_dev_validation"],
        "test_used_for_selection": False,
        "observation_only": True,
    }
    for period, metrics in periods.items():
        prefix = period.lower().replace(" ", "_")
        row[f"{prefix}_n"] = metrics["n"]
        row[f"{prefix}_h1_improvement_bp"] = metrics["h1_improvement_bp"]
        row[f"{prefix}_h1_median_bp"] = metrics["h1_median_bp"]
        row[f"{prefix}_h1_win_rate"] = metrics["h1_win_rate"]
        row[f"{prefix}_h3_improvement_bp"] = metrics["h3_improvement_bp"]
        row[f"{prefix}_c2c_observation_bp"] = metrics["c2c_observation_bp"]
    return row


def _side_pool(panel: pd.DataFrame, side: str, progress=None) -> dict[str, Any]:
    versions = tuple(POOL_VERSION_SIDES[side])
    if not versions:
        raise AssertionError(f"{side} 侧没有预注册版本")
    score_frames: dict[str, pd.DataFrame] = {}
    metric_frames: list[pd.DataFrame] = []
    raw_count = 0
    metadata: list[dict[str, Any]] = []
    version_top1_test: list[dict[str, Any]] = []
    dv_panel = panel.loc[panel.index <= VALID_END].copy()
    baseline_hashes = _dv_age_baseline_hashes(panel, dv_panel, side)
    for version_id in versions:
        if progress is not None:
            progress(f"    {version_id}: 计算评分变体和 2,592 条候选...")
        spec = VERSION_SPECS[version_id]
        scores, score_meta = _score_variants(panel, version_id, side)
        scores = scores.reindex(panel.index)
        if list(scores.columns) != [f"score_{i:02d}" for i in range(1, 9)]:
            raise AssertionError(f"{version_id}/{side} 必须有 8 个评分变体")
        score_frames[version_id] = scores
        frozen = freeze_candidates(
            dv_panel,
            scores.loc[dv_panel.index],
            side=side,
            version_id=version_id,
            core_logic_name=spec.core_logic_name,
        )
        raw_count += len(frozen["raw_registry"])
        version_top1 = _version_dv_top1(frozen, baseline_hashes)
        version_observation = _version_top1_observation(
            panel,
            scores,
            version_top1,
            version_id,
            side,
            spec,
        )
        if version_observation is not None:
            version_top1_test.append(version_observation)
        metrics = frozen["unique_metrics"].copy()
        metrics.insert(0, "version", version_id)
        metrics.insert(1, "side", side)
        metrics.insert(2, "core_logic_name", spec.core_logic_name)
        metrics.insert(3, "logic_title_cn", getattr(spec, "title_cn", spec.core_logic_name))
        metrics.insert(4, "logic_family", getattr(spec, "group", getattr(spec, "family", "")))
        metrics.insert(5, "literature", getattr(spec, "literature", ""))
        metrics["pool_rule_id"] = POOL_RULE_ID
        metrics["pool_membership_uses_test"] = False
        metrics["raw_candidate_grid_included"] = True
        metrics["score_metadata"] = [json_safe(score_meta)] * len(metrics)
        metric_frames.append(metrics)
        metadata.append({
            "version": version_id,
            "side": side,
            "core_logic_name": spec.core_logic_name,
            "raw_candidate_count": int(len(frozen["raw_registry"])),
            "deduplicated_within_version_count": int(len(frozen["unique_metrics"])),
            "version_dv_top1_candidate_id": None if version_top1 is None else version_top1["candidate_id"],
            "test_used_for_selection": False,
        })
        if progress is not None:
            if version_observation is None:
                progress(
                    f"    {version_id}: 完成；原始候选 {len(frozen['raw_registry'])}，"
                    f"DV 去重信号 {len(frozen['unique_metrics'])}，无 DV 正式通过版内 Top1"
                )
            else:
                progress(
                    f"    {version_id}: 完成；原始候选 {len(frozen['raw_registry'])}，"
                    f"DV 去重信号 {len(frozen['unique_metrics'])}；"
                    f"版内 Test n={version_observation['test_n']}，"
                    f"H1={version_observation['test_h1_improvement_bp']:.2f}bp"
                )

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_metrics = all_metrics.sort_values(
        ["joint_score", "pooled_n", "candidate_id", "version"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    all_metrics["cross_version_signal_provenance_count"] = all_metrics.groupby(
        "signal_hash_dev_validation"
    )["candidate_id"].transform("size")
    all_metrics["cross_version_signal_duplicate"] = all_metrics["cross_version_signal_provenance_count"].gt(1)
    unique_metrics = all_metrics.drop_duplicates(
        ["signal_hash_dev_validation"], keep="first"
    ).reset_index(drop=True)
    # The final company-side freeze is also DV-only.  A candidate must satisfy
    # the same formal Development/Validation gate; Test is never consulted
    # here.  Full grids remain untouched in the pool regardless of this rank.
    # Final company Top1 is DV-only and must be a genuine score/rule
    # response, not the state-age baseline in disguise. The full raw grid
    # remains intact for provenance; only the Top1 ranking applies this guard.
    all_metrics["age_only_baseline"] = [
        baseline_hashes.get(int(age)) == digest
        for age, digest in zip(all_metrics["min_state_age"], all_metrics["signal_hash_dev_validation"])
    ]
    ranked = all_metrics.loc[
        all_metrics["formal_pass"].astype(bool)
        & ~all_metrics["age_only_baseline"].astype(bool)
        & all_metrics["joint_score"].notna()
    ].copy()
    ranked = ranked.drop_duplicates(
        ["signal_hash_dev_validation"], keep="first"
    ).reset_index(drop=True)
    ranked = ranked.sort_values(
        ["joint_score", "pooled_n", "candidate_id", "version"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    if ranked.empty:
        return {
            "versions": versions,
            "metadata": metadata,
            "score_frames": score_frames,
            "raw_candidate_count": raw_count,
            "metric_rows": int(len(all_metrics)),
            "unique_signal_rows": int(len(unique_metrics)),
            "ranked": ranked,
            "top20": ranked,
            "top1": None,
            "version_top1_test": version_top1_test,
        }
    top1 = ranked.iloc[0].to_dict()
    top1.update({
        "pool_rule_id": POOL_RULE_ID,
        "side": side,
        "action": "-1→0" if side == "minus" else "1→0",
        "selection_data_end": str(VALID_END.date()),
        "model_fit_period": "Development only",
        "threshold_fit_period": "company Development only",
        "ranking_period": "company Development+Validation",
        "test_used_for_selection": False,
    })
    return {
        "versions": versions,
        "metadata": metadata,
        "score_frames": score_frames,
        "raw_candidate_count": raw_count,
        "metric_rows": int(len(all_metrics)),
        "unique_signal_rows": int(len(unique_metrics)),
        "ranked": ranked,
        "top20": ranked.head(20),
        "top1": top1,
        "version_top1_test": version_top1_test,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def json_safe(value: Any) -> Any:
    return _jsonable(value)


def run_company_side(side: str, verbose: bool = True) -> dict[str, Any]:
    """Rebuild one side's large pool, freeze its company Top1, then show Test.

    ``verbose=True`` is deliberately the notebook default: the company-side
    run can take a while, so each major stage and version is printed with
    flushing for remote, screenshot-based operation.
    """
    if side not in ("minus", "plus"):
        raise ValueError("side must be minus or plus")
    def progress(message: str) -> None:
        if verbose:
            print(f"[1545] {message}", flush=True)

    progress("1/5 读取唯一现货输入并重算八状态...")
    panel, input_manifest = build_panel()
    progress(
        f"1/5 完成：{len(panel)} 个形成日，"
        f"{panel.index.min().date()} 至 {panel.index.max().date()}"
    )
    progress(f"2/5 构建 {side} 侧跨版本全量候选池：{','.join(POOL_VERSION_SIDES[side])}")
    pool = _side_pool(panel, side, progress=progress)
    progress(
        f"2/5 完成：原始候选 {pool['raw_candidate_count']}，"
        f"跨版本去重后信号 {pool['unique_signal_rows']}"
    )
    if pool["top1"] is None:
        progress("3/5 DV 冻结失败：没有可计算的候选")
        return {
            "status": "NO_COMPUTABLE_CANDIDATE",
            "side": side,
            "pool_rule_id": POOL_RULE_ID,
            "test_used_for_selection": False,
            "input_audit": input_manifest,
            "pool": {key: pool[key] for key in ("versions", "metadata", "raw_candidate_count", "metric_rows", "unique_signal_rows")},
            "version_top1_test": pool["version_top1_test"],
        }
    progress("3/5 已按 Development+Validation 冻结 Top1；Test 尚未参与")
    top1 = pool["top1"]
    version_id = str(top1["version"])
    signal = build_candidate_signal(
        panel,
        pool["score_frames"][version_id][str(top1["score_variant"])],
        _candidate_spec(pd.Series(top1)),
        side,
    )
    signal = pd.Series(signal, index=panel.index, name="exit_to_zero")
    progress("4/5 计算冻结信号的全周期及 Development/Validation/Test 指标...")
    periods = [_metrics(panel, signal, side, period) for period in ("Development", "Validation", "Test")]
    full_cycle = _metrics(panel, signal, side, "Full cycle")
    progress("5/5 计算冻结后诊断和 Test 年度拆分...")
    diagnostics = {
        period: post_freeze_diagnostics(panel, signal, side, _period_mask(panel, period))
        for period in ("Development", "Validation", "Test")
    }
    recent = panel.loc[signal.astype(bool), [
        "effective_date", "exit_h1_date", "base_state", "state_age",
        "o2o_h1", "o2o_h3", "c2c_obs", "distance_to_natural_switch",
    ]].tail(30).copy()
    recent.insert(0, "formation_date", recent.index)
    recent.insert(1, "side", side)
    return {
        "status": "COMPLETED",
        "side": side,
        "pool_rule_id": POOL_RULE_ID,
        "test_used_for_selection": False,
        "input_audit": input_manifest,
        "pool": {
            "versions": pool["versions"],
            "metadata": pool["metadata"],
            "raw_candidate_count": pool["raw_candidate_count"],
            "metric_rows_before_cross_version_dedup": pool["metric_rows"],
            "unique_signal_rows": pool["unique_signal_rows"],
            "ranked_rows": int(len(pool["ranked"])),
        },
        "top20": pool["top20"],
        "version_top1_test": pool["version_top1_test"],
        "freeze": top1,
        "full_cycle": full_cycle,
        "periods": periods,
        "year_metrics": _year_metrics(panel, signal, side),
        "diagnostics": diagnostics,
        "recent_signals": recent.reset_index(drop=True),
    }


__all__ = ["run_company_side", "json_safe", "POOL_RULE_ID", "POOL_VERSION_SIDES"]
