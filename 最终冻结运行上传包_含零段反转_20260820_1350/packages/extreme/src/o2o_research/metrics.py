from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _finite_corr(x: np.ndarray, y: np.ndarray, rank: bool = False) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10 or np.nanstd(x[mask]) <= 1e-15 or np.nanstd(y[mask]) <= 1e-15:
        return np.nan
    if rank:
        x_use, y_use = rankdata(x[mask], method="average"), rankdata(y[mask], method="average")
    else:
        x_use, y_use = x[mask], y[mask]
    return float(np.corrcoef(x_use, y_use)[0, 1])


def label_thresholds(development: pd.DataFrame) -> dict[str, float]:
    y = development.future_open_to_open_return_1d.astype(float)
    return {"q10": float(y.quantile(0.10)), "q90": float(y.quantile(0.90))}


def event_arrays(y: np.ndarray, side: str, thresholds: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, float)
    if side == "down":
        return y <= thresholds["q10"], y >= thresholds["q90"]
    return y >= thresholds["q90"], y <= thresholds["q10"]


def _phase_metrics(
    frame: pd.DataFrame,
    score: np.ndarray,
    active: np.ndarray,
    side: str,
    thresholds: dict[str, float],
) -> dict[str, float | int]:
    y = frame.future_open_to_open_return_1d.astype(float).to_numpy()
    valid = np.isfinite(y) & np.isfinite(score)
    active = np.asarray(active, bool) & valid
    event, reverse = event_arrays(y, side, thresholds)
    event &= valid
    reverse &= valid
    signed_y = -y if side == "down" else y
    n = int(valid.sum())
    n_signal = int(active.sum())
    n_event = int(event.sum())
    direction_hit = active & (signed_y > 0)
    direction_neutral = active & (signed_y == 0)
    non_extreme_signal = active & ~event
    non_extreme_direction_hit = non_extreme_signal & (signed_y > 0)
    precision = float((active & event).sum() / n_signal) if n_signal else np.nan
    recall = float((active & event).sum() / n_event) if n_event else np.nan
    prevalence = float(n_event / n) if n else np.nan
    selected = signed_y[active]
    nonselected = signed_y[valid & ~active]
    return {
        "n": n,
        "n_signal": n_signal,
        "n_event": n_event,
        "coverage": float(n_signal / n) if n else np.nan,
        "precision": precision,
        "recall": recall,
        "event_prevalence": prevalence,
        "precision_lift": float(precision / prevalence) if prevalence and np.isfinite(precision) else np.nan,
        # Directional precision is deliberately separate from extreme-event
        # precision: an alert is directionally correct whenever its next O2O
        # return has the registered side's sign, even if it is not q10/q90.
        "direction_hit_count": int(direction_hit.sum()),
        "direction_miss_count": int((active & (signed_y < 0)).sum()),
        "direction_neutral_count": int(direction_neutral.sum()),
        "direction_accuracy": float(direction_hit.sum() / n_signal) if n_signal else np.nan,
        "same_side_sign_prevalence": float((valid & (signed_y > 0)).sum() / n) if n else np.nan,
        "direction_excess_over_baseline": (
            float(direction_hit.sum() / n_signal - (valid & (signed_y > 0)).sum() / n)
            if n_signal and n else np.nan
        ),
        "non_extreme_signal_count": int(non_extreme_signal.sum()),
        "non_extreme_direction_hit_count": int(non_extreme_direction_hit.sum()),
        "non_extreme_direction_accuracy": float(
            non_extreme_direction_hit.sum() / non_extreme_signal.sum()
        ) if non_extreme_signal.sum() else np.nan,
        "mean_o2o": float(np.nanmean(y[active])) if n_signal else np.nan,
        "median_o2o": float(np.nanmedian(y[active])) if n_signal else np.nan,
        "signed_mean_o2o": float(np.nanmean(selected)) if n_signal else np.nan,
        "signed_median_o2o": float(np.nanmedian(selected)) if n_signal else np.nan,
        "return_spread": float(np.nanmean(selected) - np.nanmean(nonselected)) if n_signal and len(nonselected) else np.nan,
        "reverse_extreme_rate": float((active & reverse).sum() / n_signal) if n_signal else np.nan,
        "rank_ic": _finite_corr(score[valid], signed_y[valid], rank=True),
        "pearson": _finite_corr(score[valid], signed_y[valid], rank=False),
    }


def _joint_score(dev: dict[str, Any], val: dict[str, Any], scale: float) -> float:
    def finite(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if np.isfinite(result) else default

    dev_ic, val_ic = finite(dev["rank_ic"]), finite(val["rank_ic"])
    dev_lift, val_lift = finite(dev["precision_lift"], 1.0), finite(val["precision_lift"], 1.0)
    dev_mean, val_mean = finite(dev["signed_mean_o2o"]), finite(val["signed_mean_o2o"])
    reverse = finite(val["reverse_extreme_rate"], 1.0)
    coverage_drift = abs(finite(dev["coverage"]) - finite(val["coverage"]))
    score = (
        0.18 * np.tanh(5 * dev_ic)
        + 0.27 * np.tanh(5 * val_ic)
        + 0.15 * np.tanh((dev_lift - 1) / 1.5)
        + 0.20 * np.tanh((val_lift - 1) / 1.5)
        + 0.07 * np.tanh(dev_mean / max(scale, 1e-9))
        + 0.13 * np.tanh(val_mean / max(scale, 1e-9))
        - 0.05 * reverse
        - 0.05 * coverage_drift
    )
    # Wrong-signed validation means and very small samples are explicit, fixed penalties.
    if val_mean <= 0:
        score -= 0.12
    if int(val["n_signal"]) < 15:
        score -= 0.08
    return float(score)


def _signal_hash(dev_signal: np.ndarray, val_signal: np.ndarray) -> str:
    bits = np.concatenate([np.asarray(dev_signal, bool), np.asarray(val_signal, bool)])
    return hashlib.sha1(np.packbits(bits).tobytes()).hexdigest()


def score_candidate_pool(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    development_scores: np.ndarray,
    validation_scores: np.ndarray,
    parameters: pd.DataFrame,
    side: str,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    if development_scores.shape[1] != 256 or validation_scores.shape[1] != 256:
        raise AssertionError("candidate scoring expects 256 base score columns")
    y_scale = float(development.future_open_to_open_return_1d.std(ddof=0))
    base_index = {f"base_{i:03d}": i - 1 for i in range(1, 257)}
    rows: list[dict[str, Any]] = []
    for parameter in parameters.itertuples(index=False):
        j = base_index[parameter.base_candidate_id]
        dev_score = development_scores[:, j]
        val_score = validation_scores[:, j]
        threshold = float(np.nanquantile(dev_score, 1 - float(parameter.coverage_config)))
        dev_active = dev_score >= threshold
        val_active = val_score >= threshold
        dev = _phase_metrics(development, dev_score, dev_active, side, thresholds)
        val = _phase_metrics(validation, val_score, val_active, side, thresholds)
        row = parameter._asdict()
        row.update({f"dev_{k}": v for k, v in dev.items()})
        row.update({f"val_{k}": v for k, v in val.items()})
        row["score_threshold_fitted_development"] = threshold
        row["joint_score"] = _joint_score(dev, val, y_scale)
        row["signal_hash_dev_validation"] = _signal_hash(dev_active, val_active)
        row["eligible_min_samples"] = bool(dev["n_signal"] >= 20 and val["n_signal"] >= 15)
        rows.append(row)
    result = pd.DataFrame(rows)
    # Deduplicate by the actual Development+Validation signal, retaining the
    # pre-test best joint score with deterministic candidate-id tie breaking.
    result = result.sort_values(
        ["signal_hash_dev_validation", "joint_score", "candidate_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result["signal_duplicate_rank"] = result.groupby("signal_hash_dev_validation").cumcount()
    result["is_unique_signal"] = result.signal_duplicate_rank.eq(0)
    leader = result.loc[result.is_unique_signal, ["signal_hash_dev_validation", "candidate_id"]].rename(
        columns={"candidate_id": "signal_duplicate_of"}
    )
    result = result.merge(leader, on="signal_hash_dev_validation", how="left")
    result = result.sort_values(
        ["is_unique_signal", "eligible_min_samples", "joint_score", "val_precision", "candidate_id"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result["selection_rank"] = np.arange(1, len(result) + 1)
    return result


def select_top1(candidate_metrics: pd.DataFrame) -> pd.Series:
    pool = candidate_metrics.loc[candidate_metrics.is_unique_signal].copy()
    if pool.empty:
        raise RuntimeError("no calculable unique candidates")
    eligible = pool.loc[pool.eligible_min_samples]
    chosen = (eligible if not eligible.empty else pool).sort_values(
        ["joint_score", "val_precision", "candidate_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]
    return chosen


def evaluate_frozen_candidate(
    frame: pd.DataFrame,
    score: np.ndarray,
    frozen: pd.Series | dict[str, Any],
    side: str,
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], np.ndarray]:
    threshold = float(frozen["score_threshold_fitted_development"])
    active = np.asarray(score, float) >= threshold
    metrics = _phase_metrics(frame, np.asarray(score, float), active, side, thresholds)
    return metrics, active


def score_bins(frame: pd.DataFrame, score: np.ndarray, side: str, bins: int = 10) -> pd.DataFrame:
    y = frame.future_open_to_open_return_1d.astype(float).to_numpy()
    c2c = frame.future_close_to_close_return_1d.astype(float).to_numpy()
    valid = np.isfinite(y) & np.isfinite(score)
    if valid.sum() < bins * 3:
        return pd.DataFrame()
    rank_pct = pd.Series(score[valid]).rank(method="first", pct=True)
    bucket = pd.cut(rank_pct, bins=np.linspace(0, 1, bins + 1), labels=False, include_lowest=True) + 1
    work = pd.DataFrame({"bucket": bucket, "o2o": y[valid], "c2c": c2c[valid], "score": score[valid]})
    result = work.groupby("bucket", as_index=False).agg(
        n=("o2o", "size"), score_min=("score", "min"), score_max=("score", "max"),
        o2o_mean=("o2o", "mean"), o2o_median=("o2o", "median"),
        c2c_mean_observation_only=("c2c", "mean"), c2c_median_observation_only=("c2c", "median"),
    )
    result["signed_o2o_mean"] = -result.o2o_mean if side == "down" else result.o2o_mean
    ordered = result.sort_values("bucket")
    result["monotonicity_spearman"] = _finite_corr(
        ordered.bucket.to_numpy(float), ordered.signed_o2o_mean.to_numpy(float), rank=True
    )
    return result


def annual_metrics(
    frame: pd.DataFrame,
    score: np.ndarray,
    active: np.ndarray,
    side: str,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for year, part in frame.assign(_score=score, _active=active).groupby(frame.date.dt.year):
        metrics = _phase_metrics(part, part._score.to_numpy(), part._active.to_numpy(bool), side, thresholds)
        rows.append({"year": int(year), **metrics})
    return pd.DataFrame(rows)


def block_bootstrap_metrics(
    frame: pd.DataFrame,
    active: np.ndarray,
    side: str,
    thresholds: dict[str, float],
    seed: int,
    draws: int = 500,
    block_length: int = 20,
) -> pd.DataFrame:
    y = frame.future_open_to_open_return_1d.astype(float).to_numpy()
    active = np.asarray(active, bool)
    valid = np.isfinite(y)
    y, active = y[valid], active[valid]
    if not len(y):
        return pd.DataFrame()
    event, reverse = event_arrays(y, side, thresholds)
    signed = -y if side == "down" else y
    rng = np.random.default_rng(seed)
    blocks = [np.arange(start, min(start + block_length, len(y))) for start in range(0, len(y), block_length)]
    rows = []
    for draw in range(draws):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        index = np.concatenate([blocks[i] for i in chosen])[: len(y)]
        act = active[index]
        n_signal = int(act.sum())
        rows.append({
            "draw": draw + 1,
            "n_signal": n_signal,
            "signed_mean_o2o": float(np.mean(signed[index][act])) if n_signal else np.nan,
            "precision": float(np.mean(event[index][act])) if n_signal else np.nan,
            "reverse_extreme_rate": float(np.mean(reverse[index][act])) if n_signal else np.nan,
        })
    return pd.DataFrame(rows)


def bootstrap_summary(samples: pd.DataFrame) -> dict[str, Any]:
    if samples.empty:
        return {"draws": 0}
    return {
        "draws": int(len(samples)),
        "p_signed_mean_positive": float((samples.signed_mean_o2o > 0).mean()),
        "signed_mean_p05": float(samples.signed_mean_o2o.quantile(0.05)),
        "signed_mean_p50": float(samples.signed_mean_o2o.quantile(0.50)),
        "signed_mean_p95": float(samples.signed_mean_o2o.quantile(0.95)),
        "precision_p05": float(samples.precision.quantile(0.05)),
        "precision_p50": float(samples.precision.quantile(0.50)),
        "precision_p95": float(samples.precision.quantile(0.95)),
    }


def error_diagnostics(
    frame: pd.DataFrame,
    score: np.ndarray,
    active: np.ndarray,
    side: str,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    y = frame.future_open_to_open_return_1d.astype(float).to_numpy()
    event, reverse = event_arrays(y, side, thresholds)
    active = np.asarray(active, bool)
    kind = np.full(len(frame), "", dtype=object)
    kind[active & ~event] = "false_positive"
    kind[~active & event] = "missed_extreme"
    kind[active & reverse] = "reverse_extreme"
    keep = kind != ""
    cols = [
        "date", "entry_date", "label_exit_date", "future_open_to_open_return_1d",
        "future_close_to_close_return_1d", "ret_1", "ret_5", "ret_20", "vol_5", "vol_20",
        "close_location", "true_range_pct", "drawdown_60", "amount_ratio_20", "amihud",
    ]
    out = frame.loc[keep, [c for c in cols if c in frame]].copy()
    out.insert(3, "error_type", kind[keep])
    out["score"] = np.asarray(score)[keep]
    return out.sort_values(["error_type", "date"])
