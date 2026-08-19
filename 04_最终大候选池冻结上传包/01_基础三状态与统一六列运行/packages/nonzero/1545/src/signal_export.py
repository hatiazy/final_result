"""Export the two already-frozen exits on an effective-date grid.

The runtime path stores the final V55/V80 score variants, thresholds and
state-age/confirmation rules. It computes those frozen scores from the new
spot input and emits the effective-date five-column view without candidate
pool rebuilding or remote reselection.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from common.candidates import build_candidate_signal
from common.candidates import CandidateSpec
from common.data import DEV_END, build_panel
from common.scores import side_state
from pool_runner import _score_variants


SIGNAL_COLUMNS = (
    "date",
    "three_state",
    "minus_exit_signal",
    "plus_exit_signal",
    "final_three_state",
)

FROZEN_NONZERO_PARAMETERS: dict[str, dict[str, Any]] = {
    "minus": {
        "version": "V55",
        "score_variant": "score_02",
        "threshold_quantile": 0.68,
        "threshold_value": 0.815842643776842,
        "min_state_age": 1,
        "confirm_days": 2,
        "cooldown_days": 0,
    },
    "plus": {
        "version": "V80",
        "score_variant": "score_02",
        "threshold_quantile": 0.90,
        "threshold_value": 0.4920211306764882,
        "min_state_age": 5,
        "confirm_days": 1,
        "cooldown_days": 0,
    },
}


def _event_exit_state(
    base_state: pd.Series,
    minus_signal: pd.Series,
    plus_signal: pd.Series,
) -> pd.Series:
    """Set only the signal date to zero; restore the base state afterwards."""
    state = pd.to_numeric(base_state, errors="raise").astype("int8")
    minus = minus_signal.astype(bool).to_numpy()
    plus = plus_signal.astype(bool).to_numpy()
    original = state.to_numpy()
    output = original.copy().astype(np.int8)
    for pos, value in enumerate(original):
        value = int(value)
        if value == -1 and minus[pos]:
            output[pos] = 0
        elif value == 1 and plus[pos]:
            output[pos] = 0
    return pd.Series(output, index=state.index, dtype="int8")


def _assert_single_forward_shift(panel: pd.DataFrame, effective_date: pd.Series) -> None:
    """Fail closed if any signal is not placed strictly after its formation close."""
    effective = pd.to_datetime(effective_date, errors="coerce")
    formation = pd.DatetimeIndex(pd.to_datetime(panel.index, errors="coerce"))
    if effective.isna().any() or formation.isna().any() or len(effective) != len(formation):
        raise RuntimeError("信号执行日映射存在无效日期")
    if not np.all(effective.to_numpy() > formation.to_numpy()):
        raise RuntimeError("信号执行日不是形成日之后的下一执行日，拒绝输出")


def _build_candidate_pool_signal_export(
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Rebuild both side Top1s and return the effective-date five-column view.

    ``three_state`` is the state observed at formation close ``t`` and is
    placed on ``date=t+1`` because that is the actual implementation/opening
    date.  The final state uses event-day exits: only a date carrying an
    actual side-specific exit signal is set to zero; later dates follow the
    base three-state value again.
    """

    panel, input_manifest = build_panel()
    pools: dict[str, Any] = {}
    signals: dict[str, pd.Series] = {}
    for side in ("minus", "plus"):
        if progress is not None:
            progress(f"构建 {side} 侧候选池并冻结 Development+Validation Top1...")
        pool = _side_pool(panel, side, progress=progress)
        if pool["top1"] is None:
            raise RuntimeError(f"{side} 侧没有可冻结的正式候选")
        top1 = pool["top1"]
        scores = pool["score_frames"][str(top1["version"])]
        score = scores[str(top1["score_variant"])]
        signal = build_candidate_signal(
            panel,
            score,
            _candidate_spec(pd.Series(top1)),
            side,
        )
        signals[side] = pd.Series(signal, index=panel.index, dtype=bool)
        pools[side] = pool

    effective_date = pd.to_datetime(panel["effective_date"], errors="coerce")
    # The latest formation row has no t+1 spot row yet, so data.py quite
    # correctly leaves its effective_date as NaT for performance alignment.
    # It is nevertheless a valid next-trading-day signal to export: when the
    # input ends at Friday 2026-08-14, the five-column signal must include
    # Monday 2026-08-17 rather than the weekend date 2026-08-15.
    pending = effective_date.isna()
    if pending.any():
        pending_dates = effective_date.index[pending.to_numpy()]
        if len(pending_dates) != 1 or pending_dates[-1] != panel.index[-1]:
            raise RuntimeError(
                "导出层只允许最后一条形成日缺少 t+1 effective_date；"
                f"实际待执行行={pending_dates.tolist()}"
            )
        effective_date.loc[pending_dates[0]] = (
            pd.Timestamp(pending_dates[0]) + pd.offsets.BDay(1)
        )
    _assert_single_forward_shift(panel, effective_date)
    output = pd.DataFrame(
        {
            "date": effective_date,
            "three_state": panel["base_state"].astype("int8"),
            "minus_exit_signal": signals["minus"].astype("int8"),
            "plus_exit_signal": signals["plus"].astype("int8"),
        },
        index=panel.index,
    )
    output = output.loc[output["date"].notna()].copy()
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    output["final_three_state"] = _event_exit_state(
        output["three_state"],
        output["minus_exit_signal"],
        output["plus_exit_signal"],
    ).to_numpy(dtype=np.int8)
    output = output.loc[:, SIGNAL_COLUMNS].reset_index(drop=True)
    output["three_state"] = output["three_state"].astype("int8")
    output["minus_exit_signal"] = output["minus_exit_signal"].astype("int8")
    output["plus_exit_signal"] = output["plus_exit_signal"].astype("int8")
    output["final_three_state"] = output["final_three_state"].astype("int8")
    metadata = {
        "input_manifest": input_manifest,
        "freeze": {
            side: pools[side]["top1"]
            for side in ("minus", "plus")
        },
        "pool_summary": {
            side: {
                "versions": pools[side]["versions"],
                "raw_candidate_count": pools[side]["raw_candidate_count"],
                "metric_rows_before_cross_version_dedup": pools[side]["metric_rows"],
                "unique_signal_rows": pools[side]["unique_signal_rows"],
                "ranked_rows": int(len(pools[side]["ranked"])),
            }
            for side in ("minus", "plus")
        },
        "date_role": "date is the next actual spot trading date after formation close; "
        "the latest pending formation row uses the next weekday until that row is available",
        "three_state_role": "base_state at formation close, aligned to its effective date",
        "exit_state_policy": "event-day only: signal date becomes 0; subsequent dates restore the base state",
        "test_used_for_selection": False,
    }
    return output, metadata, pools


def build_frozen_signal_export(
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Generate the five-column export from V55/V80 frozen parameters only."""
    panel, input_manifest = build_panel()
    signals: dict[str, pd.Series] = {}
    freezes: dict[str, dict[str, Any]] = {}
    score_metadata: dict[str, Any] = {}
    for side in ("minus", "plus"):
        cfg = dict(FROZEN_NONZERO_PARAMETERS[side])
        version = str(cfg["version"])
        if progress is not None:
            progress(f"读取冻结 {version}；只计算最终 score_02，不重建候选池")
        scores, metadata = _score_variants(panel, version, side)
        score = scores[str(cfg["score_variant"])]
        population_mask = (
            panel["base_state"].eq(side_state(side))
            & panel.index.to_series().le(DEV_END)
            & panel["exit_h1_date"].le(DEV_END)
            & score.notna()
        )
        observed_threshold = float(score.loc[population_mask].quantile(float(cfg["threshold_quantile"])))
        if not np.isclose(observed_threshold, float(cfg["threshold_value"]), rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"{version} 冻结阈值校验失败：expected={cfg['threshold_value']}, observed={observed_threshold}"
            )
        spec = CandidateSpec(
            score_variant=str(cfg["score_variant"]),
            threshold_quantile=float(cfg["threshold_quantile"]),
            threshold_value=float(cfg["threshold_value"]),
            min_state_age=int(cfg["min_state_age"]),
            confirm_days=int(cfg["confirm_days"]),
            cooldown_days=int(cfg["cooldown_days"]),
        )
        signal = build_candidate_signal(panel, score, spec, side)
        signals[side] = pd.Series(signal, index=panel.index, dtype=bool)
        freezes[side] = {
            **cfg,
            "candidate_id": spec.candidate_id,
            "side": side,
            "test_used_for_selection": False,
            "candidate_grid_rebuilt": False,
        }
        score_metadata[side] = next(
            (item for item in metadata if item.get("score_variant") == str(cfg["score_variant"])),
            {},
        )
        if progress is not None:
            progress(f"{version} 冻结信号完成；事件数={int(signal.sum())}")

    effective_date = pd.to_datetime(panel["effective_date"], errors="coerce")
    pending = effective_date.isna()
    if pending.any():
        pending_dates = effective_date.index[pending.to_numpy()]
        if len(pending_dates) != 1 or pending_dates[-1] != panel.index[-1]:
            raise RuntimeError("导出层只允许最后一条形成日缺少 t+1 effective_date")
        effective_date.loc[pending_dates[0]] = pd.Timestamp(pending_dates[0]) + pd.offsets.BDay(1)
    _assert_single_forward_shift(panel, effective_date)
    output = pd.DataFrame({
        "date": effective_date,
        "three_state": panel["base_state"].astype("int8"),
        "minus_exit_signal": signals["minus"].astype("int8"),
        "plus_exit_signal": signals["plus"].astype("int8"),
    }, index=panel.index)
    output = output.loc[output["date"].notna()].copy()
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    output["final_three_state"] = _event_exit_state(
        output["three_state"],
        output["minus_exit_signal"],
        output["plus_exit_signal"],
    ).to_numpy(dtype=np.int8)
    output = output.loc[:, SIGNAL_COLUMNS].reset_index(drop=True)
    for column in SIGNAL_COLUMNS[1:]:
        output[column] = output[column].astype("int8")
    metadata_out = {
        "input_manifest": input_manifest,
        "freeze": freezes,
        "score_metadata": score_metadata,
        "pool_summary": {
            side: {
                "mode": "frozen_parameters_only",
                "version": freezes[side]["version"],
                "candidate_id": freezes[side]["candidate_id"],
                "raw_candidate_count": 1,
                "candidate_grid_rebuilt": False,
                "reselection_performed": False,
            }
            for side in ("minus", "plus")
        },
        "date_role": "date is the next actual spot trading date after formation close; the latest pending formation row uses the next weekday until that row is available",
        "three_state_role": "base_state at formation close, aligned to its effective date",
        "exit_state_policy": "event-day only: signal date becomes 0; subsequent dates restore the base state",
        "test_used_for_selection": False,
        "candidate_grid_rebuilt": False,
    }
    return output, metadata_out, freezes


__all__ = ["SIGNAL_COLUMNS", "build_frozen_signal_export"]
