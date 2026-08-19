"""Export the two independently frozen exits on an effective-date grid.

The company package keeps the scoring/freeze code in ``pool_runner``.  This
small adapter deliberately does not store a selected parameter or threshold:
it rebuilds both side pools from the current spot input, freezes each side on
Development+Validation, and then emits the effective-date five-column view.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from common.candidates import build_candidate_signal
from common.data import build_panel
from pool_runner import _candidate_spec, _side_pool


SIGNAL_COLUMNS = (
    "date",
    "three_state",
    "minus_exit_signal",
    "plus_exit_signal",
    "final_three_state",
)


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


def build_frozen_signal_export(
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


__all__ = ["SIGNAL_COLUMNS", "build_frozen_signal_export"]
