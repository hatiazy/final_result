"""Export the event-only frozen three-state signal on an effective-date grid.

This adapter is intentionally separate from the eight-state engine.  It
rebuilds the two side-specific candidate pools from the one raw spot input,
freezes each side on Development+Validation, and then exports a compact
five-column view.  Only the actual entry-signal day can change a base-zero
state; later days in the same base-zero run remain zero unless they have a
new entry signal of their own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from company_runner import run_company_side
from spot_panel import load_spot_panel


SIGNAL_COLUMNS = (
    "date",
    "three_state",
    "minus_entry_signal",
    "plus_entry_signal",
    "final_three_state",
)

ProgressFn = Callable[[str], None]


def _event_adjusted_state(
    base_state: np.ndarray,
    minus_signal: np.ndarray,
    plus_signal: np.ndarray,
) -> np.ndarray:
    """Apply only same-day entry events to base-zero rows.

    A simultaneous down/up signal is kept neutral rather than manufacturing a
    direction.  No signal is carried through the remainder of a zero run.
    """

    base = np.asarray(base_state, dtype=int)
    minus = np.asarray(minus_signal, dtype=bool)
    plus = np.asarray(plus_signal, dtype=bool)
    if not (len(base) == len(minus) == len(plus)):
        raise ValueError("base state and entry signals must have the same length")
    output = base.copy()
    conflict = minus & plus
    output[minus & ~conflict & (base == 0)] = -1
    output[plus & ~conflict & (base == 0)] = 1
    return output.astype(np.int8)


def _aligned_entry_signal(
    path: Path,
    formation_dates: pd.DatetimeIndex,
) -> np.ndarray:
    """Read one generated side signal and align it to the research panel."""

    if not path.is_file():
        raise FileNotFoundError(
            f"missing frozen side signal {path}; run the corresponding side freeze first"
        )
    frame = pd.read_parquet(path)
    required = {"formation_date", "entry_signal"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    dates = pd.to_datetime(frame["formation_date"], errors="raise")
    if dates.duplicated().any():
        raise ValueError(f"{path.name} contains duplicate formation dates")
    if set(dates) != set(formation_dates):
        raise ValueError(f"{path.name} does not cover exactly the current spot panel dates")
    values = pd.to_numeric(frame["entry_signal"], errors="raise")
    if not values.isin([0, 1]).all():
        raise ValueError(f"{path.name} entry_signal must contain only 0/1 values")
    aligned = (
        pd.DataFrame({"formation_date": dates, "entry_signal": values.astype("int8")})
        .set_index("formation_date")
        .reindex(formation_dates)
    )
    if aligned["entry_signal"].isna().any():
        raise ValueError(f"{path.name} has an unalignable formation date")
    return aligned["entry_signal"].to_numpy(dtype=np.int8).astype(bool)


def _effective_dates(panel: pd.DataFrame) -> tuple[pd.Series, bool]:
    """Return next-row dates, filling only the final pending date for export."""

    dates = pd.to_datetime(panel["effective_date"], errors="coerce").copy()
    pending = dates.isna()
    if not pending.any():
        return dates, False
    pending_positions = np.flatnonzero(pending.to_numpy())
    if len(pending_positions) != 1 or pending_positions[0] != len(dates) - 1:
        raise RuntimeError(
            "only the final formation row may lack an effective date for export"
        )
    dates.iloc[-1] = pd.Timestamp(panel["formation_date"].iloc[-1]) + pd.offsets.BDay(1)
    return dates, True


def build_frozen_event_signal_export(
    spot_path: str | Path,
    output_dir: str | Path,
    *,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]]:
    """Rebuild both independent freezes and export the event-only five columns.

    The two calls to :func:`run_company_side` are independent: each scans its
    own registered full grid and selects on Development+Validation only.  Test
    metrics are computed by the side runner after freeze and are never used by
    this export layer.
    """

    source = Path(spot_path).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    def progress(message: str) -> None:
        if show_progress:
            print(f"[event-export] {message}", flush=True)

    progress("读取唯一现货文件并重算八状态/基础三状态")
    spot, panel, spot_audit = load_spot_panel(source)
    freezes: dict[str, dict[str, Any]] = {}
    for side, side_dir in (("down", root / "01_down"), ("up", root / "02_up")):
        progress(f"独立冻结 {side} 侧候选池（Test 保持锁定至冻结后）")
        freezes[side] = run_company_side(
            side,
            spot,
            panel,
            side_dir,
            show_progress=show_progress,
        )

    formation_dates = pd.DatetimeIndex(
        pd.to_datetime(panel["formation_date"], errors="raise")
    )
    minus_signal = _aligned_entry_signal(root / "company_signal_down.parquet", formation_dates)
    plus_signal = _aligned_entry_signal(root / "company_signal_up.parquet", formation_dates)
    effective, pending_effective_date = _effective_dates(panel)
    base = pd.to_numeric(panel["state"], errors="raise").to_numpy(dtype=np.int8)
    final = _event_adjusted_state(base, minus_signal, plus_signal)
    conflict = (base == 0) & minus_signal & plus_signal

    output = pd.DataFrame(
        {
            "date": effective.dt.strftime("%Y-%m-%d"),
            "three_state": base,
            "minus_entry_signal": minus_signal.astype(np.int8),
            "plus_entry_signal": plus_signal.astype(np.int8),
            "final_three_state": final,
        }
    ).loc[:, SIGNAL_COLUMNS]

    freeze_summary = {
        side: {
            "candidate_id": freeze["candidate_id"],
            "source_version": freeze["source_version"],
            "core_logic_name": freeze["core_logic_name"],
            "method_key": freeze["method_key"],
            "test_used_for_selection": bool(freeze["test_used_for_selection"]),
        }
        for side, freeze in freezes.items()
    }
    metadata: dict[str, Any] = {
        "input_file": str(source),
        "input_contract": "one_raw_spot_file_only",
        "spot_audit": spot_audit,
        "freeze": freeze_summary,
        "date_role": "formation close t mapped to the next actual spot row; final pending row uses the next weekday for display",
        "three_state_role": "base state computed from the remote-aligned eight-state recipes",
        "event_state_policy": "only a same-day entry signal changes base 0 to -1/+1; no carry through later zero days",
        "same_day_conflict_policy": "both signals on a base-zero day remain final_three_state=0",
        "same_day_base_zero_conflict_count": int(conflict.sum()),
        "pending_effective_date_filled": bool(pending_effective_date),
        "state_counts_base": {
            str(value): int(np.sum(base == value)) for value in (-1, 0, 1)
        },
        "state_counts_final": {
            str(value): int(np.sum(final == value)) for value in (-1, 0, 1)
        },
        "signal_counts": {
            "minus_entry_days": int(minus_signal.sum()),
            "plus_entry_days": int(plus_signal.sum()),
        },
        "test_used_for_selection": False,
        "columns": list(SIGNAL_COLUMNS),
        "csv_file": "company_event_three_state_signal.csv",
        "json_file": "company_event_three_state_signal.json",
    }
    csv_path = root / metadata["csv_file"]
    json_path = root / metadata["json_file"]
    output.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    progress(f"五列事件式三状态已输出：{csv_path}")
    return output, metadata, freezes


__all__ = ["SIGNAL_COLUMNS", "build_frozen_event_signal_export"]
