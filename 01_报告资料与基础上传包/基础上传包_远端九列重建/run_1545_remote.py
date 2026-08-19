"""Run the copied 1545 final company exporter in an isolated process."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("REMOTE_OUTPUT_DIR", str(ROOT / "远端输出"))).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_ROOT = ROOT / "vendor" / "1545" / "src"
import sys

sys.path.insert(0, str(SOURCE_ROOT))
from signal_export import SIGNAL_COLUMNS, build_frozen_signal_export  # noqa: E402


def _default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    return str(value)


def progress(message: str) -> None:
    print(f"[1545-SIGNAL] {message}", flush=True)


signal, metadata, _ = build_frozen_signal_export(progress=progress)
if list(signal.columns) != list(SIGNAL_COLUMNS):
    raise AssertionError(f"1545 exporter columns differ: {list(signal.columns)}")
output_path = OUTPUT_DIR / "remote_1545_five_columns.csv"
metadata_path = OUTPUT_DIR / "remote_1545_metadata.json"
signal.to_csv(output_path, index=False, encoding="utf-8-sig")
metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=_default) + "\n", encoding="utf-8")
print(f"[1545-SIGNAL] output={output_path}", flush=True)
print(f"[1545-SIGNAL] rows={len(signal)} date={signal['date'].min()} -> {signal['date'].max()}", flush=True)
