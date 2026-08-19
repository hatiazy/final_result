from __future__ import annotations

"""Run only the already-frozen V55/V80 nonzero-exit parameters."""

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "packages" / "nonzero" / "1545" / "src"
OUTPUT_DIR = Path(os.environ.get("UPLOAD_OUTPUT_DIR", str(PACKAGE_ROOT / "runtime_outputs"))).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SOURCE_ROOT))

from signal_export import SIGNAL_COLUMNS, build_frozen_signal_export  # noqa: E402


def _json_default(value: Any) -> Any:
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
    print(f"[非零退出] {message}", flush=True)


signal, metadata, _ = build_frozen_signal_export(progress=progress)
if list(signal.columns) != list(SIGNAL_COLUMNS):
    raise AssertionError(f"非零退出列顺序不一致: {list(signal.columns)}")

output_path = OUTPUT_DIR / "remote_nonzero_five_columns.csv"
metadata_path = OUTPUT_DIR / "remote_nonzero_metadata.json"
signal.to_csv(output_path, index=False, encoding="utf-8-sig")
metadata_path.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default) + "\n",
    encoding="utf-8",
)
print(f"[非零退出] output={output_path}", flush=True)
print(f"[非零退出] rows={len(signal)} date={signal['date'].min()} -> {signal['date'].max()}", flush=True)
