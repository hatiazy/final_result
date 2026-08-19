from __future__ import annotations

"""Refresh only the compact execution-date summary from existing engine outputs."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from remote_validation import (
    _build_execution_summary,
    _load_extreme,
    _load_nonzero,
    _resolve_spot,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PACKAGE_ROOT / "runtime_outputs"


def update_execution_summary(
    spot_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Read existing outputs and rewrite 最终执行日简表.csv without rerunning engines."""
    spot, source_kind = _resolve_spot(spot_path)
    output = Path(output_dir or os.environ.get("UPLOAD_OUTPUT_DIR", str(OUTPUT_ROOT))).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    nonzero_path = output / "remote_nonzero_five_columns.csv"
    down_path = output / "remote_down_extreme_daily.csv"
    up_path = output / "remote_up_extreme_daily.csv"
    missing = [str(path) for path in (nonzero_path, down_path, up_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "缺少已有引擎输出，先运行 00_主入口_远端验证.ipynb 或 src/remote_validation.py: "
            + ", ".join(missing)
        )

    summary_path = output / "最终执行日简表.csv"
    summary = _build_execution_summary(
        spot,
        _load_nonzero(nonzero_path),
        _load_extreme(down_path),
        _load_extreme(up_path),
        summary_path,
    )
    manifest = {
        "operation": "只更新最终执行日简表，不重跑任何信号引擎",
        "spot_path": str(spot),
        "spot_source_kind": source_kind,
        "input_files": {
            "nonzero": str(nonzero_path),
            "down": str(down_path),
            "up": str(up_path),
        },
        "output": summary,
    }
    manifest_path = output / "最终执行日简表_更新记录.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"已更新: {summary_path}")
    print(f"实际执行日: {summary['date_min']} -> {summary['date_max']}")
    print(f"输入现货: {spot} ({source_kind})")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="只根据已有引擎输出更新最终执行日简表")
    parser.add_argument("--spot", default=None, help="可选：现货文件；不填则使用环境变量或默认远端路径")
    parser.add_argument("--output", default=None, help="可选：runtime_outputs 目录")
    args = parser.parse_args()
    update_execution_summary(args.spot, args.output)


if __name__ == "__main__":
    main()
