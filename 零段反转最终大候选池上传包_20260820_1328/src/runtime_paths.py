"""Default paths used by the uploaded notebooks.

The package can be copied to any working directory.  Production notebooks
still read only the single raw spot file; environment variables are optional
overrides for a different mount.  Results default to the folder containing
the uploaded notebooks so a remote user can inspect them in place.
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_SPOT_ROOT = Path("/home/hzy/cta")
DEFAULT_SPOT_GLOB = "IC数据更新*最终固化版/现货最终版/CSI500_SPOT_md_eod_raw*最终版.parquet"
DEFAULT_REMOTE_THREE_STATE_PATH = Path(
    "/home/hzy/cta/三状态冻结/IC_1545_three_state_and_downside_warning.csv"
)


def _absolute(raw: str | Path, label: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} 必须使用绝对路径：{raw}")
    return path.resolve()


def resolve_spot_path() -> Path:
    """Resolve the configured or reference remote raw spot file."""

    configured = os.environ.get("COMPANY_SPOT_PATH", "").strip()
    if configured:
        path = _absolute(configured, "COMPANY_SPOT_PATH")
        if not path.is_file():
            raise FileNotFoundError(f"COMPANY_SPOT_PATH does not exist: {path}")
        return path
    matches = sorted(DEFAULT_SPOT_ROOT.glob(DEFAULT_SPOT_GLOB))
    if len(matches) != 1:
        raise FileNotFoundError(
            "无法在参考远端现货路径模式下唯一定位文件；"
            f"找到 {len(matches)} 个。请设置 COMPANY_SPOT_PATH 覆盖：{DEFAULT_SPOT_ROOT / DEFAULT_SPOT_GLOB}"
        )
    return matches[0].resolve()


def resolve_output_dir(package_root: str | Path) -> Path:
    """Use an explicit output override or the uploaded notebook directory."""

    configured = os.environ.get("COMPANY_OUTPUT_DIR", "").strip()
    path = _absolute(configured, "COMPANY_OUTPUT_DIR") if configured else _absolute(package_root, "上传包根目录")
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_remote_three_state_path() -> Path:
    """Resolve the reference remote baseline path for optional Notebook 05."""

    configured = os.environ.get("REMOTE_THREE_STATE_PATH", "").strip()
    return _absolute(configured, "REMOTE_THREE_STATE_PATH") if configured else DEFAULT_REMOTE_THREE_STATE_PATH


__all__ = [
    "DEFAULT_REMOTE_THREE_STATE_PATH",
    "DEFAULT_SPOT_GLOB",
    "DEFAULT_SPOT_ROOT",
    "resolve_output_dir",
    "resolve_remote_three_state_path",
    "resolve_spot_path",
]
