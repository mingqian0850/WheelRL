"""Cross-platform experiment naming and provenance helpers."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco
import torch


def default_run_dir(task: str, seed: int) -> Path:
    """Build a collision-free run directory for parallel machines."""
    host = re.sub(r"[^A-Za-z0-9_.-]+", "-", platform.node()).strip("-") or "host"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / task / f"{host}-{timestamp}-seed{seed}"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def write_run_metadata(run_dir: Path, args: Namespace) -> None:
    """Write enough machine and command data to reproduce one run."""
    cuda_devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "memory_gib": round(properties.total_memory / 2**30, 3),
                }
            )
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "mujoco": mujoco.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "git_commit": _git_commit(),
        "command": sys.argv,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
