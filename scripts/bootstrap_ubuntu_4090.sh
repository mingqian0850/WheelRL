#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

uv_bin="$(command -v uv || true)"
if [[ -z "$uv_bin" && -x "${HOME}/.local/bin/uv" ]]; then
  uv_bin="${HOME}/.local/bin/uv"
fi
if [[ -z "$uv_bin" ]]; then
  python3 -m pip install --user uv
  uv_bin="${HOME}/.local/bin/uv"
fi

export UV_LINK_MODE=hardlink
export TMPDIR=/tmp
"$uv_bin" python install 3.11
"$uv_bin" sync --extra dev

nvidia-smi \
  --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader

.venv/bin/python -m wheelrl.check_env --steps 50
.venv/bin/python -m wheelrl.check_door
.venv/bin/python - <<'PY'
import sys

import torch

print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA; check the NVIDIA driver and wheel.")

devices = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    memory_gib = properties.total_memory / 2**30
    devices.append((properties.name, memory_gib))
    print(f"device[{index}]={properties.name} vram={memory_gib:.2f} GiB")

if not any("4090" in name and memory_gib >= 22.0 for name, memory_gib in devices):
    raise SystemExit("Expected an RTX 4090 with at least 22 GiB visible VRAM.")

if sys.platform != "linux":
    raise SystemExit(f"Expected native Linux, got {sys.platform}.")
PY

echo "Ubuntu RTX 4090 environment is ready."
