#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

uv_bin="$(command -v uv || true)"
if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
  uv_bin="$HOME/.local/bin/uv"
fi
if [[ -z "$uv_bin" ]]; then
  python3 -m pip install --user --break-system-packages uv
  uv_bin="$HOME/.local/bin/uv"
fi

export UV_LINK_MODE=hardlink
export TMPDIR=/tmp
"$uv_bin" python install 3.11
"$uv_bin" sync --extra dev

.venv/bin/python -m wheelrl.check_env --steps 50
.venv/bin/python -c \
  'import torch; print("cuda_available=", torch.cuda.is_available()); print("device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")'
