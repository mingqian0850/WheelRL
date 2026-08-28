#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ ! -x .venv/bin/python ]]; then
  echo "Run scripts/bootstrap_ubuntu_4090.sh first." >&2
  exit 2
fi

task="${1:-}"
n_envs="${N_ENVS:-8}"
seed="${SEED:-142}"
timesteps="${TIMESTEPS:-3000000}"
policy_device="${POLICY_DEVICE:-cpu}"

case "$task" in
  gripper)
    exec .venv/bin/wheelrl-train \
      --timesteps "$timesteps" \
      --n-envs "$n_envs" \
      --seed "$seed" \
      --device "$policy_device" \
      --command-scale "${COMMAND_SCALE:-0.5}"
    ;;
  door)
    exec .venv/bin/wheelrl-train-door \
      --stage curriculum \
      --timesteps "$timesteps" \
      --n-envs "$n_envs" \
      --seed "$seed" \
      --device "$policy_device" \
      --randomization "${RANDOMIZATION:-0.55}"
    ;;
  coordinator)
    exec .venv/bin/wheelrl-train-hier \
      --stage "${COORDINATOR_STAGE:-curriculum}" \
      --timesteps "$timesteps" \
      --n-envs "$n_envs" \
      --seed "$seed" \
      --device "$policy_device" \
      --randomization "${RANDOMIZATION:-0.75}"
    ;;
  *)
    echo "usage: $0 {gripper|door|coordinator}" >&2
    echo "optional: N_ENVS, SEED, TIMESTEPS, POLICY_DEVICE," >&2
    echo "          COMMAND_SCALE, RANDOMIZATION, COORDINATOR_STAGE" >&2
    exit 2
    ;;
esac
