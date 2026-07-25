"""Visualize the pull-door scene or replay a trained door policy."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from wheelrl.envs import B2WZ1DoorEnv
from wheelrl.envs.b2w_z1_door import TASK_STAGES


def _normalization_path(model_path: Path, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return explicit_path
    candidates = (
        model_path.parent / "vecnormalize.pkl",
        model_path.parent.parent / "vecnormalize.pkl",
        model_path.parent.parent.parent / "vecnormalize.pkl",
    )
    return next((path for path in candidates if path.exists()), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--stage", default="full", choices=TASK_STAGES)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    base_env = B2WZ1DoorEnv(
        render_mode="human",
        task_stage=args.stage,
        domain_randomization=0.0,
    )
    model = PPO.load(args.model, device=args.device) if args.model else None
    vec_env: VecNormalize | None = None
    if model is not None:
        normalization_path = _normalization_path(args.model, args.stats)
        if normalization_path is not None:
            vec_env = VecNormalize.load(
                normalization_path,
                DummyVecEnv([lambda: base_env]),
            )
            vec_env.training = False
            vec_env.norm_reward = False
            observation = vec_env.reset()
        else:
            print("warning: no vecnormalize.pkl found; using raw observations")
            observation, _ = base_env.reset(seed=0)
    else:
        observation, _ = base_env.reset(seed=0)

    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            start = time.monotonic()
            if model is None:
                action = np.zeros(base_env.action_space.shape, dtype=np.float32)
            else:
                action, _ = model.predict(observation, deterministic=True)

            if vec_env is not None:
                observation, _, done, infos = vec_env.step(action)
                if done[0]:
                    print(
                        f"episode_end success={infos[0].get('is_success', False)} "
                        f"door_angle={infos[0].get('door_angle', 0.0):.3f}"
                    )
            else:
                observation, _, terminated, truncated, info = base_env.step(action)
                if terminated or truncated:
                    print(
                        f"episode_end success={info['is_success']} "
                        f"door_angle={info['door_angle']:.3f}"
                    )
                    observation, _ = base_env.reset()
            elapsed = time.monotonic() - start
            if elapsed < base_env.dt:
                time.sleep(base_env.dt - elapsed)
    finally:
        if vec_env is not None:
            vec_env.close()
        else:
            base_env.close()


if __name__ == "__main__":
    main()
