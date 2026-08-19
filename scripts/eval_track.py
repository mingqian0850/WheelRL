"""Deterministic evaluation of a trained task-space tracking policy.

Measures TCP position/orientation tracking error, uprightness and base
height over randomized episodes. Run from the default wheelrl environment:

    source .venv/bin/activate
    wheelrl-eval-track --model runs/track_ppo/final_model.zip \
        --stats runs/track_ppo/vecnormalize.pkl --episodes 20
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

if not os.environ.get("TMPDIR") or os.environ["TMPDIR"].startswith("/mnt/"):
    os.environ["TMPDIR"] = "/tmp"

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from wheelrl.envs import B2WZ1TrackEnv


def make_env(seed: int, curriculum: float, motion: float, randomization: float):
    def _factory():
        env = B2WZ1TrackEnv(
            tracking_curriculum=curriculum,
            target_motion=motion,
            randomization=randomization,
        )
        env.reset(seed=seed)
        return env

    return _factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--curriculum", type=float, default=1.0)
    parser.add_argument("--motion", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--randomization", type=float, default=1.0)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    env = VecNormalize(
        DummyVecEnv([make_env(10_000, args.curriculum, args.motion, args.randomization)]),
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0,
    )
    env = VecNormalize.load(args.stats, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(args.model, env=env, device=args.device)

    position_errors: list[float] = []
    orientation_errors: list[float] = []
    mean_upright: list[float] = []
    falls = 0
    for _ in range(args.episodes):
        # The underlying env is seeded by the factory; each reset advances the
        # RNG so episodes differ without needing per-episode seeding.
        observation = env.reset()
        step = 0
        episode_pos: list[float] = []
        episode_ori: list[float] = []
        upright_values: list[float] = []
        while step < args.max_steps:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, dones, info = env.step(action)
            episode_pos.append(float(info[0]["ee_error"]))
            episode_ori.append(float(info[0]["ee_orientation_error"]))
            upright_values.append(float(info[0]["reward_terms"]["upright"]))
            step += 1
            if bool(dones[0]):
                # VecEnvs auto-reset done envs; the terminal info still tells
                # us whether the episode ended in a fall or a timeout.
                if info[0].get("termination_reason") == "fallen_or_non_finite":
                    falls += 1
                break
        position_errors.append(float(np.mean(episode_pos)))
        orientation_errors.append(float(np.mean(episode_ori)))
        mean_upright.append(float(np.mean(upright_values)))

    print(
        f"episodes={args.episodes} curriculum={args.curriculum} "
        f"motion={args.motion} randomization={args.randomization}"
    )
    print(f"mean pos error  : {np.mean(position_errors):.4f} m")
    print(f"mean ori error  : {np.mean(orientation_errors):.4f} rad "
          f"({np.rad2deg(np.mean(orientation_errors)):.1f} deg)")
    print(f"mean upright    : {np.mean(mean_upright):.3f}")
    print(f"falls           : {falls}/{args.episodes}")
    env.close()


if __name__ == "__main__":
    main()
