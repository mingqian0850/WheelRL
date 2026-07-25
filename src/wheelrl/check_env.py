"""Validate the MuJoCo model and Gymnasium API."""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3.common.env_checker import check_env

from wheelrl.envs import B2WZ1Env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = B2WZ1Env(render_mode="human" if args.render else None)
    try:
        check_env(env, warn=True)
        observation, _ = env.reset(seed=7)
        rewards: list[float] = []
        resets = 0
        for _ in range(args.steps):
            action = env.action_space.sample() * 0.05
            observation, reward, terminated, truncated, _ = env.step(action)
            if not np.isfinite(observation).all() or not np.isfinite(reward):
                raise RuntimeError("Non-finite observation or reward detected")
            rewards.append(reward)
            if terminated or truncated:
                observation, _ = env.reset()
                resets += 1

        print(
            "environment_ok "
            f"nq={env.model.nq} nv={env.model.nv} nu={env.model.nu} "
            f"obs={env.observation_space.shape[0]} action={env.action_space.shape[0]} "
            f"mean_reward={np.mean(rewards):.3f} resets={resets}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
