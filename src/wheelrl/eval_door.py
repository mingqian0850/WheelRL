"""Evaluate door-opening success rate for a trained WheelRL PPO policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from wheelrl.envs import B2WZ1DoorEnv
from wheelrl.envs.b2w_z1_door import TASK_STAGES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--stage", default="full", choices=TASK_STAGES)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--randomization", type=float, default=1.0)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    base_env = B2WZ1DoorEnv(
        task_stage=args.stage,
        domain_randomization=args.randomization,
    )
    base_env.reset(seed=args.seed)
    env = VecNormalize.load(args.stats, DummyVecEnv([lambda: base_env]))
    env.training = False
    env.norm_reward = False
    model = PPO.load(args.model, env=env, device=args.device)

    successes = 0
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    observation = env.reset()
    running_reward = 0.0
    running_length = 0
    try:
        while len(episode_rewards) < args.episodes:
            action, _ = model.predict(observation, deterministic=True)
            observation, rewards, dones, infos = env.step(action)
            running_reward += float(rewards[0])
            running_length += 1
            if dones[0]:
                successes += int(bool(infos[0].get("is_success", False)))
                episode_rewards.append(running_reward)
                episode_lengths.append(running_length)
                running_reward = 0.0
                running_length = 0
    finally:
        env.close()

    print(
        f"evaluation episodes={args.episodes} stage={args.stage} "
        f"success_rate={successes / args.episodes:.3f} "
        f"reward_mean={np.mean(episode_rewards):.3f} "
        f"length_mean={np.mean(episode_lengths):.1f}"
    )


if __name__ == "__main__":
    main()
