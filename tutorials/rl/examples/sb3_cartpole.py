"""Train and evaluate a compact Stable-Baselines3 PPO example.

API usage is intentionally small and original. For complete implementations,
see the upstream Stable-Baselines3 and CleanRL projects listed in UPSTREAMS.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps <= 0 or args.n_envs <= 0 or args.eval_episodes <= 0:
        raise ValueError("timesteps, n_envs, and eval_episodes must be positive")

    train_env = make_vec_env("CartPole-v1", n_envs=args.n_envs, seed=args.seed)
    eval_env = Monitor(gym.make("CartPole-v1"))
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3.0e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        )
        if args.save is not None:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            model.save(args.save)
    finally:
        train_env.close()
        eval_env.close()

    print(
        f"evaluation episodes={args.eval_episodes} "
        f"mean_reward={mean_reward:.2f} std_reward={std_reward:.2f}"
    )
    if args.save is not None:
        print(f"model_saved={args.save.with_suffix('.zip')}")


if __name__ == "__main__":
    main()
