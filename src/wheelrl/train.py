"""Train a PPO whole-body policy for the B2-W + Z1 gripper environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Codex Desktop can forward Windows TEMP/TMP paths into WSL. Stable-Baselines3's
# fork server needs a Linux filesystem for its Unix-domain socket.
if not os.environ.get("TMPDIR") or os.environ["TMPDIR"].startswith("/mnt/"):
    os.environ["TMPDIR"] = "/tmp"

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

from wheelrl.envs import B2WZ1Env


def make_env(seed: int, rank: int, command_scale: float):
    def _factory():
        env = B2WZ1Env(command_curriculum=command_scale)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--command-scale", type=float, default=0.5)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/b2w_z1_gripper_ppo"),
    )
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access the GPU")

    if not 0.0 <= args.command_scale <= 1.0:
        raise ValueError("--command-scale must be between 0 and 1")

    env_fns = [make_env(args.seed, rank, args.command_scale) for rank in range(args.n_envs)]
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    train_env = VecNormalize(
        vec_cls(env_fns),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.99,
    )
    eval_env = VecNormalize(
        DummyVecEnv([make_env(args.seed + 10_000, 0, args.command_scale)]),
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0,
    )

    checkpoint = CheckpointCallback(
        save_freq=max(25_000 // args.n_envs, 1),
        save_path=str(args.run_dir / "checkpoints"),
        name_prefix="ppo_b2w_z1_gripper",
        save_vecnormalize=True,
    )
    evaluation = EvalCallback(
        eval_env,
        best_model_save_path=str(args.run_dir / "best"),
        log_path=str(args.run_dir / "eval"),
        eval_freq=max(50_000 // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3.0e-4,
        n_steps=512,
        batch_size=512,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        policy_kwargs={
            "log_std_init": -1.0,
            "net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
        },
        tensorboard_log=str(args.run_dir / "tensorboard"),
        device=args.device,
        seed=args.seed,
        verbose=1,
    )

    print(
        f"training_start device={model.device} n_envs={args.n_envs} "
        f"timesteps={args.timesteps} run_dir={args.run_dir}"
    )
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint, evaluation],
            progress_bar=True,
        )
        model.save(args.run_dir / "final_model")
        train_env.save(args.run_dir / "vecnormalize.pkl")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
