"""Train the hierarchical TCP tracking policy (RL target gen + WBC servo)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

if not os.environ.get("TMPDIR") or os.environ["TMPDIR"].startswith("/mnt/"):
    os.environ["TMPDIR"] = "/tmp"

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from wheelrl.envs.b2w_z1_hier import B2WZ1HierEnv
from wheelrl.runtime import default_run_dir, write_run_metadata


def make_env(seed: int, rank: int, curriculum: float, randomization: float):
    def _factory():
        env = B2WZ1HierEnv(
            tracking_curriculum=curriculum,
            randomization=randomization,
        )
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--curriculum", type=float, default=1.0)
    parser.add_argument("--randomization", type=float, default=1.0)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume-model", type=Path)
    parser.add_argument("--resume-stats", type=Path)
    args = parser.parse_args()

    if args.run_dir is None:
        args.run_dir = default_run_dir("b2w_z1_hier_ppo", args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env(args.seed, rank, args.curriculum, args.randomization)
        for rank in range(args.n_envs)
    ]
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    raw_train_env = vec_cls(env_fns)
    if args.resume_stats is not None:
        train_env = VecNormalize.load(args.resume_stats, raw_train_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(
            raw_train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99
        )
    eval_env = VecNormalize(
        DummyVecEnv(
            [make_env(args.seed + 10_000, 0, args.curriculum, args.randomization)]
        ),
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0,
    )
    eval_env.obs_rms = train_env.obs_rms

    checkpoint = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=str(args.run_dir / "checkpoints"),
        name_prefix="ppo_b2w_z1_hier",
        save_vecnormalize=True,
    )
    evaluation = EvalCallback(
        eval_env,
        best_model_save_path=str(args.run_dir / "best"),
        log_path=str(args.run_dir / "eval"),
        eval_freq=max(100_000 // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    if args.resume_model is not None:
        model = PPO.load(args.resume_model, env=train_env, device=args.device)
        model.tensorboard_log = str(args.run_dir / "tensorboard")
    else:
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
        f"hier_training_start device={model.device} n_envs={args.n_envs} "
        f"timesteps={args.timesteps} curriculum={args.curriculum} "
        f"randomization={args.randomization} run_dir={args.run_dir}"
    )
    write_run_metadata(args.run_dir, args)
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint, evaluation],
            progress_bar=True,
            reset_num_timesteps=args.resume_model is None,
        )
        model.save(args.run_dir / "final_model")
        train_env.save(args.run_dir / "vecnormalize.pkl")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
