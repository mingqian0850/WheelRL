"""Train the three-action B2-W + Z1 learned base coordinator."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

if not os.environ.get("TMPDIR") or os.environ["TMPDIR"].startswith("/mnt/"):
    os.environ["TMPDIR"] = "/tmp"

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from wheelrl.envs.b2w_z1_hier import B2WZ1HierEnv, CoordinationStage
from wheelrl.runtime import default_run_dir, write_run_metadata

CURRICULUM = (
    ("local", 0.10),
    ("forward", 0.30),
    ("lateral", 0.30),
    ("full", 0.30),
)


class SaveNormalizationOnBest(BaseCallback):
    """Save the observation statistics paired with each best checkpoint."""

    def __init__(self, path: Path) -> None:
        super().__init__(verbose=0)
        self.path = path

    def _on_step(self) -> bool:
        vecnormalize = self.model.get_vec_normalize_env()
        if vecnormalize is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            vecnormalize.save(self.path)
        return True


def make_env(
    seed: int,
    rank: int,
    curriculum: float,
    randomization: float,
    action_penalty: float,
    stage: CoordinationStage,
    max_episode_steps: int,
):
    def _factory():
        env = B2WZ1HierEnv(
            tracking_curriculum=curriculum,
            randomization=randomization,
            action_penalty=action_penalty,
            coordination_stage=stage,
            max_episode_steps=max_episode_steps,
        )
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=4_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--curriculum", type=float, default=1.0)
    parser.add_argument("--randomization", type=float, default=1.0)
    parser.add_argument("--eval-randomization", type=float, default=1.0)
    parser.add_argument(
        "--stage",
        choices=["local", "forward", "lateral", "full", "curriculum"],
        default="curriculum",
        help="target family, or the recommended four-stage curriculum",
    )
    parser.add_argument("--max-episode-steps", type=int, default=750)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument(
        "--action-penalty",
        type=float,
        default=0.03,
        help="penalty on gated base coordination effort (default 0.03)",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume-model", type=Path)
    parser.add_argument("--resume-stats", type=Path)
    args = parser.parse_args()

    if (args.resume_model is None) != (args.resume_stats is None):
        parser.error("--resume-model and --resume-stats must be supplied together")
    if args.timesteps < 1 or args.n_envs < 1:
        parser.error("--timesteps and --n-envs must be positive")

    if args.run_dir is None:
        args.run_dir = default_run_dir("b2w_z1_coordinator_ppo", args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    initial_stage: CoordinationStage = (
        "local" if args.stage == "curriculum" else args.stage
    )
    env_fns = [
        make_env(
            args.seed,
            rank,
            args.curriculum,
            args.randomization,
            args.action_penalty,
            initial_stage,
            args.max_episode_steps,
        )
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
            raw_train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            gamma=0.99,
        )

    eval_env = VecNormalize(
        DummyVecEnv(
            [
                make_env(
                    args.seed + 10_000,
                    0,
                    args.curriculum,
                    args.eval_randomization,
                    args.action_penalty,
                    initial_stage,
                    args.max_episode_steps,
                )
            ]
        ),
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0,
    )
    eval_env.obs_rms = train_env.obs_rms

    checkpoint = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(args.run_dir / "checkpoints"),
        name_prefix="ppo_b2w_z1_coordinator",
        save_vecnormalize=True,
    )
    best_stats = args.run_dir / "best" / "vecnormalize.pkl"
    evaluation = EvalCallback(
        eval_env,
        best_model_save_path=str(args.run_dir / "best"),
        log_path=str(args.run_dir / "eval"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        callback_on_new_best=SaveNormalizationOnBest(best_stats),
    )

    if args.resume_model is not None:
        model = PPO.load(args.resume_model, device=args.device)
        if model.action_space.shape != (3,):
            raise ValueError(
                "Legacy 6-action residual checkpoints cannot be resumed with "
                "the new 3-action coordinator environment."
            )
        model.set_env(train_env)
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
        f"coordinator_training_start device={model.device} n_envs={args.n_envs} "
        f"timesteps={args.timesteps} stage={args.stage} "
        f"randomization={args.randomization} run_dir={args.run_dir}"
    )
    write_run_metadata(args.run_dir, args)
    if args.stage == "curriculum":
        stages = CURRICULUM
    else:
        stages = ((args.stage, 1.0),)

    allocated = 0
    try:
        for index, (stage, fraction) in enumerate(stages):
            stage_steps = (
                args.timesteps - allocated
                if index == len(stages) - 1
                else max(int(args.timesteps * fraction), args.n_envs)
            )
            allocated += stage_steps
            train_env.env_method("set_coordination_stage", stage)
            eval_env.env_method("set_coordination_stage", stage)
            print(f"coordination_stage={stage} timesteps={stage_steps}")
            model.learn(
                total_timesteps=stage_steps,
                callback=[checkpoint, evaluation],
                progress_bar=True,
                reset_num_timesteps=(args.resume_model is None and index == 0),
            )
        model.save(args.run_dir / "final_model")
        train_env.save(args.run_dir / "vecnormalize.pkl")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
