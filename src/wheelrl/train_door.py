"""Train a staged PPO policy to unlatch and pull open the B2-W + Z1 door."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

if not os.environ.get("TMPDIR") or os.environ["TMPDIR"].startswith("/mnt/"):
    os.environ["TMPDIR"] = "/tmp"

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from wheelrl.envs import B2WZ1DoorEnv
from wheelrl.envs.b2w_z1_door import TASK_STAGES, TaskStage

CURRICULUM_WEIGHTS = {
    "reach": 0.15,
    "turn": 0.30,
    "pull": 0.25,
    "full": 0.30,
}


class BestSuccessCallback(BaseCallback):
    """Save the policy selected by success rate rather than episode return."""

    def __init__(self, evaluation: EvalCallback, save_dir: Path) -> None:
        super().__init__(verbose=0)
        self.evaluation = evaluation
        self.save_dir = save_dir
        self.best_success_rate = -1.0
        self.best_mean_reward = float("-inf")

    def _on_step(self) -> bool:
        if self.evaluation.eval_freq <= 0 or self.n_calls % self.evaluation.eval_freq:
            return True
        successes = self.evaluation._is_success_buffer
        success_rate = float(sum(successes) / len(successes)) if successes else -1.0
        mean_reward = float(self.evaluation.last_mean_reward)
        better_success = success_rate > self.best_success_rate
        better_tie_break = (
            success_rate == self.best_success_rate and mean_reward > self.best_mean_reward
        )
        if better_success or better_tie_break:
            self.best_success_rate = success_rate
            self.best_mean_reward = mean_reward
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(self.save_dir / "best_success_model")
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(self.save_dir / "vecnormalize.pkl")
            print(
                f"new_best_success_rate={success_rate:.3f} "
                f"mean_reward={mean_reward:.3f} path={self.save_dir}"
            )
        return True


def make_env(seed: int, rank: int, task_stage: TaskStage, randomization: float):
    def _factory():
        env = B2WZ1DoorEnv(
            task_stage=task_stage,
            domain_randomization=randomization,
        )
        env.reset(seed=seed + rank)
        return Monitor(env, info_keywords=("is_success", "phase"))

    return _factory


def _stage_plan(stage: str, total_timesteps: int) -> list[tuple[TaskStage, int]]:
    if stage != "curriculum":
        return [(stage, total_timesteps)]  # type: ignore[list-item]
    allocations: list[tuple[TaskStage, int]] = []
    assigned = 0
    for index, name in enumerate(TASK_STAGES):
        if index == len(TASK_STAGES) - 1:
            steps = total_timesteps - assigned
        else:
            steps = round(total_timesteps * CURRICULUM_WEIGHTS[name])
            assigned += steps
        allocations.append((name, steps))
    return allocations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--n-envs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--stage",
        default="curriculum",
        choices=[*TASK_STAGES, "curriculum"],
        help="Train one task stage or the full staged curriculum.",
    )
    parser.add_argument(
        "--randomization",
        type=float,
        default=0.35,
        help="Domain-randomization strength from 0 to 1.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/b2w_z1_pull_door_ppo"),
    )
    parser.add_argument("--resume-model", type=Path)
    parser.add_argument("--resume-stats", type=Path)
    args = parser.parse_args()

    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    if not 0.0 <= args.randomization <= 1.0:
        raise ValueError("--randomization must be between 0 and 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access the GPU")
    if (args.resume_model is None) != (args.resume_stats is None):
        raise ValueError("--resume-model and --resume-stats must be supplied together")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    plan = _stage_plan(args.stage, args.timesteps)
    initial_stage = plan[0][0]
    env_fns = [
        make_env(args.seed, rank, initial_stage, args.randomization)
        for rank in range(args.n_envs)
    ]
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    raw_train_env = vec_cls(env_fns)
    raw_eval_env = DummyVecEnv(
        [make_env(args.seed + 10_000, 0, initial_stage, args.randomization)]
    )

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
        raw_eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0,
    )

    if args.resume_model is not None:
        model = PPO.load(args.resume_model, env=train_env, device=args.device)
        model.tensorboard_log = str(args.run_dir / "tensorboard")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=2.0e-4,
            n_steps=256,
            batch_size=256,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.025,
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
        f"door_training_start device={model.device} n_envs={args.n_envs} "
        f"plan={plan} randomization={args.randomization} run_dir={args.run_dir}"
    )
    try:
        for task_stage, stage_steps in plan:
            train_env.env_method("set_task_stage", task_stage)
            eval_env.env_method("set_task_stage", task_stage)
            train_env.reset()
            eval_env.reset()
            if model.ep_info_buffer is not None:
                model.ep_info_buffer.clear()
            if model.ep_success_buffer is not None:
                model.ep_success_buffer.clear()
            stage_dir = args.run_dir / task_stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = CheckpointCallback(
                save_freq=max(50_000 // args.n_envs, 1),
                save_path=str(stage_dir / "checkpoints"),
                name_prefix=f"ppo_door_{task_stage}",
                save_vecnormalize=True,
            )
            evaluation = EvalCallback(
                eval_env,
                best_model_save_path=str(stage_dir / "best"),
                log_path=str(stage_dir / "eval"),
                eval_freq=max(25_000 // args.n_envs, 1),
                n_eval_episodes=10,
                deterministic=True,
            )
            best_success = BestSuccessCallback(evaluation, stage_dir / "best_success")
            print(f"curriculum_stage={task_stage} timesteps={stage_steps}")
            model.learn(
                total_timesteps=stage_steps,
                callback=[checkpoint, evaluation, best_success],
                progress_bar=True,
                reset_num_timesteps=False,
                tb_log_name=f"door_{task_stage}",
            )
            model.save(stage_dir / "stage_model")
            train_env.save(stage_dir / "vecnormalize.pkl")

        model.save(args.run_dir / "final_model")
        train_env.save(args.run_dir / "vecnormalize.pkl")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
