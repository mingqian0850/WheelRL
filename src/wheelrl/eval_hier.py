"""Paired evaluation of learned base coordination against pure analytic WBC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from wheelrl.envs.b2w_z1_hier import B2WZ1HierEnv, CoordinationStage


def _run_episode(
    *,
    seed: int,
    policy: PPO | None,
    stats: VecNormalize | None,
    stage: CoordinationStage,
    randomization: float,
    max_episode_steps: int,
) -> dict[str, Any]:
    learned = policy is not None
    env = B2WZ1HierEnv(
        coordination_stage=stage,
        randomization=randomization,
        max_episode_steps=max_episode_steps,
        coordinator_enabled=learned,
    )
    observation, reset_info = env.reset(seed=seed)
    start_base = env.data.xpos[env.wbc._base_body_id].copy()
    position_errors: list[float] = []
    orientation_errors: list[float] = []
    gates: list[float] = []
    action_norms: list[float] = []
    final_info: dict[str, Any] = {}
    try:
        for _ in range(max_episode_steps):
            if learned:
                assert policy is not None and stats is not None
                normalized = stats.normalize_obs(observation[None])[0]
                action, _ = policy.predict(normalized, deterministic=True)
                action = np.asarray(action, dtype=np.float32)
            else:
                action = np.zeros(3, dtype=np.float32)
            observation, _, terminated, truncated, final_info = env.step(action)
            position_errors.append(float(final_info["command_pos_error"]))
            orientation_errors.append(float(final_info["command_ori_error"]))
            gates.append(float(final_info["coordination_gate"]))
            action_norms.append(float(np.linalg.norm(action)))
            if terminated or truncated:
                break
        tail = min(50, len(position_errors))
        end_base = env.data.xpos[env.wbc._base_body_id].copy()
        return {
            "seed": seed,
            "category": reset_info["target_category"],
            "success": bool(final_info.get("success", False)),
            "fallen": final_info.get("termination_reason") == "fallen_or_non_finite",
            "steps": len(position_errors),
            "tail_position_error_m": float(np.mean(position_errors[-tail:])),
            "tail_orientation_error_rad": float(np.mean(orientation_errors[-tail:])),
            "position_jitter_m": float(np.std(position_errors[-tail:])),
            "base_travel_m": float(np.linalg.norm(end_base[:2] - start_base[:2])),
            "mean_gate": float(np.mean(gates)),
            "mean_action_norm": float(np.mean(action_norms)),
        }
    finally:
        env.close()


def _summarize(episodes: list[dict[str, Any]]) -> dict[str, float]:
    def mean(key: str) -> float:
        return float(np.mean([float(episode[key]) for episode in episodes]))

    return {
        "success_rate": mean("success"),
        "fall_rate": mean("fallen"),
        "tail_position_error_m": mean("tail_position_error_m"),
        "tail_orientation_error_deg": float(
            np.rad2deg(mean("tail_orientation_error_rad"))
        ),
        "position_jitter_m": mean("position_jitter_m"),
        "base_travel_m": mean("base_travel_m"),
        "episode_steps": mean("steps"),
        "mean_gate": mean("mean_gate"),
        "mean_action_norm": mean("mean_action_norm"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument(
        "--stage", choices=["local", "forward", "lateral", "full"], default="full"
    )
    parser.add_argument("--randomization", type=float, default=1.0)
    parser.add_argument("--max-episode-steps", type=int, default=750)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    policy = PPO.load(args.model, device=args.device)
    if policy.action_space.shape != (3,):
        raise ValueError(
            "Expected a 3-action coordinator policy; legacy 6-action residual "
            "checkpoints are intentionally incompatible."
        )
    stats = VecNormalize.load(
        args.stats,
        DummyVecEnv(
            [
                lambda: B2WZ1HierEnv(
                    coordination_stage=args.stage,
                    randomization=0.0,
                    max_episode_steps=args.max_episode_steps,
                )
            ]
        ),
    )
    stats.training = False
    stats.norm_reward = False

    learned_episodes: list[dict[str, Any]] = []
    baseline_episodes: list[dict[str, Any]] = []
    try:
        for index in range(args.episodes):
            seed = args.seed + index
            baseline_episodes.append(
                _run_episode(
                    seed=seed,
                    policy=None,
                    stats=None,
                    stage=args.stage,
                    randomization=args.randomization,
                    max_episode_steps=args.max_episode_steps,
                )
            )
            learned_episodes.append(
                _run_episode(
                    seed=seed,
                    policy=policy,
                    stats=stats,
                    stage=args.stage,
                    randomization=args.randomization,
                    max_episode_steps=args.max_episode_steps,
                )
            )
    finally:
        stats.close()

    baseline = _summarize(baseline_episodes)
    learned = _summarize(learned_episodes)
    result = {
        "config": {
            "episodes": args.episodes,
            "seed": args.seed,
            "stage": args.stage,
            "randomization": args.randomization,
        },
        "pure_wbc": baseline,
        "learned_coordinator": learned,
        "paired_delta": {
            "success_rate": learned["success_rate"] - baseline["success_rate"],
            "tail_position_error_m": (
                learned["tail_position_error_m"] - baseline["tail_position_error_m"]
            ),
            "position_jitter_m": (
                learned["position_jitter_m"] - baseline["position_jitter_m"]
            ),
        },
        "episodes": {
            "pure_wbc": baseline_episodes,
            "learned_coordinator": learned_episodes,
        },
    }
    print(json.dumps({key: result[key] for key in result if key != "episodes"}, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
