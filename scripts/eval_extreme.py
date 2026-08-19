"""Extreme-posture stability evaluation for the tracking policy.

Resets the robot with the arm at extreme poses (joint-limit corners and
random full-range poses), commands a small nearby TCP target, and measures
how often the policy falls and how well it tracks. This targets exactly the
"unstable at certain postures" failure mode of kinematic controllers.

Usage:
    .venv/bin/python scripts/eval_extreme.py \
        --model runs/track_ppo_v2/final_model.zip \
        --stats runs/track_ppo_v2/vecnormalize.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from wheelrl.envs import B2WZ1TrackEnv

LIMIT_MARGIN = 0.06  # rad, keep this far from the hard limit
TRACK_STEPS = 400  # 4 s per pose
OFFSET = 0.05  # m, target offset from the current EE at the extreme pose


def make_env(seed: int):
    def _factory():
        env = B2WZ1TrackEnv(
            tracking_curriculum=0.2,
            target_motion=0.0,
            randomization=0.0,
        )
        env.reset(seed=seed)
        return env

    return _factory


def extreme_poses(env: B2WZ1TrackEnv, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    ranges = env._arm_ranges
    lo = ranges[:, 0] + LIMIT_MARGIN
    hi = ranges[:, 1] - LIMIT_MARGIN
    poses: list[tuple[str, np.ndarray]] = []
    # each joint at its low and high extreme, others nominal
    for joint in range(6):
        for value, label in ((lo[joint], "low"), (hi[joint], "high")):
            pose = np.zeros(6)
            pose[joint] = value
            poses.append((f"j{joint+1}-{label}", pose))
    # random full-range poses (extreme combinations)
    for index in range(8):
        poses.append(
            (f"random-{index}", rng.uniform(lo, hi))
        )
    return poses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    env = VecNormalize(
        DummyVecEnv([make_env(7)]),
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0,
    )
    env = VecNormalize.load(args.stats, env)
    env.training = False
    env.norm_reward = False
    policy = PPO.load(args.model, env=env, device=args.device)
    raw_env = env.venv.envs[0]

    rng = np.random.default_rng(0)
    results = []
    for name, pose in extreme_poses(raw_env, rng):
        env.reset()
        raw_env.data.qpos[raw_env._qpos_adr[16:22]] = pose
        mujoco_forward(raw_env)
        # nearby target so the test is "stabilize + track", not "reach far"
        raw_env._ee_target_base = raw_env._current_ee_base() + rng.uniform(
            -OFFSET, OFFSET, size=3
        )
        raw_env._ee_target_rotation = raw_env._current_ee_rotation_base()
        observation = raw_env._get_obs()

        fell = False
        errors = []
        for _ in range(TRACK_STEPS):
            normalized = env.normalize_obs(np.asarray(observation)[None])[0]
            action, _ = policy.predict(normalized, deterministic=True)
            observation, _, terminated, _, info = raw_env.step(action)
            errors.append(
                (float(info["ee_error"]), float(info["ee_orientation_error"]))
            )
            if bool(terminated):
                fell = True
                break
        mean_pos = float(np.mean([e[0] for e in errors]))
        mean_ori = float(np.mean([e[1] for e in errors]))
        results.append((name, fell, mean_pos, mean_ori))
        print(
            f"{name:>12}: fell={fell}  mean_pos={mean_pos*1000:5.0f}mm  "
            f"mean_ori={np.rad2deg(mean_ori):5.1f}deg"
        )

    falls = sum(1 for _, fell, _, _ in results if fell)
    print(f"\n极端姿态摔倒率: {falls}/{len(results)}")
    print(f"全部姿态平均位置误差: {np.mean([r[2] for r in results])*1000:.0f}mm")
    env.close()


def mujoco_forward(env: B2WZ1TrackEnv) -> None:
    import mujoco

    mujoco.mj_forward(env.model, env.data)


if __name__ == "__main__":
    main()
