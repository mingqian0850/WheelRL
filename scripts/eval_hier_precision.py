"""Steady-state tracking precision of a hierarchical policy in closed loop.

Runs the training-style loop (random frozen command per episode, raw actions,
no smoothing) and reports the *settled* command error, which isolates the
policy's steady-state residual bias from any inference plumbing.

Usage:
    python scripts/eval_hier_precision.py [--model runs/hier_ppo/final_model.zip] \\
        [--stats runs/hier_ppo/vecnormalize.pkl] [--episodes 5]

Expectations:
    * policies trained without an action-magnitude penalty: ~25-30 mm offset
      (the residual policy's self-consistent nonzero fixed point);
    * with ``--action-penalty`` in training: a few mm.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from wheelrl.envs.b2w_z1_hier import B2WZ1HierEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("runs/hier_ppo/final_model.zip"))
    parser.add_argument("--stats", type=Path, default=Path("runs/hier_ppo/vecnormalize.pkl"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--randomization", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=1000, help="steps per episode")
    args = parser.parse_args()

    stats = VecNormalize.load(
        args.stats, DummyVecEnv([lambda: B2WZ1HierEnv(randomization=args.randomization)])
    )
    stats.training = False
    stats.norm_reward = False
    policy = PPO.load(args.model, device="cpu")

    all_pos, all_ori, all_act = [], [], []
    for ep in range(args.episodes):
        env = B2WZ1HierEnv(randomization=args.randomization)
        obs, _ = env.reset(seed=ep)
        pos, ori, act = [], [], []
        for t in range(args.steps):
            norm = stats.normalize_obs(np.asarray(obs)[None])[0]
            action, _ = policy.predict(norm, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if t >= args.steps - 200:  # settled window
                pos.append(info["command_pos_error"])
                ori.append(info["command_ori_error"])
                act.append(np.linalg.norm(action))
            if terminated or truncated:
                break
        env.close()
        all_pos += pos
        all_ori += ori
        all_act += act
        print(
            f"episode {ep}: pos {np.mean(pos) * 1000:6.1f} mm | "
            f"ori {np.rad2deg(np.mean(ori)):5.2f} deg | |action| {np.mean(act):5.3f}"
        )

    print(
        f"\noverall: pos {np.mean(all_pos) * 1000:6.1f} mm | "
        f"ori {np.rad2deg(np.mean(all_ori)):5.2f} deg | |action| {np.mean(all_act):5.3f}"
    )


if __name__ == "__main__":
    main()
