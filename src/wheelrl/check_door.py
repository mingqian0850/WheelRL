"""Validate the pull-door MuJoCo model, latch, and Gymnasium API."""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3.common.env_checker import check_env

from wheelrl.envs import B2WZ1DoorEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = B2WZ1DoorEnv(
        render_mode="human" if args.render else None,
        task_stage="full",
        domain_randomization=0.0,
    )
    try:
        check_env(env, warn=True)
        observation, _ = env.reset(seed=7)
        rewards: list[float] = []
        resets = 0
        for _ in range(args.steps):
            action = env.action_space.sample() * 0.15
            observation, reward, terminated, truncated, _ = env.step(action)
            if not np.isfinite(observation).all() or not np.isfinite(reward):
                raise RuntimeError("Non-finite observation or reward detected")
            rewards.append(reward)
            if terminated or truncated:
                observation, _ = env.reset()
                resets += 1

        env.reset(seed=8)
        env.data.qpos[env._door_qpos_adr] = -0.10
        env.data.qvel[env._door_dof_adr] = 0.0
        env._latch_released = False
        env._apply_door_passive_forces()
        locked_torque = float(env.data.qfrc_applied[env._door_dof_adr])
        env._latch_released = True
        env._apply_door_passive_forces()
        released_torque = float(env.data.qfrc_applied[env._door_dof_adr])
        if locked_torque <= 20.0 or released_torque >= 2.0:
            raise RuntimeError(
                f"Latch torque check failed: locked={locked_torque}, released={released_torque}"
            )

        print(
            "door_environment_ok "
            f"nq={env.model.nq} nv={env.model.nv} nu={env.model.nu} "
            f"obs={env.observation_space.shape[0]} action={env.action_space.shape[0]} "
            f"mean_reward={np.mean(rewards):.3f} resets={resets} "
            f"latch_torque={locked_torque:.1f}/{released_torque:.1f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
