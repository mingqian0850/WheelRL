from __future__ import annotations

import numpy as np
from stable_baselines3.common.env_checker import check_env

from wheelrl.envs.b2w_z1_hier import B2WZ1HierEnv


def test_hier_env_spaces_and_checker() -> None:
    env = B2WZ1HierEnv(
        randomization=0.0,
        coordination_stage="local",
        max_episode_steps=5,
    )
    try:
        check_env(env, warn=True)
        observation, _ = env.reset(seed=11)
        assert observation.shape == (78,)
        assert env.action_space.shape == (3,)
    finally:
        env.close()


def test_local_gate_ignores_policy_and_preserves_exact_tcp_target() -> None:
    env = B2WZ1HierEnv(randomization=0.0, coordination_stage="local")
    try:
        env.reset(seed=12)
        rotation = env._base_rotation()
        position = env.wbc.home_tcp_position + rotation @ np.array([0.05, 0.05, 0.0])
        env.set_command_pose(position, env.wbc.home_tcp_rotation)
        expected_position = env.wbc.target_position.copy()
        expected_rotation = env.wbc.target_rotation.copy()

        _, _, _, _, info = env.step(np.ones(3, dtype=np.float32))

        assert info["coordination_gate"] == 0.0
        np.testing.assert_allclose(info["applied_action"], 0.0)
        np.testing.assert_allclose(env.wbc.target_position, expected_position)
        np.testing.assert_allclose(env.wbc.target_rotation, expected_rotation)
    finally:
        env.close()


def test_far_target_enables_base_command_without_shifting_tcp_target() -> None:
    env = B2WZ1HierEnv(randomization=0.0, coordination_stage="forward")
    try:
        env.reset(seed=13)
        rotation = env._base_rotation()
        start_base = env.data.xpos[env.wbc._base_body_id].copy()
        position = env.wbc.home_tcp_position + rotation @ np.array([0.60, 0.0, 0.0])
        env.set_command_pose(position, env.wbc.home_tcp_rotation)

        for _ in range(35):
            env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))

        displacement = rotation.T @ (
            env.data.xpos[env.wbc._base_body_id] - start_base
        )
        assert env._coordination_gate > 0.9
        assert displacement[0] > 0.03
        np.testing.assert_allclose(env.wbc.target_position, position)
    finally:
        env.close()


def test_left_and_right_far_targets_have_symmetric_positive_gate() -> None:
    env = B2WZ1HierEnv(randomization=0.0, coordination_stage="lateral")
    try:
        env.reset(seed=14)
        rotation = env._base_rotation()
        gates = []
        for lateral in (-0.40, 0.40):
            position = env.wbc.home_tcp_position + rotation @ np.array(
                [0.0, lateral, 0.0]
            )
            env.set_command_pose(position, env.wbc.home_tcp_rotation)
            gates.append(env._coordination_gate)
        assert min(gates) > 0.9
        assert abs(gates[0] - gates[1]) < 1.0e-9
    finally:
        env.close()


def test_domain_randomization_is_applied_without_compounding() -> None:
    env = B2WZ1HierEnv(randomization=1.0, coordination_stage="local")
    try:
        nominal = env._nominal_body_mass.copy()
        env.reset(seed=15)
        first = env.model.body_mass.copy()
        env.reset(seed=16)
        second = env.model.body_mass.copy()
        massive = nominal > 1.0e-9

        assert not np.allclose(first[massive], nominal[massive])
        assert not np.allclose(second[massive], first[massive])
        assert np.all(second[massive] >= 0.85 * nominal[massive])
        assert np.all(second[massive] <= 1.15 * nominal[massive])
    finally:
        env.close()


def test_disabling_coordinator_restores_pure_wbc() -> None:
    env = B2WZ1HierEnv(
        randomization=0.0,
        coordination_stage="forward",
        coordinator_enabled=False,
    )
    try:
        env.reset(seed=17)
        rotation = env._base_rotation()
        env.set_command_pose(
            env.wbc.home_tcp_position + rotation @ np.array([0.60, 0.0, 0.0]),
            env.wbc.home_tcp_rotation,
        )
        _, _, _, _, info = env.step(np.ones(3, dtype=np.float32))
        assert info["coordination_gate"] > 0.9
        np.testing.assert_allclose(info["applied_action"], 0.0)
        assert env.wbc._coordinator_command is None
    finally:
        env.close()
