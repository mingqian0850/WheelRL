import numpy as np

from wheelrl.envs import B2WZ1DoorEnv, B2WZ1Env


def test_model_and_environment_step() -> None:
    env = B2WZ1Env(command_curriculum=0.0)
    try:
        observation, info = env.reset(seed=1)
        assert env.model.nq == 30
        assert env.model.nv == 29
        assert env.model.nu == 23
        assert observation.shape == (80,)
        assert info["command_vx"] == 0.0

        observation, reward, terminated, truncated, info = env.step(np.zeros(23, dtype=np.float32))
        assert observation.shape == (80,)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        assert not truncated
        assert "ee_error" in info
        assert isinstance(terminated, bool)
    finally:
        env.close()


def test_gripper_opens_and_closes() -> None:
    env = B2WZ1Env(command_curriculum=0.0)
    try:
        env.reset(seed=2)
        close_action = np.zeros(23, dtype=np.float32)
        close_action[-1] = 1.0
        for _ in range(30):
            _, _, terminated, truncated, _ = env.step(close_action)
            assert not (terminated or truncated)
        closed_position = env.data.qpos[env._qpos_adr[-1]]

        open_action = np.zeros(23, dtype=np.float32)
        open_action[-1] = -1.0
        for _ in range(30):
            _, _, terminated, truncated, _ = env.step(open_action)
            assert not (terminated or truncated)
        open_position = env.data.qpos[env._qpos_adr[-1]]

        assert closed_position > -0.2
        assert open_position < -1.3
    finally:
        env.close()


def test_door_environment_step_and_spaces() -> None:
    env = B2WZ1DoorEnv(task_stage="full", domain_randomization=0.0)
    try:
        observation, info = env.reset(seed=3)
        assert env.model.nq == 32
        assert env.model.nv == 31
        assert env.model.nu == 23
        assert env.action_space.shape == (9,)
        assert observation.shape == (84,)
        assert info["latch_released"] is False

        observation, reward, terminated, truncated, info = env.step(
            np.zeros(9, dtype=np.float32)
        )
        assert observation.shape == (84,)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        assert not truncated
        assert "door_angle" in info
        assert isinstance(terminated, bool)
    finally:
        env.close()


def test_door_latch_releases_only_after_handle_threshold() -> None:
    env = B2WZ1DoorEnv(task_stage="full", domain_randomization=0.0)
    try:
        env.reset(seed=4)
        env.data.qpos[env._handle_qpos_adr] = env._release_angle - 0.02
        env._update_latch_state()
        assert env._latch_released is False

        env.data.qpos[env._handle_qpos_adr] = env._release_angle + 0.02
        env._update_latch_state()
        assert env._latch_released is True
    finally:
        env.close()


def test_locked_door_has_stronger_restoring_torque() -> None:
    env = B2WZ1DoorEnv(task_stage="full", domain_randomization=0.0)
    try:
        env.reset(seed=5)
        env.data.qpos[env._door_qpos_adr] = -0.10
        env.data.qvel[env._door_dof_adr] = 0.0
        env._latch_released = False
        env._apply_door_passive_forces()
        locked_torque = env.data.qfrc_applied[env._door_dof_adr]

        env._latch_released = True
        env._apply_door_passive_forces()
        released_torque = env.data.qfrc_applied[env._door_dof_adr]
        assert locked_torque > 80.0
        assert 0.0 < released_torque < 1.0
    finally:
        env.close()
