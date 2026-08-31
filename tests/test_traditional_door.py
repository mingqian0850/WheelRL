import mujoco
import numpy as np

from wheelrl.envs import B2WZ1DoorEnv
from wheelrl.envs.b2w_z1_door import DOOR_SUCCESS_ANGLE, PREGRASP_ARM
from wheelrl.traditional_door import (
    MinkArmIK,
    TraditionalDoorController,
    TraditionalDoorPhase,
)


def test_mink_arm_ik_freezes_every_non_arm_degree_of_freedom() -> None:
    env = B2WZ1DoorEnv(task_stage="full", domain_randomization=0.0)
    try:
        env.reset(seed=0)
        target_data = mujoco.MjData(env.model)
        target_data.qpos[:] = env.data.qpos
        target_data.qpos[env._qpos_adr[16:22]] = PREGRASP_ARM
        mujoco.mj_forward(env.model, target_data)
        target_position = target_data.site_xpos[env._ee_site_id].copy()
        target_rotation = target_data.site_xmat[env._ee_site_id].reshape(3, 3).copy()

        result = MinkArmIK(env).solve_pose(
            env.data.qpos,
            target_position,
            target_rotation,
            max_iterations=120,
        )

        assert result.position_error < 1.0e-4
        assert result.orientation_error < 1.0e-3
        assert result.non_arm_displacement < 1.0e-10
        np.testing.assert_allclose(result.arm_qpos, PREGRASP_ARM, atol=3.0e-4)
    finally:
        env.close()


def test_traditional_controller_opens_door_with_base_after_unlatching() -> None:
    env = B2WZ1DoorEnv(
        task_stage="full",
        domain_randomization=0.0,
        max_episode_steps=100_000,
    )
    try:
        controller = TraditionalDoorController(env)
        controller.reset(seed=0)
        phases = [controller.phase]
        base_moved_before_release = False
        pull_start_position: np.ndarray | None = None

        for _ in range(3_000):
            diagnostic = controller.step()
            if diagnostic.phase != phases[-1]:
                phases.append(diagnostic.phase)
            if not diagnostic.latch_released:
                base_moved_before_release |= abs(diagnostic.base_forward_command) > 1.0e-9
            if diagnostic.phase == TraditionalDoorPhase.PULL and pull_start_position is None:
                pull_start_position = env.data.xpos[env._base_body_id].copy()
            if controller.terminal:
                break

        assert phases == [
            TraditionalDoorPhase.SETTLE,
            TraditionalDoorPhase.APPROACH,
            TraditionalDoorPhase.GRASP,
            TraditionalDoorPhase.UNLATCH,
            TraditionalDoorPhase.PULL,
            TraditionalDoorPhase.SUCCESS,
        ]
        assert not base_moved_before_release
        assert controller.success
        assert diagnostic.latch_released
        assert diagnostic.grasp_attached
        assert diagnostic.door_angle <= DOOR_SUCCESS_ANGLE
        assert diagnostic.upright > 0.98
        assert diagnostic.arm_joint_margin > 0.05
        assert pull_start_position is not None
        base_displacement = env.data.xpos[env._base_body_id] - pull_start_position
        assert base_displacement[0] < -0.70
        assert diagnostic.elapsed_time < 25.0
    finally:
        env.close()
