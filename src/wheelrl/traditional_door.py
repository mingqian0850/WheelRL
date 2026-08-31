"""Traditional staged controller for opening a pull door with B2-W + Z1.

The arm performs free-space pose tracking and handle rotation with Mink IK.
After the latch releases, the arm becomes a compliant connection and the
wheeled base supplies the opening motion while following the door arc.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import mink
import mujoco
import numpy as np
from numpy.typing import NDArray

from wheelrl.envs.b2w_z1 import (
    ARM_JOINTS,
    ARM_KD,
    ARM_KP,
    GRIPPER_KD,
    GRIPPER_KP,
    LEG_KD,
    LEG_KP,
    LEG_NOMINAL,
    TORQUE_LIMITS,
)
from wheelrl.envs.b2w_z1_door import (
    DOOR_SUCCESS_ANGLE,
    PREGRASP_ARM,
    B2WZ1DoorEnv,
)

FloatArray = NDArray[np.float64]


class TraditionalDoorPhase(StrEnum):
    """Phases of the deterministic pull-door behavior."""

    SETTLE = "settle"
    APPROACH = "approach"
    GRASP = "grasp"
    UNLATCH = "unlatch"
    PULL = "pull"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class TraditionalDoorConfig:
    """Timing and gain parameters for :class:`TraditionalDoorController`."""

    control_hz: float = 100.0
    settle_seconds: float = 1.0
    approach_seconds: float = 4.0
    approach_timeout: float = 6.0
    approach_distance: float = 0.085
    grasp_settle_seconds: float = 1.2
    grasp_timeout: float = 2.5
    unlatch_seconds: float = 2.8
    unlatch_timeout: float = 4.5
    pull_timeout: float = 16.0
    pull_speed: float = 0.10
    pull_ramp_seconds: float = 0.50
    yaw_gain: float = 2.0
    max_yaw_rate: float = 0.45
    wheel_radius: float = 0.10
    half_track: float = 0.235
    free_space_arm_gain: float = 1.50
    compliant_arm_gain: float = 0.20
    arm_damping_scale: float = 2.0
    handle_overtravel: float = 0.12
    ik_dt: float = 0.01
    ik_iterations: int = 260
    ik_position_tolerance: float = 2.0e-3
    ik_orientation_tolerance: float = np.deg2rad(2.0)

    def __post_init__(self) -> None:
        if self.control_hz <= 0.0:
            raise ValueError("control_hz must be positive")
        if self.pull_speed <= 0.0:
            raise ValueError("pull_speed must be positive")
        if self.wheel_radius <= 0.0 or self.half_track <= 0.0:
            raise ValueError("wheel geometry must be positive")
        if self.ik_dt <= 0.0 or self.ik_iterations < 1:
            raise ValueError("IK integration settings must be positive")


@dataclass(frozen=True)
class MinkIKResult:
    """Result of an arm-only Mink pose solve."""

    arm_qpos: FloatArray
    position_error: float
    orientation_error: float
    iterations: int
    non_arm_displacement: float


@dataclass(frozen=True)
class TraditionalDoorDiagnostics:
    """Values reported by one controller update."""

    phase: TraditionalDoorPhase
    phase_time: float
    elapsed_time: float
    door_angle: float
    handle_angle: float
    latch_released: bool
    grasp_attached: bool
    tcp_handle_distance: float
    handle_contact_force: float
    base_forward_command: float
    base_yaw_command: float
    base_yaw: float
    upright: float
    arm_joint_margin: float
    max_torque: float
    ik_position_error: float
    ik_orientation_error: float
    success: bool
    failure_reason: str | None


def _smoothstep(value: float) -> float:
    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _orientation_error(target: FloatArray, actual: FloatArray) -> float:
    relative = np.asarray(target) @ np.asarray(actual).T
    cosine = float(np.clip(0.5 * (np.trace(relative) - 1.0), -1.0, 1.0))
    return float(np.arccos(cosine))


def _se3(rotation: FloatArray, position: FloatArray) -> mink.SE3:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return mink.SE3.from_matrix(matrix)


class MinkArmIK:
    """Arm-only pose IK on a full B2-W + Z1 + door MuJoCo model.

    Mink sees every degree of freedom in the scene. A ``DofFreezingTask`` is
    therefore passed as an exact equality constraint so the solver cannot
    satisfy a TCP target by moving the floating base, wheels, door, or handle.
    """

    def __init__(
        self,
        env: B2WZ1DoorEnv,
        *,
        dt: float = 0.01,
        max_joint_velocity: float = 1.5,
    ) -> None:
        self.env = env
        self.model = env.model
        self.dt = float(dt)
        self.configuration = mink.Configuration(self.model)
        self.frame_task = mink.FrameTask(
            frame_name="z1_ee",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.25,
            lm_damping=1.0e-5,
        )
        self.posture_task = mink.PostureTask(self.model, cost=1.0e-3)

        self.arm_qpos_addresses = np.asarray(env._qpos_adr[16:22], dtype=np.int32)
        self.arm_dof_addresses = np.asarray(env._dof_adr[16:22], dtype=np.int32)
        arm_dofs = set(int(index) for index in self.arm_dof_addresses)
        frozen_dofs = [index for index in range(self.model.nv) if index not in arm_dofs]
        self.freeze_non_arm = mink.DofFreezingTask(
            model=self.model,
            dof_indices=frozen_dofs,
        )
        self.limits = [
            mink.ConfigurationLimit(model=self.model),
            mink.VelocityLimit(
                self.model,
                {name: max_joint_velocity for name in ARM_JOINTS},
            ),
        ]

    def solve_pose(
        self,
        qpos: FloatArray,
        target_position: FloatArray,
        target_rotation: FloatArray,
        *,
        max_iterations: int,
        position_tolerance: float = 1.0e-4,
        orientation_tolerance: float = 1.0e-3,
    ) -> MinkIKResult:
        """Solve a world-frame TCP pose without moving any non-arm DoF."""
        initial_qpos = np.asarray(qpos, dtype=np.float64).copy()
        if initial_qpos.shape != (self.model.nq,):
            raise ValueError(f"qpos must have shape ({self.model.nq},)")
        target_position = np.asarray(target_position, dtype=np.float64)
        target_rotation = np.asarray(target_rotation, dtype=np.float64)
        if target_position.shape != (3,) or target_rotation.shape != (3, 3):
            raise ValueError("target position/rotation must have shape (3,) and (3, 3)")

        self.configuration.update(initial_qpos)
        self.posture_task.set_target(initial_qpos)
        self.frame_task.set_target(_se3(target_rotation, target_position))

        position_error = np.inf
        orientation_error = np.inf
        iterations = 0
        for iteration in range(1, max_iterations + 1):
            iterations = iteration
            velocity = mink.solve_ik(
                self.configuration,
                [self.frame_task, self.posture_task],
                self.dt,
                "daqp",
                limits=self.limits,
                constraints=[self.freeze_non_arm],
                damping=1.0e-6,
                safety_break=True,
            )
            self.configuration.integrate_inplace(velocity, self.dt)
            pose = self.configuration.get_transform_frame_to_world("z1_ee", "site")
            matrix = pose.as_matrix()
            position_error = float(np.linalg.norm(target_position - matrix[:3, 3]))
            orientation_error = _orientation_error(target_rotation, matrix[:3, :3])
            if (
                position_error <= position_tolerance
                and orientation_error <= orientation_tolerance
            ):
                break

        solved_qpos = self.configuration.q.copy()
        non_arm_mask = np.ones(self.model.nq, dtype=bool)
        non_arm_mask[self.arm_qpos_addresses] = False
        non_arm_displacement = float(
            np.max(np.abs(solved_qpos[non_arm_mask] - initial_qpos[non_arm_mask]))
        )
        return MinkIKResult(
            arm_qpos=solved_qpos[self.arm_qpos_addresses].copy(),
            position_error=position_error,
            orientation_error=orientation_error,
            iterations=iterations,
            non_arm_displacement=non_arm_displacement,
        )


class TraditionalDoorController:
    """Deterministic IK/impedance controller for the B2-W pull-door scene."""

    def __init__(
        self,
        env: B2WZ1DoorEnv,
        config: TraditionalDoorConfig | None = None,
    ) -> None:
        self.env = env
        self.model = env.model
        self.data = env.data
        self.config = config or TraditionalDoorConfig()
        frame_skip = (1.0 / self.config.control_hz) / self.model.opt.timestep
        self.frame_skip = int(round(frame_skip))
        if self.frame_skip < 1 or not np.isclose(frame_skip, self.frame_skip):
            raise ValueError("control_hz must divide the MuJoCo simulation frequency")
        self.dt = self.frame_skip * self.model.opt.timestep
        self.ik = MinkArmIK(env, dt=self.config.ik_dt)
        self._scratch_data = mujoco.MjData(self.model)

        target_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "tcp_target",
        )
        self._target_mocap_id = (
            int(self.model.body_mocapid[target_body_id]) if target_body_id >= 0 else -1
        )

        self.phase = TraditionalDoorPhase.SETTLE
        self.failure_reason: str | None = None
        self._global_step = 0
        self._phase_step = 0
        self._leg_reference = LEG_NOMINAL.copy()
        self._arm_reference = PREGRASP_ARM.copy()
        self._phase_start_arm_reference = PREGRASP_ARM.copy()
        self._phase_goal_arm_reference = PREGRASP_ARM.copy()
        self._gripper_reference = -1.35
        self._wheel_velocity_target = np.zeros(4, dtype=np.float64)
        self._arm_gain = self.config.free_space_arm_gain
        self._last_torque = np.zeros(23, dtype=np.float64)
        self._base_forward_command = 0.0
        self._base_yaw_command = 0.0
        self._pull_start_door_angle = 0.0
        self._pull_start_base_yaw = 0.0
        self._target_position = np.zeros(3, dtype=np.float64)
        self._target_rotation = np.eye(3, dtype=np.float64)
        self._last_ik_result = MinkIKResult(
            arm_qpos=PREGRASP_ARM.copy(),
            position_error=0.0,
            orientation_error=0.0,
            iterations=0,
            non_arm_displacement=0.0,
        )

    @property
    def elapsed_time(self) -> float:
        return self._global_step * self.dt

    @property
    def phase_time(self) -> float:
        return self._phase_step * self.dt

    @property
    def terminal(self) -> bool:
        return self.phase in (
            TraditionalDoorPhase.SUCCESS,
            TraditionalDoorPhase.FAILURE,
        )

    @property
    def success(self) -> bool:
        return self.phase == TraditionalDoorPhase.SUCCESS

    def reset(self, *, seed: int | None = None) -> TraditionalDoorDiagnostics:
        """Reset the scene and initialize the staged behavior."""
        self.env.reset(seed=seed)
        self.phase = TraditionalDoorPhase.SETTLE
        self.failure_reason = None
        self._global_step = 0
        self._phase_step = 0
        self._leg_reference[:] = LEG_NOMINAL
        self._arm_reference[:] = self.data.qpos[self.env._qpos_adr[16:22]]
        self._phase_start_arm_reference[:] = self._arm_reference
        self._phase_goal_arm_reference[:] = self._arm_reference
        self._gripper_reference = -1.35
        self._wheel_velocity_target.fill(0.0)
        self._arm_gain = self.config.free_space_arm_gain
        self._last_torque.fill(0.0)
        self._base_forward_command = 0.0
        self._base_yaw_command = 0.0
        self._pull_start_door_angle = 0.0
        self._pull_start_base_yaw = 0.0
        self._target_position[:] = self.data.site_xpos[self.env._handle_site_id]
        self._target_rotation[:] = self.data.site_xmat[self.env._ee_site_id].reshape(3, 3)
        self._update_target_marker()
        return self.diagnostics()

    def _base_yaw(self) -> float:
        rotation = self.env._base_rotation()
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def _transition(self, phase: TraditionalDoorPhase) -> None:
        self.phase = phase
        self._phase_step = 0
        if phase == TraditionalDoorPhase.APPROACH:
            self._plan_approach()
        elif phase == TraditionalDoorPhase.UNLATCH:
            self._plan_unlatch()
        elif phase == TraditionalDoorPhase.PULL:
            self._pull_start_door_angle = self.env.door_angle
            self._pull_start_base_yaw = self._base_yaw()
            self._arm_reference[:] = self.data.qpos[self.env._qpos_adr[16:22]]
            self._arm_gain = self.config.compliant_arm_gain
        elif phase in (TraditionalDoorPhase.SUCCESS, TraditionalDoorPhase.FAILURE):
            self._wheel_velocity_target.fill(0.0)
            self._base_forward_command = 0.0
            self._base_yaw_command = 0.0
            self._arm_reference[:] = self.data.qpos[self.env._qpos_adr[16:22]]
            self._arm_gain = self.config.compliant_arm_gain

    def _fail(self, reason: str) -> None:
        if not self.terminal:
            self.failure_reason = reason
            self._transition(TraditionalDoorPhase.FAILURE)

    def _plan_approach(self) -> None:
        """Use Mink to generate the collision-free-side pre-grasp reference."""
        self._phase_start_arm_reference[:] = self.data.qpos[
            self.env._qpos_adr[16:22]
        ]
        door_rotation = self.data.xmat[self.env._door_body_id].reshape(3, 3)
        approach_offset_door = np.array([-0.023, -0.022, 0.0], dtype=np.float64)
        self._target_position[:] = (
            self.data.site_xpos[self.env._handle_site_id]
            + door_rotation @ approach_offset_door
        )

        # PREGRASP_ARM calibrates a sensible wrist orientation for this mount.
        self._scratch_data.qpos[:] = self.data.qpos
        self._scratch_data.qpos[self.env._qpos_adr[16:22]] = PREGRASP_ARM
        mujoco.mj_forward(self.model, self._scratch_data)
        self._target_rotation[:] = self._scratch_data.site_xmat[
            self.env._ee_site_id
        ].reshape(3, 3)
        self._last_ik_result = self.ik.solve_pose(
            self.data.qpos,
            self._target_position,
            self._target_rotation,
            max_iterations=self.config.ik_iterations,
        )
        self._phase_goal_arm_reference[:] = self._last_ik_result.arm_qpos
        if not self._ik_result_usable(self._last_ik_result):
            self._fail("approach IK target is unreachable")
        self._update_target_marker()

    def _plan_unlatch(self) -> None:
        """Solve the TCP pose corresponding to a depressed door handle."""
        self._phase_start_arm_reference[:] = self.data.qpos[
            self.env._qpos_adr[16:22]
        ]
        current_handle_rotation = self.data.site_xmat[
            self.env._handle_site_id
        ].reshape(3, 3)
        current_tcp_rotation = self.data.site_xmat[self.env._ee_site_id].reshape(3, 3)
        handle_to_tcp_rotation = current_handle_rotation.T @ current_tcp_rotation

        self._scratch_data.qpos[:] = self.data.qpos
        handle_high = float(self.model.jnt_range[self.env._handle_joint_id, 1])
        desired_handle_angle = min(
            self.env._release_angle + self.config.handle_overtravel,
            handle_high - 0.02,
        )
        self._scratch_data.qpos[self.env._handle_qpos_adr] = desired_handle_angle
        mujoco.mj_forward(self.model, self._scratch_data)
        self._target_position[:] = self._scratch_data.site_xpos[
            self.env._handle_site_id
        ]
        desired_handle_rotation = self._scratch_data.site_xmat[
            self.env._handle_site_id
        ].reshape(3, 3)
        self._target_rotation[:] = desired_handle_rotation @ handle_to_tcp_rotation

        self._last_ik_result = self.ik.solve_pose(
            self.data.qpos,
            self._target_position,
            self._target_rotation,
            max_iterations=self.config.ik_iterations,
        )
        self._phase_goal_arm_reference[:] = self._last_ik_result.arm_qpos
        if not self._ik_result_usable(self._last_ik_result):
            self._fail("unlatch IK target is unreachable")
        self._update_target_marker()

    def _ik_result_usable(self, result: MinkIKResult) -> bool:
        return bool(
            result.position_error <= self.config.ik_position_tolerance
            and result.orientation_error <= self.config.ik_orientation_tolerance
            and result.non_arm_displacement <= 1.0e-9
            and np.isfinite(result.arm_qpos).all()
        )

    def _update_commands(self) -> None:
        self._wheel_velocity_target.fill(0.0)
        self._base_forward_command = 0.0
        self._base_yaw_command = 0.0
        self._gripper_reference = -1.35
        self._arm_gain = self.config.free_space_arm_gain

        if self.phase == TraditionalDoorPhase.SETTLE:
            return
        if self.phase == TraditionalDoorPhase.APPROACH:
            alpha = _smoothstep(self.phase_time / self.config.approach_seconds)
            self._arm_reference[:] = (
                (1.0 - alpha) * self._phase_start_arm_reference
                + alpha * self._phase_goal_arm_reference
            )
            return
        if self.phase == TraditionalDoorPhase.GRASP:
            self._arm_reference[:] = self._phase_goal_arm_reference
            self._gripper_reference = 0.0
            return
        if self.phase == TraditionalDoorPhase.UNLATCH:
            alpha = _smoothstep(self.phase_time / self.config.unlatch_seconds)
            self._arm_reference[:] = (
                (1.0 - alpha) * self._phase_start_arm_reference
                + alpha * self._phase_goal_arm_reference
            )
            self._gripper_reference = 0.0
            return
        if self.phase == TraditionalDoorPhase.PULL:
            self._gripper_reference = 0.0
            self._arm_gain = self.config.compliant_arm_gain
            ramp = _smoothstep(self.phase_time / self.config.pull_ramp_seconds)
            forward_speed = -self.config.pull_speed * ramp
            desired_yaw = self._pull_start_base_yaw + (
                self.env.door_angle - self._pull_start_door_angle
            )
            yaw_error = _wrap_angle(desired_yaw - self._base_yaw())
            yaw_rate = float(
                np.clip(
                    self.config.yaw_gain * yaw_error,
                    -self.config.max_yaw_rate,
                    self.config.max_yaw_rate,
                )
            )
            right_velocity = (
                forward_speed + self.config.half_track * yaw_rate
            ) / self.config.wheel_radius
            left_velocity = (
                forward_speed - self.config.half_track * yaw_rate
            ) / self.config.wheel_radius
            self._wheel_velocity_target[:] = np.clip(
                [right_velocity, left_velocity, right_velocity, left_velocity],
                -12.0,
                12.0,
            )
            self._base_forward_command = forward_speed
            self._base_yaw_command = yaw_rate
            return

        # Terminal phases hold the final pose with zero wheel velocity.
        self._gripper_reference = 0.0
        self._arm_gain = self.config.compliant_arm_gain

    def _compute_torque(self) -> FloatArray:
        qpos, qvel = self.env._joint_state()
        torque = np.zeros(23, dtype=np.float64)
        torque[:12] = (
            2.0 * LEG_KP * (self._leg_reference - qpos[:12])
            - 6.0 * LEG_KD * qvel[:12]
        )
        torque[12:16] = 5.0 * (self._wheel_velocity_target - qvel[12:16])
        torque[16:22] = (
            self._arm_gain * ARM_KP * (self._arm_reference - qpos[16:22])
            - self.config.arm_damping_scale * ARM_KD * qvel[16:22]
            + self.data.qfrc_bias[self.env._dof_adr[16:22]]
        )
        torque[22] = (
            GRIPPER_KP * (self._gripper_reference - qpos[22])
            - GRIPPER_KD * qvel[22]
        )
        return np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)

    def _advance_physics(self) -> None:
        self._last_torque[:] = self._compute_torque()
        self.data.ctrl[self.env._actuator_ids] = self._last_torque
        for _ in range(self.frame_skip):
            self.env._update_grasp_constraint()
            self.env._update_latch_state()
            self.env._apply_door_passive_forces()
            mujoco.mj_step(self.model, self.data)
        self.env._update_latch_state()
        self.env._update_phase()

    def _check_transitions(self) -> None:
        if not self.env._is_healthy():
            self._fail("robot fell or simulation became non-finite")
            return
        if self.phase in (TraditionalDoorPhase.UNLATCH, TraditionalDoorPhase.PULL):
            if not self.env._grasp_attached:
                self._fail("gripper lost the door handle")
                return

        if self.phase == TraditionalDoorPhase.SETTLE:
            if self.phase_time >= self.config.settle_seconds:
                self._transition(TraditionalDoorPhase.APPROACH)
        elif self.phase == TraditionalDoorPhase.APPROACH:
            distance = float(np.linalg.norm(self.env._handle_error_world()))
            if (
                self.phase_time >= self.config.approach_seconds
                and distance <= self.config.approach_distance
            ):
                self._transition(TraditionalDoorPhase.GRASP)
            elif self.phase_time >= self.config.approach_timeout:
                self._fail("TCP did not reach the pre-grasp region")
        elif self.phase == TraditionalDoorPhase.GRASP:
            if (
                self.env._grasp_attached
                and self.phase_time >= self.config.grasp_settle_seconds
            ):
                self._transition(TraditionalDoorPhase.UNLATCH)
            elif self.phase_time >= self.config.grasp_timeout:
                self._fail("gripper did not attach to the handle")
        elif self.phase == TraditionalDoorPhase.UNLATCH:
            if self.env._latch_released:
                self._transition(TraditionalDoorPhase.PULL)
            elif self.phase_time >= self.config.unlatch_timeout:
                self._fail("handle rotation did not release the latch")
        elif self.phase == TraditionalDoorPhase.PULL:
            if self.env.door_angle <= DOOR_SUCCESS_ANGLE:
                self._transition(TraditionalDoorPhase.SUCCESS)
            elif self.phase_time >= self.config.pull_timeout:
                self._fail("base did not pull the door to the success angle")

    def _update_target_marker(self) -> None:
        if self._target_mocap_id < 0:
            return
        if self.phase in (
            TraditionalDoorPhase.PULL,
            TraditionalDoorPhase.SUCCESS,
            TraditionalDoorPhase.FAILURE,
        ):
            position = self.data.site_xpos[self.env._handle_site_id]
            rotation = self.data.site_xmat[self.env._ee_site_id].reshape(3, 3)
        else:
            position = self._target_position
            rotation = self._target_rotation
        self.data.mocap_pos[self._target_mocap_id] = position
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, np.asarray(rotation, dtype=float).ravel())
        self.data.mocap_quat[self._target_mocap_id] = quaternion

    def step(self) -> TraditionalDoorDiagnostics:
        """Advance one control update and return diagnostics."""
        self._update_commands()
        self._advance_physics()
        self._global_step += 1
        self._phase_step += 1
        if not self.terminal:
            self._check_transitions()
        self._update_target_marker()
        return self.diagnostics()

    def diagnostics(self) -> TraditionalDoorDiagnostics:
        rotation = self.env._base_rotation()
        arm_qpos = self.data.qpos[self.env._qpos_adr[16:22]]
        arm_ranges = self.model.jnt_range[self.env._joint_ids[16:22]]
        widths = np.maximum(arm_ranges[:, 1] - arm_ranges[:, 0], 1.0e-9)
        arm_joint_margin = float(
            np.min(
                np.minimum(
                    arm_qpos - arm_ranges[:, 0],
                    arm_ranges[:, 1] - arm_qpos,
                )
                / widths
            )
        )
        _, contact_force = self.env._gripper_contact()
        return TraditionalDoorDiagnostics(
            phase=self.phase,
            phase_time=self.phase_time,
            elapsed_time=self.elapsed_time,
            door_angle=self.env.door_angle,
            handle_angle=self.env.handle_angle,
            latch_released=self.env._latch_released,
            grasp_attached=self.env._grasp_attached,
            tcp_handle_distance=float(np.linalg.norm(self.env._handle_error_world())),
            handle_contact_force=contact_force,
            base_forward_command=self._base_forward_command,
            base_yaw_command=self._base_yaw_command,
            base_yaw=self._base_yaw(),
            upright=float(rotation[2, 2]),
            arm_joint_margin=arm_joint_margin,
            max_torque=float(np.max(np.abs(self._last_torque))),
            ik_position_error=self._last_ik_result.position_error,
            ik_orientation_error=self._last_ik_result.orientation_error,
            success=self.success,
            failure_reason=self.failure_reason,
        )
