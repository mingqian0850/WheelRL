"""MuJoCo Jacobian whole-body controller for a stationary B2-W + Z1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from wheelrl.envs.b2w_z1 import (
    ACTUATORS,
    ARM_JOINTS,
    ARM_KD,
    ARM_KP,
    ARM_NOMINAL,
    GRIPPER_KD,
    GRIPPER_KP,
    GRIPPER_NOMINAL,
    LEG_JOINTS,
    LEG_KD,
    LEG_KP,
    TORQUE_LIMITS,
    WHEEL_JOINTS,
)

FloatArray = NDArray[np.float64]

WBC_MODEL_PATH = (
    Path(__file__).resolve().parent / "assets" / "b2w_z1" / "wbc_scene.xml"
)
WBC_STAND_LEG_NOMINAL = np.tile(
    np.array([0.0, 1.0, -2.0], dtype=np.float64),
    4,
)
WHEEL_BODIES = (
    "FR_wheel_link",
    "FL_wheel_link",
    "RR_wheel_link",
    "RL_wheel_link",
)
WHEEL_RADIUS = 0.10
SPEED_PROFILE_SCALES = {
    "precision": 0.40,
    "normal": 1.00,
    "fast": 2.00,
}
ROTATION_SPEED_PROFILE_SCALES = {
    "precision": 0.40,
    "normal": 1.00,
    "fast": 1.50,
}

# Task-space loop gains. The desired twists are PD in task space:
#   v_des = Kp * error - Kd * v_measured
# The velocity feedback damps the overshoot that a pure P command produces
# when the body carries momentum through the target.
TCP_POS_GAIN = 3.0
TCP_ORI_GAIN = 3.0
TCP_VEL_DAMPING = 0.60
BASE_VEL_DAMPING = 0.80
# Arm servo gains are boosted locally (the RL envs keep their own tuning).
ARM_KP_SCALE = 1.5
ARM_KD_SCALE = 2.0


def rotation_vector(rotation: FloatArray) -> FloatArray:
    """Return the SO(3) logarithm of a 3x3 rotation matrix."""
    cosine = float(np.clip(0.5 * (np.trace(rotation) - 1.0), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = 0.5 * np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1.0e-7:
        return skew
    if np.pi - angle < 1.0e-4:
        diagonal = np.maximum(0.5 * (np.diag(rotation) + 1.0), 0.0)
        axis = np.sqrt(diagonal)
        axis[1] = np.copysign(axis[1], rotation[0, 1] + rotation[1, 0])
        axis[2] = np.copysign(axis[2], rotation[0, 2] + rotation[2, 0])
        return angle * axis / max(float(np.linalg.norm(axis)), 1.0e-12)
    return angle * skew / np.sin(angle)


def axis_angle_rotation(axis: FloatArray, angle: float) -> FloatArray:
    """Create a rotation matrix from a world-frame axis and angle."""
    unit_axis = np.asarray(axis, dtype=np.float64)
    unit_axis = unit_axis / np.linalg.norm(unit_axis)
    x, y, z = unit_axis
    cross = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def rpy_rotation(rpy_radians: FloatArray) -> FloatArray:
    """Create Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = np.asarray(rpy_radians, dtype=np.float64)
    return (
        axis_angle_rotation(np.array([0.0, 0.0, 1.0]), float(yaw))
        @ axis_angle_rotation(np.array([0.0, 1.0, 0.0]), float(pitch))
        @ axis_angle_rotation(np.array([1.0, 0.0, 0.0]), float(roll))
    )


def rotation_rpy(rotation: FloatArray) -> FloatArray:
    """Return roll, pitch, yaw for a rotation built as Rz Ry Rx."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    pitch = float(np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(np.cos(pitch)) > 1.0e-7:
        roll = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-matrix[0, 1], matrix[1, 1]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


@dataclass(frozen=True)
class WBCDiagnostics:
    """Tracking and stability values for display and tests."""

    position_error: float
    orientation_error: float
    base_position_error: float
    upright: float
    max_torque: float
    mobile_base_active: bool
    base_goal_distance: float


class B2WZ1WholeBodyController:
    """Weighted differential WBC with joint-level torque servos.

    The optimization variable is generalized velocity for the floating base,
    12 leg joints, four wheels, and six Z1 joints. Wheel rolling constraints,
    base motion/stabilization, TCP SE(3), and posture tasks are solved together.
    The commanded leg and arm velocities are integrated into joint references;
    the wheels receive velocity targets. Both are tracked by 500 Hz torque
    servos.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        control_hz: float = 100.0,
        base_assist: float = 0.25,
        auto_drive: bool = True,
        speed_profile: str = "normal",
    ) -> None:
        self.model = model
        self.data = data
        self.control_dt = 1.0 / control_hz
        frame_skip = self.control_dt / model.opt.timestep
        self.frame_skip = int(round(frame_skip))
        if self.frame_skip < 1 or not np.isclose(frame_skip, self.frame_skip):
            raise ValueError("control_hz must divide the MuJoCo simulation frequency")
        self.base_assist = float(np.clip(base_assist, 0.0, 1.0))
        self.auto_drive = auto_drive
        self._speed_profile = ""
        self._speed_scale = 1.0
        self._rotation_speed_scale = 1.0
        self.set_speed_profile(speed_profile)

        self._base_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._tcp_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "z1_ee")
        self._wheel_body_ids = self._ids(mujoco.mjtObj.mjOBJ_BODY, WHEEL_BODIES)
        self._actuator_ids = self._ids(mujoco.mjtObj.mjOBJ_ACTUATOR, ACTUATORS)

        leg_joint_ids = self._ids(mujoco.mjtObj.mjOBJ_JOINT, LEG_JOINTS)
        wheel_joint_ids = self._ids(mujoco.mjtObj.mjOBJ_JOINT, WHEEL_JOINTS)
        arm_joint_ids = self._ids(mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS)
        self._gripper_joint_id = self._id(
            mujoco.mjtObj.mjOBJ_JOINT,
            "z1_gripper_joint",
        )

        self._leg_qpos_adr = model.jnt_qposadr[leg_joint_ids]
        self._wheel_qpos_adr = model.jnt_qposadr[wheel_joint_ids]
        self._arm_qpos_adr = model.jnt_qposadr[arm_joint_ids]
        self._gripper_qpos_adr = int(model.jnt_qposadr[self._gripper_joint_id])
        self._leg_dof_adr = model.jnt_dofadr[leg_joint_ids]
        self._wheel_dof_adr = model.jnt_dofadr[wheel_joint_ids]
        self._arm_dof_adr = model.jnt_dofadr[arm_joint_ids]
        self._gripper_dof_adr = int(model.jnt_dofadr[self._gripper_joint_id])

        self._optimized_dofs = np.concatenate(
            [
                np.arange(6, dtype=np.int32),
                self._leg_dof_adr,
                self._wheel_dof_adr,
                self._arm_dof_adr,
            ]
        )
        self._arm_joint_ranges = model.jnt_range[arm_joint_ids].copy()

        target_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "tcp_target",
        )
        self._target_mocap_id = (
            int(model.body_mocapid[target_body_id]) if target_body_id >= 0 else -1
        )

        self._joint_reference = np.concatenate(
            [WBC_STAND_LEG_NOMINAL, ARM_NOMINAL]
        ).copy()
        self._gripper_target = float(GRIPPER_NOMINAL[0])
        self._active = False
        self._home_base_position = np.zeros(3, dtype=np.float64)
        self._home_base_rotation = np.eye(3, dtype=np.float64)
        self._desired_base_position = np.zeros(3, dtype=np.float64)
        self._desired_base_rotation = np.eye(3, dtype=np.float64)
        self._base_goal_distance = 0.0
        self._home_tcp_position = np.zeros(3, dtype=np.float64)
        self._home_tcp_rotation = np.eye(3, dtype=np.float64)
        self._home_tcp_position_base = np.zeros(3, dtype=np.float64)
        self._home_tcp_rotation_base = np.eye(3, dtype=np.float64)
        self._target_position = np.zeros(3, dtype=np.float64)
        self._target_rotation = np.eye(3, dtype=np.float64)
        self._last_torque = np.zeros(23, dtype=np.float64)
        self._wheel_velocity_target = np.zeros(4, dtype=np.float64)
        self._mobile_base_active = False

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return object_id

    def _ids(
        self,
        object_type: mujoco.mjtObj,
        names: tuple[str, ...],
    ) -> NDArray[np.int32]:
        return np.asarray(
            [self._id(object_type, name) for name in names],
            dtype=np.int32,
        )

    @property
    def home_tcp_position(self) -> FloatArray:
        return self._home_tcp_position.copy()

    @property
    def home_tcp_rotation(self) -> FloatArray:
        return self._home_tcp_rotation.copy()

    @property
    def target_position(self) -> FloatArray:
        return self._target_position.copy()

    @property
    def target_rotation(self) -> FloatArray:
        return self._target_rotation.copy()

    @property
    def gripper_target(self) -> float:
        return self._gripper_target

    @property
    def gripper_closed(self) -> bool:
        return self._gripper_target > -0.5

    @property
    def speed_profile(self) -> str:
        return self._speed_profile

    @property
    def speed_scale(self) -> float:
        return self._speed_scale

    @property
    def rotation_speed_scale(self) -> float:
        return self._rotation_speed_scale

    @property
    def tcp_position(self) -> FloatArray:
        return self.data.site_xpos[self._tcp_site_id].copy()

    @property
    def tcp_rotation(self) -> FloatArray:
        return self.data.site_xmat[self._tcp_site_id].reshape(3, 3).copy()

    def reset(self) -> None:
        """Reset to a crouched four-wheel stance."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = np.array(
            [0.0, 0.0, 0.55, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self.data.qpos[self._leg_qpos_adr] = WBC_STAND_LEG_NOMINAL
        self.data.qpos[self._wheel_qpos_adr] = 0.0
        self.data.qpos[self._arm_qpos_adr] = ARM_NOMINAL
        self.data.qpos[self._gripper_qpos_adr] = GRIPPER_NOMINAL[0]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._joint_reference[:] = np.concatenate(
            [WBC_STAND_LEG_NOMINAL, ARM_NOMINAL]
        )
        self._gripper_target = float(GRIPPER_NOMINAL[0])
        self._active = False
        self._mobile_base_active = False
        self._wheel_velocity_target.fill(0.0)
        self._target_position[:] = self.tcp_position
        self._target_rotation[:] = self.tcp_rotation
        self._update_target_marker()

    def capture_reference(self) -> None:
        """Capture the settled base and TCP poses, then enable WBC."""
        self._home_base_position[:] = self.data.xpos[self._base_body_id]
        self._home_base_rotation[:] = self.data.xmat[self._base_body_id].reshape(3, 3)
        self._desired_base_rotation[:] = self._home_base_rotation
        self._home_tcp_position[:] = self.tcp_position
        self._home_tcp_rotation[:] = self.tcp_rotation
        self._home_tcp_position_base[:] = self._home_base_rotation.T @ (
            self._home_tcp_position - self._home_base_position
        )
        self._home_tcp_rotation_base[:] = (
            self._home_base_rotation.T @ self._home_tcp_rotation
        )
        self._target_position[:] = self._home_tcp_position
        self._target_rotation[:] = self._home_tcp_rotation
        self._active = True
        self._update_target_marker()

    def set_target_pose(
        self,
        position: FloatArray,
        rotation: FloatArray,
    ) -> None:
        """Set an absolute world-frame TCP target pose."""
        position_array = np.asarray(position, dtype=np.float64)
        rotation_array = np.asarray(rotation, dtype=np.float64)
        if position_array.shape != (3,) or rotation_array.shape != (3, 3):
            raise ValueError("position must be (3,) and rotation must be (3, 3)")
        self._target_position[:] = position_array
        self._target_rotation[:] = rotation_array
        self._update_target_marker()

    def set_home_offset(
        self,
        position_offset: FloatArray,
        rpy_offset_radians: FloatArray,
    ) -> None:
        """Set target from a world position and local RPY offset from home."""
        self.set_target_pose(
            self._home_tcp_position + np.asarray(position_offset, dtype=np.float64),
            self._home_tcp_rotation
            @ rpy_rotation(np.asarray(rpy_offset_radians, dtype=np.float64)),
        )

    def nudge_position(self, world_delta: FloatArray) -> None:
        self._target_position += np.asarray(world_delta, dtype=np.float64)
        self._update_target_marker()

    def nudge_position_base(self, local_delta: FloatArray) -> None:
        """Nudge along base forward/left and world vertical directions."""
        delta = np.asarray(local_delta, dtype=np.float64)
        if delta.shape != (3,):
            raise ValueError("local_delta must have shape (3,)")
        base_rotation = self.data.xmat[self._base_body_id].reshape(3, 3)
        forward = base_rotation[:, 0].copy()
        forward[2] = 0.0
        forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
        left = np.array([-forward[1], forward[0], 0.0], dtype=np.float64)
        world_delta = (
            delta[0] * forward
            + delta[1] * left
            + delta[2] * np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )
        self.nudge_position(world_delta)

    def nudge_orientation(self, world_axis: FloatArray, angle: float) -> None:
        self._target_rotation[:] = (
            axis_angle_rotation(np.asarray(world_axis, dtype=np.float64), angle)
            @ self._target_rotation
        )
        self._update_target_marker()

    def go_home(self) -> None:
        self.set_target_pose(self._home_tcp_position, self._home_tcp_rotation)

    def set_gripper(self, *, closed: bool) -> None:
        self._gripper_target = 0.0 if closed else -1.35

    def toggle_gripper(self) -> None:
        self.set_gripper(closed=not self.gripper_closed)

    def set_speed_profile(self, profile: str) -> None:
        """Set a safe motion-speed multiplier without changing control rate."""
        if profile not in SPEED_PROFILE_SCALES:
            choices = ", ".join(SPEED_PROFILE_SCALES)
            raise ValueError(f"speed profile must be one of: {choices}")
        self._speed_profile = profile
        self._speed_scale = SPEED_PROFILE_SCALES[profile]
        self._rotation_speed_scale = ROTATION_SPEED_PROFILE_SCALES[profile]

    def _update_target_marker(self) -> None:
        if self._target_mocap_id < 0:
            return
        self.data.mocap_pos[self._target_mocap_id] = self._target_position
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, self._target_rotation.ravel())
        self.data.mocap_quat[self._target_mocap_id] = quaternion

    def _body_jacobian(self, body_id: int) -> FloatArray:
        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(
            self.model,
            self.data,
            jacobian_position,
            jacobian_rotation,
            body_id,
        )
        return np.vstack([jacobian_position, jacobian_rotation])

    def _site_jacobian(self, site_id: int) -> FloatArray:
        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self.model,
            self.data,
            jacobian_position,
            jacobian_rotation,
            site_id,
        )
        return np.vstack([jacobian_position, jacobian_rotation])

    def _solve_generalized_velocity(self) -> FloatArray:
        task_jacobians: list[FloatArray] = []
        task_velocities: list[FloatArray] = []
        task_weights: list[FloatArray] = []

        base_rotation = self.data.xmat[self._base_body_id].reshape(3, 3)
        forward = base_rotation[:, 0].copy()
        forward[2] = 0.0
        forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
        lateral = np.array([-forward[1], forward[0], 0.0], dtype=np.float64)
        vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        for index, wheel_body_id in enumerate(self._wheel_body_ids):
            wheel_jacobian = self._body_jacobian(int(wheel_body_id))[:3]
            rolling_jacobian = np.vstack(
                [
                    lateral @ wheel_jacobian,
                    vertical @ wheel_jacobian,
                    forward @ wheel_jacobian,
                ]
            )[:, self._optimized_dofs]
            rolling_jacobian[2, 18 + index] -= WHEEL_RADIUS
            task_jacobians.append(rolling_jacobian)
            task_velocities.append(np.zeros(3, dtype=np.float64))
            task_weights.append(np.full(3, 100.0, dtype=np.float64))

        target_delta_home = self._home_base_rotation.T @ (
            self._target_position - self._home_tcp_position
        )
        arm_allowance = np.array([0.16, 0.10], dtype=np.float64)
        retained_by_arm = np.clip(
            target_delta_home[:2],
            -arm_allowance,
            arm_allowance,
        )
        target_requires_driving = bool(
            np.any(np.abs(target_delta_home[:2]) > arm_allowance)
        )
        home_yaw = float(
            np.arctan2(
                self._home_base_rotation[1, 0],
                self._home_base_rotation[0, 0],
            )
        )
        if target_requires_driving and self.auto_drive:
            target_from_home_base = (
                self._target_position[:2] - self._home_base_position[:2]
            )
            home_tcp_heading = float(
                np.arctan2(
                    self._home_tcp_position_base[1],
                    self._home_tcp_position_base[0],
                )
            )
            desired_yaw = (
                float(
                    np.arctan2(
                        target_from_home_base[1],
                        target_from_home_base[0],
                    )
                )
                - home_tcp_heading
            )
            desired_yaw = home_yaw + self._wrap_angle(desired_yaw - home_yaw)
            desired_tcp_relative = self._home_tcp_position_base.copy()
            desired_tcp_relative[0] += 0.08
            yaw_rotation = axis_angle_rotation(
                np.array([0.0, 0.0, 1.0]),
                desired_yaw - home_yaw,
            )
            self._desired_base_rotation[:] = (
                yaw_rotation @ self._home_base_rotation
            )
            desired_tcp_offset_world = (
                self._desired_base_rotation @ desired_tcp_relative
            )
            self._desired_base_position[:2] = (
                self._target_position[:2] - desired_tcp_offset_world[:2]
            )
            self._desired_base_position[2] = (
                self._home_base_position[2]
                + np.clip(
                    self.base_assist * target_delta_home[2],
                    -0.04,
                    0.04,
                )
            )
        else:
            local_base_shift = np.array(
                [
                    self.base_assist * retained_by_arm[0],
                    self.base_assist * retained_by_arm[1],
                    np.clip(
                        self.base_assist * target_delta_home[2],
                        -0.04,
                        0.04,
                    ),
                ],
                dtype=np.float64,
            )
            self._desired_base_position[:] = (
                self._home_base_position
                + self._home_base_rotation @ local_base_shift
            )
            self._desired_base_rotation[:] = self._home_base_rotation

        base_position = self.data.xpos[self._base_body_id]
        planar_base_error = self._desired_base_position[:2] - base_position[:2]
        self._base_goal_distance = float(np.linalg.norm(planar_base_error))
        current_yaw = float(np.arctan2(base_rotation[1, 0], base_rotation[0, 0]))
        desired_yaw = float(
            np.arctan2(
                self._desired_base_rotation[1, 0],
                self._desired_base_rotation[0, 0],
            )
        )
        final_yaw_error = self._wrap_angle(desired_yaw - current_yaw)

        if self.auto_drive:
            if self._mobile_base_active:
                self._mobile_base_active = bool(
                    self._base_goal_distance > 0.090
                    or abs(final_yaw_error) > 0.015
                )
            else:
                self._mobile_base_active = bool(
                    target_requires_driving
                    and (
                        self._base_goal_distance > 0.12
                        or abs(final_yaw_error) > 0.04
                    )
                )
        else:
            self._mobile_base_active = False

        if self._mobile_base_active:
            desired_base_twist = self._mobile_base_twist(
                planar_base_error,
                final_yaw_error,
                forward,
                base_rotation,
            )
            desired_base_twist[:3] *= self._speed_scale
            desired_base_twist[3:] *= self._rotation_speed_scale
            desired_base_twist -= BASE_VEL_DAMPING * self.data.qvel[:6]
            target_from_home_current = (
                base_rotation.T @ (self._target_position - base_position)
                - self._home_tcp_position_base
            )
            reachable_delta = np.clip(
                target_from_home_current,
                np.array([-0.12, -0.10, -0.14], dtype=np.float64),
                np.array([0.16, 0.10, 0.14], dtype=np.float64),
            )
            control_tcp_position = base_position + base_rotation @ (
                self._home_tcp_position_base + reachable_delta
            )
            carry_rotation = (
                base_rotation @ self._home_tcp_rotation_base
            )
            orientation_delta = rotation_vector(
                self._target_rotation @ carry_rotation.T
            )
            orientation_blend = float(
                np.clip(1.0 - self._base_goal_distance / 0.20, 0.0, 1.0)
            )
            orientation_angle = float(np.linalg.norm(orientation_delta))
            if orientation_angle > 1.0e-9:
                control_tcp_rotation = axis_angle_rotation(
                    orientation_delta / orientation_angle,
                    orientation_blend * orientation_angle,
                ) @ carry_rotation
            else:
                control_tcp_rotation = carry_rotation
        else:
            desired_base_twist = np.concatenate(
                [
                    np.clip(
                        2.0 * (self._desired_base_position - base_position),
                        -0.10,
                        0.10,
                    ),
                    np.clip(
                        2.0
                        * rotation_vector(
                            self._desired_base_rotation @ base_rotation.T
                        ),
                        -0.20,
                        0.20,
                    ),
                ]
            )
            desired_base_twist[:3] *= self._speed_scale
            desired_base_twist[3:] *= self._rotation_speed_scale
            desired_base_twist -= BASE_VEL_DAMPING * self.data.qvel[:6]
            control_tcp_position = self._target_position
            control_tcp_rotation = self._target_rotation
        task_jacobians.append(
            self._body_jacobian(self._base_body_id)[:, self._optimized_dofs]
        )
        task_velocities.append(desired_base_twist)
        base_task_weights = (
            np.array([30.0, 30.0, 18.0, 20.0, 20.0, 30.0])
            if self._mobile_base_active
            else np.array([6.0, 6.0, 12.0, 15.0, 15.0, 8.0])
        )
        task_weights.append(base_task_weights)

        tcp_rotation = self.tcp_rotation
        # Task-space PD: brake with the measured TCP velocity so the body
        # does not oscillate through the target.
        tcp_velocity = self._site_jacobian(self._tcp_site_id) @ self.data.qvel
        desired_tcp_twist = np.concatenate(
            [
                np.clip(
                    TCP_POS_GAIN * (control_tcp_position - self.tcp_position),
                    -0.30,
                    0.30,
                ),
                np.clip(
                    TCP_ORI_GAIN
                    * rotation_vector(control_tcp_rotation @ tcp_rotation.T),
                    -0.60,
                    0.60,
                ),
            ]
        )
        desired_tcp_twist[:3] *= self._speed_scale
        desired_tcp_twist[3:] *= self._rotation_speed_scale
        desired_tcp_twist -= TCP_VEL_DAMPING * tcp_velocity
        task_jacobians.append(
            self._site_jacobian(self._tcp_site_id)[:, self._optimized_dofs]
        )
        task_velocities.append(desired_tcp_twist)
        task_weights.append(
            np.array([18.0, 18.0, 18.0, 9.0, 9.0, 9.0], dtype=np.float64)
        )

        posture_jacobian = np.zeros((18, 28), dtype=np.float64)
        posture_jacobian[:12, 6:18] = np.eye(12)
        posture_jacobian[12:, 22:] = np.eye(6)
        current_posture = np.concatenate(
            [
                self.data.qpos[self._leg_qpos_adr],
                self.data.qpos[self._arm_qpos_adr],
            ]
        )
        task_jacobians.append(posture_jacobian)
        posture_velocity = 0.3 * (
            np.concatenate([WBC_STAND_LEG_NOMINAL, ARM_NOMINAL])
            - current_posture
        )
        if self._mobile_base_active:
            posture_velocity[:12] = 0.0
            posture_weights = np.concatenate(
                [np.full(12, 12.0), np.full(6, 0.2)]
            )
        else:
            posture_weights = np.full(18, 0.2, dtype=np.float64)
        task_velocities.append(posture_velocity)
        task_weights.append(posture_weights)

        jacobian = np.vstack(task_jacobians)
        velocity = np.concatenate(task_velocities)
        weights = np.concatenate(task_weights)
        weighted_jacobian = weights[:, None] * jacobian
        weighted_velocity = weights * velocity
        normal_matrix = (
            weighted_jacobian.T @ weighted_jacobian
            + 3.0e-2 * np.eye(28)
        )
        generalized_velocity = np.linalg.solve(
            normal_matrix,
            weighted_jacobian.T @ weighted_velocity,
        )
        lower = np.array(
            [-0.40] * 3
            + [-1.0] * 3
            + [-0.35] * 12
            + [-8.0] * 4
            + [-1.0] * 6,
            dtype=np.float64,
        )
        lower[:3] *= self._speed_scale
        lower[3:6] *= self._rotation_speed_scale
        lower[6:] *= self._speed_scale
        return np.clip(generalized_velocity, lower, -lower)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _mobile_base_twist(
        self,
        planar_error: FloatArray,
        final_yaw_error: float,
        forward: FloatArray,
        base_rotation: FloatArray,
    ) -> FloatArray:
        distance = float(np.linalg.norm(planar_error))
        desired_forward = self._desired_base_rotation[:2, 0]
        desired_lateral = np.array(
            [-desired_forward[1], desired_forward[0]],
            dtype=np.float64,
        )
        longitudinal_error = float(desired_forward @ planar_error)
        lateral_error = float(desired_lateral @ planar_error)
        if abs(final_yaw_error) > 0.12:
            angular_velocity = float(
                np.clip(3.0 * final_yaw_error, -0.6, 0.6)
            )
            linear_speed = 0.0
        elif distance > 0.080:
            linear_speed = float(
                np.clip(0.9 * longitudinal_error, -0.35, 0.35)
            )
            direction = 1.0 if linear_speed >= 0.0 else -1.0
            cross_track = float(
                np.arctan2(lateral_error, max(abs(longitudinal_error), 0.10))
            )
            angular_velocity = float(
                np.clip(
                    3.0 * (final_yaw_error + 0.6 * direction * cross_track),
                    -0.6,
                    0.6,
                )
            )
        else:
            angular_velocity = float(np.clip(4.0 * final_yaw_error, -0.5, 0.5))
            if abs(final_yaw_error) > 0.015:
                angular_velocity = float(
                    np.copysign(max(abs(angular_velocity), 0.25), final_yaw_error)
                )
            linear_speed = 0.0

        linear_velocity = linear_speed * forward
        linear_velocity[2] = float(
            np.clip(
                2.0
                * (
                    self._desired_base_position[2]
                    - self.data.xpos[self._base_body_id, 2]
                ),
                -0.10,
                0.10,
            )
        )
        base_orientation_error = rotation_vector(
            self._desired_base_rotation @ base_rotation.T
        )
        angular = np.array(
            [
                np.clip(2.0 * base_orientation_error[0], -0.20, 0.20),
                np.clip(2.0 * base_orientation_error[1], -0.20, 0.20),
                angular_velocity,
            ],
            dtype=np.float64,
        )
        return np.concatenate([linear_velocity, angular])

    def _update_joint_reference(self) -> None:
        generalized_velocity = self._solve_generalized_velocity()
        actuated_velocity = np.concatenate(
            [generalized_velocity[6:18], generalized_velocity[22:]]
        )
        self._joint_reference += self.control_dt * actuated_velocity
        self._wheel_velocity_target[:] = generalized_velocity[18:22]
        self._joint_reference[:12] = np.clip(
            self._joint_reference[:12],
            WBC_STAND_LEG_NOMINAL - 0.22,
            WBC_STAND_LEG_NOMINAL + 0.22,
        )
        self._joint_reference[12:] = np.clip(
            self._joint_reference[12:],
            self._arm_joint_ranges[:, 0],
            self._arm_joint_ranges[:, 1],
        )

        actual = np.concatenate(
            [
                self.data.qpos[self._leg_qpos_adr],
                self.data.qpos[self._arm_qpos_adr],
            ]
        )
        windup_limit = np.concatenate(
            [np.full(12, 0.18), np.full(6, 0.70)]
        )
        self._joint_reference[:] = np.clip(
            self._joint_reference,
            actual - windup_limit,
            actual + windup_limit,
        )

    def _compute_torque(self) -> FloatArray:
        if self._active:
            self._update_joint_reference()

        leg_position = self.data.qpos[self._leg_qpos_adr]
        leg_velocity = self.data.qvel[self._leg_dof_adr]
        wheel_velocity = self.data.qvel[self._wheel_dof_adr]
        arm_position = self.data.qpos[self._arm_qpos_adr]
        arm_velocity = self.data.qvel[self._arm_dof_adr]
        gripper_position = float(self.data.qpos[self._gripper_qpos_adr])
        gripper_velocity = float(self.data.qvel[self._gripper_dof_adr])

        torque = np.zeros(23, dtype=np.float64)
        torque[:12] = (
            2.0 * LEG_KP * (self._joint_reference[:12] - leg_position)
            - 6.0 * LEG_KD * leg_velocity
        )
        torque[12:16] = 5.0 * (
            self._wheel_velocity_target - wheel_velocity
        )
        torque[16:22] = (
            ARM_KP_SCALE * ARM_KP * (self._joint_reference[12:] - arm_position)
            - ARM_KD_SCALE * ARM_KD * arm_velocity
            + self.data.qfrc_bias[self._arm_dof_adr]
        )
        torque[22] = (
            GRIPPER_KP * (self._gripper_target - gripper_position)
            - GRIPPER_KD * gripper_velocity
        )
        return np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)

    def step(self) -> WBCDiagnostics:
        """Run one WBC update and advance MuJoCo by one control period."""
        self._last_torque[:] = self._compute_torque()
        self.data.ctrl[self._actuator_ids] = self._last_torque
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._update_target_marker()
        return self.diagnostics()

    def diagnostics(self) -> WBCDiagnostics:
        base_rotation = self.data.xmat[self._base_body_id].reshape(3, 3)
        return WBCDiagnostics(
            position_error=float(
                np.linalg.norm(self._target_position - self.tcp_position)
            ),
            orientation_error=float(
                np.linalg.norm(
                    rotation_vector(self._target_rotation @ self.tcp_rotation.T)
                )
            ),
            base_position_error=float(
                np.linalg.norm(
                    self._desired_base_position
                    - self.data.xpos[self._base_body_id]
                )
            ),
            upright=float(base_rotation[2, 2]),
            max_torque=float(np.max(np.abs(self._last_torque))),
            mobile_base_active=self._mobile_base_active,
            base_goal_distance=self._base_goal_distance,
        )
