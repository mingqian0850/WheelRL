"""MuJoCo Jacobian whole-body controller for a stationary B2-W + Z1."""

from __future__ import annotations

from collections import deque
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
# The command is considered to be in the terminal region below these errors.
# We still command velocity damping there; only the proportional term enters a
# dead band.  This removes the tiny limit cycle that otherwise remains after
# the TCP reaches a target.
TCP_POSITION_DEADBAND = 1.0e-3
TCP_ORIENTATION_DEADBAND = np.deg2rad(0.15)
TCP_TERMINAL_POSITION = 0.06
TCP_TERMINAL_ORIENTATION = np.deg2rad(8.0)
TCP_TERMINAL_DAMPING = 1.05

# Comfortable TCP displacement from the captured arm pose, expressed in the
# current base frame.  Outside this region the base participates continuously
# instead of waiting for the arm servo to sit at a joint limit for 0.5 s.
ARM_COMFORT_DELTA_LOW = np.array([-0.12, -0.10, -0.14], dtype=np.float64)
ARM_COMFORT_DELTA_HIGH = np.array([0.16, 0.10, 0.14], dtype=np.float64)
BASE_PARTICIPATION_START = 0.82
BASE_PARTICIPATION_FULL = 1.12
# Vertical TCP tracking belongs to Z1.  B2-W may trim ride height slightly,
# but it must never keep crouching to chase a target below the arm workspace.
BASE_HEIGHT_ASSIST_LOW = -0.015
BASE_HEIGHT_ASSIST_HIGH = 0.030
# Low-reach posture: pitch the front of the body down while keeping all four
# wheels planted. The arm target is lowered only as this posture becomes
# available, preventing Z1 from folding through the top deck during entry.
LOW_TILT_START_DROP = 0.30
LOW_TILT_FULL_DROP = 0.48
LOW_TILT_MAX = np.deg2rad(14.0)
LOW_TILT_RATE = np.deg2rad(18.0)  # rad/s, smooth ~0.8 s full transition
LOW_TILT_FRONT_X = 0.20
LOW_TILT_SAFE_DROP_LEVEL = 0.38
LOW_TILT_SAFE_DROP_FULL = 0.48
LOW_TILT_DRIVE_SCALE = 0.45
LOW_TILT_HOLD_ENTER_ERROR = 0.030
LOW_TILT_HOLD_EXIT_ERROR = 0.070
# Arm servo gains are boosted locally (the RL envs keep their own tuning).
ARM_KP_SCALE = 1.5
ARM_KD_SCALE = 2.0
# Reverse mode: when the target lies behind the base (more than REVERSE_ANGLE
# from the heading), drive the base backward instead of turning around. The
# desired base parks the target at this base-frame x so the arm reaches it
# over its shoulder (measured arm workspace x extends to about -0.65 m).
REVERSE_ANGLE = 2.0  # rad (~115 deg) from the current heading: enter reverse
REVERSE_EXIT_ANGLE = 1.5  # rad (~86 deg): hysteresis exit for reverse mode
# Parking x: the target is parked just behind the base. Deep backward reach
# (x < -0.15) sits at the arm's joint-limit boundary, so the robot backs up
# nearly all the way and the arm does a short over-the-shoulder reach.
REVERSE_TARGET_X = -0.05  # m, base-frame x of the target when parked
REVERSE_REACHABLE_DELTA = np.array([-0.80, -0.10, -0.14], dtype=np.float64)
REVERSE_REACHABLE_DELTA_HIGH = np.array([-0.05, 0.10, 0.14], dtype=np.float64)

# Arm-first (macro-micro) mode: the 6-DOF arm serves the TCP alone while the
# base holds still; the base drives only when the arm demonstrably cannot
# reach the target. Reachability is decided by the arm-only servo error, so
# joint limits and workspace shape are handled automatically (the Z1 workspace
# is highly asymmetric and not a box in base coordinates).
ARM_FIRST_EXIT_ERROR = 0.08  # m: arm-only TCP error that means "unreachable"
ARM_FIRST_IMMEDIATE_EXIT_ERROR = 0.06  # m: clear new planar target command
ARM_FIRST_EXIT_STEPS = 50  # 0.5 s at 100 Hz before handing off to the base
ARM_FIRST_ENTER_ERROR = 0.015  # m: arm-only IK achievable error to re-engage
ARM_FIRST_ENTER_ORI = 0.02  # rad: arm-only IK achievable orientation error
ARM_FIRST_ENTER_STEPS = 20  # consecutive good steps before re-engaging
ARM_FIRST_BASE_SETTLE = 0.03  # m: max planar base error at re-engage
ARM_FIRST_BASE_SETTLE_YAW = 0.02  # rad: max yaw error at re-engage
# The Z1's fourth joint is locked in this model, so the arm has five
# effective dofs and cannot serve arbitrary SE(3): a target position is
# sometimes reachable only with the wrong orientation. Detect that stall
# (arm at the position, orientation stuck) and hand off to the base, whose
# yaw can help satisfy the orientation.
ARM_FIRST_STALL_POS = 0.05  # m: arm is at the target position...
ARM_FIRST_STALL_ORI = 0.12  # rad: ...but the orientation is stuck


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    """Return a C1-continuous transition from zero to one."""
    if edge1 <= edge0:
        raise ValueError("edge1 must be greater than edge0")
    scaled = float(np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0))
    return scaled * scaled * (3.0 - 2.0 * scaled)


def clip_vector_norm(vector: FloatArray, maximum: float) -> FloatArray:
    """Limit a vector magnitude without changing its direction."""
    result = np.asarray(vector, dtype=np.float64).copy()
    norm = float(np.linalg.norm(result))
    if norm > maximum > 0.0:
        result *= maximum / norm
    return result


def solve_box_least_squares(
    matrix: FloatArray,
    target: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    *,
    damping: float = 0.0,
    max_iterations: int | None = None,
) -> FloatArray:
    """Solve a small damped least-squares problem with box constraints.

    This is a bounded-variable active-set solver for

    ``min 0.5 ||A x - b||^2 + 0.5 * damping * ||x||^2``.

    The WBC has only 28 variables, so solving the free normal equations and
    updating the active set is both faster and more predictable than adding a
    heavyweight optimization dependency.  Unlike solving and clipping, the
    returned free variables are re-optimized after a joint reaches a bound.
    """
    a = np.asarray(matrix, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if a.ndim != 2 or b.shape != (a.shape[0],):
        raise ValueError("matrix and target shapes are incompatible")
    if lo.shape != (a.shape[1],) or hi.shape != (a.shape[1],):
        raise ValueError("bounds must match the number of variables")
    if np.any(lo > hi):
        raise ValueError("lower bounds must not exceed upper bounds")
    if damping < 0.0:
        raise ValueError("damping must be non-negative")

    variable_count = a.shape[1]
    iterations = max_iterations or max(4 * variable_count, 1)
    active = np.zeros(variable_count, dtype=np.int8)  # -1 lower, +1 upper
    solution = np.clip(np.zeros(variable_count, dtype=np.float64), lo, hi)
    tolerance = 1.0e-9

    for _ in range(iterations):
        free = active == 0
        fixed = ~free
        solution[active < 0] = lo[active < 0]
        solution[active > 0] = hi[active > 0]

        if np.any(free):
            residual_target = b.copy()
            if np.any(fixed):
                residual_target -= a[:, fixed] @ solution[fixed]
            free_matrix = a[:, free]
            normal = free_matrix.T @ free_matrix
            if damping:
                normal += damping * np.eye(normal.shape[0])
            right_hand_side = free_matrix.T @ residual_target
            try:
                candidate = np.linalg.solve(normal, right_hand_side)
            except np.linalg.LinAlgError:
                candidate = np.linalg.lstsq(normal, right_hand_side, rcond=None)[0]

            free_indices = np.flatnonzero(free)
            below = candidate < lo[free] - tolerance
            above = candidate > hi[free] + tolerance
            solution[free] = np.clip(candidate, lo[free], hi[free])
            if np.any(below) or np.any(above):
                active[free_indices[below]] = -1
                active[free_indices[above]] = 1
                continue

        gradient = a.T @ (a @ solution - b) + damping * solution
        lower_release = np.flatnonzero((active < 0) & (gradient < -tolerance))
        upper_release = np.flatnonzero((active > 0) & (gradient > tolerance))
        if lower_release.size == 0 and upper_release.size == 0:
            break
        candidates = np.concatenate([lower_release, upper_release])
        release = int(candidates[np.argmax(np.abs(gradient[candidates]))])
        active[release] = 0

    return np.clip(solution, lo, hi)


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
    arm_only: bool = False
    coordinator_gate: float = 0.0
    solver_residual: float = 0.0
    base_participation: float = 0.0
    tcp_speed: float = 0.0
    vertical_limited: bool = False
    tilt_assist_active: bool = False
    tilt_angle: float = 0.0


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
        reverse_mode: bool = True,
        speed_profile: str = "normal",
        arm_first: bool = True,
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
        self.reverse_mode = reverse_mode
        self._reversing = False  # hysteresis state, see _solve_generalized_velocity
        # Macro-micro decoupling: when the arm can serve the target alone the
        # base is locked (precision); otherwise the base drives (coverage).
        self.arm_first = bool(arm_first)
        self.arm_only_active = False
        self._arm_exit_counter = 0
        self._arm_enter_counter = 0
        # Arm stall recovery: blend the reference back to the canonical
        # config, from which the servo is known to converge.
        self._arm_stall_counter = 0
        self._replan_target: FloatArray | None = None
        self._replan_folded = False
        self._arm_error_window: deque[float] = deque(maxlen=30)
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
        # Optional (pose, rotation) override for the displayed target marker.
        self._marker_pose: tuple[FloatArray, FloatArray] | None = None

        self._joint_reference = np.concatenate(
            [WBC_STAND_LEG_NOMINAL, ARM_NOMINAL]
        ).copy()
        self._gripper_target = float(GRIPPER_NOMINAL[0])
        self._active = False
        self._reversing = False
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
        # Optional learned macro correction. It acts only on the floating-base
        # task; the user supplied TCP target is never modified.  The command
        # is [forward velocity (m/s), yaw rate (rad/s), base-height offset
        # (m)].  A smooth analytic reachability gate decides whether it may
        # participate at all.
        self._coordinator_command: FloatArray | None = None
        self._coordinator_gate = 0.0
        self._actuator_gain_scale = 1.0
        self._solver_residual = 0.0
        self._base_participation = 0.0
        self._tcp_speed = 0.0
        self._tilt_angle = 0.0
        self._tilt_fraction = 0.0
        self._tilt_recovering = False
        self._tilt_planar_hold = False
        self._tilt_hold_base_position = np.zeros(3, dtype=np.float64)
        self._tilt_hold_heading_rotation = np.eye(3, dtype=np.float64)
        self._vertical_target_clamped = False

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
        self.arm_only_active = False
        self._arm_exit_counter = 0
        self._arm_enter_counter = 0
        self._arm_stall_counter = 0
        self._replan_target = None
        self._replan_folded = False
        self._arm_error_window.clear()
        self._wheel_velocity_target.fill(0.0)
        self.clear_base_coordinator_command()
        self._solver_residual = 0.0
        self._base_participation = 0.0
        self._tcp_speed = 0.0
        self._tilt_angle = 0.0
        self._tilt_fraction = 0.0
        self._tilt_recovering = False
        self._tilt_planar_hold = False
        self._tilt_hold_base_position.fill(0.0)
        self._tilt_hold_heading_rotation[:] = np.eye(3)
        self._vertical_target_clamped = False
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
        # The captured pose is trivially arm-reachable.  Start macro/micro
        # control in its stable precision state; a later planar-unreachable
        # command releases the base immediately through the participation
        # gate.  Without this initialization, a low first command spends its
        # first updates in full-body mode and can translate the base before
        # arm reachability has even been evaluated.
        self.arm_only_active = self.arm_first
        self._mobile_base_active = False
        self._arm_exit_counter = 0
        self._arm_enter_counter = 0
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

    def set_base_coordinator_command(
        self,
        command: FloatArray,
        *,
        gate: float,
    ) -> None:
        """Set a learned base command without changing the TCP pose target.

        ``gate`` is supplied by an analytic reachability check. A zero gate
        disables the learned command completely and restores normal arm-first
        WBC behavior, which prevents a policy from degrading easy local
        manipulation targets.
        """
        command_array = np.asarray(command, dtype=np.float64)
        if command_array.shape != (3,) or not np.isfinite(command_array).all():
            raise ValueError("coordinator command must be a finite (3,) array")
        self._coordinator_command = np.clip(
            command_array,
            np.array([-0.45, -0.80, -0.08], dtype=np.float64),
            np.array([0.45, 0.80, 0.08], dtype=np.float64),
        )
        self._coordinator_gate = float(np.clip(gate, 0.0, 1.0))

    def clear_base_coordinator_command(self) -> None:
        """Return base planning entirely to the analytic WBC."""
        self._coordinator_command = None
        self._coordinator_gate = 0.0

    def set_actuator_gain_scale(self, scale: float) -> None:
        """Scale joint servo gains for dynamics randomization experiments."""
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("actuator gain scale must be finite and positive")
        self._actuator_gain_scale = float(np.clip(scale, 0.70, 1.30))

    def _update_target_marker(self) -> None:
        if self._target_mocap_id < 0:
            return
        # _marker_pose overrides the displayed pose for compatible wrappers.
        if self._marker_pose is not None:
            position, rotation = self._marker_pose
        else:
            position, rotation = self._target_position, self._target_rotation
        self.data.mocap_pos[self._target_mocap_id] = position
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, np.asarray(rotation, dtype=float).ravel())
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

    def _arm_ik_from(
        self,
        q0: FloatArray,
        position: FloatArray,
        rotation: FloatArray,
        iterations: int,
    ) -> tuple[FloatArray, float, float]:
        """Run arm-only DLS IK from an initial qpos; return (q, pos, ori)."""
        saved_qpos = self.data.qpos.copy()
        q = q0.copy()
        jacobian = np.zeros((6, self.model.nv))
        for _ in range(iterations):
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_jacSite(
                self.model, self.data, jacobian[:3], jacobian[3:], self._tcp_site_id
            )
            arm_jacobian = jacobian[:, self._arm_dof_adr]
            pos_error = position - self.data.site_xpos[self._tcp_site_id]
            current_rotation = self.data.site_xmat[self._tcp_site_id].reshape(3, 3)
            ori_error = rotation_vector(rotation @ current_rotation.T)
            error = np.concatenate([pos_error, ori_error])
            velocity = np.linalg.solve(
                arm_jacobian.T @ arm_jacobian + 3.0e-2 * np.eye(6),
                arm_jacobian.T @ error,
            )
            q[self._arm_qpos_adr] += np.clip(velocity, -0.5, 0.5) * 0.5
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        pos_error = float(
            np.linalg.norm(position - self.data.site_xpos[self._tcp_site_id])
        )
        ori_error = float(
            np.linalg.norm(rotation_vector(rotation @ self.tcp_rotation.T))
        )
        self.data.qpos[:] = saved_qpos
        mujoco.mj_forward(self.model, self.data)
        return q, pos_error, ori_error

    def _arm_can_reach(
        self,
        position: FloatArray | None = None,
        rotation: FloatArray | None = None,
        tolerance: float = ARM_FIRST_ENTER_ERROR,
    ) -> bool:
        """Cheap arm-only DLS IK feasibility from the current arm pose.

        The Z1 workspace is joint-limit-shaped (not a box in base
        coordinates), so reachability is decided by actually trying the IK.
        """
        if position is None:
            position = self._target_position
        if rotation is None:
            rotation = self._target_rotation
        _, pos_error, ori_error = self._arm_ik_from(
            self.data.qpos.copy(), position, rotation, 12
        )
        return bool(pos_error < tolerance and ori_error < ARM_FIRST_ENTER_ORI)

    def _base_participation_score(
        self,
        base_position: FloatArray,
        base_rotation: FloatArray,
    ) -> float:
        """Return a smooth measure of how much the base should help the arm.

        The primary signal is the target's utilization of the comfortable arm
        workspace.  Joint-limit and manipulability risks add smaller signals
        only while a meaningful TCP error remains.  Consequently the base is
        quiet for precise local tracking but joins immediately for an
        unreachable or awkward target.
        """
        target_in_base = base_rotation.T @ (self._target_position - base_position)
        delta = target_in_base - self._home_tcp_position_base
        center = 0.5 * (ARM_COMFORT_DELTA_LOW + ARM_COMFORT_DELTA_HIGH)
        half_width = 0.5 * (ARM_COMFORT_DELTA_HIGH - ARM_COMFORT_DELTA_LOW)
        # Driving can improve x/y reach, but cannot solve a pure z error.  Do
        # not interpret a low TCP target as a request to move or crouch B2-W.
        planar_utilization = float(
            np.max(np.abs((delta[:2] - center[:2]) / half_width[:2]))
        )
        workspace_need = smoothstep(
            BASE_PARTICIPATION_START,
            BASE_PARTICIPATION_FULL,
            planar_utilization,
        )
        planar_opportunity = smoothstep(
            0.04,
            0.14,
            float(np.linalg.norm(delta[:2])),
        )

        arm_position = self.data.qpos[self._arm_qpos_adr]
        arm_width = np.maximum(
            self._arm_joint_ranges[:, 1] - self._arm_joint_ranges[:, 0],
            1.0e-6,
        )
        normalized_margin = np.minimum(
            arm_position - self._arm_joint_ranges[:, 0],
            self._arm_joint_ranges[:, 1] - arm_position,
        ) / arm_width
        joint_need = 1.0 - smoothstep(
            0.04,
            0.18,
            float(np.min(normalized_margin)),
        )

        arm_jacobian = self._site_jacobian(self._tcp_site_id)[
            :3, self._arm_dof_adr
        ]
        smallest_singular = float(
            np.min(np.linalg.svd(arm_jacobian, compute_uv=False))
        )
        singular_need = 1.0 - smoothstep(0.04, 0.12, smallest_singular)
        error_need = smoothstep(
            0.015,
            0.08,
            float(np.linalg.norm(self._target_position - self.tcp_position)),
        )
        return float(
            np.clip(
                max(
                    workspace_need,
                    0.65 * planar_opportunity * error_need * joint_need,
                    0.45 * planar_opportunity * error_need * singular_need,
                ),
                0.0,
                1.0,
            )
        )

    def _generalized_velocity_bounds(self) -> tuple[FloatArray, FloatArray]:
        """Build speed and one-step joint-position bounds for the WBC QP."""
        lower = np.array(
            [-0.40] * 3
            + [-1.0] * 3
            + [-0.35] * 12
            + [-8.0] * 4
            + [-1.0] * 6,
            dtype=np.float64,
        )
        upper = -lower
        lower[:3] *= self._speed_scale
        upper[:3] *= self._speed_scale
        lower[3:6] *= self._rotation_speed_scale
        upper[3:6] *= self._rotation_speed_scale
        lower[6:] *= self._speed_scale
        upper[6:] *= self._speed_scale

        leg_lower = WBC_STAND_LEG_NOMINAL - 0.22
        leg_upper = WBC_STAND_LEG_NOMINAL + 0.22
        lower[6:18] = np.maximum(
            lower[6:18],
            (leg_lower - self._joint_reference[:12]) / self.control_dt,
        )
        upper[6:18] = np.minimum(
            upper[6:18],
            (leg_upper - self._joint_reference[:12]) / self.control_dt,
        )

        joint_margin = 1.0e-3
        arm_lower = self._arm_joint_ranges[:, 0] + joint_margin
        arm_upper = self._arm_joint_ranges[:, 1] - joint_margin
        lower[22:] = np.maximum(
            lower[22:],
            (arm_lower - self._joint_reference[12:]) / self.control_dt,
        )
        upper[22:] = np.minimum(
            upper[22:],
            (arm_upper - self._joint_reference[12:]) / self.control_dt,
        )
        return lower, upper

    def _desired_tcp_twist(
        self,
        control_position: FloatArray,
        control_rotation: FloatArray,
    ) -> FloatArray:
        """Generate a position-priority, terminally damped SE(3) command."""
        position_error = control_position - self.tcp_position
        position_error_norm = float(np.linalg.norm(position_error))
        if position_error_norm < TCP_POSITION_DEADBAND:
            position_error = np.zeros(3, dtype=np.float64)

        orientation_error = rotation_vector(
            control_rotation @ self.tcp_rotation.T
        )
        orientation_error_norm = float(np.linalg.norm(orientation_error))
        if orientation_error_norm < TCP_ORIENTATION_DEADBAND:
            orientation_error = np.zeros(3, dtype=np.float64)

        # Position dominates while the TCP is far away.  Full orientation
        # authority returns smoothly in the terminal region.
        orientation_priority = 1.0 - 0.75 * smoothstep(
            0.05,
            0.25,
            position_error_norm,
        )
        linear = clip_vector_norm(TCP_POS_GAIN * position_error, 0.30)
        angular = orientation_priority * clip_vector_norm(
            TCP_ORI_GAIN * orientation_error,
            0.60,
        )
        linear *= self._speed_scale
        angular *= self._rotation_speed_scale

        tcp_velocity = self._site_jacobian(self._tcp_site_id) @ self.data.qvel
        self._tcp_speed = float(np.linalg.norm(tcp_velocity[:3]))
        near_position = 1.0 - smoothstep(
            0.01,
            TCP_TERMINAL_POSITION,
            position_error_norm,
        )
        near_orientation = 1.0 - smoothstep(
            np.deg2rad(1.0),
            TCP_TERMINAL_ORIENTATION,
            orientation_error_norm,
        )
        linear_damping = TCP_VEL_DAMPING + near_position * (
            TCP_TERMINAL_DAMPING - TCP_VEL_DAMPING
        )
        angular_damping = TCP_VEL_DAMPING + near_orientation * (
            TCP_TERMINAL_DAMPING - TCP_VEL_DAMPING
        )
        linear -= linear_damping * tcp_velocity[:3]
        angular -= angular_damping * tcp_velocity[3:]
        return np.concatenate(
            [
                clip_vector_norm(linear, 0.45 * self._speed_scale),
                clip_vector_norm(angular, 0.90 * self._rotation_speed_scale),
            ]
        )

    def _update_low_tilt_assist(
        self,
        target_delta_home: FloatArray,
        base_position: FloatArray,
        base_rotation: FloatArray,
    ) -> None:
        """Smoothly request a nose-down posture for a low target in front.

        Tilting is deliberately separate from ``mobile_base_active``: wheels
        remain at their planar location while front/rear leg extension creates
        pitch. Far planar targets are served first, then the tilt fades in as
        the target enters the local arm workspace.
        """
        target_in_base = base_rotation.T @ (self._target_position - base_position)
        target_drop = max(0.0, -float(target_delta_home[2]))
        depth_fraction = smoothstep(
            LOW_TILT_START_DROP,
            LOW_TILT_FULL_DROP,
            target_drop,
        )
        front_fraction = smoothstep(
            LOW_TILT_FRONT_X - 0.08,
            LOW_TILT_FRONT_X + 0.08,
            float(target_in_base[0]),
        )
        lateral_offset = abs(
            float(target_in_base[1] - self._home_tcp_position_base[1])
        )
        lateral_fraction = 1.0 - smoothstep(0.18, 0.35, lateral_offset)
        # Enter tilt only after an initially far target has been brought near.
        # Once the low posture exists, latch it while the target remains low
        # and in front so forward jogging does not make the body stand up and
        # lift the TCP during every base relocation.
        if (
            self._tilt_recovering
            and depth_fraction > 0.0
            and front_fraction > 0.5
            and lateral_fraction > 0.5
        ):
            local_fraction = 1.0
        else:
            local_fraction = 1.0 - smoothstep(
                0.20,
                0.80,
                self._base_participation,
            )
        requested_angle = (
            LOW_TILT_MAX
            * depth_fraction
            * front_fraction
            * lateral_fraction
            * local_fraction
        )
        max_step = LOW_TILT_RATE * self.control_dt
        self._tilt_angle += float(
            np.clip(requested_angle - self._tilt_angle, -max_step, max_step)
        )
        if abs(self._tilt_angle) < np.deg2rad(0.05):
            self._tilt_angle = 0.0
        self._tilt_fraction = float(
            np.clip(self._tilt_angle / LOW_TILT_MAX, 0.0, 1.0)
        )
        actual_rotation_vector = rotation_vector(
            base_rotation @ self._home_base_rotation.T
        )
        actual_pitch = float(
            actual_rotation_vector @ self._home_base_rotation[:, 1]
        )
        if requested_angle > np.deg2rad(0.10) or self._tilt_angle > 0.0:
            self._tilt_recovering = True
        elif self._tilt_recovering and abs(actual_pitch) < np.deg2rad(0.5):
            self._tilt_recovering = False

    def _safe_low_reach_drop(self) -> float:
        """Return the collision-tested TCP drop released by current tilt."""
        return float(
            LOW_TILT_SAFE_DROP_LEVEL
            + self._tilt_fraction
            * (LOW_TILT_SAFE_DROP_FULL - LOW_TILT_SAFE_DROP_LEVEL)
        )

    def _maybe_replan_arm(self) -> None:
        """Unwedge the arm-only servo from a local minimum.

        The incremental DLS servo can get wedged against a joint limit or in
        a wrong homotopy class after repeated nudges (the Z1 workspace is
        highly non-convex). The WBC servo reliably converges from the
        canonical config (ARM_NOMINAL), so on stall the joint reference is
        blended back there and the servo re-converges, the base stays locked.
        """
        if not (self.arm_first and self.arm_only_active):
            self._arm_stall_counter = 0
            self._replan_target = None
            self._replan_folded = False
            self._arm_error_window.clear()
            return
        if self._replan_target is not None:
            step = np.clip(
                self._replan_target - self._joint_reference[12:], -0.06, 0.06
            )
            self._joint_reference[12:] += step
            if (
                np.linalg.norm(self._replan_target - self._joint_reference[12:])
                < 0.02
            ):
                self._replan_target = None
            return
        pos_error = float(
            np.linalg.norm(self._target_position - self.tcp_position)
        )
        ori_error = float(
            np.linalg.norm(
                rotation_vector(self._target_rotation @ self.tcp_rotation.T)
            )
        )
        self._arm_error_window.append(pos_error)
        if pos_error < 0.03 and ori_error < 0.05:
            self._arm_stall_counter = 0
            self._replan_folded = False  # recovered: a later wedge may fold again
            return
        # A wedge is a *plateau*, not a mere large error: the arm is stuck
        # only if the position error has not improved over the last 0.3 s
        # AND the arm joints are actually idle (a slow step-response tail
        # keeps moving, so it never looks wedged).
        window = self._arm_error_window
        arm_idle = float(np.max(np.abs(self.data.qvel[self._arm_dof_adr]))) < 0.04
        plateau = bool(
            len(window) == window.maxlen
            and pos_error - min(window) < 0.015
            and arm_idle
        )
        if not plateau:
            self._arm_stall_counter = 0
            return
        self._arm_stall_counter += 1
        if self._arm_stall_counter < 25:  # 0.25 s on the plateau before folding
            return
        if self._replan_folded:
            return  # one fold per stint; let the exit hand the task to the base
        if pos_error > ARM_FIRST_EXIT_ERROR or ori_error > 0.12:
            return  # clearly unreachable: the state-machine exit handles it
        self._replan_target = ARM_NOMINAL.copy()
        self._replan_folded = True
        self._arm_stall_counter = 0
        self._arm_exit_counter = 0

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
        self._base_participation = self._base_participation_score(
            self.data.xpos[self._base_body_id],
            base_rotation,
        )

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
            or self._base_participation > 0.05
        )
        home_yaw = float(
            np.arctan2(
                self._home_base_rotation[1, 0],
                self._home_base_rotation[0, 0],
            )
        )
        reversing = self._reversing
        if target_requires_driving and self.auto_drive:
            target_from_home_base = (
                self._target_position[:2] - self._home_base_position[:2]
            )
            # Reverse mode: with the target behind the current heading, back up
            # instead of turning around. Decide from the CURRENT base frame so
            # the choice tracks the robot as it moves.
            base_position_now = self.data.xpos[self._base_body_id]
            base_rotation_now = self.data.xmat[self._base_body_id].reshape(3, 3)
            target_in_base = base_rotation_now.T @ (
                self._target_position - base_position_now
            )
            target_angle = float(np.arctan2(target_in_base[1], target_in_base[0]))
            # Hysteresis: once reversing, stay reversing until the target is
            # clearly in the front half-plane (x > 0.10). The angle is
            # ill-defined near the base center (atan2 noise flips +/-pi), so
            # the exit uses the base-frame x coordinate instead.
            if self.reverse_mode:
                if self._reversing:
                    self._reversing = target_in_base[0] <= 0.10
                else:
                    self._reversing = abs(target_angle) > REVERSE_ANGLE
            else:
                self._reversing = False
            reversing = self._reversing
            home_tcp_heading = float(
                np.arctan2(
                    self._home_tcp_position_base[1],
                    self._home_tcp_position_base[0],
                )
            )
            if reversing:
                # Keep the current heading; park the target at a backward
                # reachable base-frame x so the arm works over its shoulder.
                self._desired_base_rotation[:] = base_rotation_now
                self._desired_base_position[:] = (
                    self._target_position
                    - base_rotation_now
                    @ np.array([REVERSE_TARGET_X, 0.0, 0.0], dtype=np.float64)
                )
                self._desired_base_position[2] = (
                    self._home_base_position[2]
                    + np.clip(
                        self.base_assist * target_delta_home[2],
                        BASE_HEIGHT_ASSIST_LOW,
                        BASE_HEIGHT_ASSIST_HIGH,
                    )
                )
            else:
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
                        BASE_HEIGHT_ASSIST_LOW,
                        BASE_HEIGHT_ASSIST_HIGH,
                    )
                )
        else:
            local_base_shift = np.array(
                [
                    self.base_assist * retained_by_arm[0],
                    self.base_assist * retained_by_arm[1],
                    np.clip(
                        self.base_assist * target_delta_home[2],
                        BASE_HEIGHT_ASSIST_LOW,
                        BASE_HEIGHT_ASSIST_HIGH,
                    ),
                ],
                dtype=np.float64,
            )
            self._desired_base_position[:] = (
                self._home_base_position
                + self._home_base_rotation @ local_base_shift
            )
            self._desired_base_rotation[:] = self._home_base_rotation

        self._update_low_tilt_assist(
            target_delta_home,
            self.data.xpos[self._base_body_id],
            base_rotation,
        )
        tilt_assist_active = self._tilt_recovering
        if tilt_assist_active:
            tcp_position_error = float(
                np.linalg.norm(self._target_position - self.tcp_position)
            )
            target_remains_low = bool(
                -float(target_delta_home[2]) > LOW_TILT_START_DROP
            )
            if self._tilt_planar_hold:
                if (
                    tcp_position_error > LOW_TILT_HOLD_EXIT_ERROR
                    or not target_remains_low
                ):
                    self._tilt_planar_hold = False
            elif (
                target_remains_low
                and tcp_position_error < LOW_TILT_HOLD_ENTER_ERROR
                and self._base_participation < 0.40
            ):
                # The TCP, rather than a redundant nominal parking pose, says
                # when forward relocation is complete. Capture that wheel
                # footprint and heading so the planner cannot keep creeping.
                self._tilt_planar_hold = True
                self._tilt_hold_base_position[:] = self.data.xpos[
                    self._base_body_id
                ]
                current_yaw_for_hold = float(
                    np.arctan2(base_rotation[1, 0], base_rotation[0, 0])
                )
                home_yaw_for_hold = float(
                    np.arctan2(
                        self._home_base_rotation[1, 0],
                        self._home_base_rotation[0, 0],
                    )
                )
                yaw_delta = self._wrap_angle(
                    current_yaw_for_hold - home_yaw_for_hold
                )
                self._tilt_hold_heading_rotation[:] = (
                    axis_angle_rotation(
                        np.array([0.0, 0.0, 1.0], dtype=np.float64),
                        yaw_delta,
                    )
                    @ self._home_base_rotation
                )
            if self._tilt_planar_hold:
                self._desired_base_position[:2] = (
                    self._tilt_hold_base_position[:2]
                )
                self._desired_base_rotation[:] = (
                    self._tilt_hold_heading_rotation
                )
            # Positive rotation about the body's left axis sends its forward
            # axis downward: front legs shorten while rear legs extend.
            tilt_axis_world = self._desired_base_rotation[:, 1].copy()
            self._desired_base_rotation[:] = (
                axis_angle_rotation(tilt_axis_world, self._tilt_angle)
                @ self._desired_base_rotation
            )
            # Rotate around a nominal-height body center rather than obtaining
            # extra reach through an uncontrolled whole-body crouch.
            self._desired_base_position[2] = self._home_base_position[2]
        else:
            self._tilt_planar_hold = False

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

        # ---- arm-first state machine (macro-micro) -----------------------
        coordinator_active = bool(
            not tilt_assist_active
            and self.auto_drive
            and self._coordinator_command is not None
            and self._coordinator_gate > 1.0e-3
        )
        if tilt_assist_active:
            # Keep the wheels at their planar location but unlock the legs so
            # the explicit base-pitch task can form a front-low/rear-high
            # stance. This is posture assistance, not locomotion.
            self.arm_only_active = False
            self._arm_exit_counter = 0
            self._arm_enter_counter = 0
        elif coordinator_active:
            # The analytic gate has declared that the base is needed.  Do not
            # wait for the arm-only plateau detector before accepting the
            # learned macro command.
            self.arm_only_active = False
            self._arm_exit_counter = 0
            self._arm_enter_counter = 0
        elif self.arm_first:
            arm_tcp_position_error = float(
                np.linalg.norm(self._target_position - self.tcp_position)
            )
            tcp_error_base = base_rotation.T @ (
                self._target_position - self.tcp_position
            )
            arm_tcp_planar_error = float(np.linalg.norm(tcp_error_base[:2]))
            vertical_only_residual = bool(
                arm_tcp_planar_error < 0.02
                and abs(float(tcp_error_base[2])) > 0.02
            )
            arm_tcp_orientation_error = float(
                np.linalg.norm(
                    rotation_vector(self._target_rotation @ self.tcp_rotation.T)
                )
            )
            if (
                self.arm_only_active
                and self._base_participation > 0.15
                and arm_tcp_planar_error > ARM_FIRST_IMMEDIATE_EXIT_ERROR
            ):
                # A newly commanded target has left the comfortable arm
                # workspace.  Release the base immediately instead of waiting
                # for an avoidable arm-only tracking plateau. Only planar
                # error can release a wheeled base; a vertical residual stays
                # with the arm and its joint limits.
                self.arm_only_active = False
                self._arm_exit_counter = 0
                self._arm_enter_counter = 0
            elif self.arm_only_active and self._replan_target is None:
                # Leave arm-only when the arm demonstrably cannot serve the
                # target: position plateau at joint limits / outside the
                # workspace, or a stall at the target position with the
                # orientation stuck (the Z1 has a locked joint). Suspended
                # while the stall-recovery fold is in flight.
                base_can_help_position = bool(
                    self._base_participation > 0.15
                    or np.any(np.abs(target_delta_home[:2]) > arm_allowance)
                )
                arm_failed = bool(
                    (
                        arm_tcp_planar_error > ARM_FIRST_EXIT_ERROR
                        and base_can_help_position
                    )
                    or (
                        arm_tcp_position_error < ARM_FIRST_STALL_POS
                        and arm_tcp_orientation_error > ARM_FIRST_STALL_ORI
                    )
                )
                if arm_failed:
                    self._arm_exit_counter += 1
                else:
                    self._arm_exit_counter = 0
                if self._arm_exit_counter >= ARM_FIRST_EXIT_STEPS:
                    self.arm_only_active = False
                    self._arm_exit_counter = 0
            else:
                # Re-engage only once the arm IK can reach the target AND the
                # base has settled, so the hand-off does not chatter. A pure
                # vertical residual is also handed to the arm: driving cannot
                # reduce it, and the arm will stop safely at its joint limit.
                arm_can_take_over = self._arm_can_reach() or vertical_only_residual
                if self._base_participation < 0.05 and arm_can_take_over:
                    self._arm_enter_counter += 1
                else:
                    self._arm_enter_counter = 0
                planned_base_settled = (
                    self._base_goal_distance < ARM_FIRST_BASE_SETTLE
                    and abs(final_yaw_error) < ARM_FIRST_BASE_SETTLE_YAW
                )
                # The nominal parking pose is only a guide, not a task-space
                # requirement.  If the moving base has already brought the
                # target into a comfortable, IK-feasible arm workspace, stop
                # there instead of continuing to rotate toward a redundant
                # parking yaw.  This hand-off is what removes slow orbiting
                # around an already reached TCP target.
                comfortably_reached = bool(
                    self._base_participation < 0.05
                    and (
                        arm_tcp_position_error < 0.02
                        or vertical_only_residual
                    )
                    and arm_tcp_orientation_error < 0.08
                )
                base_settled = planned_base_settled or comfortably_reached
                if (
                    base_settled
                    and self._arm_enter_counter >= ARM_FIRST_ENTER_STEPS
                ):
                    self.arm_only_active = True
                    self._arm_enter_counter = 0
                    self._mobile_base_active = False
        if self.arm_only_active:
            # Lock the base exactly where it is; only the arm serves the TCP.
            self._desired_base_position[:] = base_position
            self._desired_base_rotation[:] = base_rotation

        if coordinator_active:
            self._mobile_base_active = True
        elif self.auto_drive and not self.arm_only_active:
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
            if coordinator_active:
                # Residual-on-analytic is intentionally safer than replacing
                # the planner: a zero policy action is the original WBC, and
                # PPO only learns small corrections when they improve it.
                assert self._coordinator_command is not None
                command = self._coordinator_gate * self._coordinator_command
                desired_base_twist[:3] += command[0] * forward
                desired_base_twist[2] = float(
                    np.clip(
                        2.0
                        * (
                            self._desired_base_position[2]
                            + command[2]
                            - base_position[2]
                        ),
                        -0.10,
                        0.10,
                    )
                )
                desired_base_twist[5] = float(
                    np.clip(desired_base_twist[5] + command[1], -0.80, 0.80)
                )
            # Near the edge of the comfortable workspace the base joins
            # gradually; clearly unreachable targets retain full authority.
            drive_blend = max(
                self._base_participation,
                self._coordinator_gate if coordinator_active else 0.0,
            )
            desired_base_twist[[0, 1, 5]] *= 0.20 + 0.80 * drive_blend
            desired_base_twist[:3] *= self._speed_scale
            desired_base_twist[3:] *= self._rotation_speed_scale
            if tilt_assist_active:
                # Four-wheel driving remains available in the low posture,
                # but is deliberately slower to preserve contact and pitch.
                desired_base_twist[[0, 1, 5]] *= LOW_TILT_DRIVE_SCALE
            desired_base_twist -= BASE_VEL_DAMPING * self.data.qvel[:6]
            target_from_home_current = (
                base_rotation.T @ (self._target_position - base_position)
                - self._home_tcp_position_base
            )
            if reversing:
                # Backward reach: the arm works over its shoulder, so the TCP
                # control point is clamped to the backward workspace.
                reachable_delta = np.clip(
                    target_from_home_current,
                    REVERSE_REACHABLE_DELTA,
                    REVERSE_REACHABLE_DELTA_HIGH,
                )
            else:
                reachable_low = -0.14
                if tilt_assist_active:
                    reachable_low = -self._safe_low_reach_drop()
                reachable_delta = np.clip(
                    target_from_home_current,
                    np.array([-0.12, -0.10, reachable_low], dtype=np.float64),
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

        # Sequence the low reach safely: at level stance the arm may descend
        # only to the collision-tested local floor; more depth is released as
        # the nose-down posture becomes available. The visible/user target is
        # unchanged, so any remaining error is reported honestly.
        safe_vertical_drop = self._safe_low_reach_drop()
        safe_tcp_z = self._home_tcp_position[2] - safe_vertical_drop
        self._vertical_target_clamped = bool(
            control_tcp_position[2] < safe_tcp_z - 1.0e-6
        )
        if self._vertical_target_clamped:
            control_tcp_position = control_tcp_position.copy()
            control_tcp_position[2] = safe_tcp_z
        task_jacobians.append(
            self._body_jacobian(self._base_body_id)[:, self._optimized_dofs]
        )
        task_velocities.append(desired_base_twist)
        base_hold_weights = np.array([6.0, 6.0, 12.0, 15.0, 15.0, 8.0])
        # Once the target is outside the comfortable arm workspace, base
        # motion must win over the arm's tendency to preserve the current
        # posture.  Yaw gets the strongest authority because a wheeled base
        # can only serve a lateral TCP target after turning into it; the arm
        # simultaneously compensates that turn through the SE(3) TCP task.
        base_drive_weights = np.array([42.0, 42.0, 24.0, 24.0, 24.0, 55.0])
        base_weight_blend = (
            self._base_participation if self._mobile_base_active else 0.0
        )
        if coordinator_active:
            base_weight_blend = max(base_weight_blend, self._coordinator_gate)
        base_task_weights = (
            base_hold_weights
            + base_weight_blend * (base_drive_weights - base_hold_weights)
        )
        if tilt_assist_active:
            # Pitch and height are the safety-critical parts of the low-reach
            # posture; x/y/yaw merely hold the current wheel footprint.
            base_task_weights = np.array(
                [28.0, 28.0, 42.0, 42.0, 60.0, 32.0]
            )
        elif self.arm_only_active:
            # Hold the floating base against arm reactions after hand-off.
            # A weak hold lets the wheeled body drift a few centimetres over
            # long runs, which eventually makes an otherwise local target
            # look unreachable again and restarts the mobile-base planner.
            base_task_weights = np.array(
                [45.0, 45.0, 35.0, 35.0, 35.0, 45.0]
            )
        task_weights.append(base_task_weights)

        desired_tcp_twist = self._desired_tcp_twist(
            control_tcp_position,
            control_tcp_rotation,
        )
        tcp_jacobian = self._site_jacobian(self._tcp_site_id)[
            :, self._optimized_dofs
        ].copy()
        arm_columns = np.isin(self._optimized_dofs, self._arm_dof_adr)
        # A wheel base can translate in the ground plane, not vertically.
        # Allocate the TCP-z row exclusively to Z1 so an unreachable low
        # target saturates safely at the arm limit instead of commanding the
        # legs to lower the whole body indefinitely.
        tcp_jacobian[2, ~arm_columns] = 0.0
        if self.arm_only_active:
            # Keep the 28-column layout for the stacked solve; zero out every
            # dof except the arm so the base cannot participate in the servo.
            # (_optimized_dofs is not sorted, so locate the arm columns by
            # matching true dof indices rather than by position.)
            tcp_jacobian[:, ~arm_columns] = 0.0
        task_jacobians.append(tcp_jacobian)
        task_velocities.append(desired_tcp_twist)
        if self.arm_only_active:
            # Full-priority arm-only servo: the base cannot help, so the arm
            # gets the strongest possible position+orientation weights.
            task_weights.append(
                np.array([25.0, 25.0, 25.0, 12.0, 12.0, 12.0], dtype=np.float64)
            )
        elif tilt_assist_active:
            # Low reach is position critical. Orientation remains controlled,
            # but it must not force the elbow to fold back into the body while
            # the legs are creating additional vertical workspace.
            task_weights.append(
                np.array([23.0, 23.0, 27.0, 7.0, 7.0, 7.0], dtype=np.float64)
            )
        elif reversing:
            # Position-first in reverse: folding the arm backward rotates the
            # TCP far from its home orientation, so the orientation task is
            # relaxed to a weak preference and position wins.
            task_weights.append(
                np.array([18.0, 18.0, 18.0, 1.0, 1.0, 1.0], dtype=np.float64)
            )
        else:
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
        arm_width = np.maximum(
            self._arm_joint_ranges[:, 1] - self._arm_joint_ranges[:, 0],
            1.0e-6,
        )
        arm_margin = np.minimum(
            current_posture[12:] - self._arm_joint_ranges[:, 0],
            self._arm_joint_ranges[:, 1] - current_posture[12:],
        ) / arm_width
        arm_limit_weights = 0.2 + 2.8 * np.array(
            [1.0 - smoothstep(0.04, 0.18, float(margin)) for margin in arm_margin],
            dtype=np.float64,
        )
        if tilt_assist_active:
            # The QP already enforces hard predictive joint bounds. During a
            # low reach, soften only the comfort-centering term so Z1 may use
            # the extra workspace created by the tilted body.
            arm_limit_weights = 0.15 + 0.85 * (
                (arm_limit_weights - 0.2) / 2.8
            )
        if self._mobile_base_active:
            posture_velocity[:12] = 0.0
            posture_weights = np.concatenate(
                [np.full(12, 12.0), arm_limit_weights]
            )
        else:
            posture_weights = np.concatenate(
                [np.full(12, 0.2), arm_limit_weights]
            )
        if self.arm_only_active:
            # The arm posture rows would fight the TCP task on the same six
            # dofs; drop them (the TCP task fully determines the arm).
            posture_weights = np.concatenate([np.full(12, 0.2), np.zeros(6)])
        task_velocities.append(posture_velocity)
        task_weights.append(posture_weights)

        jacobian = np.vstack(task_jacobians)
        velocity = np.concatenate(task_velocities)
        weights = np.concatenate(task_weights)
        weighted_jacobian = weights[:, None] * jacobian
        weighted_velocity = weights * velocity
        lower, upper = self._generalized_velocity_bounds()
        generalized_velocity = solve_box_least_squares(
            weighted_jacobian,
            weighted_velocity,
            lower,
            upper,
            damping=3.0e-2,
        )
        self._solver_residual = float(
            np.linalg.norm(weighted_jacobian @ generalized_velocity - weighted_velocity)
            / np.sqrt(max(weighted_velocity.size, 1))
        )
        return generalized_velocity

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
            # Proportional terminal motion replaces the old minimum yaw rate.
            # A fixed 0.25 rad/s kick repeatedly crossed the goal and was a
            # direct source of visible target-adjacent oscillation.
            angular_velocity = float(
                np.clip(3.0 * final_yaw_error, -0.30, 0.30)
            )
            if abs(final_yaw_error) < 0.005:
                angular_velocity = 0.0
            linear_speed = float(
                np.clip(0.7 * longitudinal_error, -0.08, 0.08)
            )
            if distance < 0.015:
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
        # Unwedge the arm from local minima before the torque is computed.
        self._maybe_replan_arm()

        leg_position = self.data.qpos[self._leg_qpos_adr]
        leg_velocity = self.data.qvel[self._leg_dof_adr]
        wheel_velocity = self.data.qvel[self._wheel_dof_adr]
        arm_position = self.data.qpos[self._arm_qpos_adr]
        arm_velocity = self.data.qvel[self._arm_dof_adr]
        gripper_position = float(self.data.qpos[self._gripper_qpos_adr])
        gripper_velocity = float(self.data.qvel[self._gripper_dof_adr])

        torque = np.zeros(23, dtype=np.float64)
        torque[:12] = (
            self._actuator_gain_scale
            * (
                2.0 * LEG_KP * (self._joint_reference[:12] - leg_position)
                - 6.0 * LEG_KD * leg_velocity
            )
        )
        torque[12:16] = self._actuator_gain_scale * 5.0 * (
            self._wheel_velocity_target - wheel_velocity
        )
        torque[16:22] = (
            self._actuator_gain_scale
            * (
                ARM_KP_SCALE
                * ARM_KP
                * (self._joint_reference[12:] - arm_position)
                - ARM_KD_SCALE * ARM_KD * arm_velocity
            )
            + self.data.qfrc_bias[self._arm_dof_adr]
        )
        torque[22] = self._actuator_gain_scale * (
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
        tcp_error_base = base_rotation.T @ (
            self._target_position - self.tcp_position
        )
        arm_position = self.data.qpos[self._arm_qpos_adr]
        arm_width = np.maximum(
            self._arm_joint_ranges[:, 1] - self._arm_joint_ranges[:, 0],
            1.0e-6,
        )
        minimum_arm_margin = float(
            np.min(
                np.minimum(
                    arm_position - self._arm_joint_ranges[:, 0],
                    self._arm_joint_ranges[:, 1] - arm_position,
                )
                / arm_width
            )
        )
        actual_tilt_angle = float(
            rotation_vector(base_rotation @ self._home_base_rotation.T)
            @ self._home_base_rotation[:, 1]
        )
        vertical_limited = bool(
            self._vertical_target_clamped
            or (
                (self.arm_only_active or self._tilt_recovering)
                and np.linalg.norm(tcp_error_base[:2]) < 0.08
                and abs(float(tcp_error_base[2])) > 0.03
                and minimum_arm_margin < 0.02
            )
        )
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
            arm_only=self.arm_only_active,
            coordinator_gate=self._coordinator_gate,
            solver_residual=self._solver_residual,
            base_participation=self._base_participation,
            tcp_speed=self._tcp_speed,
            vertical_limited=vertical_limited,
            tilt_assist_active=self._tilt_recovering,
            tilt_angle=actual_tilt_angle,
        )
