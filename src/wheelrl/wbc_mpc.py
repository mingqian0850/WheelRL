"""MuJoCo <-> MPC bridge for the B2-W + Z1 whole-body controller.

This module connects the MuJoCo scene (physics + state) to the inverse-dynamics
MPC built on Pinocchio/CasADi.  It handles the representation differences:

  - MuJoCo stores the free-joint quaternion as ``(w, x, y, z)`` while Pinocchio
    uses ``(x, y, z, w)``,
  - joint ordering differs between the MJCF and the URDF, so all mapping is done
    by name,
  - MuJoCo has 23 actuated joints (including the gripper); the MPC optimizes the
    22 non-gripper joints and the gripper keeps its own PD target.

The controller runs a receding horizon at a user-specified rate: at every
``control`` step it reads the MuJoCo state, solves the OCP, and applies the
first-node joint torques plus a base-stabilizing posture wrench.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import casadi as ca
import mujoco
import numpy as np
from numpy.typing import NDArray

from wheelrl.envs.b2w_z1 import (
    ACTUATORS,
    ARM_JOINTS,
    GRIPPER_JOINTS,
    GRIPPER_KP,
    GRIPPER_KD,
    GRIPPER_NOMINAL,
    LEG_JOINTS,
    TORQUE_LIMITS,
    WHEEL_JOINTS,
)
from wheelrl.mpc.ocp_whole_body_rnea import make_ocp
from wheelrl.mpc.robot import B2WZ1, URDF_JOINT_NAMES

FloatArray = NDArray[np.float64]

MPC_SPEED_PROFILE_SCALES = {
    "precision": 0.40,
    "normal": 1.00,
    "fast": 2.00,
}
MPC_ROTATION_SPEED_PROFILE_SCALES = {
    "precision": 0.40,
    "normal": 1.00,
    "fast": 1.50,
}

SCENE_PATH = (
    Path(__file__).resolve().parent / "assets" / "b2w_z1" / "wbc_scene.xml"
)
COMPILED_SOLVER_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "b2w_z1"
    / "mpc"
    / "codegen"
    / "lib"
    / "libsolver_function.so"
)

SOLVER_ARGS = {
    "ipopt": {
        "opts": {
            "expand": True,
            "ipopt.print_level": 0,
            "ipopt.max_iter": 500,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
        }
    },
    "fatrop": {
        "opts": {
            "expand": True,
            "structure_detection": "auto",
            "debug": False,
            "fatrop.print_level": 0,
            "fatrop.max_iter": 50,
            "fatrop.tol": 1e-3,
            "fatrop.mu_init": 1e-4,
            "fatrop.warm_start_init_point": True,
        }
    },
}


class B2WZ1MPCController:
    """Receding-horizon whole-body controller bridging MuJoCo and MPC."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        control_hz: float = 20.0,
        solver: str = "fatrop",
        load_compiled_solver: bool = True,
        nodes: int = 10,
        tau_nodes: int = 3,
        dt_min: float = 0.03,
        dt_max: float = 0.15,
        warm_start: bool = True,
        base_vel_des: FloatArray | None = None,
        arm_vel_des: FloatArray | None = None,
        arm_force_des: FloatArray | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.control_dt = 1.0 / control_hz
        frame_skip = self.control_dt / model.opt.timestep
        self.frame_skip = int(round(frame_skip))
        if self.frame_skip < 1 or not np.isclose(frame_skip, self.frame_skip):
            raise ValueError("control_hz must divide the MuJoCo simulation rate")

        # --- Pinocchio robot + OCP -------------------------------------------
        self.robot = B2WZ1()
        self.robot.set_gait_sequence("stand", gait_period=0.5)
        self.ocp = make_ocp(
            self.robot,
            nodes=nodes,
            tau_nodes=tau_nodes,
            warm_start=warm_start,
            include_acc=True,
        )
        self.ocp.set_time_params(dt_min, dt_max)
        self.ocp.set_swing_params(0.07, [0.1, -0.2])

        self.base_vel_des = (
            np.zeros(6) if base_vel_des is None else np.asarray(base_vel_des, float)
        )
        self.arm_vel_des = (
            np.zeros(3) if arm_vel_des is None else np.asarray(arm_vel_des, float)
        )
        self.arm_force_des = (
            np.zeros(3) if arm_force_des is None else np.asarray(arm_force_des, float)
        )
        self.ocp.set_tracking_targets(
            self.base_vel_des, self.arm_vel_des, self.arm_force_des
        )
        self.ocp.init_solver(solver, SOLVER_ARGS[solver])
        self._compiled_solver_function = None
        if load_compiled_solver:
            if not COMPILED_SOLVER_PATH.exists():
                raise FileNotFoundError(
                    "compiled solver not found; run codegen first "
                    "(python -m wheelrl.mpc.codegen and build the CMake project)"
                )
            self._compiled_solver_function = ca.external(
                "solver_function", str(COMPILED_SOLVER_PATH)
            )

        # --- name-based joint mapping ----------------------------------------
        self._mujoco_joint_ids = {}
        for name in (*LEG_JOINTS, *WHEEL_JOINTS, *ARM_JOINTS, *GRIPPER_JOINTS):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"MuJoCo joint not found: {name}")
            self._mujoco_joint_ids[name] = jid

        # URDF name -> pinocchio qpos index
        self._pin_q_index: dict[str, int] = {}
        for name in URDF_JOINT_NAMES:
            jid = self.robot.model.getJointId(name)
            self._pin_q_index[name] = self.robot.model.joints[jid].idx_q

        # MuJoCo -> URDF name map
        self._mujoco_to_urdf = {}
        for mj_name, urdf_name in self._joint_name_map().items():
            self._mujoco_to_urdf[mj_name] = urdf_name

        self._mujoco_qpos_adr = {
            name: int(model.jnt_qposadr[jid]) for name, jid in self._mujoco_joint_ids.items()
        }
        self._mujoco_dof_adr = {
            name: int(model.jnt_dofadr[jid]) for name, jid in self._mujoco_joint_ids.items()
        }
        self._actuator_ids = [
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
            for name in ACTUATORS
        ]
        self._gripper_actuator_id = self._actuator_ids[-1]
        self._base_body_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        )

        self._gripper_target = float(GRIPPER_NOMINAL[0])
        self._t = 0.0
        self._step_count = 0
        self._last_tau = np.zeros(22)
        self._solve_times: list[float] = []
        self._constraint_violations: list[float] = []

        # --- compatibility layer with wheelrl.wbc.B2WZ1WholeBodyController ---
        self.control_dt = self.control_dt  # already set above
        self.auto_drive = True  # accepted for API compatibility (wheels roll via contact)
        self._speed_profile = "normal"
        self._speed_scale = 1.0
        self._rotation_speed_scale = 1.0
        self._active = False
        self._home_tcp_position = np.zeros(3)
        self._home_tcp_rotation = np.eye(3)
        self._target_position = np.zeros(3)
        self._target_rotation = np.eye(3)
        self._position_gain = 1.2  # outer-loop P gain for TCP position tracking
        self._orientation_gain = 2.0

    # ------------------------------------------------------------------ mapping
    def _joint_name_map(self) -> dict[str, str]:
        """Map MuJoCo joint names onto URDF joint names (by name)."""
        leg_map = {
            "FR_hip_joint": "FR_hip_joint",
            "FR_thigh_joint": "FR_thigh_joint",
            "FR_calf_joint": "FR_calf_joint",
            "FL_hip_joint": "FL_hip_joint",
            "FL_thigh_joint": "FL_thigh_joint",
            "FL_calf_joint": "FL_calf_joint",
            "RR_hip_joint": "RR_hip_joint",
            "RR_thigh_joint": "RR_thigh_joint",
            "RR_calf_joint": "RR_calf_joint",
            "RL_hip_joint": "RL_hip_joint",
            "RL_thigh_joint": "RL_thigh_joint",
            "RL_calf_joint": "RL_calf_joint",
        }
        wheel_map = {
            "FR_wheel_joint": "FR_foot_joint",
            "FL_wheel_joint": "FL_foot_joint",
            "RR_wheel_joint": "RR_foot_joint",
            "RL_wheel_joint": "RL_foot_joint",
        }
        arm_map = {
            mj: urdf
            for mj, urdf in zip(
                ARM_JOINTS, [f"arm_joint{i}" for i in range(1, 7)]
            )
        }
        return {**leg_map, **wheel_map, **arm_map}

    # --------------------------------------------------------------- state I/O
    def _mujoco_to_pin_q(self) -> FloatArray:
        """Convert MuJoCo qpos (wxyz) to Pinocchio q (xyzw)."""
        q = np.zeros(self.robot.nq)
        mj_qpos = self.data.qpos
        # Base position
        q[0:3] = mj_qpos[0:3]
        # Quaternion: MuJoCo (w,x,y,z) -> Pinocchio (x,y,z,w)
        q[3:7] = np.array(
            [mj_qpos[4], mj_qpos[5], mj_qpos[6], mj_qpos[3]]
        )
        for mj_name, urdf_name in self._mujoco_to_urdf.items():
            q[self._pin_q_index[urdf_name]] = mj_qpos[
                self._mujoco_qpos_adr[mj_name]
            ]
        return q

    def _mujoco_to_pin_v(self) -> FloatArray:
        v = np.zeros(self.robot.nv)
        # Base linear/angular velocity: MuJoCo stores (vx,vy,vz, wx,wy,wz) in
        # the world frame, which is exactly what Pinocchio expects.
        v[0:6] = self.data.qvel[0:6]
        # Joint velocities are written in URDF order via the qpos mapping.
        pin_v_j = np.zeros(self.robot.nj)
        for mj_name, urdf_name in self._mujoco_to_urdf.items():
            urdf_idx = list(URDF_JOINT_NAMES).index(urdf_name)
            pin_v_j[urdf_idx] = self.data.qvel[self._mujoco_dof_adr[mj_name]]
        v[6:] = pin_v_j
        return v

    def _pin_tau_to_mujoco(self, tau_j: FloatArray) -> FloatArray:
        """Map 22 joint torques from URDF order onto MuJoCo actuators (23)."""
        ctrl = np.zeros(23)
        for mj_name, urdf_name in self._mujoco_to_urdf.items():
            urdf_idx = list(URDF_JOINT_NAMES).index(urdf_name)
            mj_act_idx = list(
                (*LEG_JOINTS, *WHEEL_JOINTS, *ARM_JOINTS)
            ).index(mj_name)
            ctrl[mj_act_idx] = tau_j[urdf_idx]
        # Gripper PD target (not optimized by the MPC).
        gripper_pos = float(self.data.qpos[self._mujoco_qpos_adr["z1_gripper_joint"]])
        gripper_vel = float(self.data.qvel[self._mujoco_dof_adr["z1_gripper_joint"]])
        ctrl[-1] = (
            GRIPPER_KP * (self._gripper_target - gripper_pos)
            - GRIPPER_KD * gripper_vel
        )
        return np.clip(ctrl, -TORQUE_LIMITS, TORQUE_LIMITS)

    # -------------------------------------------------------------- controller
    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = np.array([0.0, 0.0, 0.62, 1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self._mujoco_qpos_adr["FL_hip_joint"]] = 0.0
        self.data.qpos[self._mujoco_qpos_adr["FL_thigh_joint"]] = 0.60
        self.data.qpos[self._mujoco_qpos_adr["FL_calf_joint"]] = -1.20
        self.data.qpos[self._mujoco_qpos_adr["FR_hip_joint"]] = 0.0
        self.data.qpos[self._mujoco_qpos_adr["FR_thigh_joint"]] = 0.60
        self.data.qpos[self._mujoco_qpos_adr["FR_calf_joint"]] = -1.20
        self.data.qpos[self._mujoco_qpos_adr["RR_hip_joint"]] = 0.0
        self.data.qpos[self._mujoco_qpos_adr["RR_thigh_joint"]] = 0.60
        self.data.qpos[self._mujoco_qpos_adr["RR_calf_joint"]] = -1.20
        self.data.qpos[self._mujoco_qpos_adr["RL_hip_joint"]] = 0.0
        self.data.qpos[self._mujoco_qpos_adr["RL_thigh_joint"]] = 0.60
        self.data.qpos[self._mujoco_qpos_adr["RL_calf_joint"]] = -1.20
        for name in WHEEL_JOINTS:
            self.data.qpos[self._mujoco_qpos_adr[name]] = 0.0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._t = 0.0
        self._step_count = 0

    def set_targets(
        self,
        base_vel_des: FloatArray | None = None,
        arm_vel_des: FloatArray | None = None,
        arm_force_des: FloatArray | None = None,
    ) -> None:
        if base_vel_des is not None:
            self.base_vel_des = np.asarray(base_vel_des, float)
        if arm_vel_des is not None:
            self.arm_vel_des = np.asarray(arm_vel_des, float)
        if arm_force_des is not None:
            self.arm_force_des = np.asarray(arm_force_des, float)
        self.ocp.set_tracking_targets(
            self.base_vel_des, self.arm_vel_des, self.arm_force_des
        )

    def set_gripper(self, *, closed: bool) -> None:
        self._gripper_target = 0.0 if closed else -1.35

    def toggle_gripper(self) -> None:
        self.set_gripper(closed=not self.gripper_closed)

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
        ee_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "z1_ee")
        )
        return self.data.site_xpos[ee_id].copy()

    @property
    def tcp_rotation(self) -> FloatArray:
        ee_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "z1_ee")
        )
        return self.data.site_xmat[ee_id].reshape(3, 3).copy()

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

    def capture_reference(self) -> None:
        """Capture the settled TCP pose as home and enable tracking."""
        self._home_tcp_position[:] = self.tcp_position
        self._home_tcp_rotation[:] = self.tcp_rotation
        self._target_position[:] = self._home_tcp_position
        self._target_rotation[:] = self._home_tcp_rotation
        self._active = True

    def set_target_pose(
        self,
        position: FloatArray,
        rotation: FloatArray,
    ) -> None:
        position_array = np.asarray(position, dtype=np.float64)
        rotation_array = np.asarray(rotation, dtype=np.float64)
        if position_array.shape != (3,) or rotation_array.shape != (3, 3):
            raise ValueError("position must be (3,) and rotation must be (3, 3)")
        self._target_position[:] = position_array
        self._target_rotation[:] = rotation_array

    def set_home_offset(
        self,
        position_offset: FloatArray,
        rpy_offset_radians: FloatArray,
    ) -> None:
        rotation = self._home_tcp_rotation.copy()
        rpy = np.asarray(rpy_offset_radians, dtype=np.float64)
        # Rz(yaw) Ry(pitch) Rx(roll), matching wheelrl.wbc.rpy_rotation.
        rz = np.array(
            [
                [np.cos(rpy[2]), -np.sin(rpy[2]), 0.0],
                [np.sin(rpy[2]), np.cos(rpy[2]), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        ry = np.array(
            [
                [np.cos(rpy[1]), 0.0, np.sin(rpy[1])],
                [0.0, 1.0, 0.0],
                [-np.sin(rpy[1]), 0.0, np.cos(rpy[1])],
            ]
        )
        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(rpy[0]), -np.sin(rpy[0])],
                [0.0, np.sin(rpy[0]), np.cos(rpy[0])],
            ]
        )
        self.set_target_pose(
            self._home_tcp_position + np.asarray(position_offset, dtype=np.float64),
            rotation @ (rz @ ry @ rx),
        )

    def go_home(self) -> None:
        self.set_target_pose(self._home_tcp_position, self._home_tcp_rotation)

    def nudge_position(self, world_delta: FloatArray) -> None:
        self._target_position += np.asarray(world_delta, dtype=np.float64)

    def nudge_position_base(self, local_delta: FloatArray) -> None:
        delta = np.asarray(local_delta, dtype=np.float64)
        base_rotation = self.data.xmat[self._base_body_id].reshape(3, 3)
        forward = base_rotation[:, 0].copy()
        forward[2] = 0.0
        forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
        left = np.array([-forward[1], forward[0], 0.0], dtype=np.float64)
        self.nudge_position(
            delta[0] * forward
            + delta[1] * left
            + delta[2] * np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )

    def nudge_orientation(self, world_axis: FloatArray, angle: float) -> None:
        axis = np.asarray(world_axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        cross = np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
        )
        rotation = (
            np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
        )
        self._target_rotation[:] = rotation @ self._target_rotation

    def set_speed_profile(self, profile: str) -> None:
        if profile not in MPC_SPEED_PROFILE_SCALES:
            raise ValueError(f"speed profile must be one of {MPC_SPEED_PROFILE_SCALES}")
        self._speed_profile = profile
        self._speed_scale = MPC_SPEED_PROFILE_SCALES[profile]
        self._rotation_speed_scale = MPC_ROTATION_SPEED_PROFILE_SCALES[profile]

    def _tcp_tracking_targets(self) -> tuple[FloatArray, FloatArray]:
        """Outer-loop position/attitude error -> arm EE velocity command."""
        base_rotation = self.data.xmat[self._base_body_id].reshape(3, 3)
        position_error = self._target_position - self.tcp_position
        arm_vel = self._position_gain * (base_rotation.T @ position_error)
        arm_vel *= self._speed_scale
        arm_vel = np.clip(arm_vel, -0.30, 0.30)

        target_rotation = self._target_rotation
        tcp_rotation = self.tcp_rotation
        # rotation vector of target relative to current orientation
        relative = target_rotation @ tcp_rotation.T
        cosine = float(np.clip(0.5 * (np.trace(relative) - 1.0), -1.0, 1.0))
        angle = float(np.arccos(cosine))
        if angle < 1.0e-7:
            axis = np.zeros(3)
        else:
            skew = 0.5 * np.array(
                [
                    relative[2, 1] - relative[1, 2],
                    relative[0, 2] - relative[2, 0],
                    relative[1, 0] - relative[0, 1],
                ],
                dtype=np.float64,
            )
            axis = skew / np.sin(angle)
        angular_vel = self._orientation_gain * angle * axis
        angular_vel *= self._rotation_speed_scale
        angular_vel = np.clip(angular_vel, -0.5, 0.5)
        # arm_vel_des is linear only in the OCP; add angular via a small
        # linear proxy is not meaningful, so we keep it linear for now and
        # expose orientation error in diagnostics.
        return arm_vel, angular_vel

    def step(self) -> dict[str, float]:
        """Solve MPC from the current state and advance MuJoCo one control step."""
        if self._active:
            arm_vel, angular_vel = self._tcp_tracking_targets()
            self.ocp.set_tracking_targets(
                self.base_vel_des, arm_vel, self.arm_force_des
            )
        q = self._mujoco_to_pin_q()
        v = self._mujoco_to_pin_v()
        x_init = np.concatenate([q, v])
        self.ocp.update_params(x_init, self._t)
        params = self.ocp.get_solver_params()
        if self._compiled_solver_function is not None:
            sol_x = self._compiled_solver_function(*params)
        else:
            sol_x = self.ocp.solver_function(*params)
        self.ocp.retract_stacked_sol(sol_x, retract_all=False)
        tau_j = np.asarray(self.ocp.U_prev[0][self.ocp.tau_idx :]).flatten()
        if tau_j.size != 22:
            # torque nodes may not include all joints in the first node; pad
            raise RuntimeError(
                f"unexpected tau size {tau_j.size}; tau_nodes={self.ocp.tau_nodes}"
            )
        self._last_tau[:] = np.asarray(tau_j, float)
        ctrl = self._pin_tau_to_mujoco(self._last_tau)
        self.data.ctrl[self._actuator_ids] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._t += self.control_dt
        self._step_count += 1
        return self.diagnostics()

    def diagnostics(self) -> dict[str, float]:
        base_rotation = self.data.xmat[self._base_body_id].reshape(3, 3)
        position_error = float(
            np.linalg.norm(self._target_position - self.tcp_position)
        )
        relative = self._target_rotation @ self.tcp_rotation.T
        cosine = float(np.clip(0.5 * (np.trace(relative) - 1.0), -1.0, 1.0))
        orientation_error = float(np.arccos(cosine))
        return SimpleNamespace(
            t=self._t,
            base_height=float(self.data.qpos[2]),
            upright=float(base_rotation[2, 2]),
            position_error=position_error,
            orientation_error=orientation_error,
            base_position_error=0.0,
            max_torque=(
                float(np.max(np.abs(self._last_tau))) if self._last_tau.size else 0.0
            ),
            mobile_base_active=False,
            base_goal_distance=0.0,
        )

    def close(self) -> None:
        pass
