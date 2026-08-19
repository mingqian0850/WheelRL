"""Hierarchical B2-W + Z1 task for unlatching and pulling open a door."""

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from wheelrl.envs.b2w_z1 import (
    ARM_ACTION_SCALE,
    ARM_KD,
    ARM_KP,
    ARM_NOMINAL,
    CONTROL_JOINTS,
    GRIPPER_KD,
    GRIPPER_KP,
    GRIPPER_NOMINAL,
    JOINT_NOMINAL,
    LEG_KD,
    LEG_KP,
    LEG_NOMINAL,
    TORQUE_LIMITS,
    B2WZ1Env,
)

FloatArray = NDArray[np.float64]
TaskStage = Literal["reach", "turn", "pull", "full"]
TASK_STAGES: tuple[TaskStage, ...] = ("reach", "turn", "pull", "full")

# The robot starts 0.02 m further forward than before the Z1 mount moved from
# x=0.23 to x=0.211 (plate front flush with the body front), keeping the
# arm-to-handle geometry identical for the trained policy (everything
# relative to the base is unchanged).
BASE_START = np.array([0.14, -0.47, 0.68], dtype=np.float64)
PREGRASP_ARM = np.array(
    [-0.16904, 1.98459, -1.84672, 0.03933, -0.06931, 0.0],
    dtype=np.float64,
)
GRASP_ARM = np.array(
    [-0.13625, 1.88349, -1.65623, 0.19033, -0.06136, 0.0],
    dtype=np.float64,
)
HANDLE_RELEASE_ANGLE = 0.35
DOOR_SUCCESS_ANGLE = -0.72
PREGRASP_DISTANCE = 0.18
GRASP_DISTANCE = 0.11


class B2WZ1DoorEnv(B2WZ1Env):
    """Open a pull door using a fixed low-level base controller and Z1 targets.

    The nine normalized actions are base forward velocity, base yaw rate, six
    arm joint targets, and the gripper target. Door and handle state are exposed
    to this first privileged policy; a later student policy can replace them
    with perception and history.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 750,
        task_stage: TaskStage = "full",
        domain_randomization: float = 1.0,
    ) -> None:
        if task_stage not in TASK_STAGES:
            raise ValueError(f"task_stage must be one of {TASK_STAGES}, got {task_stage!r}")
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            command_curriculum=0.0,
            _model_filename="door_scene.xml",
        )
        self.task_stage: TaskStage = task_stage
        self.domain_randomization = float(np.clip(domain_randomization, 0.0, 1.0))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(9,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(84,),
            dtype=np.float32,
        )

        self._door_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "door_hinge_joint")
        self._handle_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "door_handle_joint")
        self._door_qpos_adr = int(self.model.jnt_qposadr[self._door_joint_id])
        self._door_dof_adr = int(self.model.jnt_dofadr[self._door_joint_id])
        self._handle_qpos_adr = int(self.model.jnt_qposadr[self._handle_joint_id])
        self._handle_dof_adr = int(self.model.jnt_dofadr[self._handle_joint_id])
        self._handle_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "door_handle_grasp")
        self._door_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "door_panel")
        self._door_frame_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "door_frame")
        self._handle_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "door_handle")
        self._grasp_equality_id = self._id(
            mujoco.mjtObj.mjOBJ_EQUALITY, "gripper_handle_connect"
        )
        self._handle_geom_ids = {
            self._id(mujoco.mjtObj.mjOBJ_GEOM, "door_handle_hub"),
            self._id(mujoco.mjtObj.mjOBJ_GEOM, "door_handle_lever"),
        }
        self._gripper_geom_ids = {
            self._id(mujoco.mjtObj.mjOBJ_GEOM, "z1_gripper_stator_collision"),
            self._id(mujoco.mjtObj.mjOBJ_GEOM, "z1_gripper_mover_collision"),
        }

        self._nominal_door_mass = float(self.model.body_mass[self._door_body_id])
        self._nominal_door_inertia = self.model.body_inertia[self._door_body_id].copy()
        self._nominal_frame_pos = self.model.body_pos[self._door_frame_body_id].copy()
        self._base_command = np.zeros(2, dtype=np.float64)
        self._last_high_action = np.zeros(9, dtype=np.float64)
        self._latch_released = False
        self._grasp_attached = False
        self._phase = 0
        self._prev_phase = 0
        self._prev_handle_angle = 0.0
        self._prev_door_angle = 0.0
        self._prev_handle_distance = 0.0
        self._release_angle = HANDLE_RELEASE_ANGLE
        self._handle_spring = 2.8
        self._handle_damping = 0.12
        self._door_closer = 2.0
        self._door_damping = 1.4
        self._latch_stiffness = 850.0
        self._latch_damping = 45.0

    def set_task_stage(self, task_stage: TaskStage) -> None:
        """Change curriculum stage; the next reset adopts the new initial state."""
        if task_stage not in TASK_STAGES:
            raise ValueError(f"task_stage must be one of {TASK_STAGES}, got {task_stage!r}")
        self.task_stage = task_stage

    @property
    def door_angle(self) -> float:
        return float(self.data.qpos[self._door_qpos_adr])

    @property
    def handle_angle(self) -> float:
        return float(self.data.qpos[self._handle_qpos_adr])

    def _handle_position_base(self) -> FloatArray:
        rotation = self._base_rotation()
        delta = self.data.site_xpos[self._handle_site_id] - self.data.xpos[self._base_body_id]
        return rotation.T @ delta

    def _handle_error_world(self) -> FloatArray:
        return self.data.site_xpos[self._handle_site_id] - self.data.site_xpos[self._ee_site_id]

    def _door_normal_base(self) -> FloatArray:
        rotation = self._base_rotation()
        door_rotation = self.data.xmat[self._door_body_id].reshape(3, 3)
        return rotation.T @ door_rotation[:, 0]

    def _handle_axis_base(self) -> FloatArray:
        rotation = self._base_rotation()
        handle_rotation = self.data.xmat[self._handle_body_id].reshape(3, 3)
        return rotation.T @ handle_rotation[:, 1]

    def _gripper_contact(self) -> tuple[bool, float]:
        normal_force = 0.0
        in_contact = False
        contact_force = np.zeros(6, dtype=np.float64)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & self._handle_geom_ids and pair & self._gripper_geom_ids:
                in_contact = True
                mujoco.mj_contactForce(self.model, self.data, contact_index, contact_force)
                normal_force += max(0.0, float(contact_force[0]))
        return in_contact, normal_force

    def _update_phase(self) -> None:
        distance = float(np.linalg.norm(self._handle_error_world()))
        if self._phase == 0 and distance < PREGRASP_DISTANCE:
            self._phase = 1
        if self._phase == 1 and self._grasp_attached:
            self._phase = 2
        if self._latch_released:
            self._phase = max(self._phase, 3)
        if self.door_angle <= DOOR_SUCCESS_ANGLE:
            self._phase = 4

    def _update_latch_state(self) -> None:
        if self.handle_angle >= self._release_angle:
            self._latch_released = True

    def _update_grasp_constraint(self) -> None:
        gripper_position = float(self.data.qpos[self._qpos_adr[-1]])
        distance = float(np.linalg.norm(self._handle_error_world()))
        if (
            not self._grasp_attached
            and gripper_position > -0.60
            and distance < 0.085
        ):
            self.data.eq_active[self._grasp_equality_id] = 1
            self._grasp_attached = True
        elif self._grasp_attached and gripper_position < -0.95:
            self.data.eq_active[self._grasp_equality_id] = 0
            self._grasp_attached = False

    def _apply_door_passive_forces(self) -> None:
        handle_angle = self.handle_angle
        handle_velocity = float(self.data.qvel[self._handle_dof_adr])
        door_angle = self.door_angle
        door_velocity = float(self.data.qvel[self._door_dof_adr])

        self.data.qfrc_applied[self._handle_dof_adr] = (
            -self._handle_spring * handle_angle - self._handle_damping * handle_velocity
        )
        door_torque = -self._door_closer * door_angle - self._door_damping * door_velocity
        if not self._latch_released:
            door_torque += (
                -self._latch_stiffness * door_angle - self._latch_damping * door_velocity
            )
        self.data.qfrc_applied[self._door_dof_adr] = door_torque

    def _compute_high_level_torque(self, action: FloatArray) -> FloatArray:
        qpos, qvel = self._joint_state()
        torque = np.zeros(23, dtype=np.float64)

        torque[:12] = LEG_KP * (LEG_NOMINAL - qpos[:12]) - LEG_KD * qvel[:12]

        desired_command = np.array([0.55 * action[0], 0.90 * action[1]])
        self._base_command += 0.18 * (desired_command - self._base_command)
        rotation = self._base_rotation()
        local_velocity = rotation.T @ self.data.qvel[:3]
        local_angular_velocity = rotation.T @ self.data.qvel[3:6]
        tracked_forward_velocity = self._base_command[0] + 5.0 * (
            self._base_command[0] - local_velocity[0]
        )
        tracked_yaw_rate = self._base_command[1] + 1.5 * (
            self._base_command[1] - local_angular_velocity[2]
        )
        wheel_radius = 0.12
        half_track = 0.235
        right_velocity = (
            tracked_forward_velocity + half_track * tracked_yaw_rate
        ) / wheel_radius
        left_velocity = (
            tracked_forward_velocity - half_track * tracked_yaw_rate
        ) / wheel_radius
        wheel_targets = np.array(
            [right_velocity, left_velocity, right_velocity, left_velocity],
            dtype=np.float64,
        )
        wheel_targets = np.clip(wheel_targets, -22.0, 22.0)
        torque[12:16] = 4.0 * (wheel_targets - qvel[12:16])

        # Door policies act as residuals around a reachable pre-grasp. This
        # keeps zero-mean exploration at the handle instead of pulling the arm
        # back to its generic transport pose after a curriculum reset.
        arm_targets = PREGRASP_ARM + ARM_ACTION_SCALE * action[2:8]
        joint_ranges = self.model.jnt_range[self._joint_ids[16:22]]
        arm_targets = np.clip(arm_targets, joint_ranges[:, 0], joint_ranges[:, 1])
        torque[16:22] = ARM_KP * (arm_targets - qpos[16:22]) - ARM_KD * qvel[16:22]

        gripper_range = self.model.jnt_range[self._joint_ids[-1]]
        gripper_target = 0.5 * (
            gripper_range[0]
            + gripper_range[1]
            + action[8] * (gripper_range[1] - gripper_range[0])
        )
        torque[22] = (
            GRIPPER_KP * (gripper_target - qpos[22]) - GRIPPER_KD * qvel[22]
        )
        return np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)

    def _get_obs(self) -> np.ndarray:
        rotation = self._base_rotation()
        qpos, qvel = self._joint_state()
        base_linear_velocity = rotation.T @ self.data.qvel[:3]
        base_angular_velocity = rotation.T @ self.data.qvel[3:6]
        projected_gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        joint_positions = np.concatenate(
            [
                qpos[:12] - LEG_NOMINAL,
                qpos[16:22] - ARM_NOMINAL,
                qpos[22:] - GRIPPER_NOMINAL,
            ]
        )
        phase_one_hot = np.zeros(5, dtype=np.float64)
        phase_one_hot[self._phase] = 1.0
        handle_error_base = rotation.T @ self._handle_error_world()
        observation = np.concatenate(
            [
                np.array([self.data.qpos[2]], dtype=np.float64),
                base_linear_velocity,
                base_angular_velocity,
                projected_gravity,
                joint_positions,
                0.1 * qvel,
                self._handle_position_base(),
                handle_error_base,
                self._door_normal_base(),
                self._handle_axis_base(),
                np.array(
                    [
                        self.handle_angle,
                        0.1 * self.data.qvel[self._handle_dof_adr],
                        self.door_angle,
                        0.1 * self.data.qvel[self._door_dof_adr],
                        float(self._latch_released),
                        float(self._grasp_attached),
                    ],
                    dtype=np.float64,
                ),
                phase_one_hot,
                self._last_high_action,
            ]
        )
        if observation.shape != self.observation_space.shape:
            raise RuntimeError(
                f"Door observation has shape {observation.shape}, "
                f"expected {self.observation_space.shape}"
            )
        return observation.astype(np.float32)

    def _reward(
        self,
        action: FloatArray,
        torque: FloatArray,
    ) -> tuple[float, dict[str, float], bool]:
        rotation = self._base_rotation()
        distance = float(np.linalg.norm(self._handle_error_world()))
        upright = float(np.clip(rotation[2, 2], -1.0, 1.0))
        height_error = float(self.data.qpos[2] - 0.46)
        gripper_contact, contact_force = self._gripper_contact()
        handle_progress = float(
            np.clip(self.handle_angle / max(self._release_angle, 1.0e-6), 0.0, 1.2)
        )
        door_progress = float(np.clip(self.door_angle / DOOR_SUCCESS_ANGLE, 0.0, 1.2))
        handle_delta = self.handle_angle - self._prev_handle_angle
        door_delta = self._prev_door_angle - self.door_angle
        reach_delta = self._prev_handle_distance - distance
        gripper_position = float(self.data.qpos[self._qpos_adr[-1]])
        alignment = abs(
            float(np.dot(self._door_normal_base(), np.array([1.0, 0.0, 0.0])))
        )

        terms = {
            "reach": float(np.exp(-10.0 * distance**2)),
            "near_handle": float(distance < 0.10),
            "gripper_contact": float(gripper_contact),
            "contact_force": min(contact_force / 40.0, 1.0),
            "gripper_closed_near": float(
                distance < GRASP_DISTANCE and gripper_position > -0.55
            ),
            "reach_delta": reach_delta,
            "handle_progress": handle_progress,
            "handle_delta": handle_delta,
            "door_progress": door_progress,
            "door_delta": door_delta,
            "phase_advance": float(max(0, self._phase - self._prev_phase)),
            "alignment": alignment,
            "upright": max(0.0, upright),
            "height": float(np.exp(-12.0 * height_error**2)),
            "energy_penalty": -1.0e-5
            * float(np.sum(np.abs(torque * self.data.qvel[self._dof_adr]))),
            "action_rate_penalty": -0.01
            * float(np.mean(np.square(action - self._last_high_action))),
        }
        reward = (
            -0.02
            + 20.0 * terms["reach_delta"]
            + 0.01 * terms["gripper_contact"]
            + 0.005 * terms["contact_force"]
            + 2.0 * terms["phase_advance"]
            - 0.02 * (1.0 - terms["alignment"])
            - 0.03 * (1.0 - terms["upright"])
            - 0.02 * (1.0 - terms["height"])
            + terms["energy_penalty"]
            + terms["action_rate_penalty"]
        )
        if self.task_stage in ("turn", "pull", "full"):
            reward += (
                0.01 * terms["gripper_closed_near"]
                + 40.0 * terms["handle_delta"]
            )
        if self.task_stage in ("pull", "full"):
            reward += 50.0 * terms["door_delta"]

        success = (
            (self.task_stage == "reach" and self._phase >= 1)
            or (self.task_stage == "turn" and self._latch_released)
            or (self.task_stage in ("pull", "full") and self.door_angle <= DOOR_SUCCESS_ANGLE)
        )
        if success:
            reward += {"reach": 10.0, "turn": 20.0, "pull": 40.0, "full": 60.0}[
                self.task_stage
            ]
        return float(reward), terms, success

    def _randomize_door(self) -> None:
        scale = self.domain_randomization
        mass_scale = self.np_random.uniform(1.0 - 0.35 * scale, 1.0 + 0.35 * scale)
        self.model.body_mass[self._door_body_id] = self._nominal_door_mass * mass_scale
        self.model.body_inertia[self._door_body_id] = self._nominal_door_inertia * mass_scale
        frame_offset = self.np_random.uniform(-0.025, 0.025, size=3) * scale
        frame_offset[2] = 0.0
        self.model.body_pos[self._door_frame_body_id] = self._nominal_frame_pos + frame_offset
        self._release_angle = HANDLE_RELEASE_ANGLE + self.np_random.uniform(
            -0.07, 0.07
        ) * scale
        self._handle_spring = 2.8 * self.np_random.uniform(
            1.0 - 0.30 * scale, 1.0 + 0.30 * scale
        )
        self._door_closer = 2.0 * self.np_random.uniform(
            1.0 - 0.40 * scale, 1.0 + 0.40 * scale
        )
        self._door_damping = 1.4 * self.np_random.uniform(
            1.0 - 0.30 * scale, 1.0 + 0.30 * scale
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        if options and "task_stage" in options:
            self.set_task_stage(options["task_stage"])
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_door()

        base_noise = self.np_random.uniform(-0.025, 0.025, size=3)
        yaw = self.np_random.uniform(-0.05, 0.05)
        self.data.qpos[:7] = np.array(
            [
                BASE_START[0] + base_noise[0],
                BASE_START[1] + base_noise[1],
                BASE_START[2],
                np.cos(0.5 * yaw),
                0.0,
                0.0,
                np.sin(0.5 * yaw),
            ],
            dtype=np.float64,
        )
        joint_noise = self.np_random.uniform(-0.015, 0.015, size=len(CONTROL_JOINTS))
        joint_noise[12:16] = 0.0
        initial_joints = JOINT_NOMINAL.copy()
        if self.task_stage == "turn":
            initial_joints[16:22] = PREGRASP_ARM
            initial_joints[22] = -1.25
        elif self.task_stage == "pull":
            initial_joints[16:22] = GRASP_ARM
            initial_joints[22] = -0.40
        self.data.qpos[self._qpos_adr] = initial_joints + joint_noise

        self._latch_released = False
        self._grasp_attached = False
        self.data.eq_active[self._grasp_equality_id] = 0
        if self.task_stage == "pull":
            self.data.qpos[self._handle_qpos_adr] = HANDLE_RELEASE_ANGLE + 0.08
            self.data.qpos[self._door_qpos_adr] = -0.04
            self._latch_released = True
        self.data.qvel[:] = self.np_random.normal(0.0, 0.006, size=self.model.nv)
        self._base_command.fill(0.0)
        self._last_high_action.fill(0.0)
        self._step_count = 0
        self._phase = 3 if self._latch_released else 0
        mujoco.mj_forward(self.model, self.data)
        self._update_grasp_constraint()
        self._update_phase()
        self._prev_phase = self._phase
        self._prev_handle_angle = self.handle_angle
        self._prev_door_angle = self.door_angle
        self._prev_handle_distance = float(np.linalg.norm(self._handle_error_world()))

        observation = self._get_obs()
        info = self._info(success=False)
        if self.render_mode == "human":
            self.render()
        return observation, info

    def _info(self, success: bool) -> dict[str, Any]:
        contact, contact_force = self._gripper_contact()
        return {
            "task_stage": self.task_stage,
            "phase": self._phase,
            "is_success": success,
            "door_angle": self.door_angle,
            "handle_angle": self.handle_angle,
            "latch_released": self._latch_released,
            "grasp_attached": self._grasp_attached,
            "ee_handle_distance": float(np.linalg.norm(self._handle_error_world())),
            "gripper_contact": contact,
            "handle_contact_force": contact_force,
            "base_command_vx": float(self._base_command[0]),
            "base_command_yaw": float(self._base_command[1]),
        }

    def step(
        self,
        action: NDArray[np.floating[Any]],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        clipped_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        torque = self._compute_high_level_torque(clipped_action)
        self.data.ctrl[self._actuator_ids] = torque
        for _ in range(self.frame_skip):
            self._update_grasp_constraint()
            self._update_latch_state()
            self._apply_door_passive_forces()
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        self._update_latch_state()
        self._update_phase()

        reward, reward_terms, success = self._reward(clipped_action, torque)
        healthy = self._is_healthy()
        if not healthy and not success:
            reward -= 20.0
            reward_terms["fall_penalty"] = -20.0
        terminated = bool(success or not healthy)
        truncated = self._step_count >= self.max_episode_steps
        self._last_high_action[:] = clipped_action
        self._prev_handle_angle = self.handle_angle
        self._prev_door_angle = self.door_angle
        self._prev_handle_distance = float(np.linalg.norm(self._handle_error_world()))
        self._prev_phase = self._phase
        observation = self._get_obs()

        info = self._info(success=success)
        info["reward_terms"] = reward_terms
        if success:
            info["termination_reason"] = "task_success"
        elif not healthy:
            info["termination_reason"] = "fallen_or_non_finite"
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info
