"""Gymnasium environment for whole-body B2-W + Z1 control in MuJoCo."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

LEG_JOINTS = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
WHEEL_JOINTS = (
    "FR_wheel_joint",
    "FL_wheel_joint",
    "RR_wheel_joint",
    "RL_wheel_joint",
)
ARM_JOINTS = tuple(f"z1_joint{i}" for i in range(1, 7))
GRIPPER_JOINTS = ("z1_gripper_joint",)
CONTROL_JOINTS = LEG_JOINTS + WHEEL_JOINTS + ARM_JOINTS + GRIPPER_JOINTS

ACTUATORS = (
    (
        "FR_hip",
        "FR_thigh",
        "FR_calf",
        "FL_hip",
        "FL_thigh",
        "FL_calf",
        "RR_hip",
        "RR_thigh",
        "RR_calf",
        "RL_hip",
        "RL_thigh",
        "RL_calf",
        "FR_wheel",
        "FL_wheel",
        "RR_wheel",
        "RL_wheel",
    )
    + tuple(f"z1_joint{i}_motor" for i in range(1, 7))
    + ("z1_gripper_motor",)
)

LEG_NOMINAL = np.tile(np.array([0.0, 0.60, -1.20], dtype=np.float64), 4)
ARM_NOMINAL = np.array([0.0, 1.35, -1.65, 0.30, 0.0, 0.0], dtype=np.float64)
GRIPPER_NOMINAL = np.array([-0.90], dtype=np.float64)
JOINT_NOMINAL = np.concatenate([LEG_NOMINAL, np.zeros(4), ARM_NOMINAL, GRIPPER_NOMINAL])

LEG_ACTION_SCALE = np.tile(np.array([0.45, 0.65, 0.55], dtype=np.float64), 4)
ARM_ACTION_SCALE = np.array([0.8, 0.75, 0.75, 0.65, 0.6, 0.9], dtype=np.float64)
LEG_KP = np.tile(np.array([260.0, 300.0, 350.0], dtype=np.float64), 4)
LEG_KD = np.tile(np.array([5.0, 6.0, 7.0], dtype=np.float64), 4)
ARM_KP = np.array([35.0, 65.0, 45.0, 30.0, 22.0, 16.0], dtype=np.float64)
ARM_KD = np.array([1.5, 2.2, 1.6, 1.2, 0.9, 0.7], dtype=np.float64)
GRIPPER_KP = 12.0
GRIPPER_KD = 0.6
TORQUE_LIMITS = np.array(
    [200, 200, 300] * 4 + [20] * 4 + [30, 60, 30, 30, 30, 30, 15],
    dtype=np.float64,
)


class B2WZ1Env(gym.Env[np.ndarray, np.ndarray]):
    """Starter whole-body control task with 23 normalized actions.

    Actions control 12 leg joint targets, four wheel velocity targets, and six
    Z1 joint targets plus one gripper target. MuJoCo receives clipped PD torques
    at 500 Hz while the RL policy runs at 50 Hz.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
        command_curriculum: float = 1.0,
        _model_filename: str = "scene.xml",
    ) -> None:
        super().__init__()
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"Unsupported render mode: {render_mode}")

        model_path = (
            Path(__file__).resolve().parents[1] / "assets" / "b2w_z1" / _model_filename
        )
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.command_curriculum = float(np.clip(command_curriculum, 0.0, 1.0))
        self.frame_skip = round(0.02 / self.model.opt.timestep)
        self.dt = self.frame_skip * self.model.opt.timestep

        self.action_space = spaces.Box(-1.0, 1.0, shape=(23,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(80,),
            dtype=np.float32,
        )

        self._joint_ids = self._ids(mujoco.mjtObj.mjOBJ_JOINT, CONTROL_JOINTS)
        self._qpos_adr = self.model.jnt_qposadr[self._joint_ids]
        self._dof_adr = self.model.jnt_dofadr[self._joint_ids]
        self._actuator_ids = self._ids(mujoco.mjtObj.mjOBJ_ACTUATOR, ACTUATORS)
        self._base_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._ee_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "z1_ee")

        self._step_count = 0
        self._command = np.zeros(2, dtype=np.float64)
        self._ee_target_base = np.zeros(3, dtype=np.float64)
        self._last_action = np.zeros(23, dtype=np.float64)
        self._viewer: Any | None = None
        self._renderer: mujoco.Renderer | None = None

    def _id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return obj_id

    def _ids(self, obj_type: mujoco.mjtObj, names: tuple[str, ...]) -> NDArray[np.int32]:
        return np.asarray([self._id(obj_type, name) for name in names], dtype=np.int32)

    def _base_rotation(self) -> FloatArray:
        return self.data.xmat[self._base_body_id].reshape(3, 3).copy()

    def _current_ee_base(self) -> FloatArray:
        rotation = self._base_rotation()
        delta_world = self.data.site_xpos[self._ee_site_id] - self.data.xpos[self._base_body_id]
        return rotation.T @ delta_world

    def _joint_state(self) -> tuple[FloatArray, FloatArray]:
        return self.data.qpos[self._qpos_adr].copy(), self.data.qvel[self._dof_adr].copy()

    def _get_obs(self) -> np.ndarray:
        rotation = self._base_rotation()
        qpos, qvel = self._joint_state()
        base_linear_velocity = rotation.T @ self.data.qvel[:3]
        base_angular_velocity = rotation.T @ self.data.qvel[3:6]
        projected_gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        bounded_joint_positions = np.concatenate(
            [
                qpos[:12] - LEG_NOMINAL,
                qpos[16:22] - ARM_NOMINAL,
                qpos[22:] - GRIPPER_NOMINAL,
            ]
        )
        ee_error = self._ee_target_base - self._current_ee_base()

        observation = np.concatenate(
            [
                np.array([self.data.qpos[2]], dtype=np.float64),
                base_linear_velocity,
                base_angular_velocity,
                projected_gravity,
                bounded_joint_positions,
                0.1 * qvel,
                self._command,
                ee_error,
                self._last_action,
            ]
        )
        return observation.astype(np.float32)

    def _compute_torque(self, action: FloatArray) -> FloatArray:
        qpos, qvel = self._joint_state()
        torque = np.zeros(23, dtype=np.float64)

        leg_targets = LEG_NOMINAL + LEG_ACTION_SCALE * action[:12]
        torque[:12] = LEG_KP * (leg_targets - qpos[:12]) - LEG_KD * qvel[:12]

        wheel_velocity_targets = 24.0 * action[12:16]
        torque[12:16] = 1.2 * (wheel_velocity_targets - qvel[12:16])

        arm_targets = ARM_NOMINAL + ARM_ACTION_SCALE * action[16:22]
        joint_ranges = self.model.jnt_range[self._joint_ids[16:22]]
        arm_targets = np.clip(arm_targets, joint_ranges[:, 0], joint_ranges[:, 1])
        torque[16:22] = ARM_KP * (arm_targets - qpos[16:22]) - ARM_KD * qvel[16:22]

        gripper_range = self.model.jnt_range[self._joint_ids[22]]
        gripper_target = 0.5 * (
            gripper_range[0] + gripper_range[1] + action[22] * (gripper_range[1] - gripper_range[0])
        )
        torque[22] = GRIPPER_KP * (gripper_target - qpos[22]) - GRIPPER_KD * qvel[22]

        return np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)

    def _reward(
        self,
        action: FloatArray,
        torque: FloatArray,
    ) -> tuple[float, dict[str, float]]:
        rotation = self._base_rotation()
        local_velocity = rotation.T @ self.data.qvel[:3]
        local_angular_velocity = rotation.T @ self.data.qvel[3:6]
        upright = float(np.clip(rotation[2, 2], -1.0, 1.0))
        height_error = self.data.qpos[2] - 0.46
        ee_error = np.linalg.norm(self._ee_target_base - self._current_ee_base())

        terms = {
            "track_velocity": float(np.exp(-2.0 * ((local_velocity[0] - self._command[0]) ** 2))),
            "track_yaw": float(
                np.exp(-2.0 * ((local_angular_velocity[2] - self._command[1]) ** 2))
            ),
            "upright": max(0.0, upright),
            "height": float(np.exp(-12.0 * height_error**2)),
            "ee_tracking": float(np.exp(-18.0 * ee_error**2)),
            "alive": 1.0,
            "energy_penalty": -2.0e-5
            * float(np.sum(np.abs(torque * self.data.qvel[self._dof_adr]))),
            "action_rate_penalty": -0.01 * float(np.mean(np.square(action - self._last_action))),
            "lateral_penalty": -0.08 * float(local_velocity[1] ** 2),
        }
        reward = (
            0.85 * terms["track_velocity"]
            + 0.30 * terms["track_yaw"]
            + 0.35 * terms["upright"]
            + 0.25 * terms["height"]
            + 0.55 * terms["ee_tracking"]
            + 0.10 * terms["alive"]
            + terms["energy_penalty"]
            + terms["action_rate_penalty"]
            + terms["lateral_penalty"]
        )
        return float(reward), terms

    def _is_healthy(self) -> bool:
        rotation = self._base_rotation()
        height = float(self.data.qpos[2])
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        return bool(finite and 0.38 < height < 1.20 and rotation[2, 2] > 0.35)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:7] = np.array([0.0, 0.0, 0.68, 1.0, 0.0, 0.0, 0.0])
        reset_noise = self.np_random.uniform(-0.025, 0.025, size=23)
        reset_noise[12:16] = 0.0
        self.data.qpos[self._qpos_adr] = JOINT_NOMINAL + reset_noise
        self.data.qvel[:] = self.np_random.normal(0.0, 0.01, size=self.model.nv)
        mujoco.mj_forward(self.model, self.data)

        scale = self.command_curriculum
        self._command = np.array(
            [
                self.np_random.uniform(-0.2, 1.0) * scale,
                self.np_random.uniform(-0.6, 0.6) * scale,
            ],
            dtype=np.float64,
        )
        self._ee_target_base = self._current_ee_base() + self.np_random.uniform(
            low=np.array([-0.07, -0.10, -0.06]) * scale,
            high=np.array([0.07, 0.10, 0.08]) * scale,
        )
        self._last_action.fill(0.0)
        self._step_count = 0

        observation = self._get_obs()
        info = {
            "command_vx": float(self._command[0]),
            "command_yaw": float(self._command[1]),
            "gripper_position": float(self.data.qpos[self._qpos_adr[22]]),
        }
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(
        self,
        action: NDArray[np.floating[Any]],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        clipped_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        torque = self._compute_torque(clipped_action)
        self.data.ctrl[self._actuator_ids] = torque
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._step_count += 1

        reward, reward_terms = self._reward(clipped_action, torque)
        terminated = not self._is_healthy()
        truncated = self._step_count >= self.max_episode_steps
        self._last_action[:] = clipped_action
        observation = self._get_obs()

        info: dict[str, Any] = {
            "reward_terms": reward_terms,
            "base_height": float(self.data.qpos[2]),
            "ee_error": float(np.linalg.norm(self._ee_target_base - self._current_ee_base())),
            "command_vx": float(self._command[0]),
            "command_yaw": float(self._command[1]),
            "gripper_position": float(self.data.qpos[self._qpos_adr[22]]),
        }
        if terminated:
            info["termination_reason"] = "fallen_or_non_finite"
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer as mujoco_viewer

                self._viewer = mujoco_viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self) -> None:
        if self._viewer is not None:
            viewer = self._viewer
            viewer.close()
            deadline = time.monotonic() + 2.0
            while viewer.is_running() and time.monotonic() < deadline:
                time.sleep(0.01)
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
