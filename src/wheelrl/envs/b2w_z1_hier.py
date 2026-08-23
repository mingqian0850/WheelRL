"""Hierarchical TCP tracking: RL target generation + WBC precision servo.

The policy runs at 50 Hz and outputs residual corrections to a commanded
TCP pose (position offset + RPY offset, base frame). The wrapped
``B2WZ1WholeBodyController`` serves the resulting target at 100 Hz with its
analytic damped-least-squares whole-body IK (0.5 mm / 0.04 deg class
precision). The reward measures how well the *achieved* TCP pose matches the
commanded one, plus stability and effort terms.

This splits the task the way the literature does: the analytic controller
provides precision and constraints, the RL policy provides coordination,
robustness and the learned command-to-target mapping.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from wheelrl.envs.b2w_z1_track import (
    EE_POS_BOUNDS,
    POS_OFFSET_HIGH,
    POS_OFFSET_LOW,
    rotation_vector,
    rpy_rotation,
)
from wheelrl.wbc import WBC_MODEL_PATH, B2WZ1WholeBodyController

FloatArray = NDArray[np.float64]

# Command pose offsets at full curriculum (same magnitudes as the track env).
COMMAND_POS_LOW = POS_OFFSET_LOW
COMMAND_POS_HIGH = POS_OFFSET_HIGH
COMMAND_ORI_MAX = float(np.deg2rad(25.0))
# Residual action limits (the WBC does the coarse motion; the policy nudges).
ACTION_POS_LIMIT = 0.05  # m
ACTION_ORI_LIMIT = float(np.deg2rad(10.0))
SETTLE_STEPS = 30  # WBC steps (0.3 s) before capturing the reference


class B2WZ1HierEnv(gym.Env[np.ndarray, np.ndarray]):
    """RL-over-WBC hierarchical TCP tracking environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
        tracking_curriculum: float = 1.0,
        randomization: float = 1.0,
        action_penalty: float = 0.5,
        arm_first: bool = False,
        model: mujoco.MjModel | None = None,
        data: mujoco.MjData | None = None,
    ) -> None:
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.tracking_curriculum = float(np.clip(tracking_curriculum, 0.0, 1.0))
        self.randomization = float(np.clip(randomization, 0.0, 1.0))
        # Penalize |action|^2 so the residual only moves the servo target when
        # it genuinely helps (unreachable commands, transients). Without it the
        # reward is flat within ~±2 cm and PPO settles at an arbitrary
        # self-consistent nonzero residual, which shows up as a ~25 mm
        # steady-state tracking offset.
        self.action_penalty = float(np.clip(action_penalty, 0.0, 5.0))
        # Arm-first macro-micro servo: the policy was trained with the base
        # participating, so default off here (keeps the trained dynamics).
        self.arm_first = bool(arm_first)

        if model is None or data is None:
            model = mujoco.MjModel.from_xml_path(str(WBC_MODEL_PATH))
            data = mujoco.MjData(model)
        self.model = model
        self.data = data
        self.wbc = B2WZ1WholeBodyController(
            self.model,
            self.data,
            control_hz=100.0,
            base_assist=0.25,
            auto_drive=True,
            arm_first=self.arm_first,
        )
        self.dt = 0.02  # policy period (WBC steps twice per env step)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        # base obs (63) + command rpy (3) + achieved-command
        # error (6) + last action (6) = 98
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(78,), dtype=np.float32
        )

        self._base_body_id = self.wbc._base_body_id
        self._ee_site_id = self.wbc._tcp_site_id
        self._command_pos_world = np.zeros(3, dtype=np.float64)
        self._command_rpy = np.zeros(3, dtype=np.float64)
        self._command_rotation_world = np.eye(3)
        self._last_action = np.zeros(6, dtype=np.float64)
        self._step_count = 0
        self._viewer: Any | None = None
        self._renderer: Any | None = None

    # ------------------------------------------------------------------ task
    def _base_rotation(self) -> FloatArray:
        return self.data.xmat[self._base_body_id].reshape(3, 3).copy()

    def _ee_base(self) -> FloatArray:
        rotation = self._base_rotation()
        delta = self.data.site_xpos[self._ee_site_id] - self.data.xpos[self._base_body_id]
        return rotation.T @ delta

    def _ee_rotation_base(self) -> FloatArray:
        rotation = self._base_rotation()
        return rotation.T @ self.data.site_xmat[self._ee_site_id].reshape(3, 3)

    def _command_error(self) -> FloatArray:
        pos_error = self._command_pos_world - self.data.site_xpos[self._ee_site_id]
        current = self.data.site_xmat[self._ee_site_id].reshape(3, 3)
        return np.concatenate(
            [pos_error, rotation_vector(self._command_rotation_world @ current.T)]
        )

    def _apply_action(self, action: FloatArray) -> None:
        """Map the residual action onto the commanded pose and serve it with WBC."""
        clipped = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # The command is expressed in the base frame; the WBC works in world.
        base_rotation = self._base_rotation()
        position_offset = clipped[:3] * ACTION_POS_LIMIT
        rpy_offset = clipped[3:] * ACTION_ORI_LIMIT
        target_position = self._command_pos_world + base_rotation @ position_offset
        target_rotation = (
            base_rotation
            @ rpy_rotation(rpy_offset)
            @ (base_rotation.T @ self._command_rotation_world)
        )
        self.wbc.set_target_pose(target_position, target_rotation)

    def _stability_terms(self) -> dict[str, float]:
        upright = float(np.clip(self.data.xmat[self._base_body_id].reshape(3, 3)[2, 2], -1.0, 1.0))
        height = float(self.data.qpos[2])
        return {
            "upright": max(0.0, upright),
            "tilt_penalty": -0.60 * max(0.0, 0.80 - upright),
            "height_penalty": -0.5 * max(0.0, abs(height - 0.62) - 0.12),
        }

    # ------------------------------------------------------------------ API
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.wbc.reset()
        for _ in range(SETTLE_STEPS):
            self.wbc.step()
        self.wbc.capture_reference()

        scale = self.tracking_curriculum
        # Command pose in the base frame at reset, then frozen in the world.
        command_pos_base = self._ee_base() + self.np_random.uniform(
            COMMAND_POS_LOW * scale, COMMAND_POS_HIGH * scale
        )
        command_pos_base = np.clip(command_pos_base, EE_POS_BOUNDS[:, 0], EE_POS_BOUNDS[:, 1])
        self._command_rpy = self.np_random.uniform(
            -COMMAND_ORI_MAX, COMMAND_ORI_MAX, size=3
        ) * scale
        base_rotation = self._base_rotation()
        self._command_pos_world = (
            self.data.xpos[self._base_body_id] + base_rotation @ command_pos_base
        )
        self._command_rotation_world = (
            base_rotation
            @ rpy_rotation(self._command_rpy)
            @ (base_rotation.T @ self._ee_rotation_base())
        )
        self._last_action.fill(0.0)
        self._step_count = 0
        return self._get_obs(), {"command_pos": self._command_pos_world.copy()}

    def step(
        self,
        action: NDArray[np.floating[Any]],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._apply_action(action)
        for _ in range(2):  # 100 Hz WBC servo inside a 50 Hz policy step
            self.wbc.step()
        self._step_count += 1

        error = self._command_error()
        pos_error = float(np.linalg.norm(error[:3]))
        ori_error = float(np.linalg.norm(error[3:]))
        terms = {
            "command_position": float(np.exp(-18.0 * pos_error**2)),
            "command_orientation": float(np.exp(-25.0 * ori_error**2)),
            "sync_gate": float(np.exp(-8.0 * pos_error**2) * np.exp(-8.0 * ori_error**2)),
        }
        terms.update(self._stability_terms())
        control_vel = np.concatenate(
            [
                self.data.qvel[self.wbc._leg_dof_adr],
                self.data.qvel[self.wbc._wheel_dof_adr],
                self.data.qvel[self.wbc._arm_dof_adr],
                self.data.qvel[[self.wbc._gripper_dof_adr]],
            ]
        )
        terms["energy_penalty"] = -2.0e-5 * float(
            np.sum(np.abs(self.wbc._last_torque * control_vel))
        )
        terms["action_rate_penalty"] = -0.01 * float(
            np.mean(np.square(np.asarray(action, float) - self._last_action))
        )
        terms["action_magnitude_penalty"] = -self.action_penalty * float(
            np.mean(np.square(np.asarray(action, float)))
        )
        reward = (
            1.0 * terms["command_position"]
            + 0.6 * terms["command_orientation"]
            + 0.5 * terms["sync_gate"]
            + 0.2 * terms["upright"]
            + terms["tilt_penalty"]
            + terms["height_penalty"]
            + terms["energy_penalty"]
            + terms["action_rate_penalty"]
            + terms["action_magnitude_penalty"]
        )
        upright = float(np.clip(self.data.xmat[self._base_body_id].reshape(3, 3)[2, 2], -1.0, 1.0))
        healthy = bool(
            np.isfinite(self.data.qpos).all()
            and 0.20 < self.data.qpos[2] < 1.30
            and upright > 0.35
        )
        terminated = not healthy
        if terminated:
            reward -= 20.0
        truncated = self._step_count >= self.max_episode_steps
        self._last_action[:] = np.asarray(action, dtype=np.float64)

        info = {
            "reward_terms": terms,
            "command_pos_error": pos_error,
            "command_ori_error": ori_error,
            "upright": upright,
            "termination_reason": "fallen_or_non_finite" if terminated else "truncated",
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        # 80-dim base observation mirrors B2WZ1Env._get_obs (control order).
        wbc = self.wbc
        rotation = self._base_rotation()
        qpos = self.data.qpos
        qvel = self.data.qvel
        base_linear_velocity = rotation.T @ qvel[:3]
        base_angular_velocity = rotation.T @ qvel[3:6]
        projected_gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        joint_positions = np.concatenate(
            [
                qpos[wbc._leg_qpos_adr] - np.tile(np.array([0.0, 1.0, -2.0]), 4),
                qpos[wbc._arm_qpos_adr] - np.array([0.0, 1.35, -1.65, 0.30, 0.0, 0.0]),
                qpos[[wbc._gripper_qpos_adr]] - np.array([-0.90]),
            ]
        )
        control_vel = np.concatenate(
            [
                qvel[wbc._leg_dof_adr],
                qvel[wbc._wheel_dof_adr],
                qvel[wbc._arm_dof_adr],
                qvel[[wbc._gripper_dof_adr]],
            ]
        )
        base_obs = np.concatenate(
            [
                np.array([qpos[2]], dtype=np.float64),
                base_linear_velocity,
                base_angular_velocity,
                projected_gravity,
                joint_positions,
                0.1 * control_vel,
                np.zeros(2),  # command slot (unused here)
                rotation.T @ (self._command_pos_world - self.data.site_xpos[self._ee_site_id]),
                self._last_action[:6],
            ]
        )
        return np.concatenate(
            [base_obs, self._command_rpy, self._command_error(), self._last_action]
        ).astype(np.float32)

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
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
