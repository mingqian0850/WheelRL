"""Hierarchical whole-body controller: RL target generation over WBC servo.

``B2WZ1HierController`` presents the same interface as
:class:`wheelrl.wbc.B2WZ1WholeBodyController` so the play-wbc panel can drive
it. The commanded TCP pose (from the panel) is the reference; every 50 Hz
step a small residual policy (6-dim position/RPY offsets) corrects it, and
the wrapped 100 Hz WBC serves the result with its analytic DLS IK.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from wheelrl.envs.b2w_z1_hier import B2WZ1HierEnv
from wheelrl.envs.b2w_z1_track import rotation_vector, rpy_rotation
from wheelrl.wbc import WBCDiagnostics

FloatArray = np.ndarray


class B2WZ1HierController:
    """Panel-compatible controller: RL residuals over the WBC servo."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        policy_path: Path,
        stats_path: Path,
        control_hz: float = 100.0,
        auto_drive: bool = True,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.data = data
        # The env owns the command bookkeeping and the WBC servo (100 Hz).
        self.env = B2WZ1HierEnv(model=model, data=data, randomization=0.0)
        self.wbc = self.env.wbc
        self.wbc.auto_drive = auto_drive
        self.control_dt = 1.0 / control_hz

        # Observation normalization from the training statistics.
        stats_venv = VecNormalize.load(
            stats_path,
            DummyVecEnv([lambda: B2WZ1HierEnv()]),
        )
        stats_venv.training = False
        stats_venv.norm_reward = False
        self._stats = stats_venv
        self.policy = PPO.load(policy_path, device=device)

        self._speed_profile = "normal"
        self._speed_scale = 1.0
        self._rotation_speed_scale = 1.0
        self._active = False
        self._last_action = np.zeros(6, dtype=np.float64)
        # Low-passed residual: the raw policy output jumps ~20 mm per 50 Hz
        # step (sensitive to obs noise), which makes the commanded target and
        # its marker visibly shake. Smoothing keeps the servo target calm
        # without changing the steady-state behavior.
        self._residual_smoothed = np.zeros(6, dtype=np.float64)
        self._residual_smoothing = 0.30

    # ------------------------------------------------------------ pose state
    @property
    def tcp_position(self) -> FloatArray:
        return self.data.site_xpos[self.wbc._tcp_site_id].copy()

    @property
    def tcp_rotation(self) -> FloatArray:
        return self.data.site_xmat[self.wbc._tcp_site_id].reshape(3, 3).copy()

    @property
    def target_position(self) -> FloatArray:
        return self.env._command_pos_world.copy()

    @property
    def target_rotation(self) -> FloatArray:
        return self.env._command_rotation_world.copy()

    @property
    def home_tcp_position(self) -> FloatArray:
        return self.wbc.home_tcp_position

    @property
    def home_tcp_rotation(self) -> FloatArray:
        return self.wbc.home_tcp_rotation

    @property
    def gripper_closed(self) -> bool:
        return self.wbc.gripper_closed

    @property
    def auto_drive(self) -> bool:
        return self.wbc.auto_drive

    @auto_drive.setter
    def auto_drive(self, value: bool) -> None:
        self.wbc.auto_drive = value

    @property
    def speed_profile(self) -> str:
        return self._speed_profile

    @property
    def speed_scale(self) -> float:
        return self._speed_scale

    @property
    def rotation_speed_scale(self) -> float:
        return self._rotation_speed_scale

    # ------------------------------------------------------------ commands
    def reset(self) -> None:
        self.env.reset(seed=7)
        self.capture_reference()

    def capture_reference(self) -> None:
        self.wbc.capture_reference()
        # The command starts at the settled TCP pose.
        self.env._command_pos_world[:] = self.wbc.home_tcp_position
        self.env._command_rotation_world[:] = self.wbc.home_tcp_rotation
        self.env._command_rpy.fill(0.0)
        self._active = True

    def set_target_pose(self, position: FloatArray, rotation: FloatArray) -> None:
        self.env._command_pos_world[:] = np.asarray(position, dtype=float)
        self.env._command_rotation_world[:] = np.asarray(rotation, dtype=float)
        self.env._command_rpy.fill(0.0)

    def set_home_offset(
        self,
        position_offset: FloatArray,
        rpy_offset_radians: FloatArray,
    ) -> None:
        self.env._command_pos_world[:] = (
            self.wbc.home_tcp_position + np.asarray(position_offset, dtype=float)
        )
        self.env._command_rotation_world[:] = (
            self.wbc.home_tcp_rotation
            @ rpy_rotation(np.asarray(rpy_offset_radians, dtype=float))
        )
        self.env._command_rpy.fill(0.0)

    def go_home(self) -> None:
        self.env._command_pos_world[:] = self.wbc.home_tcp_position
        self.env._command_rotation_world[:] = self.wbc.home_tcp_rotation

    def nudge_position(self, world_delta: FloatArray) -> None:
        self.env._command_pos_world += np.asarray(world_delta, dtype=float)

    def nudge_position_base(self, local_delta: FloatArray) -> None:
        delta = np.asarray(local_delta, dtype=float)
        base_rotation = self.data.xmat[self.wbc._base_body_id].reshape(3, 3)
        forward = base_rotation[:, 0].copy()
        forward[2] = 0.0
        forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
        left = np.array([-forward[1], forward[0], 0.0], dtype=float)
        self.nudge_position(
            delta[0] * forward
            + delta[1] * left
            + delta[2] * np.array([0.0, 0.0, 1.0], dtype=float)
        )

    def nudge_orientation(self, world_axis: FloatArray, angle: float) -> None:
        axis = np.asarray(world_axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        cross = np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float
        )
        rotation = (
            np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
        )
        self.env._command_rotation_world[:] = rotation @ self.env._command_rotation_world

    def set_gripper(self, *, closed: bool) -> None:
        self.wbc.set_gripper(closed=closed)

    def toggle_gripper(self) -> None:
        self.wbc.toggle_gripper()

    def set_speed_profile(self, profile: str) -> None:
        from wheelrl.wbc import ROTATION_SPEED_PROFILE_SCALES, SPEED_PROFILE_SCALES

        if profile not in SPEED_PROFILE_SCALES:
            raise ValueError(f"speed profile must be one of {SPEED_PROFILE_SCALES}")
        self._speed_profile = profile
        self._speed_scale = SPEED_PROFILE_SCALES[profile]
        self._rotation_speed_scale = ROTATION_SPEED_PROFILE_SCALES[profile]

    # --------------------------------------------------------------- control
    def step(self) -> WBCDiagnostics:
        """One 50 Hz policy step = one residual correction + two WBC steps."""
        observation = self.env._get_obs()
        normalized = self._stats.normalize_obs(np.asarray(observation)[None])[0]
        action, _ = self.policy.predict(normalized, deterministic=True)
        self._last_action[:] = np.asarray(action, dtype=float)
        self._residual_smoothed += self._residual_smoothing * (
            self._last_action - self._residual_smoothed
        )
        self.env.step(self._residual_smoothed)
        # The marker (mocap triad) shows the USER's commanded pose, which is
        # stable; the servo target (command + smoothed residual) is internal
        # and is refreshed at the next step anyway.
        self.wbc._target_position[:] = self.env._command_pos_world
        self.wbc._target_rotation[:] = self.env._command_rotation_world
        return self.diagnostics()

    def diagnostics(self) -> WBCDiagnostics:
        position_error = float(
            np.linalg.norm(
                self.env._command_pos_world - self.data.site_xpos[self.wbc._tcp_site_id]
            )
        )
        orientation_error = float(
            np.linalg.norm(
                rotation_vector(
                    self.env._command_rotation_world
                    @ self.data.site_xmat[self.wbc._tcp_site_id].reshape(3, 3).T
                )
            )
        )
        base_rotation = self.data.xmat[self.wbc._base_body_id].reshape(3, 3)
        return WBCDiagnostics(
            position_error=position_error,
            orientation_error=orientation_error,
            base_position_error=0.0,
            upright=float(base_rotation[2, 2]),
            max_torque=float(np.max(np.abs(self.wbc._last_torque))),
            mobile_base_active=self.wbc._mobile_base_active,
            base_goal_distance=self.wbc._base_goal_distance,
        )

    def close(self) -> None:
        self._stats.close()
