"""RL base coordination over an exact-pose analytic whole-body controller.

The policy never edits the user TCP target. It outputs only a three element
macro correction -- base forward velocity, yaw rate and height offset -- while the
100 Hz analytic WBC continues to servo the exact TCP pose. A deterministic,
smooth reachability gate disables the learned command for ordinary arm-local
targets, so learning cannot degrade the WBC's millimetre-class local tracking.
"""

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from wheelrl.envs.b2w_z1 import ARM_NOMINAL, GRIPPER_NOMINAL
from wheelrl.envs.b2w_z1_track import rotation_vector, rpy_rotation
from wheelrl.wbc import (
    WBC_MODEL_PATH,
    WBC_STAND_LEG_NOMINAL,
    B2WZ1WholeBodyController,
)

FloatArray = NDArray[np.float64]
CoordinationStage = Literal["local", "forward", "lateral", "full"]

COMMAND_ORI_MAX = float(np.deg2rad(25.0))
SETTLE_STEPS = 30
SUCCESS_POSITION = 0.010
SUCCESS_ORIENTATION = float(np.deg2rad(3.0))
SUCCESS_HOLD_STEPS = 25
ACTION_SCALE = np.array([0.20, 0.35, 0.05], dtype=np.float64)

# Arm-comfort region around the captured home TCP, expressed in the current
# base frame. Outside it the gate ramps smoothly over REACH_RAMP.
REACH_LOW = np.array([-0.12, -0.11, -0.14], dtype=np.float64)
REACH_HIGH = np.array([0.16, 0.11, 0.14], dtype=np.float64)
REACH_RAMP = np.array([0.18, 0.18, 0.14], dtype=np.float64)


def _smoothstep(value: float) -> float:
    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


class B2WZ1HierEnv(gym.Env[np.ndarray, np.ndarray]):
    """Train a learned macro base coordinator around the analytic WBC."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 750,
        tracking_curriculum: float = 1.0,
        randomization: float = 1.0,
        action_penalty: float = 0.03,
        arm_first: bool = True,
        coordination_stage: CoordinationStage = "full",
        coordinator_enabled: bool = True,
        model: mujoco.MjModel | None = None,
        data: mujoco.MjData | None = None,
    ) -> None:
        if coordination_stage not in {"local", "forward", "lateral", "full"}:
            raise ValueError(
                "coordination_stage must be local, forward, lateral, or full"
            )
        self.render_mode = render_mode
        self.max_episode_steps = int(max_episode_steps)
        self.tracking_curriculum = float(np.clip(tracking_curriculum, 0.0, 1.0))
        self.randomization = float(np.clip(randomization, 0.0, 1.0))
        self.action_penalty = float(np.clip(action_penalty, 0.0, 5.0))
        self.arm_first = bool(arm_first)
        self.coordination_stage: CoordinationStage = coordination_stage
        self.coordinator_enabled = bool(coordinator_enabled)

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
        self.dt = 0.02
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(78,), dtype=np.float32
        )

        self._base_body_id = self.wbc._base_body_id
        self._ee_site_id = self.wbc._tcp_site_id
        self._command_pos_world = np.zeros(3, dtype=np.float64)
        self._command_rotation_world = np.eye(3, dtype=np.float64)
        self._last_action = np.zeros(3, dtype=np.float64)
        self._coordination_gate = 0.0
        self._raw_coordination_gate = 0.0
        self._step_count = 0
        self._success_hold = 0
        self._previous_pose_metric = 0.0
        self._target_category = coordination_stage
        self._action_delay = 0
        self._action_history: list[FloatArray] = []
        self._observation_noise = 0.0
        self._viewer: Any | None = None
        self._renderer: Any | None = None

        # Randomization is restored from these immutable nominal values on
        # every reset, preventing multiplicative drift between episodes.
        self._nominal_body_mass = self.model.body_mass.copy()
        self._nominal_body_inertia = self.model.body_inertia.copy()
        self._nominal_geom_friction = self.model.geom_friction.copy()
        self._nominal_dof_damping = self.model.dof_damping.copy()

    # ---------------------------------------------------------------- task
    def set_coordination_stage(self, stage: CoordinationStage) -> None:
        if stage not in {"local", "forward", "lateral", "full"}:
            raise ValueError("unknown coordination stage")
        self.coordination_stage = stage

    def set_coordinator_enabled(self, enabled: bool) -> None:
        """Enable RL base commands or run the unmodified WBC baseline."""
        self.coordinator_enabled = bool(enabled)
        if not self.coordinator_enabled:
            self.wbc.clear_base_coordinator_command()

    def set_command_pose(self, position: FloatArray, rotation: FloatArray) -> None:
        """Set the immutable world-frame pose reference used by reward and WBC."""
        position_array = np.asarray(position, dtype=np.float64)
        rotation_array = np.asarray(rotation, dtype=np.float64)
        if position_array.shape != (3,) or rotation_array.shape != (3, 3):
            raise ValueError("position must be (3,) and rotation must be (3, 3)")
        self._command_pos_world[:] = position_array
        self._command_rotation_world[:] = rotation_array
        self.wbc.set_target_pose(position_array, rotation_array)
        if self.wbc._active:
            raw_gate, _ = self._compute_reachability_gate()
            self._raw_coordination_gate = raw_gate
            self._coordination_gate = raw_gate

    def _base_rotation(self) -> FloatArray:
        return self.data.xmat[self._base_body_id].reshape(3, 3).copy()

    def _command_error(self) -> FloatArray:
        position_error = self._command_pos_world - self.wbc.tcp_position
        orientation_error = rotation_vector(
            self._command_rotation_world @ self.wbc.tcp_rotation.T
        )
        return np.concatenate([position_error, orientation_error])

    def _arm_quality(self) -> tuple[FloatArray, float, float]:
        q = self.data.qpos[self.wbc._arm_qpos_adr]
        low = self.wbc._arm_joint_ranges[:, 0]
        high = self.wbc._arm_joint_ranges[:, 1]
        half_span = 0.5 * (high - low)
        margins = np.ones(6, dtype=np.float64)
        movable = half_span > 1.0e-6
        margins[movable] = np.minimum(
            q[movable] - low[movable], high[movable] - q[movable]
        ) / half_span[movable]
        margins = np.clip(margins, 0.0, 1.0)

        arm_jacobian = self.wbc._site_jacobian(self._ee_site_id)[
            :3, self.wbc._arm_dof_adr
        ]
        singular_values = np.linalg.svd(arm_jacobian, compute_uv=False)
        manipulability = float(np.min(singular_values))
        return margins, float(np.min(margins)), manipulability

    def _compute_reachability_gate(self) -> tuple[float, dict[str, float]]:
        base_rotation = self._base_rotation()
        target_in_base = base_rotation.T @ (
            self._command_pos_world - self.data.xpos[self._base_body_id]
        )
        delta = target_in_base - self.wbc._home_tcp_position_base
        outside = np.maximum(np.maximum(REACH_LOW - delta, delta - REACH_HIGH), 0.0)
        workspace_gate = _smoothstep(float(np.max(outside / REACH_RAMP)))

        margins, minimum_margin, manipulability = self._arm_quality()
        error = self._command_error()
        position_error = float(np.linalg.norm(error[:3]))
        orientation_error = float(np.linalg.norm(error[3:]))
        dwell = _smoothstep((self._step_count - 15) / 35.0)
        unresolved = _smoothstep((position_error - 0.025) / 0.075)
        joint_gate = dwell * unresolved * _smoothstep((0.08 - minimum_margin) / 0.06)
        manipulation_gate = (
            dwell
            * unresolved
            * _smoothstep((0.015 - manipulability) / 0.012)
        )
        # The model has one locked Z1 joint. If position is already close but
        # orientation remains stuck, base yaw supplies the missing macro DOF.
        orientation_stall_gate = (
            dwell
            * _smoothstep((orientation_error - 0.12) / 0.25)
            * _smoothstep((0.07 - position_error) / 0.05)
        )
        raw_gate = float(
            np.clip(
                max(
                    workspace_gate,
                    joint_gate,
                    manipulation_gate,
                    orientation_stall_gate,
                ),
                0.0,
                1.0,
            )
        )
        return raw_gate, {
            "workspace_gate": workspace_gate,
            "joint_gate": joint_gate,
            "manipulation_gate": manipulation_gate,
            "orientation_stall_gate": orientation_stall_gate,
            "joint_limit_margin": minimum_margin,
            "manipulability": manipulability,
            "arm_margin_mean": float(np.mean(margins)),
        }

    def _sample_target(self) -> tuple[FloatArray, FloatArray, str]:
        stage = self.coordination_stage
        category = stage
        if stage == "full":
            category = str(
                self.np_random.choice(
                    np.array(["local", "forward", "lateral"]),
                    p=np.array([0.20, 0.40, 0.40]),
                )
            )

        if category == "local":
            low = np.array([-0.07, -0.10, -0.06])
            high = np.array([0.07, 0.10, 0.08])
            offset = self.np_random.uniform(low, high)
        elif category == "forward":
            direction = -1.0 if self.np_random.random() < 0.20 else 1.0
            x = direction * self.np_random.uniform(0.20, 0.75)
            offset = np.array(
                [
                    x,
                    self.np_random.uniform(-0.14, 0.14),
                    self.np_random.uniform(-0.10, 0.12),
                ]
            )
        else:
            side = -1.0 if self.np_random.random() < 0.5 else 1.0
            offset = np.array(
                [
                    self.np_random.uniform(-0.15, 0.45),
                    side * self.np_random.uniform(0.18, 0.55),
                    self.np_random.uniform(-0.10, 0.12),
                ]
            )

        offset *= self.tracking_curriculum
        position = self.wbc.home_tcp_position + self._base_rotation() @ offset
        position[2] = np.clip(position[2], 0.55, 1.45)
        rpy = self.np_random.uniform(-COMMAND_ORI_MAX, COMMAND_ORI_MAX, size=3)
        rpy *= self.tracking_curriculum
        base_rotation = self._base_rotation()
        rotation = (
            base_rotation
            @ rpy_rotation(rpy)
            @ base_rotation.T
            @ self.wbc.home_tcp_rotation
        )
        return position, rotation, category

    # ------------------------------------------------------- randomization
    def _restore_nominal_dynamics(self) -> None:
        self.model.body_mass[:] = self._nominal_body_mass
        self.model.body_inertia[:] = self._nominal_body_inertia
        self.model.geom_friction[:] = self._nominal_geom_friction
        self.model.dof_damping[:] = self._nominal_dof_damping

    def _apply_domain_randomization(self) -> dict[str, float]:
        scale = self.randomization
        self._restore_nominal_dynamics()
        if scale > 0.0:
            body_scale = self.np_random.uniform(
                1.0 - 0.15 * scale, 1.0 + 0.15 * scale, self.model.nbody
            )
            body_scale[0] = 1.0
            self.model.body_mass[:] *= body_scale
            self.model.body_inertia[:] *= body_scale[:, None]
            friction_scale = self.np_random.uniform(
                1.0 - 0.30 * scale, 1.0 + 0.30 * scale, self.model.ngeom
            )
            self.model.geom_friction[:] *= friction_scale[:, None]
            damping_scale = float(
                self.np_random.uniform(1.0 - 0.12 * scale, 1.0 + 0.12 * scale)
            )
            self.model.dof_damping[:] *= damping_scale
            gain_scale = float(
                self.np_random.uniform(1.0 - 0.12 * scale, 1.0 + 0.12 * scale)
            )
        else:
            damping_scale = 1.0
            gain_scale = 1.0

        self.wbc.set_actuator_gain_scale(gain_scale)
        self._observation_noise = 0.003 * scale
        max_delay = int(round(2.0 * scale))
        self._action_delay = (
            int(self.np_random.integers(0, max_delay + 1)) if max_delay else 0
        )
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        massive = self._nominal_body_mass > 1.0e-9
        frictional = self._nominal_geom_friction[:, 0] > 1.0e-9
        return {
            "mass_scale_mean": float(
                np.mean(self.model.body_mass[massive] / self._nominal_body_mass[massive])
            ),
            "friction_scale_mean": float(
                np.mean(
                    self.model.geom_friction[frictional, 0]
                    / self._nominal_geom_friction[frictional, 0]
                )
            ),
            "damping_scale": damping_scale,
            "gain_scale": gain_scale,
            "action_delay": float(self._action_delay),
        }

    # -------------------------------------------------------------- action
    def _apply_action(self, action: FloatArray) -> FloatArray:
        clipped = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if clipped.shape != (3,):
            raise ValueError("action must have shape (3,)")
        self._action_history.append(clipped.copy())
        applied = self._action_history.pop(0)
        if self.coordinator_enabled and self._coordination_gate > 1.0e-3:
            self.wbc.set_base_coordinator_command(
                ACTION_SCALE * applied,
                gate=self._coordination_gate,
            )
        else:
            self.wbc.clear_base_coordinator_command()
            applied = np.zeros(3, dtype=np.float64)
        return applied

    def _stability_terms(self) -> dict[str, float]:
        upright = float(np.clip(self._base_rotation()[2, 2], -1.0, 1.0))
        height = float(self.data.qpos[2])
        return {
            "upright": max(0.0, upright),
            "tilt_penalty": -0.80 * max(0.0, 0.82 - upright),
            "height_penalty": -0.60 * max(0.0, abs(height - 0.62) - 0.12),
        }

    # ---------------------------------------------------------------- API
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._restore_nominal_dynamics()
        self.wbc.reset()
        randomization_info = self._apply_domain_randomization()
        for _ in range(SETTLE_STEPS):
            self.wbc.step()
        self.wbc.capture_reference()

        position, rotation, category = self._sample_target()
        self.set_command_pose(position, rotation)
        self._last_action.fill(0.0)
        self._step_count = 0
        self._success_hold = 0
        self._target_category = category
        self._action_history = [
            np.zeros(3, dtype=np.float64) for _ in range(self._action_delay)
        ]
        self._raw_coordination_gate, gate_info = self._compute_reachability_gate()
        self._coordination_gate = self._raw_coordination_gate
        error = self._command_error()
        self._previous_pose_metric = float(
            np.linalg.norm(error[:3]) + 0.25 * np.linalg.norm(error[3:])
        )
        info: dict[str, Any] = {
            "command_pos": self._command_pos_world.copy(),
            "target_category": category,
            "coordination_gate": self._coordination_gate,
            **gate_info,
            **randomization_info,
        }
        return self._get_obs(), info

    def step(
        self,
        action: NDArray[np.floating[Any]],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        applied_action = self._apply_action(np.asarray(action, dtype=np.float64))
        for _ in range(2):
            self.wbc.step()
        self._step_count += 1

        error = self._command_error()
        position_error = float(np.linalg.norm(error[:3]))
        orientation_error = float(np.linalg.norm(error[3:]))
        pose_metric = position_error + 0.25 * orientation_error
        progress = float(
            np.clip(self._previous_pose_metric - pose_metric, -0.05, 0.05)
        )
        self._previous_pose_metric = pose_metric

        raw_gate, gate_info = self._compute_reachability_gate()
        self._raw_coordination_gate = raw_gate
        gate_alpha = 0.30 if raw_gate > self._coordination_gate else 0.12
        self._coordination_gate += gate_alpha * (raw_gate - self._coordination_gate)
        if raw_gate == 0.0 and self._coordination_gate < 1.0e-3:
            self._coordination_gate = 0.0

        success_now = bool(
            position_error < SUCCESS_POSITION
            and orientation_error < SUCCESS_ORIENTATION
        )
        self._success_hold = self._success_hold + 1 if success_now else 0
        success = self._success_hold >= SUCCESS_HOLD_STEPS

        terms = {
            "position_tracking": float(np.exp(-10.0 * position_error)),
            "orientation_tracking": float(np.exp(-3.0 * orientation_error)),
            "progress": 12.0 * progress,
        }
        terms.update(self._stability_terms())
        control_velocity = np.concatenate(
            [
                self.data.qvel[self.wbc._leg_dof_adr],
                self.data.qvel[self.wbc._wheel_dof_adr],
                self.data.qvel[self.wbc._arm_dof_adr],
                self.data.qvel[[self.wbc._gripper_dof_adr]],
            ]
        )
        terms["energy_penalty"] = -1.0e-5 * float(
            np.sum(np.abs(self.wbc._last_torque * control_velocity))
        )
        terms["action_rate_penalty"] = -0.01 * self._coordination_gate * float(
            np.mean(np.square(applied_action - self._last_action))
        )
        terms["action_magnitude_penalty"] = (
            -self.action_penalty
            * self._coordination_gate
            * float(np.mean(np.square(applied_action)))
        )
        reward = (
            1.4 * terms["position_tracking"]
            + 0.5 * terms["orientation_tracking"]
            + terms["progress"]
            + 0.20 * terms["upright"]
            + terms["tilt_penalty"]
            + terms["height_penalty"]
            + terms["energy_penalty"]
            + terms["action_rate_penalty"]
            + terms["action_magnitude_penalty"]
        )
        upright = float(np.clip(self._base_rotation()[2, 2], -1.0, 1.0))
        healthy = bool(
            np.isfinite(self.data.qpos).all()
            and 0.20 < self.data.qpos[2] < 1.30
            and upright > 0.35
        )
        fallen = not healthy
        terminated = fallen or success
        if fallen:
            reward -= 20.0
        elif success:
            reward += 25.0
        truncated = self._step_count >= self.max_episode_steps
        self._last_action[:] = applied_action

        if fallen:
            termination_reason = "fallen_or_non_finite"
        elif success:
            termination_reason = "success"
        elif truncated:
            termination_reason = "time_limit"
        else:
            termination_reason = "running"
        diagnostics = self.wbc.diagnostics()
        info = {
            "reward_terms": terms,
            "command_pos_error": position_error,
            "command_ori_error": orientation_error,
            "upright": upright,
            "success": success,
            "success_hold": self._success_hold,
            "target_category": self._target_category,
            "coordination_gate": self._coordination_gate,
            "raw_coordination_gate": raw_gate,
            "applied_action": applied_action.copy(),
            "base_goal_distance": diagnostics.base_goal_distance,
            "solver_residual": diagnostics.solver_residual,
            "termination_reason": termination_reason,
            **gate_info,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        wbc = self.wbc
        rotation = self._base_rotation()
        qpos = self.data.qpos
        qvel = self.data.qvel
        base_linear_velocity = rotation.T @ qvel[:3]
        base_angular_velocity = rotation.T @ qvel[3:6]
        projected_gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        joint_positions = np.concatenate(
            [
                qpos[wbc._leg_qpos_adr] - WBC_STAND_LEG_NOMINAL,
                qpos[wbc._arm_qpos_adr] - ARM_NOMINAL,
                qpos[[wbc._gripper_qpos_adr]] - GRIPPER_NOMINAL,
            ]
        )
        control_velocity = np.concatenate(
            [
                qvel[wbc._leg_dof_adr],
                qvel[wbc._wheel_dof_adr],
                qvel[wbc._arm_dof_adr],
                qvel[[wbc._gripper_dof_adr]],
            ]
        )
        target_in_base = rotation.T @ (
            self._command_pos_world - self.data.xpos[self._base_body_id]
        )
        error = self._command_error()
        position_error_base = rotation.T @ error[:3]
        orientation_error_base = rotation.T @ error[3:]
        arm_margins, _, manipulability = self._arm_quality()
        wheel_slip = 0.1 * (
            wbc._wheel_velocity_target - qvel[wbc._wheel_dof_adr]
        )
        observation = np.concatenate(
            [
                np.array([qpos[2]], dtype=np.float64),
                base_linear_velocity,
                base_angular_velocity,
                projected_gravity,
                joint_positions,
                0.1 * control_velocity,
                target_in_base,
                position_error_base,
                orientation_error_base,
                arm_margins,
                np.array(
                    [
                        manipulability,
                        wbc._base_goal_distance,
                        self._coordination_gate,
                        float(wbc._mobile_base_active),
                    ],
                    dtype=np.float64,
                ),
                wheel_slip,
                self._last_action,
            ]
        )
        if observation.shape != (78,):
            raise RuntimeError(f"hierarchical observation has shape {observation.shape}")
        if self._observation_noise > 0.0:
            observation = observation + self.np_random.normal(
                0.0, self._observation_noise, size=observation.shape
            )
        return observation.astype(np.float32)

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
