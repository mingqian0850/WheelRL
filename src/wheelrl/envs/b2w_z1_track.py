"""Task-space whole-body pose tracking for the B2-W + Z1.

RFM-inspired (`Learning Whole-Body Loco-Manipulation for Omni-Directional
Task Space Pose Tracking with a Wheeled-Quadrupedal-Manipulator`, RA-L 2024):
the policy must drive the whole body -- wheels, legs and arm -- so the Z1
tool center point (TCP) tracks a *moving* 6-DoF pose target in the base
frame.  The target can translate (position) and rotate (orientation), and a
curriculum scales the initial offset, orientation error, and target speed.

Differences from :class:`B2WZ1Env`:

- the observation is 86-dimensional: the 80-dim base observation plus the
  TCP orientation error (rotation vector, 3) and the target velocity in the
  base frame (3);
- the reward adds an orientation-tracking term and a multiplicative
  "sync gate" (a light RFM-style nonlinear fusion) so position and
  orientation must be satisfied together, not additively traded off;
- the EE target can move during the episode; its motion is reflected at the
  reachable-workspace bounds so the target stays in the arm/base workspace;
- the base velocity command is disabled (``command_curriculum=0``): forward
  motion emerges from the wheel actions the policy must learn to coordinate
  with the arm.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from wheelrl.envs.b2w_z1 import B2WZ1Env

FloatArray = NDArray[np.float64]

# Reachable TCP position bounds in the base frame [m], per axis (low, high).
# The Z1 workspace is wide (measured grid: x in [-0.63, 1.09], y in [-0.56,
# 0.56], z in [-0.26, 1.09]); the tracking target is confined to the working
# zone around the nominal EE pose [0.64, 0.0, 0.67] so the arm stays in a
# comfortable region and the base is not forced to over-drive.
EE_POS_BOUNDS = np.array(
    [[0.28, 0.98], [-0.45, 0.45], [0.25, 1.00]], dtype=np.float64
)
# Maximum per-axis orientation offset [rad] at full curriculum.
ORI_MAX = float(np.deg2rad(25.0))
# Maximum target speed [m/s] at full curriculum.
TARGET_VMAX = 0.10
# Position-offset magnitudes at full curriculum (matches B2WZ1Env).
POS_OFFSET_LOW = np.array([-0.07, -0.10, -0.06], dtype=np.float64)
POS_OFFSET_HIGH = np.array([0.07, 0.10, 0.08], dtype=np.float64)

# --- domain randomization and posture quality (randomization >= 0) ----------
# Initial arm-pose randomization: sampled UNIFORMLY over the full joint
# ranges so the policy must stabilize from stretched, folded and
# near-singular postures - not only around the nominal pose.
# Dynamics randomization strengths at randomization=1.0.
MASS_RANGE = 0.15      # per-body mass/inertia scale
FRICTION_RANGE = 0.30  # floor friction
PD_RANGE = 0.10        # overall PD gain scale
# Posture-quality shaping: stay away from joint limits and low
# manipulability (arm Jacobian singular values), where kinematic controllers
# degrade.
JOINT_MARGIN_MIN = 0.20  # rad, penalty starts below this distance to a limit
JOINT_MARGIN_WEIGHT = 2.0
MANIP_MIN = 0.15         # smallest arm-Jacobian singular value threshold
MANIP_WEIGHT = 0.05


def rotation_vector(rotation: FloatArray) -> FloatArray:
    """SO(3) logarithm of a rotation matrix (axis * angle)."""
    cosine = float(np.clip(0.5 * (np.trace(rotation) - 1.0), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1.0e-8:
        return np.zeros(3, dtype=np.float64)
    skew = 0.5 * np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    return skew * (angle / np.sin(angle))


def rpy_rotation(rpy: FloatArray) -> FloatArray:
    """Rz(yaw) @ Ry(pitch) @ Rx(roll), matching the rest of WheelRL."""
    roll, pitch, yaw = rpy
    rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ry = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ]
    )
    return rz @ ry @ rx


class B2WZ1TrackEnv(B2WZ1Env):
    """Whole-body task-space tracking of a moving 6-DoF TCP target."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
        tracking_curriculum: float = 1.0,
        target_motion: float = 0.0,
        randomization: float = 0.0,
        orientation_weight: float = 0.6,
        _model_filename: str = "scene.xml",
    ) -> None:
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            command_curriculum=0.0,
            _model_filename=_model_filename,
        )
        self.tracking_curriculum = float(np.clip(tracking_curriculum, 0.0, 1.0))
        self.target_motion = float(np.clip(target_motion, 0.0, 1.0))
        self.randomization = float(np.clip(randomization, 0.0, 1.0))
        self.orientation_weight = float(orientation_weight)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(86,), dtype=np.float32
        )

        self._ee_target_rpy = np.zeros(3, dtype=np.float64)
        self._ee_target_rotation = np.eye(3)
        self._ee_target_vel = np.zeros(3, dtype=np.float64)
        # Low-pass filtered actual target velocity (smooth start and bounces).
        self._ee_vel = np.zeros(3, dtype=np.float64)

        self._wheel_body_ids = {
            self._id(mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("FL_wheel_link", "FR_wheel_link", "RL_wheel_link", "RR_wheel_link")
        }
        self._floor_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Domain-randomization baselines (per-env model copies).
        self._nominal_body_mass = self.model.body_mass.copy()
        self._nominal_body_inertia = self.model.body_inertia.copy()
        self._nominal_floor_friction = self.model.geom_friction[self._floor_geom_id].copy()
        self._pd_scale = 1.0
        self._arm_ranges = self.model.jnt_range[self._joint_ids[16:22]].copy()

    # ------------------------------------------------------------ task space
    def _current_ee_rotation_base(self) -> FloatArray:
        rotation = self._base_rotation()
        ee_world = self.data.site_xmat[self._ee_site_id].reshape(3, 3)
        return rotation.T @ ee_world

    def _ee_orientation_error(self) -> FloatArray:
        """Rotation vector that takes the current TCP orientation to the target."""
        current = self._current_ee_rotation_base()
        return rotation_vector(self._ee_target_rotation @ current.T)

    def _advance_target(self) -> None:
        if self.target_motion <= 0.0:
            return
        # Low-pass the commanded velocity: the target starts smoothly and
        # direction changes at the workspace bounds do not produce steps.
        self._ee_vel += 0.05 * (self._ee_target_vel - self._ee_vel)
        self._ee_target_base += self._ee_vel * self.dt
        for axis in range(3):
            if self._ee_target_base[axis] < EE_POS_BOUNDS[axis, 0]:
                self._ee_target_base[axis] = EE_POS_BOUNDS[axis, 0]
                self._ee_target_vel[axis] = abs(self._ee_target_vel[axis])
            elif self._ee_target_base[axis] > EE_POS_BOUNDS[axis, 1]:
                self._ee_target_base[axis] = EE_POS_BOUNDS[axis, 1]
                self._ee_target_vel[axis] = -abs(self._ee_target_vel[axis])

    def _wheels_in_contact(self) -> int:
        """Number of wheels currently touching the floor (0..4)."""
        wheels: set[int] = set()
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if contact.dist > 0.0:
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            if geom1 == self._floor_geom_id and body2 in self._wheel_body_ids:
                wheels.add(body2)
            elif geom2 == self._floor_geom_id and body1 in self._wheel_body_ids:
                wheels.add(body1)
        return len(wheels)

    def _is_healthy(self) -> bool:
        """Track-env health check: height is deliberately permissive.

        Squatting to lower the center of mass during fast arm/wheel motions is
        a legitimate whole-body strategy for this task, so the episode only
        ends on a real failure: tipping (upright), non-finite state, or an
        extreme height that indicates collapse or a jump. The height-band
        reward shapes the preferred stance instead.
        """
        rotation = self._base_rotation()
        height = float(self.data.qpos[2])
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        return bool(finite and 0.20 < height < 1.30 and rotation[2, 2] > 0.35)

    # ---------------------------------------------------------------- env API
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)

        if self.randomization > 0.0:
            scale = self.randomization
            # Dynamics domain randomization: per-body mass/inertia, floor
            # friction, and overall PD gain scale.
            mass_scale = 1.0 + self.np_random.uniform(
                -MASS_RANGE, MASS_RANGE, size=self.model.nbody
            ) * scale
            self.model.body_mass[:] = self._nominal_body_mass * mass_scale
            self.model.body_inertia[:] = self._nominal_body_inertia * mass_scale[:, None]
            self.model.geom_friction[self._floor_geom_id] = self._nominal_floor_friction * (
                1.0 + self.np_random.uniform(-FRICTION_RANGE, FRICTION_RANGE) * scale
            )
            self._pd_scale = float(
                1.0 + self.np_random.uniform(-PD_RANGE, PD_RANGE) * scale
            )
            # Initial arm-pose randomization over the FULL joint ranges: the
            # policy must stabilize from stretched/folded/singular postures.
            initial_arm = self.np_random.uniform(
                self._arm_ranges[:, 0], self._arm_ranges[:, 1]
            )
            self.data.qpos[self._qpos_adr[16:22]] = initial_arm
            mujoco.mj_forward(self.model, self.data)

        scale = self.tracking_curriculum
        self._ee_target_rpy = self.np_random.uniform(-ORI_MAX, ORI_MAX, size=3) * scale
        # The target orientation is the current TCP orientation plus the sampled
        # RPY offset, so a zero offset means "hold the current orientation".
        self._ee_target_rotation = rpy_rotation(self._ee_target_rpy) @ (
            self._current_ee_rotation_base()
        )
        # The parent sampled a zero offset (command_curriculum=0); re-sample the
        # position target with our own curriculum and keep it in the box.
        self._ee_target_base = self._current_ee_base() + self.np_random.uniform(
            POS_OFFSET_LOW * scale, POS_OFFSET_HIGH * scale
        )
        self._ee_target_base = np.clip(
            self._ee_target_base, EE_POS_BOUNDS[:, 0], EE_POS_BOUNDS[:, 1]
        )

        vmax = TARGET_VMAX * self.target_motion
        velocity = self.np_random.uniform(-vmax, vmax, size=3)
        if vmax > 0.0 and np.max(np.abs(velocity)) < 0.01 * vmax:
            velocity[:2] = self.np_random.uniform(0.3 * vmax, vmax, size=2) * self.np_random.choice(
                [-1.0, 1.0], size=2
            )
        self._ee_target_vel = velocity
        self._ee_vel.fill(0.0)

        # Rebuild the observation now that the target is fully defined.
        observation = self._get_obs()
        info["ee_orientation_error"] = float(np.linalg.norm(self._ee_orientation_error()))
        info["target_speed"] = float(np.linalg.norm(self._ee_vel))
        return observation, info

    def step(
        self,
        action: NDArray[np.floating[Any]],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._advance_target()
        observation, reward, terminated, truncated, info = super().step(action)
        if terminated:
            reward -= 20.0
            info["reward_terms"]["fall_penalty"] = -20.0
        info["ee_orientation_error"] = float(np.linalg.norm(self._ee_orientation_error()))
        info["target_speed"] = float(np.linalg.norm(self._ee_vel))
        return observation, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        base_observation = super()._get_obs()
        extended = np.concatenate(
            [
                base_observation,
                self._ee_orientation_error(),
                self._ee_vel,
            ]
        )
        return extended.astype(np.float32)

    def _compute_torque(self, action: FloatArray) -> FloatArray:
        torque = super()._compute_torque(action)
        if self._pd_scale != 1.0:
            torque = torque * self._pd_scale
        return torque

    def _reward(
        self,
        action: FloatArray,
        torque: FloatArray,
    ) -> tuple[float, dict[str, float]]:
        rotation = self._base_rotation()
        local_velocity = rotation.T @ self.data.qvel[:3]
        local_angular_velocity = rotation.T @ self.data.qvel[3:6]
        upright = float(np.clip(rotation[2, 2], -1.0, 1.0))
        height_error = self.data.qpos[2] - 0.62
        position_error = float(np.linalg.norm(self._ee_target_base - self._current_ee_base()))
        orientation_error = float(np.linalg.norm(self._ee_orientation_error()))
        wheels_off = 4 - self._wheels_in_contact()
        # Stability gate: tracking rewards vanish as the robot tilts toward a
        # fall, so chasing the target can never be worth falling over.
        stability_gate = float(np.clip((upright - 0.45) / 0.45, 0.0, 1.0))
        # Height band: squatting/rising beyond +/-0.12 m around the natural
        # height costs reward linearly, so crouching for stability is bounded.
        height_band = max(0.0, abs(height_error) - 0.12)
        # Posture quality: stay away from joint limits and from arm-Jacobian
        # singularities, where kinematic controllers degrade.
        arm_qpos = self.data.qpos[self._qpos_adr[16:22]]
        margin = np.minimum(
            arm_qpos - self._arm_ranges[:, 0], self._arm_ranges[:, 1] - arm_qpos
        )
        joint_margin = float(np.sum(np.maximum(0.0, JOINT_MARGIN_MIN - margin) ** 2))
        arm_jacobian = np.zeros((6, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self.model, self.data, arm_jacobian[:3], arm_jacobian[3:], self._ee_site_id
        )
        arm_jacobian = arm_jacobian[:, self._dof_adr[16:22]]
        sigma_min = float(np.linalg.svd(arm_jacobian, compute_uv=False)[-1])
        manipulability = max(0.0, MANIP_MIN - sigma_min)

        terms = {
            "ee_position": float(np.exp(-18.0 * position_error**2)) * stability_gate,
            "ee_orientation": float(np.exp(-25.0 * orientation_error**2)) * stability_gate,
            # RFM-inspired multiplicative gate: both pose parts must be small.
            "sync_gate": float(
                np.exp(-8.0 * position_error**2) * np.exp(-8.0 * orientation_error**2)
            )
            * stability_gate,
            "stability_gate": stability_gate,
            "upright": max(0.0, upright),
            # Continuous stability shaping: tilting, pitching/rolling and wheel
            # lift are penalized every step, not only at termination, so the
            # policy cannot learn to chase targets by driving itself over.
            "tilt_penalty": -0.60 * max(0.0, 0.80 - upright),
            "pitch_rate_penalty": -0.02 * float(np.sum(local_angular_velocity[:2] ** 2)),
            "wheel_lift_penalty": -0.25 * float(wheels_off),
            "height": float(np.exp(-12.0 * height_error**2)),
            "height_band_penalty": -1.00 * float(height_band),
            "joint_margin_penalty": -JOINT_MARGIN_WEIGHT * joint_margin,
            "manipulability_penalty": -MANIP_WEIGHT * manipulability,
            "alive": 1.0,
            "energy_penalty": -2.0e-5
            * float(np.sum(np.abs(torque * self.data.qvel[self._dof_adr]))),
            "action_rate_penalty": -0.01 * float(np.mean(np.square(action - self._last_action))),
            "lateral_penalty": -0.05 * float(local_velocity[1] ** 2),
        }
        reward = (
            1.00 * terms["ee_position"]
            + self.orientation_weight * terms["ee_orientation"]
            + 0.50 * terms["sync_gate"]
            + 0.25 * terms["upright"]
            + terms["tilt_penalty"]
            + terms["pitch_rate_penalty"]
            + terms["wheel_lift_penalty"]
            + terms["height_band_penalty"]
            + terms["joint_margin_penalty"]
            + terms["manipulability_penalty"]
            + 0.15 * terms["height"]
            + 0.05 * terms["alive"]
            + terms["energy_penalty"]
            + terms["action_rate_penalty"]
            + terms["lateral_penalty"]
        )
        return float(reward), terms
