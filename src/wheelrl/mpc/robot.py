"""B2-W + Z1 robot model for the whole-body MPC (Pinocchio)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin

from wheelrl.envs.b2w_z1 import (
    ARM_JOINTS,
    ARM_NOMINAL,
    LEG_NOMINAL,
    LEG_JOINTS,
    TORQUE_LIMITS,
    WHEEL_JOINTS,
)

URDF_PATH = (
    Path(__file__).resolve().parents[1] / "assets" / "b2w_z1" / "mpc" / "b2w_z1.urdf"
)

# MuJoCo joint names in the order they appear in qpos (after the free joint).
# The URDF uses FL/FR/RL/RR order while MuJoCo uses FR/FL/RR/RL; we map by name.
URDF_JOINT_NAMES = tuple(
    [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FL_foot_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FR_foot_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RL_foot_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RR_foot_joint",
        *[f"arm_joint{i}" for i in range(1, 7)],
    ]
)

WHEEL_RADIUS = 0.10  # [m], matches WheelRL assets


class B2WZ1:
    """Pinocchio model of the wheel-legged B2-W + Z1 for MPC use."""

    def __init__(self, urdf_path: Path = URDF_PATH, reference_pose: str | None = None):
        self.urdf_path = urdf_path
        # buildModelFromUrdf only parses dynamics (inertia/joints) and does not
        # require mesh files, which keeps the MPC lightweight.
        self.model = pin.buildModelFromUrdf(
            str(urdf_path), pin.JointModelFreeFlyer()
        )
        self.data = self.model.createData()
        # The aCodeDog URDF has no SRDF; the nominal pose is set to match the
        # WheelRL MuJoCo scene so MPC and MuJoCo share the same reference.
        self.q0 = self._default_q0()

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nj = self.nq - 7
        self.mass = sum(
            self.model.inertias[i].mass for i in range(self.model.njoints)
        )
        # forces at the four wheel contact points + the arm end-effector
        self.nf = 3 * len(("FL_contact", "FR_contact", "RL_contact", "RR_contact")) + 3

        # Contact frames (wheel bottoms) added to the URDF.
        self.contact_frames = [
            self.model.getFrameId(f, type=pin.FrameType.FIXED_JOINT)
            for f in ("FL_contact", "FR_contact", "RL_contact", "RR_contact")
        ]
        self.foot_frames = self.contact_frames
        self.arm_ee_frame = self.model.getFrameId("arm_link06")

        # Joint limits (skip base).
        self.joint_pos_min = self.model.lowerPositionLimit[7:]
        self.joint_pos_max = self.model.upperPositionLimit[7:]
        self.joint_vel_max = self.model.velocityLimit[6:]
        # Map MuJoCo torque limits (23 values) onto the 22 actuated joints here.
        self.joint_torque_max = np.asarray(TORQUE_LIMITS[:22], dtype=float)

        self.arm_joints = len(ARM_JOINTS)
        self.front_force_ratio = 0.4
        self.gait_sequence = None

        # Name -> qpos index map for the 22 actuated joints.
        self.joint_q_indices: dict[str, int] = {}
        for name in URDF_JOINT_NAMES:
            jid = self.model.getJointId(name)
            self.joint_q_indices[name] = self.model.joints[jid].idx_q

        # MuJoCo joint name -> pinocchio qpos index (filled by the controller).
        self.mujoco_to_pin_qpos: dict[str, int] = {}

    def _default_q0(self) -> np.ndarray:
        q = np.zeros(self.model.nq)
        q[2] = 0.62  # base height [m]; calibrated below
        # Pinocchio FreeFlyer quaternion order is (x, y, z, w).
        q[6] = 1.0  # identity quaternion
        self._apply_default_joint_pose(q)
        # Calibrate the base height so the mean wheel-contact height is 0.
        q = self._calibrate_base_height(q)
        return q

    def _calibrate_base_height(self, q: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        pin.framesForwardKinematics(self.model, self.data, q)
        heights = []
        for name in ("FL_contact", "FR_contact", "RL_contact", "RR_contact"):
            fid = self.model.getFrameId(name, type=pin.FrameType.FIXED_JOINT)
            heights.append(float(self.data.oMf[fid].translation[2]))
        mean_height = float(np.mean(heights))
        q[2] -= mean_height
        return q

    def _apply_default_joint_pose(self, q: np.ndarray) -> None:
        # URDF joint order: FL, FR, RL, RR hips/thighs/calves, then wheels.
        leg_names = [
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ]
        arm_names = [f"arm_joint{i}" for i in range(1, 7)]

        for name, value in zip(leg_names, LEG_NOMINAL):
            jid = self.model.getJointId(name)
            q[self.model.joints[jid].idx_q] = value
        # Wheels start at zero rotation.
        for name in ("FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"):
            jid = self.model.getJointId(name)
            q[self.model.joints[jid].idx_q] = 0.0
        for name, value in zip(arm_names, ARM_NOMINAL):
            jid = self.model.getJointId(name)
            q[self.model.joints[jid].idx_q] = value

    def set_gait_sequence(self, gait_type: str, gait_period: float) -> None:
        """B2-W wheels are always in contact: only 'stand' is supported."""
        from wheelrl.mpc.gait_sequence import GaitSequence

        self.gait_sequence = GaitSequence(gait_type, gait_period)
