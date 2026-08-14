"""Whole-body inverse-dynamics MPC for the B2-W + Z1 (ported from wb-mpc-locoman).

The controller solves a receding-horizon optimal control problem with:
  - full-order inverse dynamics (RNEA) as the dynamics constraint,
  - contact forces at the four wheel-ground contact points with friction cones,
  - joint torque limits,
  - base, wheel-rolling, and TCP tracking objectives.

This is the port of https://github.com/lukasmolnar/wb-mpc-locoman adapted to the
wheel-legged Unitree B2-W carrying a Unitree Z1 arm.
"""

from wheelrl.mpc.dynamics import Dynamics
from wheelrl.mpc.dynamics_whole_body_torque import DynamicsWholeBodyTorque
from wheelrl.mpc.ocp import OCP
from wheelrl.mpc.ocp_whole_body_rnea import OCPWholeBodyRNEA, make_ocp
from wheelrl.mpc.robot import B2WZ1

__all__ = [
    "Dynamics",
    "DynamicsWholeBodyTorque",
    "OCP",
    "OCPWholeBodyRNEA",
    "make_ocp",
    "B2WZ1",
]
