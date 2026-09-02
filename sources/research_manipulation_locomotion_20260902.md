# Source record: Deep Whole-Body Control

Retrieved: 2026-09-02

Query: assess whether *Deep Whole-Body Control: Learning a Unified Policy for
Manipulation and Locomotion* is reusable for B2-W + Z1 six-DoF TCP tracking,
and compare it with MLM.

## Primary sources

- Official project page: <https://manipulation-locomotion.github.io/>
- Official arXiv record: <https://arxiv.org/abs/2210.10044>
- arXiv full text: <https://arxiv.org/html/2210.10044v1>
- Official code: <https://github.com/MarkFzp/Deep-Whole-Body-Control>

## Verified facts

- CoRL 2022 Oral; Best Systems Paper Award finalist.
- The policy observes robot state, an SE(3) end-effector command, a separately
  supplied forward/yaw base-velocity command, previous action, and randomized
  environment extrinsics. It outputs 12 leg and 6 arm joint-position targets,
  which are converted to torque by joint PD control.
- Training uses PPO with Advantage Mixing and Regularized Online Adaptation.
- The simulator is NVIDIA Isaac Gym. The paper reports 5,000 environments,
  200 Hz physics, 50 Hz policy control, 40 environment steps per batch, and
  10,000 training batches (about two billion simulated samples).
- End-effector targets are sampled in spherical coordinates around the arm
  base. The target construction follows the base horizontal position but is
  independent of torso height, roll, and pitch. Consequently, the learned legs
  bend, stretch, roll, and pitch to extend arm workspace; base translation is
  commanded separately rather than inferred from a fixed world-frame TCP goal.
- The paper demonstrates teleoperation, AprilTag-guided position tracking, and
  demonstration replay. It does not demonstrate force-controlled lever turning
  or hinged-door pulling.

## Public-code audit

Repository HEAD inspected: `8159e4ed8695b2d3f62a40d2ab8d88205ac5021a`
(2024-01-14).

- The repository contains modified `legged_gym`, modified `rsl_rl`, the
  `widowGo1` environment/configuration, and combined Go1 + WidowX assets.
- No trained whole-body policy checkpoint was found.
- The top-level README contains only the citation; installation instructions
  live in the inherited `legged_gym/README.md` and target Isaac Gym Preview 3,
  Python 3.8, PyTorch 1.10, and CUDA 11.3.
- In the released default `widowGo1_config.py`, orientation target ranges and
  orientation reward weights are zero, and the last three arm action scales
  are zero. Therefore the released default is primarily a three-dimensional
  end-effector-position tracker, despite the paper's general SE(3) formulation;
  full orientation tracking requires implementation and retuning.

## Relevance assessment

The release is a useful executable baseline for joint-level whole-body
coordination, posture adaptation, Advantage Mixing, and online dynamics
adaptation. It is not a drop-in B2-W + Z1 tracker: the robot, arm, contacts,
wheel actions, command frame, orientation objective, collision model, and
deployment interfaces all need to be replaced or ported.
