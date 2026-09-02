# WheelRL handoff

Last updated: 2026-09-02 (Europe/Berlin)

## User objective

Develop a Unitree B2-W + Z1 + gripper controller that can perform accurate,
smooth, whole-body TCP pose tracking and later open a pull door by approaching
the lever, grasping it, rotating it to release the latch, and moving backward
while maintaining the grasp.

The user prioritizes **high TCP precision** over reproducing any one paper.
Existing approaches may be replaced when a cleaner architecture performs
better. The Visual Whole-Body Control (VBC) scheme is preserved as a useful
vision/body-coordination baseline, not accepted as the final precision
controller.

## Repository state

- WSL repository: `/home/chenm/WheelRL`
- Remote: `https://github.com/mingqian0850/WheelRL.git`
- Working branch: `feature/traditional-compliant-door-control`
- Branch point: `ba6bc23` (`backup/vbc-vision-hybrid-tcp-20260829`)
- Preserved VBC/camera content commit: `555ada3`
- The handoff itself is committed after `555ada3`; use the branch HEAD rather
  than expecting this document to contain its own commit hash.
- This feature branch is based on the preserved VBC/high-precision handoff
  branch. It has not been merged into `ubuntu-4090-training`.

The next native-Ubuntu task should branch from the current remote head as
`feature/isaac-lab-tcp-tracker`, not from `ubuntu-4090-training`. Its complete
environment, method, asset-validation and acceptance-test contract is in:

- `docs/research/ISAACLAB_TCP_TRACKING_HANDOFF_2026-09-02.md`

At the time this handoff was written, the branch contained no unrelated user
changes. Always run `git status --short --branch` before editing.

### Traditional pull-door baseline (current feature branch)

The current branch adds a non-RL baseline in
`src/wheelrl/traditional_door.py`. It uses the general MuJoCo-native `mink`
Python library for arm-only TCP pose IK, with every non-Z1 DoF frozen as an
exact QP constraint. A state machine approaches the handle, closes the
gripper, rotates the lever, detects latch release, lowers Z1 stiffness, and
then commands B2-W to reverse and yaw with the door angle.

Use:

```bash
wheelrl-play-door-traditional
wheelrl-play-door-traditional --headless --no-realtime
```

The nominal headless episode opens the door to about -0.721 rad in 19.35
simulated seconds. Ten seeds at each door-randomization strength 0.0, 0.5, and
1.0 completed successfully during implementation. The grasp remains a soft
idealized MuJoCo equality and is not yet a real-gripper validation. Details and
limitations are in `docs/TRADITIONAL_DOOR_CONTROLLER.md`.

## Completed work

### B2-W + Z1 simulation and control baseline

The repository already contains:

- B2-W, Z1, and one-DoF gripper MuJoCo assets;
- state-based PPO environments and trained-policy play paths;
- a pull-door scene and hierarchical door task;
- interactive browser controls for TCP pose and gripper commands;
- a bounded WBC with arm-first macro/micro coordination;
- automatic base relocation for arm-unreachable TCP targets;
- low-reach body tilt coordination;
- WSL and native Ubuntu RTX 4090 setup paths.

Relevant earlier commits immediately below this branch include:

- `df21347`: bounded tracking and low-reach tilt coordination;
- `a019040`: arm-first macro/micro mode;
- `75afc64`: residual penalty to reduce steady-state tracking offset.

### VBC-style dual-camera baseline

The robot now contains two fixed MuJoCo cameras:

- `head_cam`, mounted on `base_link`;
- `z1_cam`, mounted beside the gripper on `z1_link06`.

The old wrist-camera orientation was wrong because the model treated camera
`+Z` as the observation direction. MuJoCo observes along camera `-Z`. The
corrected wrist-camera frame is:

- optical/view axis: `z1_link06 +X`;
- image up: `z1_link06 +Z`;
- image right: `z1_link06 -Y`.

The physical camera housing is also oriented laterally rather than along the
forearm. Both simulated cameras use `fovy=42.272558950` degrees, which
corresponds to the released VBC configuration's 69-degree horizontal field of
view at 96 x 54 pixels.

Do not copy the paper's B1 wrist-camera translation directly onto real B2-W
hardware. Measure `T_base_headcam` and `T_link06_wristcam` from CAD/hand-eye
calibration before sim-to-real deployment.

Key files:

- `src/wheelrl/assets/b2w_z1/b2w_z1.xml`
- `docs/VISION_CAMERA_SETUP.md`
- `tests/test_env.py::test_vbc_style_cameras_face_forward`

The released VBC code uses three image-history frames (12 full-input channels:
two views x mask/depth x three frames), although the paper appendix says four
steps. WheelRL records and follows the executable implementation rather than
silently choosing one.

### Preserved VBC architecture

The reference architecture is:

```text
two RGB-D views
    -> target masks + segmented depth
    -> high-level visual policy (~6.25-8.33 Hz)
    -> [6D TCP increment, base forward velocity, yaw rate, gripper]
    -> learned quadruped policy (50 Hz) + arm IK (~800 Hz on the real system)
    -> joint controllers
```

This separation remains useful. Images should update task-space subgoals; they
should not replace the fast proprioceptive TCP servo.

The paper's reported goal-reaching errors are approximately:

- standing: 4.7 cm position, 0.850 rad orientation;
- walking: 2.9 cm position, 0.332 rad orientation.

Those results outperform direct joint prediction but are insufficient for the
desired lever-handle precision. The baseline is therefore archived rather than
used as the final controller.

Primary external references:

- <https://arxiv.org/abs/2403.16967>
- <https://wholebody-b1.github.io/>
- <https://github.com/Ericonaldo/visual_wholebody>

## Preferred high-precision direction

The agreed architecture is:

```text
head/wrist perception (30 Hz)
    -> handle/object pose + uncertainty
    -> task policy or door state machine (8-15 Hz)
    -> smooth world-/door-frame TCP subgoal
    -> reference governor (100-200 Hz)
    -> constrained whole-body QP (200-500 Hz)
    -> joint torque/position loops (500-1000 Hz)
```

RL should learn slow coordination choices such as base forward/reverse motion,
yaw, body height/pitch, reachability posture, QP-weight residuals, and bounded
TCP twist residuals. A deterministic constrained WBC should retain authority
over precise tracking and safety.

Use geometric SE(3) error:

```text
e_p = p_target - p_tcp
e_R = Log(R_target * R_tcp^T)
```

Avoid Euler-angle subtraction in the precision loop. Add bounded target
velocity/acceleration, critically damped filtering, terminal deadband and
hysteresis, adaptive singularity damping, null-space joint centering, and
anti-windup for any near-target integral term.

Hard WBC constraints should ultimately cover joint/torque/velocity limits,
wheel rolling and no-lateral-slip, ground contact/friction, self/body/ground
collision, base attitude/height safety, and a defined infeasibility fallback.

### Learning-method decision added 2026-09-02

Use MLM as the reference for short-horizon six-DoF TCP trajectories, recent
state history, future reference and curriculum learning. Use Deep Whole-Body
Control as a code-level reference for unified body/arm representations,
posture coordination, separate action heads and optional Advantage Mixing.

Do not directly reproduce either policy. MLM has no official implementation or
checkpoint available as of this date. Deep Whole-Body Control has code, but its
released default is predominantly position-only, its horizontal TCP target
follows the base, and base velocity is commanded separately. None of those
choices alone solves WheelRL's fixed world-target relocation requirement.

For the first Isaac Lab experiment, train a slow reachability/posture
coordinator around a frozen deterministic SE(3) servo. Treat a direct unified
joint policy as an ablation. Keep free-space tracking separate from the
grasp/turn/pull contact controller.

The complete design and acceptance criteria are in:

- `docs/HIGH_PRECISION_TCP_TRACKING_ROADMAP.md`

Initial simulation targets are:

- static reachable setpoints: <= 5 mm RMS and <= 2 degrees RMS;
- moving target with stationary base: <= 10 mm RMS and <= 5 degrees RMS;
- whole-body relocation/tilt: <= 15 mm RMS and <= 7 degrees RMS;
- door pre-grasp terminal error: <= 10 mm and <= 5 degrees.

These are engineering goals, not current measured performance.

## Door-control requirement

Do not use one rigid pose controller throughout the door episode. Use explicit
control modes:

1. approach and collision-aware alignment;
2. precise pre-grasp SE(3) tracking;
3. gripper closure and Cartesian compliance;
4. handle-axis rotation with radial/contact regulation;
5. pulling with maintained grasp/tension and negative base velocity;
6. traversal/release.

After contact, hybrid motion/force or impedance control is more important than
minimizing an unconstrained absolute TCP pose error. Joint-torque estimates or
a wrist force/torque sensor should be considered before real-door deployment.

## Recommended next implementation steps

Work in this order so changes are measurable:

1. Add a repeatable headless SE(3) benchmark covering static targets, smooth
   trajectories, unreachable targets, low targets, base relocation, and tilt.
2. Log RMS, 95th percentile, terminal error, settling time, overshoot, settled
   jitter, joint-limit margin, manipulability, collision margin, and QP
   infeasibility.
3. Replace any Euler orientation residual in the active precision path with an
   SO(3) logarithmic residual and unit-test wraparound near +/-pi.
4. Add reference filtering, terminal hysteresis, adaptive damping, and
   null-space joint centering to the arm-only path.
5. Integrate rolling, collision, joint, and base-posture constraints in one
   velocity-level QP; move to inverse-dynamics/torque QP after the benchmark is
   stable.
6. Train a slow reachability/posture coordinator in Isaac Sim while freezing
   the precision QP.
7. Add handle-frame impedance and force-aware turn/pull modes.
8. Connect the dual-camera visual policy only after state-based control meets
   the tracking benchmark.

The immediate next task should be step 1, not more reward tuning.

## Validation and common commands

Environment setup:

```bash
cd /home/chenm/WheelRL
source .venv/bin/activate
```

Run the complete test suite:

```bash
pytest
```

At the VBC backup commit, all 29 tests passed and `git diff --check` was clean.

Interactive WBC:

```bash
wheelrl-play-wbc --seconds 0
```

Headless reproducible pose command:

```bash
wheelrl-play-wbc \
  --headless --seconds 10 \
  --target-offset 0.10 0.04 -0.04 \
  --rpy-offset-deg 4 -6 8
```

Read `README.md`, `docs/VISION_CAMERA_SETUP.md`, and
`docs/HIGH_PRECISION_TCP_TRACKING_ROADMAP.md` before changing the controller.

## Safety and repository guidance

- Preserve user checkpoints and `runs/`; most run data are intentionally
  gitignored and may exist only locally or as GitHub Release assets.
- Never assume two machines combine training automatically. WSL and Ubuntu
  4090 runs are independent trials unless distributed training is explicitly
  implemented.
- Do not replace the camera transforms with paper values without frame and
  calibration verification.
- Do not claim the precision goals are achieved until the new benchmark
  measures them.
- Keep the VBC backup branch recoverable while developing the higher-precision
  controller on a separate branch.
