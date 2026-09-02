# Isaac Lab TCP tracking handoff (2026-09-02)

This is the implementation handoff for the next agent working on the native
Ubuntu experiment machine. The immediate goal is a reproducible Isaac
Sim/Isaac Lab training setup and a smoke-tested B2-W + Z1 world-frame TCP
tracking environment. Do not start a long training run until the asset and
coordinate-frame gates below pass.

## Repository starting point

- Remote: `https://github.com/mingqian0850/WheelRL.git`
- Source branch: `feature/traditional-compliant-door-control`
- Verified implementation parent: `e036874` (`door: add Mink IK compliant pull controller`)
- Create a new implementation branch from the current remote head, preferably
  `feature/isaac-lab-tcp-tracker`.
- Do not start from the older `ubuntu-4090-training` branch.
- Keep the existing MuJoCo `.venv` untouched. Give Isaac Lab its own managed
  environment and record every version in a machine-readable lock/metadata
  file.

Before editing, read:

- `HANDOFF.md`
- `docs/HIGH_PRECISION_TCP_TRACKING_ROADMAP.md`
- `docs/research/RESEARCH_STATUS_2026-08-28.md`
- `docs/TRADITIONAL_DOOR_CONTROLLER.md`
- `sources/research_mlm_20260902.md`
- `sources/research_manipulation_locomotion_20260902.md`
- `src/wheelrl/envs/b2w_z1_track.py`
- `src/wheelrl/envs/b2w_z1_hier.py`
- `src/wheelrl/assets/b2w_z1/b2w_z1.xml`

## Agreed method selection

Do not reproduce either paper verbatim.

### Borrow from MLM

- a short past/current/future six-DoF TCP reference window;
- recent proprioceptive history;
- continuous rotation representation for policy input;
- asymmetric actor-critic training and dynamics randomization;
- curriculum across target range and task difficulty.

MLM is paper-only as of this handoff. Its trajectory predictor is unnecessary
for the first version because a synthetic trajectory generator provides future
samples exactly. Add prediction only for headset or other historical-only
teleoperation.

### Borrow from Deep Whole-Body Control

- one shared representation for body and arm coordination;
- separate lower-body and arm output heads;
- Advantage Mixing as an optional credit-assignment ablation;
- proprioceptive history for online dynamics adaptation;
- the command construction that makes body height, pitch, and roll useful for
  extending the arm workspace.

Do not inherit its moving target frame. Its horizontal TCP goal follows the
base, and locomotion velocity is commanded separately, so it does not learn to
drive to a fixed world target solely from TCP error. The official release also
defaults to position-only tracking: orientation sampling/reward and the final
three arm action scales are zero.

### WheelRL target architecture

The primary experiment is hierarchical:

```text
world/door-frame SE(3) target + desired twist/limits
        -> smooth short-horizon reference manager
        -> slow learned reachability/posture coordinator
        -> constrained deterministic SE(3) QP/WBC
        -> leg position, wheel velocity/torque, Z1 position/torque servos
```

The learned coordinator proposes base forward velocity, yaw rate, body height,
pitch/roll, arm null-space posture, QP weights, or a tightly bounded TCP twist
residual. It must not change the user's final target. The deterministic WBC
retains final pose precision and hard safety constraints.

For comparison, implement a unified-policy baseline with three output heads:

```text
12 leg joint-position residuals
4 wheel velocity or torque commands
6 Z1 joint-position residuals
```

That is 22 tracking actions. Keep the one-DoF gripper out of the free-space
tracking policy; add it only at the door-task layer. A direct 23-action policy
including the gripper is a later door-task ablation, not the first benchmark.

## Training contract

### Command and coordinate frames

- Store the final target in `world` or `door` frame.
- Re-express it in the current heading/base frame at every policy step; never
  reconstruct it by adding an offset to the current base pose.
- Position error is `p_target - p_tcp` in one declared frame.
- Orientation error for evaluation/control is
  `Log(R_target * R_tcp^T)`. Do not subtract Euler angles.
- Policy rotation input may use a normalized 6D rotation representation.
- Include desired TCP linear/angular velocity and bounded acceleration/jerk.

### Actor observation

- five or more frames of projected gravity/base angular velocity;
- leg, wheel, and Z1 joint position/velocity;
- wheel contact/rolling velocity and slip estimate;
- previous action;
- current TCP pose/twist and geometric target error;
- past/current/future TCP reference samples, initially `t-3:t+4`;
- joint-limit margin, arm manipulability, and coordinator/WBC mode where useful.

The critic may additionally use true base velocity, contact forces, randomized
mass/friction/motor strength, sensor delay, terrain and external disturbance.
Do not expose privileged signals to the deployment actor.

### Reward and constraints

The main tracking term should require position and orientation simultaneously,
for example a product or gated fusion of the two bounded rewards. Add explicit
terms for desired twist, settling time and terminal TCP velocity. Penalize
terminal jitter, action rate/acceleration, unnecessary base travel, wheel slip,
energy, joint-limit proximity, poor manipulability and collisions.

Joint, torque, velocity, rolling/no-lateral-slip, ground contact/friction,
self/body/ground collision and safe body-attitude limits belong in the
controller/environment constraints, not only in soft rewards.

### Curriculum

1. Reachable static targets with base motion disabled.
2. Smooth moving targets inside the arm workspace.
3. World-fixed targets requiring base translation/yaw.
4. Low, high, lateral and rear targets requiring natural posture allocation.
5. Dynamics, mass/COM, friction, motor strength, sensor noise and control-delay
   randomization.
6. Only after free-space acceptance: pre-grasp trajectories and door contact.

Begin with no RGB/depth observations. State-based tracking must converge before
camera rendering or visual encoders are added.

### Frequencies and scaling

- Begin near the published baselines: 200--400 Hz physics and 50 Hz policy.
- Run headless for throughput tests and GUI only for small evaluation batches.
- Scale environment count through measured powers of two; do not assume the
  maximum count or GPU utilization in advance.
- Record simulation steps/s, policy-update time, GPU memory/utilization, reset
  rate and real-time factor for every benchmark.
- Do not assume two GPUs pool memory. Treat the RTX 4090 and any RTX PRO 6000 as
  independent workers unless distributed PPO is explicitly configured.

## Setup and smoke-test sequence

1. Record Ubuntu, kernel, GPU model/VRAM, driver, CUDA visibility, CPU, RAM and
   free disk space. Do not infer the exact `PRO 6000` model from its name.
2. From the current official Isaac Lab documentation, choose a mutually
   compatible Isaac Sim, Isaac Lab, Python and PyTorch/CUDA stack. Save the
   exact commit/tag and installation method; do not copy the 2022 Isaac Gym
   Preview 3 environment.
3. Run one official headless example and one GUI example before adding WheelRL.
4. Import the robot asset and validate every link/joint before training.
5. Run deterministic reset/step tests at 1, 16 and 64 environments.
6. Run a very small PPO smoke test, verify finite observations/actions/losses,
   checkpoint save/load and deterministic evaluation.
7. Benchmark increasing environment counts and select by throughput rather
   than VRAM occupancy alone.
8. Commit the environment lock, setup notes, asset-conversion provenance,
   smoke-test command and measured result before starting a long run.

## Asset and frame gates

All of the following must pass:

- 12 leg joints, 4 continuous wheel joints, 6 Z1 joints and the separate
  one-DoF gripper are present and mapped by explicit names, not assumed order.
- Joint axes, signs, limits, home pose, mass, COM and inertia are checked.
- Z1 mounting transform, TCP frame and gripper geometry agree with MuJoCo.
- Wheel radius, rotation sign, forward direction and lateral slip convention
  are tested.
- Left/right TCP commands move in the correct world direction.
- Quaternion ordering and multiplication convention have unit tests.
- A stored world target remains numerically invariant while the base translates,
  rotates, changes height and tilts.
- Isaac Lab and MuJoCo FK agree at the home pose and randomized configurations
  within a declared numerical tolerance.
- Randomized parameters reset from nominal values and do not accumulate across
  episodes.
- Self-collision and body-ground collision are active and observable.

## Acceptance benchmark

Use a fixed set of at least 30--50 target cases covering front/side/rear,
near/far, high/low, orientation-only changes and smooth trajectories. Compare
the learned controller against pure deterministic QP using identical seeds.

| Test | Position target | Orientation target |
|---|---:|---:|
| Static reachable setpoints | <= 5 mm RMS | <= 2 deg RMS |
| Moving target, stationary base | <= 10 mm RMS | <= 5 deg RMS |
| Whole-body relocation/tilt | <= 15 mm RMS | <= 7 deg RMS |
| Door pre-grasp terminal pose | <= 10 mm | <= 5 deg |

Also report position/orientation P95 and maximum, rise/settling time, overshoot,
settled TCP linear/angular velocity RMS, base travel/yaw, body attitude,
minimum joint/collision margin, manipulability, wheel slip, energy/action rate,
QP infeasibility/deadline misses and simulation throughput.

Go/no-go rules:

- A local reachable target with the learned coordinator enabled must not be
  worse than pure QP.
- An unreachable target must cause bounded, purposeful base relocation rather
  than a distorted arm pose or perpetual base oscillation.
- A low target must produce a collision-free height/tilt allocation and retain
  tracking when the target subsequently moves forward.
- Reaching the terminal tolerance must reduce TCP velocity and base motion;
  reward alone is not evidence of convergence.
- Do not add door contact or vision until these free-space gates pass.

## Handoff prompt for the next agent

```text
Read docs/research/ISAACLAB_TCP_TRACKING_HANDOFF_2026-09-02.md and all files it
marks as required. On the native Ubuntu experiment machine, clone
origin/feature/traditional-compliant-door-control and create
feature/isaac-lab-tcp-tracker. Inspect the actual hardware and current official
Isaac Lab compatibility guidance before installing anything. Build a separate,
version-locked Isaac Sim/Isaac Lab environment, run official headless and GUI
smoke tests, then import and validate B2-W + Z1 + gripper. Implement only the
minimal world-frame state-based six-DoF TCP tracking environment and a tiny PPO
smoke test. Do not start a long training run, add cameras, or add door contact
until asset, coordinate-frame, finite-value, save/load and deterministic
evaluation gates pass. Record exact commands, versions and measured throughput,
commit the reproducible setup, and report any blocking mismatch instead of
silently changing the robot model.
```
