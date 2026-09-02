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

- Current WSL repository: `/home/mingqian/WheelRL`
- Remote: `https://github.com/mingqian0850/WheelRL.git`
- Working branch: `backup/vbc-vision-hybrid-tcp-20260829`
- Branch point: `a546616` (`ubuntu-4090-training`)
- Preserved VBC/camera content commit: `555ada3`
- The handoff itself is committed after `555ada3`; use the branch HEAD rather
  than expecting this document to contain its own commit hash.
- This branch is a backup/experimental branch. It has not been merged into
  `ubuntu-4090-training`.

At the end of the 2026-08-31 session, the branch had two intentionally
untracked release archives, `track_ppo_v3.zip` and `track_ppo_v4.zip`. Preserve
them. Always run `git status --short --branch` before editing and do not assume
that ignored `runs/` artifacts belong to a particular Git branch.

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

### Isaac Sim / Isaac Lab bring-up

The local NVIDIA simulation environment was installed and accepted under the
NVIDIA Omniverse EULA:

- Miniforge: `/home/mingqian/miniforge3`;
- conda environment: `env_isaaclab51`;
- Isaac Sim package: `5.1.0.0`;
- Isaac Lab editable checkout: `/home/mingqian/IsaacLab` at `b0542fe2`;
- Isaac Lab Python package: `0.54.4`;
- Isaac Lab RL package: `0.5.2`;
- PyTorch: `2.7.0+cu128`;
- RSL-RL: `5.0.1`;
- MuJoCo in that environment: `3.12.0`.

Activate it with:

```bash
eval "$(/home/mingqian/miniforge3/bin/conda shell.bash hook)"
conda activate env_isaaclab51
export OMNI_KIT_ACCEPT_EULA=YES
```

The external baseline repository is cloned at
`/home/mingqian/LeggedManip_Lab`, branch `master`, commit `4e12109d`. Its remote
has only `master`; there is no alternate branch containing a better B2 + Z1
policy. At handoff time that external checkout had modified Git-LFS USD files
for GO1/GO2 manipulator variants. They were not part of WheelRL work and must
not be reset, committed, or pushed without a separate audit.

LeggedManip already contains a B2 + Z1 USD and registered Isaac tasks
`B2-Z1-WBC` / `B2-Z1-WBC-Play`. Earlier loading exposed unresolved foot-visual
references in the USD dependency graph. Physics, articulation, observations,
rewards, and a training smoke run still worked, but the visual references must
be repaired or visually checked before relying on rendered data.

Do not confuse the platforms:

- WheelRL models **B2-W + Z1**, including four driven wheels;
- LeggedManip models **B2 + Z1**, a conventional quadruped without B2-W wheel
  actions.

A policy trained for one morphology is not directly deployable on the other.
Choose the intended physical robot before doing a long training run.

### B2 + Z1 learned-policy validation

The bundled LeggedManip MuJoCo WBC policy is:

```text
/home/mingqian/LeggedManip_Lab/mujoco/deploy/policy/b2_z1/wbc/policy.pt
```

It was verified as a finite TorchScript network with a `210 -> 18` interface:
three frames of 70 observations produce 18 leg/arm joint-position residuals.
The MuJoCo runner loads it, but the observed posture leans forward and TCP
tracking is not good enough for the precision objective.

The only locally trained B2 + Z1 checkpoint is the one-iteration smoke run:

```text
/home/mingqian/LeggedManip_Lab/logs/rsl_rl/b2_z1_wbc/
    2026-08-31_13-49-01/model_0.pt
```

It reported roughly `-3.37` mean reward, `0.1545 m` mean TCP position error,
and `0.0965 rad` mean orientation error. It is not a useful trained policy,
even though an exported `policy.pt` and `policy.onnx` exist beside it.

The inspected B2 + Z1 simulation contains no explicit external payload on the
TCP. Do not infer real payload capacity from simulated arm-link mass. Unitree
Z1 payload ratings depend on the exact Air/Pro variant and operating envelope.

### What the existing LeggedManip `B2-Z1-WBC` task really trains

Despite its name, this task is not an analytic whole-body QP controller. It is
an end-to-end PPO policy that outputs all 18 joint-position residuals at 50 Hz.
The audit found several reasons not to continue training it unchanged for
arbitrary world-frame TCP tracking:

- TCP x/y commands are relative to `link0`, z is a world-frame height, and
  orientation is relative to `link0`; the target therefore moves in world x/y
  when the robot walks.
- A random base-velocity command is sampled independently of the TCP goal, so
  the policy is rewarded for two potentially conflicting tasks.
- TCP commands are piecewise constant for 8-10 seconds and come from a
  hand-written Cartesian box. The full box is x `[0.55, 0.85]`, y
  `[-0.35, 0.35]`, and z `[0.1, 1.0]`.
- The actor receives the raw goal but not explicit current TCP pose or SE(3)
  error; current TCP pose is privileged critic information only.
- The position reward has a broad standard deviation of about `0.316 m`, so
  it does not strongly optimize millimetre-level precision.
- Disabled curriculum logic currently promotes the sampler directly to its
  full limit box. Orientation sampling also bypasses the intended curriculum.

The forward-only goal distribution, arm-deviation cost, independent locomotion
command, and comparatively weak upright penalty plausibly explain the learned
forward lean. This is an engineering inference from the task definition, not
a completed ablation study.

### WheelRL branch and published-policy audit

The remote has four branches:

- `main` at `da0eddf`;
- `agent/rl-drl-tutorial` at `f17a223`;
- `ubuntu-4090-training` at `a546616`;
- `backup/vbc-vision-hybrid-tcp-20260829` at this handoff branch.

No branch tracks a learned WBC/hierarchical checkpoint. `main` and the tutorial
branch do not contain `wheelrl-play-wbc`; the Ubuntu and backup branches contain
the current controller and training code. Their WBC implementation is
identical, while the backup branch additionally preserves the VBC cameras and
documentation.

The best published learned hybrid is the GitHub Release `hier-policies-v2`:

- asset: `hier_ppo_v2.zip`;
- reported result: about `4.5 mm / 0.7 deg`, with `0/10` falls under its test;
- checkpoint interface: 78 observations and six TCP-residual actions;
- compatible source commit: `75afc64`.

The current source expects a new three-action base coordinator and deliberately
rejects that six-action checkpoint. The release tags misleadingly point to the
old `main`; use commit `75afc64`, not the tag, for replay. A headless replay of
the exact release/commit pair loaded successfully. Use a separate worktree if
it must be reproduced so the current branch remains intact.

The current `track_ppo_v3` and `track_ppo_v4` policies are direct 23-action
whole-body trackers, not WBC policies. They expect 86 observations, while the
current tracking environment emits 92, so the current README play command
fails normalization-space validation. Do not load either archive into the
three-action hierarchical runner.

The best immediately playable precision controller in WheelRL remains the
current analytic WBC:

```bash
cd /home/mingqian/WheelRL
source .venv/bin/activate
wheelrl-play-wbc --controller wbc --controls panel --seconds 0
```

A two-second headless smoke run completed normally and held the neutral target
at approximately `0.8 mm / 0.04 deg`. This is only a smoke result, not the full
acceptance benchmark described below.

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

The complete design and acceptance criteria are in:

- `docs/HIGH_PRECISION_TCP_TRACKING_ROADMAP.md`

Initial simulation targets are:

- static reachable setpoints: <= 5 mm RMS and <= 2 degrees RMS;
- moving target with stationary base: <= 10 mm RMS and <= 5 degrees RMS;
- whole-body relocation/tilt: <= 15 mm RMS and <= 7 degrees RMS;
- door pre-grasp terminal error: <= 10 mm and <= 5 degrees.

These are engineering goals, not current measured performance.

### Agreed no-manual-reach TCP training design

The user does not want to hand-label an arm-reachable Cartesian region. The
controller should accept an exact world-frame TCP target and decide whether
arm motion, base relocation, or body posture is required.

The recommended architecture is an always-active, goal-conditioned macro
policy around a deterministic precision WBC:

```text
exact world-frame TCP pose + desired twist
    -> learned coordinator at 10-25 Hz
    -> base velocity/yaw + body height/pitch
    -> deterministic SE(3) WBC/IK at 100-500 Hz
    -> joint position/torque loops
```

For B2 + Z1, a suitable macro action is
`[base vx, base vy, yaw rate, body height, body pitch]`. For nonholonomic
B2-W + Z1, omit lateral base velocity. The policy must not translate or rotate
the user's TCP target.

Do not replace the Cartesian box with another manually tuned binary gate.
Derive feasible training goals from the robot model instead:

1. Sample a collision-safe ghost/goal robot configuration from USD/MJCF soft
   joint limits, including a stable base/body pose.
2. Run batched forward kinematics and use the resulting TCP SE(3) pose as the
   exact target.
3. Reject only physical invalidity: collision, ground intersection, inadequate
   joint margin, or very low manipulability.
4. Join valid goal poses with minimum-jerk/spline trajectories and expose the
   desired TCP linear/angular velocity.
5. Later displace the ghost base from the real base so goals require learned
   relocation while remaining feasible by construction.

This removes a hand-authored arm workspace, but it does not remove physical
safety bounds or the need for a finite training arena. Impossible targets must
produce an explicit infeasibility/fallback state rather than an unsafe motion.

Recommended actor observations are geometric position and SO(3) orientation
error, desired TCP twist, current TCP pose, base velocity/gravity/height,
joint position/velocity/limit margins, Jacobian singular values or
manipulability, contacts or wheel slip, WBC status, and previous action.

Use unconditional costs on base motion, body tilt, action magnitude/rate,
energy, slip, collision, and joint-limit proximity. Then zero base motion is
naturally optimal for a locally solvable goal, while relocation becomes worth
its cost when it reduces persistent TCP error. Train fixed-horizon tracking
episodes; success thresholds are evaluation criteria, not reachability gates.

The staged curriculum should be:

1. stationary FK-derived local poses;
2. smooth six-DoF trajectories and desired twist;
3. fixed world goals requiring base relocation;
4. payload, mass, friction, latency, sensor-noise, and push randomization.

Use Isaac Lab's GPU-batched differential IK or operational-space control in a
4,096-environment loop. Do not put Pink's per-environment CPU IK loop inside
that training path.

Preserve the upstream `B2-Z1-WBC` baseline and register a new task, provisionally
`B2-Z1-TCP`, for this design. That task does **not** exist yet. Its observation
and action interface will differ from the current `210 -> 18` policy, so the
MuJoCo deploy runner must be updated alongside it.

### Agreed randomization and terrain curriculum

Do not inherit the upstream task's full randomization during initial controller
development. The current LeggedManip base configuration applies friction
`0.5-1.2`, all-body mass scaling `0.9-1.1`, base COM shifts up to 5 cm, and
gain/inertia scaling up to +/-20% from startup; its delayed actuators can also
add 0-20 ms. This is too much uncertainty for diagnosing a new precision task.

Use a performance-gated curriculum instead:

1. nominal flat plane, no observation corruption, delay, payload, or pushes;
2. stationary FK-derived targets, followed by smooth six-DoF trajectories;
3. ghost-base displacement that requires whole-body relocation;
4. mild flat-ground randomization: roughly +/-5% mass/inertia, +/-1 cm COM,
   +/-10% gains, small measured sensor noise, and 0-10 ms delay;
5. measured deployment randomization: roughly +/-10% mass, +/-15% inertia,
   +/-3 cm COM, +/-20% gains, payload mass/COM, and 0-20 ms delay;
6. modest pushes last, after randomized-flat tracking passes;
7. deployment-matched terrain only after the preceding stages pass.

Keep about 20% nominal environments during the full-randomization stage to
anchor precision. For an indoor door task, a suitable final terrain mixture is
approximately 70% plane, 20% low height variation (start at +/-5 mm and cap
near +/-10 mm), and 10% slopes up to about +/-3 degrees. Do not train on
stairs or rubble unless the intended deployment requires them. Texture and
lighting augmentation are irrelevant to the state-only controller and should
be added only when camera observations enter the actor.

### Expected precision while the base moves

The Z1's advertised approximately 0.1 mm repeatability is not mobile,
world-frame TCP accuracy. Base-state error, foot impact and slip, mount/TCP
calibration, arm compliance, payload deflection, latency, and target perception
dominate once the quadruped moves. A one-degree base-attitude error at 0.7 m
reach alone creates about 12 mm of TCP displacement.

The following are defensible engineering ranges, not demonstrated guarantees
for the current policy:

| Mode | Simulation RMS target | Real-hardware RMS estimate |
|---|---:|---:|
| Standing, arm motion only | 3-8 mm, 1-2 deg | 5-15 mm, 1-3 deg |
| Slow coordinated walking | 10-25 mm, 2-5 deg | 20-50 mm, 3-8 deg |
| Touchdown, slip, fast gait, or payload transients | 20-50 mm | 50-100+ mm |

For the door task, use `transit -> slow approach -> settle -> precise grasp ->
compliant pull`, rather than trying to hold millimetre accuracy while trotting.
Slow the base below roughly 0.1-0.15 m/s during approach, settle for a measured
0.3-1.0 s stable window, execute the final 10-20 mm arm-dominant alignment at
no more than about 20-50 mm/s, and use Cartesian impedance/admittance at about
5-20 mm/s after contact. These values are starting points to validate, not
hard-coded universal limits.

Supporting references:

- Unitree Z1 specifications and stiffness caveat:
  <https://www.unitree.com/mobile/z1/>
- Whole-body end-effector tracking results on another quadruped-manipulator:
  <https://arxiv.org/html/2507.08656>
- Exact B2 + Z1 inverse-dynamics MPC work, which demonstrates coordinated
  tracking but does not publish a TCP RMSE:
  <https://arxiv.org/html/2511.19709>

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
6. Confirm whether the target platform is B2 or B2-W and use a matching USD;
   do not train a long B2 run for intended B2-W deployment.
7. Add the separate `B2-Z1-TCP` Isaac Lab task with world-frame goals,
   FK-derived target generation, smooth trajectories, and an always-on macro
   coordinator while freezing the precision WBC.
8. Smoke-test task dimensions/reward terms, then train and evaluate the staged
   curriculum with multiple deterministic seeds.
9. Implement the matching MuJoCo observation/controller path before claiming
   cross-simulator deployment.
10. Add handle-frame impedance and force-aware turn/pull modes.
11. Connect the dual-camera visual policy only after state-based control meets
    the tracking benchmark.

The immediate next task should be step 1, not more reward tuning.

## Validation and common commands

Environment setup:

```bash
cd /home/mingqian/WheelRL
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

After the new Isaac task is implemented and registered, its intended commands
are:

```bash
eval "$(/home/mingqian/miniforge3/bin/conda shell.bash hook)"
conda activate env_isaaclab51
cd /home/mingqian/LeggedManip_Lab
export OMNI_KIT_ACCEPT_EULA=YES

# Interface/reward smoke test.
python scripts/rsl_rl/train.py \
  --task B2-Z1-TCP --num_envs 64 --headless --max_iterations 2

# Full RTX 4090 run only after the smoke test and deterministic evaluation pass.
python scripts/rsl_rl/train.py \
  --task B2-Z1-TCP --num_envs 4096 --device cuda:0 --headless \
  --max_iterations 5000 --run_name world_tcp_fk_v1
```

Do not run those commands until `B2-Z1-TCP` is registered. Do not substitute
the existing `B2-Z1-WBC` task and assume it implements the new design.

## Safety and repository guidance

- Preserve user checkpoints and `runs/`; most run data are intentionally
  gitignored and may exist only locally or as GitHub Release assets.
- Preserve the untracked `track_ppo_v3.zip` and `track_ppo_v4.zip` archives in
  `/home/mingqian/WheelRL`.
- Treat `/home/mingqian/LeggedManip_Lab` as an external, currently dirty
  checkout. Do not push its Git-LFS USD changes to the upstream repository.
- Never assume two machines combine training automatically. WSL and Ubuntu
  4090 runs are independent trials unless distributed training is explicitly
  implemented.
- Do not replace the camera transforms with paper values without frame and
  calibration verification.
- Do not claim the precision goals are achieved until the new benchmark
  measures them.
- Keep the VBC backup branch recoverable while developing the higher-precision
  controller on a separate branch.
