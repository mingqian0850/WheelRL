# WheelRL: Unitree B2-W + Z1 in MuJoCo

This project provides a runnable WSL2 training starter for whole-body control of
a Unitree B2-W carrying a six-axis Unitree Z1 arm and its one-DoF gripper.

The current research status, literature audit, open-source baseline ranking, and
recommended B2+Z1-first roadmap are recorded in
[`docs/research/RESEARCH_STATUS_2026-08-28.md`](docs/research/RESEARCH_STATUS_2026-08-28.md).
The roadmap treats this repository as a measurable baseline, not as a required
architecture: a clean-slate model, controller, simulator, or training stack is
explicitly allowed when it produces a stronger reproducible result.

The environment has:

- the official Unitree B2-W MuJoCo body and wheel assets;
- official Z1 and gripper masses, inertias, limits, and meshes;
- 23 normalized actions: 12 leg targets, four wheel velocity targets, six arm
  targets, and one gripper open/close target;
- 500 Hz MuJoCo/PD control and a 50 Hz RL policy;
- an 80-dimensional proprioceptive observation including gripper position and
  velocity;
- PPO training, checkpoints, TensorBoard logs, evaluation, and visualization.

It also contains a separate pull-door task in which the Z1 must reach the
lever, close the gripper, rotate the lever far enough to release a latch, and
command the B2-W backward while pulling the door open.

The final action component controls the gripper: `-1` is fully open
(`-pi/2 rad`) and `+1` is fully closed (`0 rad`).

## WSL setup

From this project directory:

```bash
bash scripts/bootstrap_wsl.sh
source .venv/bin/activate
```

The bootstrap script installs an isolated Python 3.11 environment, resolves the
locked dependencies, validates the MuJoCo model, and prints CUDA availability.
A separate CUDA Toolkit (`nvcc`) is not required for PyTorch wheels.

### Separate Ubuntu RTX 4090 machine

The 4090 machine should use its own native-Linux Git clone and virtual
environment:

```bash
git clone --branch ubuntu-4090-training \
  https://github.com/mingqian0850/WheelRL.git ~/WheelRL
cd ~/WheelRL
bash scripts/bootstrap_ubuntu_4090.sh
source .venv/bin/activate
```

The bootstrap checks both robot scenes and fails clearly if PyTorch cannot see
an RTX 4090 with roughly 24 GB VRAM. WSL and Ubuntu runs can proceed at the
same time, but Stable-Baselines3 PPO does not combine gradients across the two
processes. Treat them as independent seeds or hyperparameter trials:

```bash
# WSL: CPU MuJoCo trial
wheelrl-train-door \
  --stage curriculum --timesteps 3000000 --n-envs 6 \
  --device cpu --randomization 0.35 --seed 42
```

```bash
# Native Ubuntu 4090: independent trial
scripts/train_ubuntu_4090.sh door
```

The same launcher accepts `gripper` and `coordinator`. Override its defaults with environment
variables, for example:

```bash
SEED=143 N_ENVS=12 TIMESTEPS=5000000 \
  RANDOMIZATION=0.7 scripts/train_ubuntu_4090.sh door
```

When `--run-dir` is omitted, each trainer now writes to a collision-free path
containing the task, host name, UTC timestamp, and seed. Every run also stores
`metadata.json` with the OS, command, Git commit, MuJoCo/PyTorch versions, and
visible CUDA devices. The current state-only MLP is normally faster with
`--device cpu`, because MuJoCo physics remains CPU-based. Benchmark
`--device cuda` on the 4090, but expect that GPU to become important mainly
after adding RGB-D perception or a larger policy.

To continue one selected checkpoint on another machine, copy the model and
normalization statistics together:

```text
wheelrl-train-door --resume-model <model.zip> --resume-stats <vecnormalize.pkl> ...
wheelrl-train      --resume-model <model.zip> --resume-stats <vecnormalize.pkl> ...
wheelrl-train-hier --resume-model <model.zip> --resume-stats <vecnormalize.pkl> ...
```

### Sharing trained checkpoints (e.g. to the 4090 machine)

`runs/` is gitignored (only the verified full-task door checkpoint is kept in
Git), so `git pull` does **not** fetch trained policies. Published policies are
distributed as GitHub Release assets instead, one zip per run directory
(contains `final_model.zip` + `vecnormalize.pkl`):

```bash
# on the target machine, inside a WheelRL clone
gh release download track-policies-v1 --clobber
unzip -o track_ppo_v3.zip -d runs/
unzip -o track_ppo_v4.zip -d runs/
```

No `gh` login? Download the zips from the release page directly
(<https://github.com/mingqian0850/WheelRL/releases>) and unzip them into
`runs/`. Play the policy as usual:

```bash
wheelrl-play-track --model runs/track_ppo_v4/final_model.zip \
  --stats runs/track_ppo_v4/vecnormalize.pkl --seconds 0 --motion 0.5
```

To publish a new policy, zip the run directory and upload it as a release
asset (or commit it like the door checkpoint if it is the verified final one):

```bash
cd runs && zip -r track_ppo_vX.zip track_ppo_vX/final_model.zip track_ppo_vX/vecnormalize.pkl
gh release create track-policies-v2 track_ppo_vX.zip --title "Track policies v2"
```

Note: `scp` from this WSL machine usually does not work for other machines,
because WSL uses a NAT address that changes on reboot.

## Verify and view the robot

Headless model and API check:

```bash
wheelrl-check
```

Check the door model, handle spring, latch, and high-level action interface:

```bash
wheelrl-check-door
```

Open the MuJoCo viewer with a neutral standing controller:

```bash
wheelrl-play --seconds 30
```

Open the interactive TCP-pose whole-body controller:

```bash
wheelrl-play-wbc --seconds 120
```

By default this prints and opens a localhost browser control panel. It provides
numeric TCP `x/y/z + roll/pitch/yaw` inputs, jog buttons, separate gripper
open/close buttons, Home, automatic-base-motion control, and live tracking
error. MuJoCo keeps exclusive ownership of its keyboard shortcuts, so camera,
pause, help, and visualization controls no longer conflict with robot commands.
The yellow/red-green-blue triad in MuJoCo is the commanded Z1 TCP pose.

The browser also changes real robot motion limits while the simulation runs:

- `Precision` (`0.4x`): 2 mm / 0.5 degree jogs for grasp and contact work;
- `Normal` (`1x`): 1 cm / 2 degree jogs;
- `Fast` (up to `2x`, rotation `1.5x`): 3 cm / 5 degree jogs for large moves.

The profile scales TCP, base, and WBC joint velocities. It is separate from
MuJoCo's playback-speed control. Set the initial profile without the browser
with `wheelrl-play-wbc --speed-profile fast`.

If the browser cannot be opened automatically, visit the printed URL (normally
`http://127.0.0.1:8765/`). Choose a free port automatically with
`--panel-port 0`, or suppress automatic browser launch with
`--no-open-browser`.

See [`docs/MUJOCO_VIEWER_GUIDE.md`](docs/MUJOCO_VIEWER_GUIDE.md) for every
viewer panel, mouse action, and keyboard shortcut relevant to WheelRL.

The robot model also carries a VBC-style dual RGB-D layout: `head_cam` on the
front of B2-W and `z1_cam` beside the gripper. Their poses, projection,
96 x 54 visual-policy interface, real-camera options, and sim-to-real noise
model are documented in
[`docs/VISION_CAMERA_SETUP.md`](docs/VISION_CAMERA_SETUP.md).
The preserved VBC baseline and the separate constrained high-precision TCP
tracking roadmap are recorded in
[`docs/HIGH_PRECISION_TCP_TRACKING_ROADMAP.md`](docs/HIGH_PRECISION_TCP_TRACKING_ROADMAP.md).

The original keyboard controls remain as an explicit legacy mode:

```bash
wheelrl-play-wbc --controls keyboard
```

In that mode, `W/S` move forward/backward, `A/D` move left/right, `R/F` move
along `z`, `U/O`, `I/K`, and `J/L` rotate the target, `H` returns home, `C`
toggles the gripper, and `P` prints the pose. Because MuJoCo receives these keys
too, the browser panel is the recommended play interface.

A repeatable headless pose command is also available:

```bash
wheelrl-play-wbc \
  --headless \
  --seconds 10 \
  --target-offset 0.10 0.04 -0.04 \
  --rpy-offset-deg 4 -6 8
```

At 100 Hz the example solves wheel rolling constraints, floating-base motion
and stabilization, TCP SE(3), and posture tasks in one bounded, damped
least-squares QP. Joint-speed and predictive joint-position bounds are enforced
inside the solve instead of clipping its result afterward. The resulting leg
and arm velocities are integrated into references, while four wheel-velocity
targets drive the nonholonomic base. All targets are tracked through the
existing 500 Hz torque interface.

Targets inside a conservative Z1 workspace envelope mainly use the arm, with
limited leg/base help from `--base-assist` (default `0.25`). Base participation
increases smoothly near the workspace boundary, at an arm joint limit, or near
a poor-manipulability pose. For a clearly unreachable target it starts in the
first control update: automatic drive turns and/or translates B2-W while Z1
simultaneously moves toward the reachable part of the same world-frame target.
Once the TCP is IK-feasible and settled, control hands back to a strongly
damped arm-only precision mode; the nominal base parking yaw is discarded so
it cannot produce a slow limit cycle around the reached target. The browser and
MuJoCo overlay show base participation and TCP speed. Use `--no-auto-drive` to
disable automatic base motion.

Vertical tracking is handled separately from planar driving. For a sufficiently
low target in front, B2-W keeps its wheel footprint fixed and smoothly pitches
forward by at most 14 degrees: the front legs shorten while the rear legs
extend. This supplies useful low workspace without an uncontrolled whole-body
crouch. The TCP is released downward progressively as the tilt develops, using
a collision-tested local floor; targets below that floor remain visibly
commanded but are projected to the nearest safe servo point. The browser shows
`low-reach tilt` and its angle, or `vertical target limited` when even maximum
tilt cannot serve the request. For a target that is both far and low, planar
driving completes first and the low-reach tilt enters afterward. Once already
tilted, forward TCP jogging keeps the pitch latched and permits slow four-wheel
relocation inside the low workspace; it no longer stands up and lifts the TCP
between successive forward commands. The planar base stops when TCP error, not
a redundant nominal parking pose, says the relocation is complete.

This remains a simulation controller rather than a robot-ready inverse-dynamics
or force controller: friction-cone/contact-force optimization, collision-aware
trajectory planning, and hardware safety remain separate steps. The WBC scene
also excludes one proximal `base_link`/`z1_link02` collision pair because the
assumed mounting bracket overlaps the conservative simulated lidar collision
box; the real mount transform and collision geometry must be measured.

View the robot at the pull door:

```bash
wheelrl-play-door --stage turn --seconds 30
```

Run the deterministic traditional-control baseline that performs the complete
door sequence with Mink IK, gravity-compensated Z1 control, compliant pulling,
and B2-W door-arc following:

```bash
wheelrl-play-door-traditional
```

Fast headless verification:

```bash
wheelrl-play-door-traditional --headless --no-realtime
```

The base remains stopped through approach, grasp, and lever rotation. Only
after the handle crosses the latch-release angle does Z1 switch to a low
stiffness connection and B2-W reverse while yawing with the door angle. See
[`docs/TRADITIONAL_DOOR_CONTROLLER.md`](docs/TRADITIONAL_DOOR_CONTROLLER.md)
for the controller phases, Mink constraints, measured baseline, tuning, and
remaining sim-to-real limitations.

The viewer uses WSLg. If no window appears, verify that `echo $DISPLAY` is
non-empty and that GUI applications work in the same WSL session.

## Train

Short smoke run:

```bash
wheelrl-train --timesteps 16384 --n-envs 2 --device cpu --run-dir runs/smoke
```

Recommended first real run for this machine:

```bash
wheelrl-train --timesteps 2000000 --n-envs 8 --device cpu --command-scale 0.5
```

Inspect learning curves:

```bash
tensorboard --logdir runs
```

Play a trained model:

```bash
wheelrl-play --model <RUN_DIR>/best/best_model.zip
```

The environment ID is `WheelRL-B2WZ1Grip-v0`. Checkpoints produced by the
earlier 22-action/no-gripper environment are intentionally kept separate and
cannot be loaded into this 23-action environment.

## Task-space TCP tracking training

`B2WZ1TrackEnv` (`wheelrl-train-track`) is an RFM-inspired task-space variant
of the base environment: the policy must drive the whole body -- wheels, legs
and arm -- so the Z1 TCP tracks a *moving* 6-DoF pose target in the base
frame. The 86-dimensional observation adds the TCP orientation error
(rotation vector) and the target velocity to the 80-dim base observation.

```bash
# smoke run
wheelrl-train-track --timesteps 16384 --n-envs 2 --device cpu \
  --curriculum 0.5 --motion 0.5 --run-dir runs/track_smoke
```

Recommended first real run for this machine:

```bash
wheelrl-train-track --timesteps 2000000 --n-envs 8 --device cpu \
  --curriculum 1.0 --motion 1.0
```

- `--curriculum 0..1` scales the initial position offset and orientation
  offset of the TCP target at reset (0 = hold the current pose).
- `--motion 0..1` scales the peak target speed (0.10 * motion m/s); the
  target reflects at the reachable-workspace bounds so it stays inside the
  working zone.
- The base velocity command is disabled; forward motion must emerge from the
  wheel actions coordinated with the arm.
- The reward includes position and orientation tracking terms plus a
  multiplicative "sync gate" (a light reward-fusion step) so both pose parts
  must be satisfied together.

## Learned base coordinator for TCP-pose WBC

`wheelrl-train-hier` now trains a deliberately small policy around the
analytic whole-body controller. Its action is:

```text
[base forward-velocity correction, base yaw-rate correction, base height offset]
```

The correction is added to the analytic base planner, so zero policy action is
the original WBC rather than a stopped robot. The policy cannot translate or
rotate the requested TCP pose. The exact panel
pose remains the WBC target at all times. A smooth analytic reachability gate
sets the learned action to zero for arm-local targets and activates it only
outside the comfortable Z1 workspace, near joint limits/poor manipulability,
or when the locked arm joint leaves a persistent orientation error. This is
the intended macro/micro split: RL repositions B2-W; WBC supplies final TCP
precision.

The 78-value observation contains base velocities and gravity, leg/wheel/arm
state, the target and TCP error in the base frame, six arm joint-limit
margins, translational manipulability, WBC base-goal distance, gate state,
wheel slip and the previous 3-D action. Reset randomization now really changes
body mass/inertia, friction, damping, joint-servo gains, sensor noise and
zero-to-two policy steps of command latency.

Smoke-test the interface in WSL:

```bash
wheelrl-train-hier --stage local --timesteps 4096 --n-envs 1 \
  --device cpu --randomization 0 --run-dir /tmp/wheelrl_coord_smoke
```

Recommended first curriculum run (standard MuJoCo physics remains CPU-side):

```bash
wheelrl-train-hier --stage curriculum --timesteps 4000000 \
  --n-envs 8 --device cpu --randomization 0.75
```

On the native Ubuntu RTX 4090 clone, the same run is available through:

```bash
TIMESTEPS=4000000 N_ENVS=12 POLICY_DEVICE=cpu \
  scripts/train_ubuntu_4090.sh coordinator
```

For this state-only MLP, benchmark `POLICY_DEVICE=cuda` rather than assuming it
is faster: the GPU trains the small network but does not accelerate standard
MuJoCo rigid-body simulation. Keep `final_model.zip` paired with its
`vecnormalize.pkl`. Compare a trained policy against pure WBC using identical
episode seeds and randomized dynamics:

```bash
wheelrl-eval-hier \
  --model <RUN_DIR>/final_model.zip \
  --stats <RUN_DIR>/vecnormalize.pkl \
  --stage full --episodes 20 --randomization 1
```

Deploy it in the existing browser-controlled simulator:

```bash
wheelrl-play-wbc --controller hier \
  --hier-model <RUN_DIR>/final_model.zip \
  --hier-stats <RUN_DIR>/vecnormalize.pkl
```

Legacy hierarchical checkpoints output six TCP-residual actions and are
intentionally rejected; otherwise they could reintroduce the steady-state
pose offset this coordinator removes.

## Pull-door training

The door task uses a hierarchical nine-dimensional action:

```text
[base forward velocity, base yaw rate, 6 x Z1 joint target, gripper target]
```

The fixed low-level controller holds the 12 leg joints and tracks the base
velocity through the four wheels at 500 Hz. The six arm actions are residual
joint targets around a door-specific, reachable pre-grasp pose, so zero-mean
exploration remains near the handle. The PPO policy runs at 50 Hz. Its
84-dimensional privileged observation contains robot proprioception, relative
handle pose, direct handle angle/velocity, door angle/velocity, latch and
grasp-assist state, the task phase, and the previous action.

The MuJoCo door has two dynamic joints. The handle has a return spring and the
door has a closer. Until the handle crosses a randomized release angle, a
high-stiffness latch holds the door shut. The opening direction is negative
door angle because this is a pull door. When the fingers close within the
handle capture radius, a compliant MuJoCo equality constraint supplies a
soft-grasp approximation. It can be released by opening the gripper and avoids
requiring a high-fidelity finger-pad model during this first policy stage.

Run a small smoke training:

```bash
wheelrl-train-door \
  --stage reach \
  --timesteps 16384 \
  --n-envs 2 \
  --device cpu \
  --randomization 0 \
  --run-dir runs/door_smoke
```

Run the recommended staged curriculum for this WSL machine:

```bash
wheelrl-train-door \
  --stage curriculum \
  --timesteps 3000000 \
  --n-envs 6 \
  --device cpu \
  --randomization 0.35
```

The curriculum keeps one policy and optimizer, then changes the reset state and
success condition in this order:

1. `reach`: move the end effector into the handle region;
2. `turn`: begin from a collision-free pre-grasp pose, close the gripper, and
   rotate the lever past the latch threshold;
3. `pull`: begin near-grasp and unlatched, then learn coordinated backward motion;
4. `full`: start from the nominal arm pose and perform the complete sequence.

Each stage writes its own best model, checkpoints, evaluation log, and
normalization statistics. In addition to reward-based checkpoints, the trainer
explicitly saves the best policy by deterministic task success rate. After a
curriculum run, prefer:

```text
<RUN_DIR>/full/best_success/best_success_model.zip
<RUN_DIR>/full/best_success/vecnormalize.pkl
```

If no full-stage evaluation has produced a successful policy yet, the trainer
still writes `final_model.zip` and `vecnormalize.pkl` at the run root so that
training can be resumed.

Replay the best full-task policy:

```bash
wheelrl-play-door \
  --model <RUN_DIR>/full/best_success/best_success_model.zip \
  --stats <RUN_DIR>/full/best_success/vecnormalize.pkl \
  --stage full
```

Evaluate 100 randomized episodes:

```bash
wheelrl-eval-door \
  --model <RUN_DIR>/full/best_success/best_success_model.zip \
  --stats <RUN_DIR>/full/best_success/vecnormalize.pkl \
  --episodes 100 \
  --randomization 1
```

Start with the privileged policy above. Once it has a reliable randomized
success rate, use its successful trajectories to train a recurrent student
that removes direct handle angle, door angle, and latch state from the
observation and replaces them with perception and observation history.

The current domain randomization covers door mass and pose, latch release
angle, handle spring, door closer, and hinge damping. Before real deployment,
also calibrate sensor/actuator delay, B2-W base mass and center of mass, Z1
joint friction, and the actual mounting transform.

On the detected WSL machine, six parallel door environments sustain roughly
1,200--1,400 simulation steps/s during PPO training. The GPU installation is
working, but this small MLP does not amortize transfer overhead. Use
`--device cuda` later for a larger or vision-based policy.

The pull-door task is heavier because it adds articulated contacts and a
stateful latch. On this WSL installation, a verified two-environment PPO smoke
run reaches roughly 900 simulation steps/s. Six CPU environments reach roughly
1,400 steps/s and are the
recommended starting point; the 8 GB GPU is available but the small MLP usually
does not offset CPU-to-GPU transfer overhead.

## Important modeling boundary

The arm is mounted at `x=0.211 m, y=0, z=0.105 m` relative to the B2-W base,
so the Z1 base-plate front edge aligns with the front of the dog's main body
(`x=0.25`), leaving a ~0.09 m gap to the lidar mount (`x=0.342`). This is a
documented simulation assumption. Before sim-to-real deployment, replace it
with the measured bracket transform and update the combined mass/inertia
calibration. The included policy is a research starting point, not
a robot-ready safety controller.
