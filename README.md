# WheelRL: Unitree B2-W + Z1 in MuJoCo

This project provides a runnable WSL2 training starter for whole-body control of
a Unitree B2-W carrying a six-axis Unitree Z1 arm and its one-DoF gripper.

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

The same launcher accepts `gripper`. Override its defaults with environment
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
```

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
and stabilization, TCP SE(3), and posture tasks in one weighted damped
least-squares system. The resulting leg and arm velocities are integrated into
references, while four wheel-velocity targets drive the nonholonomic base. All
targets are tracked through the existing 500 Hz torque interface.

Targets inside a conservative Z1 workspace envelope mainly use the arm, with
limited leg/base help from `--base-assist` (default `0.25`). When a target
exceeds that envelope, automatic drive turns and/or translates B2-W while Z1
simultaneously moves toward the reachable part of the same world-frame target.
Use `--no-auto-drive` to disable this behavior.

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

The arm is mounted at `x=0.0 m, y=0, z=0.105 m` relative to the B2-W base
(the center of the top deck, ~0.34 m behind the lidar mount at `x=0.342`),
matching the reference `b2w_z1.urdf` from
[aCodeDog/legged-robots-manipulation](https://github.com/aCodeDog/legged-robots-manipulation)
(`base_arm_joint0` origin `xyz="0 0 0.056"`). This is a documented simulation
assumption. Before sim-to-real deployment, replace it with the measured
bracket transform and update the combined mass/inertia calibration. The
included policy is a research starting point, not
a robot-ready safety controller.
