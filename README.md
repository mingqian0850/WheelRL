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
wheelrl-play --model runs/b2w_z1_gripper_ppo/best/best_model.zip
```

The environment ID is `WheelRL-B2WZ1Grip-v0`. Checkpoints produced by the
earlier 22-action/no-gripper environment are intentionally kept separate and
cannot be loaded into this 23-action environment.

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
runs/b2w_z1_pull_door_ppo/full/best_success/best_success_model.zip
runs/b2w_z1_pull_door_ppo/full/best_success/vecnormalize.pkl
```

If no full-stage evaluation has produced a successful policy yet, the trainer
still writes `final_model.zip` and `vecnormalize.pkl` at the run root so that
training can be resumed.

Replay the best full-task policy:

```bash
wheelrl-play-door \
  --model runs/b2w_z1_pull_door_ppo/full/best_success/best_success_model.zip \
  --stats runs/b2w_z1_pull_door_ppo/full/best_success/vecnormalize.pkl \
  --stage full
```

Evaluate 100 randomized episodes:

```bash
wheelrl-eval-door \
  --model runs/b2w_z1_pull_door_ppo/full/best_success/best_success_model.zip \
  --stats runs/b2w_z1_pull_door_ppo/full/best_success/vecnormalize.pkl \
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

The arm is mounted at `x=0.23 m, y=0, z=0.105 m` relative to the B2-W base.
This is a documented simulation assumption. Before sim-to-real deployment,
replace it with the measured bracket transform and update the combined
mass/inertia calibration. The included policy is a research starting point, not
a robot-ready safety controller.
