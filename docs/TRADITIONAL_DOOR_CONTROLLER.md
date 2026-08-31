# Traditional B2-W + Z1 pull-door controller

This baseline opens the MuJoCo pull door without a learned policy. It assigns
precise free-space motion to Z1 and the opening work to B2-W:

```text
Mink arm-only pose IK
    -> Z1 joint references + gravity-compensated PD
    -> grasp and rotate the lever
    -> latch-release event
    -> low-stiffness Z1 joint impedance
    -> B2-W reverse velocity + door-angle yaw following
```

Run it in the viewer:

```bash
source .venv/bin/activate
wheelrl-play-door-traditional
```

Run the same episode as a fast headless check:

```bash
wheelrl-play-door-traditional --headless --no-realtime
```

Use `--domain-randomization 1.0` to exercise the door mass, frame pose, latch
angle, handle spring, closer, and damping ranges already implemented by
`B2WZ1DoorEnv`. `--pull-speed` changes the positive speed magnitude; the
controller applies it as a reverse command only after the latch releases.

## Why Mink

[Mink](https://github.com/kevinzakka/mink) is a general Python differential-IK
library built directly on MuJoCo. The controller uses `FrameTask` for the
`z1_ee` TCP, `ConfigurationLimit` and `VelocityLimit` for the Z1 joints, and
the v1.3 `DofFreezingTask` as an exact constraint. Freezing is essential: the
full scene also contains a floating base, legs, wheels, gripper, door, and
handle, and an unconstrained IK solve could move those DoFs instead of solving
with the arm.

The current dependency is `mink>=1.3,<2`; the lock file resolves Mink 1.3.0,
DAQP 0.9.1, and MuJoCo 3.10.0. Mink returns a kinematic velocity/reference, not
a motor torque. WheelRL copies only the six solved Z1 positions into a
gravity-compensated joint servo and lets `mj_step` evolve the real dynamics.

Other general Python choices include Pinocchio/Pink, Robotics Toolbox,
PyKDL, and IKPy. Mink is used here because it shares the exact MJCF model and
frame names with the simulator, so there is no second URDF model or joint-map
to keep synchronized.

## State machine

| Phase | Arm | Base | Transition |
|---|---|---|---|
| `settle` | Hold initial joints | Wheels stopped | 1 s elapsed |
| `approach` | Smooth Mink IK reference to the handle | Wheels stopped | TCP enters 8.5 cm grasp region |
| `grasp` | Hold pre-grasp pose; close gripper | Wheels stopped | Soft grasp attached and settled |
| `unlatch` | Mink IK to the depressed-handle pose | Wheels stopped | Handle passes randomized release angle |
| `pull` | Low-stiffness impedance; gravity compensation remains | Reverse and yaw with door angle | Door angle reaches -0.72 rad |
| `success` | Compliant hold | Wheels stopped | Terminal |

The pre-grasp and unlatch references are solved at runtime from the actual
randomized handle pose. During unlatching, the desired wrist orientation is
the TCP-to-handle relative rotation captured at grasp, transported by the
rotating handle frame.

During pulling, the desired base heading is

```text
yaw_desired = yaw_at_release + (door_angle - door_angle_at_release)
```

and a bounded yaw-rate servo tracks it while the wheels reverse. This follows
the handle's circular hinge trajectory much better than driving straight
backward. The yellow RGB-axis marker shows the active TCP target. Once pulling
starts it follows the handle, emphasizing that the arm is now a compliant
connection rather than a rigid world-frame pose servo.

## Current reproducible result

On 2026-08-31, the headless baseline completed 30/30 sampled episodes: ten
seeds each at door-randomization strengths 0.0, 0.5, and 1.0. The nominal seed
0 episode reached `door_angle=-0.721 rad` in 19.35 simulated seconds with
`upright=0.999`. This is a regression result for the present simplified scene,
not a sim-to-real success claim.

The simulated grasp is still an idealized soft MuJoCo `connect` equality. It
models limited compliance but not realistic finger friction, slip, handle
geometry, or gripper structural limits. Pulling work comes from the base, yet
the force path remains door -> handle -> gripper -> Z1 -> B2-W. Real-hardware
work therefore still requires measured door forces, wrist/estimated wrench
limits, overload release, collision checking, an emergency stop, and preferably
a wrist force/torque sensor or a chassis-anchored load-bearing hook/tether.

## Relevant files

- `src/wheelrl/traditional_door.py`: Mink wrapper, state machine, impedance,
  wheel-arc controller, diagnostics.
- `src/wheelrl/play_door_traditional.py`: viewer/headless launcher.
- `src/wheelrl/assets/b2w_z1/door_scene.xml`: door, latch, soft grasp, target
  marker.
- `tests/test_traditional_door.py`: non-arm IK freeze and full opening test.

