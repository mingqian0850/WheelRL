# Whole-Body Inverse-Dynamics MPC for B2-W + Z1

Port of [wb-mpc-locoman](https://github.com/lukasmolnar/wb-mpc-locoman)
(ETH, *Whole-Body Inverse Dynamics MPC for Legged Loco-Manipulation*, RA-L 2025)
adapted from the four-legged Unitree B2 to the **wheel-legged Unitree B2-W**
carrying a Unitree Z1 arm.

## What was ported

- `dynamics.py` — full-order RNEA dynamics built on Pinocchio CasADi, with
  external wrenches at the four wheel contact frames and the arm end-effector.
- `ocp.py` / `ocp_whole_body_rnea.py` — receding-horizon OCP (the original
  wb-mpc-locoman implementation, kept verbatim so the fatrop structure
  detection and codegen path work) with:
  - RNEA as the dynamics constraint (`tau_rnea[:6] == 0`, joint torques equal
    RNEA output),
  - friction cones and zero-slip contact at the four wheel-ground contact
    points,
  - joint torque limits on the first `tau_nodes` nodes,
  - joint position/velocity limits,
  - base velocity, arm end-effector velocity and force tracking.
- `robot.py` — Pinocchio model loaded from `b2w_z1.urdf` (dynamics only, no
  meshes required). The URDF was taken from
  [aCodeDog/legged-robots-manipulation](https://github.com/aCodeDog/legged-robots-manipulation)
  and modified:
  - `continuous` wheel joints fixed to `revolute` so Pinocchio parses them as
    plain 1-DoF revolute joints,
  - four `*_contact` fixed frames inserted below each wheel, positioned at the
    MuJoCo wheel collision bottom so MPC contacts match the physics scene,
  - base mass bumped to match the MuJoCo model total mass.
- `gait_sequence.py` — original gait code; B2-W wheels are always in contact
  (`stand` schedule keeps all four contacts, no swing).
- `wbc_mpc.py` — `B2WZ1MPCController` bridges MuJoCo state to the MPC and maps
  the 22 optimized joint torques back onto the 23 MuJoCo actuators (the
  gripper keeps its own PD target).

## Differences from the original (B2 -> B2-W)

| Aspect | wb-mpc-locoman (B2+Z1) | This port (B2-W+Z1) |
|---|---|---|
| Contact | 4 feet, trot/walk/stand gait | 4 wheels, always in contact (`stand`) |
| Wheel joints | feet are passive | 4 revolute wheel joints with zero position weight (free rolling), no-slip enforced by contact constraints |
| Quaternion | Pinocchio xyzw | handled in `_mujoco_to_pin_q` (MuJoCo wxyz) |
| Joint mapping | — | by-name map between MJCF and URDF |
| Gripper | — | separate PD, not optimized |

## Verification

The `wheelrl` package is installed (editable) in the `wheelrl-mpc`
micromamba environment together with MuJoCo, Pinocchio and CasADi:

```bash
# Single-shot OCP solve (ipopt, ~5 s; fatrop interpreted, ~0.1 s)
micromamba activate wheelrl-mpc
python -m wheelrl.mpc.smoke_test

# Closed-loop MuJoCo standing (ipopt, ~6 s per control step)
python -m wheelrl.mpc.smoke_test --mujoco --steps 25
```

Closed-loop result (i5-13600KF, ipopt): robot settles at base height ≈ 0.65 m
and stays upright over 25 steps without falling.

## Compiled fatrop solver (near real time)

`python -m wheelrl.mpc.codegen` writes `solver_function.c`; the CMake project
in `assets/b2w_z1/mpc/codegen` builds `libsolver_function.so` (see the script
docstring). With the compiled solver the OCP solves in **~50 ms/step on this
machine** (≈19 Hz), and `B2WZ1MPCController` loads it by default.

## Interactive control (play-wbc)

The existing browser/keyboard whole-body-control interface accepts the MPC
backend:

```bash
# Headless MPC run with a small TCP target offset
micromamba activate wheelrl-mpc
python -m wheelrl.play_wbc --controller mpc --headless \
  --target-offset 0.08 0.03 0.02

# With viewer + browser panel (recommended)
python -m wheelrl.play_wbc --controller mpc
```

> Note: `python -m wheelrl.play_wbc` (without `--controller`) still runs the
> original velocity-level WBC, and also works from this environment.

The MPC backend reuses the panel actions (pose offsets, home, gripper, speed
profile) through a compatibility layer. TCP **position** tracking is enabled
via an outer P loop feeding arm velocity targets into the OCP. TCP
**orientation** tracking is not yet part of the OCP (the original wb-mpc OCP
only tracks linear EE velocity), so orientation drifts during large moves.
`auto_drive` is accepted but has no effect yet (wheels roll via the contact
constraints).

With the compiled fatrop solver the interactive MPC run advances at ~20 Hz.

## Why the original OCP code is used verbatim

The first port rewrote the OCP with an equivalent constraint layout, but the
CasADi 3.7 fatrop plugin rejected it under `structure_detection: "auto"` (and
the compiled solver corrupted memory). Keeping the original wb-mpc-locoman OCP
code (imports adapted, wheel joints added to the weights) preserves the exact
constraint ordering that fatrop's structure detection expects, so both the
interpreted fatrop path (~100 ms) and the codegen-compiled path (~50 ms) work.

## Performance notes

- **ipopt** works reliably but is slow (~5-7 s per solve), far from real time.
- **fatrop** (interpreted, CasADi 3.7) initializes with
  `structure_detection: "none"` and solves in ~4 s but can crash with a
  corrupted double-linked list on exit. The production path is the
  codegen-compiled fatrop solver (as in the original repo); see
  `~/wb-mpc-locoman/RUN_WSL.md` for the compile procedure.

## Next steps

1. Compile the fatrop solver for this OCP (codegen) to reach ~20-80 Hz.
2. Add MPC frequency reduction (larger `dt_min`), warm-start interpolation,
   and a state estimator before real-hardware use.
3. Port the door-opening and TCP-tracking tasks onto this controller.
