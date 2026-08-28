# VBC-style dual RGB-D camera setup

WheelRL uses the camera layout from **Visual Whole-Body Control for Legged
Loco-Manipulation** as its starting point:

- one forward RGB-D camera fixed to the robot base;
- one eye-in-hand RGB-D camera fixed to Z1 `link06` near the gripper;
- a 96 x 54 policy image for each view;
- a 69-degree horizontal field of view;
- three-frame stacks of the target mask and target-segmented depth image.

The paper appendix says "4-step images", but the released implementation sets
`camera_history_len = 3` and reshapes the full two-camera input into 12 channels
(two views x mask/depth x three frames). WheelRL follows the executable public
code; this discrepancy should be kept explicit when comparing checkpoints.

The paper used two Intel RealSense D435 cameras. Its public Isaac Gym
configuration places the forward optical center at `[0.425, 0.04, 0.12]` m in
the B1 base frame and uses `[0.0955, 0.22, -0.03175]` m for the wrist-camera
position in its own Z1 `link06` frame. The first transform is used by the
WheelRL B2-W simulation. The second is **not** copied blindly: WheelRL retains
its existing side-mounted wrist transform because the B1 bracket, B2-W deck,
and the two model frame conventions must be measured before real deployment.

## MuJoCo cameras

| View | Camera name | Parent | Optical-center position | View direction |
|---|---|---|---|---|
| Forward | `head_cam` | `base_link` | `[0.425, 0.04, 0.12]` m | base `+x` |
| Wrist | `z1_cam` | `z1_link06` | `[0.065, 0.09, 0.0]` m | link06 `+x` |

MuJoCo specifies vertical rather than horizontal field of view. For a 96:54
image, a 69-degree horizontal field of view is equivalent to
`fovy=42.272558950` degrees. MuJoCo also looks along the camera frame's `-z`
axis; both camera quaternions account for that convention and keep image-up
aligned with the parent link's `+z` axis.

The fixed camera elements define projection and pose, but render resolution is
selected by the renderer. The visual-policy data path should request exactly
`width=96, height=54` after rendering or downsample a larger raw image to that
size.

## Policy interface

The cameras should belong to the high-level visual policy, not the fast
stabilization controller:

```text
head/wrist RGB-D (30 Hz)
        -> target segmentation and depth preprocessing
        -> 3-frame [mask, segmented depth] stack (96 x 54)
        -> visual policy (about 8-15 Hz)
        -> TCP pose increment + base velocity/yaw + gripper command
        -> WBC / IK tracking (50-100 Hz)
        -> joint control (500 Hz)
```

For sim-to-real training, randomize the following independently:

- each camera translation (start with +/- 10 mm) and rotation (start with
  +/- 5 degrees);
- 80-120 ms visual latency and occasional dropped frames;
- depth holes, blur, quantization, Gaussian noise, and reflective-handle
  dropout;
- the near-depth clip in the 0.18-0.25 m interval when emulating D435;
- segmentation erosion, dilation, partial occlusion, and tracker loss.

The public VBC code clips depth to 2 m, randomizes a near clip around
0.18-0.25 m, and applies erasing, blur, noise, and small image rotations.

## Recommended real hardware for the door task

There are two reasonable choices:

1. **Closest reproduction:** two D435/D435i cameras, one at each simulated
   pose. This minimizes the initial sim-to-real mismatch with VBC.
2. **Door-optimized setup:** D435i on the base plus D405 at the wrist. D405 is
   better for the final 7-50 cm approach, where a D435 can lose close-range
   depth. Keep the simulated camera interface identical and change only the
   wrist depth-noise/range model.

Mount both cameras rigidly. Record `T_base_headcam` and
`T_link06_wristcam` from measured CAD/hand-eye calibration rather than using
the values above as real-robot calibration. Use camera timestamps, robot-state
timestamps, and inference timestamps from one clock. Two cameras should use
reliable USB 3 connections; separate host controllers are preferable when
both stream depth continuously.

For opening a lever-handle pull door, depth alone cannot reliably reveal latch
release or pulling force. Retain joint-torque/contact observations in the
controller and plan for a wrist force/torque sensor or force estimate in the
later contact-rich policy.
