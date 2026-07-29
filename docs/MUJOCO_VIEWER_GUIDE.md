# MuJoCo viewer guide for WheelRL

`wheelrl-play-wbc` uses MuJoCo's passive `Simulate` viewer. The Python control
loop advances physics, so a few generic viewer controls such as Play/Pause and
single-step do not stop or advance WheelRL itself. The browser panel owns all
TCP and gripper commands; the viewer keyboard and mouse are left to MuJoCo.

Press `F1` at any time to display MuJoCo's built-in shortcut help. Hold the
right mouse button over a UI item to display that item's shortcut.

## Window and mouse

| Input | Effect |
|---|---|
| Left drag | Orbit the camera |
| Right drag | Pan vertically |
| Shift + right drag | Pan horizontally |
| Wheel or middle drag | Zoom |
| Left double-click | Select a body |
| Right double-click | Center the camera on that point |
| Ctrl + right double-click | Track the selected body |
| Ctrl + drag | Rotate the selected dynamic body |
| Ctrl + right drag | Translate it in a vertical plane |
| Ctrl + Shift + right drag | Translate it in a horizontal plane |
| Esc | Return to the free camera |
| `[` / `]` | Previous/next fixed camera |
| Ctrl + A | Align the free camera |

Dragging a robot body with Ctrl applies a viewer perturbation; it is useful for
testing recovery but is not a TCP command.

## Top-level UI sections

The exact rows vary slightly by MuJoCo version and model:

| Section | What its controls do in this project |
|---|---|
| File | Reload the XML/model, copy or save model/data, and manage screenshots. Reloading discards current simulation state. |
| Option | Change UI spacing, color scheme and font scaling; these do not affect physics. |
| Simulation | Run/pause, reset, speed, keyframes and history. WheelRL advances a passive viewer from Python, so pause/step controls do not pause its controller loop. Close the window or use **End simulation** in the browser panel instead. |
| Watch | Inspect a named scalar or array element from `mjModel`/`mjData`; it is read-only diagnostics. |
| Physics | Display or change timestep, integrator, contact cone, Jacobian type, solver, iterations, tolerance and physics-disable flags. Changing these alters the experiment and should not be done during policy evaluation. |
| Rendering | Select camera and adjust display-only rendering features such as shadows, reflections, skybox, fog and wireframe. |
| Visualization | Toggle frames, labels, transparency, contacts, contact forces, center of mass, inertia boxes, constraints, BVH and other debug overlays. These normally do not change physics. |
| Group enable | Show/hide geom, site, joint, tendon and actuator groups 0–5. This changes visibility only, not collision or control. |
| Joint | View or edit joint positions with sliders. WheelRL continuously updates the state, so sliders can fight the controller and should be used only while diagnosing a paused/custom run. |
| Control | View or edit actuator controls. WheelRL overwrites `data.ctrl` every control cycle, so these sliders do not provide persistent commands. |
| Equality | Enable/disable equality constraints. In the door scene this includes the compliant gripper–handle grasp constraint; changing it alters task physics. |

## Function keys and keyboard shortcuts

| Key | Effect |
|---|---|
| F1 | Help overlay |
| F2 | Simulation information overlay |
| F3 | Physics profiler |
| F4 | Sensor plots |
| F5 | Full screen |
| F6 | Cycle coordinate-frame visualization |
| F7 | Cycle object labels |
| Tab / Shift + Tab | Show/hide left/right UI panel |
| Space | Play/pause in an active viewer; no physics effect in WheelRL passive mode |
| Right / Left arrow | Forward/backward history step in an active viewer; no control-loop effect here |
| `=` / `-` | Increase/decrease real-time speed setting |
| Backspace | Reset MuJoCo data; the WheelRL controller may immediately command it again |
| Ctrl + C | Copy state |
| Ctrl + L | Reload model |
| Ctrl + P | Screenshot |
| Ctrl + Q | Quit |
| Page Up | Select parent of the selected body |
| `0`…`5` | Toggle geom visibility groups |
| Shift + `0`…`5` | Toggle site visibility groups |

Alt plus a section's underlined letter expands/collapses that UI section.
Rendering shortcuts such as `S` (shadow), `W` (wireframe), `R` (reflection),
`L` (additive), `K` (skybox), and `G` (fog) explain why the old robot-control
letter keys conflicted with the viewer.

Reference: <https://mujoco.readthedocs.io/en/latest/programming/samples.html#simulate>
