"""Interactively command a Z1 TCP SE(3) target using whole-body control."""

from __future__ import annotations

import argparse
import queue
import time
from typing import Any

import mujoco
import numpy as np

from wheelrl.control_panel import WBCControlPanel
from wheelrl.wbc import (
    SPEED_PROFILE_SCALES,
    WBC_MODEL_PATH,
    B2WZ1WholeBodyController,
    rotation_rpy,
)

JOG_STEPS = {
    "precision": (0.002, np.deg2rad(0.5)),
    "normal": (0.010, np.deg2rad(2.0)),
    "fast": (0.030, np.deg2rad(5.0)),
}


def _print_pose(controller: B2WZ1WholeBodyController) -> None:
    diagnostics = controller.diagnostics()
    position = " ".join(f"{value:+.4f}" for value in controller.tcp_position)
    target = " ".join(f"{value:+.4f}" for value in controller.target_position)
    print(
        f"tcp=[{position}] target=[{target}] "
        f"position_error={diagnostics.position_error:.4f} m "
        f"orientation_error={np.rad2deg(diagnostics.orientation_error):.2f} deg "
        f"upright={diagnostics.upright:.3f} "
        f"speed={controller.speed_profile}"
        f"({controller.speed_scale:.2f}x/"
        f"{controller.rotation_speed_scale:.2f}x-rot) "
        f"auto_drive={'on' if diagnostics.mobile_base_active else 'off'} "
        f"base_goal={diagnostics.base_goal_distance:.3f} m"
    )


def _handle_key(key: int, controller: B2WZ1WholeBodyController) -> None:
    try:
        letter = chr(key).upper()
    except ValueError:
        return

    translation_step, rotation_step = JOG_STEPS[controller.speed_profile]
    translations = {
        "W": np.array([translation_step, 0.0, 0.0]),
        "S": np.array([-translation_step, 0.0, 0.0]),
        "A": np.array([0.0, translation_step, 0.0]),
        "D": np.array([0.0, -translation_step, 0.0]),
        "R": np.array([0.0, 0.0, translation_step]),
        "F": np.array([0.0, 0.0, -translation_step]),
    }
    rotations = {
        "U": (np.array([1.0, 0.0, 0.0]), rotation_step),
        "O": (np.array([1.0, 0.0, 0.0]), -rotation_step),
        "I": (np.array([0.0, 1.0, 0.0]), rotation_step),
        "K": (np.array([0.0, 1.0, 0.0]), -rotation_step),
        "J": (np.array([0.0, 0.0, 1.0]), rotation_step),
        "L": (np.array([0.0, 0.0, 1.0]), -rotation_step),
    }
    if letter in translations:
        controller.nudge_position_base(translations[letter])
    elif letter in rotations:
        axis, angle = rotations[letter]
        controller.nudge_orientation(axis, angle)
    elif letter == "H":
        controller.go_home()
    elif letter == "C":
        controller.toggle_gripper()
    elif letter == "P":
        _print_pose(controller)


def _vector(
    command: dict[str, Any],
    name: str,
    *,
    limit: float,
) -> np.ndarray:
    value = np.asarray(command.get(name), dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain three finite values")
    if np.any(np.abs(value) > limit):
        raise ValueError(f"{name} exceeds the safety limit {limit}")
    return value


def _apply_panel_command(
    command: dict[str, Any],
    controller: B2WZ1WholeBodyController,
) -> bool:
    """Apply one browser command; return False when play should stop."""
    action = command.get("action")
    if action == "set_pose":
        position_offset = _vector(command, "position_offset", limit=3.0)
        rpy_offset_deg = _vector(command, "rpy_offset_deg", limit=360.0)
        controller.set_home_offset(position_offset, np.deg2rad(rpy_offset_deg))
    elif action == "home":
        controller.go_home()
    elif action == "gripper":
        closed = command.get("closed")
        if not isinstance(closed, bool):
            raise ValueError("gripper closed must be a boolean")
        controller.set_gripper(closed=closed)
    elif action == "auto_drive":
        enabled = command.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError("auto_drive enabled must be a boolean")
        controller.auto_drive = enabled
    elif action == "speed_profile":
        profile = command.get("profile")
        if not isinstance(profile, str):
            raise ValueError("speed profile must be a string")
        controller.set_speed_profile(profile)
    elif action == "print_pose":
        _print_pose(controller)
    elif action == "quit":
        return False
    else:
        raise ValueError(f"unknown panel action: {action!r}")
    return True


def _panel_state(controller: B2WZ1WholeBodyController) -> dict[str, Any]:
    diagnostics = controller.diagnostics()
    relative_rotation = controller.home_tcp_rotation.T @ controller.target_rotation
    return {
        "position_offset": (
            controller.target_position - controller.home_tcp_position
        ).tolist(),
        "rpy_offset_deg": np.rad2deg(rotation_rpy(relative_rotation)).tolist(),
        "tcp_position": controller.tcp_position.tolist(),
        "target_position": controller.target_position.tolist(),
        "position_error": diagnostics.position_error,
        "orientation_error_deg": float(
            np.rad2deg(diagnostics.orientation_error)
        ),
        "mobile_base_active": diagnostics.mobile_base_active,
        "base_goal_distance": diagnostics.base_goal_distance,
        "gripper_closed": controller.gripper_closed,
        "auto_drive": controller.auto_drive,
        "speed_profile": controller.speed_profile,
        "speed_scale": controller.speed_scale,
        "rotation_speed_scale": controller.rotation_speed_scale,
    }


def _viewer_overlay(
    controller: B2WZ1WholeBodyController,
    panel_url: str | None,
) -> list[tuple[object, object, str, str]]:
    """Build viewer text while caller owns the data lock.

    Handle.set_texts() takes the native viewer mutex internally, so it must be
    called only after releasing viewer.lock().
    """
    diagnostics = controller.diagnostics()
    lines = [
        f"TCP error  {diagnostics.position_error:.4f} m",
        f"Rotation   {np.rad2deg(diagnostics.orientation_error):.2f} deg",
        "Base       "
        + ("moving" if diagnostics.mobile_base_active else "arm workspace"),
        "Gripper    " + ("closed" if controller.gripper_closed else "open"),
        f"Speed      {controller.speed_profile} "
        f"({controller.speed_scale:.2f}x, "
        f"rot {controller.rotation_speed_scale:.2f}x)",
    ]
    texts = [
        (
            mujoco.mjtFontScale.mjFONTSCALE_100,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "WheelRL WBC",
            "\n".join(lines),
        )
    ]
    if panel_url is not None:
        texts.append(
            (
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                "TCP and gripper controls",
                panel_url,
            )
        )
    return texts


def _run_settle(
    controller: B2WZ1WholeBodyController,
    seconds: float,
    viewer: object | None,
) -> None:
    steps = int(round(seconds / controller.control_dt))
    for _ in range(steps):
        start = time.monotonic()
        if viewer is not None:
            with viewer.lock():
                controller.step()
            viewer.sync()
            elapsed = time.monotonic() - start
            if elapsed < controller.control_dt:
                time.sleep(controller.control_dt - elapsed)
        else:
            controller.step()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--base-assist", type=float, default=0.25)
    parser.add_argument(
        "--speed-profile",
        choices=tuple(SPEED_PROFILE_SCALES),
        default="normal",
        help="initial motion limits; the browser can change this live",
    )
    parser.add_argument(
        "--no-auto-drive",
        action="store_true",
        help="disable automatic wheel motion for targets outside arm reach",
    )
    parser.add_argument(
        "--target-offset",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        default=(0.0, 0.0, 0.0),
        help="world-frame metres from the settled TCP pose",
    )
    parser.add_argument(
        "--rpy-offset-deg",
        type=float,
        nargs=3,
        metavar=("ROLL", "PITCH", "YAW"),
        default=(0.0, 0.0, 0.0),
        help="local RPY degrees from the settled TCP orientation",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without WSLg and print tracking diagnostics",
    )
    parser.add_argument(
        "--controls",
        choices=("panel", "keyboard", "both", "none"),
        help="input mode; defaults to panel with a viewer and none headless",
    )
    parser.add_argument(
        "--panel-port",
        type=int,
        default=8765,
        help="localhost browser-control port; use 0 to select a free port",
    )
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="print the panel URL without trying to open a browser",
    )
    args = parser.parse_args()
    controls = args.controls or ("none" if args.headless else "panel")
    keyboard_enabled = controls in ("keyboard", "both")
    panel_enabled = controls in ("panel", "both")

    model = mujoco.MjModel.from_xml_path(str(WBC_MODEL_PATH))
    data = mujoco.MjData(model)
    controller = B2WZ1WholeBodyController(
        model,
        data,
        control_hz=args.control_hz,
        base_assist=args.base_assist,
        auto_drive=not args.no_auto_drive,
        speed_profile=args.speed_profile,
    )
    controller.reset()

    key_queue: queue.SimpleQueue[int] = queue.SimpleQueue()

    def key_callback(key: int) -> None:
        key_queue.put(key)

    viewer = None
    panel = None
    try:
        if not args.headless:
            from mujoco import viewer as mujoco_viewer

            viewer = mujoco_viewer.launch_passive(
                model,
                data,
                key_callback=key_callback if keyboard_enabled else None,
            )
            viewer.cam.lookat[:] = np.array([0.1, 0.0, 0.55])
            viewer.cam.distance = 2.4
            viewer.cam.azimuth = -125.0
            viewer.cam.elevation = -18.0

        print(f"Settling the four-wheel stance for {args.settle_seconds:.1f} s...")
        _run_settle(controller, args.settle_seconds, viewer)
        if viewer is not None:
            with viewer.lock():
                controller.capture_reference()
                controller.set_home_offset(
                    np.asarray(args.target_offset, dtype=np.float64),
                    np.deg2rad(np.asarray(args.rpy_offset_deg, dtype=np.float64)),
                )
        else:
            controller.capture_reference()
            controller.set_home_offset(
                np.asarray(args.target_offset, dtype=np.float64),
                np.deg2rad(np.asarray(args.rpy_offset_deg, dtype=np.float64)),
            )

        if panel_enabled:
            panel = WBCControlPanel(port=args.panel_port)
            panel.publish(_panel_state(controller))
            panel.start()
            print(f"TCP and gripper control panel: {panel.url}")
            if not args.no_open_browser and not panel.open_browser():
                print("Could not open a browser automatically; open the URL above.")
        if keyboard_enabled:
            print(
                "TCP target controls: W/S forward/back, A/D left/right, R/F z; "
                "U/O roll, I/K pitch, J/L yaw; H home, C gripper, P pose."
            )
            print(
                "Keyboard mode shares keys with MuJoCo and is intended only "
                "as a legacy option."
            )
        if viewer is not None:
            with viewer.lock():
                _print_pose(controller)
                overlay = _viewer_overlay(
                    controller,
                    panel.url if panel is not None else None,
                )
            viewer.set_texts(overlay)
        else:
            _print_pose(controller)

        deadline = time.monotonic() + args.seconds
        next_report = time.monotonic() + 1.0
        next_ui_update = time.monotonic()
        keep_running = True
        while time.monotonic() < deadline and (
            viewer is None or viewer.is_running()
        ) and keep_running:
            start = time.monotonic()
            if viewer is not None:
                panel_state = None
                overlay = None
                with viewer.lock():
                    while not key_queue.empty():
                        _handle_key(key_queue.get_nowait(), controller)
                    if panel is not None:
                        for command in panel.drain_commands():
                            try:
                                keep_running = _apply_panel_command(
                                    command,
                                    controller,
                                )
                            except ValueError as error:
                                print(f"Ignored panel command: {error}")
                            if not keep_running:
                                break
                    if keep_running:
                        controller.step()
                        if time.monotonic() >= next_ui_update:
                            overlay = _viewer_overlay(
                                controller,
                                panel.url if panel is not None else None,
                            )
                            if panel is not None:
                                panel_state = _panel_state(controller)
                            next_ui_update = time.monotonic() + 0.10
                # Both methods take their own native/Python locks. Calling
                # either while viewer.lock() is held deadlocks the GUI.
                if overlay is not None:
                    viewer.set_texts(overlay)
                if panel is not None and panel_state is not None:
                    panel.publish(panel_state)
                viewer.sync()
                elapsed = time.monotonic() - start
                if elapsed < controller.control_dt:
                    time.sleep(controller.control_dt - elapsed)
            else:
                while not key_queue.empty():
                    _handle_key(key_queue.get_nowait(), controller)
                if panel is not None:
                    for command in panel.drain_commands():
                        try:
                            keep_running = _apply_panel_command(
                                command,
                                controller,
                            )
                        except ValueError as error:
                            print(f"Ignored panel command: {error}")
                        if not keep_running:
                            break
                if keep_running:
                    controller.step()
                    if panel is not None:
                        panel.publish(_panel_state(controller))
                if time.monotonic() >= next_report:
                    _print_pose(controller)
                    next_report = time.monotonic() + 1.0
                if panel is not None:
                    elapsed = time.monotonic() - start
                    if elapsed < controller.control_dt:
                        time.sleep(controller.control_dt - elapsed)
        if viewer is not None and viewer.is_running():
            with viewer.lock():
                _print_pose(controller)
        else:
            _print_pose(controller)
    finally:
        if panel is not None:
            panel.close()
        if viewer is not None:
            if viewer.is_running():
                viewer.close()
            # close() signals the native render thread asynchronously. Giving
            # it time to leave avoids a WSL/glibc teardown race; querying the
            # handle with is_running() after close is itself unsafe.
            time.sleep(0.25)


if __name__ == "__main__":
    main()
