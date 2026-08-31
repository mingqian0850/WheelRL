"""Play the deterministic Mink-IK + compliant-base pull-door controller."""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from typing import Any

import mujoco.viewer

from wheelrl.envs import B2WZ1DoorEnv
from wheelrl.traditional_door import (
    TraditionalDoorConfig,
    TraditionalDoorController,
    TraditionalDoorPhase,
)


def _status_line(controller: TraditionalDoorController) -> str:
    diagnostic = controller.diagnostics()
    return (
        f"t={diagnostic.elapsed_time:5.2f}s phase={diagnostic.phase.value:<8} "
        f"door={diagnostic.door_angle:+.3f} handle={diagnostic.handle_angle:+.3f} "
        f"tcp_error={diagnostic.tcp_handle_distance:.3f}m "
        f"grasp={int(diagnostic.grasp_attached)} latch={int(diagnostic.latch_released)} "
        f"base=[{diagnostic.base_forward_command:+.2f}m/s, "
        f"{diagnostic.base_yaw_command:+.2f}rad/s]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="pace simulation to wall time (default: viewer only)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--pull-speed", type=float, default=0.10)
    parser.add_argument(
        "--domain-randomization",
        type=float,
        default=0.0,
        help="door randomization strength in [0, 1]",
    )
    args = parser.parse_args()

    realtime = (not args.headless) if args.realtime is None else args.realtime
    env = B2WZ1DoorEnv(
        render_mode=None,
        task_stage="full",
        domain_randomization=args.domain_randomization,
        max_episode_steps=100_000,
    )
    controller = TraditionalDoorController(
        env,
        TraditionalDoorConfig(pull_speed=args.pull_speed),
    )
    controller.reset(seed=args.seed)

    viewer_context: Any
    if args.headless:
        viewer_context = nullcontext(None)
    else:
        viewer_context = mujoco.viewer.launch_passive(
            env.model,
            env.data,
            show_left_ui=True,
            show_right_ui=True,
        )

    last_phase = controller.phase
    next_report = 0.0
    terminal_time: float | None = None
    aborted = False
    try:
        with viewer_context as viewer:
            while controller.elapsed_time < args.max_seconds:
                if viewer is not None and not viewer.is_running():
                    aborted = True
                    break
                start = time.monotonic()
                diagnostic = controller.step()
                if viewer is not None:
                    viewer.sync()

                if diagnostic.phase != last_phase:
                    print(_status_line(controller), flush=True)
                    last_phase = diagnostic.phase
                elif diagnostic.elapsed_time >= next_report:
                    print(_status_line(controller), flush=True)
                    next_report = diagnostic.elapsed_time + 1.0

                if controller.terminal:
                    if terminal_time is None:
                        terminal_time = controller.elapsed_time
                    if (
                        args.headless
                        or controller.elapsed_time - terminal_time >= args.hold_seconds
                    ):
                        break

                elapsed = time.monotonic() - start
                if realtime and elapsed < controller.dt:
                    time.sleep(controller.dt - elapsed)
    finally:
        env.close()

    diagnostic = controller.diagnostics()
    if aborted:
        print("viewer closed by user")
        return
    if diagnostic.phase == TraditionalDoorPhase.SUCCESS:
        print(
            f"SUCCESS: opened door to {diagnostic.door_angle:+.3f} rad in "
            f"{diagnostic.elapsed_time:.2f} s; upright={diagnostic.upright:.3f}"
        )
        return
    if diagnostic.phase == TraditionalDoorPhase.FAILURE:
        raise SystemExit(f"FAILURE: {diagnostic.failure_reason}")
    raise SystemExit(
        f"TIMEOUT: phase={diagnostic.phase.value}, door={diagnostic.door_angle:+.3f} rad"
    )


if __name__ == "__main__":
    main()
