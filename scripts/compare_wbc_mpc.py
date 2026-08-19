"""Head-to-head: velocity-level geometric WBC vs inverse-dynamics MPC.

Runs both controllers on the same wbc_scene.xml through identical target
protocols (5 s of simulated time) and reports steady-state tracking error,
uprightness, base height and max torque.

Run inside the wheelrl-mpc environment:

    ~/.local/share/mamba/envs/wheelrl-mpc/bin/python scripts/compare_wbc_mpc.py
"""

from __future__ import annotations

import numpy as np
import mujoco

from wheelrl.wbc import WBC_MODEL_PATH, B2WZ1WholeBodyController
from wheelrl.wbc_mpc import B2WZ1MPCController

OFFSETS = {
    "rest": (np.zeros(3), np.zeros(3)),
    "small": (np.array([0.05, 0.02, -0.02]), np.zeros(3)),
    "larger": (
        np.array([0.10, 0.04, -0.04]),
        np.deg2rad(np.array([4.0, -6.0, 8.0])),
    ),
}
SIM_SECONDS = 5.0


def run_wbc(offset, rpy):
    model = mujoco.MjModel.from_xml_path(str(WBC_MODEL_PATH))
    data = mujoco.MjData(model)
    ctrl = B2WZ1WholeBodyController(model, data, base_assist=0.25, control_hz=100.0)
    ctrl.reset()
    for _ in range(200):  # settle 2 s
        ctrl.step()
    ctrl.capture_reference()
    ctrl.set_home_offset(offset, rpy)
    max_torque = 0.0
    min_upright = 1.0
    for _ in range(int(SIM_SECONDS * 100)):
        d = ctrl.step()
        max_torque = max(max_torque, d.max_torque)
        min_upright = min(min_upright, d.upright)
    d = ctrl.diagnostics()
    return {
        "position_error_m": d.position_error,
        "orientation_error_deg": np.rad2deg(d.orientation_error),
        "upright": d.upright,
        "min_upright": min_upright,
        "base_height": float(data.qpos[2]),
        "max_torque_Nm": max_torque,
    }


def run_mpc(offset, rpy):
    model = mujoco.MjModel.from_xml_path(str(WBC_MODEL_PATH))
    data = mujoco.MjData(model)
    ctrl = B2WZ1MPCController(
        model, data, control_hz=20.0, solver="fatrop", load_compiled_solver=True
    )
    ctrl.reset()
    ctrl.step()  # single settle step (matches play-wbc MPC mode)
    ctrl.capture_reference()
    ctrl.set_home_offset(offset, rpy)
    max_torque = 0.0
    min_upright = 1.0
    for _ in range(int(SIM_SECONDS * 20)):
        d = ctrl.step()
        max_torque = max(max_torque, d.max_torque)
        min_upright = min(min_upright, d.upright)
    d = ctrl.diagnostics()
    return {
        "position_error_m": d.position_error,
        "orientation_error_deg": np.rad2deg(d.orientation_error),
        "upright": d.upright,
        "min_upright": min_upright,
        "base_height": float(data.qpos[2]),
        "max_torque_Nm": max_torque,
    }


def main() -> None:
    print(f"{'scenario':<8} {'method':<5} {'pos_err[m]':>10} {'ori_err[deg]':>12} "
          f"{'upright':>8} {'min_up':>7} {'height[m]':>9} {'max_tau[Nm]':>11}")
    for name, (offset, rpy) in OFFSETS.items():
        for method, fn in (("WBC", run_wbc), ("MPC", run_mpc)):
            try:
                r = fn(offset, rpy)
                print(f"{name:<8} {method:<5} {r['position_error_m']:>10.4f} "
                      f"{r['orientation_error_deg']:>12.2f} {r['upright']:>8.3f} "
                      f"{r['min_upright']:>7.3f} {r['base_height']:>9.3f} "
                      f"{r['max_torque_Nm']:>11.1f}")
            except Exception as e:
                print(f"{name:<8} {method:<5} FAILED: {e}")


if __name__ == "__main__":
    main()
