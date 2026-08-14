"""Smoke test for the B2-W + Z1 whole-body MPC port."""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np
import pinocchio as pin

from wheelrl.mpc.ocp_whole_body_rnea import make_ocp
from wheelrl.mpc.robot import B2WZ1
from wheelrl.wbc_mpc import B2WZ1MPCController, SCENE_PATH


def check_frames() -> None:
    """Verify the four wheel-contact frames touch the ground at q0."""
    robot = B2WZ1()
    q = robot.q0.copy()
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.framesForwardKinematics(robot.model, robot.data, q)
    for name in ("FL_contact", "FR_contact", "RL_contact", "RR_contact"):
        fid = robot.model.getFrameId(name, type=pin.FrameType.FIXED_JOINT)
        pos = robot.data.oMf[fid].translation
        print(f"{name}: z={pos[2]:.4f} (should be ~0)")
    return robot


def check_ocp(robot: B2WZ1, solver: str = "ipopt") -> None:
    robot.set_gait_sequence("stand", gait_period=0.5)
    ocp = make_ocp(robot, nodes=8, tau_nodes=3, warm_start=True)
    ocp.set_time_params(0.03, 0.12)
    ocp.set_swing_params(0.07, [0.1, -0.2])
    ocp.set_tracking_targets(
        np.zeros(6), np.zeros(3), np.zeros(3)
    )
    from wheelrl.wbc_mpc import SOLVER_ARGS

    ocp.init_solver(solver, SOLVER_ARGS[solver])
    x_init = np.concatenate([robot.q0, np.zeros(robot.nv)])
    ocp.update_params(x_init, 0.0)
    params = ocp.get_solver_params()
    start = time.time()
    sol_x = ocp.solver_function(*params)
    elapsed = time.time() - start
    ocp.retract_stacked_sol(sol_x, retract_all=False)
    stacked_params = np.array(ocp.opti.value(ocp.opti.p, ocp.opti.initial()))
    g, lbg, ubg = ocp.g_data(sol_x, stacked_params)
    cv = ocp.constr_viol_norm_inf(g, lbg, ubg)
    tau0 = ocp.U_prev[0][ocp.tau_idx :]
    print(
        f"OCP {solver}: solve={elapsed * 1000:.1f} ms cv={cv:.2e} "
        f"tau0_norm={np.linalg.norm(tau0):.3f}"
    )
    return ocp


def run_mujoco(solver: str = "ipopt", steps: int = 50) -> None:
    """Closed-loop MuJoCo smoke run with the MPC controller."""
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    ctrl = B2WZ1MPCController(model, data, solver=solver, control_hz=20.0)
    ctrl.reset()
    print("closed-loop run starting")
    fallen = False
    for i in range(steps):
        diag = ctrl.step()
        if i % 10 == 0 or i == steps - 1:
            print(
                f"step={i:3d} t={diag.t:.2f} h={diag.base_height:.3f} "
                f"upright={diag.upright:.3f}"
            )
        if diag.base_height < 0.35 or diag.upright < 0.35:
            fallen = True
            print(f"FALLEN at step {i}")
            break
    print("RESULT:", "OK" if not fallen else "FALLEN")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="ipopt", choices=["ipopt", "fatrop"])
    parser.add_argument("--mujoco", action="store_true", help="run closed-loop MuJoCo")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    robot = check_frames()
    check_ocp(robot, solver=args.solver)
    if args.mujoco:
        run_mujoco(solver=args.solver, steps=args.steps)


if __name__ == "__main__":
    main()
