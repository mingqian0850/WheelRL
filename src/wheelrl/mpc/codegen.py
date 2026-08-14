"""Generate the fatrop C solver for the B2-W + Z1 whole-body MPC.

Usage:
    python -m wheelrl.mpc.codegen

This writes ``solver_function.c`` in the current working directory. Compile it
with the CMake project in ``assets/b2w_z1/mpc/codegen`` (mirrors the procedure
from wb-mpc-locoman):

    export CONDA_PREFIX=$CONDA_PREFIX   # wheelrl-mpc env
    cd <dir>/codegen && mkdir -p build && cd build
    cmake .. && make -j4
    cp libsolver_function.so ../lib/
"""

from __future__ import annotations

import numpy as np

from wheelrl.mpc.ocp_whole_body_rnea import make_ocp
from wheelrl.mpc.robot import B2WZ1
from wheelrl.wbc_mpc import SOLVER_ARGS


def main() -> None:
    robot = B2WZ1()
    robot.set_gait_sequence("stand", gait_period=0.5)
    ocp = make_ocp(robot, nodes=10, tau_nodes=3, warm_start=True)
    ocp.set_time_params(0.03, 0.12)
    ocp.set_swing_params(0.07, [0.1, -0.2])
    ocp.set_tracking_targets(np.zeros(6), np.zeros(3), np.zeros(3))
    # structure_detection "none" is required for the casadi 3.7 fatrop plugin
    # with this OCP layout.
    ocp.init_solver("fatrop", SOLVER_ARGS["fatrop"])
    ocp.compile_solver()
    print("wrote solver_function.c")


if __name__ == "__main__":
    main()
