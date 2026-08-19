"""Step-response benchmark for the WBC TCP tracking loop.

Settles the controller, commands a 10 cm + 8 deg TCP step, and records the
position/orientation error trace at 100 Hz. Reports settling time, overshoot,
oscillation crossings, and final error. Run twice (before/after tuning) and
compare.

Usage:
    .venv/bin/python scripts/bench_wbc_step.py [--out baseline.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from wheelrl.wbc import WBC_MODEL_PATH, B2WZ1WholeBodyController

SETTLE_STEPS = 200
TRACK_STEPS = 500  # 5 s at 100 Hz
STEP_POS = np.array([0.10, 0.04, -0.04], dtype=np.float64)
STEP_RPY = np.deg2rad(np.array([4.0, -6.0, 8.0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/wbc_step.json"))
    parser.add_argument("--seconds", type=float, default=0.0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(WBC_MODEL_PATH))
    data = mujoco.MjData(model)
    ctrl = B2WZ1WholeBodyController(model, data, base_assist=0.25, control_hz=100.0)
    ctrl.reset()

    for _ in range(SETTLE_STEPS):
        ctrl.step()
    ctrl.capture_reference()
    ctrl.set_home_offset(STEP_POS, STEP_RPY)

    trace = []
    for step_index in range(TRACK_STEPS):
        d = ctrl.step()
        trace.append(
            {
                # sim time, not wall time: the benchmark runs faster than real time
                "t": float(step_index / 100.0),
                "pos_err": d.position_error,
                "ori_err": d.orientation_error,
                "upright": d.upright,
                "max_torque": d.max_torque,
            }
        )

    pos = np.array([r["pos_err"] for r in trace])
    ori = np.array([r["ori_err"] for r in trace])

    def settle_time(err: np.ndarray, threshold: float) -> float | None:
        crossed = np.nonzero(err < threshold)[0]
        if not crossed.size:
            return None
        first = crossed[0]
        # require the remainder to stay below the threshold
        if np.all(err[first:] < threshold):
            return first / 100.0
        return None

    def crossings(err: np.ndarray) -> int:
        # count sign changes of (err - final) after the first 0.3 s
        tail = err[int(0.3 * 100):]
        final = float(np.mean(tail[-50:]))
        sign = np.sign(tail - final)
        return int(np.sum(np.abs(np.diff(sign)) > 0) / 2)

    pos_final = float(pos[-50:].mean())
    ori_final = float(np.rad2deg(ori[-50:].mean()))
    result = {
        "pos_final_m": round(pos_final, 5),
        "pos_settle_5mm_s": settle_time(pos, 0.005),
        "pos_settle_2mm_s": settle_time(pos, 0.002),
        "pos_crossings": crossings(pos),
        "ori_final_deg": round(ori_final, 4),
        "ori_settle_1deg_s": settle_time(ori, np.deg2rad(1.0)),
        "ori_crossings": crossings(ori),
        "min_upright": round(float(trace[-1]["upright"]), 4),
        "max_torque": round(float(max(r["max_torque"] for r in trace)), 1),
    }
    print(json.dumps(result, indent=2))
    args.out.write_text(json.dumps({"metrics": result, "trace": trace}, indent=2))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
