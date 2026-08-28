import mujoco
import numpy as np

from wheelrl.play_wbc import (
    _apply_panel_command,
    _handle_key,
    _panel_state,
    _viewer_overlay,
)
from wheelrl.wbc import (
    WBC_MODEL_PATH,
    B2WZ1WholeBodyController,
    rotation_rpy,
    rotation_vector,
    rpy_rotation,
    solve_box_least_squares,
)


def _non_floor_contact_pairs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> set[tuple[str | None, str | None]]:
    pairs = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        body_names = []
        for geom_id in (contact.geom1, contact.geom2):
            body_id = int(model.geom_bodyid[geom_id])
            body_names.append(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            )
        if "world" not in body_names:
            pairs.add((body_names[0], body_names[1]))
    return pairs


def test_rotation_vector_and_rpy_round_trip() -> None:
    rpy = np.deg2rad(np.array([5.0, -8.0, 12.0]))
    rotation = rpy_rotation(rpy)
    assert np.linalg.norm(rotation_vector(rotation)) > 0.1
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
    assert np.isclose(np.linalg.det(rotation), 1.0)
    assert np.allclose(rotation_rpy(rotation), rpy, atol=1.0e-12)


def test_box_least_squares_reoptimizes_after_a_bound_activates() -> None:
    # The unconstrained solution is [2, -2]. Once x[0] reaches its upper
    # bound, a real bounded solve re-optimizes x[1] to -1; solve-and-clip
    # would incorrectly leave it at -2.
    matrix = np.array([[1.0, 0.0], [1.0, 1.0]])
    target = np.array([2.0, 0.0])
    solution = solve_box_least_squares(
        matrix,
        target,
        np.array([-10.0, -10.0]),
        np.array([1.0, 10.0]),
    )
    assert np.allclose(solution, [1.0, -1.0], atol=1.0e-10)


def test_panel_sets_pose_gripper_and_auto_drive_without_viewer_keys() -> None:
    model = mujoco.MjModel.from_xml_path(str(WBC_MODEL_PATH))
    data = mujoco.MjData(model)
    controller = B2WZ1WholeBodyController(model, data)
    controller.reset()
    controller.capture_reference()

    assert _apply_panel_command(
        {
            "action": "set_pose",
            "position_offset": [0.1, 0.04, -0.03],
            "rpy_offset_deg": [4.0, -6.0, 8.0],
        },
        controller,
    )
    state = _panel_state(controller)
    assert np.allclose(state["position_offset"], [0.1, 0.04, -0.03])
    assert np.allclose(state["rpy_offset_deg"], [4.0, -6.0, 8.0])

    assert _apply_panel_command(
        {"action": "gripper", "closed": True},
        controller,
    )
    assert controller.gripper_closed
    assert _apply_panel_command(
        {"action": "auto_drive", "enabled": False},
        controller,
    )
    assert not controller.auto_drive
    assert _apply_panel_command(
        {"action": "speed_profile", "profile": "fast"},
        controller,
    )
    assert controller.speed_profile == "fast"
    assert controller.speed_scale == 2.0
    assert controller.rotation_speed_scale == 1.5
    assert _panel_state(controller)["speed_profile"] == "fast"
    assert not _apply_panel_command({"action": "quit"}, controller)

    overlay = _viewer_overlay(controller, "http://127.0.0.1:8765/")
    assert len(overlay) == 2
    assert overlay[0][2] == "WheelRL WBC"
    assert "Speed      fast (2.00x, rot 1.50x)" in overlay[0][3]
    assert overlay[1][3] == "http://127.0.0.1:8765/"


def test_fast_profile_tracks_a_pose_faster_than_precision() -> None:
    def error_after_half_second(profile: str) -> float:
        model = mujoco.MjModel.from_xml_path(str(WBC_MODEL_PATH))
        data = mujoco.MjData(model)
        controller = B2WZ1WholeBodyController(
            model,
            data,
            speed_profile=profile,
        )
        controller.reset()
        for _ in range(200):
            controller.step()
        controller.capture_reference()
        controller.set_home_offset(np.array([0.10, 0.0, 0.0]), np.zeros(3))
        for _ in range(50):
            diagnostics = controller.step()
        return diagnostics.position_error

    precision_error = error_after_half_second("precision")
    fast_error = error_after_half_second("fast")
    assert fast_error < 0.7 * precision_error


def test_wbc_tracks_tcp_position_and_orientation_target() -> None:
    model = mujoco.MjModel.from_xml_path(str(WBC_MODEL_PATH))
    data = mujoco.MjData(model)
    controller = B2WZ1WholeBodyController(model, data, base_assist=0.25)
    controller.reset()

    for _ in range(200):
        controller.step()
    controller.capture_reference()
    controller.set_home_offset(
        np.array([0.10, 0.04, -0.04]),
        np.deg2rad(np.array([4.0, -6.0, 8.0])),
    )

    diagnostics = controller.diagnostics()
    for _ in range(800):
        diagnostics = controller.step()

    assert diagnostics.position_error < 0.015
    assert diagnostics.orientation_error < np.deg2rad(1.5)
    assert diagnostics.upright > 0.98
    assert diagnostics.max_torque <= 300.0


def test_a_and_d_keys_follow_base_left_and_right() -> None:
    model = mujoco.MjModel.from_xml_path(str(WBC_MODEL_PATH))
    data = mujoco.MjData(model)
    controller = B2WZ1WholeBodyController(model, data)
    controller.reset()
    controller.capture_reference()
    initial_position = controller.target_position
    base_rotation = data.xmat[controller._base_body_id].reshape(3, 3)
    forward = base_rotation[:, 0].copy()
    forward[2] = 0.0
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0], 0.0])

    _handle_key(ord("A"), controller)
    assert np.allclose(controller.target_position, initial_position + 0.01 * left)
    _handle_key(ord("D"), controller)
    assert np.allclose(controller.target_position, initial_position)


def test_far_target_moves_base_and_arm_synchronously() -> None:
    model = mujoco.MjModel.from_xml_path(str(WBC_MODEL_PATH))
    data = mujoco.MjData(model)
    controller = B2WZ1WholeBodyController(model, data)
    controller.reset()
    for _ in range(200):
        controller.step()
    controller.capture_reference()

    base_position_start = data.xpos[controller._base_body_id].copy()
    tcp_position_start = controller.tcp_position
    controller.set_home_offset(np.array([0.60, 0.0, 0.0]), np.zeros(3))

    diagnostics = controller.step()
    assert diagnostics.mobile_base_active
    assert not diagnostics.arm_only
    assert diagnostics.base_participation > 0.9

    mobile_was_active = diagnostics.mobile_base_active
    for _ in range(299):
        diagnostics = controller.step()
        mobile_was_active |= diagnostics.mobile_base_active
    base_motion = data.xpos[controller._base_body_id] - base_position_start
    tcp_motion = controller.tcp_position - tcp_position_start
    assert base_motion[0] > 0.30
    assert tcp_motion[0] > 0.25
    assert mobile_was_active

    for _ in range(700):
        diagnostics = controller.step()
    assert diagnostics.position_error < 0.015
    assert diagnostics.orientation_error < np.deg2rad(1.0)
    assert diagnostics.upright > 0.98


def test_lateral_far_target_turns_base_and_converges() -> None:
    model = mujoco.MjModel.from_xml_path(str(WBC_MODEL_PATH))
    data = mujoco.MjData(model)
    controller = B2WZ1WholeBodyController(model, data)
    controller.reset()
    for _ in range(200):
        controller.step()
    controller.capture_reference()
    controller.set_home_offset(np.array([0.0, 0.40, 0.0]), np.zeros(3))

    for _ in range(3000):
        diagnostics = controller.step()

    base_rotation = data.xmat[controller._base_body_id].reshape(3, 3)
    base_yaw = np.arctan2(base_rotation[1, 0], base_rotation[0, 0])
    assert base_yaw > 0.35
    assert diagnostics.position_error < 0.025
    assert diagnostics.orientation_error < np.deg2rad(1.5)
    assert diagnostics.upright > 0.98

    # Once the moving base has delivered the target to the arm workspace, the
    # controller must hand back to arm-only precision mode. Keeping a
    # redundant base-yaw objective active used to produce a slow limit cycle
    # around the already reached target.
    terminal_errors = []
    terminal_speeds = []
    for _ in range(1500):
        diagnostics = controller.step()
        terminal_errors.append(diagnostics.position_error)
        terminal_speeds.append(diagnostics.tcp_speed)
    assert diagnostics.arm_only
    assert not diagnostics.mobile_base_active
    assert max(terminal_errors[-500:]) < 0.005
    assert np.mean(terminal_speeds[-500:]) < 0.002


def test_unreachable_low_target_tilts_without_driving_or_collision() -> None:
    model = mujoco.MjModel.from_xml_path(str(WBC_MODEL_PATH))
    data = mujoco.MjData(model)
    controller = B2WZ1WholeBodyController(model, data)
    controller.reset()
    for _ in range(200):
        controller.step()
    controller.capture_reference()

    base_start = data.xpos[controller._base_body_id].copy()
    controller.set_home_offset(np.array([0.0, 0.0, -0.60]), np.zeros(3))
    mobile_was_active = False
    unexpected_contacts: set[tuple[str | None, str | None]] = set()
    for _ in range(1200):
        diagnostics = controller.step()
        mobile_was_active |= diagnostics.mobile_base_active
        unexpected_contacts |= _non_floor_contact_pairs(model, data)

    base_motion = data.xpos[controller._base_body_id] - base_start
    base_rotation = data.xmat[controller._base_body_id].reshape(3, 3)
    base_pitch = float(
        np.arctan2(
            -base_rotation[2, 0],
            np.hypot(base_rotation[0, 0], base_rotation[1, 0]),
        )
    )
    front_hip_height = np.mean(
        [
            data.xpos[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name), 2
            ]
            for name in ("FR_hip", "FL_hip")
        ]
    )
    rear_hip_height = np.mean(
        [
            data.xpos[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name), 2
            ]
            for name in ("RR_hip", "RL_hip")
        ]
    )
    assert not mobile_was_active
    assert diagnostics.tilt_assist_active
    assert diagnostics.vertical_limited
    assert np.linalg.norm(base_motion[:2]) < 0.01
    assert abs(base_motion[2]) < 0.02
    assert np.deg2rad(10.0) < base_pitch < np.deg2rad(16.0)
    assert rear_hip_height - front_hip_height > 0.10
    assert diagnostics.upright > 0.96
    assert not unexpected_contacts
    # The target is deliberately below the safe Z1 workspace: remaining
    # error is preferable to destabilizing the whole body.
    assert diagnostics.position_error > 0.05

    controller.go_home()
    for _ in range(700):
        diagnostics = controller.step()
        unexpected_contacts |= _non_floor_contact_pairs(model, data)
    base_rotation = data.xmat[controller._base_body_id].reshape(3, 3)
    recovered_pitch = float(
        np.arctan2(
            -base_rotation[2, 0],
            np.hypot(base_rotation[0, 0], base_rotation[1, 0]),
        )
    )
    assert not diagnostics.tilt_assist_active
    assert diagnostics.arm_only
    assert abs(recovered_pitch) < np.deg2rad(1.0)
    assert diagnostics.position_error < 0.005
    assert not unexpected_contacts


def test_far_low_target_drives_then_hands_vertical_residual_to_arm() -> None:
    model = mujoco.MjModel.from_xml_path(str(WBC_MODEL_PATH))
    data = mujoco.MjData(model)
    controller = B2WZ1WholeBodyController(model, data)
    controller.reset()
    for _ in range(200):
        controller.step()
    controller.capture_reference()

    base_start = data.xpos[controller._base_body_id].copy()
    controller.set_home_offset(np.array([0.35, 0.15, -0.45]), np.zeros(3))
    mobile_was_active = False
    unexpected_contacts: set[tuple[str | None, str | None]] = set()
    for _ in range(1600):
        diagnostics = controller.step()
        mobile_was_active |= diagnostics.mobile_base_active
        unexpected_contacts |= _non_floor_contact_pairs(model, data)

    base_motion = data.xpos[controller._base_body_id] - base_start
    assert mobile_was_active
    assert base_motion[0] > 0.15
    assert abs(base_motion[2]) < 0.025
    assert diagnostics.tilt_assist_active
    assert not diagnostics.mobile_base_active
    assert diagnostics.upright > 0.96
    assert diagnostics.position_error < 0.04
    assert diagnostics.orientation_error < np.deg2rad(10.0)
    assert not unexpected_contacts


def test_forward_jog_preserves_low_tilt_and_tcp_height() -> None:
    model = mujoco.MjModel.from_xml_path(str(WBC_MODEL_PATH))
    data = mujoco.MjData(model)
    controller = B2WZ1WholeBodyController(model, data)
    controller.reset()
    for _ in range(200):
        controller.step()
    controller.capture_reference()
    controller.set_home_offset(np.array([0.0, 0.0, -0.45]), np.zeros(3))
    for _ in range(1200):
        controller.step()

    base_start = data.xpos[controller._base_body_id].copy()
    target_z = float(controller.target_position[2])
    controller.set_home_offset(np.array([0.40, 0.0, -0.45]), np.zeros(3))
    mobile_was_active = False
    minimum_pitch = np.inf
    maximum_tcp_height_error = 0.0
    unexpected_contacts: set[tuple[str | None, str | None]] = set()
    for _ in range(1000):
        diagnostics = controller.step()
        mobile_was_active |= diagnostics.mobile_base_active
        base_rotation = data.xmat[controller._base_body_id].reshape(3, 3)
        pitch = float(
            np.arctan2(
                -base_rotation[2, 0],
                np.hypot(base_rotation[0, 0], base_rotation[1, 0]),
            )
        )
        minimum_pitch = min(minimum_pitch, pitch)
        maximum_tcp_height_error = max(
            maximum_tcp_height_error,
            abs(float(controller.tcp_position[2]) - target_z),
        )
        unexpected_contacts |= _non_floor_contact_pairs(model, data)

    base_motion = data.xpos[controller._base_body_id] - base_start
    assert mobile_was_active
    assert not diagnostics.mobile_base_active
    assert diagnostics.tilt_assist_active
    assert base_motion[0] > 0.12
    assert minimum_pitch > np.deg2rad(10.0)
    assert maximum_tcp_height_error < 0.06
    assert diagnostics.position_error < 0.01
    assert diagnostics.orientation_error < np.deg2rad(2.0)
    assert not unexpected_contacts


def test_reverse_mode_backs_up_without_turning() -> None:
    model = mujoco.MjModel.from_xml_path(str(WBC_MODEL_PATH))
    data = mujoco.MjData(model)
    controller = B2WZ1WholeBodyController(model, data)
    controller.reset()
    for _ in range(200):
        controller.step()
    controller.capture_reference()
    # Target 1.0 m behind the TCP (behind the base): back up, do not turn.
    controller.set_target_pose(
        controller.home_tcp_position - np.array([1.0, 0.0, 0.0]),
        controller.home_tcp_rotation,
    )
    base_rotation = data.xmat[controller._base_body_id].reshape(3, 3)
    start_x = float(data.xpos[controller._base_body_id][0])
    start_yaw = float(np.arctan2(base_rotation[1, 0], base_rotation[0, 0]))
    diagnostics = None
    for _ in range(700):
        diagnostics = controller.step()
    base_rotation = data.xmat[controller._base_body_id].reshape(3, 3)
    base_yaw = float(np.arctan2(base_rotation[1, 0], base_rotation[0, 0]))
    dx = float(data.xpos[controller._base_body_id][0]) - start_x
    dyaw = (base_yaw - start_yaw + np.pi) % (2 * np.pi) - np.pi
    assert dx < -0.20
    assert abs(dyaw) < 0.6
    assert diagnostics is not None
    assert diagnostics.position_error < 0.06
    assert diagnostics.upright > 0.9
