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
)


def test_rotation_vector_and_rpy_round_trip() -> None:
    rpy = np.deg2rad(np.array([5.0, -8.0, 12.0]))
    rotation = rpy_rotation(rpy)
    assert np.linalg.norm(rotation_vector(rotation)) > 0.1
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
    assert np.isclose(np.linalg.det(rotation), 1.0)
    assert np.allclose(rotation_rpy(rotation), rpy, atol=1.0e-12)


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

    mobile_was_active = False
    for _ in range(300):
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
