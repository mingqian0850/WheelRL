import numpy as np

from wheelrl.envs import B2WZ1TrackEnv
from wheelrl.envs.b2w_z1_track import EE_POS_BOUNDS


def test_track_env_spaces_and_step() -> None:
    env = B2WZ1TrackEnv(tracking_curriculum=1.0, target_motion=1.0)
    try:
        observation, info = env.reset(seed=1)
        assert observation.shape == (86,)
        assert env.action_space.shape == (23,)
        # The actual velocity ramps up smoothly from zero after reset.
        assert info["target_speed"] == 0.0
        assert np.linalg.norm(env._ee_target_vel) > 0.0

        for _ in range(30):
            env.step(np.zeros(23, dtype=np.float32))
        assert np.linalg.norm(env._ee_vel) > 0.0

        observation, reward, terminated, truncated, info = env.step(
            np.zeros(23, dtype=np.float32)
        )
        assert observation.shape == (86,)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        assert "ee_orientation_error" in info
        assert not truncated
    finally:
        env.close()


def test_track_zero_curriculum_holds_pose() -> None:
    env = B2WZ1TrackEnv(tracking_curriculum=0.0, target_motion=0.0)
    try:
        observation, info = env.reset(seed=2)
        position_error = np.linalg.norm(
            env._ee_target_base - env._current_ee_base()
        )
        orientation_error = np.linalg.norm(env._ee_orientation_error())
        assert position_error < 1.0e-9
        assert orientation_error < 1.0e-9
        assert info["target_speed"] == 0.0
    finally:
        env.close()


def test_track_moving_target_stays_in_workspace_bounds() -> None:
    env = B2WZ1TrackEnv(tracking_curriculum=1.0, target_motion=1.0)
    try:
        env.reset(seed=3)
        moved = False
        start = env._ee_target_base.copy()
        for _ in range(500):
            env._advance_target()
            target = env._ee_target_base
            assert np.all(target >= EE_POS_BOUNDS[:, 0] - 1.0e-6)
            assert np.all(target <= EE_POS_BOUNDS[:, 1] + 1.0e-6)
            if not np.allclose(target, start):
                moved = True
        assert moved
    finally:
        env.close()


def test_track_reward_contains_orientation_and_gate_terms() -> None:
    env = B2WZ1TrackEnv(tracking_curriculum=1.0, target_motion=0.0)
    try:
        env.reset(seed=4)
        _, reward, _, _, info = env.step(np.zeros(23, dtype=np.float32))
        terms = info["reward_terms"]
        for key in ("ee_position", "ee_orientation", "sync_gate"):
            assert key in terms
        assert np.isfinite(reward)
        assert 0.0 < terms["sync_gate"] <= 1.0
    finally:
        env.close()


def test_track_aligned_target_gives_near_max_pose_reward() -> None:
    env = B2WZ1TrackEnv(tracking_curriculum=0.0, target_motion=0.0)
    try:
        env.reset(seed=5)
        _, _, _, _, info = env.step(np.zeros(23, dtype=np.float32))
        terms = info["reward_terms"]
        assert terms["ee_position"] > 0.9
        assert terms["ee_orientation"] > 0.9
        assert terms["sync_gate"] > 0.9
    finally:
        env.close()
