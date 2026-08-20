"""Play a trained task-space tracking policy with a live target marker.

Loads a ``B2WZ1TrackEnv`` scene with two extra marker sites injected into a
temporary copy of the model (red sphere = target position, orange box = target
orientation), runs the trained policy deterministically, and shows live
tracking error in the viewer overlay. Press ``R`` to reset the episode.

Usage:

    wheelrl-play-track --model runs/track_ppo/final_model.zip \
        --stats runs/track_ppo/vecnormalize.pkl --seconds 60
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from wheelrl.envs import B2WZ1TrackEnv

SCENE_XML = (
    Path(__file__).resolve().parent / "assets" / "b2w_z1" / "scene.xml"
)

MARKER_SITES = """
    <site name="track_target_pos" type="sphere" size="0.035"
      rgba="1 0.3 0 0.55" group="3"/>
    <site name="track_target_ori" type="box" size="0.055 0.013 0.013"
      rgba="1 0.6 0 0.95" group="3"/>
"""


def _build_scene_with_markers() -> Path:
    """Write a temp scene (same dir, so relative includes resolve) with markers."""
    xml = SCENE_XML.read_text(encoding="utf-8")
    xml = xml.replace("<worldbody>", "<worldbody>\n" + MARKER_SITES, 1)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".xml",
        dir=str(SCENE_XML.parent),
        delete=False,
        encoding="utf-8",
    )
    handle.write(xml)
    handle.close()
    return Path(handle.name)


def _world_target(env: B2WZ1TrackEnv) -> tuple[np.ndarray, np.ndarray]:
    rotation = env._base_rotation()
    position = env.data.xpos[env._base_body_id]
    world_pos = position + rotation @ env._ee_target_base
    world_rot = rotation @ env._ee_target_rotation
    return world_pos, world_rot


def _update_markers(model: mujoco.MjModel, data: mujoco.MjData, env: B2WZ1TrackEnv) -> None:
    """Move the target markers to the current target pose.

    The viewer renders sites from ``data.site_xpos``/``site_xmat``, so the
    markers are overwritten there after each physics step (model-level site
    positions of world-attached sites are cached at compile time and cannot
    be changed at runtime).
    """
    world_pos, world_rot = _world_target(env)
    for name in ("track_target_pos", "track_target_ori"):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        data.site_xpos[site_id] = world_pos
        data.site_xmat[site_id] = world_rot.reshape(9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--curriculum", type=float, default=1.0)
    parser.add_argument("--motion", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true", help="no WSLg viewer")
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    args = parser.parse_args()

    temp_scene = _build_scene_with_markers()
    try:
        # render_mode is always None: the play script owns the (single) viewer
        # below; an env render_mode of "human" would open a second passive
        # viewer from env.step(), and two viewers on one context close each
        # other.
        env = B2WZ1TrackEnv(
            render_mode=None,
            tracking_curriculum=args.curriculum,
            target_motion=args.motion,
            _model_filename=temp_scene.name,
        )
        model = env.model
        env.reset(seed=args.seed)

        # Observation normalization from the training stats.
        vec_env = VecNormalize.load(
            args.stats,
            DummyVecEnv([lambda: B2WZ1TrackEnv(tracking_curriculum=0.0, target_motion=0.0)]),
        )
        vec_env.training = False
        vec_env.norm_reward = False

        policy = PPO.load(args.model, device=args.device)

        viewer = None
        if not args.headless:
            from mujoco import viewer as mujoco_viewer

            def key_callback(key: int) -> None:
                if chr(key).upper() == "R":
                    env.reset(seed=args.seed)

            viewer = mujoco_viewer.launch_passive(
                model, env.data, key_callback=key_callback
            )
            viewer.cam.lookat[:] = np.array([0.4, 0.0, 0.6])
            viewer.cam.distance = 2.6
            viewer.cam.azimuth = -125.0
            viewer.cam.elevation = -18.0

        deadline = time.monotonic() + args.seconds
        episode = 0
        next_report = time.monotonic() + 1.0
        try:
            while time.monotonic() < deadline and (
                viewer is None or viewer.is_running()
            ):
                start = time.monotonic()
                normalized = vec_env.normalize_obs(np.asarray(env._get_obs())[None])[0]
                action, _ = policy.predict(normalized, deterministic=True)
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    episode += 1
                    env.reset(seed=args.seed + episode)
                _update_markers(model, env.data, env)
                mujoco.mj_forward(model, env.data)

                if viewer is not None:
                    viewer.sync()
                    if time.monotonic() >= next_report:
                        overlay = [
                            (
                                mujoco.mjtFontScale.mjFONTSCALE_100,
                                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                "WheelRL track",
                                "\n".join(
                                    [
                                        f"Episode   {episode}",
                                        f"Pos err   {info['ee_error']:.3f} m",
                                        f"Ori err   "
                                        f"{np.rad2deg(info['ee_orientation_error']):.1f} deg",
                                        f"Height    {env.data.qpos[2]:.2f} m",
                                        f"Upright   {info['reward_terms']['upright']:.2f}",
                                        f"Target    {args.motion:.2f} m/s (R: reset)",
                                    ]
                                ),
                            )
                        ]
                        viewer.set_texts(overlay)
                        next_report = time.monotonic() + 0.25
                else:
                    if time.monotonic() >= next_report:
                        print(
                            f"ep={episode} pos_err={info['ee_error']:.3f} m "
                            f"ori_err={np.rad2deg(info['ee_orientation_error']):.1f} deg "
                            f"height={env.data.qpos[2]:.2f} "
                            f"upright={info['reward_terms']['upright']:.2f}"
                        )
                        next_report = time.monotonic() + 1.0

                elapsed = time.monotonic() - start
                if elapsed < env.dt:
                    time.sleep(env.dt - elapsed)
        finally:
            if viewer is not None:
                viewer.close()
                time.sleep(0.25)
            vec_env.close()
            env.close()
    finally:
        os.unlink(temp_scene)


if __name__ == "__main__":
    main()
