"""WheelRL package and Gymnasium environment registration."""

from gymnasium.envs.registration import register

register(
    id="WheelRL-B2WZ1Grip-v0",
    entry_point="wheelrl.envs.b2w_z1:B2WZ1Env",
    max_episode_steps=1000,
)

register(
    id="WheelRL-B2WZ1Door-v0",
    entry_point="wheelrl.envs.b2w_z1_door:B2WZ1DoorEnv",
    max_episode_steps=750,
)

__all__ = ["envs"]
