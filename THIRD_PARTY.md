# Third-party robot assets

The B2-W MJCF and B2-W meshes under `src/wheelrl/assets/b2w_z1/assets/b2w`
come from `unitreerobotics/unitree_mujoco`, commit
`ae6a8403e272733e9996ef59990880330496177f`.

The Z1 arm and gripper meshes and physical parameters used by the combined MJCF come from
`unitreerobotics/unitree_ros`, commit
`aa0f5c68b5aba347bad409e71b6430407da758d7`.

Both upstream license texts are preserved under `src/wheelrl/assets/licenses`.
The Z1 mounting transform and the RL task/control implementation are WheelRL
project assumptions, not an official Unitree B2-W + Z1 calibration.
