# 上游开源项目索引

核验日期：2026-07-27。下面的 commit 是编写本教程时查看的快照，用于提供
永久链接；实际安装仍由 WheelRL 的 `pyproject.toml` 和 `uv.lock` 决定。

| 项目 | 本教程中的角色 | 编写时快照 | 许可证 | 建议阅读入口 |
|---|---|---|---|---|
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | 环境 API、spaces、终止语义、环境检查 | [`6b90bd0`](https://github.com/Farama-Foundation/Gymnasium/commit/6b90bd094c8b84dee17bd24616681f07cb26a7f2) | MIT | [基础用法](https://gymnasium.farama.org/introduction/basic_usage/) |
| [Spinning Up](https://github.com/openai/spinningup) | 经典 RL 公式、算法分类、伪代码 | [`038665d`](https://github.com/openai/spinningup/commit/038665d62d569055401d91856abb287263096178) | MIT | [RL Introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | 单文件 DQN/PPO/SAC 实现导读、实现细节核对 | [`fe8d8a0`](https://github.com/vwxyzjn/cleanrl/commit/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb) | MIT；个别文件另有声明 | [PPO 文档](https://docs.cleanrl.dev/rl-algorithms/ppo/) |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | WheelRL 的 PPO 工程实现、向量环境、回调与保存 | [`06f6135`](https://github.com/DLR-RM/stable-baselines3/commit/06f613544574aa3157eba0ccee8570f5a8a8e1c9) | MIT | [RL Tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) |
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl) | 高吞吐腿式机器人 on-policy 训练和 actor-critic 组织方式 | [`c281c32`](https://github.com/leggedrobotics/rsl_rl/commit/c281c32e5746e75570c1a29bfa78bca0200cc5ff) | BSD-3-Clause | [README](https://github.com/leggedrobotics/rsl_rl/blob/main/README.md) |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | 大规模机器人环境、任务配置、域随机化参考 | [`af1bab4`](https://github.com/isaac-sim/IsaacLab/commit/af1bab4dc173ba69b08fab779c14ead61d13fd33) | BSD-3-Clause | [官方文档](https://isaac-sim.github.io/IsaacLab/) |
| [rliable](https://github.com/google-research/rliable) | 多任务/多种子区间估计、IQM 与性能分布 | [`3ccd9f4`](https://github.com/google-research/rliable/commit/3ccd9f4dea577a04d3d2b557f259aac08badbd81) | Apache-2.0 | [README](https://github.com/google-research/rliable/blob/master/README.md) |
| [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics) | goal-conditioned 机器人环境和 HER 接口参考 | [`410d810`](https://github.com/Farama-Foundation/Gymnasium-Robotics/commit/410d810d71f363d6b6fdab457583aba8be0a5c9f) | MIT | [官方文档](https://robotics.farama.org/) |
| [Minari](https://github.com/Farama-Foundation/Minari) | 离线 RL 数据集格式、版本与加载接口 | [`1880118`](https://github.com/Farama-Foundation/Minari/commit/1880118a0d5f67addd30a540316c11fdcb337014) | 以 MIT 为主，仓库内含第三方声明 | [官方文档](https://minari.farama.org/) |
| [imitation](https://github.com/HumanCompatibleAI/imitation) | BC、DAgger 等模仿学习接口参考 | [`e5ef188`](https://github.com/HumanCompatibleAI/imitation/commit/e5ef18806c449ca47153b494a02471c5e2ae3a14) | MIT | [官方文档](https://imitation.readthedocs.io/) |

## 使用边界

- 本目录不是这些项目的镜像，也不锁定它们的完整源代码。
- 教程中的公式推导、图表、练习和示例代码均围绕 WheelRL 重新组织。
- 若将来从上游复制或修改实质性代码，必须在文件头或 `THIRD_PARTY.md`
  中记录来源、固定 commit，并遵守对应许可证和 NOTICE 要求。
- Spinning Up 适合学习经典公式，但其最后一次主分支提交较早；新项目的接口
  应以当前 Gymnasium、PyTorch 和 Stable-Baselines3 文档为准。
- rliable 仓库在核验时已归档；其论文方法仍有教学价值，但不要把归档仓库
  当作活跃维护的核心依赖。

## 章节到上游源码的阅读映射

| 教程主题 | 先读 | 再读 | 回到 WheelRL |
|---|---|---|---|
| 环境接口 | Gymnasium basic usage / custom env | SB3 env checker | `src/wheelrl/envs/` |
| DQN | Spinning Up 的 value learning 背景 | CleanRL `dqn.py` | 用离散小环境练习，不直接用于当前 23 维连续动作 |
| PPO | Spinning Up PPO | CleanRL `ppo.py`、SB3 PPO | `src/wheelrl/train.py` |
| 连续控制 | Spinning Up SAC/TD3 | CleanRL `sac_continuous_action.py`、`td3_continuous_action.py` | 作为 WheelRL PPO 的样本效率对照 |
| 机器人 PPO | RSL-RL runner/algorithm/storage | Isaac Lab RL task examples | `B2WZ1Env`、`B2WZ1DoorEnv` |
| 评估 | SB3 EvalCallback | rliable 指标和 bootstrap 思路 | `src/wheelrl/eval_door.py` |
