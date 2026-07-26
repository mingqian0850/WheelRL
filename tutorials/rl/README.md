# 强化学习 / 深度强化学习教程

这套教程面向两类读者：

1. 希望从马尔可夫决策过程、Bellman 方程一路理解到 PPO、SAC 的学习者；
2. 希望看懂并改进 WheelRL 轮腿机器人训练代码的工程实践者。

内容采用“推导一条公式、运行一个最小例子、阅读一个成熟开源实现、回到
WheelRL 做一次实验”的循环。教程中的代码是为本仓库原创的最小示例；上游
项目只做链接、导读和接口对照，没有复制其实现。开源项目、版本快照和许可证
见 [上游项目索引](UPSTREAMS.md)。

## 你最终应该掌握什么

完成主线后，你应该能：

- 用 MDP 描述机器人任务，区分状态、观测、动作、奖励、终止和截断；
- 从 Bellman 期望方程推到 TD、Q-learning、DQN；
- 从 log-derivative trick 推出策略梯度，并理解 baseline 为什么不引入偏差；
- 从 TD 残差推到 GAE，再看懂 PPO 的 clipped surrogate objective；
- 解释 DDPG、TD3、SAC 的核心差异，并按动作空间与样本预算选择算法；
- 检查 WheelRL 的 Gymnasium API、奖励量纲、控制频率、归一化和评估流程；
- 用多随机种子、置信区间和成功率报告一个可复现的机器人 RL 结果。

## 学习地图

| 阶段 | 章节 | 产出 | 建议时间 |
|---|---|---|---:|
| 预备 | [00 学习路线与实验约定](00_learning_route.md) | 能运行最小示例，建立实验记录模板 | 0.5 天 |
| 基础 | [01 MDP 与 Bellman 方程](01_mdp_bellman.md) | 手算一个小 MDP 的价值函数 | 1 天 |
| 价值学习 | [02 从表格 Q-learning 到 DQN](02_value_learning.md) | 跑通 Q-learning，能解释 DQN 三个稳定化组件 | 1–2 天 |
| 策略学习 | [03 策略梯度](03_policy_gradient.md) | 推导 REINFORCE 与 baseline | 1 天 |
| 主算法 | [04 Actor-Critic、GAE 与 PPO](04_actor_critic_ppo.md) | 看懂 WheelRL 的 PPO 配置 | 2 天 |
| 连续控制 | [05 DDPG、TD3 与 SAC](05_continuous_control.md) | 能为连续动作任务选算法 | 1–2 天 |
| 机器人实战 | [06 WheelRL 任务建模](06_wheelrl_practice.md) | 完成一次奖励/观测消融 | 2–4 天 |
| 科学评估 | [07 调试、评估与复现](07_evaluation.md) | 生成多种子评估表和结论 | 1–2 天 |
| 扩展 | [08 模型学习、离线 RL 与模仿学习](08_advanced_topics.md) | 知道何时不应继续堆在线 PPO | 1 天 |
| 项目制 | [09 练习与毕业项目](09_projects.md) | 形成可验收的 WheelRL 实验报告 | 1–3 周 |
| 查阅 | [数学附录](math_appendix.md) | 快速复习概率、梯度和常见公式 | 按需 |

若目标是尽快读懂本仓库，走这条短路线：

```text
00 → 01 的 Bellman 小节 → 03 → 04 → 06 → 07
```

若目标是系统学习，按编号顺序完成，并在每章结束做“检查点”。

## 三层开源参考法

同一个算法建议看三遍，但每遍目的不同：

1. **概念层**：先读 [OpenAI Spinning Up](https://github.com/openai/spinningup)
   的关键公式和伪代码。它适合理解经典连续控制算法，但代码年代较早，不作为
   当前依赖模板。
2. **透明实现层**：再读 [CleanRL](https://github.com/vwxyzjn/cleanrl) 的
   单文件实现。重点追踪张量形状、数据采集、优势估计、损失和日志，而不是直接
   粘贴代码。
3. **工程层**：最后读
   [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) 的 API、
   callbacks、向量环境和保存/加载流程。WheelRL 当前使用这一层。

机器人扩展对照
[RSL-RL](https://github.com/leggedrobotics/rsl_rl) 与
[Isaac Lab](https://github.com/isaac-sim/IsaacLab)；统计评估方法参考
[rliable](https://github.com/google-research/rliable)。rliable 仓库现已归档，
因此本教程只采用其评估思想，不把它设为 WheelRL 的运行依赖。

## 快速开始

在仓库根目录完成已有 WSL 安装后：

```bash
source .venv/bin/activate
python tutorials/rl/examples/tabular_q_learning.py
python tutorials/rl/examples/gae_walkthrough.py
python tutorials/rl/examples/sb3_cartpole.py --timesteps 10000
wheelrl-check
```

前两个示例只依赖 NumPy；第三个示例使用仓库已有的 Gymnasium 和
Stable-Baselines3。示例用途是验证概念，不是提供性能基准。

## 阅读代码时始终问的五个问题

1. 一次 `env.step()` 究竟代表多少秒？策略频率和物理仿真频率分别是多少？
2. `terminated` 和 `truncated` 是否被正确区分？时间上限是否错误地阻断了
   bootstrap？
3. 网络看到的是原始量、归一化量，还是带特权状态的量？
4. 日志中的回报提升来自真正的任务成功，还是奖励塑形项变大？
5. 评估时是否冻结了归一化统计、关闭探索噪声，并使用了未见过的随机种子？

这五个问题比盲目调整学习率更常决定机器人 RL 实验是否可信。
