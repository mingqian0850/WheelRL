# 08 模型学习、离线 RL 与模仿学习

当在线 PPO 卡住时，先判断瓶颈属于哪一类。不同瓶颈需要不同工具。

## 8.1 目标条件化与 HER

如果任务有明确目标 $g$，可学习

$$
\pi(a|s,g),\qquad Q(s,a,g).
$$

Hindsight Experience Replay 把失败轨迹中实际达到的状态重新当作目标，从而
把“没有达到原目标”的数据转成“成功达到另一个目标”的监督。

它适合：

- reach、push、pose tracking 等可定义 achieved goal 的任务；
- 奖励主要由目标误差决定；
- off-policy 算法和 replay buffer。

它不自动解决：

- 门锁阶段逻辑；
- 不安全探索；
- achieved goal 不可测；
- 任务成功还依赖接触模式和历史。

[Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)
提供 goal-conditioned 环境接口范例，可用来理解 `observation`、
`achieved_goal`、`desired_goal` 的分离。

## 8.2 模型式强化学习

学习动力学模型

$$
\hat s_{t+1}=f_\psi(s_t,a_t)
$$

或概率模型

$$
p_\psi(s_{t+1},r_t|s_t,a_t),
$$

再用于：

- Dyna：真实数据更新 model，model rollout 增加训练数据；
- MPC：每一步在模型中规划一段动作；
- world model：在潜变量空间想象 rollout；
- system identification：估计质量、摩擦、延迟等物理参数。

机器人接触会让长期预测误差快速累积。模型应报告多步误差、不确定性和
out-of-distribution 检测，而不是只报告单步 MSE。

对 WheelRL，更务实的第一步通常不是端到端 world model，而是：

1. 标定门质量、弹簧、阻尼和执行器延迟；
2. 对可解释参数做 system identification；
3. 用短视野 MPC 或 residual policy；
4. 只在模型可信区域内规划。

## 8.3 离线强化学习

离线 RL 只使用固定数据集

$$
\mathcal{D}=\{(s,a,r,s',d)\},
$$

训练期间不与环境交互。核心难题是 distribution shift：策略可能选择数据集没
覆盖的动作，而 critic 会对这些动作产生没有依据的高估。

常见路线：

- **Behavior Cloning**：只模仿数据动作，稳定但受示范上限和 covariate shift
  限制；
- **IQL**：避免显式评估任意 OOD 动作，用 expectile value 和优势加权行为克隆；
- **CQL**：对数据外动作的 Q 值施加保守惩罚；
- **Decision Transformer**：把轨迹建模为条件序列预测。

离线数据必须带上：

- observation/action 精确定义和版本；
- terminated 与 truncated；
- episode 边界；
- reward 与 success；
- 原始或可恢复的未归一化数据；
- 采集 policy、seed 和域参数；
- 安全事件和失败轨迹，不能只保留成功。

[Minari](https://github.com/Farama-Foundation/Minari) 可作为 Gymnasium 离线
数据集格式和版本管理的参考。若为 WheelRL 建数据集，先定义 schema 和数据
验证，不要先挑离线算法。

## 8.4 模仿学习与 DAgger

Behavior Cloning 最小化

$$
L_{\text{BC}}
=-\mathbb{E}_{(s,a)\sim\mathcal{D}_{expert}}
\log\pi_\theta(a|s)
$$

或连续动作的 MSE。问题是训练时只见专家状态，部署时一个小错误会把策略带到
未见状态，错误继续累积。

DAgger 的思路是：

1. 用当前 learner rollout；
2. 让 expert 给 learner 实际访问状态标注动作；
3. 聚合新旧数据；
4. 重新训练 learner。

它要求专家能对 learner 的危险状态提供可靠动作。真机上必须有安全监督和
可中止机制。

[HumanCompatibleAI/imitation](https://github.com/HumanCompatibleAI/imitation)
提供与 Gymnasium / Stable-Baselines3 对接的 BC、DAgger 等实现，可作为接口
参考。

## 8.5 Privileged teacher → deployable student

WheelRL 门任务很适合这条路线：

```text
仿真特权 PPO teacher
  输入：真实状态、门角、锁舌、phase
        ↓ 采集成功和恢复轨迹
student behavior cloning
  输入：本体状态、视觉/力觉、历史
        ↓
仿真随机化 + imitation/RL fine-tuning
        ↓
hold-out perception noise / delay 测试
        ↓
受监督的低速真机验证
```

student 不一定要复制 teacher 的每个动作。可以蒸馏：

- 动作均值；
- value/advantage；
- 中间表征；
- 高层技能标签；
- teacher 轨迹的 subgoal。

若 student 观测不足以区分两个需要不同动作的状态，增加网络容量也无法解决；
必须加入历史、传感器或状态估计。

## 8.6 何时使用哪条路线

| 瓶颈 | 优先尝试 |
|---|---|
| 奖励稀疏但目标清晰 | goal conditioning、HER、课程 |
| 有大量成功示范 | BC 起步，再 DAgger/RL |
| 真实交互昂贵 | 离线 RL、模型学习、仿真预训练 |
| 仿真参数不准 | system identification、域随机化 |
| 特权策略成功但可部署策略失败 | teacher–student、历史/感知 |
| 接触规划要求精确约束 | MPC/trajectory optimization + residual RL |
| 安全约束是硬条件 | safety layer、shield、监督状态机，不只靠惩罚项 |

### 检查点

1. HER 要求任务能提供哪三个 goal-related 概念？
2. 离线 critic 为什么容易高估数据外动作？
3. BC 的 covariate shift 从哪里来？
4. teacher 成功而 student 失败时，怎样区分感知不足和策略容量不足？
