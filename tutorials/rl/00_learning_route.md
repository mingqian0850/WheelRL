# 00 学习路线与实验约定

## 先建立正确的心智模型

强化学习不是“给神经网络一个奖励，它自然就会学会”。完整闭环是：

```text
任务定义 → 环境动力学 → 数据采集 → 目标估计 → 参数更新
    ↑                                      ↓
评估与诊断 ← 日志、回放、成功率、失败类型 ← 新策略
```

任何一处错误都会表现为“训练不稳定”。因此先固定实验约定，再谈算法。

## 预备知识最低线

你不需要先学完整本概率论，但应能使用：

- 条件概率、期望、方差；
- 向量、矩阵乘法和梯度；
- Python、NumPy、PyTorch 的张量与自动微分；
- 神经网络的前向、损失、反向传播与优化器。

欠缺的数学可以边做边查 [数学附录](math_appendix.md)。

## 推荐的 6 周路线

### 第 1 周：不用神经网络

- 完成第 01、02 章；
- 手算一个 3 状态 MDP；
- 运行 `tabular_q_learning.py`；
- 改变折扣因子和探索率，解释策略变化。

### 第 2 周：策略梯度

- 完成第 03、04 章；
- 手推 REINFORCE、baseline、GAE；
- 运行 `gae_walkthrough.py`；
- 对照 CleanRL PPO，标出 rollout、GAE、minibatch、loss 四段。

### 第 3 周：成熟框架

- 运行 `sb3_cartpole.py`；
- 阅读 SB3 的 `Monitor`、`VecEnv`、`EvalCallback`；
- 解释模型权重和 `VecNormalize` 统计为什么必须成对保存。

### 第 4 周：WheelRL 环境

- 阅读 `B2WZ1Env` 的 spaces、`reset()`、`step()`、奖励与终止；
- 用零动作和随机动作检查量纲；
- 画出每个奖励项随时间的曲线；
- 完成一个不训练的环境消融。

### 第 5 周：训练与评估

- 做 3–5 个随机种子的小规模 PPO 实验；
- 比较成功率、episode return、能耗与稳定性；
- 不根据单条“最好曲线”下结论。

### 第 6 周：机器人专题

- 完成门任务的 reach → turn → pull → full 课程；
- 增加域随机化；
- 按失败类型统计：未到达、未抓住、未解锁、未拉开、跌倒；
- 写一页 sim-to-real 风险清单。

## 每次实验的最小记录

建议为每组实验保存以下信息：

```yaml
experiment:
  commit: "<git sha>"
  command: "<完整命令>"
  seed: 42
  environment: "WheelRL-B2WZ1Grip-v0"
  total_timesteps: 2000000
  train_randomization: 0.35
  evaluation_seeds: [10001, 10002, 10003, 10004, 10005]
  primary_metric: "success_rate"
  secondary_metrics: ["return", "energy", "fall_rate", "episode_length"]
  hardware: "<CPU/GPU 与操作系统>"
  notes: "<唯一改变的变量与预期>"
```

如果一次实验同时改奖励、网络、学习率和随机化，结果几乎无法解释。一次只改变
一个研究问题，工程调通阶段除外。

## 运行检查

在仓库根目录：

```bash
source .venv/bin/activate
wheelrl-check
python tutorials/rl/examples/tabular_q_learning.py --seed 7
python tutorials/rl/examples/gae_walkthrough.py
```

### 检查点

不看资料，尝试回答：

1. “算法”“环境”“策略网络”“训练脚本”分别承担什么责任？
2. 为什么训练回报和真实任务成功率可能相反？
3. 为什么比较两个算法时要固定环境版本和评估种子？

能用自己的话回答，再进入第 01 章。
