# 教学示例

这些脚本用于把公式变成可观察的数值，不是新的训练框架，也不复制上游项目
代码。

## 1. 表格 Q-learning

```bash
python tutorials/rl/examples/tabular_q_learning.py
```

可尝试：

```bash
python tutorials/rl/examples/tabular_q_learning.py \
  --episodes 4000 \
  --slip-probability 0.3 \
  --gamma 0.95 \
  --seed 7
```

观察不同折扣、滑移和探索退火对 Q 表及贪心策略的影响。

## 2. GAE 数值展开

```bash
python tutorials/rl/examples/gae_walkthrough.py
```

脚本验证：

- $\lambda=0$ 等于一步 TD residual；
- terminal 会切断 bootstrap；
- timeout/truncation 可保留最后状态的 bootstrap value；
- $\lambda=1$ 在真正终止的有限轨迹上等于 Monte Carlo return 减去 value。

## 3. Stable-Baselines3 PPO

```bash
python tutorials/rl/examples/sb3_cartpole.py --timesteps 10000
```

这是熟悉 SB3 API 的小实验。CartPole 是离散动作，WheelRL 是连续动作，但
`learn → predict → evaluate → save` 的工程流程相同。

## 4. 成功率汇总

```bash
python tutorials/rl/examples/summarize_success.py \
  --successes 78 71 83 69 80 \
  --episodes 100
```

输入的是每个独立训练 seed 的成功 episode 数。脚本同时展示 seed 间变异和
把 episode 合并后的 Wilson 区间；正式结论应优先关注 seed-level interval。
