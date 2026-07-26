# 02 从表格 Q-learning 到 DQN

## 2.1 三种价值估计

### Monte Carlo

等到 episode 结束，用完整回报监督价值：

$$
V(s_t)\leftarrow V(s_t)+\alpha\left(G_t-V(s_t)\right).
$$

优点是 target 不依赖当前价值估计；缺点是必须等到终局且方差大。

### TD(0)

只看一步，用下一个状态的估计 bootstrap：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),
$$

$$
V(s_t)\leftarrow V(s_t)+\alpha\delta_t.
$$

TD target 有偏但方差通常更小，并能在线更新。

### n-step

在两者之间取平衡：

$$
G_t^{(n)}
=r_t+\gamma r_{t+1}+\cdots+\gamma^{n-1}r_{t+n-1}
+\gamma^nV(s_{t+n}).
$$

$n=1$ 是 TD(0)，到 episode 末尾则接近 Monte Carlo。GAE 会把多个 n-step
估计按指数权重组合起来。

## 2.2 SARSA 与 Q-learning

SARSA 使用行为策略实际选择的下一动作：

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha[
r_t+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)].
$$

它是 on-policy：评估并改进当前实际执行的策略。

Q-learning 使用贪心下一动作：

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha[
r_t+\gamma\max_aQ(s_{t+1},a)-Q(s_t,a_t)].
$$

采集数据可以来自 $\epsilon$-greedy 行为策略，但 target 指向贪心策略，所以它是
off-policy。行为策略常写为

$$
a_t=
\begin{cases}
\text{随机动作}, & \text{概率 }\epsilon,\\
\arg\max_aQ(s_t,a), & \text{概率 }1-\epsilon.
\end{cases}
$$

运行本仓库示例：

```bash
python tutorials/rl/examples/tabular_q_learning.py --episodes 4000 --seed 7
```

然后分别尝试 `--gamma 0.5`、`--epsilon-end 0.5` 和更强的滑移概率。观察
Q 表和贪心策略，而不只看最后一次 episode。

## 2.3 为什么要 DQN

当状态是图像或高维连续向量时，Q 表无法枚举。DQN 用网络
$Q_\theta(s,a)$ 近似动作价值，最基本的平方 TD 损失为

$$
L(\theta)=
\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}
\left[
\left(
y-Q_\theta(s,a)
\right)^2
\right],
$$

其中

$$
y=r+\gamma(1-d)\max_{a'}Q_{\bar\theta}(s',a').
$$

$d$ 表示真正终止，$\bar\theta$ 是 target network 参数。

直接把表格更新换成神经网络通常不稳定。DQN 的两个核心工程组件是：

1. **Experience replay**：把转移存入 buffer，再近似 i.i.d. 地随机采样，提升
   数据复用并减弱相邻样本相关性；
2. **Target network**：让 target 参数 $\bar\theta$ 延迟更新，避免网络一边
   追目标、一边让目标以同样速度移动。

加上函数逼近、bootstrap 和 off-policy 数据后形成经典“deadly triad”。
它不是说算法必然发散，而是说表格算法的简单收敛直觉不再够用。

## 2.4 一次 DQN 更新的张量形状

设 batch size 为 $B$，离散动作数为 $K$：

```text
obs             [B, obs_dim]
actions         [B]
Q_online(obs)   [B, K]
Q(s,a)          [B]      # gather actions
next_Q_target   [B, K]
target          [B]
loss            scalar
```

阅读 [CleanRL 的 DQN](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py)
时，先只沿这六个张量追踪。确认 replay buffer 存了什么、何时开始学习、target
network 多久同步一次、epsilon 怎样退火，再看日志和命令行参数。

## 2.5 常见改进

### Double DQN

普通 max 同时负责“选动作”和“估价值”，噪声会产生过估计。Double DQN 用
online network 选动作、target network 估价值：

$$
a^*=\arg\max_a Q_\theta(s',a),
$$

$$
y=r+\gamma(1-d)Q_{\bar\theta}(s',a^*).
$$

### Dueling network

把输出分成状态价值和动作优势：

$$
Q(s,a)=V(s)+A(s,a)-\frac{1}{|\mathcal{A}|}\sum_{a'}A(s,a').
$$

减去均值是为消除 $V$ 与 $A$ 分解的不唯一性。它在很多动作价值相近的状态中
有帮助。

### Prioritized replay

按 TD error 大小提高“意外样本”的采样概率，但会改变数据分布，需要
importance sampling 权重修正。它增加复杂度，不应在基础 DQN 尚未通过环境、
shape 和终止语义检查前加入。

## 2.6 DQN 不适合当前 WheelRL 动作

WheelRL 的动作是 23 维或 9 维连续 `Box`。DQN 需要对所有离散动作计算
$\arg\max_aQ(s,a)$。若每个动作维度仅离散成 5 档，23 维就有
$5^{23}$ 种组合，完全不可行。

可选路线：

- 使用 PPO 的连续高斯策略；
- 使用 DDPG/TD3/SAC 的连续 actor；
- 仅在高层决策很少时离散化，例如选择 `reach/turn/pull` 技能，而底层控制仍
  连续。

不要因为 DQN 著名，就把连续机器人控制粗暴离散化。

## 2.7 调试清单

- 随机策略能否覆盖奖励非零区域？
- replay 中 `terminated` 与 timeout 是否分开？
- target 是否错误地参与梯度？
- `gather` 的动作索引 shape 是否正确？
- epsilon 是否过早降到很小？
- reward scale 是否造成巨大 TD error？
- 评估是否关闭 epsilon 探索？

### 检查点

1. Monte Carlo 和 TD 的偏差—方差权衡是什么？
2. Q-learning 为什么能用随机行为数据学习贪心策略？
3. replay buffer 和 target network 分别解决什么不稳定来源？
4. 为什么 DQN 不能自然处理 WheelRL 的 23 维连续动作？
