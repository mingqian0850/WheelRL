# 03 策略梯度

价值方法间接得到策略：先学 $Q$，再取 $\arg\max$。策略梯度直接参数化
$\pi_\theta(a|s)$ 并最大化期望回报，天然适合连续动作和随机策略。

## 3.1 轨迹概率

长度 $T$ 的轨迹记为

$$
\tau=(s_0,a_0,r_0,\ldots,s_T).
$$

其概率为

$$
p_\theta(\tau)
=\rho_0(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t).
$$

目标函数是

$$
J(\theta)
=\mathbb{E}_{\tau\sim p_\theta(\tau)}[R(\tau)].
$$

为让基础推导清楚，本节先令
$R(\tau)=\sum_{t=0}^{T-1}r_t$。折扣目标会在 3.3 节单独说明时间权重。

## 3.2 从期望回报推到 REINFORCE

先对轨迹分布求导：

$$
\nabla_\theta J
=\int \nabla_\theta p_\theta(\tau)R(\tau)d\tau.
$$

使用 log-derivative trick：

$$
\nabla_\theta J
=\mathbb{E}_{\tau\sim p_\theta}
\left[
R(\tau)\nabla_\theta\log p_\theta(\tau)
\right].
$$

展开轨迹 log-prob：

$$
\log p_\theta(\tau)
=\log\rho_0(s_0)
+\sum_t\log\pi_\theta(a_t|s_t)
+\sum_t\log P(s_{t+1}|s_t,a_t).
$$

环境初始分布和动力学不含 $\theta$，求导后只剩策略：

$$
\nabla_\theta J
=\mathbb{E}
\left[
\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)R(\tau)
\right].
$$

这个结果很重要：即使不知道 MuJoCo 动力学的导数，也可以用采样得到无偏策略
梯度估计。

## 3.3 当前动作不该为此前奖励负责

时刻 $t$ 的动作不能影响过去奖励。利用因果性，可把整条轨迹回报替换成
reward-to-go：

$$
\hat{G}_t=\sum_{k=t}^{T-1}r_k,
$$

$$
\hat{g}
=\sum_t
\nabla_\theta\log\pi_\theta(a_t|s_t)\hat{G}_t.
$$

这仍是 REINFORCE，但去掉了与当前动作无关的过去奖励，方差更小。

若严格从初始状态优化折扣目标
$R_\gamma(\tau)=\sum_k\gamma^k r_k$，对应的轨迹估计为

$$
\sum_t\gamma^t\nabla_\theta\log\pi_\theta(a_t|s_t)
\left(\sum_{k=t}^{T-1}\gamma^{k-t}r_k\right).
$$

策略梯度定理也可把折扣吸收到状态访问分布中。不同教材/实现可能不显式写
外层 $\gamma^t$；比较公式时要先确认它优化的 objective 与采样分布。PPO
实现通常用 `gamma` 构造 return/advantage，再对 rollout 时间步取平均。

实现中通常最小化负目标：

$$
L_{\text{policy}}
=-\frac{1}{B}\sum_t
\log\pi_\theta(a_t|s_t)\hat{G}_t.
$$

## 3.4 为什么可以减 baseline

对任何只依赖状态、不依赖当前动作的 $b(s)$：

$$
\mathbb{E}_{a\sim\pi}
\left[
\nabla_\theta\log\pi_\theta(a|s)b(s)
\right]
=b(s)\nabla_\theta\sum_a\pi_\theta(a|s)
=b(s)\nabla_\theta1=0.
$$

所以

$$
\hat{g}
=\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t)
\left(\hat{G}_t-b(s_t)\right)
$$

不改变期望梯度。最常用的 baseline 是学习出的 $V_\phi(s_t)$，于是括号内成为
优势估计：

$$
\hat{A}_t=\hat{G}_t-V_\phi(s_t).
$$

baseline 的作用是降方差，不是“奖励的一部分”。不要把 value loss 和环境奖励
混为一谈。

## 3.5 Actor 与 Critic

- **Actor**：$\pi_\theta(a|s)$，决定动作；
- **Critic**：$V_\phi(s)$ 或 $Q_\phi(s,a)$，估计 actor 的表现；
- **Actor-Critic**：critic 提供低方差优势估计，actor 用它更新策略。

REINFORCE 用完整 Monte Carlo 回报训练；actor-critic 通常用 bootstrap，因此
引入一些偏差以换取更低方差和更频繁更新。

## 3.6 连续动作策略

WheelRL 的 PPO actor 可理解为对每个动作维度给出高斯分布参数：

$$
a_t\sim\mathcal{N}(\mu_\theta(s_t),\operatorname{diag}(\sigma^2)).
$$

训练时采样以探索，评估时通常取均值。`log_std_init=-1.0` 对应初始标准差约

$$
\exp(-1)\approx0.368.
$$

对于归一化到 $[-1,1]$ 的动作，这已经是明显的探索幅度。动作最终如何裁剪或
缩放必须结合框架实现阅读；简单 `clip` 会让边界处的实际动作分布不同于原始
高斯。

## 3.7 熵正则

为避免策略过早变成近似确定性，可在最大化目标中加入熵：

$$
J_{\text{total}}
=J_{\text{policy}}+\beta\,
\mathbb{E}_s[\mathcal{H}(\pi_\theta(\cdot|s))].
$$

在最小化 loss 的代码中，熵项符号通常相反。检查实现时不要只看变量名，
应确认最后是鼓励还是惩罚熵。

较大的熵系数并不总是更好：

- 稀疏奖励早期可能需要更多探索；
- 接触丰富的机器人任务中，过大动作方差会导致跌倒或饱和；
- 动作维度越高，总熵量级也越大；
- 课程阶段切换时可能需要重新审视探索强度。

## 3.8 On-policy 的含义

策略梯度公式中的轨迹来自当前 $\pi_\theta$。策略更新后，旧数据不再严格服从
新策略。REINFORCE、A2C 和 PPO 因此属于 on-policy 家族，数据复用有限。

PPO 会用概率比率允许对同一批 rollout 做数个 epoch，但仍不能像 SAC 那样长期
保存一个大 replay buffer 反复训练。样本效率和训练稳定性之间的权衡，是机器人
任务中比较 PPO 与 SAC 的关键。

## 3.9 阅读开源实现

建议顺序：

1. 读 Spinning Up 的策略优化介绍，确认公式；
2. 打开 CleanRL `ppo.py`，只标出 `get_action_and_value()`、rollout、
   advantage、minibatch loss；
3. 打开 SB3 PPO 文档，理解用户可配置的超参数；
4. 回到 WheelRL `train.py`，把每个参数写在第 04 章的公式旁边。

### 检查点

1. 为什么策略梯度不需要对环境动力学求导？
2. reward-to-go 为什么能降方差？
3. baseline 依赖动作会发生什么？
4. 训练时随机、评估时确定性分别解决什么问题？
