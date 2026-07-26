# 05 连续控制：DDPG、TD3 与 SAC

WheelRL 的动作空间是高维连续区间。除了 PPO，常见候选还有 DDPG、TD3 和
SAC。它们都是 off-policy actor-critic，可用 replay buffer 复用样本。

## 5.1 确定性策略梯度与 DDPG

确定性 actor 输出

$$
a=\mu_\theta(s).
$$

critic 学习 $Q_\phi(s,a)$。actor 目标为最大化 critic：

$$
J(\theta)
=\mathbb{E}_{s\sim\mathcal{D}}
\left[Q_\phi(s,\mu_\theta(s))\right].
$$

链式法则给出

$$
\nabla_\theta J
\approx
\mathbb{E}_{s\sim\mathcal{D}}
\left[
\nabla_a Q_\phi(s,a)|_{a=\mu_\theta(s)}
\nabla_\theta\mu_\theta(s)
\right].
$$

critic target 为

$$
y=r+\gamma(1-d)
Q_{\bar\phi}(s',\mu_{\bar\theta}(s')).
$$

DDPG 使用：

- actor 和 critic target networks；
- replay buffer；
- 在确定性动作上外加探索噪声；
- Polyak averaging 慢更新 target。

问题是 critic 的局部误差会直接把 actor 推向虚假的高 Q 区域，训练对超参数和
reward scale 敏感。

## 5.2 TD3 的三个修正

TD3 针对 DDPG 的过估计与脆弱性加入三项：

### Clipped double Q

训练两个 critic，target 取较小者：

$$
y=r+\gamma(1-d)
\min_{i=1,2}Q_{\bar\phi_i}(s',a').
$$

取 min 会偏保守，但能抑制 max/actor 对估计噪声的利用。

### Target policy smoothing

target action 加裁剪噪声：

$$
a'=\operatorname{clip}
\left(
\mu_{\bar\theta}(s')+\operatorname{clip}(\epsilon,-c,c),
a_{\min},a_{\max}
\right).
$$

这相当于要求 critic 在动作邻域内平滑，而不是依赖一个尖锐的虚假峰值。

### Delayed policy update

critic 更新若干次后才更新 actor 和 target networks，让 value estimate 先跟上。

阅读实现时可对照 CleanRL 的
[`td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py)。

## 5.3 最大熵强化学习

SAC 不只最大化奖励，还最大化策略熵：

$$
J(\pi)
=\mathbb{E}_\pi
\left[
\sum_t\gamma^t
\left(r_t+\alpha\mathcal{H}(\pi(\cdot|s_t))\right)
\right].
$$

$\alpha$ 是温度：

- 大 $\alpha$：更重视随机性和探索；
- 小 $\alpha$：更重视高回报；
- 常见做法是自动调节 $\alpha$，逼近目标熵。

soft state value 满足

$$
V(s)
=\mathbb{E}_{a\sim\pi}
\left[
Q(s,a)-\alpha\log\pi(a|s)
\right].
$$

双 critic target 为

$$
y=r+\gamma(1-d)
\left[
\min_iQ_{\bar\phi_i}(s',a')
-\alpha\log\pi_\theta(a'|s')
\right].
$$

actor 最小化

$$
J_\pi(\theta)
=\mathbb{E}_{s\sim\mathcal{D},a\sim\pi_\theta}
\left[
\alpha\log\pi_\theta(a|s)
-\min_iQ_{\phi_i}(s,a)
\right].
$$

## 5.4 Reparameterization 与 tanh

SAC 常用

$$
u=\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I),
$$

$$
a=\tanh(u).
$$

这样随机性来自与 $\theta$ 无关的 $\epsilon$，梯度可以穿过 $u$ 和 `tanh`。
但动作变换后 log-prob 必须包含 Jacobian 修正：

$$
\log\pi(a|s)
=\log\mathcal{N}(u;\mu,\sigma)
-\sum_i\log(1-\tanh^2(u_i)).
$$

实际代码会加数值稳定项。漏掉修正会让 entropy/temperature 目标错误。

## 5.5 PPO 与 SAC 如何选

| 维度 | PPO | SAC |
|---|---|---|
| 数据使用 | on-policy，rollout 后丢弃 | off-policy，replay 多次复用 |
| 并行仿真 | 很适合大量同步环境 | 也能并行，但实现更复杂 |
| 样本效率 | 通常较低 | 通常较高 |
| wall-clock | 高吞吐仿真下常很强 | 单环境或昂贵仿真可能更占优 |
| 稳定性 | clipping 后相对稳健 | 对 Q scale、target、更新比敏感 |
| 探索 | 高斯策略 + entropy bonus | 最大熵目标 |
| recurrent/privileged | 机器人框架中常见 | 可做，但工程支持差异较大 |

对 WheelRL：

- 当前 MuJoCo CPU 仿真可并行，PPO 是合理基线；
- 若仿真单步很贵、真实采样有限，SAC 的 replay 更有吸引力；
- 门任务含明显阶段结构和稀疏成功，先解决任务可达性与课程，再比较算法；
- 比较时按**环境交互步数**和**实际耗时**各画一张曲线，不能只选对某算法有利
  的横轴。

## 5.6 机器人 off-policy 的额外陷阱

1. **Replay 过期**：课程、域随机化范围或奖励函数改变后，旧 buffer 不再来自
   当前任务分布；
2. **大动作维度**：critic 更容易出现 extrapolation error；
3. **接触不连续**：Q 对动作的局部梯度可能很嘈杂；
4. **归一化漂移**：buffer 中旧样本与更新后的 observation statistics 不一致；
5. **UTD ratio**：每个环境步做过多 gradient update，可能在有限数据上过拟合。

因此“off-policy 更省样本”不等于“把更新次数调大就更好”。

### 检查点

1. DDPG 的 actor 梯度怎样通过 critic 传播？
2. TD3 三个修正分别针对什么问题？
3. SAC 的温度 $\alpha$ 变大时，策略行为怎样改变？
4. 为什么 `tanh` squashing 后要修正 log-prob？
