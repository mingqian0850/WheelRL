# 数学附录

## A.1 期望的线性

不要求随机变量独立：

$$
\mathbb{E}[aX+bY]=a\mathbb{E}[X]+b\mathbb{E}[Y].
$$

策略梯度推导中反复使用这一性质。

## A.2 方差和 bias–variance

$$
\operatorname{Var}(X)=\mathbb{E}[(X-\mathbb{E}X)^2].
$$

估计量 $\hat{x}$ 的均方误差可分为

$$
\mathbb{E}[(\hat{x}-x)^2]
=\operatorname{Var}(\hat{x})+\operatorname{Bias}(\hat{x})^2.
$$

Monte Carlo target 通常低偏高方差；bootstrap target 通常增加偏差、降低方差。

## A.3 log-derivative trick

因为

$$
\nabla_\theta\log p_\theta(x)
=\frac{\nabla_\theta p_\theta(x)}{p_\theta(x)},
$$

所以

$$
\nabla_\theta p_\theta(x)
=p_\theta(x)\nabla_\theta\log p_\theta(x).
$$

由此

$$
\nabla_\theta\mathbb{E}_{x\sim p_\theta}[f(x)]
=\mathbb{E}_{x\sim p_\theta}
\left[f(x)\nabla_\theta\log p_\theta(x)\right].
$$

这是 REINFORCE 的核心。

## A.4 梯度停止

构造 TD target 时，target 一般被视为常数：

```python
with torch.no_grad():
    target = reward + gamma * next_value
loss = (value - target).square().mean()
```

若不停止梯度，优化器可能通过改变 target 一侧来减小损失，目标就不再是预期的
半梯度更新。不同算法若明确需要对 target 求导会另行说明。

## A.5 高斯策略

连续动作策略常设

$$
\pi_\theta(a|s)=\mathcal{N}(\mu_\theta(s),\operatorname{diag}(\sigma_\theta^2)).
$$

一维对数概率为

$$
\log\pi(a|s)
=-\frac{1}{2}\left[
\frac{(a-\mu)^2}{\sigma^2}+2\log\sigma+\log(2\pi)
\right].
$$

多维对角高斯的 log-prob 是各维之和。实现中要明确：

- 网络输出的是 $\sigma$、$\log\sigma$ 还是 state-independent 参数；
- 动作是否再经过 `tanh`；
- `tanh` 后是否加入 change-of-variables 的 log-prob 修正；
- 环境动作缩放是否在策略分布内部还是 wrapper 中完成。

PPO 与 SAC 对这些选择的处理并不完全相同。

## A.6 KL 散度

$$
D_{\mathrm{KL}}(P\|Q)
=\mathbb{E}_{x\sim P}
\left[\log\frac{P(x)}{Q(x)}\right]\ge 0.
$$

KL 不对称。PPO 常把新旧策略的近似 KL 当监控或 early stopping 指标；它不是
clipped objective 本身的硬约束。

## A.7 熵

离散分布：

$$
\mathcal{H}(\pi(\cdot|s))
=-\sum_a\pi(a|s)\log\pi(a|s).
$$

连续分布使用微分熵。熵奖励鼓励策略保持随机性，但其数值依赖动作维度和分布
参数化，不能跨任务机械复用相同系数。

## A.8 指数移动平均

target network 的 Polyak 更新常写成

$$
\bar\theta\leftarrow
\tau\theta+(1-\tau)\bar\theta,
$$

其中小 $\tau$ 表示慢更新。部分论文使用 $\rho=1-\tau$，阅读代码时要看公式，
不要只看参数名。
