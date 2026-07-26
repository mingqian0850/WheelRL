# 04 Actor-Critic、GAE 与 PPO

PPO 是 WheelRL 当前主算法。本章不把它记成一组默认参数，而是从 TD residual
逐步构造它的训练目标。

## 4.1 用 critic 构造一步优势

critic 估计 $V_\phi(s_t)$。一步 TD residual 为

$$
\delta_t
=r_t+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t),
$$

其中 $d_t$ 只在真正 terminal 时为 1。若 critic 完全准确，
$\delta_t$ 是当前动作优势的一个样本估计。

一步估计方差小，但很依赖 critic；完整 Monte Carlo 优势偏差小，但方差高。
GAE 在两者之间连续调节。

## 4.2 GAE 推导

广义优势估计定义为 TD residual 的指数加权和：

$$
\hat{A}^{\text{GAE}(\gamma,\lambda)}_t
=\delta_t
+(\gamma\lambda)(1-d_t)\delta_{t+1}
+(\gamma\lambda)^2(1-d_t)(1-d_{t+1})\delta_{t+2}
+\cdots.
$$

反向递推实现为

$$
\hat{A}_t
=\delta_t+\gamma\lambda(1-d_t)\hat{A}_{t+1}.
$$

- $\lambda=0$：只用一步 TD，方差低、对 critic 偏差敏感；
- $\lambda\to1$：接近 Monte Carlo，偏差低、方差高；
- 实践常用 $\gamma=0.99,\lambda=0.95$，但应理解为起点而非定律。

value target 常取

$$
\hat{R}_t=\hat{A}_t+V_\phi(s_t).
$$

运行数值示例：

```bash
python tutorials/rl/examples/gae_walkthrough.py
```

脚本会展示 $\lambda=0$、0.95、1 三种结果，并验证 terminal mask。

## 4.3 为什么不能无限制做策略梯度步

同一批样本是在旧策略 $\pi_{\theta_\text{old}}$ 下采集的。若用新策略计算目标，
通过 importance ratio

$$
r_t(\theta)
=\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_\text{old}}(a_t|s_t)}
=\exp\left(
\log\pi_\theta(a_t|s_t)
-\log\pi_{\theta_\text{old}}(a_t|s_t)
\right)
$$

进行修正。普通 surrogate objective 为

$$
L^{\text{PG}}(\theta)
=\mathbb{E}_t[r_t(\theta)\hat{A}_t].
$$

但当比率偏离 1 很多时，少数样本会推动过大的策略变化。

## 4.4 PPO clipped objective

PPO-Clip 最大化

$$
L^{\text{CLIP}}(\theta)
=\mathbb{E}_t
\left[
\min\left(
r_t(\theta)\hat{A}_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right].
$$

分情况理解：

- 若 $\hat{A}_t>0$，提高该动作概率是好事，但超过 $1+\epsilon$ 后收益被截平；
- 若 $\hat{A}_t<0$，降低该动作概率是好事，但低于 $1-\epsilon$ 后收益被截平；
- 对“朝坏方向”移动的更新，`min` 不会替你截掉惩罚。

因此 clipping 不是把概率比简单限制在区间内，也不是严格的 KL 约束。它只是
让继续把比率推远在目标上不再有利。实现中仍应监控：

- approximate KL；
- clip fraction；
- policy entropy；
- explained variance；
- value loss 和 gradient norm。

## 4.5 完整损失

实现通常最小化

$$
L
=-L^{\text{CLIP}}
+c_vL_V
-c_e\mathcal{H}[\pi_\theta]
$$

其中

$$
L_V
=\mathbb{E}_t[(V_\phi(s_t)-\hat{R}_t)^2].
$$

三个部分的量级可能相差很大。`vf_coef` 与 `ent_coef` 是损失权重，不是环境
奖励权重。

优势常在 minibatch 或整批上标准化：

$$
\tilde{A}_t
=\frac{\hat{A}_t-\mu_A}{\sigma_A+\varepsilon}.
$$

这改善数值条件，但也意味着 policy loss 的绝对大小不能直接跨实验比较。

## 4.6 PPO 数据流

```text
N 个并行环境 × T 步
        ↓
obs, action, reward, value, log_prob, terminal
        ↓ 反向递推
advantage, return
        ↓ 打乱
minibatches × K epochs
        ↓
policy loss + value loss + entropy → optimizer step
        ↓
丢弃 rollout，使用新策略采下一批
```

读任何 PPO 实现时，按这条数据流逐段核对，比从 class 层级开始更容易。

## 4.7 逐项解读 WheelRL 参数

`src/wheelrl/train.py` 的核心配置：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `n_envs` | 8 | 并行环境数 $N$ |
| `n_steps` | 512 | 每环境 rollout 长度 $T$ |
| rollout size | 4096 | 一次更新的数据量 $N\times T$ |
| `batch_size` | 512 | 每个 minibatch 样本数 |
| `n_epochs` | 5 | 同一 rollout 重用 5 遍 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE 偏差—方差旋钮 |
| `clip_range` | 0.2 | $\epsilon$ |
| `ent_coef` | 0.01 | 熵奖励权重 |
| `vf_coef` | 0.5 | value loss 权重 |
| `max_grad_norm` | 0.5 | 梯度范数裁剪 |
| `target_kl` | 0.02 | KL 过大时提前停止该轮更新 |
| `log_std_init` | -1.0 | 初始动作标准差约 0.368 |

一次 rollout 有 4096 个样本，每 epoch 约 8 个 minibatch，5 个 epoch 约做
40 次 optimizer step。若只增大 `n_epochs`，数据复用增强，但新旧策略可能偏离
更快，clip fraction 与 KL 也会升高。

门任务 `train_door.py` 默认 $N=6,T=256$，一次 rollout 为 1536 个样本；
`batch_size=256`，每 epoch 6 个 minibatch。任务切换课程阶段时继续保留同一个
策略与优化器。

## 4.8 归一化为什么属于算法的一部分

WheelRL 使用 `VecNormalize`：

- observation 用运行均值和方差标准化；
- 训练奖励/回报也归一化；
- observation 裁剪到 `[-10,10]`；
- 评估环境不更新统计，也不归一化奖励。

保存模型时还必须保存 `vecnormalize.pkl`。只加载网络、不加载相同 observation
统计，相当于给策略换了一套输入坐标系。

评估时应：

1. 从训练环境同步或加载 normalization statistics；
2. 设置 `training=False`，冻结统计；
3. 设置 `norm_reward=False`，报告原始任务回报；
4. 对 PPO 通常使用 deterministic action；
5. 使用独立于训练的种子和随机化实例。

## 4.9 PPO 失败时的排查顺序

1. **环境正确性**：spaces、finite observation、终止、时间尺度；
2. **奖励可达性**：随机/脚本策略是否能触发关键奖励与 success；
3. **量纲**：观测、动作、奖励项、扭矩和速度是否合理；
4. **数据吞吐**：实际 simulation steps/s 与 rollout 大小；
5. **训练诊断**：KL、clip fraction、entropy、explained variance；
6. **最后才调超参数**：学习率、网络宽度、epoch、clip 等。

若 policy loss 很“漂亮”但 success 始终为零，优先怀疑任务和奖励，而不是继续
把网络加深。

### 检查点

1. 从 $\delta_t$ 写出 GAE 的反向递推式。
2. PPO clipping 对正优势和负优势分别怎样起作用？
3. WheelRL 主任务一次 rollout 有多少样本、约做多少次 optimizer step？
4. 为什么 `best_model.zip` 和 `vecnormalize.pkl` 应视为一个整体？
