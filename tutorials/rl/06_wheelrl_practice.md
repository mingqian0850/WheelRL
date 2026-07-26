# 06 WheelRL 任务建模实战

本章把前面的公式落到当前代码。所有数字以本分支基于的 WheelRL `main`
实现为准；未来修改环境时应同步更新这一章。

## 6.1 控制层级与时间尺度

主环境 `B2WZ1Env`：

```text
PPO policy：50 Hz，每 0.02 s 输出一次归一化动作
        ↓
关节目标 / 车轮速度目标
        ↓
PD torque：送入 500 Hz MuJoCo 仿真
        ↓
累计 10 个物理步后返回 observation、reward
```

策略没有直接输出 500 Hz torque，这降低了探索空间的刚性和高频风险。代价是
策略能力受固定 PD 结构、增益和目标缩放限制。

一个容易忽略的事实：算法中的“一步”是 0.02 s，而不是 MuJoCo 的一个内部
步。所有 $\gamma$、episode length、动作平滑项都应按策略步解释。

## 6.2 主任务动作空间

23 维动作都先裁剪到 $[-1,1]$：

| 索引 | 维度 | 物理含义 | 映射 |
|---|---:|---|---|
| `0:12` | 12 | 四条腿关节位置残差 | nominal + 每关节 scale × action |
| `12:16` | 4 | 四个车轮速度目标 | $24a$ rad/s |
| `16:22` | 6 | Z1 关节位置残差 | nominal + scale × action，再按关节限位裁剪 |
| `22` | 1 | 夹爪位置 | 线性映射到关节范围 |

PD torque 形式为

$$
\tau=K_p(q_{\text{target}}-q)-K_d\dot q,
$$

最后按每个 actuator 的 torque limit 裁剪。

动作含义决定了策略优化的几何：

- `action=0` 不是零扭矩，而是 nominal 姿态/零轮速；
- 每一维的相同高斯标准差对应不同关节幅度；
- 夹爪动作是绝对归一化位置，而腿/臂是相对 nominal 的残差；
- reward 中 action-rate 惩罚约束的是归一化动作变化，不是直接扭矩变化。

## 6.3 主任务 80 维观测

| 片段 | 维度 | 累计 |
|---|---:|---:|
| base height | 1 | 1 |
| base-frame linear velocity | 3 | 4 |
| base-frame angular velocity | 3 | 7 |
| projected gravity | 3 | 10 |
| 腿 12 + 臂 6 + 夹爪 1 的相对关节位置 | 19 | 29 |
| 23 个受控关节速度 × 0.1 | 23 | 52 |
| forward / yaw command | 2 | 54 |
| base-frame end-effector target error | 3 | 57 |
| previous action | 23 | 80 |

`previous action` 让策略看到动作平滑项所需的信息，也为执行器动态提供一小段
历史。它并不能解决所有部分可观测性。

检查观测时要问：

- 世界坐标和机体坐标是否混用？
- 角度是否有 wrap discontinuity？
- 不同量纲在 `VecNormalize` 前相差多少？
- 仿真中可见的量在真实机器人上是否可测、延迟多大？
- 训练和部署是否使用完全相同的顺序与缩放？

## 6.4 主任务奖励

主任务奖励可以概括为

$$
\begin{aligned}
r={}&0.85r_{\text{vx}}
+0.30r_{\text{yaw}}
+0.35r_{\text{upright}}
+0.25r_{\text{height}}\\
&+0.55r_{\text{ee}}
+0.10r_{\text{alive}}
+p_{\text{energy}}
+p_{\Delta a}
+p_{\text{lateral}}.
\end{aligned}
$$

跟踪项使用指数核，例如

$$
r_{\text{vx}}
=\exp[-2(v_x-v_x^{cmd})^2],
$$

$$
r_{\text{ee}}
=\exp[-18\|e_{\text{ee}}\|^2].
$$

指数核有界、平滑，但尺度系数决定“多大误差就几乎没有梯度”。例如
$\exp(-18e^2)$ 在 $e=0.2$ m 时只剩约 0.49。调系数前先画出 reward–error
曲线。

能耗项近似使用

$$
p_{\text{energy}}
=-2\times10^{-5}\sum_i|\tau_i\dot q_i|.
$$

它是功率幅值的代理，而不是严格的电能模型。动作变化和横向速度也被惩罚。

### 奖励审计

做一次 200–1000 步 rollout，分别累计每个 `reward_terms`：

1. 比较每项**加权后**的均值、标准差和极值；
2. 检查某一正奖励是否在失败策略上也接近饱和；
3. 检查负项是否大到盖过所有任务信号；
4. 对成功轨迹和失败轨迹分别统计，不要只看混合平均。

## 6.5 门任务为什么改成 9 维高层动作

`B2WZ1DoorEnv` 固定腿部站立控制，让策略输出：

| 索引 | 维度 | 含义 |
|---|---:|---|
| `0` | 1 | base forward velocity command |
| `1` | 1 | base yaw-rate command |
| `2:8` | 6 | 围绕 pre-grasp 姿态的 Z1 残差目标 |
| `8` | 1 | gripper target |

这是一种 action-space shaping：

- 降低策略维度；
- 把轮子非完整约束和腿部站立交给固定控制器；
- 让零均值探索从可达的 pre-grasp 附近开始；
- 减少“还没碰到把手就跌倒”的无效样本。

它也引入偏置：若固定控制器或 pre-grasp 不足以完成某类轨迹，PPO 无法突破
这个边界。

## 6.6 门任务 84 维特权观测

前 52 维与主任务的机体/关节信息相似，之后加入：

| 片段 | 维度 | 累计 |
|---|---:|---:|
| base/关节信息 | 52 | 52 |
| handle position in base | 3 | 55 |
| end-effector-to-handle error | 3 | 58 |
| door normal in base | 3 | 61 |
| handle axis in base | 3 | 64 |
| handle angle/velocity、door angle/velocity、latch、grasp | 6 | 70 |
| phase one-hot | 5 | 75 |
| previous high-level action | 9 | 84 |

直接的 handle/door angle、latch 和 grasp 状态是仿真特权信息。合理路线是：

1. 先训练 privileged teacher，验证任务和奖励可学习；
2. 收集 teacher 的成功轨迹；
3. 构造只用真实可测传感器、视觉/力觉和历史的 student；
4. 做 behavior cloning / distillation；
5. 再用受限 RL 微调，并单独评估 perception error 与 policy error。

## 6.7 进度奖励与课程

门任务大量使用差分进度：

$$
r_{\text{reach-progress}}
\propto d_{t-1}-d_t,
$$

$$
r_{\text{handle-progress}}
\propto \theta^{handle}_t-\theta^{handle}_{t-1},
$$

$$
r_{\text{door-progress}}
\propto \theta^{door}_{t-1}-\theta^{door}_t.
$$

差分项比持续给“离目标近”的奖励更不容易诱导原地刷分，但策略可能前后振荡；
需要结合 episode 常数成本、action-rate、成功终止和回放检查。

课程阶段为：

1. `reach`：到达把手区域；
2. `turn`：从 pre-grasp 开始，闭合夹爪并转动把手；
3. `pull`：从近抓取、已解锁状态开始拉门；
4. `full`：从 nominal 姿态完成全流程。

默认时间分配是 15%、30%、25%、30%。同一策略和优化器连续训练，因此后期可能
发生遗忘。每阶段都要保留：

- 当前阶段成功率；
- 前面阶段的回归评估；
- value/normalization statistics；
- 按失败类型的计数。

仅看 `full` 的平均回报无法判断失败发生在哪个技能。

## 6.8 域随机化

当前门任务随机化包括：

- 门质量和惯量同尺度变化；
- 门框平面位置；
- 把手释放角；
- 把手弹簧；
- 关门器强度；
- 门铰链阻尼；
- reset 的 base/joint 噪声。

随机化强度不是越大越好。推荐课程：

```text
固定模型学会任务
    ↓
小范围随机化保持高成功率
    ↓
逐步扩大到标定不确定性范围
    ↓
在未参与训练的 hold-out 参数组合上评估
```

随机化分布应来自实物标定或明确的误差预算。把所有参数任意放大 50%，可能训练
出保守但无用的策略，也可能让任务物理上不可达。

## 6.9 WheelRL 实验阶梯

### 实验 A：零训练环境审计

```bash
wheelrl-check
wheelrl-check-door
wheelrl-play-door --stage turn --seconds 30
```

确认动作方向、坐标系、终止条件和 success 事件。

### 实验 B：过拟合单一条件

```bash
wheelrl-train-door \
  --stage reach \
  --timesteps 100000 \
  --n-envs 2 \
  --device cpu \
  --randomization 0 \
  --run-dir runs/tutorial_reach_seed42
```

若固定环境的 reach 都学不会，不要先加域随机化。

### 实验 C：多种子基线

至少运行 3 个种子，记录 success rate，而不是挑最好的一个 checkpoint。

### 实验 D：单变量消融

从以下问题任选一个：

- 去掉 previous action；
- 把 `reach_delta` 系数减半；
- 改变 `log_std_init`；
- 课程阶段重新分配；
- 把随机化从 0、0.2、0.35 逐级比较。

每次只改一个因素，并提前写出假设。

## 6.10 Sim-to-real 最小清单

仿真成功不等于可以上真机。至少需要：

- 测量 B2-W + Z1 安装变换、总质量、质心与惯量；
- 标定关节摩擦、扭矩/速度限制和 PD 增益；
- 模拟 command、observation、actuator 的延迟和丢包；
- 用真实可测 observation 替代特权状态；
- 加入自碰撞、环境碰撞和力/扭矩安全约束；
- 在策略外设置硬限位、速度限制、急停和监督状态机；
- 先离线回放，再低增益、低速度、被动安全环境测试；
- 将“训练奖励高”与“满足硬安全条件”彻底分开。

### 检查点

1. 为什么主任务的零动作不等于零 torque？
2. 写出 80 维 observation 的分段并验证总和。
3. 门任务降成 9 维动作带来什么能力与偏置？
4. privileged teacher 到真实 student 中间还缺哪些模块？
5. 为什么域随机化范围应来自标定，而不是越宽越好？
