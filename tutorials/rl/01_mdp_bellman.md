# 01 MDP 与 Bellman 方程

## 1.1 从交互过程到 MDP

马尔可夫决策过程写作

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},P,R,\rho_0,\gamma).
$$

- $s_t\in\mathcal{S}$：时刻 $t$ 的状态；
- $a_t\in\mathcal{A}$：智能体选择的动作；
- $P(s'|s,a)$：状态转移概率；
- $R(s,a,s')$：一步奖励；
- $\rho_0$：初始状态分布；
- $\gamma\in[0,1)$：折扣因子。

策略 $\pi(a|s)$ 给出在状态 $s$ 选择动作 $a$ 的概率。环境和策略共同生成

$$
s_0,a_0,r_0,s_1,a_1,r_1,\ldots
$$

这条轨迹。马尔可夫性要求：给定当前状态和动作后，下一状态不再依赖更早历史：

$$
P(s_{t+1}|s_t,a_t,s_{t-1},a_{t-1},\ldots)
=P(s_{t+1}|s_t,a_t).
$$

机器人实际得到的通常是观测 $o_t$，不一定包含完整状态。例如 WheelRL 门任务
的首个策略直接看到门角、把手角和锁舌状态，这些是仿真中的**特权观测**。真实
系统若无法直接测得这些量，策略面对的是部分可观测问题；需要历史堆叠、循环
网络、状态估计器或教师—学生蒸馏，而不能仅把变量名从 observation 中删掉。

## 1.2 回报和价值函数

从 $t$ 开始的折扣回报为

$$
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}.
$$

折扣有三种直觉：

1. 表达“越早得到奖励越好”；
2. 让无限时域的和在奖励有界时收敛；
3. 决定信用分配的有效时间尺度，粗略为 $1/(1-\gamma)$ 个策略步。

WheelRL 的策略周期是 $0.02$ s。若 $\gamma=0.99$，有效尺度约为 100 步，
也就是约 2 s。这个换算比单独记住 `gamma=0.99` 更有意义。

状态价值和动作价值分别是

$$
V^\pi(s)=\mathbb{E}_\pi[G_t|s_t=s],
$$

$$
Q^\pi(s,a)=\mathbb{E}_\pi[G_t|s_t=s,a_t=a].
$$

优势函数衡量一个动作相对于当前策略平均选择好多少：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

## 1.3 Bellman 期望方程

把回报拆出第一步：

$$
G_t=r_t+\gamma G_{t+1}.
$$

对策略和环境的随机性取期望：

$$
V^\pi(s)
=\mathbb{E}_{a\sim\pi,s'\sim P}
\left[R(s,a,s')+\gamma V^\pi(s')\right].
$$

同理，

$$
Q^\pi(s,a)
=\mathbb{E}_{s'\sim P}
\left[R(s,a,s')+\gamma
\mathbb{E}_{a'\sim\pi}Q^\pi(s',a')\right].
$$

这不是额外假设，而是回报定义的递归展开。价值学习的大部分算法，只是在回答
“怎样用有限样本逼近右侧期望”。

对有限状态 MDP，固定策略后可写成矩阵形式：

$$
\mathbf{v}_\pi=\mathbf{r}_\pi+\gamma P_\pi\mathbf{v}_\pi,
$$

因此

$$
\mathbf{v}_\pi=(I-\gamma P_\pi)^{-1}\mathbf{r}_\pi.
$$

小问题可以直接求逆；大问题无法枚举状态，只能用采样、函数逼近和迭代更新。

## 1.4 Bellman 最优方程

最优价值定义为所有策略中的最大值：

$$
V^*(s)=\max_\pi V^\pi(s),\qquad
Q^*(s,a)=\max_\pi Q^\pi(s,a).
$$

最优动作价值满足

$$
Q^*(s,a)
=\mathbb{E}_{s'\sim P}
\left[R(s,a,s')+\gamma\max_{a'}Q^*(s',a')\right].
$$

若知道 $Q^*$，贪心策略

$$
\pi^*(s)\in\arg\max_a Q^*(s,a)
$$

就是最优策略。Q-learning 和 DQN 正是在逼近这个固定点。

当 $\gamma<1$ 时，Bellman 最优算子在最大范数下是 $\gamma$-压缩：

$$
\|\mathcal{T}Q_1-\mathcal{T}Q_2\|_\infty
\le \gamma\|Q_1-Q_2\|_\infty.
$$

这解释了表格值迭代为何会收敛到唯一固定点。神经网络情形不再自动继承这一
保证，因为更新同时包含采样误差、非线性函数逼近和不断变化的数据分布。

## 1.5 一个两步任务

有状态 `start`、`near_goal`、`terminal`。固定策略总是向前：

- `start → near_goal`，奖励 0；
- `near_goal → terminal`，奖励 1；
- terminal 之后无奖励。

若 $\gamma=0.9$：

$$
V(\text{near_goal})=1,
$$

$$
V(\text{start})=0+0.9V(\text{near_goal})=0.9.
$$

若在 `start` 增加一个立即获得 0.95 并终止的动作，智能体会偏好立即奖励；
若奖励是 0.8，则偏好两步路径。改变 $\gamma$ 会改变这个选择。

这和机器人奖励塑形直接相关：很大的“靠近把手”即时奖励可能让策略停在把手
附近，而不是完成延迟更长的开门成功。

## 1.6 终止与截断

Gymnasium 把 episode 结束分成：

- `terminated=True`：MDP 内生终止，例如机器人跌倒或任务成功；
- `truncated=True`：外部时间上限或采集限制，底层状态仍可能继续。

TD target 应写成

$$
y_t=r_t+\gamma(1-\mathbb{1}_{\text{terminated}})V(s_{t+1}).
$$

仅因时间上限 `truncated=True` 时通常仍应 bootstrap。如果把两者合并成
`done` 并一律置零，会在时间边界产生系统性低估。使用成熟库时也要确认 wrapper
如何传递 `terminal_observation` 和 timeout 信息。

## 1.7 对照 WheelRL

打开 `src/wheelrl/envs/b2w_z1.py`，逐项定位：

- $\mathcal{S}$：仿真器完整 `qpos`、`qvel`、接触等内部状态；
- $o_t$：`_get_obs()` 输出的 80 维向量；
- $\mathcal{A}$：`Box(-1, 1, (23,))`；
- $P$：MuJoCo 动力学、PD 控制和 `frame_skip`；
- $R$：`_reward()` 中的任务项与正则项；
- 内生终止：`_is_healthy()` 失败；
- 外生截断：达到 `max_episode_steps`。

### 检查点

1. 为什么 observation 不一定等于 MDP state？
2. $V^\pi$、$Q^\pi$ 和 $A^\pi$ 各回答什么问题？
3. 若策略频率不变，$\gamma$ 从 0.99 改到 0.995，信用分配时间尺度怎样变化？
4. 机器人达到时间上限时，价值 target 是否应自动清零？
