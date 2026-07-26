# 07 调试、评估与复现

强化学习训练曲线噪声很大。可靠结论来自预先定义的指标、多个训练种子、独立
评估和不确定性区间，而不是一条最漂亮的曲线。

## 7.1 先区分四类随机性

1. **训练种子**：网络初始化、动作采样、minibatch 顺序；
2. **环境种子**：reset noise、命令采样、域随机化；
3. **评估种子**：未参与训练和 checkpoint 选择的测试 episode；
4. **系统随机性**：并行调度、GPU 非确定算子和浮点归约顺序。

仅固定 `seed=42` 只能帮助复现某一次运行，不能说明算法稳定。

## 7.2 WheelRL 的指标层级

### 主指标

门任务首选 success rate：

$$
\hat p=\frac{\text{成功 episode 数}}{\text{总 episode 数}}.
$$

成功条件是任务定义，而 episode return 混合了大量塑形项。checkpoint 选择也应
优先成功率，再用回报打破平局；当前 `BestSuccessCallback` 正是这样做。

### 次指标

- episode return；
- episode length；
- reach / turn / pull / full 各阶段成功率；
- 跌倒率、超时率；
- 到把手距离、把手角、门角；
- 平均/峰值 torque、机械功率代理；
- action rate、接触力、任务完成时间。

### 诊断指标

- PPO approximate KL、clip fraction；
- policy entropy、action std；
- value loss、explained variance；
- gradient norm；
- simulation steps/s；
- 每个 reward term 的加权分布。

诊断指标帮助解释训练，不应代替任务指标。

## 7.3 训练、选择与最终测试分离

推荐三层数据：

```text
train：更新策略和 normalization statistics
validation：EvalCallback 选择 checkpoint
test：训练结束后一次性报告，不再据此调参
```

若看了 test 结果后继续调奖励和超参数，test 就变成 validation，必须再建立新的
hold-out 条件。

域随机化也要分离：

- train 参数范围：训练时采样；
- interpolation test：范围内未见组合；
- extrapolation test：略超出范围；
- nominal test：固定标称模型，检查鲁棒性是否牺牲太多标称性能。

## 7.4 多种子报告

最低建议：

- 调试阶段：1 个训练种子，20 个评估 episode；
- 初步比较：3 个训练种子，每个至少 50–100 个 episode；
- 正式结论：5–10 个训练种子，并报告不确定性区间。

每个训练种子产生一个独立 policy。不要把同一 policy 的 500 个 episode 当成
500 次独立训练。环境随机性估计“这个 policy 的表现”，训练种子之间的差异才
估计“训练算法的稳定性”。

可以报告：

- 每个 seed 的 success rate；
- across-seed mean / median；
- 25% trimmed mean 或 IQM；
- seed-level bootstrap 95% confidence interval；
- worst-seed 和失败类型。

运行示例：

```bash
python tutorials/rl/examples/summarize_success.py \
  --successes 78 71 83 69 80 \
  --episodes 100 \
  --bootstrap 20000
```

脚本会给出各 seed rate、mean、IQM、Wilson interval 和按 seed 重采样的区间。
正式跨多个任务比较时可参考 rliable 的 stratified bootstrap、performance
profiles 与 probability of improvement；该仓库目前已归档，因此这里把方法
写进实验协议，而不新增运行依赖。

## 7.5 为什么平均回报会骗人

假设两个策略：

| 策略 | 90 个失败 episode | 10 个成功 episode | 平均回报 | 成功率 |
|---|---:|---:|---:|---:|
| A | 每次 30 | 每次 80 | 35 | 10% |
| B | 每次 5 | 每次 70 | 37.5 | 50% |

B 的平均回报只略高，但任务成功率是 A 的五倍。反过来，若靠近把手的塑形奖励
很高，完全失败的 A 甚至可能有更高平均回报。

所以报告中至少同时给 success rate 与 return，并用失败类型解释差异。

## 7.6 失败分类

为门任务建立互斥或层级标签：

```text
fall
timeout_before_reach
reached_but_no_grasp
grasped_but_no_latch_release
latched_released_but_no_door_open
success
```

每个 episode 在结束时生成一个标签。比较两个实验时画失败占比，而不是只说
“成功率提高 8%”。这样才能决定下一步是改 perception、grasp、课程还是 base
pulling。

## 7.7 常见训练曲线症状

| 现象 | 优先检查 | 不要先做 |
|---|---|---|
| 回报完全不变 | 环境 step、reward 是否可触发、参数是否在更新 | 加大网络 |
| 回报升而成功率为 0 | 奖励投机、成功阈值、关键阶段不可达 | 继续训练十倍步数 |
| entropy 很快归零 | 初始 std、entropy coefficient、动作裁剪/饱和 | 只改 gamma |
| KL 突然很高 | 学习率、优势异常值、epoch、batch | 加更多 epoch |
| value loss 巨大 | reward scale、terminal mask、normalization | 只调 actor |
| explained variance 长期 ≤ 0 | observation、return target、critic 容量 | 宣称 critic 无关 |
| 单 seed 很好其余失败 | 任务脆弱、探索不足、课程/初始化依赖 | 只发布最好 seed |

## 7.8 环境级验证

在训练前做：

1. `reset(seed=x)` 是否可重复；
2. observation/action 是否落在声明 space；
3. 连续运行 1000 步是否始终 finite；
4. 成功、跌倒、超时能否分别触发；
5. reward sum 是否等于记录的各项加权和；
6. 环境关闭后是否释放 viewer/renderer；
7. vectorized 环境中每个 rank 是否使用不同但可复现的 seed。

现有仓库测试已经覆盖基础 shape、finite、门锁释放和被锁门的恢复力；新增任务
时应继续把物理不变量写成测试。

## 7.9 可复现性清单

发布结果时记录：

- Git commit 和 dirty 状态；
- Python 与锁文件版本；
- 完整命令与所有非默认参数；
- 训练/评估 seed；
- 环境 ID 与 observation/action 版本；
- reward 与 success 定义；
- normalization statistics；
- checkpoint 选择规则；
- 硬件、操作系统、耗时和仿真吞吐；
- 所有训练 seed，包括失败运行。

### 检查点

1. 同一 policy 的 100 个 episode 与 5 个训练 seed 分别衡量什么？
2. 为什么 test set 一旦用于调参就不再是 test？
3. 设计一个能解释门任务失败位置的标签体系。
4. 哪些 PPO 日志能提示策略更新过大？
