# 09 练习与毕业项目

练习按“可以验收的产出”设计。不要只提交训练截图。

## 项目 1：表格方法与 Bellman 验证

### 任务

扩展 `examples/tabular_q_learning.py`：

- 增加 SARSA；
- 比较 Q-learning 与 SARSA 在高滑移概率下的策略；
- 对一个确定性小环境，用动态规划求 $Q^*$ 并验证学习误差。

### 验收

- 固定 seed 可复现；
- 画出至少 5 个 seed 的 mean ± interval；
- 解释 on-policy 与 off-policy 导致的差异；
- 单元测试验证 terminal 后不 bootstrap。

## 项目 2：从零实现最小 PPO

### 任务

在 CartPole 或 Pendulum 上写一个教学版 PPO，只包含：

- vectorized rollout；
- actor/critic；
- GAE；
- clipped objective；
- value loss、entropy；
- minibatch 与多 epoch。

### 限制

- 不复制 CleanRL 或 SB3 源码；
- 可以逐段对照它们检查公式和 shape；
- 文件头记录参考链接；
- 总实现应足够短，能逐行解释。

### 验收

- shape 注释完整；
- GAE 有数值测试；
- 能保存/加载模型；
- 至少 3 个 seed；
- 报告 KL、clip fraction、entropy、explained variance。

## 项目 3：WheelRL 奖励审计

### 研究问题

主任务或门任务中，哪些奖励项真正区分成功与失败？

### 方法

1. 不改训练，先对脚本/随机/已有策略采样；
2. 保存每步 `reward_terms` 与 episode outcome；
3. 对成功和失败分别画加权项分布；
4. 找出饱和项、被忽略项和可能被投机的项；
5. 提出一个最小改动；
6. 用至少 3 个训练 seed 做消融。

### 验收

- 预注册主指标；
- 原配置与新配置只有一个明确差异；
- success、return、fall rate、energy 同时报告；
- 展示失败分类而非只给平均曲线。

## 项目 4：PPO 与 SAC 的公平比较

### 研究问题

在 WheelRL 的连续控制中，SAC 的样本复用是否抵消 PPO 的并行吞吐优势？

### 公平性约束

- 相同 observation、action、reward、termination；
- 相同训练环境随机化；
- 相同 evaluation seeds；
- 报告 environment steps 与 wall-clock 两套横轴；
- 分别做合理但有限的超参数预算；
- 统计所有尝试，不只保留最好结果。

### 验收

- 每个算法至少 5 个训练 seed；
- success rate 的置信区间；
- simulation / update 时间分解；
- replay ratio、PPO batch reuse 说明；
- 对结果限制给出诚实结论。

## 项目 5：门任务特权教师蒸馏

### 任务

把直接 handle angle、door angle、latch 和 phase 从 student observation 中移除，
用可测本体状态、带噪目标位姿与 observation history 训练 student。

### 阶段

1. 评估 privileged teacher；
2. 收集成功、失败和恢复轨迹；
3. BC student；
4. 加传感器噪声与延迟；
5. 必要时进行 DAgger-style 仿真标注或 RL fine-tuning；
6. 在 hold-out 门参数上测试。

### 验收

- 清楚列出 teacher-only 与 deployable observation；
- 数据集 schema、版本和 train/validation/test split；
- teacher 与 student 的成功率/失败类型；
- perception error 与 control error 分离；
- 不把仿真成功宣称为真机安全。

## 毕业报告模板

```markdown
# 标题

## 1. 问题与假设
一句话说明只改变什么、预期为什么。

## 2. 环境与算法
commit、observation、action、reward、termination、超参数。

## 3. 实验协议
训练种子、评估种子、步数、硬件、checkpoint 选择、主指标。

## 4. 结果
所有 seed、置信区间、训练步数和 wall-clock 曲线。

## 5. 失败分析
按失败类型、阶段、随机化参数拆解。

## 6. 消融与反例
说明改动在什么条件下无效或退化。

## 7. 结论与限制
只回答数据支持的问题，列出 sim-to-real 缺口。

## 8. 复现命令
完整命令、依赖锁文件和产物路径。
```
