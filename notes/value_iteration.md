# 值迭代 (Value Iteration) 学习笔记

## 目录
1. [核心思想](#核心思想)
2. [Bellman 最优方程](#bellman-最优方程)
3. [迭代收敛原理](#迭代收敛原理)
4. [代码实现分析](#代码实现分析)
5. [实现优化讨论](#实现优化讨论)
6. [与其他方法的关系](#与其他方法的关系)

---

## 核心思想

值迭代是一种**动态规划**方法，在**已知 MDP 完整模型**（转移概率 P 和奖励函数 R）的前提下，通过反复应用 Bellman 最优算子来求解最优价值函数 V*，再从 V* 推导出最优策略 π*。

```
前提：已知 P(s'|s,a) 和 R(s,a,s')
目标：找到 V*(s) 和 π*(s)
方法：反复迭代 V(s) ← max_a Q(s,a)，直到收敛
```

### 与多臂老虎机的关键区别

| 维度 | 多臂老虎机 | 值迭代 (MDP) |
|------|-----------|-------------|
| **状态** | 无状态 | 有状态，动作改变状态 |
| **目标** | 最大化即时奖励 | 最大化长期累计奖励 |
| **学习方式** | 在线试错 | 离线规划（已知模型） |
| **核心挑战** | 探索 vs 利用 | 信用分配（哪步决策导致了最终奖励） |

---

## Bellman 最优方程

### 数学形式

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

$$\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

### 拆解理解

```
V*(s) = max_a Q*(s, a)

其中：
Q*(s, a) = Σ P(s'|s,a) × [R(s,a,s') + γ × V*(s')]
           ─────────────   ─────────   ───────────
           转移概率加权      即时奖励     折扣后的未来价值
```

- **max_a**：选择最优动作（贪心）
- **Σ P(s'|s,a)**：对所有可能的下一状态求期望（处理随机性）
- **R(s,a,s')**：即时奖励
- **γ V\*(s')**：未来价值的折扣

### 折扣因子 γ 的作用

| γ 值 | 含义 | 行为倾向 |
|------|------|---------|
| 0 | 完全忽略未来 | 只看即时奖励，极度短视 |
| 0.5 | 未来价值快速衰减 | 偏短视 |
| 0.9 | 未来价值缓慢衰减 | 平衡短期和长期（常用） |
| 1.0 | 未来和现在同等重要 | 可能不收敛（无限期任务） |

---

## 迭代收敛原理

### 为什么需要多次迭代？

**核心原因**：V(s) 的更新依赖于 V(s') 的值，而 V(s') 本身也在被更新。信息需要从"已知价值的状态"（如终点）逐步传播到所有状态。

### 信息的波浪式传播

以 4×4 确定性网格、γ=0.9 为例：

```
初始：V(s) = 0 对所有状态

迭代 1：只有终点的直接邻居获得有意义的值
  ┌──────┬──────┬──────┬──────┐
  │  -0.1│  -0.1│  -0.1│  -0.1│
  ├──────┼──────┼──────┼──────┤
  │  -0.1│   X  │  -0.1│  -0.1│
  ├──────┼──────┼──────┼──────┤
  │  -0.1│  -0.1│   X  │  1.0 │ ← 终点邻居知道"旁边有宝藏"
  ├──────┼──────┼──────┼──────┤
  │  -0.1│  -0.1│  1.0 │   G  │
  └──────┴──────┴──────┴──────┘

迭代 2：信息向外扩散一圈
  (2,3) 的邻居现在知道"走两步能到终点"
  V(1,3) = -0.1 + 0.9 × 1.0 = 0.8  ← 变了！

迭代 3：再扩散一圈
  ...

迭代 N：信息传播到起点，所有状态都有了准确的估值
```

**每次迭代，"终点有奖励"这个信息通过 `γ × V(s')` 向外传播一层。**

### 原地更新 (In-place Update)

当前代码使用**原地更新**：在同一次迭代中，先更新的状态的新 V 值会立刻被后面的状态使用。

```python
# 原地更新：V[state_idx] 被修改后，后续状态立刻用到新值
self.V[state_idx] = max(q_values)
```

这类似于 **Gauss-Seidel 方法**，通常比标准的"同步更新"收敛更快，但信息传播方向受遍历顺序影响：
- 遍历顺序 (0,0) → (3,3)：信息从左上到右下传播快，反方向需要下一次迭代
- 这就是为什么即使用了原地更新，仍然需要多次迭代

### 收敛判断

```python
# 当所有状态的 V 值变化都小于 θ 时，认为收敛
if delta < self.theta:  # theta = 1e-6
    return iteration + 1
```

**收敛保证**：只要 γ < 1，值迭代一定收敛到 V*。这是因为 Bellman 最优算子是一个 **γ-压缩映射**（contraction mapping），每次迭代都会将误差缩小至少 γ 倍。

---

## 代码实现分析

### 整体流程

```
┌─────────────────────────────────────────┐
│  1. 初始化 V(s) = 0, ∀s                │
│                                         │
│  2. value_iteration():                  │
│     ┌─────────────────────────────────┐ │
│     │ for each iteration:             │ │
│     │   for each state s:             │ │
│     │     Q(s,a) = Σ P·[R + γV(s')]   │ │
│     │     V(s) = max_a Q(s,a)         │ │
│     │   if max_change < θ: STOP       │ │
│     └─────────────────────────────────┘ │
│                                         │
│  3. extract_policy():                   │
│     π*(s) = argmax_a Q(s,a)            │
│                                         │
│  4. 用 π* 执行决策                      │
└─────────────────────────────────────────┘
```

### compute_q_value() - Q 值计算

```python
def compute_q_value(self, state, action):
    """Q(s, a) = Σ P(s'|s,a) [R(s,a,s') + γV(s')]"""
    q = 0
    current_state = self.env.idx_to_state(state)

    for next_row in range(self.env.size):
        for next_col in range(self.env.size):
            next_state = (next_row, next_col)
            prob = self.env.get_transition_prob(current_state, action, next_state)
            if prob > 0:
                reward = self.env.get_reward(current_state, next_state)
                q += prob * (reward + self.gamma * self.V[next_state_idx])
    return q
```

**对应公式**：这段代码是 Bellman 方程 Σ P(s'|s,a)[R + γV(s')] 的直接翻译。遍历所有可能的 s'，用转移概率加权求和。

### value_iteration() - 主循环

关键设计点：
- **跳过终点**：终点 V=0（已到达，无未来奖励）
- **跳过障碍物**：不可达状态，V=0
- **delta 追踪**：记录每轮最大变化量，用于判断收敛

### extract_policy() - 策略提取

值迭代收敛后，V* 已经确定。策略提取只需对每个状态选择使 Q(s,a) 最大的动作：

```python
π*(s) = argmax_a Q(s, a)
```

这是一个独立的步骤，不参与迭代过程。

---

## 实现优化讨论

### compute_q_value 的冗余问题

当前实现遍历所有 size×size 个状态来查找非零转移概率，但实际上每个 (state, action) 对最多只有 4 个可能的下一状态（上下左右）。

**当前写法**：O(S) 复杂度，每次调用 `get_transition_prob` 内部都重新计算概率分布

```python
# 遍历 16 个状态，但只有 ≤4 个有非零概率
for next_row in range(self.env.size):
    for next_col in range(self.env.size):
        prob = self.env.get_transition_prob(...)  # 每次都重新计算
```

**更高效的写法**：直接获取非零转移，O(实际转移数) 复杂度

```python
def compute_q_value(self, state, action):
    q = 0
    current_state = self.env.idx_to_state(state)

    if self.env.stochastic:
        transitions = self.env._get_stochastic_probs(current_state, action)
    else:
        next_s = self.env._get_next_state(current_state, action)
        transitions = {next_s: 1.0}

    for next_state, prob in transitions.items():
        reward = self.env.get_reward(current_state, next_state)
        next_state_idx = self.env.state_to_idx(next_state)
        q += prob * (reward + self.gamma * self.V[next_state_idx])

    return q
```

**改进效果**：4×4 网格中从遍历 16 个状态降到 ≤4 个，且避免了重复计算概率分布。

### 更贴近标准 RL 框架的接口

Gymnasium 等标准框架通常提供 `env.P[s][a]` 接口，直接返回 `[(prob, next_state, reward, done), ...]`：

```python
# Gymnasium 风格
for prob, next_state, reward, done in env.P[state][action]:
    q += prob * (reward + gamma * V[next_state])
```

这种设计将"转移动力学"封装在环境内部，算法代码更简洁通用。

---

## 与其他方法的关系

### 值迭代 vs 策略迭代

| 维度 | 值迭代 | 策略迭代 |
|------|--------|---------|
| **更新目标** | 直接更新 V(s) | 交替更新策略 π 和 V^π |
| **每步操作** | V(s) = max_a Q(s,a) | 评估 π → 改进 π → 重复 |
| **收敛速度** | 较慢（线性收敛） | 较快（通常更少外层迭代） |
| **每步代价** | 低（一次 max 操作） | 高（需要完整策略评估） |
| **适用场景** | 状态空间较大时 | 状态空间较小时 |

### 值迭代 vs Q-Learning

| 维度 | 值迭代 | Q-Learning |
|------|--------|-----------|
| **模型依赖** | 需要已知 P 和 R | 不需要（无模型） |
| **学习方式** | 离线规划 | 在线试错 |
| **数据来源** | 直接查表 P(s'\|s,a) | 与环境交互采样 |
| **状态空间** | 必须有限且可枚举 | 可扩展到连续空间 |
| **核心公式** | V(s) = max_a Σ P·[R+γV] | Q(s,a) ← Q + α·[r+γ·max Q - Q] |

### 学习路线图

```
值迭代 (已知模型，离线规划)
  │
  ├── 策略迭代 (已知模型，另一种 DP)
  │
  └── Q-Learning (未知模型，在线学习)
        │
        ├── SARSA (on-policy 变体)
        │
        └── Deep Q-Network (函数逼近，处理大状态空间)
              │
              └── Policy Gradient / Actor-Critic ...
```

值迭代是理解后续所有方法的基础：
- **Q-Learning** 本质上是在"不知道 P 和 R"的情况下，通过采样来近似值迭代的更新
- **DQN** 是用神经网络替代 Q 表，处理连续/高维状态空间

---

## 关键洞察

### 1. 值迭代 = 信息传播

每次迭代将"终点有奖励"的信息通过 γ·V(s') 向外传播一层。迭代次数 ≈ 最远状态到终点的"信息距离"。

### 2. 已知模型是奢侈的

值迭代要求完全已知 P(s'|s,a) 和 R(s,a,s')。现实中这几乎不可能--你不知道股市的转移概率，不知道用户点击的精确模型。这就是为什么后续要学习 Q-Learning 等无模型方法。

### 3. 状态空间爆炸

4×4 网格只有 16 个状态，值迭代轻松搞定。但围棋有 ~10^170 个状态，值迭代完全不可行。这推动了函数逼近（如 DQN）的发展。

---

- **最后更新**：2026-03-26
- **关联代码**：`phase2_mdp/value_iteration.py`、`phase2_mdp/mdp_gridworld.py`
- **前置知识**：`notes/mdp_gridworld.md`
- **难度等级**：⭐⭐⭐ (中等)

