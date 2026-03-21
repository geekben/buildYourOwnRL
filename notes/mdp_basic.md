# 马尔可夫决策过程（MDP）：从单步决策到序列决策

## 为什么需要 MDP？

多臂老虎机是最简单的强化学习问题，但它有一个关键限制：**没有状态**。

- 多臂老虎机：每次选择都独立，动作只影响即时奖励
- 现实世界：动作会改变环境状态，当前选择影响未来

**MDP 引入了状态的概念**，让智能体能够在序列决策中考虑长期影响。

## MDP 的五个元素

一个马尔可夫决策过程定义为五元组 $\langle S, A, P, R, \gamma \rangle$：

### 1. 状态空间（State Space）$S$

所有可能的环境状态集合。

网格世界示例：
- 4×4 网格，共 16 个状态
- 状态表示：$(row, col)$，如 $(0, 0)$ 是起点，$(3, 3)$ 是终点
- 障碍物状态：$(1, 1)$, $(2, 2)$

### 2. 动作空间（Action Space）$A$

智能体可执行的所有动作集合。

网格世界示例：
- 四个动作：上、下、左、右
- 动作编号：$A = \{0, 1, 2, 3\}$

### 3. 转移概率（Transition Probability）$P$

$$
P(s' | s, a) = \Pr(S_{t+1} = s' | S_t = s, A_t = a)
$$

描述在状态 $s$ 执行动作 $a$ 后，转移到状态 $s'$ 的概率。

**确定性环境**：
$$
P(s' | s, a) = \begin{cases} 1 & \text{如果 } s' \text{ 是预期下一状态} \\ 0 & \text{否则} \end{cases}
$$

**随机环境**（网格世界中加入滑倒概率）：
- 80% 概率按预期方向移动
- 20% 概率滑倒到其他方向

马尔可夫性质：
$$
P(s' | s, a) = P(s' | s, a, s_{t-1}, a_{t-1}, \ldots)
$$
即：未来只依赖于当前状态和动作，与历史无关。

### 4. 奖励函数（Reward Function）$R$

$$
R(s, a, s') = \text{在状态 } s \text{ 执行 } a \text{ 转移到 } s' \text{ 获得的奖励}
$$

网格世界的奖励设计：

| 情况 | 奖励 | 设计意图 |
|------|------|----------|
| 到达终点 | +1.0 | 鼓励完成任务 |
| 撞到障碍 | -1.0 | 惩罚危险行为 |
| 每步移动 | -0.1 | 鼓励效率，避免原地打转 |

### 5. 折扣因子（Discount Factor）$\gamma$

$$
\gamma \in [0, 1]
$$

用于计算累计奖励（回报）：

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

**γ 的作用**：

| γ 值 | 含义 | 适用场景 |
|------|------|----------|
| 0 | 只关心即时奖励 | 贪婪、短视 |
| 0.9 | 平衡即时与未来 | 最常用 |
| 1 | 未来与现在同等重要 | 有限horizon任务 |

**为什么需要折扣因子**：
1. 避免无限循环中的无穷回报
2. 模型不确定性：远期奖励更难预测
3. 经济学意义：即时收益比未来收益更有价值

## 目标：最大化期望回报

智能体的目标是找到一个策略 $\pi$，使得从任意状态出发的期望回报最大：

$$
\pi^* = \arg\max_\pi \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R_t \mid \pi\right]
$$

## 状态价值函数 $V(s)$

在状态 $s$ 下，遵循策略 $\pi$ 能获得的期望回报：

$$
V_\pi(s) = \mathbb{E}_\pi\left[G_t \mid S_t = s\right]
$$

**Bellman 方程**（递归定义）：

$$
V_\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V_\pi(s') \right]
$$

直觉：当前状态的价值 = 即时奖励期望 + 未来奖励期望的折扣值

## 动作价值函数 $Q(s, a)$

在状态 $s$ 执行动作 $a$ 后，遵循策略 $\pi$ 的期望回报：

$$
Q_\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid S_t = s, A_t = a\right]
$$

**Q 与 V 的关系**：

$$
V_\pi(s) = \sum_a \pi(a|s) Q_\pi(s, a)
$$

$$
Q_\pi(s, a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V_\pi(s') \right]
$$

## 最优价值函数

最优状态价值函数：
$$
V^*(s) = \max_\pi V_\pi(s)
$$

最优动作价值函数：
$$
Q^*(s, a) = \max_\pi Q_\pi(s, a)
$$

**Bellman 最优方程**：

$$
V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]
$$

$$
Q^*(s, a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \max_{a'} Q^*(s', a') \right]
$$

一旦知道 $Q^*$，最优策略就是：

$$
\pi^*(s) = \arg\max_a Q^*(s, a)
$$

---

## 实验结果

### 网格世界环境

![网格世界](../phase2_mdp/images/gridworld_demo.png)

- **S** (蓝色)：起点 (0, 0)
- **G** (绿色)：终点 (3, 3)
- **X** (红色)：障碍物

### 确定性 vs 随机环境

使用随机策略（等概率选择动作）：

| 环境 | 到达终点比例 | 平均累计奖励 | 平均步数 |
|------|-------------|-------------|---------|
| 确定性 | 44.8% | -3.55 | 40.5 |
| 随机 (slip=0.2) | 46.4% | -3.49 | 40.1 |

随机环境下，由于滑倒概率的存在，策略执行的不确定性增加。

---

## 与多臂老虎机的对比

| 特性 | 多臂老虎机 | MDP |
|------|-----------|-----|
| 状态 | 无 | 有 |
| 动作影响 | 只影响即时奖励 | 影响状态转移和未来奖励 |
| 决策类型 | 单步 | 序列 |
| 策略 | 动作选择 | 状态到动作的映射 |
| 核心问题 | 探索与利用 | 规划与学习 |

## 环境交互接口

标准的强化学习交互循环：

```
state = env.reset()  # 初始化
done = False

while not done:
    action = agent.select_action(state)  # 根据状态选择动作
    next_state, reward, done = env.step(action)  # 执行动作，观察结果
    agent.update(state, action, reward, next_state)  # 学习
    state = next_state
```

这个接口被 gym/gymnasium 等框架广泛采用。

## 已知模型 vs 未知模型

**已知模型（Model-based）**：
- 知道转移概率 $P$ 和奖励函数 $R$
- 可以使用动态规划（值迭代、策略迭代）直接求解最优策略
- 本节假设：我们完全知道环境的 MDP 结构

**未知模型（Model-free）**：
- 不知道 $P$ 和 $R$
- 需要通过交互来学习
- 后续的 Q-Learning、SARSA 属于此类

## 下一步

既然我们知道了 MDP 的完整信息，如何找到最优策略？

- **值迭代（Value Iteration）**：迭代求解 Bellman 最优方程
- **策略迭代（Policy Iteration）**：交替进行策略评估和策略改进

---

## 代码运行

```bash
cd phase2_mdp

# 运行网格世界演示
python mdp_gridworld.py
```

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) - Sutton & Barto, Chapter 3
- [马尔可夫决策过程 Wiki](https://en.wikipedia.org/wiki/Markov_decision_process)
