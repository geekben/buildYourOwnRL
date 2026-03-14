# Build Your Own RL 🎮

从零开始，一步步构建强化学习算法。从最简单的多臂老虎机逐步演进到 PPO、SAC 等前沿算法，每个版本只引入一个新概念，帮助你真正理解 RL 的核心原理。

## 快速开始

```bash
pip install -r requirements.txt
python bandit_epsilon_greedy.py
```

运行后会生成对比图，展示不同探索策略的效果差异。

## 学习路线图

```
bandit.py                         # 多臂老虎机：贪婪策略（RL 的 Hello World）
       ↓
bandit_epsilon_greedy.py          # ε-greedy：引入探索与利用的平衡
       ↓
bandit_ucb.py                     # UCB：基于不确定性的智能探索
       ↓
bandit_thompson.py                # Thompson Sampling：贝叶斯探索（扩展）
       ↓
mdp_gridworld.py                  # 网格世界：引入状态、动作、转移
       ↓
value_iteration.py                # 值迭代：动态规划求解
       ↓
policy_iteration.py               # 策略迭代：另一种 DP 方法
       ↓
q_learning.py                     # Q-Learning：表格方法核心
       ↓
sarsa.py                          # SARSA：在线策略 vs Q-Learning 离线策略
       ↓
dqn_v1.py                         # DQN：从表格到神经网络
       ↓
dqn_v2_replay_buffer.py           # 经验回放
       ↓
dqn_v3_target_network.py          # 目标网络
       ↓
ppo_v1.py                         # PPO：剪切目标函数
       ↓
sac_v1.py                         # SAC：最大熵强化学习
```

## 演进路线

项目按版本递进，每一步只引入**一个新概念**：

### 阶段一：探索与利用（多臂老虎机）

| 文件 | 说明 | 学习笔记 |
|------|------|----------|
| `bandit.py` | 多臂老虎机基础，贪婪策略演示探索与利用的困境 | `notes/bandit_basic.md` |
| `bandit_epsilon_greedy.py` | ε-greedy 策略，以 ε 概率探索，对比不同 ε 值效果 | |
| `bandit_ucb.py` | UCB 上置信界，基于不确定性的智能探索 | |
| `bandit_thompson.py` | Thompson Sampling，贝叶斯后验采样探索（扩展） | |

### 阶段二：马尔可夫决策过程（表格方法）

| 文件 | 引入的新概念 | 学习笔记 |
|------|-------------|----------|
| `mdp_gridworld.py` | **MDP 基础**，状态、动作、转移概率、折扣因子 | |
| `value_iteration.py` | **值迭代**，动态规划求解最优价值函数 | |
| `policy_iteration.py` | **策略迭代**，策略评估 + 策略改进交替 | |
| `q_learning.py` | **Q-Learning**，时序差分，无模型学习 | |
| `sarsa.py` | **SARSA**，在线策略学习，与 Q-Learning 对比 | |

### 阶段三：深度强化学习

| 文件 | 引入的新概念 | 学习笔记 |
|------|-------------|----------|
| `dqn_v1.py` | **神经网络近似 Q 函数**，从表格到深度网络 | |
| `dqn_v2_replay_buffer.py` | **经验回放**，打破数据相关性，提高样本效率 | |
| `dqn_v3_target_network.py` | **目标网络**，稳定训练，避免移动目标问题 | |
| `dqn_v4_double_dqn.py` | **Double DQN**，解耦动作选择与评估，减少过估计 | |
| `dqn_v5_dueling.py` | **Dueling 架构**，分离状态价值与动作优势 | |

### 阶段四：策略梯度方法

| 文件 | 引入的新概念 | 学习笔记 |
|------|-------------|----------|
| `policy_gradient.py` | **REINFORCE**，直接优化策略参数，蒙特卡洛采样 | |
| `policy_gradient_baseline.py` | **基线**，降低方差，加速收敛 | |
| `actor_critic.py` | **Actor-Critic**，结合策略梯度和价值函数 | |

### 阶段五：现代强化学习算法

| 文件 | 引入的新概念 | 学习笔记 |
|------|-------------|----------|
| `ppo_v1.py` | **PPO Clip**，限制策略更新幅度，稳定训练 | |
| `ppo_v2_gae.py` | **GAE 广义优势估计**，平衡偏差与方差 | |
| `sac_v1.py` | **SAC 基础**，最大熵框架，鼓励探索 | |
| `sac_v2_entropy_tuning.py` | **自动熵调节**，自适应控制探索程度 | |

## 最终模型架构

```
Observation → Encoder (MLP/CNN)
           → Actor Network → Mean/LogStd → Tanh Squash → Action
           → Critic Network (Q1, Q2) → Min Q → Value Estimation
           → Entropy Target (auto-tuned)
           → Policy Gradient + Entropy Bonus → Loss
```

## 核心概念演进

```
多臂老虎机（无状态）
    ↓ 引入状态
马尔可夫决策过程（已知模型）
    ↓ 无模型学习
Q-Learning / SARSA（表格方法）
    ↓ 函数近似
DQN 系列（值函数近似）
    ↓ 直接优化策略
Policy Gradient（策略梯度）
    ↓ 结合两者
Actor-Critic → PPO / SAC（现代算法）
```

## 实验环境

| 环境 | 适用阶段 | 说明 |
|------|---------|------|
| `BanditEnv` | 阶段一 | 多臂老虎机，验证探索策略 |
| `GridWorld` | 阶段二 | 网格世界，直观理解 MDP |
| `CartPole-v1` | 阶段三 | 经典控制，DQN 系列 |
| `LunarLander-v2` | 阶段三-四 | 离散动作，中等难度 |
| `Pendulum-v1` | 阶段五 | 连续动作，策略梯度验证 |
| `HalfCheetah-v4` | 阶段五 | MuJoCo，完整算法测试 |

## 学习笔记

`notes/` 目录包含学习过程中的笔记和实验记录：

**概念解析**
- `bandit_basic.md` - 多臂老虎机基础：探索与利用的困境

## 依赖

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Gymnasium (替代 OpenAI Gym)
- wandb（可选，实验跟踪）

```bash
pip install -r requirements.txt
```

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) - Sutton & Barto
- [Deep Q-Learning Paper](https://arxiv.org/abs/1312.5602) - Mnih et al., 2015
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [SAC Paper](https://arxiv.org/abs/1801.01290) - Haarnoja et al., 2018
- [Spinning Up in Deep RL](https://spinningup.openai.com/) - OpenAI
