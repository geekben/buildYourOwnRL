"""
dqn_v2_replay_buffer.py - DQN v2：引入经验回放

在 v1 的基础上只增加一个新概念：经验回放（Experience Replay）

v1 的问题回顾：
    1. 数据相关性：连续的 (s, a, r, s') 来自同一条轨迹，高度相关
       → 网络过度拟合当前轨迹，忘记之前学过的
    2. 移动目标：TD target 用同一个网络计算（v3 解决）

经验回放如何解决数据相关性？
    v1（在线学习）：每步产生一个样本，立即用来更新，然后丢弃
        → 连续样本高度相关 → 梯度方向偏 → 训练不稳定

    v2（经验回放）：每步产生的样本存入缓冲区，训练时随机抽一批
        → 随机采样打破时间相关性 → 梯度方向更均匀 → 训练更稳定
        → 同一条经验可以被多次使用 → 样本效率更高

类比：
    v1 = 考试复习只看刚做过的题目 → 只记住最近做的，忘记之前的
    v2 = 把所有做过的题目存起来，每次随机抽几道复习 → 均匀学习

学习要点：
1. 经验回放的动机与设计
2. ReplayBuffer 的实现（环形缓冲区）
3. 从单样本更新到 mini-batch 更新
4. v1 vs v2 的训练稳定性对比
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque

# 复用 v1 的 Q 网络（网络结构完全不变）
from dqn_v1 import QNetwork


# ============================================================
# 经验回放缓冲区：v2 的核心新组件
# ============================================================

class ReplayBuffer:
    """
    经验回放缓冲区（Experience Replay Buffer）

    核心思想：
        不再"用完就丢"，而是把经历存起来，训练时随机采样

    工作流程：
        1. 智能体与环境交互，产生 (s, a, r, s', done)
        2. 存入缓冲区（如果满了，覆盖最旧的）
        3. 训练时，从缓冲区随机采样一个 mini-batch
        4. 用这个 batch 做一次梯度下降

    为什么有效：
        - 打破时间相关性：随机采样让 batch 中的样本来自不同时间步和不同轨迹
        - 提高样本效率：同一条经验可以被采样多次，充分利用数据
        - 平滑训练：batch 中包含多种状态，梯度方向更均匀

    实现细节：
        - 使用 deque（双端队列）作为环形缓冲区
        - 容量有限（如 10000），满了自动丢弃最旧的经验
        - 为什么不无限存？旧经验对应的策略太差，用处不大
    """

    def __init__(self, capacity=10000):
        """
        Args:
            capacity: 缓冲区最大容量。太小 → 采样多样性不够；太大 → 旧数据太多
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """存入一条经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        随机采样一个 mini-batch

        返回的每个元素都是 numpy 数组，shape = (batch_size, ...)
        这是经验回放的核心操作：随机采样打破了数据的时间相关性
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ============================================================
# DQN v2 智能体
# ============================================================

class DQNAgentV2:
    """
    DQN v2：DQN v1 + 经验回放

    与 v1 的差异（只有两处）：
    ┌─────────────────┬──────────────────────┬──────────────────────────┐
    │                 │ DQN v1               │ DQN v2                   │
    ├─────────────────┼──────────────────────┼──────────────────────────┤
    │ 数据来源        │ 当前步的单个样本      │ 从 buffer 随机采样 batch │
    │ 更新方式        │ 单样本梯度下降        │ mini-batch 梯度下降      │
    │ 样本使用次数    │ 1 次（用完就丢）      │ 多次（存在 buffer 中）   │
    │ 数据相关性      │ 高（连续样本）        │ 低（随机采样）           │
    └─────────────────┴──────────────────────┴──────────────────────────┘

    其他一切（网络结构、ε-greedy、TD target 计算）与 v1 完全相同。
    """

    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_capacity=10000, batch_size=64, seed=42):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon: 初始探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
            buffer_capacity: 经验回放缓冲区容量（新增）
            batch_size: 每次从 buffer 中采样的样本数（新增）
            seed: 随机种子
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # 随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        # Q 网络（与 v1 完全相同）
        self.q_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # 经验回放缓冲区（v2 新增！）
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # 记录训练过程
        self.episode_rewards = []
        self.episode_steps = []
        self.losses = []
        self.epsilon_history = []

    def select_action(self, state):
        """ε-greedy 选择动作（与 v1 完全相同）"""
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        存储一条经验到缓冲区（v2 新增）

        v1 中没有这一步——v1 直接在 update() 中使用当前样本
        v2 先存起来，update() 时从 buffer 中随机采样
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """
        从缓冲区采样 mini-batch 并更新（v2 的核心改动）

        与 v1 的对比：
            v1.update(state, action, reward, next_state, done):
                用单个样本计算 loss，做一次梯度下降

            v2.update():
                从 buffer 随机采样 batch_size 个样本
                用这些样本的平均 loss，做一次梯度下降

        batch 更新的好处：
            1. 随机采样打破数据相关性
            2. 多个样本的梯度平均更稳定（方差更小）
            3. GPU 可以并行处理 batch（虽然 CartPole 用不到）
        """
        # buffer 中样本不够时不更新
        if len(self.replay_buffer) < self.batch_size:
            return

        # 1. 从 buffer 随机采样（核心操作！）
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # 2. numpy → torch
        states_tensor = torch.FloatTensor(states)            # (batch, state_dim)
        actions_tensor = torch.LongTensor(actions)            # (batch,)
        rewards_tensor = torch.FloatTensor(rewards)           # (batch,)
        next_states_tensor = torch.FloatTensor(next_states)   # (batch, state_dim)
        dones_tensor = torch.FloatTensor(dones)               # (batch,)

        # 3. 计算当前 Q 值：Q_θ(s)[a]
        #    对比 v1：v1 只算一个样本的 Q 值，v2 算一个 batch 的
        q_values = self.q_network(states_tensor)              # (batch, action_dim)
        current_q = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        #                     ^^^^^^ 从所有动作的 Q 值中，选出实际执行的那个动作的 Q 值
        #    .gather(1, actions) = Q(s_i, a_i) for each i in batch

        # 4. 计算 TD target：r + γ max_a' Q_θ(s', a')
        #    注意：仍然用同一个网络（v3 才引入目标网络）
        with torch.no_grad():
            next_q_values = self.q_network(next_states_tensor)  # (batch, action_dim)
            max_next_q = next_q_values.max(dim=1).values        # (batch,)
            td_targets = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q

        # 5. 计算损失：batch 中所有样本的 MSE 平均值
        loss = nn.functional.mse_loss(current_q, td_targets)

        # 6. 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================================
# 训练函数
# ============================================================

def train_dqn_v2(env, agent, n_episodes=300, max_steps=500):
    """
    训练 DQN v2（经验回放）

    与 v1 训练循环的对比：
        v1: 选动作 → 执行 → 立即更新（单样本）
        v2: 选动作 → 执行 → 存入 buffer → 从 buffer 采样更新（batch）

    训练循环中只有两行代码不同：
        + agent.store_transition(...)   ← 存入 buffer
        - agent.update(s, a, r, s', d)  ← v1 的单样本更新
        + agent.update()                ← v2 的 batch 更新
    """
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # v2 的两步：先存，再从 buffer 采样更新
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            total_reward += reward
            state = next_state

            if done:
                break

        agent.episode_rewards.append(total_reward)
        agent.episode_steps.append(step + 1)
        agent.epsilon_history.append(agent.epsilon)
        agent.decay_epsilon()

        if (episode + 1) % 50 == 0:
            recent_rewards = agent.episode_rewards[-50:]
            recent_avg = np.mean(recent_rewards)
            buffer_size = len(agent.replay_buffer)
            print(f"  Episode {episode + 1:4d} | "
                  f"Avg Reward: {recent_avg:6.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Buffer: {buffer_size:5d} | "
                  f"Loss: {np.mean(agent.losses[-50:]) if agent.losses else 0:.4f}")


# ============================================================
# v1 的训练函数（用于对比，从 dqn_v1 导入 agent）
# ============================================================

def train_dqn_v1_for_comparison(env, agent, n_episodes=300, max_steps=500):
    """训练 v1 agent（适配本文件的对比实验）"""
    from dqn_v1 import DQNAgentV1
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break

        agent.episode_rewards.append(total_reward)
        agent.episode_steps.append(step + 1)
        agent.epsilon_history.append(agent.epsilon)
        agent.decay_epsilon()

        if (episode + 1) % 50 == 0:
            recent_rewards = agent.episode_rewards[-50:]
            print(f"  Episode {episode + 1:4d} | "
                  f"Avg Reward: {np.mean(recent_rewards):6.1f} | "
                  f"ε: {agent.epsilon:.3f}")


# ============================================================
# 可视化函数
# ============================================================

def visualize_comparison(agents, labels, title="DQN v1 vs v2",
                         window=20, save_path=None):
    """绘制多个 agent 的学习曲线对比"""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#9E9E9E', '#2196F3', '#4CAF50', '#FF5722']

    for idx, (agent, label) in enumerate(zip(agents, labels)):
        rewards = agent.episode_rewards
        color = colors[idx % len(colors)]
        ax.plot(rewards, alpha=0.15, color=color)
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(rewards)), smoothed,
                    color=color, linewidth=2, label=f'{label} (滑动平均)')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title(title)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=500, color='green', linestyle='--', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_buffer_effect(agent_v2, save_path=None):
    """
    可视化经验回放的效果

    展示 buffer 中的样本分布 vs 在线学习的样本分布
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：loss 曲线（v2 的 loss 应该更平滑）
    if agent_v2.losses:
        losses = agent_v2.losses
        axes[0].plot(losses, alpha=0.2, color='#2196F3')
        window = 100
        if len(losses) >= window:
            smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
            axes[0].plot(range(window - 1, len(losses)), smoothed,
                         color='#2196F3', linewidth=2, label='Loss (滑动平均)')
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('DQN v2 训练 Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # 右图：buffer 中样本的 reward 分布
    if len(agent_v2.replay_buffer) > 0:
        buffer_rewards = [t[2] for t in agent_v2.replay_buffer.buffer]
        axes[1].hist(buffer_rewards, bins=30, color='#2196F3', alpha=0.7, edgecolor='white')
        axes[1].set_xlabel('Reward')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Buffer 中的 Reward 分布\n(共 {len(agent_v2.replay_buffer)} 条经验)')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_stability(rewards_runs_v1, rewards_runs_v2, save_path=None):
    """对比 v1 和 v2 在多次运行中的稳定性"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    window = 20

    # 左图：v1 的多次运行
    for i, rewards in enumerate(rewards_runs_v1):
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
            axes[0].plot(range(window - 1, len(rewards)), smoothed,
                         alpha=0.7, label=f'Run {i + 1}')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward (滑动平均)')
    axes[0].set_title('DQN v1（无经验回放）\n不同运行之间差异大')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=500, color='green', linestyle='--', alpha=0.3)

    # 右图：v2 的多次运行
    for i, rewards in enumerate(rewards_runs_v2):
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
            axes[1].plot(range(window - 1, len(rewards)), smoothed,
                         alpha=0.7, label=f'Run {i + 1}')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Total Reward (滑动平均)')
    axes[1].set_title('DQN v2（有经验回放）\n不同运行之间更一致')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=500, color='green', linestyle='--', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================
# 实验一：v1 vs v2 直接对比
# ============================================================

def experiment_v1_vs_v2():
    """
    核心实验：v1（无回放）vs v2（有回放）

    预期：
    - v2 学习更快、更稳定
    - v2 的曲线更平滑（方差更小）
    - v2 最终性能更高
    """
    print("=" * 60)
    print("实验一：DQN v1 vs v2 直接对比")
    print("=" * 60)
    print()
    print("v1：在线学习（每步立即更新，单样本）")
    print("v2：经验回放（存入 buffer，随机采样 batch 更新）")
    print()

    from dqn_v1 import DQNAgentV1

    # v1
    print("训练 DQN v1（无经验回放）...")
    env_v1 = gym.make('CartPole-v1')
    agent_v1 = DQNAgentV1(
        state_dim=4, action_dim=2,
        learning_rate=1e-3, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
        seed=42,
    )
    train_dqn_v1_for_comparison(env_v1, agent_v1, n_episodes=200)
    env_v1.close()

    # v2
    print("\n训练 DQN v2（有经验回放）...")
    env_v2 = gym.make('CartPole-v1')
    agent_v2 = DQNAgentV2(
        state_dim=4, action_dim=2,
        learning_rate=1e-3, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
        buffer_capacity=10000, batch_size=64,
        seed=42,
    )
    train_dqn_v2(env_v2, agent_v2, n_episodes=200)
    env_v2.close()

    # 对比评估
    print("\n" + "=" * 60)
    print("策略评估（关闭探索，20 次运行）")
    print("=" * 60)

    for agent, name in [(agent_v1, 'v1'), (agent_v2, 'v2')]:
        eval_env = gym.make('CartPole-v1')
        eval_rewards = []
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0

        for i in range(20):
            state, _ = eval_env.reset(seed=i + 1000)
            total_reward = 0
            for step in range(500):
                action = agent.select_action(state)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            eval_rewards.append(total_reward)

        agent.epsilon = original_epsilon
        eval_env.close()
        print(f"  DQN {name}: 平均奖励 = {np.mean(eval_rewards):.1f} ± {np.std(eval_rewards):.1f}")

    return agent_v1, agent_v2


# ============================================================
# 实验二：稳定性对比（多次运行）
# ============================================================

def experiment_stability():
    """
    多次独立运行，对比 v1 和 v2 的稳定性

    预期：
    - v1 不同运行之间差异大
    - v2 不同运行之间更一致
    """
    print("\n\n" + "=" * 60)
    print("实验二：训练稳定性对比（3 次独立运行）")
    print("=" * 60)

    from dqn_v1 import DQNAgentV1

    rewards_runs_v1 = []
    rewards_runs_v2 = []

    for run in range(3):
        print(f"\n  --- Run {run + 1}/3 ---")

        # v1
        print(f"  训练 v1 ...")
        env = gym.make('CartPole-v1')
        agent = DQNAgentV1(
            state_dim=4, action_dim=2,
            learning_rate=1e-3, gamma=0.99,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
            seed=run * 100 + 42,
        )
        train_dqn_v1_for_comparison(env, agent, n_episodes=150)
        rewards_runs_v1.append(agent.episode_rewards)
        env.close()
        print(f"    v1 最后 50 轮平均: {np.mean(agent.episode_rewards[-50:]):.1f}")

        # v2
        print(f"  训练 v2 ...")
        env = gym.make('CartPole-v1')
        agent = DQNAgentV2(
            state_dim=4, action_dim=2,
            learning_rate=1e-3, gamma=0.99,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
            buffer_capacity=10000, batch_size=64,
            seed=run * 100 + 42,
        )
        train_dqn_v2(env, agent, n_episodes=150)
        rewards_runs_v2.append(agent.episode_rewards)
        env.close()
        print(f"    v2 最后 50 轮平均: {np.mean(agent.episode_rewards[-50:]):.1f}")

    # 计算稳定性指标
    v1_final_avgs = [np.mean(r[-50:]) for r in rewards_runs_v1]
    v2_final_avgs = [np.mean(r[-50:]) for r in rewards_runs_v2]

    print(f"\n  v1 各运行最终平均: {v1_final_avgs}")
    print(f"  v2 各运行最终平均: {v2_final_avgs}")
    print(f"  v1 运行间标准差: {np.std(v1_final_avgs):.1f}")
    print(f"  v2 运行间标准差: {np.std(v2_final_avgs):.1f}")

    # 可视化
    visualize_stability(
        rewards_runs_v1, rewards_runs_v2,
        save_path='images/dqn_v2_stability_comparison.png',
    )

    return rewards_runs_v1, rewards_runs_v2


# ============================================================
# 实验三：buffer 大小和 batch 大小的影响
# ============================================================

def experiment_hyperparams():
    """
    探索经验回放的关键超参数

    1. buffer 容量：太小 → 多样性不够；太大 → 旧数据太多
    2. batch 大小：太小 → 梯度方差大；太大 → 训练慢
    """
    print("\n\n" + "=" * 60)
    print("实验三：经验回放超参数的影响")
    print("=" * 60)

    configs = [
        {'buffer_capacity': 1000,  'batch_size': 64, 'label': 'Buffer=1K, Batch=64'},
        {'buffer_capacity': 10000, 'batch_size': 64, 'label': 'Buffer=10K, Batch=64'},
        {'buffer_capacity': 10000, 'batch_size': 128, 'label': 'Buffer=10K, Batch=128'},
    ]

    agents = []
    labels = []

    for config in configs:
        print(f"\n  训练 {config['label']} ...")
        env = gym.make('CartPole-v1')
        agent = DQNAgentV2(
            state_dim=4, action_dim=2,
            learning_rate=1e-3, gamma=0.99,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
            buffer_capacity=config['buffer_capacity'],
            batch_size=config['batch_size'],
            seed=42,
        )
        train_dqn_v2(env, agent, n_episodes=200)
        agents.append(agent)
        labels.append(config['label'])
        env.close()

        avg_last_50 = np.mean(agent.episode_rewards[-50:])
        print(f"    最后 50 轮平均: {avg_last_50:.1f}")

    # 可视化
    visualize_comparison(
        agents, labels,
        title='经验回放超参数对比',
        save_path='images/dqn_v2_hyperparams.png',
    )

    print()
    print("分析：")
    print("  - Buffer 太小（1K）：多样性不够，效果打折扣")
    print("  - Buffer 适中（10K）：多样性足够，训练稳定")
    print("  - Batch 太小（32）：梯度方差较大")
    print("  - Batch 适中（64-128）：梯度更稳定，但每步计算量更大")

    return agents, labels


# ============================================================
# 主函数
# ============================================================

if __name__ == '__main__':
    os.makedirs('images', exist_ok=True)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # --- 实验一：v1 vs v2 ---
    agent_v1, agent_v2 = experiment_v1_vs_v2()

    # 可视化对比
    visualize_comparison(
        [agent_v1, agent_v2],
        ['DQN v1（无经验回放）', 'DQN v2（有经验回放）'],
        title='CartPole：DQN v1 vs v2\n经验回放显著提升训练稳定性和性能',
        save_path='images/dqn_v2_vs_v1.png',
    )

    # v2 的 buffer 效果可视化
    visualize_buffer_effect(agent_v2, save_path='images/dqn_v2_buffer_effect.png')

    # --- 实验二：稳定性对比 ---
    rewards_v1, rewards_v2 = experiment_stability()

    # --- 实验三：超参数 ---
    agents_hp, labels_hp = experiment_hyperparams()

    # --- 总结 ---
    print("\n\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print()
    print("DQN v2 = DQN v1 + 经验回放")
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  v1：选动作 → 执行 → 立即更新（单样本，用完就丢）          │")
    print("  │  v2：选动作 → 执行 → 存入 buffer → 随机采样 batch 更新    │")
    print("  │                                                             │")
    print("  │  核心改变：                                                 │")
    print("  │    1. 新增 ReplayBuffer（环形缓冲区，存储经验）             │")
    print("  │    2. update() 改为 batch 更新（随机采样打破相关性）        │")
    print("  │                                                             │")
    print("  │  效果：                                                     │")
    print("  │    ✅ 训练更稳定（随机采样打破数据相关性）                   │")
    print("  │    ✅ 样本效率更高（同一条经验可以被多次使用）              │")
    print("  │    ✅ 学习更快（batch 梯度比单样本梯度方向更准）            │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print("  ❌ 仍未解决的问题：")
    print("     移动目标：TD target 仍然用同一个网络计算")
    print("     → v3 解决方案：引入目标网络，固定 target 一段时间")
    print()
    print("  回忆：为什么 SARSA 不能用经验回放？")
    print("     SARSA 是 on-policy → 需要当前策略的 a'")
    print("     经验回放中的 a' 是旧策略选的 → 不代表当前策略")
    print("     Q-Learning/DQN 是 off-policy → 用 max，不需要 a' → 可以用经验回放")
    print()
    print("  下一步：")
    print("     dqn_v3.py → 引入目标网络，稳定 TD target")
