"""
dqn_v4_double_dqn.py - DQN v4：Double DQN

在 v3 的基础上只改一个地方：TD target 的计算方式

v3 的遗留问题：max 操作的过估计偏差
    TD target = r + γ max_a' Q_θ⁻(s', a')
                      ^^^ 同一个网络既选动作又评估
    max 会系统性地挑中 Q 值被高估的动作
    → Q 值越来越膨胀 → target 越来越大 → 正反馈循环

Double DQN 如何解决：
    TD target = r + γ Q_θ⁻(s', argmax_a' Q_θ(s', a'))
    "选动作"用 q_network（θ），"评估"用 target_network（θ⁻）
    两个网络的估计误差不相关 → max 选中的动作不一定在 θ⁻ 中也被高估

类比：
    v3 = 让同一个老师出题并打分 → 总给自己出的题高分
    v4 = 让 A 老师出题，B 老师打分 → A 觉得好的题，B 不一定给高分

学习要点：
1. max 操作为什么导致过估计（Jensen 不等式）
2. "选"和"评"解耦的思想
3. 代码改动极小（只改 TD target 计算的 2-3 行）
4. Q 值过估计的可视化验证
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from dqn_v1 import QNetwork
from dqn_v2_replay_buffer import ReplayBuffer


# ============================================================
# DQN v4 智能体
# ============================================================

class DQNAgentV4:
    """
    DQN v4：Double DQN（v3 + 选评解耦）

    与 v3 的差异（只有一处）：
    ┌─────────────────┬──────────────────────────┬──────────────────────────────┐
    │                 │ DQN v3                   │ DQN v4 (Double DQN)          │
    ├─────────────────┼──────────────────────────┼──────────────────────────────┤
    │ TD target       │ max_a' Q_θ⁻(s', a')      │ Q_θ⁻(s', argmax_a' Q_θ(s')) │
    │                 │ θ⁻ 选动作 + θ⁻ 评估      │ θ 选动作 + θ⁻ 评估           │
    └─────────────────┴──────────────────────────┴──────────────────────────────┘

    其他一切（双网络架构、ε-greedy、经验回放、目标网络同步）与 v3 完全相同。
    """

    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_capacity=10000, batch_size=64,
                 target_update_freq=500, seed=42):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        self.q_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.train_step = 0

        self.episode_rewards = []
        self.episode_steps = []
        self.losses = []
        self.epsilon_history = []
        self.target_sync_steps = []
        # v4 新增：追踪 Q 值估计，用于过估计可视化
        self.q_value_estimates = []

    def select_action(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.action_dim)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """
        Double DQN 的核心改动在这里

        v3: next_q = target_network(s') → max
            target_network 既选动作又评估 → 过估计

        v4: best_action = q_network(s').argmax()     ← q_network 选
            next_q = target_network(s')[best_action]  ← target_network 评
            选和评用不同网络 → 过估计大幅减少
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)

        q_values = self.q_network(states_tensor)
        current_q = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # ========== v4 的核心改动（对比 v3 只改了这 4 行）==========
        with torch.no_grad():
            # 第一步：用 q_network 选出最优动作（"选"）
            next_q_online = self.q_network(next_states_tensor)
            best_actions = next_q_online.argmax(dim=1)

            # 第二步：用 target_network 评估该动作的 Q 值（"评"）
            next_q_target = self.target_network(next_states_tensor)
            max_next_q = next_q_target.gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)

            td_targets = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q
        # ===========================================================

        loss = nn.functional.mse_loss(current_q, td_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())

        # 记录 Q 值估计（用于过估计分析）
        with torch.no_grad():
            self.q_value_estimates.append(q_values.max(dim=1).values.mean().item())

        self.train_step += 1
        self._update_target_network()

    def _update_target_network(self):
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_sync_steps.append(self.train_step)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================================
# 为 v3 也添加 Q 值追踪（用于公平对比）
# ============================================================

class DQNAgentV3WithQTracking:
    """v3 的副本，增加 Q 值追踪，用于实验 2 的公平对比"""

    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_capacity=10000, batch_size=64,
                 target_update_freq=500, seed=42):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        self.q_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.train_step = 0

        self.episode_rewards = []
        self.episode_steps = []
        self.losses = []
        self.epsilon_history = []
        self.target_sync_steps = []
        self.q_value_estimates = []

    def select_action(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.action_dim)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """v3 的 update：target_network 既选又评（会过估计）"""
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)

        q_values = self.q_network(states_tensor)
        current_q = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # v3 的 TD target：target_network 既选又评
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor)
            max_next_q = next_q_values.max(dim=1).values
            td_targets = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q

        loss = nn.functional.mse_loss(current_q, td_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())

        with torch.no_grad():
            self.q_value_estimates.append(q_values.max(dim=1).values.mean().item())

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_sync_steps.append(self.train_step)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================================
# 训练函数
# ============================================================

def train_dqn(env, agent, n_episodes=300, max_steps=500, verbose=True):
    """通用训练循环，兼容 v3/v4 agent"""
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

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

        if verbose and (episode + 1) % 50 == 0:
            recent_rewards = agent.episode_rewards[-50:]
            recent_avg = np.mean(recent_rewards)
            recent_loss = np.mean(agent.losses[-200:]) if agent.losses else 0
            print(f"  Episode {episode + 1:4d} | "
                  f"Avg Reward: {recent_avg:6.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Loss: {recent_loss:.4f}")


def evaluate_agent(agent, n_episodes=20, max_steps=500, seed_base=1000):
    eval_env = gym.make('CartPole-v1')
    rewards = []
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    for i in range(n_episodes):
        state, _ = eval_env.reset(seed=seed_base + i)
        total_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    agent.epsilon = original_epsilon
    eval_env.close()
    return np.mean(rewards), np.std(rewards), rewards


# ============================================================
# 可视化函数
# ============================================================

def smooth(values, window):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode='valid')


def visualize_v3_vs_v4(agent_v3, agent_v4, save_path=None):
    """实验 1：v3 vs v4 reward + loss 对比"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    window = 20

    for agent, label, color in [(agent_v3, 'DQN v3 (标准 DQN)', '#FF9800'),
                                  (agent_v4, 'DQN v4 (Double DQN)', '#2196F3')]:
        rewards = agent.episode_rewards
        axes[0].plot(rewards, alpha=0.15, color=color)
        smoothed = smooth(rewards, window)
        axes[0].plot(range(window - 1, len(rewards)), smoothed,
                     color=color, linewidth=2, label=f'{label}')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Reward 曲线')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=500, color='green', linestyle='--', alpha=0.3)

    loss_window = 100
    for agent, label, color in [(agent_v3, 'DQN v3', '#FF9800'),
                                  (agent_v4, 'DQN v4', '#2196F3')]:
        losses = agent.losses
        if losses:
            smoothed = smooth(losses, loss_window)
            axes[1].plot(range(loss_window - 1, len(losses)), smoothed,
                         color=color, linewidth=2, label=f'{label} loss')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_yscale('log')
    axes[1].set_title('TD Loss 曲线（log 刻度）')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, which='both')

    plt.suptitle('实验 1: DQN v3 vs v4（Double DQN）', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_overestimation(agent_v3, agent_v4, save_path=None):
    """
    实验 2：Q 值过估计可视化——Double DQN 的核心卖点

    上：v3 和 v4 的 Q 值估计随训练步数的变化
    下：每个 episode 的实际 reward（作为"真实价值"的参考）
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[1.2, 1])
    q_window = 500

    # 上图：Q 值估计
    for agent, label, color in [(agent_v3, 'DQN v3 (标准 DQN)', '#FF9800'),
                                  (agent_v4, 'DQN v4 (Double DQN)', '#2196F3')]:
        q_vals = agent.q_value_estimates
        if q_vals:
            axes[0].plot(q_vals, alpha=0.08, color=color)
            smoothed = smooth(q_vals, q_window)
            axes[0].plot(range(q_window - 1, len(q_vals)), smoothed,
                         color=color, linewidth=2, label=label)
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Estimated max Q(s, ·)')
    axes[0].set_title('训练过程中的 Q 值估计\n'
                       '(v3 的 Q 值显著高于 v4 → 过估计)')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 下图：实际 reward
    reward_window = 20
    for agent, label, color in [(agent_v3, 'DQN v3', '#FF9800'),
                                  (agent_v4, 'DQN v4', '#2196F3')]:
        rewards = agent.episode_rewards
        smoothed = smooth(rewards, reward_window)
        axes[1].plot(range(reward_window - 1, len(rewards)), smoothed,
                     color=color, linewidth=2, label=label)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Actual Episode Reward')
    axes[1].set_title('实际 Reward（参考）\n'
                       '(注意：Q 值是折扣回报，不能直接与原始 reward 比较)')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('实验 2: Q 值过估计可视化——max 操作的系统性偏差',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_acrobot(agent_v3, agent_v4, save_path=None):
    """实验 3：Acrobot 上 v3 vs v4 对比"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    window = 30

    # 左：reward
    for agent, label, color in [(agent_v3, 'DQN v3', '#FF9800'),
                                  (agent_v4, 'DQN v4', '#2196F3')]:
        rewards = agent.episode_rewards
        axes[0].plot(rewards, alpha=0.1, color=color)
        smoothed = smooth(rewards, window)
        axes[0].plot(range(window - 1, len(rewards)), smoothed,
                     color=color, linewidth=2, label=label)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Acrobot Reward\n(越接近 0 越好)')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 中：loss
    loss_window = 100
    for agent, label, color in [(agent_v3, 'DQN v3', '#FF9800'),
                                  (agent_v4, 'DQN v4', '#2196F3')]:
        losses = agent.losses
        if losses:
            smoothed = smooth(losses, loss_window)
            axes[1].plot(range(loss_window - 1, len(losses)), smoothed,
                         color=color, linewidth=2, label=label)
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_yscale('log')
    axes[1].set_title('TD Loss（log 刻度）')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, which='both')

    # 右：Q 值估计
    q_window = 300
    for agent, label, color in [(agent_v3, 'DQN v3', '#FF9800'),
                                  (agent_v4, 'DQN v4', '#2196F3')]:
        q_vals = agent.q_value_estimates
        if q_vals:
            smoothed = smooth(q_vals, q_window)
            axes[2].plot(range(q_window - 1, len(q_vals)), smoothed,
                         color=color, linewidth=2, label=label)
    axes[2].set_xlabel('Training Step')
    axes[2].set_ylabel('Estimated max Q(s, ·)')
    axes[2].set_title('Q 值估计\n(v3 过估计更明显)')
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('实验 3: Acrobot-v1——3 个动作下过估计效应更明显',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================
# 实验 1：v3 vs v4 直接对比
# ============================================================

def experiment_v3_vs_v4():
    """
    核心实验：v3（标准 DQN）vs v4（Double DQN）

    v3 的 max 操作会过估计 Q 值；v4 用选评解耦减少偏差
    """
    print("=" * 60)
    print("实验 1: DQN v3 vs v4 (Double DQN)")
    print("=" * 60)
    print()
    print("v3: TD target 中 target_network 既选又评 → 过估计")
    print("v4: q_network 选动作, target_network 评估 → 选评解耦")
    print()

    n_eps = 500

    # v3（带 Q 值追踪）
    print("训练 DQN v3 ...")
    env = gym.make('CartPole-v1')
    agent_v3 = DQNAgentV3WithQTracking(
        state_dim=4, action_dim=2,
        learning_rate=1e-3, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
        buffer_capacity=10000, batch_size=64,
        target_update_freq=500, seed=42,
    )
    train_dqn(env, agent_v3, n_episodes=n_eps)
    env.close()

    # v4
    print("\n训练 DQN v4 (Double DQN) ...")
    env = gym.make('CartPole-v1')
    agent_v4 = DQNAgentV4(
        state_dim=4, action_dim=2,
        learning_rate=1e-3, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
        buffer_capacity=10000, batch_size=64,
        target_update_freq=500, seed=42,
    )
    train_dqn(env, agent_v4, n_episodes=n_eps)
    env.close()

    # 评估
    print("\n" + "=" * 60)
    print("策略评估（关闭探索，20 次运行）")
    print("=" * 60)
    for agent, name in [(agent_v3, 'v3'), (agent_v4, 'v4')]:
        mean, std, _ = evaluate_agent(agent)
        print(f"  DQN {name}: 平均奖励 = {mean:.1f} ± {std:.1f}")

    # Q 值对比
    if agent_v3.q_value_estimates and agent_v4.q_value_estimates:
        v3_q_mean = np.mean(agent_v3.q_value_estimates[-1000:])
        v4_q_mean = np.mean(agent_v4.q_value_estimates[-1000:])
        print(f"\n  v3 末段 Q 值均值: {v3_q_mean:.2f}")
        print(f"  v4 末段 Q 值均值: {v4_q_mean:.2f}")
        print(f"  差值 (v3 - v4): {v3_q_mean - v4_q_mean:.2f} (正值 = v3 过估计)")

    # Loss 对比
    if agent_v3.losses and agent_v4.losses:
        v3_loss = np.mean(agent_v3.losses[-1000:])
        v4_loss = np.mean(agent_v4.losses[-1000:])
        print(f"\n  v3 末段 loss: {v3_loss:.4f}")
        print(f"  v4 末段 loss: {v4_loss:.4f}")

    return agent_v3, agent_v4


# ============================================================
# 实验 2：Q 值过估计可视化（与实验 1 共享 agent）
# ============================================================

def experiment_overestimation(agent_v3, agent_v4):
    """
    用实验 1 训练好的 agent 做过估计分析

    对比：
    - v3 和 v4 的 Q 值估计随训练的变化
    - 实际 reward 作为"真实价值"的参考
    """
    print("\n\n" + "=" * 60)
    print("实验 2: Q 值过估计分析")
    print("=" * 60)

    v3_q = agent_v3.q_value_estimates
    v4_q = agent_v4.q_value_estimates

    if v3_q and v4_q:
        v3_final = np.mean(v3_q[-1000:])
        v4_final = np.mean(v4_q[-1000:])
        v3_max = np.max(v3_q)
        v4_max = np.max(v4_q)
        v3_reward = np.mean(agent_v3.episode_rewards[-100:])
        v4_reward = np.mean(agent_v4.episode_rewards[-100:])

        print(f"\n  v3 Q 值末段均值: {v3_final:.2f} (峰值: {v3_max:.2f})")
        print(f"  v4 Q 值末段均值: {v4_final:.2f} (峰值: {v4_max:.2f})")
        print(f"\n  v3 - v4 差值: {v3_final - v4_final:.2f}")
        print(f"  v3 相对 v4 高出: {(v3_final - v4_final) / max(abs(v4_final), 1e-6) * 100:.1f}%")
        print(f"  (正值 = v3 过估计，差值越大过估计越严重)")


# ============================================================
# 实验 3：Acrobot（更难的任务，3 个动作）
# ============================================================

def experiment_acrobot():
    """
    Acrobot-v1 上的 v3 vs v4 对比

    Acrobot 有 3 个动作（vs CartPole 的 2 个）
    动作越多，max 过估计越严重 → Double DQN 的优势越明显
    """
    print("\n\n" + "=" * 60)
    print("实验 3: Acrobot-v1（3 个动作）上 v3 vs v4")
    print("=" * 60)
    print()
    print("Acrobot: 6 维状态, 3 个动作")
    print("动作数更多 → max 过估计更严重 → Double DQN 优势更明显")
    print()

    n_eps = 500

    print("训练 DQN v3 ...")
    env = gym.make('Acrobot-v1')
    agent_v3 = DQNAgentV3WithQTracking(
        state_dim=6, action_dim=3,
        learning_rate=1e-3, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.01,
        buffer_capacity=10000, batch_size=64,
        target_update_freq=500, seed=42,
    )
    train_dqn(env, agent_v3, n_episodes=n_eps)
    env.close()

    print("\n训练 DQN v4 (Double DQN) ...")
    env = gym.make('Acrobot-v1')
    agent_v4 = DQNAgentV4(
        state_dim=6, action_dim=3,
        learning_rate=1e-3, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.01,
        buffer_capacity=10000, batch_size=64,
        target_update_freq=500, seed=42,
    )
    train_dqn(env, agent_v4, n_episodes=n_eps)
    env.close()

    print("\n" + "=" * 60)
    print("结果对比")
    print("=" * 60)

    for agent, name in [(agent_v3, 'v3'), (agent_v4, 'v4')]:
        last_100 = np.mean(agent.episode_rewards[-100:])
        loss_mean = np.mean(agent.losses[-500:]) if agent.losses else 0
        q_mean = np.mean(agent.q_value_estimates[-1000:]) if agent.q_value_estimates else 0
        print(f"  DQN {name}: reward={last_100:.1f} | loss={loss_mean:.4f} | Q 值={q_mean:.2f}")

    return agent_v3, agent_v4


# ============================================================
# 实验 4：多 seed 聚合（消除单次运行的偶然性）
# ============================================================

def experiment_multi_seed():
    """
    多 seed 聚合对比，消除单次运行的偶然性

    CartPole 只有 2 个动作，过估计信号弱——需要多次运行取平均
    """
    print("\n\n" + "=" * 60)
    print("实验 4: 多 seed 聚合（5 个 seed）")
    print("=" * 60)

    n_eps = 500
    seeds = [42, 142, 242, 342, 442]

    v3_all_rewards = []
    v4_all_rewards = []
    v3_all_q = []
    v4_all_q = []
    v3_evals = []
    v4_evals = []

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")

        env = gym.make('CartPole-v1')
        agent_v3 = DQNAgentV3WithQTracking(
            state_dim=4, action_dim=2,
            learning_rate=1e-3, gamma=0.99,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
            buffer_capacity=10000, batch_size=64,
            target_update_freq=500, seed=seed,
        )
        train_dqn(env, agent_v3, n_episodes=n_eps, verbose=False)
        env.close()

        env = gym.make('CartPole-v1')
        agent_v4 = DQNAgentV4(
            state_dim=4, action_dim=2,
            learning_rate=1e-3, gamma=0.99,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
            buffer_capacity=10000, batch_size=64,
            target_update_freq=500, seed=seed,
        )
        train_dqn(env, agent_v4, n_episodes=n_eps, verbose=False)
        env.close()

        v3_last = np.mean(agent_v3.episode_rewards[-100:])
        v4_last = np.mean(agent_v4.episode_rewards[-100:])
        v3_q = np.mean(agent_v3.q_value_estimates[-1000:])
        v4_q = np.mean(agent_v4.q_value_estimates[-1000:])

        v3_all_rewards.append(v3_last)
        v4_all_rewards.append(v4_last)
        v3_all_q.append(v3_q)
        v4_all_q.append(v4_q)

        v3_eval, _, _ = evaluate_agent(agent_v3)
        v4_eval, _, _ = evaluate_agent(agent_v4)
        v3_evals.append(v3_eval)
        v4_evals.append(v4_eval)

        print(f"    v3: reward={v3_last:.1f}, Q={v3_q:.1f}, eval={v3_eval:.1f}")
        print(f"    v4: reward={v4_last:.1f}, Q={v4_q:.1f}, eval={v4_eval:.1f}")

    print(f"\n  === 聚合结果（{len(seeds)} 个 seed 的平均）===")
    print(f"  v3 评估 reward: {np.mean(v3_evals):.1f} ± {np.std(v3_evals):.1f}")
    print(f"  v4 评估 reward: {np.mean(v4_evals):.1f} ± {np.std(v4_evals):.1f}")
    print(f"  v3 Q 值均值: {np.mean(v3_all_q):.2f}")
    print(f"  v4 Q 值均值: {np.mean(v4_all_q):.2f}")
    print(f"  Q 值差 (v3 - v4): {np.mean(v3_all_q) - np.mean(v4_all_q):.2f}")

    return {
        'v3_rewards': v3_all_rewards, 'v4_rewards': v4_all_rewards,
        'v3_q': v3_all_q, 'v4_q': v4_all_q,
        'v3_evals': v3_evals, 'v4_evals': v4_evals,
    }


# ============================================================
# 主函数
# ============================================================

if __name__ == '__main__':
    os.makedirs('images', exist_ok=True)

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # --- 实验 1：v3 vs v4 ---
    agent_v3, agent_v4 = experiment_v3_vs_v4()
    visualize_v3_vs_v4(agent_v3, agent_v4,
                        save_path='images/dqn_v4_vs_v3.png')

    # --- 实验 2：Q 值过估计（用实验 1 的 agent）---
    experiment_overestimation(agent_v3, agent_v4)
    visualize_overestimation(agent_v3, agent_v4,
                              save_path='images/dqn_v4_overestimation.png')

    # --- 实验 3：Acrobot ---
    agent_v3_acro, agent_v4_acro = experiment_acrobot()
    visualize_acrobot(agent_v3_acro, agent_v4_acro,
                       save_path='images/dqn_v4_acrobot.png')

    # --- 实验 4：多 seed 聚合 ---
    multi_seed_results = experiment_multi_seed()

    # --- 总结 ---
    print("\n\n" + "=" * 60)
    print("总结：Double DQN")
    print("=" * 60)
    print()
    print("DQN v4 = DQN v3 + 选评解耦")
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  v3: TD target = r + γ max_a' Q_θ⁻(s', a')                 │")
    print("  │       θ⁻ 既选又评 → max 导致过估计                         │")
    print("  │                                                             │")
    print("  │  v4: TD target = r + γ Q_θ⁻(s', argmax_a' Q_θ(s', a'))     │")
    print("  │       θ 选动作, θ⁻ 评估 → 选评解耦 → 过估计减少             │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print("  代码改动：仅 update() 中 TD target 的 2-3 行")
    print("  不增加任何网络、超参数或计算量")
    print()
    print("  DQN 演进路线：")
    print("    v1 = 神经网络近似 Q 函数")
    print("    v2 = + 经验回放（解决数据相关性）")
    print("    v3 = + 目标网络（解决移动目标）")
    print("    v4 = + 选评解耦（解决 max 过估计）← Double DQN")
    print()
    print("  下一步可探索：")
    print("    - Dueling DQN：分离 V(s) 和 A(s,a)")
    print("    - Prioritized Replay：让重要经验被采样更多")
    print("    - Rainbow：所有改进的集大成者")
