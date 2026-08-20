# -*- coding: utf-8 -*-
"""
dqn_v5_dueling_dqn.py - DQN v5：Dueling DQN

在 v4 的基础上改一个地方：Q 网络的架构

v4 的遗留问题：Q 网络的表达效率
    Q_θ(s) → [Q(s,a0), Q(s,a1), ..., Q(s,an)]
    网络必须为每个 (s, a) 独立估计 Q 值
    但很多状态下，"状态本身好不好" 比 "选哪个动作" 重要得多

    例如 CartPole：
    - 杆子快要倒了 → 不管往哪推，都要完蛋 → V(s) 低，A(s,a) 差别不大
    - 杆子很稳 → 往哪推都行 → V(s) 高，A(s,a) 差别不大
    - 只有杆子微偏时，动作选择才真正重要 → A(s,a) 有显著差异

Dueling DQN 如何解决：
    将 Q(s,a) 分解为两部分：
        Q(s, a) = V(s) + A(s, a) - mean_a'(A(s, a'))
                   ^^^^   ^^^^^^^^
                   状态价值   动作优势

    - V(s)：待在状态 s 本身有多好（跟动作无关）
    - A(s,a)：在状态 s 下选动作 a 比平均水平好多少
    - 减去 mean 是为了可辨识性（identifiability）

类比：
    v4 网络 = 对每道菜直接打分（分数之间没有结构）
    v5 网络 = "餐厅评分" + "每道菜相对于餐厅平均水平的偏差"
    好处：如果餐厅整体很好（V 高），你不需要尝遍每道菜就知道分数不会低

学习要点：
1. V(s) 和 A(s,a) 的分解思想——引出 Actor-Critic 的前奏
2. 为什么需要减去 mean(A)——可辨识性问题
3. 网络架构的变化（共享特征层 + 两个独立流）
4. 代码改动集中在 DuelingQNetwork，算法（Double DQN）不变
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from dqn_v2_replay_buffer import ReplayBuffer


# ============================================================
# Dueling Q 网络：v5 的核心新组件
# ============================================================

class DuelingQNetwork(nn.Module):
    """
    Dueling Q 网络：将 Q 分解为 V + A

    架构对比：
    ┌─────────────────────────────────────────────────────────────┐
    │  v1-v4 Q 网络:                                              │
    │    state → [shared layers] → Q(s, a) for all a              │
    │                                                             │
    │  v5 Dueling 网络:                                            │
    │    state → [shared layers] ─┬─ [value stream]     → V(s)    │
    │                             └─ [advantage stream] → A(s, a) │
    │                                                             │
    │    Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))               │
    └─────────────────────────────────────────────────────────────┘

    为什么要减去 mean(A)？
        如果 Q = V + A（不减 mean），那么对于同一组 Q 值：
            V=10, A=[2, -2] → Q=[12, 8]
            V=12, A=[0, -4] → Q=[12, 8]   ← 同样的 Q，但 V 和 A 不同！
        网络无法区分 "哪些是 V 的贡献，哪些是 A 的贡献"
        减去 mean 后：A 的均值被强制为 0，V 就唯一地等于 Q 的均值
            V = mean(Q(s, ·))，A(s, a) = Q(s, a) - mean(Q(s, ·))
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.action_dim = action_dim

        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # 价值流（Value Stream）：输出标量 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 优势流（Advantage Stream）：输出 A(s, a) for each a
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        """
        前向传播：state → Q(s, a) = V(s) + A(s, a) - mean(A)

        Returns:
            Q 值张量，shape = (batch_size, action_dim)
            接口与 v1 的 QNetwork 完全一致——外部代码无感知
        """
        features = self.feature(state)

        value = self.value_stream(features)           # (batch, 1)
        advantage = self.advantage_stream(features)   # (batch, action_dim)

        # Q = V + (A - mean(A))
        # 减去 mean 使得 advantage 的均值为 0 → V 唯一代表状态价值
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def get_value_and_advantage(self, state):
        """额外接口：返回分解后的 V 和 A（用于可视化分析）"""
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value, advantage


# ============================================================
# DQN v5 智能体
# ============================================================

class DQNAgentV5:
    """
    DQN v5：Dueling DQN（v4 + V/A 分解架构）

    与 v4 的差异（只有网络架构）：
    ┌─────────────────┬───────────────────────────┬────────────────────────────────┐
    │                 │ DQN v4 (Double DQN)       │ DQN v5 (Dueling DQN)           │
    ├─────────────────┼───────────────────────────┼────────────────────────────────┤
    │ 网络架构        │ state → Q(s, a)           │ state → V(s) + A(s,a) - mean  │
    │ TD target      │ Double DQN（选评解耦）     │ 同 v4（Double DQN）            │
    │ 经验回放        │ ✓                         │ ✓                              │
    │ 目标网络        │ ✓                         │ ✓（也用 Dueling 架构）         │
    └─────────────────┴───────────────────────────┴────────────────────────────────┘

    Dueling 的好处：
    - 在很多状态下，动作之间差别不大 → V 流可以高效学习状态价值
    - 不需要每个 (s,a) 都被访问过就能估计 V(s) → 泛化更好
    - 尤其在动作空间大时优势显著（有些动作几乎等价）
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

        # v5 的核心改动：使用 DuelingQNetwork 替代 QNetwork
        self.q_network = DuelingQNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.target_network = DuelingQNetwork(state_dim, action_dim)
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
        # v5 新增：追踪 V 和 A 的分解情况
        self.value_estimates = []
        self.advantage_magnitudes = []

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
        更新逻辑与 v4 完全一致（Double DQN）
        唯一的区别是 q_network 和 target_network 内部是 Dueling 架构
        从外部看，forward() 的输入输出格式不变 → 算法代码零改动
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

        # Double DQN TD target（与 v4 完全一致）
        with torch.no_grad():
            next_q_online = self.q_network(next_states_tensor)
            best_actions = next_q_online.argmax(dim=1)

            next_q_target = self.target_network(next_states_tensor)
            max_next_q = next_q_target.gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)

            td_targets = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q

        loss = nn.functional.mse_loss(current_q, td_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())

        # 追踪 Q 值和 V/A 分解
        with torch.no_grad():
            self.q_value_estimates.append(q_values.max(dim=1).values.mean().item())
            value, advantage = self.q_network.get_value_and_advantage(states_tensor)
            self.value_estimates.append(value.mean().item())
            self.advantage_magnitudes.append(advantage.abs().mean().item())

        self.train_step += 1
        self._update_target_network()

    def _update_target_network(self):
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_sync_steps.append(self.train_step)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================================
# v4 agent 的副本（用于公平对比）
# ============================================================

class DQNAgentV4ForComparison:
    """v4 (Double DQN with standard QNetwork) 用于与 v5 对比"""

    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_capacity=10000, batch_size=64,
                 target_update_freq=500, hidden_dim=128, seed=42):
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

        # 标准 Q 网络（与 v5 的 Dueling 对比）
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
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

        with torch.no_grad():
            next_q_online = self.q_network(next_states_tensor)
            best_actions = next_q_online.argmax(dim=1)
            next_q_target = self.target_network(next_states_tensor)
            max_next_q = next_q_target.gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)
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


def evaluate_agent(agent, env_name='CartPole-v1', n_episodes=20,
                   max_steps=500, seed_base=1000):
    eval_env = gym.make(env_name)
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


def visualize_v4_vs_v5(agent_v4, agent_v5, save_path=None):
    """实验 1：v4 vs v5 reward + loss 对比"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    window = 20

    for agent, label, color in [(agent_v4, 'DQN v4 (Double DQN)', '#FF9800'),
                                (agent_v5, 'DQN v5 (Dueling DQN)', '#2196F3')]:
        rewards = agent.episode_rewards
        axes[0].plot(rewards, alpha=0.15, color=color)
        smoothed = smooth(rewards, window)
        axes[0].plot(range(window - 1, len(rewards)), smoothed,
                     color=color, linewidth=2, label=label)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Reward Curve: Dueling Architecture Accelerates Learning')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=500, color='green', linestyle='--', alpha=0.3)

    loss_window = 100
    for agent, label, color in [(agent_v4, 'DQN v4', '#FF9800'),
                                (agent_v5, 'DQN v5', '#2196F3')]:
        losses = agent.losses
        if losses:
            smoothed = smooth(losses, loss_window)
            axes[1].plot(range(loss_window - 1, len(losses)), smoothed,
                         color=color, linewidth=2, label=f'{label} loss')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_yscale('log')
    axes[1].set_title('TD Loss (log scale)')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, which='both')

    plt.suptitle('Exp 1: DQN v4 vs v5 (Dueling DQN)', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_va_decomposition(agent_v5, save_path=None):
    """
    实验 2：V/A 分解可视化——Dueling 的核心洞察

    上：V(s) 随训练的变化 → 网络学到了"状态本身好不好"
    中：|A(s,a)| 的均值 → 反映"动作选择有多重要"
    下：Q = V + A 的对比
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    window = 500

    # V(s) 的变化
    v_vals = agent_v5.value_estimates
    if v_vals:
        axes[0].plot(v_vals, alpha=0.08, color='#4CAF50')
        smoothed = smooth(v_vals, window)
        axes[0].plot(range(window - 1, len(v_vals)), smoothed,
                     color='#4CAF50', linewidth=2, label='V(s) state value')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('V(s)')
    axes[0].set_title('Value Stream: learned state value V(s)\n'
                      '(grows with training - network learns "how good is this state")')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # |A(s,a)| 的变化
    a_vals = agent_v5.advantage_magnitudes
    if a_vals:
        axes[1].plot(a_vals, alpha=0.08, color='#FF5722')
        smoothed = smooth(a_vals, window)
        axes[1].plot(range(window - 1, len(a_vals)), smoothed,
                     color='#FF5722', linewidth=2, label='|A(s,a)| advantage magnitude')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('mean |A(s,a)|')
    axes[1].set_title('Advantage Stream: magnitude of action differences\n'
                      '(small |A| = action choice mostly irrelevant, V dominates)')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Q 值 = V + A
    q_vals = agent_v5.q_value_estimates
    if q_vals and v_vals:
        q_smoothed = smooth(q_vals, window)
        v_smoothed = smooth(v_vals, window)
        n = min(len(q_smoothed), len(v_smoothed))
        axes[2].plot(range(window - 1, window - 1 + n), q_smoothed[:n],
                     color='#2196F3', linewidth=2, label='Q(s,a) = V + A')
        axes[2].plot(range(window - 1, window - 1 + n), v_smoothed[:n],
                     color='#4CAF50', linewidth=2, label='V(s) component')
        axes[2].fill_between(
            range(window - 1, window - 1 + n),
            v_smoothed[:n], q_smoothed[:n],
            alpha=0.2, color='#FF5722', label='A(s,a) component'
        )
    axes[2].set_xlabel('Training Step')
    axes[2].set_ylabel('Value')
    axes[2].set_title('Q = V + A decomposition\n'
                      '(V dominates, A is just fine-tuning - Dueling learns V efficiently)')
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Exp 2: Dueling DQN V/A Decomposition', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_advantage_heatmap(agent_v5, save_path=None):
    """
    实验 3：Advantage 热力图——展示"何时动作选择重要"

    在 CartPole 状态空间中扫描，看 A(s,a) 的幅度：
    - 幅度大：这个状态下选对动作很关键
    - 幅度小：随便选都差不多（V 主导）
    """
    positions = np.linspace(-2.4, 2.4, 40)
    angles = np.linspace(-0.21, 0.21, 40)

    advantage_magnitude = np.zeros((len(angles), len(positions)))
    value_landscape = np.zeros((len(angles), len(positions)))
    best_action = np.zeros((len(angles), len(positions)))

    agent_v5.q_network.eval()
    with torch.no_grad():
        for i, angle in enumerate(angles):
            for j, pos in enumerate(positions):
                state = torch.FloatTensor([[pos, 0.0, angle, 0.0]])
                v, a = agent_v5.q_network.get_value_and_advantage(state)
                value_landscape[i, j] = v.item()
                advantage_magnitude[i, j] = a.abs().max().item()
                q = agent_v5.q_network(state)
                best_action[i, j] = q.argmax(dim=1).item()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # V(s) 地形图
    im0 = axes[0].imshow(value_landscape,
                          extent=[positions[0], positions[-1],
                                  angles[-1], angles[0]],
                          aspect='auto', cmap='RdYlGn')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Angle')
    axes[0].set_title('V(s): State Value\n(high in center, low at edges - stable states are valuable)')
    plt.colorbar(im0, ax=axes[0])

    # |A| 热力图
    im1 = axes[1].imshow(advantage_magnitude,
                          extent=[positions[0], positions[-1],
                                  angles[-1], angles[0]],
                          aspect='auto', cmap='hot')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Angle')
    axes[1].set_title('max|A(s,a)|: Action Selection Importance\n(bright = choosing right action is critical)')
    plt.colorbar(im1, ax=axes[1])

    # 最优动作
    im2 = axes[2].imshow(best_action,
                          extent=[positions[0], positions[-1],
                                  angles[-1], angles[0]],
                          aspect='auto', cmap='coolwarm')
    axes[2].set_xlabel('Position')
    axes[2].set_ylabel('Angle')
    axes[2].set_title('Best Action (blue=left, red=right)\n(bright regions in |A| heatmap = decision boundary)')
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle('Exp 3: Dueling DQN learned V/A landscape\n'
                 '(velocity=0, angular_velocity=0)', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_acrobot(agent_v4, agent_v5, save_path=None):
    """实验 4：Acrobot 上 v4 vs v5 对比（动作空间更大 → Dueling 优势更明显）"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    window = 30

    for agent, label, color in [(agent_v4, 'DQN v4', '#FF9800'),
                                (agent_v5, 'DQN v5 (Dueling)', '#2196F3')]:
        rewards = agent.episode_rewards
        axes[0].plot(rewards, alpha=0.1, color=color)
        smoothed = smooth(rewards, window)
        axes[0].plot(range(window - 1, len(rewards)), smoothed,
                     color=color, linewidth=2, label=label)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Acrobot Reward\n(closer to 0 is better)')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    loss_window = 100
    for agent, label, color in [(agent_v4, 'DQN v4', '#FF9800'),
                                (agent_v5, 'DQN v5', '#2196F3')]:
        losses = agent.losses
        if losses:
            smoothed = smooth(losses, loss_window)
            axes[1].plot(range(loss_window - 1, len(losses)), smoothed,
                         color=color, linewidth=2, label=label)
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_yscale('log')
    axes[1].set_title('TD Loss (log scale)')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, which='both')

    q_window = 300
    for agent, label, color in [(agent_v4, 'DQN v4', '#FF9800'),
                                (agent_v5, 'DQN v5', '#2196F3')]:
        q_vals = agent.q_value_estimates
        if q_vals:
            smoothed = smooth(q_vals, q_window)
            axes[2].plot(range(q_window - 1, len(q_vals)), smoothed,
                         color=color, linewidth=2, label=label)
    axes[2].set_xlabel('Training Step')
    axes[2].set_ylabel('Estimated max Q(s, ·)')
    axes[2].set_title('Q Value Estimates')
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Exp 4: Acrobot-v1 - Dueling advantage more visible with more actions',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================
# 实验 1：v4 vs v5 直接对比
# ============================================================

def experiment_v4_vs_v5():
    """
    核心实验：v4（标准网络 + Double DQN）vs v5（Dueling 网络 + Double DQN）

    Dueling 的优势在于 V 流可以从所有经验中学习（不依赖具体动作）
    → 学到好的状态价值估计更快 → 整体收敛更快
    """
    print("=" * 60)
    print("实验 1: DQN v4 vs v5 (Dueling DQN)")
    print("=" * 60)
    print()
    print("v4: state → [layers] → Q(s,a)              （标准网络）")
    print("v5: state → [layers] → V(s) + A(s,a) - mean （Dueling 网络）")
    print()
    print("Dueling 的核心洞察：")
    print("  很多状态下，动作选择无关紧要（|A| ≈ 0）")
    print("  此时 Q ≈ V，V 流可以从所有经验中高效学习")
    print()

    n_eps = 500

    print("训练 DQN v4 (Double DQN, 标准网络) ...")
    env = gym.make('CartPole-v1')
    agent_v4 = DQNAgentV4ForComparison(
        state_dim=4, action_dim=2,
        learning_rate=1e-3, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
        buffer_capacity=10000, batch_size=64,
        target_update_freq=500, seed=42,
    )
    train_dqn(env, agent_v4, n_episodes=n_eps)
    env.close()

    print("\n训练 DQN v5 (Dueling DQN) ...")
    env = gym.make('CartPole-v1')
    agent_v5 = DQNAgentV5(
        state_dim=4, action_dim=2,
        learning_rate=1e-3, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
        buffer_capacity=10000, batch_size=64,
        target_update_freq=500, seed=42,
    )
    train_dqn(env, agent_v5, n_episodes=n_eps)
    env.close()

    print("\n" + "=" * 60)
    print("策略评估（关闭探索，20 次运行）")
    print("=" * 60)
    for agent, name in [(agent_v4, 'v4'), (agent_v5, 'v5')]:
        mean, std, _ = evaluate_agent(agent)
        print(f"  DQN {name}: 平均奖励 = {mean:.1f} ± {std:.1f}")

    return agent_v4, agent_v5


# ============================================================
# 实验 2：V/A 分解分析（与实验 1 共享 agent）
# ============================================================

def experiment_va_analysis(agent_v5):
    """
    分析 Dueling 网络学到的 V/A 分解

    关键观察：
    - V(s) 随训练增长 → 网络学到了状态价值
    - |A(s,a)| 相对 V 很小 → 大部分 Q 值由 V 主导
    - 这说明 CartPole 中"状态好不好"比"选哪个动作"更重要
    """
    print("\n\n" + "=" * 60)
    print("实验 2: V/A 分解分析")
    print("=" * 60)

    v_vals = agent_v5.value_estimates
    a_vals = agent_v5.advantage_magnitudes
    q_vals = agent_v5.q_value_estimates

    if v_vals and a_vals and q_vals:
        v_final = np.mean(v_vals[-1000:])
        a_final = np.mean(a_vals[-1000:])
        q_final = np.mean(q_vals[-1000:])

        print(f"\n  训练末段统计（最后 1000 步）：")
        print(f"  V(s) 均值:     {v_final:.2f}")
        print(f"  |A(s,a)| 均值: {a_final:.2f}")
        print(f"  Q(s,a) 均值:   {q_final:.2f}")
        print(f"  V / Q 比例:    {v_final / max(abs(q_final), 1e-6) * 100:.1f}%")
        print(f"  |A| / Q 比例:  {a_final / max(abs(q_final), 1e-6) * 100:.1f}%")
        print()
        print("  解读：")
        print("  - V 占 Q 的绝大部分 → '状态好不好' 是主要因素")
        print("  - |A| 相对很小 → '选哪个动作' 通常影响不大")
        print("  - Dueling 让网络高效学习 V，不需要为每个 (s,a) 单独估计")


# ============================================================
# 实验 3：Advantage 热力图
# ============================================================

def experiment_advantage_landscape(agent_v5):
    """在 CartPole 状态空间中可视化 V 和 A 的分布"""
    print("\n\n" + "=" * 60)
    print("实验 3: Advantage 热力图")
    print("=" * 60)
    print()
    print("扫描 [位置, 角度] 平面（固定速度=0, 角速度=0）")
    print("  V(s): 中心区域高 → 杆子稳定时状态好")
    print("  |A|:  边缘/倾斜处大 → 快要倒时动作选择关键")
    print('  → 在大部分"安全"区域，随便选动作都行（|A|≈0）')


# ============================================================
# 实验 4：Acrobot（3 个动作）
# ============================================================

def experiment_acrobot():
    """
    Acrobot-v1 上的 v4 vs v5 对比

    Acrobot 有 3 个动作（vs CartPole 的 2 个）
    动作越多，Dueling 优势越大：
    - 标准网络需要为 3 个动作分别估计 Q 值
    - Dueling 只需学好 V(s)，A 的 3 个值围绕 0 波动
    """
    print("\n\n" + "=" * 60)
    print("实验 4: Acrobot-v1（3 个动作）上 v4 vs v5")
    print("=" * 60)
    print()
    print("Acrobot: 6 维状态, 3 个动作")
    print("动作数更多 → 标准网络需要学 3 个独立 Q 值")
    print("Dueling → V 流学状态价值，A 流只学 3 个小偏差")
    print()

    n_eps = 500

    print("训练 DQN v4 ...")
    env = gym.make('Acrobot-v1')
    agent_v4 = DQNAgentV4ForComparison(
        state_dim=6, action_dim=3,
        learning_rate=1e-3, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.01,
        buffer_capacity=10000, batch_size=64,
        target_update_freq=500, seed=42,
    )
    train_dqn(env, agent_v4, n_episodes=n_eps)
    env.close()

    print("\n训练 DQN v5 (Dueling) ...")
    env = gym.make('Acrobot-v1')
    agent_v5 = DQNAgentV5(
        state_dim=6, action_dim=3,
        learning_rate=1e-3, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.01,
        buffer_capacity=10000, batch_size=64,
        target_update_freq=500, seed=42,
    )
    train_dqn(env, agent_v5, n_episodes=n_eps)
    env.close()

    print("\n" + "=" * 60)
    print("结果对比")
    print("=" * 60)
    for agent, name in [(agent_v4, 'v4'), (agent_v5, 'v5')]:
        last_100 = np.mean(agent.episode_rewards[-100:])
        loss_mean = np.mean(agent.losses[-500:]) if agent.losses else 0
        q_mean = np.mean(agent.q_value_estimates[-1000:]) if agent.q_value_estimates else 0
        print(f"  DQN {name}: reward={last_100:.1f} | loss={loss_mean:.4f} | Q={q_mean:.2f}")

    return agent_v4, agent_v5


# ============================================================
# 实验 5：多 seed 聚合
# ============================================================

def experiment_multi_seed():
    """多 seed 聚合对比，验证 Dueling 的稳定优势"""
    print("\n\n" + "=" * 60)
    print("实验 5: 多 seed 聚合（3 个 seed）")
    print("=" * 60)

    n_eps = 300
    seeds = [42, 142, 242]

    v4_evals = []
    v5_evals = []

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")

        env = gym.make('CartPole-v1')
        agent_v4 = DQNAgentV4ForComparison(
            state_dim=4, action_dim=2,
            learning_rate=1e-3, gamma=0.99,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
            buffer_capacity=10000, batch_size=64,
            target_update_freq=500, seed=seed,
        )
        train_dqn(env, agent_v4, n_episodes=n_eps, verbose=False)
        env.close()

        env = gym.make('CartPole-v1')
        agent_v5 = DQNAgentV5(
            state_dim=4, action_dim=2,
            learning_rate=1e-3, gamma=0.99,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
            buffer_capacity=10000, batch_size=64,
            target_update_freq=500, seed=seed,
        )
        train_dqn(env, agent_v5, n_episodes=n_eps, verbose=False)
        env.close()

        v4_eval, _, _ = evaluate_agent(agent_v4)
        v5_eval, _, _ = evaluate_agent(agent_v5)
        v4_evals.append(v4_eval)
        v5_evals.append(v5_eval)

        print(f"    v4: eval={v4_eval:.1f}")
        print(f"    v5: eval={v5_eval:.1f}")

    print(f"\n  === 聚合结果（{len(seeds)} 个 seed 的均值）===")
    print(f"  v4 评估 reward: {np.mean(v4_evals):.1f} +/- {np.std(v4_evals):.1f}")
    print(f"  v5 评估 reward: {np.mean(v5_evals):.1f} +/- {np.std(v5_evals):.1f}")

    return {'v4_evals': v4_evals, 'v5_evals': v5_evals}


# ============================================================
# 主函数
# ============================================================

if __name__ == '__main__':
    os.makedirs('images', exist_ok=True)

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # --- 实验 1：v4 vs v5 ---
    agent_v4, agent_v5 = experiment_v4_vs_v5()
    visualize_v4_vs_v5(agent_v4, agent_v5,
                       save_path='images/dqn_v5_vs_v4.png')

    # --- 实验 2：V/A 分解分析 ---
    experiment_va_analysis(agent_v5)
    visualize_va_decomposition(agent_v5,
                               save_path='images/dqn_v5_va_decomposition.png')

    # --- 实验 3：Advantage 热力图 ---
    experiment_advantage_landscape(agent_v5)
    visualize_advantage_heatmap(agent_v5,
                                save_path='images/dqn_v5_advantage_heatmap.png')

    # --- 实验 4：Acrobot ---
    agent_v4_acro, agent_v5_acro = experiment_acrobot()
    visualize_acrobot(agent_v4_acro, agent_v5_acro,
                      save_path='images/dqn_v5_acrobot.png')

    # --- 实验 5：多 seed 聚合 ---
    multi_seed_results = experiment_multi_seed()

    # --- 总结 ---
    print("\n\n" + "=" * 60)
    print("总结：Dueling DQN")
    print("=" * 60)
    print()
    print("DQN v5 = DQN v4 + V/A 分解架构")
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  v4: Q_θ(s) = network(s)                                   │")
    print("  │      网络直接输出 Q(s,a) for each a                         │")
    print("  │                                                             │")
    print("  │  v5: Q(s,a) = V(s) + A(s,a) - mean_a'(A(s,a'))            │")
    print("  │      V stream: 状态本身有多好（跟动作无关）                  │")
    print("  │      A stream: 选这个动作比平均好多少                        │")
    print("  │      减 mean: 保证 V 唯一代表状态价值（可辨识性）            │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print("  架构改动：")
    print("    共享特征层 → 分叉为 Value 流和 Advantage 流 → 合并输出 Q")
    print("    算法完全不变（仍用 Double DQN + 经验回放 + 目标网络）")
    print()
    print("  核心洞察：")
    print("    很多状态下，'状态本身好不好' 比 '选哪个动作' 重要")
    print("    V 流可以从所有经验中学习 → 不需要每个 (s,a) 都被访问")
    print("    动作空间越大，优势越明显")
    print()
    print("  DQN 演进路线：")
    print("    v1 = 神经网络近似 Q 函数")
    print("    v2 = + 经验回放（解决数据相关性）")
    print("    v3 = + 目标网络（解决移动目标）")
    print("    v4 = + 选评解耦（解决 max 过估计）← Double DQN")
    print("    v5 = + V/A 分解（更高效的网络架构）← Dueling DQN")
    print()
    print("  下一步可探索：")
    print("    - Prioritized Replay：让重要经验被采样更多")
    print("    - Noisy Networks：用参数噪声替代 ε-greedy 探索")
    print("    - Rainbow：所有改进的集大成者")
