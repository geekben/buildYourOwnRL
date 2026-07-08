"""
dqn_v3_target_network.py - DQN v3：引入目标网络

在 v2 的基础上只增加一个新概念：目标网络（Target Network）

v2 仍未解决的问题：移动目标（Moving Target）
    TD target = r + γ max Q_θ(s', ·)
                       ^^^ 同一个网络！
    每次更新 θ → Q_θ 变 → TD target 变 → 像在追一个会跑的靶子
    经验回放让训练更频繁 → 反而加剧了这个问题（loss 出现几十万的峰值）

目标网络如何解决移动目标？
    引入第二个网络 θ⁻（target_network）专门负责计算 TD target：
        TD target = r + γ max Q_θ⁻(s', ·)
                          ^^^^ 旧版网络，参数固定一段时间
    每隔 C 步把 θ 复制给 θ⁻（hard update）
    或每步用 τ 做加权平均：θ⁻ ← τ θ + (1-τ) θ⁻（soft update）

类比：
    v2 = 射箭，但靶子在动 → 永远追不上
    v3 = 把靶子钉住一段时间，瞄准再换 → 能稳定命中

完成 Nature 2015 论文级别的完整 DQN：
    v1 (神经网络) + v2 (经验回放) + v3 (目标网络) = DQN

学习要点：
1. 移动目标问题的根源与影响
2. 目标网络的双网络架构
3. 两种同步策略：Hard update（每 C 步）vs Soft update（Polyak averaging）
4. v2 vs v3 的训练稳定性对比（这才是 v3 的真正胜场）
5. C / τ 的取舍
"""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# 复用 v1 的 Q 网络（结构完全一致）
from dqn_v1 import QNetwork
# 复用 v2 的经验回放（v3 仍然依赖它）
from dqn_v2_replay_buffer import ReplayBuffer


# ============================================================
# DQN v3 智能体
# ============================================================

class DQNAgentV3:
    """
    DQN v3：DQN v2 + 目标网络

    与 v2 的差异（只有三处）：
    ┌─────────────────┬──────────────────────┬──────────────────────────┐
    │                 │ DQN v2               │ DQN v3                   │
    ├─────────────────┼──────────────────────┼──────────────────────────┤
    │ 网络数量        │ 1 个 Q 网络           │ 2 个（Q 网络 + 目标网络）│
    │ TD target 计算  │ 用 Q 网络             │ 用目标网络               │
    │ 同步策略        │ —                    │ Hard / Soft update       │
    └─────────────────┴──────────────────────┴──────────────────────────┘

    其他一切（网络结构、ε-greedy、经验回放）与 v2 完全相同。

    两种同步策略：
        - Hard update（Nature 2015 经典）：每 C 步硬复制 θ⁻ ← θ
            优点：实现简单，目标完全稳定 C 步
            缺点：每次同步都是"突变"，训练曲线有"心跳"

        - Soft update（Polyak averaging）：每步软更新 θ⁻ ← τ θ + (1-τ) θ⁻
            优点：目标连续平滑变化，无突变
            缺点：每步多一次参数复制，τ 取值更难调

    数学等价关系：τ ≈ 1/C 时两者更新速率相近（但 hard 是阶梯，soft 是斜坡）
    """

    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_capacity=10000, batch_size=64,
                 target_update_strategy='hard', target_update_freq=500, tau=0.005,
                 seed=42):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon: 初始探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
            buffer_capacity: 经验回放缓冲区容量
            batch_size: mini-batch 大小
            target_update_strategy: 'hard' 或 'soft'（v3 新增）
            target_update_freq: hard update 的同步频率 C（每 C 步同步一次）
            tau: soft update 的混合系数 τ（每步 θ⁻ ← τ θ + (1-τ) θ⁻）
            seed: 随机种子
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # v3 新增：目标网络相关参数
        assert target_update_strategy in ('hard', 'soft'), \
            f"target_update_strategy 必须是 'hard' 或 'soft'，得到 {target_update_strategy}"
        self.target_update_strategy = target_update_strategy
        self.target_update_freq = target_update_freq
        self.tau = tau

        # 随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        # Q 网络（与 v2 相同，被反向传播更新）
        self.q_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # 目标网络（v3 新增！）
        # 创建一份独立的网络副本，初始权重与 q_network 相同
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        # 关键：目标网络永远不被反向传播，关闭其梯度计算
        for param in self.target_network.parameters():
            param.requires_grad = False

        # 经验回放（与 v2 完全一致）
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # 训练步数（用于判断 hard update 的同步时机）
        self.train_step = 0

        # 记录训练过程
        self.episode_rewards = []
        self.episode_steps = []
        self.losses = []
        self.epsilon_history = []
        # v3 新增：记录目标网络的同步事件（用于可视化"心跳"）
        self.target_sync_steps = []

    def select_action(self, state):
        """ε-greedy 选择动作（用 q_network，不是 target_network）"""
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """存储一条经验到缓冲区（与 v2 完全一致）"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """
        从缓冲区采样 mini-batch 并更新

        与 v2 的唯一差别：TD target 用 self.target_network 而非 self.q_network

        关键：
            1. 反向传播 ONLY 更新 q_network
            2. target_network 通过 _update_target_network() 同步
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        # 1. 从 buffer 随机采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # 2. numpy → torch
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)

        # 3. 当前 Q 值（用 q_network——这是被训练的网络）
        q_values = self.q_network(states_tensor)
        current_q = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # 4. TD target（用 target_network——这是 v3 的核心改动！）
        with torch.no_grad():
            #                          ↓ v2 用 q_network，v3 用 target_network
            next_q_values = self.target_network(next_states_tensor)
            max_next_q = next_q_values.max(dim=1).values
            td_targets = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q

        # 5. 损失 + 梯度下降（只更新 q_network）
        loss = nn.functional.mse_loss(current_q, td_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())

        # 6. 更新目标网络（v3 新增）
        self.train_step += 1
        self._update_target_network()

    def _update_target_network(self):
        """
        同步目标网络

        Hard update（Nature 2015 经典）：
            每 C 步整体复制：θ⁻ ← θ
            "把靶子钉住 C 步，到点换一次"

        Soft update（Polyak averaging）：
            每步加权平均：θ⁻ ← τ θ + (1-τ) θ⁻
            "靶子缓慢漂向当前网络，永远不一蹴而就"
        """
        if self.target_update_strategy == 'hard':
            if self.train_step % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                self.target_sync_steps.append(self.train_step)

        elif self.target_update_strategy == 'soft':
            # θ⁻ ← τ θ + (1-τ) θ⁻
            with torch.no_grad():
                for target_p, source_p in zip(
                    self.target_network.parameters(),
                    self.q_network.parameters()
                ):
                    target_p.data.mul_(1.0 - self.tau)
                    target_p.data.add_(source_p.data, alpha=self.tau)

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================================
# 训练函数
# ============================================================

def train_dqn_v3(env, agent, n_episodes=300, max_steps=500, verbose=True):
    """
    训练 DQN v3

    与 v2 的训练循环完全一致——目标网络同步是在 agent.update() 内部自动处理的
    """
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
            n_syncs = len(agent.target_sync_steps)
            print(f"  Episode {episode + 1:4d} | "
                  f"Avg Reward: {recent_avg:6.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Loss: {recent_loss:.4f} | "
                  f"Target syncs: {n_syncs}")


def evaluate_agent(agent, n_episodes=20, max_steps=500, seed_base=1000):
    """关闭探索后评估策略（与 v1/v2 共用）"""
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


def visualize_v2_vs_v3(agent_v2, agent_v3, save_path=None):
    """
    实验 1：v2 vs v3 直接对比（200 episodes）

    左：reward 曲线（200 轮内 v2 更快达到高 reward）
    右：loss 曲线（v3 的 loss 量级显著低于 v2）
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    window = 20

    # 左：reward 曲线
    for agent, label, color in [(agent_v2, 'DQN v2 (无目标网络)', '#FF9800'),
                                  (agent_v3, 'DQN v3 (有目标网络)', '#2196F3')]:
        rewards = agent.episode_rewards
        axes[0].plot(rewards, alpha=0.15, color=color)
        smoothed = smooth(rewards, window)
        axes[0].plot(range(window - 1, len(rewards)), smoothed,
                     color=color, linewidth=2, label=f'{label} (滑动平均)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Reward 曲线\n(200 轮内 v2 更快达到高 reward)')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=500, color='green', linestyle='--', alpha=0.3)

    # 右：loss 曲线（log 刻度，因为 v2 的 loss 量级巨大）
    loss_window = 100
    for agent, label, color in [(agent_v2, 'DQN v2', '#FF9800'),
                                  (agent_v3, 'DQN v3', '#2196F3')]:
        losses = agent.losses
        if losses:
            axes[1].plot(losses, alpha=0.1, color=color)
            smoothed = smooth(losses, loss_window)
            axes[1].plot(range(loss_window - 1, len(losses)), smoothed,
                         color=color, linewidth=2, label=f'{label} loss (滑动平均)')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_yscale('log')
    axes[1].set_title('TD Loss 曲线（log 刻度）\n(v3 的 loss 量级显著更低且更稳定)')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, which='both')

    plt.suptitle('实验 1: DQN v2 vs v3——目标网络如何稳定 TD target',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_target_freq(agents, labels, save_path=None):
    """
    实验 2：目标网络更新频率 C 的影响

    单图，多条曲线对比 reward 滑动平均
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#F44336', '#FF9800', '#2196F3', '#9C27B0']
    window = 20

    for idx, (agent, label) in enumerate(zip(agents, labels)):
        rewards = agent.episode_rewards
        color = colors[idx % len(colors)]
        ax.plot(rewards, alpha=0.10, color=color)
        smoothed = smooth(rewards, window)
        ax.plot(range(window - 1, len(rewards)), smoothed,
                color=color, linewidth=2, label=label)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward (滑动平均)')
    ax.set_title('实验 2: 目标网络更新频率 C 的影响\n'
                 'C 太小 → 退化为 v2（移动目标问题再现）；C 太大 → 目标过时')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=500, color='green', linestyle='--', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_hard_vs_soft(agent_hard, agent_soft, save_path=None):
    """
    实验 3：Hard vs Soft update 对比

    左：reward 曲线（应该相近）
    右：训练过程中 |θ⁻ - θ| 的范数（hard 是阶梯式跳变，soft 是平滑漂移）

    简化处理：通过记录 loss 的"心跳"来间接展示同步事件
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    window = 20

    # 左：reward 曲线
    for agent, label, color in [(agent_hard, 'Hard update (C=500)', '#E91E63'),
                                 (agent_soft, 'Soft update (τ=0.005)', '#00BCD4')]:
        rewards = agent.episode_rewards
        axes[0].plot(rewards, alpha=0.15, color=color)
        smoothed = smooth(rewards, window)
        axes[0].plot(range(window - 1, len(rewards)), smoothed,
                     color=color, linewidth=2, label=label)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Reward 曲线\n(两种策略最终性能相近)')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=500, color='green', linestyle='--', alpha=0.3)

    # 右：loss 曲线 + hard update 的同步标记
    loss_window = 50
    for agent, label, color in [(agent_hard, 'Hard update', '#E91E63'),
                                 (agent_soft, 'Soft update', '#00BCD4')]:
        losses = agent.losses
        if losses:
            smoothed = smooth(losses, loss_window)
            axes[1].plot(range(loss_window - 1, len(losses)), smoothed,
                         color=color, linewidth=2, label=label, alpha=0.85)

    # 标出 hard update 的同步时刻（"心跳"）
    for sync_step in agent_hard.target_sync_steps:
        axes[1].axvline(x=sync_step, color='#E91E63', alpha=0.15, linestyle=':')

    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_yscale('log')
    axes[1].set_title('TD Loss 曲线\n(hard 的虚线为同步时刻——能看到"心跳"震荡)')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, which='both')

    plt.suptitle('实验 3: Hard update vs Soft update', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_stability(rewards_runs_v2, rewards_runs_v3, save_path=None):
    """
    实验 4：v2 vs v3 多 seed 稳定性对比

    左：v2 多次运行；右：v3 多次运行
    v3 的"运行间一致性"应该比 v2 显著更好
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    window = 20

    # 左：v2 多次运行
    for i, rewards in enumerate(rewards_runs_v2):
        if len(rewards) >= window:
            smoothed = smooth(rewards, window)
            axes[0].plot(range(window - 1, len(rewards)), smoothed,
                         alpha=0.8, label=f'Run {i + 1}', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward (滑动平均)')
    axes[0].set_title('DQN v2（无目标网络）')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=500, color='green', linestyle='--', alpha=0.3)

    # 右：v3 多次运行
    for i, rewards in enumerate(rewards_runs_v3):
        if len(rewards) >= window:
            smoothed = smooth(rewards, window)
            axes[1].plot(range(window - 1, len(rewards)), smoothed,
                         alpha=0.8, label=f'Run {i + 1}', linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Total Reward (滑动平均)')
    axes[1].set_title('DQN v3（有目标网络）')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=500, color='green', linestyle='--', alpha=0.3)

    # y 轴对齐（用相同范围便于对比）
    all_rewards = [r for runs in [rewards_runs_v2, rewards_runs_v3] for run in runs for r in run]
    if all_rewards:
        y_max = max(all_rewards) * 1.05
        axes[0].set_ylim(0, y_max)
        axes[1].set_ylim(0, y_max)

    plt.suptitle('实验 4: 多 seed 稳定性对比（150 episodes）', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================
# 实验 1：v2 vs v3 直接对比
# ============================================================

def experiment_v2_vs_v3():
    """
    核心实验：v2（仅经验回放）vs v3（经验回放 + 目标网络）

    预期：
    - v3 的 loss 量级显著低于 v2（目标网络稳定了 TD target）
    - v3 的 reward 曲线更平滑
    - v3 评估性能略高于 v2
    """
    print("=" * 60)
    print("实验 1: DQN v2 vs v3 直接对比")
    print("=" * 60)
    print()
    print("v2: 经验回放（loss 巨大波动 → 移动目标问题加剧）")
    print("v3: 经验回放 + 目标网络（loss 大幅下降 → 训练稳定）")
    print()

    from dqn_v2_replay_buffer import DQNAgentV2, train_dqn_v2

    # v2
    print("训练 DQN v2 ...")
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

    # v3（hard update，Nature 2015 经典配置）
    print("\n训练 DQN v3（hard update, C=500） ...")
    env_v3 = gym.make('CartPole-v1')
    agent_v3 = DQNAgentV3(
        state_dim=4, action_dim=2,
        learning_rate=1e-3, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
        buffer_capacity=10000, batch_size=64,
        target_update_strategy='hard', target_update_freq=500,
        seed=42,
    )
    train_dqn_v3(env_v3, agent_v3, n_episodes=200)
    env_v3.close()

    # 评估
    print("\n" + "=" * 60)
    print("策略评估（关闭探索，20 次运行）")
    print("=" * 60)
    for agent, name in [(agent_v2, 'v2'), (agent_v3, 'v3')]:
        mean, std, _ = evaluate_agent(agent)
        print(f"  DQN {name}: 平均奖励 = {mean:.1f} ± {std:.1f}")

    # loss 量级对比
    print()
    if agent_v2.losses and agent_v3.losses:
        v2_loss_mean = np.mean(agent_v2.losses[-500:])
        v3_loss_mean = np.mean(agent_v3.losses[-500:])
        print(f"  v2 loss 末段平均: {v2_loss_mean:.2f}")
        print(f"  v3 loss 末段平均: {v3_loss_mean:.2f}")
        print(f"  比值: v2 / v3 = {v2_loss_mean / max(v3_loss_mean, 1e-6):.1f}x")

    return agent_v2, agent_v3


# ============================================================
# 实验 2：目标网络更新频率 C 的影响
# ============================================================

def experiment_target_freq():
    """
    探索 hard update 频率 C 的影响

    C=1：每步同步，等价于 v2（移动目标问题再现）
    C=100：频繁同步，目标变化较快但不至于像 v2 那样
    C=500：经典配置（Nature 2015 默认）
    C=2000：长时间不同步，目标"过时"
    """
    print("\n\n" + "=" * 60)
    print("实验 2: 目标网络更新频率 C 的影响")
    print("=" * 60)

    configs = [
        {'C': 1,    'label': 'C=1 (退化为 v2)'},
        {'C': 100,  'label': 'C=100 (频繁同步)'},
        {'C': 500,  'label': 'C=500 (经典配置)'},
        {'C': 2000, 'label': 'C=2000 (目标过时)'},
    ]

    agents = []
    labels = []
    for config in configs:
        print(f"\n  训练 {config['label']} ...")
        env = gym.make('CartPole-v1')
        agent = DQNAgentV3(
            state_dim=4, action_dim=2,
            learning_rate=1e-3, gamma=0.99,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
            buffer_capacity=10000, batch_size=64,
            target_update_strategy='hard', target_update_freq=config['C'],
            seed=42,
        )
        train_dqn_v3(env, agent, n_episodes=200, verbose=False)
        agents.append(agent)
        labels.append(config['label'])
        env.close()

        avg_last_50 = np.mean(agent.episode_rewards[-50:])
        loss_mean = np.mean(agent.losses[-500:]) if agent.losses else 0
        print(f"    最后 50 轮平均: {avg_last_50:.1f} | loss 末段平均: {loss_mean:.2f}")

    print()
    print("分析：")
    print("  - C=1：每步都同步，目标网络等于 q_network，退化为 v2")
    print("  - C=100：同步太频繁，目标稳定性不够")
    print("  - C=500：经典默认值，平衡了稳定性和时效性")
    print("  - C=2000：长期不同步，目标过时，学习效率受影响")

    return agents, labels


# ============================================================
# 实验 3：Hard vs Soft update
# ============================================================

def experiment_hard_vs_soft():
    """
    Hard update（每 C 步整体复制）vs Soft update（每步 τ 加权）

    数学等价：τ ≈ 1/C 时两者更新速率相近，但
        Hard：靶子 C 步不动，到点突变
        Soft：靶子每步缓慢漂向新位置，永远不突变
    """
    print("\n\n" + "=" * 60)
    print("实验 3: Hard update vs Soft update")
    print("=" * 60)

    # Hard update
    print("\n  训练 Hard update (C=500) ...")
    env = gym.make('CartPole-v1')
    agent_hard = DQNAgentV3(
        state_dim=4, action_dim=2,
        learning_rate=1e-3, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
        buffer_capacity=10000, batch_size=64,
        target_update_strategy='hard', target_update_freq=500,
        seed=42,
    )
    train_dqn_v3(env, agent_hard, n_episodes=200, verbose=False)
    env.close()
    avg_hard = np.mean(agent_hard.episode_rewards[-50:])
    print(f"    最后 50 轮平均: {avg_hard:.1f} | 同步次数: {len(agent_hard.target_sync_steps)}")

    # Soft update
    print("\n  训练 Soft update (τ=0.005) ...")
    env = gym.make('CartPole-v1')
    agent_soft = DQNAgentV3(
        state_dim=4, action_dim=2,
        learning_rate=1e-3, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
        buffer_capacity=10000, batch_size=64,
        target_update_strategy='soft', tau=0.005,
        seed=42,
    )
    train_dqn_v3(env, agent_soft, n_episodes=200, verbose=False)
    env.close()
    avg_soft = np.mean(agent_soft.episode_rewards[-50:])
    print(f"    最后 50 轮平均: {avg_soft:.1f}")

    print()
    print("分析：")
    print("  - 数学上，τ ≈ 1/C 时两者更新速率相当（这里 τ=0.005 对应 ~C=200）")
    print("  - Hard update：靶子突变，loss 在同步时有'心跳'式跳变")
    print("  - Soft update：靶子平滑漂移，loss 曲线更连续")
    print("  - 最终性能往往相近，soft 在某些任务上略优（如连续控制 DDPG）")

    return agent_hard, agent_soft


# ============================================================
# 实验 4：多 seed 稳定性对比
# ============================================================

def experiment_stability():
    """
    多次独立运行（不同 seed），对比 v2 和 v3 的稳定性

    v2 的笔记里提到：v2 的真正收益是性能而非稳定性
    v3 的真正收益才是稳定性——本实验验证这一点
    """
    print("\n\n" + "=" * 60)
    print("实验 4: 多 seed 稳定性对比（3 次独立运行）")
    print("=" * 60)

    from dqn_v2_replay_buffer import DQNAgentV2, train_dqn_v2

    rewards_runs_v2 = []
    rewards_runs_v3 = []

    for run in range(3):
        print(f"\n  --- Run {run + 1}/3 ---")
        seed = run * 100 + 42

        # v2
        print(f"    训练 v2 (seed={seed}) ...")
        env = gym.make('CartPole-v1')
        agent_v2 = DQNAgentV2(
            state_dim=4, action_dim=2,
            learning_rate=1e-3, gamma=0.99,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
            buffer_capacity=10000, batch_size=64,
            seed=seed,
        )
        train_dqn_v2(env, agent_v2, n_episodes=150)
        rewards_runs_v2.append(agent_v2.episode_rewards)
        env.close()
        print(f"      v2 最后 50 轮平均: {np.mean(agent_v2.episode_rewards[-50:]):.1f}")

        # v3
        print(f"    训练 v3 (seed={seed}) ...")
        env = gym.make('CartPole-v1')
        agent_v3 = DQNAgentV3(
            state_dim=4, action_dim=2,
            learning_rate=1e-3, gamma=0.99,
            epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
            buffer_capacity=10000, batch_size=64,
            target_update_strategy='hard', target_update_freq=500,
            seed=seed,
        )
        train_dqn_v3(env, agent_v3, n_episodes=150, verbose=False)
        rewards_runs_v3.append(agent_v3.episode_rewards)
        env.close()
        print(f"      v3 最后 50 轮平均: {np.mean(agent_v3.episode_rewards[-50:]):.1f}")

    # 稳定性指标
    v2_final_avgs = [np.mean(r[-50:]) for r in rewards_runs_v2]
    v3_final_avgs = [np.mean(r[-50:]) for r in rewards_runs_v3]

    print(f"\n  v2 各运行最终平均: {[f'{x:.1f}' for x in v2_final_avgs]}")
    print(f"  v3 各运行最终平均: {[f'{x:.1f}' for x in v3_final_avgs]}")
    print(f"  v2 运行间标准差: {np.std(v2_final_avgs):.1f}")
    print(f"  v3 运行间标准差: {np.std(v3_final_avgs):.1f}")
    print(f"  → v3 的运行间标准差应当显著低于 v2（稳定性的真正胜场）")

    return rewards_runs_v2, rewards_runs_v3


# ============================================================
# 实验 5：CartPole 长训练（1000 episodes）
# ============================================================

def visualize_long_run(agent_v2, agent_v3, save_path=None):
    """v2 vs v3 在 1000 episodes 下的对比"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    window = 50

    for agent, label, color in [(agent_v2, 'DQN v2', '#FF9800'),
                                  (agent_v3, 'DQN v3 (C=500)', '#2196F3')]:
        rewards = agent.episode_rewards
        axes[0].plot(rewards, alpha=0.1, color=color)
        smoothed = smooth(rewards, window)
        axes[0].plot(range(window - 1, len(rewards)), smoothed,
                     color=color, linewidth=2, label=f'{label} (滑动平均)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Reward 曲线（1000 episodes）')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=500, color='green', linestyle='--', alpha=0.3)

    loss_window = 200
    for agent, label, color in [(agent_v2, 'DQN v2', '#FF9800'),
                                  (agent_v3, 'DQN v3', '#2196F3')]:
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

    plt.suptitle('实验 5: CartPole 长训练（1000 episodes）——v3 是否追平 v2？',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def experiment_long_run():
    """
    CartPole 上跑 1000 episodes，验证 v3 是否在足够长的训练后追平 v2
    """
    print("\n\n" + "=" * 60)
    print("实验 5: CartPole 长训练（1000 episodes）")
    print("=" * 60)
    print()
    print("假说：v3 的'慢但稳'在足够多的 episode 后会追平甚至超过 v2")
    print()

    from dqn_v2_replay_buffer import DQNAgentV2, train_dqn_v2

    print("训练 DQN v2（1000 episodes）...")
    env = gym.make('CartPole-v1')
    agent_v2 = DQNAgentV2(
        state_dim=4, action_dim=2,
        learning_rate=1e-3, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
        buffer_capacity=10000, batch_size=64,
        seed=42,
    )
    train_dqn_v2(env, agent_v2, n_episodes=1000)
    env.close()

    print("\n训练 DQN v3（1000 episodes, C=500）...")
    env = gym.make('CartPole-v1')
    agent_v3 = DQNAgentV3(
        state_dim=4, action_dim=2,
        learning_rate=1e-3, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
        buffer_capacity=10000, batch_size=64,
        target_update_strategy='hard', target_update_freq=500,
        seed=42,
    )
    train_dqn_v3(env, agent_v3, n_episodes=1000)
    env.close()

    print("\n" + "=" * 60)
    print("策略评估（关闭探索，20 次运行）")
    print("=" * 60)
    for agent, name in [(agent_v2, 'v2'), (agent_v3, 'v3')]:
        mean, std, _ = evaluate_agent(agent)
        print(f"  DQN {name}: 平均奖励 = {mean:.1f} ± {std:.1f}")

    for window_name, start in [('前 200 轮', (0, 200)),
                                 ('中 400-600 轮', (400, 600)),
                                 ('后 200 轮', (-200, None))]:
        s, e = start
        v2_avg = np.mean(agent_v2.episode_rewards[s:e])
        v3_avg = np.mean(agent_v3.episode_rewards[s:e])
        print(f"  {window_name}: v2={v2_avg:.1f}, v3={v3_avg:.1f}")

    return agent_v2, agent_v3


# ============================================================
# 实验 6：Acrobot-v1（更难的任务）
# ============================================================

def visualize_acrobot(agent_v2, agent_v3, save_path=None):
    """Acrobot 上 v2 vs v3 对比"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    window = 30

    for agent, label, color in [(agent_v2, 'DQN v2', '#FF9800'),
                                  (agent_v3, 'DQN v3 (C=500)', '#2196F3')]:
        rewards = agent.episode_rewards
        axes[0].plot(rewards, alpha=0.1, color=color)
        smoothed = smooth(rewards, window)
        axes[0].plot(range(window - 1, len(rewards)), smoothed,
                     color=color, linewidth=2, label=f'{label} (滑动平均)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Acrobot Reward 曲线\n(reward 越接近 0 越好)')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    loss_window = 100
    for agent, label, color in [(agent_v2, 'DQN v2', '#FF9800'),
                                  (agent_v3, 'DQN v3', '#2196F3')]:
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

    plt.suptitle('实验 6: Acrobot-v1——更难的任务上 v2 vs v3',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def experiment_acrobot():
    """
    在 Acrobot-v1 上对比 v2 和 v3

    Acrobot 比 CartPole 更难：
    - 6 维状态（2 个关节的 cos/sin + 角速度）
    - 3 个动作（扭矩 -1, 0, +1）
    - 奖励：每步 -1，越快摆到顶部奖励越高（越接近 0）
    - 最大 500 步
    """
    print("\n\n" + "=" * 60)
    print("实验 6: Acrobot-v1（更难的任务）上 v2 vs v3")
    print("=" * 60)
    print()
    print("Acrobot: 6 维状态, 3 个动作, 比 CartPole 更难")
    print("奖励每步 -1，越快到达目标越好（reward 越接近 0 越好）")
    print()

    from dqn_v2_replay_buffer import DQNAgentV2, train_dqn_v2

    n_eps = 500

    print(f"训练 DQN v2（{n_eps} episodes）...")
    env = gym.make('Acrobot-v1')
    agent_v2 = DQNAgentV2(
        state_dim=6, action_dim=3,
        learning_rate=1e-3, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.01,
        buffer_capacity=10000, batch_size=64,
        seed=42,
    )
    train_dqn_v2(env, agent_v2, n_episodes=n_eps)
    env.close()

    print(f"\n训练 DQN v3（{n_eps} episodes, C=500）...")
    env = gym.make('Acrobot-v1')
    agent_v3 = DQNAgentV3(
        state_dim=6, action_dim=3,
        learning_rate=1e-3, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.01,
        buffer_capacity=10000, batch_size=64,
        target_update_strategy='hard', target_update_freq=500,
        seed=42,
    )
    train_dqn_v3(env, agent_v3, n_episodes=n_eps)
    env.close()

    print("\n" + "=" * 60)
    print("结果对比")
    print("=" * 60)

    for agent, name in [(agent_v2, 'v2'), (agent_v3, 'v3')]:
        last_100 = np.mean(agent.episode_rewards[-100:])
        loss_mean = np.mean(agent.losses[-500:]) if agent.losses else 0
        print(f"  DQN {name}: 最后 100 轮平均 reward = {last_100:.1f} | loss 末段平均 = {loss_mean:.2f}")

    if agent_v2.losses and agent_v3.losses:
        v2_loss = np.mean(agent_v2.losses[-500:])
        v3_loss = np.mean(agent_v3.losses[-500:])
        print(f"  Loss 比值: v2 / v3 = {v2_loss / max(v3_loss, 1e-6):.1f}x")

    return agent_v2, agent_v3


# ============================================================
# 主函数
# ============================================================

if __name__ == '__main__':
    os.makedirs('images', exist_ok=True)

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # --- 实验 1：v2 vs v3 ---
    agent_v2, agent_v3 = experiment_v2_vs_v3()
    visualize_v2_vs_v3(agent_v2, agent_v3,
                        save_path='images/dqn_v3_vs_v2.png')

    # --- 实验 2：C 频率 ---
    agents_freq, labels_freq = experiment_target_freq()
    visualize_target_freq(agents_freq, labels_freq,
                           save_path='images/dqn_v3_target_freq.png')

    # --- 实验 3：Hard vs Soft ---
    agent_hard, agent_soft = experiment_hard_vs_soft()
    visualize_hard_vs_soft(agent_hard, agent_soft,
                            save_path='images/dqn_v3_hard_vs_soft.png')

    # --- 实验 4：多 seed 稳定性 ---
    rewards_v2, rewards_v3 = experiment_stability()
    visualize_stability(rewards_v2, rewards_v3,
                         save_path='images/dqn_v3_stability.png')

    # --- 实验 5：CartPole 长训练 ---
    agent_v2_long, agent_v3_long = experiment_long_run()
    visualize_long_run(agent_v2_long, agent_v3_long,
                        save_path='images/dqn_v3_long_run.png')

    # --- 实验 6：Acrobot ---
    agent_v2_acro, agent_v3_acro = experiment_acrobot()
    visualize_acrobot(agent_v2_acro, agent_v3_acro,
                       save_path='images/dqn_v3_acrobot.png')

    # --- 总结 ---
    print("\n\n" + "=" * 60)
    print("总结：完整的 Nature 2015 DQN")
    print("=" * 60)
    print()
    print("DQN v3 = DQN v2 + 目标网络")
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  v2: TD target = r + γ max Q_θ(s', ·)                       │")
    print("  │       同一个网络 → 移动目标 → loss 巨大波动                  │")
    print("  │                                                             │")
    print("  │  v3: TD target = r + γ max Q_θ⁻(s', ·)                      │")
    print("  │       独立的目标网络 θ⁻ → 靶子稳定 → loss 平稳                │")
    print("  │                                                             │")
    print("  │  同步策略：                                                  │")
    print("  │    - Hard update：每 C 步 θ⁻ ← θ                            │")
    print("  │    - Soft update：每步 θ⁻ ← τ θ + (1-τ) θ⁻                  │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print("  ✅ v3 的真正收益：训练稳定性（loss 量级↓、运行间方差↓）")
    print()
    print("  完整 DQN（Nature 2015）:")
    print("    1️⃣ v1 = 神经网络近似 Q 函数（替代 Q 表）")
    print("    2️⃣ v2 = 经验回放（解决数据相关性）")
    print("    3️⃣ v3 = 目标网络（解决移动目标）")
    print()
    print("  下一步可探索的方向：")
    print("    - Double DQN：解决 max 操作带来的过估计偏差")
    print("    - Dueling DQN：分离 V(s) 和 A(s, a)")
    print("    - Prioritized Replay：让重要的经验被采样更多")
    print("    - Rainbow：上述改进的集大成者")
