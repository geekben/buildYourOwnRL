"""
dqn_v1.py - DQN v1：从 Q 表到神经网络

这是从表格方法到深度强化学习的第一步：
    Q-Learning（表格）：Q 表 → Q[s, a]，查表得到 Q 值
    DQN v1（神经网络）：Q_θ(s) → 输入状态 s，输出所有动作的 Q 值

为什么需要这一步？
    表格方法的 Q 表大小 = |S| × |A|
    - GridWorld 4×4 = 16 个状态 → Q 表 16×4 = 64 个值 ✅
    - CartPole：状态是 [位置, 速度, 角度, 角速度]，连续值 → Q 表无穷大 ❌
    - Atari：210×160×3 像素 → 状态数 ≈ 256^(210×160×3) → 不可能建表 ❌

解决方案：用神经网络近似 Q 函数
    Q_θ(s) ≈ Q*(s, ·)
    输入：状态向量 s（如 CartPole 的 4 维向量）
    输出：每个动作的 Q 值（如 CartPole 的 [Q(s,左), Q(s,右)]）

v1 有意不加经验回放和目标网络，暴露两个问题：
    1. 数据相关性：连续的 (s, a, r, s') 高度相关 → 梯度方向偏
    2. 移动目标：TD target 用的是同一个网络 → 自己追自己

这些问题将在 v2（经验回放）和 v3（目标网络）中逐步解决。

学习要点：
1. 为什么 Q 表无法处理连续/大状态空间
2. 神经网络如何替代 Q 表
3. 损失函数：MSE(Q_θ(s,a), r + γ max Q_θ(s',·))
4. v1 的训练不稳定现象及原因分析
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


# ============================================================
# Q 网络：用神经网络替代 Q 表
# ============================================================

class QNetwork(nn.Module):
    """
    Q 网络：输入状态，输出所有动作的 Q 值

    对比 Q 表：
        Q 表：Q[state_idx, action_idx] → 一个数
        Q 网络：Q_θ(state_vector) → [Q(s,a0), Q(s,a1), ..., Q(s,an)]

    为什么输出所有动作的 Q 值而不是单个？
        如果输入 (s, a) 输出 Q(s,a)，选动作时需要对每个 a 前向传播一次
        输入 s 输出所有 Q(s, ·)，一次前向传播就能比较所有动作
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Args:
            state_dim: 状态向量维度（如 CartPole 是 4）
            action_dim: 动作数量（如 CartPole 是 2）
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            # 注意：最后一层没有激活函数！
            # Q 值可以是任意实数（正、负、零都有可能）
        )

    def forward(self, state):
        """
        前向传播：状态 → Q 值

        Args:
            state: 状态张量，shape = (batch_size, state_dim) 或 (state_dim,)

        Returns:
            Q 值张量，shape = (batch_size, action_dim) 或 (action_dim,)
        """
        return self.network(state)


# ============================================================
# DQN v1 智能体
# ============================================================

class DQNAgentV1:
    """
    DQN v1：最朴素的深度 Q 网络

    与 Q-Learning 的对比：
    ┌─────────────┬──────────────────────────┬──────────────────────────┐
    │             │ Q-Learning（表格）        │ DQN v1（神经网络）        │
    ├─────────────┼──────────────────────────┼──────────────────────────┤
    │ Q 函数存储  │ Q 表 (numpy array)       │ 神经网络权重 θ            │
    │ 查询 Q 值   │ Q[s, a]                  │ Q_θ(s)[a]                │
    │ 更新方式    │ Q[s,a] += α * td_error   │ 梯度下降最小化 MSE loss  │
    │ 泛化能力    │ 无（每个 (s,a) 独立）     │ 有（相似状态共享权重）    │
    │ 状态空间    │ 离散、小                  │ 连续、大                  │
    └─────────────┴──────────────────────────┴──────────────────────────┘

    v1 的已知问题（有意不解决，留给 v2/v3）：
    1. 没有经验回放 → 训练数据高度相关，梯度方向偏
    2. 没有目标网络 → TD target 不稳定，"自己追自己"
    """

    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, seed=42):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            learning_rate: 神经网络学习率（替代 Q-Learning 的 α）
            gamma: 折扣因子
            epsilon: 初始探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
            seed: 随机种子
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # 设置随机种子
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        # Q 网络（替代 Q 表）
        self.q_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # 记录训练过程
        self.episode_rewards = []
        self.episode_steps = []
        self.losses = []
        self.epsilon_history = []

    def select_action(self, state):
        """
        ε-greedy 策略选择动作

        与 Q-Learning 的区别：
            Q-Learning: action = argmax Q[state_idx]
            DQN:        action = argmax Q_θ(state_tensor)

        需要先把 numpy 状态转成 torch 张量
        """
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.action_dim)

        # 贪心：用 Q 网络计算所有动作的 Q 值，选最大的
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # (state_dim,) → (1, state_dim)
            q_values = self.q_network(state_tensor)               # (1, action_dim)
            return q_values.argmax(dim=1).item()                  # 标量

    def update(self, state, action, reward, next_state, done):
        """
        DQN 的单步更新（在线学习，没有经验回放）

        对比 Q-Learning 的更新：
            Q-Learning: Q[s,a] += α * (r + γ max Q[s'] - Q[s,a])
            DQN:        θ -= lr * ∇_θ L(θ)
                        L(θ) = (Q_θ(s)[a] - (r + γ max Q_θ(s')[·]))²

        本质相同：都是让 Q(s,a) 逼近 TD target = r + γ max Q(s', ·)
        区别在于：
            - Q-Learning 直接修改 Q 表的一个值
            - DQN 用梯度下降修改整个网络的权重
              → 修改一个 (s,a) 的 Q 值，会同时影响其他相似状态的 Q 值（泛化！）
        """
        # numpy → torch
        state_tensor = torch.FloatTensor(state).unsqueeze(0)       # (1, state_dim)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward])
        done_tensor = torch.FloatTensor([float(done)])

        # 1. 计算当前 Q 值：Q_θ(s)[a]
        q_values = self.q_network(state_tensor)            # (1, action_dim)
        current_q = q_values[0, action]                     # 标量：选出执行的动作对应的 Q 值

        # 2. 计算 TD target：r + γ max_a' Q_θ(s', a')
        #    注意：这里用的是同一个网络！（v3 会引入单独的目标网络）
        with torch.no_grad():
            next_q_values = self.q_network(next_state_tensor)  # (1, action_dim)
            max_next_q = next_q_values.max(dim=1).values       # 标量
            td_target = reward_tensor + (1 - done_tensor) * self.gamma * max_next_q

        # 3. 计算损失：MSE(当前Q, TD target)
        loss = nn.functional.mse_loss(current_q.unsqueeze(0), td_target)

        # 4. 梯度下降
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

def train_dqn_v1(env, agent, n_episodes=500, max_steps=500):
    """
    训练 DQN v1（在线学习，每步立即更新）

    训练循环与 Q-Learning 完全相同：
        选动作 → 执行 → 观察 (r, s') → 更新 → 重复

    唯一的区别是 agent.update() 内部用梯度下降而不是直接修改 Q 表
    """
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
            recent_avg = np.mean(recent_rewards)
            print(f"  Episode {episode + 1:4d} | "
                  f"Avg Reward: {recent_avg:6.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Loss: {np.mean(agent.losses[-50:]):.4f}")


# ============================================================
# 可视化函数
# ============================================================

def visualize_learning_curve(agents, labels, title="DQN v1 Learning Curve",
                             window=20, save_path=None):
    """绘制学习曲线（支持多个 agent 对比）"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']

    # 上图：奖励曲线
    for idx, (agent, label) in enumerate(zip(agents, labels)):
        rewards = agent.episode_rewards
        color = colors[idx % len(colors)]
        axes[0].plot(rewards, alpha=0.2, color=color)
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
            axes[0].plot(range(window - 1, len(rewards)), smoothed,
                         color=color, linewidth=2, label=f'{label} (滑动平均)')
        else:
            axes[0].plot(rewards, color=color, linewidth=2, label=label)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 下图：损失曲线
    for idx, (agent, label) in enumerate(zip(agents, labels)):
        if hasattr(agent, 'losses') and agent.losses:
            losses = agent.losses
            color = colors[idx % len(colors)]
            # 按 episode 聚合 loss
            episode_losses = []
            loss_idx = 0
            for steps in agent.episode_steps:
                if loss_idx + steps <= len(losses):
                    episode_losses.append(np.mean(losses[loss_idx:loss_idx + steps]))
                loss_idx += steps
            if episode_losses:
                axes[1].plot(episode_losses, alpha=0.2, color=color)
                if len(episode_losses) >= window:
                    smoothed = np.convolve(episode_losses,
                                           np.ones(window) / window, mode='valid')
                    axes[1].plot(range(window - 1, len(episode_losses)), smoothed,
                                 color=color, linewidth=2, label=label)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_q_landscape(agent, env_name='CartPole-v1', save_path=None):
    """
    可视化 Q 网络在不同状态下的输出

    固定 [速度=0, 角速度=0]，扫描 [位置, 角度] 平面
    展示 Q 网络学到的价值地形图
    """
    positions = np.linspace(-2.4, 2.4, 50)
    angles = np.linspace(-0.21, 0.21, 50)

    q_left = np.zeros((len(angles), len(positions)))
    q_right = np.zeros((len(angles), len(positions)))

    agent.q_network.eval()
    with torch.no_grad():
        for i, angle in enumerate(angles):
            for j, pos in enumerate(positions):
                state = torch.FloatTensor([[pos, 0.0, angle, 0.0]])
                q_values = agent.q_network(state)
                q_left[i, j] = q_values[0, 0].item()
                q_right[i, j] = q_values[0, 1].item()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Q(s, 左)
    im0 = axes[0].imshow(q_left, extent=[positions[0], positions[-1],
                                          angles[-1], angles[0]],
                          aspect='auto', cmap='RdYlGn')
    axes[0].set_xlabel('位置 (position)')
    axes[0].set_ylabel('角度 (angle)')
    axes[0].set_title('Q(s, 左)')
    plt.colorbar(im0, ax=axes[0])

    # Q(s, 右)
    im1 = axes[1].imshow(q_right, extent=[positions[0], positions[-1],
                                           angles[-1], angles[0]],
                          aspect='auto', cmap='RdYlGn')
    axes[1].set_xlabel('位置 (position)')
    axes[1].set_ylabel('角度 (angle)')
    axes[1].set_title('Q(s, 右)')
    plt.colorbar(im1, ax=axes[1])

    # 最优动作：argmax(Q(s, 左), Q(s, 右))
    best_action = (q_right > q_left).astype(float)
    im2 = axes[2].imshow(best_action, extent=[positions[0], positions[-1],
                                               angles[-1], angles[0]],
                          aspect='auto', cmap='coolwarm')
    axes[2].set_xlabel('位置 (position)')
    axes[2].set_ylabel('角度 (angle)')
    axes[2].set_title('最优动作 (蓝=左, 红=右)')
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle('DQN v1 学到的 Q 值地形图\n(固定速度=0, 角速度=0)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_instability(rewards_runs, title="DQN v1 训练不稳定性", save_path=None):
    """可视化多次运行的不稳定性"""
    fig, ax = plt.subplots(figsize=(12, 6))
    window = 20

    for i, rewards in enumerate(rewards_runs):
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(rewards)), smoothed,
                    alpha=0.6, label=f'Run {i + 1}')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward (滑动平均)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=475, color='green', linestyle='--', alpha=0.5, label='目标 (475)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================
# 实验一：为什么 Q 表不行？
# ============================================================

def experiment_qtable_fails():
    """
    展示 Q 表在连续状态空间中的困境

    CartPole 的状态是 4 维连续向量：[位置, 速度, 角度, 角速度]
    - 如果离散化为每维 10 个桶 → 10^4 = 10000 个状态 → 还能勉强维护 Q 表
    - 如果每维 100 个桶 → 100^4 = 1 亿个状态 → Q 表爆炸
    - 而且离散化太粗会丢失信息，太细又稀疏

    这里用粗糙的离散化 + Q-Learning 来演示：
    - 精度不够 → 学不好
    - 很多 (s, a) 从未被访问过 → Q 值为 0 → 决策盲目
    """
    print("=" * 60)
    print("实验一：Q 表在 CartPole 中的困境")
    print("=" * 60)
    print()
    print("CartPole 状态：[位置, 速度, 角度, 角速度]")
    print("  每个维度都是连续值，无法直接建 Q 表")
    print("  → 必须离散化，但离散化会丢失精度")
    print()

    env = gym.make('CartPole-v1')

    # 粗糙的离散化参数
    n_bins = 8  # 每个维度离散为 8 个桶
    state_bounds = [
        (-2.4, 2.4),      # 位置
        (-3.0, 3.0),       # 速度（实际无界，截断）
        (-0.21, 0.21),     # 角度（约 ±12°）
        (-3.0, 3.0),       # 角速度（实际无界，截断）
    ]
    total_states = n_bins ** 4  # 8^4 = 4096 个离散状态

    print(f"  离散化：每维 {n_bins} 个桶 → {total_states} 个状态")
    print(f"  Q 表大小：{total_states} × 2 = {total_states * 2} 个值")
    print()

    def discretize_state(state):
        """将连续状态离散化为索引"""
        indices = []
        for i, (low, high) in enumerate(state_bounds):
            val = np.clip(state[i], low, high)
            bin_idx = int((val - low) / (high - low) * (n_bins - 1))
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            indices.append(bin_idx)
        state_idx = 0
        for idx in indices:
            state_idx = state_idx * n_bins + idx
        return state_idx

    # 用 Q-Learning + 离散化训练
    q_table = np.zeros((total_states, 2))
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    rng = np.random.default_rng(42)
    rewards_history = []
    visited_states = set()

    n_episodes = 200
    for episode in range(n_episodes):
        state, _ = env.reset(seed=episode)
        state_idx = discretize_state(state)
        total_reward = 0

        for step in range(500):
            visited_states.add(state_idx)

            if rng.random() < epsilon:
                action = rng.integers(0, 2)
            else:
                action = np.argmax(q_table[state_idx])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_idx = discretize_state(next_state)

            if done:
                td_target = reward
            else:
                td_target = reward + gamma * np.max(q_table[next_state_idx])

            q_table[state_idx, action] += alpha * (td_target - q_table[state_idx, action])

            state = next_state
            state_idx = next_state_idx
            total_reward += reward

            if done:
                break

        rewards_history.append(total_reward)
        epsilon = max(0.01, epsilon * 0.995)

    env.close()

    avg_last_50 = np.mean(rewards_history[-50:])
    coverage = len(visited_states) / total_states * 100

    print(f"训练 {n_episodes} 轮后：")
    print(f"  最后 50 轮平均奖励：{avg_last_50:.1f}（满分 500）")
    print(f"  状态覆盖率：{coverage:.1f}%（{len(visited_states)}/{total_states}）")
    print()
    print("问题分析：")
    print(f"  1. 状态覆盖率仅 {coverage:.1f}%，大量 (s,a) 从未被访问")
    print("  2. 离散化太粗，丢失了状态之间的连续关系")
    print("  3. 增加精度（如每维 100 桶）→ 状态数爆炸到 1 亿")
    print("  4. 即使某状态被访问过，相邻的连续状态也无法利用这个经验")
    print()
    print("→ 我们需要一种能从连续状态中泛化学习的方法 → 神经网络！")
    print()

    return rewards_history


# ============================================================
# 实验二：DQN v1 在 CartPole 上的表现
# ============================================================

def experiment_dqn_v1():
    """
    用 DQN v1 训练 CartPole

    CartPole 任务：
    - 一根杆子通过铰链连接在小车上
    - 目标：通过左右移动小车，让杆子保持直立
    - 状态：[位置, 速度, 角度, 角速度]（4 维连续向量）
    - 动作：0（向左推）或 1（向右推）
    - 奖励：每步 +1（杆子没倒就得分）
    - 终止：杆子倾斜超过 ±12°，或小车超出边界
    - 满分：500（坚持 500 步不倒）
    """
    print("\n" + "=" * 60)
    print("实验二：DQN v1 训练 CartPole")
    print("=" * 60)
    print()
    print("CartPole 任务：让杆子保持直立")
    print("  状态：[位置, 速度, 角度, 角速度]（4 维连续向量）")
    print("  动作：向左推 / 向右推")
    print("  目标：坚持 500 步不倒 → 满分 500")
    print()

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n              # 2

    print(f"  状态维度：{state_dim}")
    print(f"  动作数量：{action_dim}")
    print()

    agent = DQNAgentV1(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        seed=42,
    )

    print("训练 DQN v1（在线学习，无经验回放，无目标网络）...")
    train_dqn_v1(env, agent, n_episodes=300, max_steps=500)

    env.close()

    # 评估
    print("\n策略评估（关闭探索，20 次运行）:")
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

    print(f"  平均奖励：{np.mean(eval_rewards):.1f} ± {np.std(eval_rewards):.1f}")
    print(f"  最高/最低：{max(eval_rewards):.0f} / {min(eval_rewards):.0f}")

    return agent


# ============================================================
# 实验三：DQN v1 的不稳定性分析
# ============================================================

def experiment_instability():
    """
    用多次运行展示 DQN v1 的训练不稳定性

    DQN v1 有两个已知问题：
    1. 数据相关性（Correlated Data）
       - 连续的 (s, a, r, s') 来自同一条轨迹，高度相关
       - 类比：考试复习只看一章 → 这章学得特别好，其他全忘
       - 解决方案：v2 引入经验回放，随机打乱数据

    2. 移动目标（Moving Target）
       - TD target = r + γ max Q_θ(s', ·)
       - Q_θ 在不断更新，所以 TD target 也在变
       - 类比：射箭的靶子在动 → 很难瞄准
       - 解决方案：v3 引入目标网络，固定 target 一段时间
    """
    print("\n" + "=" * 60)
    print("实验三：DQN v1 的训练不稳定性")
    print("=" * 60)
    print()
    print("运行 3 次相同配置的 DQN v1，观察训练曲线的差异...")
    print("不稳定的原因：")
    print("  1. 数据相关性：连续样本来自同一轨迹，梯度方向偏")
    print("  2. 移动目标：TD target 随网络更新而变化")
    print()

    rewards_all_runs = []

    for run in range(3):
        print(f"  Run {run + 1}/3 ...")
        env = gym.make('CartPole-v1')
        agent = DQNAgentV1(
            state_dim=4,
            action_dim=2,
            learning_rate=1e-3,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            seed=run * 100 + 42,
        )
        train_dqn_v1(env, agent, n_episodes=200, max_steps=500)
        rewards_all_runs.append(agent.episode_rewards)
        env.close()

        avg_last_50 = np.mean(agent.episode_rewards[-50:])
        print(f"    最后 50 轮平均奖励：{avg_last_50:.1f}")

    # 可视化不稳定性
    visualize_instability(
        rewards_all_runs,
        title="DQN v1 训练不稳定性（3 次独立运行）",
        save_path='images/dqn_v1_instability.png',
    )

    print()
    print("观察：")
    print("  - 不同运行之间差异很大（有的学会了，有的没学会）")
    print("  - 即使学会了，奖励曲线也会剧烈波动（突然崩溃）")
    print("  - 这就是 v1 的核心问题 → v2/v3 将逐步解决")

    return rewards_all_runs


# ============================================================
# 实验四：与 Q 表的 GridWorld 对比
# ============================================================

def experiment_gridworld_comparison():
    """
    在简单的 GridWorld 中对比 Q 表和 DQN

    在小状态空间（16个状态）中，Q 表反而更好：
    - Q 表：精确、稳定、不需要调参
    - DQN：过度参数化，反而不如 Q 表

    这说明 DQN 的优势只在状态空间大/连续时才体现
    """
    print("\n" + "=" * 60)
    print("实验四：GridWorld 中 Q 表 vs DQN")
    print("=" * 60)
    print()
    print("在 4×4 GridWorld（16 个状态）中对比：")
    print("  Q 表在小状态空间中更精确、更稳定")
    print("  DQN 的优势只在大/连续状态空间中才体现")
    print()

    # 使用一个简单的 GridWorld 环境（gymnasium 接口封装）
    class SimpleGridWorldGym:
        """将 GridWorld 封装为类 gymnasium 接口"""

        def __init__(self, size=4, seed=42):
            self.size = size
            self.goal = (size - 1, size - 1)
            self.obstacles = [(1, 1), (2, 2)]
            self.rng = np.random.default_rng(seed)
            self.state = None
            self.n_states = size * size
            self.n_actions = 4
            # 为 DQN 提供状态向量（one-hot 编码）
            self.observation_space_shape = (size * size,)
            self.action_space_n = 4

        def reset(self, seed=None):
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            self.state = (0, 0)
            return self._state_to_vector(self.state)

        def step(self, action):
            row, col = self.state
            moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
            dr, dc = moves[action]
            new_row = np.clip(row + dr, 0, self.size - 1)
            new_col = np.clip(col + dc, 0, self.size - 1)
            new_state = (new_row, new_col)

            if new_state in self.obstacles:
                new_state = self.state

            self.state = new_state

            if self.state == self.goal:
                return self._state_to_vector(self.state), 1.0, True
            elif self.state in self.obstacles:
                return self._state_to_vector(self.state), -1.0, False
            else:
                return self._state_to_vector(self.state), -0.1, False

        def _state_to_vector(self, state):
            """状态 → one-hot 向量"""
            vec = np.zeros(self.n_states, dtype=np.float32)
            idx = state[0] * self.size + state[1]
            vec[idx] = 1.0
            return vec

    # Q 表训练
    print("训练 Q-Learning（Q 表）...")
    grid_env = SimpleGridWorldGym(seed=42)
    q_table = np.zeros((grid_env.n_states, grid_env.n_actions))
    alpha, gamma, epsilon = 0.1, 0.9, 1.0
    rng = np.random.default_rng(42)
    qtable_rewards = []

    for episode in range(300):
        state_vec = grid_env.reset(seed=episode)
        state_idx = np.argmax(state_vec)
        total_reward = 0

        for step in range(100):
            if rng.random() < epsilon:
                action = rng.integers(0, 4)
            else:
                action = np.argmax(q_table[state_idx])

            next_state_vec, reward, done = grid_env.step(action)
            next_state_idx = np.argmax(next_state_vec)

            if done:
                td_target = reward
            else:
                td_target = reward + gamma * np.max(q_table[next_state_idx])

            q_table[state_idx, action] += alpha * (td_target - q_table[state_idx, action])

            state_vec = next_state_vec
            state_idx = next_state_idx
            total_reward += reward

            if done:
                break

        qtable_rewards.append(total_reward)
        epsilon = max(0.01, epsilon * 0.99)

    # DQN 训练
    print("训练 DQN v1...")
    grid_env2 = SimpleGridWorldGym(seed=42)
    dqn_agent = DQNAgentV1(
        state_dim=grid_env2.n_states,
        action_dim=grid_env2.n_actions,
        learning_rate=1e-3,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
        seed=42,
    )

    for episode in range(300):
        state_vec = grid_env2.reset(seed=episode)
        total_reward = 0

        for step in range(100):
            action = dqn_agent.select_action(state_vec)
            next_state_vec, reward, done = grid_env2.step(action)
            dqn_agent.update(state_vec, action, reward, next_state_vec, done)

            state_vec = next_state_vec
            total_reward += reward

            if done:
                break

        dqn_agent.episode_rewards.append(total_reward)
        dqn_agent.episode_steps.append(step + 1)
        dqn_agent.decay_epsilon()

    # 对比结果
    qtable_last50 = np.mean(qtable_rewards[-50:])
    dqn_last50 = np.mean(dqn_agent.episode_rewards[-50:])

    print()
    print("结果对比（最后 50 轮平均奖励）：")
    print(f"  Q-Learning（Q 表）：{qtable_last50:.2f}")
    print(f"  DQN v1（神经网络）：{dqn_last50:.2f}")
    print()
    print("分析：")
    print("  在 16 个状态的小环境中，Q 表通常表现更好：")
    print("  - Q 表为每个 (s,a) 独立存储，精确且稳定")
    print("  - DQN 用 128 维隐藏层的网络去近似 16×4=64 个值，过度参数化")
    print("  - DQN 的泛化能力在这里没有用武之地")
    print()
    print("  → DQN 的优势在 CartPole 等连续/大状态空间中才体现！")

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 5))
    window = 20
    if len(qtable_rewards) >= window:
        smoothed_q = np.convolve(qtable_rewards, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(qtable_rewards)), smoothed_q,
                color='#2196F3', linewidth=2, label='Q-Learning（Q 表）')
    if len(dqn_agent.episode_rewards) >= window:
        smoothed_d = np.convolve(dqn_agent.episode_rewards,
                                  np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(dqn_agent.episode_rewards)), smoothed_d,
                color='#FF5722', linewidth=2, label='DQN v1（神经网络）')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward (滑动平均)')
    ax.set_title('GridWorld：Q 表 vs DQN v1\n(在小状态空间中，Q 表更优)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/dqn_v1_gridworld_comparison.png', dpi=150, bbox_inches='tight')
    print("图像已保存到 images/dqn_v1_gridworld_comparison.png")
    plt.close()


# ============================================================
# 主函数
# ============================================================

if __name__ == '__main__':
    os.makedirs('images', exist_ok=True)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # --- 实验一：Q 表的困境 ---
    qtable_rewards = experiment_qtable_fails()

    # --- 实验二：DQN v1 训练 ---
    agent = experiment_dqn_v1()

    # 可视化 Q 表 vs DQN 在 CartPole 上的对比
    fig, ax = plt.subplots(figsize=(12, 6))
    window = 20
    if len(qtable_rewards) >= window:
        smoothed_qt = np.convolve(qtable_rewards, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(qtable_rewards)), smoothed_qt,
                color='#9E9E9E', linewidth=2, label='Q-Learning + 离散化（Q 表）')
        ax.plot(qtable_rewards, alpha=0.15, color='#9E9E9E')
    dqn_rewards = agent.episode_rewards
    if len(dqn_rewards) >= window:
        smoothed_dqn = np.convolve(dqn_rewards, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(dqn_rewards)), smoothed_dqn,
                color='#2196F3', linewidth=2, label='DQN v1（神经网络）')
        ax.plot(dqn_rewards, alpha=0.15, color='#2196F3')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('CartPole：Q 表 vs DQN v1\nQ 表无法处理连续状态空间，DQN 通过泛化有效学习')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=500, color='green', linestyle='--', alpha=0.4, label='满分 (500)')
    plt.tight_layout()
    plt.savefig('images/dqn_v1_cartpole_comparison.png', dpi=150, bbox_inches='tight')
    print(f"图像已保存到 images/dqn_v1_cartpole_comparison.png")
    plt.close()

    # 可视化 DQN 单独的学习曲线（含 loss）
    visualize_learning_curve(
        [agent], ['DQN v1'],
        title='DQN v1 在 CartPole 上的学习曲线',
        save_path='images/dqn_v1_learning_curve.png',
    )

    # 可视化 Q 值地形图
    visualize_q_landscape(agent, save_path='images/dqn_v1_q_landscape.png')

    # --- 实验三：不稳定性分析 ---
    rewards_runs = experiment_instability()

    # --- 实验四：GridWorld 对比 ---
    experiment_gridworld_comparison()

    # --- 总结 ---
    print("\n\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print()
    print("DQN v1：用神经网络替代 Q 表")
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Q-Learning:  Q 表 → Q[s, a]                              │")
    print("  │  DQN v1:      Q 网络 → Q_θ(s)[a]                          │")
    print("  │                                                             │")
    print("  │  更新方式：                                                 │")
    print("  │    Q-Learning: Q[s,a] += α * (target - Q[s,a])            │")
    print("  │    DQN v1:     θ -= lr * ∇_θ (Q_θ(s)[a] - target)²       │")
    print("  │                                                             │")
    print("  │  target = r + γ max_a' Q(s', a')                           │")
    print("  │  （两者的 target 计算方式完全相同！）                        │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print("  ✅ 解决的问题：")
    print("     - 连续状态空间（如 CartPole 的 4 维连续向量）")
    print("     - 状态泛化（相似状态共享知识）")
    print()
    print("  ❌ 未解决的问题：")
    print("     1. 数据相关性 → 训练不稳定")
    print("        连续样本来自同一轨迹，高度相关")
    print("        → v2 解决方案：经验回放（随机采样打乱数据）")
    print("     2. 移动目标 → 训练振荡")
    print("        TD target 用同一个网络计算，网络更新 target 就变了")
    print("        → v3 解决方案：目标网络（固定 target 一段时间）")
    print()
    print("  下一步：")
    print("     dqn_v2.py → 引入经验回放，打破数据相关性")
    print("     dqn_v3.py → 引入目标网络，稳定 TD target")
