"""
sarsa.py - SARSA：On-policy 时序差分学习

与 Q-Learning 唯一的区别：
    Q-Learning: Q(s,a) ← Q + α [r + γ max_a' Q(s',a') - Q]   ← 用 max（贪心）
    SARSA:      Q(s,a) ← Q + α [r + γ Q(s',a') - Q]           ← 用实际选的 a'

这一个字母的区别（max vs 实际 a'）导致了截然不同的行为：
    - Q-Learning（off-policy）：学习最优策略，但可能走危险路线
    - SARSA（on-policy）：学习的是"我实际会怎么走"，更保守、更安全

学习要点：
1. On-policy vs Off-policy 的本质区别
2. SARSA 名字的由来：(S, A, R, S', A')
3. 在"悬崖行走"环境中的经典对比
4. Expected SARSA：介于两者之间的方法

核心算法：
    选择 a = ε-greedy(Q, s)
    执行 a，得到 r, s'
    选择 a' = ε-greedy(Q, s')    ← 关键：先选好下一步动作
    Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]
    s ← s', a ← a'
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mdp_gridworld import GridWorld, RandomAgent, run_episode


# ============================================================
# 悬崖行走环境：展示 SARSA vs Q-Learning 的经典场景
# ============================================================

class CliffWalk:
    """
    悬崖行走环境（Sutton & Barto 经典例子）

    4×12 网格：
    - 起点：左下角 (3, 0)
    - 终点：右下角 (3, 11)
    - 悬崖：底部一排中间位置 (3, 1) 到 (3, 10)
    - 掉下悬崖：回到起点，奖励 -100
    - 每步：奖励 -1

    这个环境完美展示了 SARSA 和 Q-Learning 的区别：
    - Q-Learning 学到沿悬崖边走的最短路径（最优但危险）
    - SARSA 学到远离悬崖的安全路径（次优但安全）
    """

    def __init__(self, seed=42):
        self.rows = 4
        self.cols = 12
        self.size = max(self.rows, self.cols)
        self.n_states = self.rows * self.cols
        self.n_actions = 4
        self.action_names = ['Up', 'Down', 'Left', 'Right']
        self.rng = np.random.default_rng(seed)

        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = [(3, c) for c in range(1, 11)]
        self.obstacles = []

        self.state = None
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def state_to_idx(self, state):
        return state[0] * self.cols + state[1]

    def idx_to_state(self, idx):
        return (idx // self.cols, idx % self.cols)

    def step(self, action):
        row, col = self.state

        if action == 0:    # Up
            row = max(0, row - 1)
        elif action == 1:  # Down
            row = min(self.rows - 1, row + 1)
        elif action == 2:  # Left
            col = max(0, col - 1)
        elif action == 3:  # Right
            col = min(self.cols - 1, col + 1)

        next_state = (row, col)

        if next_state in self.cliff:
            # 掉下悬崖：回到起点，大惩罚
            self.state = self.start
            return self.start, -100, False
        elif next_state == self.goal:
            self.state = next_state
            return next_state, -1, True
        else:
            self.state = next_state
            return next_state, -1, False

    def render(self):
        grid = [['.' for _ in range(self.cols)] for _ in range(self.rows)]
        for cliff_pos in self.cliff:
            grid[cliff_pos[0]][cliff_pos[1]] = 'C'
        grid[self.start[0]][self.start[1]] = 'S'
        grid[self.goal[0]][self.goal[1]] = 'G'
        if self.state not in (self.start, self.goal) and self.state not in self.cliff:
            grid[self.state[0]][self.state[1]] = 'A'
        for row in grid:
            print(' '.join(row))


# ============================================================
# SARSA 智能体
# ============================================================

class SarsaAgent:
    """
    SARSA 智能体

    与 Q-Learning 的唯一区别：
    - Q-Learning 的 TD target 用 max_a' Q(s', a')
    - SARSA 的 TD target 用 Q(s', a')，其中 a' 是 ε-greedy 实际选出的动作

    名字由来：每次更新需要 (S, A, R, S', A') 五元组
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, seed=42):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.default_rng(seed)

        self.Q = np.zeros((n_states, n_actions))

        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_history = []
        self.td_errors = []

    def select_action(self, state):
        """ε-greedy 策略选择动作（与 Q-Learning 完全相同）"""
        state_idx = self._state_to_idx(state)

        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)
        else:
            return np.argmax(self.Q[state_idx])

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA 更新规则（核心！）

        Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]
                                       ─────────
                                       用实际选的 a'，不是 max！

        对比 Q-Learning：
            Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
                                           ─────────────────
                                           用 max（假设未来最优）
        """
        state_idx = self._state_to_idx(state)
        next_state_idx = self._state_to_idx(next_state)

        if done:
            td_target = reward
        else:
            # SARSA 的关键：用实际选的 next_action，而不是 max
            td_target = reward + self.gamma * self.Q[next_state_idx, next_action]

        td_error = td_target - self.Q[state_idx, action]
        self.Q[state_idx, action] += self.alpha * td_error
        self.td_errors.append(abs(td_error))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        return np.argmax(self.Q, axis=1)

    def get_value_function(self):
        return np.max(self.Q, axis=1)

    def _state_to_idx(self, state):
        if isinstance(state, (tuple, list)):
            # 自动检测是 GridWorld(4x4) 还是 CliffWalk(4x12)
            if self.n_states == 48:
                return state[0] * 12 + state[1]
            else:
                size = int(np.sqrt(self.n_states))
                return state[0] * size + state[1]
        return state


# ============================================================
# Expected SARSA 智能体
# ============================================================

class ExpectedSarsaAgent:
    """
    Expected SARSA 智能体

    介于 Q-Learning 和 SARSA 之间：
    - Q-Learning：用 max_a' Q(s', a')
    - SARSA：用 Q(s', a')，a' 是一个采样
    - Expected SARSA：用 E_π[Q(s', a')]，对所有 a' 取期望

    Expected SARSA 消除了 SARSA 中因 a' 采样带来的方差，
    同时保留了 on-policy 的特性。
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, seed=42):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.default_rng(seed)

        self.Q = np.zeros((n_states, n_actions))

        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_history = []
        self.td_errors = []

    def select_action(self, state):
        state_idx = self._state_to_idx(state)
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)
        else:
            return np.argmax(self.Q[state_idx])

    def update(self, state, action, reward, next_state, done):
        """
        Expected SARSA 更新规则

        Q(s,a) ← Q(s,a) + α [r + γ E_π[Q(s',a')] - Q(s,a)]

        E_π[Q(s',a')] = Σ_a' π(a'|s') Q(s',a')
                       = (1-ε) max_a' Q(s',a') + ε/|A| Σ_a' Q(s',a')
        """
        state_idx = self._state_to_idx(state)
        next_state_idx = self._state_to_idx(next_state)

        if done:
            td_target = reward
        else:
            # 计算 ε-greedy 策略下的期望 Q 值
            q_next = self.Q[next_state_idx]
            best_action = np.argmax(q_next)
            expected_q = 0.0
            for action_idx in range(self.n_actions):
                if action_idx == best_action:
                    prob = (1 - self.epsilon) + self.epsilon / self.n_actions
                else:
                    prob = self.epsilon / self.n_actions
                expected_q += prob * q_next[action_idx]

            td_target = reward + self.gamma * expected_q

        td_error = td_target - self.Q[state_idx, action]
        self.Q[state_idx, action] += self.alpha * td_error
        self.td_errors.append(abs(td_error))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        return np.argmax(self.Q, axis=1)

    def get_value_function(self):
        return np.max(self.Q, axis=1)

    def _state_to_idx(self, state):
        if isinstance(state, (tuple, list)):
            if self.n_states == 48:
                return state[0] * 12 + state[1]
            else:
                size = int(np.sqrt(self.n_states))
                return state[0] * size + state[1]
        return state


# ============================================================
# 训练函数
# ============================================================

def train_sarsa(env, agent, n_episodes=500, max_steps=200):
    """
    训练 SARSA 智能体

    与 Q-Learning 训练的关键区别：
    - Q-Learning：选 a → 执行 → 更新 → 选下一个 a
    - SARSA：选 a → 执行 → 选 a' → 用 a' 更新 → a←a'

    SARSA 需要在更新前就选好下一步动作 a'
    """
    for episode in range(n_episodes):
        state = env.reset()
        action = agent.select_action(state)  # SARSA 在回合开始时就选好第一个动作
        total_reward = 0

        for step in range(max_steps):
            next_state, reward, done = env.step(action)
            next_action = agent.select_action(next_state)  # 先选好 a'

            # 用 (S, A, R, S', A') 更新
            agent.update(state, action, reward, next_state, next_action, done)

            total_reward += reward
            state = next_state
            action = next_action  # a ← a'

            if done:
                break

        agent.episode_rewards.append(total_reward)
        agent.episode_steps.append(step + 1)
        agent.epsilon_history.append(agent.epsilon)
        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            recent_rewards = agent.episode_rewards[-100:]
            print(f"  Episode {episode + 1:4d} | "
                  f"Avg Reward: {np.mean(recent_rewards):7.1f} | "
                  f"ε: {agent.epsilon:.3f}")


def train_q_learning_for_comparison(env, agent, n_episodes=500, max_steps=200):
    """训练 Q-Learning 智能体（用于对比，适配不同环境）"""
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break

        agent.episode_rewards.append(total_reward)
        agent.episode_steps.append(step + 1)
        agent.epsilon_history.append(agent.epsilon)
        agent.decay_epsilon()


def train_expected_sarsa(env, agent, n_episodes=500, max_steps=200):
    """训练 Expected SARSA 智能体"""
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break

        agent.episode_rewards.append(total_reward)
        agent.episode_steps.append(step + 1)
        agent.epsilon_history.append(agent.epsilon)
        agent.decay_epsilon()


# ============================================================
# Q-Learning 智能体（本地定义，避免循环导入问题）
# ============================================================

class QLearningAgentLocal:
    """Q-Learning 智能体（用于对比实验）"""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, seed=42):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions))
        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_history = []
        self.td_errors = []

    def select_action(self, state):
        state_idx = self._state_to_idx(state)
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)
        else:
            return np.argmax(self.Q[state_idx])

    def update(self, state, action, reward, next_state, done):
        state_idx = self._state_to_idx(state)
        next_state_idx = self._state_to_idx(next_state)
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state_idx])
        td_error = td_target - self.Q[state_idx, action]
        self.Q[state_idx, action] += self.alpha * td_error
        self.td_errors.append(abs(td_error))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        return np.argmax(self.Q, axis=1)

    def get_value_function(self):
        return np.max(self.Q, axis=1)

    def _state_to_idx(self, state):
        if isinstance(state, (tuple, list)):
            if self.n_states == 48:
                return state[0] * 12 + state[1]
            else:
                size = int(np.sqrt(self.n_states))
                return state[0] * size + state[1]
        return state


# ============================================================
# 可视化函数
# ============================================================

def visualize_cliff_policy(env, policy, title="Policy", save_path=None):
    """可视化悬崖行走环境的策略"""
    fig, ax = plt.subplots(figsize=(14, 4))

    action_arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    # 背景色
    grid_colors = np.zeros((env.rows, env.cols, 3))
    for row in range(env.rows):
        for col in range(env.cols):
            if (row, col) in env.cliff:
                grid_colors[row, col] = [0.8, 0.2, 0.2]  # 红色：悬崖
            elif (row, col) == env.start:
                grid_colors[row, col] = [0.3, 0.5, 0.9]  # 蓝色：起点
            elif (row, col) == env.goal:
                grid_colors[row, col] = [0.3, 0.8, 0.3]  # 绿色：终点
            else:
                grid_colors[row, col] = [0.95, 0.95, 0.95]  # 浅灰：普通

    ax.imshow(grid_colors, origin='upper', aspect='equal')

    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)
            if state == env.goal:
                ax.text(col, row, 'G', ha='center', va='center',
                        fontsize=14, fontweight='bold', color='white')
            elif state in env.cliff:
                ax.text(col, row, 'C', ha='center', va='center',
                        fontsize=10, fontweight='bold', color='white')
            elif state == env.start:
                state_idx = env.state_to_idx(state)
                action = policy[state_idx]
                ax.text(col, row, f'S{action_arrows[action]}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color='white')
            else:
                state_idx = env.state_to_idx(state)
                action = policy[state_idx]
                ax.text(col, row, action_arrows[action], ha='center', va='center',
                        fontsize=14, fontweight='bold')

    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_learning_comparison(agents_data, title="Learning Curve Comparison",
                                  save_path=None):
    """对比多个智能体的学习曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['steelblue', 'coral', 'green', 'purple']
    window = 50

    for idx, (name, rewards) in enumerate(agents_data):
        if len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
        else:
            smoothed = rewards
        ax.plot(smoothed, label=name, linewidth=1.5,
                color=colors[idx % len(colors)])

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cumulative Reward (smoothed)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_policy_comparison(env, policies, titles, save_path=None):
    """并排对比多个策略"""
    num_policies = len(policies)
    fig, axes = plt.subplots(1, num_policies, figsize=(14 * num_policies // 2, 4))
    if num_policies == 1:
        axes = [axes]

    action_arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    for ax_idx, (policy, title) in enumerate(zip(policies, titles)):
        ax = axes[ax_idx]

        grid_colors = np.zeros((env.rows, env.cols, 3))
        for row in range(env.rows):
            for col in range(env.cols):
                if (row, col) in env.cliff:
                    grid_colors[row, col] = [0.8, 0.2, 0.2]
                elif (row, col) == env.start:
                    grid_colors[row, col] = [0.3, 0.5, 0.9]
                elif (row, col) == env.goal:
                    grid_colors[row, col] = [0.3, 0.8, 0.3]
                else:
                    grid_colors[row, col] = [0.95, 0.95, 0.95]

        ax.imshow(grid_colors, origin='upper', aspect='equal')

        for row in range(env.rows):
            for col in range(env.cols):
                state = (row, col)
                if state == env.goal:
                    ax.text(col, row, 'G', ha='center', va='center',
                            fontsize=12, fontweight='bold', color='white')
                elif state in env.cliff:
                    ax.text(col, row, 'C', ha='center', va='center',
                            fontsize=8, fontweight='bold', color='white')
                elif state == env.start:
                    state_idx = env.state_to_idx(state)
                    action = policy[state_idx]
                    ax.text(col, row, f'S{action_arrows[action]}', ha='center', va='center',
                            fontsize=10, fontweight='bold', color='white')
                else:
                    state_idx = env.state_to_idx(state)
                    action = policy[state_idx]
                    ax.text(col, row, action_arrows[action], ha='center', va='center',
                            fontsize=12, fontweight='bold')

        ax.set_xlim(-0.5, env.cols - 0.5)
        ax.set_ylim(env.rows - 0.5, -0.5)
        ax.set_xticks(range(env.cols))
        ax.set_yticks(range(env.rows))
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_gridworld_policy(env, policy, V=None, title="Policy", save_path=None):
    """可视化 GridWorld 策略（与 q_learning.py 保持一致）"""
    fig, ax = plt.subplots(figsize=(6, 6))

    if V is not None:
        value_grid = np.zeros((env.size, env.size))
        for row in range(env.size):
            for col in range(env.size):
                state_idx = env.state_to_idx((row, col))
                value_grid[row, col] = V[state_idx]
        ax.imshow(value_grid, cmap='RdYlGn', origin='upper', alpha=0.3)

    action_arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    for row in range(env.size):
        for col in range(env.size):
            state = (row, col)
            if state == env.goal:
                ax.text(col, row, 'G', ha='center', va='center',
                        fontsize=20, fontweight='bold', color='green')
            elif state in env.obstacles:
                ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                           color='red', alpha=0.5))
                ax.text(col, row, 'X', ha='center', va='center',
                        fontsize=16, fontweight='bold', color='white')
            elif state == env.start:
                action = policy[env.state_to_idx(state)]
                ax.text(col, row, 'S', ha='center', va='center',
                        fontsize=14, fontweight='bold', color='blue')
                ax.text(col + 0.3, row + 0.3, action_arrows[action],
                        ha='center', va='center', fontsize=12, color='darkblue')
            else:
                state_idx = env.state_to_idx(state)
                action = policy[state_idx]
                ax.text(col, row, action_arrows[action], ha='center', va='center',
                        fontsize=16, fontweight='bold')

    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================
# 评估函数
# ============================================================

def evaluate_policy(env, agent, n_runs=100, max_steps=200):
    """评估策略（关闭探索）"""
    original_epsilon = agent.epsilon
    agent.epsilon = 0

    total_rewards = []
    total_steps = []
    success_count = 0

    for _ in range(n_runs):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
            if done:
                break

        total_rewards.append(episode_reward)
        total_steps.append(step + 1)
        if done:
            success_count += 1

    agent.epsilon = original_epsilon

    return {
        "success_rate": success_count / n_runs,
        "avg_reward": np.mean(total_rewards),
        "avg_steps": np.mean(total_steps),
    }


# ============================================================
# 核心对比实验：悬崖行走
# ============================================================

def cliff_walk_experiment():
    """
    经典实验：在悬崖行走环境中对比 SARSA vs Q-Learning

    这是 Sutton & Barto 教科书中的经典例子，
    完美展示了 on-policy 和 off-policy 的行为差异。
    """
    print("=" * 60)
    print("悬崖行走实验：SARSA vs Q-Learning vs Expected SARSA")
    print("=" * 60)
    print()
    print("环境：4×12 网格，底部有悬崖")
    print("  S . . . . . . . . . . G")
    print("  掉下悬崖：回到起点，奖励 -100")
    print("  每步：奖励 -1")
    print()
    print("预期行为：")
    print("  Q-Learning：学到沿悬崖边走的最短路径（最优但危险）")
    print("  SARSA：学到远离悬崖的安全路径（次优但安全）")
    print("  Expected SARSA：介于两者之间")
    print()

    env = CliffWalk(seed=42)
    n_episodes = 500
    # 使用固定 epsilon 以更清晰地展示差异
    fixed_epsilon = 0.1

    # SARSA
    print("训练 SARSA...")
    sarsa_agent = SarsaAgent(
        n_states=env.n_states, n_actions=env.n_actions,
        alpha=0.5, gamma=1.0,
        epsilon=fixed_epsilon, epsilon_decay=1.0, epsilon_min=fixed_epsilon,
        seed=42,
    )
    train_sarsa(env, sarsa_agent, n_episodes=n_episodes, max_steps=200)

    # Q-Learning
    print("\n训练 Q-Learning...")
    env_q = CliffWalk(seed=42)
    q_agent = QLearningAgentLocal(
        n_states=env_q.n_states, n_actions=env_q.n_actions,
        alpha=0.5, gamma=1.0,
        epsilon=fixed_epsilon, epsilon_decay=1.0, epsilon_min=fixed_epsilon,
        seed=42,
    )
    train_q_learning_for_comparison(env_q, q_agent, n_episodes=n_episodes, max_steps=200)

    # Expected SARSA
    print("\n训练 Expected SARSA...")
    env_es = CliffWalk(seed=42)
    expected_sarsa_agent = ExpectedSarsaAgent(
        n_states=env_es.n_states, n_actions=env_es.n_actions,
        alpha=0.5, gamma=1.0,
        epsilon=fixed_epsilon, epsilon_decay=1.0, epsilon_min=fixed_epsilon,
        seed=42,
    )
    train_expected_sarsa(env_es, expected_sarsa_agent, n_episodes=n_episodes, max_steps=200)

    # 学习曲线对比
    print()
    visualize_learning_comparison(
        [
            ("SARSA (on-policy)", sarsa_agent.episode_rewards),
            ("Q-Learning (off-policy)", q_agent.episode_rewards),
            ("Expected SARSA", expected_sarsa_agent.episode_rewards),
        ],
        title="Cliff Walk: SARSA vs Q-Learning vs Expected SARSA",
        save_path='images/sarsa_cliff_learning_curve.png',
    )

    # 策略对比
    sarsa_policy = sarsa_agent.get_policy()
    q_policy = q_agent.get_policy()
    es_policy = expected_sarsa_agent.get_policy()

    visualize_policy_comparison(
        env,
        [sarsa_policy, q_policy, es_policy],
        ["SARSA (safe path)", "Q-Learning (optimal path)", "Expected SARSA"],
        save_path='images/sarsa_cliff_policy_comparison.png',
    )

    # 评估对比
    print("\n" + "=" * 60)
    print("策略评估（关闭探索，100 次运行）")
    print("=" * 60)

    sarsa_result = evaluate_policy(env, sarsa_agent)
    q_result = evaluate_policy(env_q, q_agent)
    es_result = evaluate_policy(env_es, expected_sarsa_agent)

    print(f"\n  SARSA：")
    print(f"    成功率: {sarsa_result['success_rate'] * 100:.0f}% | "
          f"平均奖励: {sarsa_result['avg_reward']:.1f} | "
          f"平均步数: {sarsa_result['avg_steps']:.1f}")

    print(f"  Q-Learning：")
    print(f"    成功率: {q_result['success_rate'] * 100:.0f}% | "
          f"平均奖励: {q_result['avg_reward']:.1f} | "
          f"平均步数: {q_result['avg_steps']:.1f}")

    print(f"  Expected SARSA：")
    print(f"    成功率: {es_result['success_rate'] * 100:.0f}% | "
          f"平均奖励: {es_result['avg_reward']:.1f} | "
          f"平均步数: {es_result['avg_steps']:.1f}")

    print("\n分析：")
    print("  Q-Learning 学到了最短路径（沿悬崖边走），关闭探索后表现最优")
    print("  但训练过程中，ε-greedy 的随机探索会导致频繁掉下悬崖")
    print("  → 训练期间的平均奖励更低（曲线更低）")
    print()
    print("  SARSA 知道自己会探索（on-policy），所以学到了远离悬崖的安全路径")
    print("  训练期间的平均奖励更高（少掉悬崖），但最终策略路径更长")
    print()
    print("  Expected SARSA 消除了采样方差，表现介于两者之间")

    return sarsa_agent, q_agent, expected_sarsa_agent


# ============================================================
# GridWorld 对比实验
# ============================================================

def gridworld_experiment():
    """在 GridWorld 中对比 SARSA vs Q-Learning"""
    print("\n\n" + "=" * 60)
    print("GridWorld 实验：SARSA vs Q-Learning")
    print("=" * 60)
    print()
    print("在没有悬崖的普通 GridWorld 中，两者差异较小。")
    print("因为没有「危险区域」，on-policy 和 off-policy 的保守性差异不明显。")
    print()

    # SARSA
    env_s = GridWorld(stochastic=False, seed=42)
    sarsa_agent = SarsaAgent(
        n_states=env_s.n_states, n_actions=env_s.n_actions,
        alpha=0.1, gamma=0.9,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
        seed=42,
    )
    print("训练 SARSA...")
    train_sarsa(env_s, sarsa_agent, n_episodes=1000, max_steps=100)

    # Q-Learning
    env_q = GridWorld(stochastic=False, seed=42)
    q_agent = QLearningAgentLocal(
        n_states=env_q.n_states, n_actions=env_q.n_actions,
        alpha=0.1, gamma=0.9,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
        seed=42,
    )
    print("\n训练 Q-Learning...")
    train_q_learning_for_comparison(env_q, q_agent, n_episodes=1000, max_steps=100)

    # 学习曲线对比
    print()
    visualize_learning_comparison(
        [
            ("SARSA", sarsa_agent.episode_rewards),
            ("Q-Learning", q_agent.episode_rewards),
        ],
        title="GridWorld: SARSA vs Q-Learning",
        save_path='images/sarsa_gridworld_learning_curve.png',
    )

    # 策略对比
    sarsa_policy = sarsa_agent.get_policy()
    q_policy = q_agent.get_policy()
    sarsa_values = sarsa_agent.get_value_function()
    q_values = q_agent.get_value_function()

    visualize_gridworld_policy(env_s, sarsa_policy, sarsa_values,
                               title="SARSA Policy (GridWorld)",
                               save_path='images/sarsa_gridworld_policy.png')
    visualize_gridworld_policy(env_q, q_policy, q_values,
                               title="Q-Learning Policy (GridWorld)",
                               save_path='images/sarsa_gridworld_q_policy.png')

    # 策略一致性
    valid_match = 0
    valid_total = 0
    for state_idx in range(env_s.n_states):
        state = env_s.idx_to_state(state_idx)
        if state == env_s.goal or state in env_s.obstacles:
            continue
        valid_total += 1
        if sarsa_policy[state_idx] == q_policy[state_idx]:
            valid_match += 1

    print(f"\n策略一致率: {valid_match}/{valid_total} "
          f"({valid_match / valid_total * 100:.1f}%)")
    print("在没有危险区域的环境中，SARSA 和 Q-Learning 通常学到相似的策略。")


# ============================================================
# 主演示
# ============================================================

if __name__ == "__main__":
    os.makedirs('images', exist_ok=True)

    # 核心实验：悬崖行走
    sarsa_agent, q_agent, es_agent = cliff_walk_experiment()

    # GridWorld 对比
    gridworld_experiment()

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print()
    print("SARSA vs Q-Learning：一个字母的区别")
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Q-Learning: Q(s,a) ← Q + α [r + γ max Q(s',·) - Q]     │")
    print("  │                                   ─── 假设未来最优         │")
    print("  │                                                             │")
    print("  │  SARSA:      Q(s,a) ← Q + α [r + γ Q(s',a') - Q]        │")
    print("  │                                   ──── 用实际选的 a'       │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print("  ┌──────────────┬──────────────────────┬──────────────────────┐")
    print("  │              │ Q-Learning           │ SARSA                │")
    print("  ├──────────────┼──────────────────────┼──────────────────────┤")
    print("  │ 策略类型     │ Off-policy           │ On-policy            │")
    print("  │ TD target    │ r + γ max Q(s',·)    │ r + γ Q(s',a')      │")
    print("  │ 学习目标     │ 最优策略             │ 当前策略的价值       │")
    print("  │ 行为特点     │ 激进（忽略探索风险） │ 保守（考虑探索风险） │")
    print("  │ 悬崖行走     │ 沿悬崖边走（最短）   │ 远离悬崖（安全）     │")
    print("  │ 经验回放     │ ✅ 可以              │ ❌ 不行              │")
    print("  │ 收敛到       │ Q*（最优）           │ Q^π（当前策略）      │")
    print("  └──────────────┴──────────────────────┴──────────────────────┘")
    print()
    print("Expected SARSA：两者的折中")
    print("  Q(s,a) ← Q + α [r + γ E_π[Q(s',·)] - Q]")
    print("  - 用期望替代采样，消除方差")
    print("  - 当 ε→0 时退化为 Q-Learning")
    print("  - 通常比 SARSA 更稳定")
    print()
    print("核心洞察：")
    print("  On-policy 学的是「我实际会怎么走」（包括犯错）")
    print("  Off-policy 学的是「如果我不犯错，最优路线是什么」")
    print("  哪个更好取决于场景：安全关键 → SARSA，追求最优 → Q-Learning")
    print()
    print("下一步：")
    print("  DQN：用神经网络替代 Q 表")
    print("  - Q 表无法处理大状态空间（如 Atari 像素）")
    print("  - 神经网络 Q_θ(s) 可以泛化到未见过的状态")
    print("  - 需要经验回放和目标网络来稳定训练")
