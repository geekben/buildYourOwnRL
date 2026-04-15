"""
q_learning.py - Q-Learning：从规划到学习的跨越

这是整个学习路线中最关键的转折点：
- 值迭代/策略迭代：需要已知 P(s'|s,a) 和 R(s,a,s')，是"规划"
- Q-Learning：不需要知道 P 和 R，通过与环境交互来"学习"

学习要点：
1. 从 V(s) 到 Q(s,a)：为什么需要 Q 函数
2. 时序差分（TD）：不等回合结束就更新
3. ε-greedy 探索：在 Q-Learning 中的角色
4. Off-policy vs On-policy：Q-Learning 为什么是 off-policy

核心算法：
    Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
                              ─────────────────────────────────
                                       TD target - 当前估计 = TD error
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mdp_gridworld import GridWorld, RandomAgent, run_episode


class QLearningAgent:
    """
    Q-Learning 智能体

    核心变化（相比值迭代）：
    - 不需要知道 P 和 R
    - 维护 Q(s, a) 而不是 V(s)
    - 通过与环境交互来更新 Q 值
    - 使用 ε-greedy 策略平衡探索与利用
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, seed=42):
        """
        Args:
            n_states: 状态数量
            n_actions: 动作数量
            alpha: 学习率（步长），控制每次更新的幅度
            gamma: 折扣因子，与值迭代中的含义完全相同
            epsilon: 初始探索率
            epsilon_decay: 每个回合后 epsilon 的衰减系数
            epsilon_min: epsilon 的下限
            seed: 随机种子
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.default_rng(seed)

        # Q 表：Q(s, a)，初始化为 0
        # 值迭代只维护 V(s)，Q-Learning 维护 Q(s, a)
        self.Q = np.zeros((n_states, n_actions))

        # 记录学习过程
        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_history = []
        self.td_errors = []

    def select_action(self, state):
        """
        ε-greedy 策略选择动作

        与 bandit 中的 ε-greedy 完全相同的思想：
        - 以 ε 概率随机探索
        - 以 1-ε 概率贪心利用当前 Q 值
        """
        state_idx = state if isinstance(state, int) else state[0] * int(np.sqrt(self.n_states)) + state[1]

        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)
        else:
            # 贪心：选择 Q 值最大的动作
            return np.argmax(self.Q[state_idx])

    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning 更新规则（核心！）

        Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]

        对比值迭代的更新：
            V(s) ← max_a Σ P(s'|s,a) [R + γV(s')]

        关键区别：
        1. 值迭代用 P 和 R 计算期望 → Q-Learning 用实际采样 (r, s')
        2. 值迭代遍历所有 s' → Q-Learning 只用一个采样的 s'
        3. 值迭代一步到位 → Q-Learning 用学习率 α 渐进更新
        """
        state_idx = self._state_to_idx(state)
        next_state_idx = self._state_to_idx(next_state)

        # TD target：r + γ max_a' Q(s', a')
        # 如果是终止状态，没有未来价值
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state_idx])

        # TD error：TD target - 当前估计
        td_error = td_target - self.Q[state_idx, action]

        # 更新 Q 值：向 TD target 方向迈一小步
        self.Q[state_idx, action] += self.alpha * td_error

        self.td_errors.append(abs(td_error))

    def decay_epsilon(self):
        """衰减探索率：随着学习进行，逐渐减少探索"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        """从 Q 表提取贪心策略：π(s) = argmax_a Q(s, a)"""
        return np.argmax(self.Q, axis=1)

    def get_value_function(self):
        """从 Q 表提取价值函数：V(s) = max_a Q(s, a)"""
        return np.max(self.Q, axis=1)

    def _state_to_idx(self, state):
        """将状态转换为索引"""
        if isinstance(state, (tuple, list)):
            size = int(np.sqrt(self.n_states))
            return state[0] * size + state[1]
        return state


# ============================================================
# 训练函数
# ============================================================

def train_q_learning(env, agent, n_episodes=500, max_steps=100):
    """
    训练 Q-Learning 智能体

    与值迭代的根本区别：
    - 值迭代：一次性计算，不需要与环境交互
    - Q-Learning：需要反复与环境交互，逐步学习

    Args:
        env: GridWorld 环境
        agent: QLearningAgent 智能体
        n_episodes: 训练回合数
        max_steps: 每回合最大步数
    """
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # 选择动作（ε-greedy）
            action = agent.select_action(state)

            # 与环境交互（这是 Q-Learning 的数据来源！）
            next_state, reward, done = env.step(action)

            # 更新 Q 值（每一步都更新，不等回合结束）
            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            if done:
                break

        # 记录本回合数据
        agent.episode_rewards.append(total_reward)
        agent.episode_steps.append(step + 1)
        agent.epsilon_history.append(agent.epsilon)

        # 衰减探索率
        agent.decay_epsilon()

        # 打印进度
        if (episode + 1) % 100 == 0:
            recent_rewards = agent.episode_rewards[-100:]
            recent_steps = agent.episode_steps[-100:]
            success_count = sum(1 for s in recent_steps if s < max_steps)
            print(f"  Episode {episode + 1:4d} | "
                  f"Avg Reward: {np.mean(recent_rewards):6.2f} | "
                  f"Avg Steps: {np.mean(recent_steps):5.1f} | "
                  f"Success: {success_count}% | "
                  f"ε: {agent.epsilon:.3f}")


# ============================================================
# 可视化函数
# ============================================================

def visualize_q_table(env, Q, title="Q-Table Heatmap", save_path=None):
    """可视化 Q 表：展示每个状态下各动作的 Q 值"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    action_names = env.action_names

    for action_idx in range(env.n_actions):
        ax = axes[action_idx]
        q_grid = np.zeros((env.size, env.size))

        for row in range(env.size):
            for col in range(env.size):
                state_idx = env.state_to_idx((row, col))
                q_grid[row, col] = Q[state_idx, action_idx]

        im = ax.imshow(q_grid, cmap='RdYlGn', origin='upper')

        for row in range(env.size):
            for col in range(env.size):
                state = (row, col)
                if state == env.goal:
                    ax.text(col, row, 'G', ha='center', va='center',
                            fontsize=14, fontweight='bold', color='white')
                elif state in env.obstacles:
                    ax.text(col, row, 'X', ha='center', va='center',
                            fontsize=14, fontweight='bold', color='white')
                else:
                    ax.text(col, row, f'{q_grid[row, col]:.2f}',
                            ha='center', va='center', fontsize=9)

        ax.set_title(f'Q(s, {action_names[action_idx]})', fontsize=12)
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_learning_curve(agent, title="Q-Learning Learning Curve", save_path=None):
    """可视化学习曲线：奖励、步数、探索率"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # 1. 累计奖励（滑动平均）
    ax = axes[0]
    rewards = agent.episode_rewards
    window = min(50, len(rewards) // 5) if len(rewards) > 10 else 1
    smoothed_rewards = np.convolve(rewards, np.ones(window) / window, mode='valid')
    ax.plot(smoothed_rewards, linewidth=1.5, color='steelblue')
    ax.set_ylabel('Cumulative Reward', fontsize=11)
    ax.set_title('Episode Reward (smoothed)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 2. 步数
    ax = axes[1]
    steps = agent.episode_steps
    smoothed_steps = np.convolve(steps, np.ones(window) / window, mode='valid')
    ax.plot(smoothed_steps, linewidth=1.5, color='coral')
    ax.set_ylabel('Steps', fontsize=11)
    ax.set_title('Episode Steps (smoothed)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 3. 探索率
    ax = axes[2]
    ax.plot(agent.epsilon_history, linewidth=1.5, color='green')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Epsilon', fontsize=11)
    ax.set_title('Exploration Rate Decay', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_policy(env, policy, V=None, title="Policy", save_path=None):
    """可视化策略（箭头），与值迭代/策略迭代保持一致的风格"""
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


def visualize_value_function(env, V, title="Value Function", save_path=None):
    """可视化价值函数"""
    fig, ax = plt.subplots(figsize=(6, 6))

    value_grid = np.zeros((env.size, env.size))
    for row in range(env.size):
        for col in range(env.size):
            state_idx = env.state_to_idx((row, col))
            value_grid[row, col] = V[state_idx]

    im = ax.imshow(value_grid, cmap='RdYlGn', origin='upper')

    for row in range(env.size):
        for col in range(env.size):
            state = (row, col)
            if state in env.obstacles:
                ax.text(col, row, 'X', ha='center', va='center',
                        fontsize=16, fontweight='bold', color='white')
            elif state == env.goal:
                ax.text(col, row, 'G', ha='center', va='center',
                        fontsize=16, fontweight='bold', color='white')
            elif state == env.start:
                ax.text(col, row, 'S', ha='center', va='center',
                        fontsize=12, fontweight='bold', color='blue')
            else:
                ax.text(col, row, f'{value_grid[row, col]:.2f}',
                        ha='center', va='center', fontsize=10)

    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax, label='Value')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================
# 对比实验
# ============================================================

def compare_with_value_iteration(env, q_agent):
    """对比 Q-Learning 学到的策略 vs 值迭代的最优策略"""
    from value_iteration import ValueIterationAgent

    print("\n" + "=" * 60)
    print("Q-Learning vs 值迭代 对比")
    print("=" * 60)

    # 值迭代（已知模型的精确解）
    vi_agent = ValueIterationAgent(env, gamma=0.9, theta=1e-6)
    vi_agent.value_iteration()
    vi_agent.extract_policy()

    # Q-Learning 的结果
    q_policy = q_agent.get_policy()
    q_values = q_agent.get_value_function()

    # 价值函数对比
    value_diff = np.max(np.abs(q_values - vi_agent.V))
    mean_value_diff = np.mean(np.abs(q_values - vi_agent.V))
    print(f"\n价值函数最大差异: {value_diff:.4f}")
    print(f"价值函数平均差异: {mean_value_diff:.4f}")

    # 策略对比（排除终点和障碍物）
    policy_match = 0
    valid_states = 0
    for state_idx in range(env.n_states):
        state = env.idx_to_state(state_idx)
        if state == env.goal or state in env.obstacles:
            continue
        valid_states += 1
        if q_policy[state_idx] == vi_agent.policy[state_idx]:
            policy_match += 1

    print(f"\n策略一致率: {policy_match}/{valid_states} "
          f"({policy_match / valid_states * 100:.1f}%)")

    print("\n分析：")
    print("  - 值迭代：已知模型，精确计算，一定收敛到最优")
    print("  - Q-Learning：未知模型，采样学习，近似最优")
    print("  - Q-Learning 的结果质量取决于：训练回合数、学习率、探索策略")
    print("  - 在足够多的训练后，Q-Learning 可以收敛到与值迭代相同的结果")

    return vi_agent


def compare_hyperparameters(env):
    """对比不同超参数对 Q-Learning 的影响"""
    print("\n" + "=" * 60)
    print("超参数对比实验")
    print("=" * 60)

    configs = [
        {"name": "α=0.1 (standard)", "alpha": 0.1, "epsilon_decay": 0.995},
        {"name": "α=0.5 (aggressive)", "alpha": 0.5, "epsilon_decay": 0.995},
        {"name": "α=0.01 (conservative)", "alpha": 0.01, "epsilon_decay": 0.995},
        {"name": "slow explore decay", "alpha": 0.1, "epsilon_decay": 0.999},
    ]

    all_agents = []

    for config in configs:
        env_copy = GridWorld(stochastic=False, seed=42)
        agent = QLearningAgent(
            n_states=env_copy.n_states,
            n_actions=env_copy.n_actions,
            alpha=config["alpha"],
            gamma=0.9,
            epsilon=1.0,
            epsilon_decay=config["epsilon_decay"],
            seed=42,
        )

        rewards_history = []
        for episode in range(1000):
            state = env_copy.reset()
            total_reward = 0
            for step in range(100):
                action = agent.select_action(state)
                next_state, reward, done = env_copy.step(action)
                agent.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                if done:
                    break
            rewards_history.append(total_reward)
            agent.decay_epsilon()

        all_agents.append((config["name"], rewards_history))

    # 绘制对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['steelblue', 'coral', 'green', 'purple']
    window = 50

    for idx, (name, rewards) in enumerate(all_agents):
        smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax.plot(smoothed, label=name, linewidth=1.5, color=colors[idx])

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cumulative Reward (smoothed)', fontsize=12)
    ax.set_title('Q-Learning: Hyperparameter Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/q_learning_hyperparams.png', dpi=150, bbox_inches='tight')
    print("图像已保存到 images/q_learning_hyperparams.png")
    plt.close()

    print("\n观察：")
    print("  - α 太大（0.5）：学习不稳定，Q 值震荡")
    print("  - α 太小（0.01）：学习太慢，需要更多回合")
    print("  - α=0.1：通常是不错的默认值")
    print("  - ε 衰减太慢：探索过多，利用不足，收敛慢")


def evaluate_learned_policy(env, agent, n_runs=100, max_steps=50):
    """评估学到的策略（关闭探索）"""
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # 关闭探索，纯利用

    success_count = 0
    total_rewards = []
    total_steps = []

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
        if step + 1 < max_steps:
            success_count += 1

    agent.epsilon = original_epsilon  # 恢复

    return {
        "success_rate": success_count / n_runs,
        "avg_reward": np.mean(total_rewards),
        "avg_steps": np.mean(total_steps),
    }


# ============================================================
# 主演示
# ============================================================

def demonstrate_q_learning():
    """演示 Q-Learning 算法"""
    print("=" * 60)
    print("Q-Learning - 从规划到学习的跨越")
    print("=" * 60)
    print()
    print("核心变化：")
    print("  值迭代/策略迭代：")
    print("    ✅ 已知 P(s'|s,a) 和 R(s,a,s')")
    print("    ✅ 遍历所有状态，精确计算")
    print("    ❌ 现实中几乎不可能知道完整模型")
    print()
    print("  Q-Learning：")
    print("    ✅ 不需要知道 P 和 R")
    print("    ✅ 通过与环境交互来学习")
    print("    ✅ 每一步都更新（时序差分）")
    print("    ❌ 需要大量交互数据")
    print()
    print("更新公式：")
    print("  Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]")
    print("                        ───────────────────────────────")
    print("                              TD error（惊喜信号）")
    print()

    # 创建环境（确定性）
    env = GridWorld(stochastic=False, seed=42)

    # 创建 Q-Learning 智能体
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        seed=42,
    )

    # 训练
    print("开始训练...")
    print("-" * 60)
    train_q_learning(env, agent, n_episodes=1000, max_steps=100)

    # 可视化学习曲线
    visualize_learning_curve(agent, save_path='images/q_learning_curve.png')

    # 可视化 Q 表
    visualize_q_table(env, agent.Q, title="Learned Q-Table",
                      save_path='images/q_learning_q_table.png')

    # 提取并可视化策略
    learned_policy = agent.get_policy()
    learned_values = agent.get_value_function()

    visualize_value_function(env, learned_values,
                             title="Learned Value Function (Q-Learning)",
                             save_path='images/q_learning_value.png')

    visualize_policy(env, learned_policy, learned_values,
                     title="Learned Policy (Q-Learning)",
                     save_path='images/q_learning_policy.png')

    # 打印详细结果
    print("\n" + "=" * 60)
    print("学到的价值函数 V(s) = max_a Q(s,a)")
    print("=" * 60)
    for row in range(env.size):
        row_values = []
        for col in range(env.size):
            state_idx = env.state_to_idx((row, col))
            row_values.append(f"{learned_values[state_idx]:6.3f}")
        print(" ".join(row_values))

    print("\n" + "=" * 60)
    print("学到的策略")
    print("=" * 60)
    action_names = env.action_names
    for row in range(env.size):
        row_actions = []
        for col in range(env.size):
            state = (row, col)
            if state == env.goal:
                row_actions.append("  G  ")
            elif state in env.obstacles:
                row_actions.append("  X  ")
            else:
                state_idx = env.state_to_idx(state)
                action = learned_policy[state_idx]
                row_actions.append(f"{action_names[action]:>5}")
        print(" ".join(row_actions))

    # 评估学到的策略
    print("\n" + "=" * 60)
    print("策略评估（100 次运行，关闭探索）")
    print("=" * 60)
    result = evaluate_learned_policy(env, agent)
    print(f"  成功率: {result['success_rate'] * 100:.1f}%")
    print(f"  平均奖励: {result['avg_reward']:.2f}")
    print(f"  平均步数: {result['avg_steps']:.1f}")

    # 与随机策略对比
    random_agent = RandomAgent(env.n_actions)
    random_rewards = []
    random_steps = []
    random_success = 0
    for _ in range(100):
        env.reset()
        total_reward, steps, _ = run_episode(env, random_agent, max_steps=50)
        random_rewards.append(total_reward)
        random_steps.append(steps)
        if steps < 50:
            random_success += 1

    print(f"\n随机策略对比：")
    print(f"  成功率: {random_success}%")
    print(f"  平均奖励: {np.mean(random_rewards):.2f}")
    print(f"  平均步数: {np.mean(random_steps):.1f}")

    # 与值迭代对比
    compare_with_value_iteration(env, agent)

    return env, agent


def demonstrate_stochastic_env():
    """演示随机环境中的 Q-Learning"""
    print("\n\n" + "=" * 60)
    print("随机环境中的 Q-Learning")
    print("=" * 60)
    print()
    print("值迭代在随机环境中需要知道 P(s'|s,a) 的精确值。")
    print("Q-Learning 不需要！它通过多次采样自动学习期望值。")
    print()

    env = GridWorld(stochastic=True, slip_prob=0.2, seed=42)

    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        seed=42,
    )

    print("训练中（随机环境需要更多回合）...")
    print("-" * 60)
    train_q_learning(env, agent, n_episodes=2000, max_steps=100)

    learned_policy = agent.get_policy()
    learned_values = agent.get_value_function()

    visualize_value_function(env, learned_values,
                             title="Learned V (Stochastic, Q-Learning)",
                             save_path='images/q_learning_stochastic_value.png')
    visualize_policy(env, learned_policy, learned_values,
                     title="Learned Policy (Stochastic, Q-Learning)",
                     save_path='images/q_learning_stochastic_policy.png')

    result = evaluate_learned_policy(env, agent, n_runs=100)
    print(f"\n策略评估（随机环境）：")
    print(f"  成功率: {result['success_rate'] * 100:.1f}%")
    print(f"  平均奖励: {result['avg_reward']:.2f}")
    print(f"  平均步数: {result['avg_steps']:.1f}")

    print("\n观察：")
    print("  - Q-Learning 在随机环境中同样有效")
    print("  - 不需要知道滑倒概率，通过多次经历自动学习")
    print("  - 随机环境需要更多训练回合（采样方差更大）")
    print("  - 这正是 Q-Learning 的核心优势：无模型！")


if __name__ == "__main__":
    # 确保 images 目录存在
    os.makedirs('images', exist_ok=True)

    # 确定性环境
    env, agent = demonstrate_q_learning()

    # 超参数对比
    compare_hyperparameters(env)

    # 随机环境
    demonstrate_stochastic_env()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print()
    print("Q-Learning 的核心突破：")
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  从「规划」到「学习」                                  │")
    print("  │                                                         │")
    print("  │  值迭代：V(s) ← max_a Σ P(s'|s,a) [R + γV(s')]       │")
    print("  │          ─────────── 需要 P 和 R ──────────            │")
    print("  │                                                         │")
    print("  │  Q-Learning：Q(s,a) ← Q + α [r + γ max Q(s',·) - Q]  │")
    print("  │              ──── 只需要一个采样 (s,a,r,s') ────       │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    print("  ┌──────────────┬──────────────────┬──────────────────────┐")
    print("  │              │ 值迭代/策略迭代  │ Q-Learning           │")
    print("  ├──────────────┼──────────────────┼──────────────────────┤")
    print("  │ 模型依赖     │ 需要 P 和 R     │ 不需要（无模型）     │")
    print("  │ 数据来源     │ 查表计算         │ 与环境交互采样       │")
    print("  │ 更新时机     │ 全状态遍历       │ 每步即时更新（TD）   │")
    print("  │ 更新目标     │ V(s)             │ Q(s, a)              │")
    print("  │ 探索策略     │ 不需要           │ ε-greedy             │")
    print("  │ 策略类型     │ —                │ Off-policy           │")
    print("  │ 收敛保证     │ 精确收敛         │ 渐近收敛（需条件）   │")
    print("  └──────────────┴──────────────────┴──────────────────────┘")
    print()
    print("Q-Learning 是 Off-policy 的：")
    print("  - 行为策略（behavior policy）：ε-greedy，用于探索")
    print("  - 目标策略（target policy）：greedy，max_a' Q(s',a')")
    print("  - 用一个策略收集数据，学习另一个策略")
    print("  - 这意味着可以从任何数据中学习（包括他人的经验）")
    print()
    print("下一步：")
    print("  SARSA：On-policy 的 TD 学习")
    print("  - 行为策略和目标策略相同")
    print("  - Q(s,a) ← Q + α [r + γ Q(s',a') - Q]")
    print("  -                         ↑ 用实际选的 a'，不是 max")
    print("  - 更保守，更安全，但可能不是最优")
