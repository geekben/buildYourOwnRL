"""
value_iteration.py - 值迭代：动态规划求解最优策略

学习要点：
1. Bellman 最优方程的迭代求解
2. 值函数的收敛过程
3. 从 V* 推导最优策略 π*
4. 已知模型 vs 未知模型的区别

核心算法：
    V(s) ← max_a Σ P(s'|s,a) [R(s,a,s') + γV(s')]
    重复直到 V 收敛
    π*(s) = argmax_a Σ P(s'|s,a) [R(s,a,s') + γV(s')]
"""

import numpy as np
import matplotlib.pyplot as plt
from mdp_gridworld import GridWorld, RandomAgent, run_episode


class ValueIterationAgent:
    """
    值迭代智能体
    
    通过动态规划求解最优价值函数 V*，然后推导最优策略 π*
    
    前提条件：已知 MDP 的完整信息（P, R）
    """
    
    def __init__(self, env, gamma=0.9, theta=1e-6):
        """
        Args:
            env: GridWorld 环境
            gamma: 折扣因子
            theta: 收敛阈值（V 的最大变化小于 theta 时停止）
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        
        # 价值函数：V(s)
        self.V = np.zeros(env.n_states)
        
        # 策略：π(s) → a
        self.policy = np.zeros(env.n_states, dtype=int)
        
        # 记录收敛过程
        self.value_history = []
        self.delta_history = []
    
    def compute_q_value(self, state, action):
        """
        计算 Q(s, a) = Σ P(s'|s,a) [R(s,a,s') + γV(s')]
        
        这是 Bellman 方程的核心：计算某个状态下某个动作的价值
        """
        q = 0
        current_state = self.env.idx_to_state(state)
        
        # 遍历所有可能的下一状态
        for next_row in range(self.env.size):
            for next_col in range(self.env.size):
                next_state = (next_row, next_col)
                next_state_idx = self.env.state_to_idx(next_state)
                
                # 获取转移概率
                prob = self.env.get_transition_prob(current_state, action, next_state)
                
                if prob > 0:
                    # 获取奖励
                    reward = self.env.get_reward(current_state, next_state)
                    
                    # 累加：P(s'|s,a) * [R + γV(s')]
                    q += prob * (reward + self.gamma * self.V[next_state_idx])
        
        return q
    
    def value_iteration(self, max_iterations=1000):
        """
        值迭代主循环
        
        反复应用 Bellman 最优算子，直到 V 收敛
        
        Returns:
            iterations: 实际迭代次数
        """
        for iteration in range(max_iterations):
            delta = 0  # 记录本次迭代中 V 的最大变化
            
            # 遍历所有状态
            for state_idx in range(self.env.n_states):
                state = self.env.idx_to_state(state_idx)
                
                # 跳过终点和障碍物（它们的价值已经确定）
                if state == self.env.goal:
                    self.V[state_idx] = 0  # 终点价值为 0（已经到达，没有未来奖励）
                    continue
                if state in self.env.obstacles:
                    self.V[state_idx] = 0  # 障碍物不可达
                    continue
                
                # 保存旧值
                v_old = self.V[state_idx]
                
                # Bellman 最优更新：V(s) = max_a Q(s, a)
                q_values = [self.compute_q_value(state_idx, a) for a in range(self.env.n_actions)]
                self.V[state_idx] = max(q_values)
                
                # 更新 delta
                delta = max(delta, abs(self.V[state_idx] - v_old))
            
            # 记录历史
            self.value_history.append(self.V.copy())
            self.delta_history.append(delta)
            
            # 检查收敛
            if delta < self.theta:
                print(f"值迭代在第 {iteration + 1} 次迭代后收敛")
                return iteration + 1
        
        print(f"达到最大迭代次数 {max_iterations}，未完全收敛")
        return max_iterations
    
    def extract_policy(self):
        """
        从最优价值函数 V* 提取最优策略 π*
        
        π*(s) = argmax_a Q(s, a)
        """
        for state_idx in range(self.env.n_states):
            state = self.env.idx_to_state(state_idx)
            
            # 终点和障碍物不需要策略
            if state == self.env.goal or state in self.env.obstacles:
                self.policy[state_idx] = 0  # 默认值
                continue
            
            # 选择 Q 值最大的动作
            q_values = [self.compute_q_value(state_idx, a) for a in range(self.env.n_actions)]
            self.policy[state_idx] = np.argmax(q_values)
    
    def select_action(self, state):
        """根据最优策略选择动作"""
        state_idx = self.env.state_to_idx(state)
        return self.policy[state_idx]
    
    def update(self, state, action, reward, next_state):
        """值迭代是离线规划，不需要在线更新"""
        pass


def visualize_value_function(env, V, title="Value Function", save_path=None):
    """可视化价值函数"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 创建价值矩阵
    value_grid = np.zeros((env.size, env.size))
    for row in range(env.size):
        for col in range(env.size):
            state_idx = env.state_to_idx((row, col))
            value_grid[row, col] = V[state_idx]
    
    # 绘制热力图
    im = ax.imshow(value_grid, cmap='RdYlGn', origin='upper')
    
    # 添加数值标注
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
    ax.set_xticklabels(range(env.size))
    ax.set_yticklabels(range(env.size))
    ax.set_title(title, fontsize=14)
    
    plt.colorbar(im, ax=ax, label='Value')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_policy(env, policy, V=None, title="Optimal Policy", save_path=None):
    """可视化策略（箭头）"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 如果提供了 V，绘制背景热力图
    if V is not None:
        value_grid = np.zeros((env.size, env.size))
        for row in range(env.size):
            for col in range(env.size):
                state_idx = env.state_to_idx((row, col))
                value_grid[row, col] = V[state_idx]
        ax.imshow(value_grid, cmap='RdYlGn', origin='upper', alpha=0.3)
    
    # 箭头方向
    action_arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    action_deltas = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}
    
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


def visualize_convergence(delta_history, save_path=None):
    """可视化收敛过程"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(delta_history, linewidth=2)
    ax.axhline(y=1e-6, color='r', linestyle='--', label='Convergence threshold')
    ax.set_yscale('log')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Max Value Change (δ)', fontsize=12)
    ax.set_title('Value Iteration Convergence', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_policies(env, optimal_agent, n_runs=100, max_steps=50):
    """比较最优策略 vs 随机策略"""
    
    # 最优策略
    optimal_rewards = []
    optimal_steps = []
    optimal_success = 0
    
    for _ in range(n_runs):
        env.reset()
        total_reward, steps, _ = run_episode(env, optimal_agent, max_steps)
        optimal_rewards.append(total_reward)
        optimal_steps.append(steps)
        if steps < max_steps:
            optimal_success += 1
    
    # 随机策略
    random_agent = RandomAgent(env.n_actions)
    random_rewards = []
    random_steps = []
    random_success = 0
    
    for _ in range(n_runs):
        env.reset()
        total_reward, steps, _ = run_episode(env, random_agent, max_steps)
        random_rewards.append(total_reward)
        random_steps.append(steps)
        if steps < max_steps:
            random_success += 1
    
    print("\n" + "=" * 60)
    print("策略对比（100 次运行）")
    print("=" * 60)
    print(f"\n最优策略（值迭代）：")
    print(f"  到达终点比例: {optimal_success / n_runs * 100:.1f}%")
    print(f"  平均累计奖励: {np.mean(optimal_rewards):.2f}")
    print(f"  平均步数: {np.mean(optimal_steps):.1f}")
    
    print(f"\n随机策略：")
    print(f"  到达终点比例: {random_success / n_runs * 100:.1f}%")
    print(f"  平均累计奖励: {np.mean(random_rewards):.2f}")
    print(f"  平均步数: {np.mean(random_steps):.1f}")
    
    print(f"\n提升：")
    print(f"  成功率提升: {optimal_success / n_runs * 100 - random_success / n_runs * 100:.1f}%")
    print(f"  平均奖励提升: {np.mean(optimal_rewards) - np.mean(random_rewards):.2f}")
    print(f"  平均步数减少: {np.mean(random_steps) - np.mean(optimal_steps):.1f} 步")


def demonstrate_value_iteration():
    """演示值迭代算法"""
    print("=" * 60)
    print("值迭代 - 动态规划求解最优策略")
    print("=" * 60)
    print()
    print("核心思想：")
    print("  1. 初始化 V(s) = 0")
    print("  2. 反复应用 Bellman 最优更新：")
    print("     V(s) ← max_a Σ P(s'|s,a) [R(s,a,s') + γV(s')]")
    print("  3. 当 V 收敛后，提取最优策略：")
    print("     π*(s) = argmax_a Q(s, a)")
    print()
    
    # 创建环境
    env = GridWorld(stochastic=False)
    
    # 创建值迭代智能体
    agent = ValueIterationAgent(env, gamma=0.9, theta=1e-6)
    
    # 显示初始价值函数
    print("初始价值函数（全为 0）：")
    visualize_value_function(env, agent.V, "Initial Value Function", 
                            save_path='images/value_iteration_initial.png')
    
    # 运行值迭代
    print("\n运行值迭代...")
    iterations = agent.value_iteration()
    
    # 显示收敛过程
    visualize_convergence(agent.delta_history, 
                         save_path='images/value_iteration_convergence.png')
    
    # 显示最终价值函数
    print("\n最终价值函数：")
    visualize_value_function(env, agent.V, "Optimal Value Function", 
                            save_path='images/value_iteration_optimal_v.png')
    
    # 提取最优策略
    agent.extract_policy()
    
    # 显示最优策略
    print("\n最优策略：")
    visualize_policy(env, agent.policy, agent.V, "Optimal Policy", 
                    save_path='images/value_iteration_optimal_policy.png')
    
    # 打印详细结果
    print("\n" + "=" * 60)
    print("价值函数详细结果")
    print("=" * 60)
    for row in range(env.size):
        row_values = []
        for col in range(env.size):
            state_idx = env.state_to_idx((row, col))
            row_values.append(f"{agent.V[state_idx]:6.3f}")
        print(" ".join(row_values))
    
    print("\n" + "=" * 60)
    print("最优策略详细结果")
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
                action = agent.policy[state_idx]
                row_actions.append(f"{action_names[action]:>5}")
        print(" ".join(row_actions))
    
    # 比较策略效果
    compare_policies(env, agent)
    
    return env, agent


def demonstrate_stochastic_env():
    """演示随机环境中的值迭代"""
    print("\n" + "=" * 60)
    print("随机环境中的值迭代")
    print("=" * 60)
    print()
    print("在随机环境中，转移概率 P(s'|s,a) 不是确定性的。")
    print("值迭代算法仍然适用，只是 Q(s,a) 的计算包含期望。")
    print()
    
    # 创建随机环境
    env = GridWorld(stochastic=True, slip_prob=0.2)
    
    # 运行值迭代
    agent = ValueIterationAgent(env, gamma=0.9, theta=1e-6)
    agent.value_iteration()
    agent.extract_policy()
    
    # 显示结果
    visualize_value_function(env, agent.V, "Optimal V (Stochastic)", 
                            save_path='images/value_iteration_stochastic_v.png')
    visualize_policy(env, agent.policy, agent.V, "Optimal Policy (Stochastic)", 
                    save_path='images/value_iteration_stochastic_policy.png')
    
    # 比较策略
    compare_policies(env, agent)
    
    print("\n观察：")
    print("  - 随机环境中，最优策略需要考虑滑倒的可能性")
    print("  - 策略可能会更加保守，远离障碍物")
    print("  - 成功率比确定性环境低，但仍远优于随机策略")


if __name__ == "__main__":
    # 确定性环境
    env, agent = demonstrate_value_iteration()
    
    # 随机环境
    demonstrate_stochastic_env()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print()
    print("值迭代的特点：")
    print("  ✅ 理论保证：收敛到最优价值函数 V*")
    print("  ✅ 适用于确定性环境和随机环境")
    print("  ✅ 不需要实际交互，纯计算求解")
    print()
    print("值迭代的局限：")
    print("  ❌ 需要知道完整的 MDP 模型（P, R）")
    print("  ❌ 状态空间大时计算代价高")
    print("  ❌ 无法处理连续状态/动作空间")
    print()
    print("下一步：")
    print("  - 策略迭代：另一种 DP 方法")
    print("  - Q-Learning：无模型学习，不需要知道 P 和 R")
