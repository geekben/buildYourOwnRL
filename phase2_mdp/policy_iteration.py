"""
policy_iteration.py - 策略迭代：策略评估 + 策略改进

上一课（值迭代）的回顾：
    V(s) ← max_a Σ P(s'|s,a) [R(s,a,s') + γV(s')]
    值迭代在每一步都取 max，同时更新价值和隐含策略。

本课引入的新概念：策略迭代
    将"求解最优策略"拆成两个交替的步骤：
    
    步骤 1 - 策略评估（Policy Evaluation）：
        给定固定策略 π，求解其价值函数 V^π
        V^π(s) ← Σ P(s'|s,π(s)) [R(s,π(s),s') + γV^π(s')]
        注意：这里没有 max，因为策略已经固定了！
    
    步骤 2 - 策略改进（Policy Improvement）：
        根据 V^π 贪心地更新策略
        π'(s) = argmax_a Σ P(s'|s,a) [R(s,a,s') + γV^π(s')]
    
    重复步骤 1 和 2，直到策略不再变化（收敛）。

与值迭代的对比：
    值迭代：每次迭代都做 max（评估和改进混在一起）
    策略迭代：先完整评估当前策略，再改进策略（分开做）
    
    策略迭代通常迭代次数更少，但每次迭代更昂贵（需要完整评估）
"""

import numpy as np
import matplotlib.pyplot as plt
from mdp_gridworld import GridWorld, RandomAgent, run_episode


class PolicyIterationAgent:
    """
    策略迭代智能体

    交替执行策略评估和策略改进，直到策略收敛。

    前提条件：已知 MDP 的完整信息（P, R）
    """

    def __init__(self, env, gamma=0.9, theta=1e-6):
        """
        Args:
            env: GridWorld 环境
            gamma: 折扣因子
            theta: 策略评估的收敛阈值
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta

        # 价值函数：V^π(s)
        self.V = np.zeros(env.n_states)

        # 策略：π(s) → a，初始化为全部向右（任意初始策略都行）
        self.policy = np.zeros(env.n_states, dtype=int)
        self.policy[:] = 3  # 初始策略：全部向右

        # 记录过程
        self.eval_iterations_history = []  # 每轮策略评估的迭代次数
        self.policy_history = []           # 每轮的策略快照
        self.value_history = []            # 每轮的价值函数快照

    def compute_q_value(self, state, action):
        """
        计算 Q(s, a) = Σ P(s'|s,a) [R(s,a,s') + γV(s')]

        与值迭代中完全相同的计算
        """
        q_value = 0
        current_state = self.env.idx_to_state(state)

        for next_row in range(self.env.size):
            for next_col in range(self.env.size):
                next_state = (next_row, next_col)
                next_state_idx = self.env.state_to_idx(next_state)

                prob = self.env.get_transition_prob(current_state, action, next_state)

                if prob > 0:
                    reward = self.env.get_reward(current_state, next_state)
                    q_value += prob * (reward + self.gamma * self.V[next_state_idx])

        return q_value

    def policy_evaluation(self, max_iterations=1000):
        """
        策略评估：给定固定策略 π，计算 V^π

        这是策略迭代与值迭代的关键区别：
        - 值迭代：V(s) ← max_a Q(s, a)        ← 取 max
        - 策略评估：V(s) ← Q(s, π(s))          ← 用固定策略的动作

        Returns:
            iterations: 评估收敛所需的迭代次数
        """
        for iteration in range(max_iterations):
            delta = 0

            for state_idx in range(self.env.n_states):
                state = self.env.idx_to_state(state_idx)

                # 跳过终点和障碍物
                if state == self.env.goal:
                    self.V[state_idx] = 0
                    continue
                if state in self.env.obstacles:
                    self.V[state_idx] = 0
                    continue

                v_old = self.V[state_idx]

                # 关键区别：使用当前策略的动作，而不是 max
                action = self.policy[state_idx]
                self.V[state_idx] = self.compute_q_value(state_idx, action)

                delta = max(delta, abs(self.V[state_idx] - v_old))

            if delta < self.theta:
                return iteration + 1

        return max_iterations

    def policy_improvement(self):
        """
        策略改进：根据当前 V^π 贪心更新策略

        π'(s) = argmax_a Q(s, a)

        Returns:
            policy_stable: 策略是否已经稳定（没有变化）
        """
        policy_stable = True

        for state_idx in range(self.env.n_states):
            state = self.env.idx_to_state(state_idx)

            # 终点和障碍物不需要策略
            if state == self.env.goal or state in self.env.obstacles:
                continue

            old_action = self.policy[state_idx]

            # 选择使 Q 值最大的动作
            q_values = [self.compute_q_value(state_idx, a) for a in range(self.env.n_actions)]
            self.policy[state_idx] = np.argmax(q_values)

            # 检查策略是否发生了变化
            if old_action != self.policy[state_idx]:
                policy_stable = False

        return policy_stable

    def policy_iteration(self, max_outer_iterations=100):
        """
        策略迭代主循环

        交替执行策略评估和策略改进，直到策略稳定。

        Returns:
            outer_iterations: 外层迭代次数（策略改变了多少次）
        """
        for outer_iteration in range(max_outer_iterations):
            # 记录当前策略
            self.policy_history.append(self.policy.copy())

            # 步骤 1：策略评估
            eval_iters = self.policy_evaluation()
            self.eval_iterations_history.append(eval_iters)
            self.value_history.append(self.V.copy())

            print(f"  第 {outer_iteration + 1} 轮：策略评估用了 {eval_iters} 次迭代", end="")

            # 步骤 2：策略改进
            policy_stable = self.policy_improvement()

            if policy_stable:
                print(" → 策略已稳定，收敛！")
                self.policy_history.append(self.policy.copy())
                print(f"\n策略迭代在第 {outer_iteration + 1} 轮后收敛")
                return outer_iteration + 1
            else:
                # 统计有多少个状态的策略发生了变化
                changed_count = np.sum(
                    self.policy_history[-1] != self.policy
                )
                print(f" → 策略改进，{changed_count} 个状态的动作发生变化")

        print(f"达到最大迭代次数 {max_outer_iterations}，未完全收敛")
        return max_outer_iterations

    def select_action(self, state):
        """根据当前策略选择动作"""
        state_idx = self.env.state_to_idx(state)
        return self.policy[state_idx]

    def update(self, state, action, reward, next_state):
        """策略迭代是离线规划，不需要在线更新"""
        pass


# ============================================================
# 可视化函数
# ============================================================

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


def visualize_policy(env, policy, V=None, title="Policy", save_path=None):
    """可视化策略（箭头）"""
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


def visualize_policy_evolution(env, policy_history, save_path=None):
    """可视化策略的演化过程：展示每一轮策略改进后策略的变化"""
    num_policies = min(len(policy_history), 6)  # 最多展示 6 个
    if num_policies <= 1:
        return

    # 均匀选取要展示的轮次
    indices = np.linspace(0, len(policy_history) - 1, num_policies, dtype=int)

    fig, axes = plt.subplots(1, num_policies, figsize=(4 * num_policies, 4))
    if num_policies == 1:
        axes = [axes]

    action_arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    for ax_idx, policy_idx in enumerate(indices):
        ax = axes[ax_idx]
        policy = policy_history[policy_idx]

        for row in range(env.size):
            for col in range(env.size):
                state = (row, col)
                if state == env.goal:
                    ax.text(col, row, 'G', ha='center', va='center',
                            fontsize=14, fontweight='bold', color='green')
                elif state in env.obstacles:
                    ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                               color='red', alpha=0.5))
                    ax.text(col, row, 'X', ha='center', va='center',
                            fontsize=12, fontweight='bold', color='white')
                else:
                    state_idx = env.state_to_idx(state)
                    action = policy[state_idx]
                    ax.text(col, row, action_arrows[action],
                            ha='center', va='center', fontsize=14, fontweight='bold')

        ax.set_xlim(-0.5, env.size - 0.5)
        ax.set_ylim(env.size - 0.5, -0.5)
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        if policy_idx == 0:
            ax.set_title("初始策略", fontsize=12)
        elif policy_idx == len(policy_history) - 1:
            ax.set_title("最终策略 ✓", fontsize=12)
        else:
            ax.set_title(f"第 {policy_idx} 轮", fontsize=12)

    plt.suptitle("策略演化过程", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_eval_iterations(eval_iterations_history, save_path=None):
    """可视化每轮策略评估所需的迭代次数"""
    fig, ax = plt.subplots(figsize=(8, 4))

    rounds = range(1, len(eval_iterations_history) + 1)
    ax.bar(rounds, eval_iterations_history, color='steelblue', alpha=0.8)
    ax.set_xlabel('Policy Iteration Round', fontsize=12)
    ax.set_ylabel('Evaluation Iterations', fontsize=12)
    ax.set_title('Policy Evaluation Cost per Round', fontsize=14)
    ax.set_xticks(list(rounds))
    ax.grid(True, alpha=0.3, axis='y')

    # 标注数值
    for round_num, iters in zip(rounds, eval_iterations_history):
        ax.text(round_num, iters + 0.5, str(iters), ha='center', fontsize=10)

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

def compare_with_value_iteration(env, pi_agent):
    """对比策略迭代 vs 值迭代的结果"""
    from value_iteration import ValueIterationAgent

    print("\n" + "=" * 60)
    print("策略迭代 vs 值迭代 对比")
    print("=" * 60)

    # 值迭代
    vi_agent = ValueIterationAgent(env, gamma=0.9, theta=1e-6)
    vi_iters = vi_agent.value_iteration()
    vi_agent.extract_policy()

    # 价值函数差异
    value_diff = np.max(np.abs(pi_agent.V - vi_agent.V))
    print(f"\n价值函数最大差异: {value_diff:.10f}")

    # 策略差异
    policy_match = np.sum(pi_agent.policy == vi_agent.policy)
    total_states = env.n_states - len(env.obstacles) - 1  # 排除障碍物和终点
    print(f"策略一致的状态数: {policy_match}/{env.n_states} "
          f"(有效状态 {total_states} 个)")

    # 迭代次数对比
    pi_total_eval_iters = sum(pi_agent.eval_iterations_history)
    print(f"\n值迭代：{vi_iters} 次迭代")
    print(f"策略迭代：{len(pi_agent.eval_iterations_history)} 轮外层迭代，"
          f"共 {pi_total_eval_iters} 次内层评估迭代")

    print("\n分析：")
    print("  - 两种方法最终收敛到相同的最优策略和价值函数")
    print("  - 策略迭代的外层迭代次数通常很少（策略空间有限）")
    print("  - 但每轮策略评估本身需要多次迭代才能收敛")
    print("  - 值迭代可以看作策略迭代的特例：每轮只做一次评估就改进")


def compare_policies(env, optimal_agent, n_runs=100, max_steps=50):
    """比较最优策略 vs 随机策略"""
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
    print(f"\n最优策略（策略迭代）：")
    print(f"  到达终点比例: {optimal_success / n_runs * 100:.1f}%")
    print(f"  平均累计奖励: {np.mean(optimal_rewards):.2f}")
    print(f"  平均步数: {np.mean(optimal_steps):.1f}")

    print(f"\n随机策略：")
    print(f"  到达终点比例: {random_success / n_runs * 100:.1f}%")
    print(f"  平均累计奖励: {np.mean(random_rewards):.2f}")
    print(f"  平均步数: {np.mean(random_steps):.1f}")


# ============================================================
# 主演示
# ============================================================

def demonstrate_policy_iteration():
    """演示策略迭代算法"""
    print("=" * 60)
    print("策略迭代 - 策略评估 + 策略改进")
    print("=" * 60)
    print()
    print("核心思想：")
    print("  值迭代把评估和改进混在一起（每步都取 max）")
    print("  策略迭代把它们分开：")
    print()
    print("  步骤 1 - 策略评估：")
    print("    给定策略 π，反复计算直到 V^π 收敛")
    print("    V^π(s) ← Σ P(s'|s,π(s)) [R + γV^π(s')]")
    print()
    print("  步骤 2 - 策略改进：")
    print("    根据 V^π 贪心更新策略")
    print("    π'(s) = argmax_a Q(s, a)")
    print()
    print("  重复直到策略不再变化")
    print()

    # 创建环境
    env = GridWorld(stochastic=False)

    # 创建策略迭代智能体
    agent = PolicyIterationAgent(env, gamma=0.9, theta=1e-6)

    # 显示初始策略
    print("初始策略（全部向右 →）：")
    visualize_policy(env, agent.policy, title="Initial Policy (all →)",
                     save_path='images/policy_iteration_initial.png')

    # 运行策略迭代
    print("\n运行策略迭代...")
    print("-" * 50)
    outer_iterations = agent.policy_iteration()

    # 可视化策略演化
    visualize_policy_evolution(env, agent.policy_history,
                               save_path='images/policy_iteration_evolution.png')

    # 可视化每轮评估的迭代次数
    visualize_eval_iterations(agent.eval_iterations_history,
                              save_path='images/policy_iteration_eval_cost.png')

    # 显示最终价值函数
    print("\n最终价值函数：")
    visualize_value_function(env, agent.V, "Optimal Value Function (Policy Iteration)",
                             save_path='images/policy_iteration_optimal_v.png')

    # 显示最终策略
    print("\n最终策略：")
    visualize_policy(env, agent.policy, agent.V, "Optimal Policy (Policy Iteration)",
                     save_path='images/policy_iteration_optimal_policy.png')

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

    # 与值迭代对比
    compare_with_value_iteration(env, agent)

    return env, agent


def demonstrate_stochastic_env():
    """演示随机环境中的策略迭代"""
    print("\n\n" + "=" * 60)
    print("随机环境中的策略迭代")
    print("=" * 60)
    print()
    print("在随机环境中，策略评估需要考虑转移的不确定性。")
    print("但算法结构完全相同：评估 → 改进 → 评估 → ...")
    print()

    env = GridWorld(stochastic=True, slip_prob=0.2)

    agent = PolicyIterationAgent(env, gamma=0.9, theta=1e-6)

    print("运行策略迭代（随机环境）...")
    print("-" * 50)
    agent.policy_iteration()

    visualize_value_function(env, agent.V, "Optimal V (Stochastic, Policy Iteration)",
                             save_path='images/policy_iteration_stochastic_v.png')
    visualize_policy(env, agent.policy, agent.V,
                     "Optimal Policy (Stochastic, Policy Iteration)",
                     save_path='images/policy_iteration_stochastic_policy.png')

    compare_policies(env, agent)

    print("\n观察：")
    print("  - 策略迭代在随机环境中同样能找到最优策略")
    print("  - 随机环境可能需要更多轮策略评估迭代（V 收敛更慢）")
    print("  - 但外层迭代次数（策略改进次数）通常仍然很少")


if __name__ == "__main__":
    # 确定性环境
    env, agent = demonstrate_policy_iteration()

    # 随机环境
    demonstrate_stochastic_env()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print()
    print("策略迭代 vs 值迭代：")
    print("  ┌─────────────┬──────────────────────┬──────────────────────┐")
    print("  │             │ 值迭代               │ 策略迭代             │")
    print("  ├─────────────┼──────────────────────┼──────────────────────┤")
    print("  │ 更新方式    │ V(s) ← max_a Q(s,a) │ 评估 V^π → 改进 π   │")
    print("  │ 每步操作    │ 评估+改进混合        │ 评估和改进分开       │")
    print("  │ 外层迭代    │ 通常较多             │ 通常很少（2-5 轮）   │")
    print("  │ 每轮代价    │ 较低（一次遍历）     │ 较高（完整评估）     │")
    print("  │ 总计算量    │ 中等                 │ 中等                 │")
    print("  │ 收敛结果    │ 相同的 V* 和 π*      │ 相同的 V* 和 π*      │")
    print("  └─────────────┴──────────────────────┴──────────────────────┘")
    print()
    print("两种方法的共同局限：")
    print("  ❌ 都需要知道完整的 MDP 模型（P, R）")
    print("  ❌ 都需要遍历所有状态（状态空间大时不可行）")
    print()
    print("下一步：")
    print("  Q-Learning：无模型学习！")
    print("  - 不需要知道 P 和 R")
    print("  - 通过与环境交互来学习")
    print("  - 从「规划」转向「学习」")

