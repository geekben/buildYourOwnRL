"""
bandit_ucb.py - UCB 上置信界探索：更聪明的探索策略

在 bandit_epsilon_greedy.py 的基础上，引入一个新的概念：
- UCB (Upper Confidence Bound)：基于不确定性的智能探索

学习要点：
1. 为什么随机探索不是最优的？
2. 不确定性如何指导探索？
3. UCB 公式的直觉理解
"""

import numpy as np
import matplotlib.pyplot as plt


class BanditEnv:
    """多臂老虎机环境"""
    
    def __init__(self, n_arms=10, seed=42):
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)
        self.true_rewards = self.rng.normal(0, 1, n_arms)
        
    def pull(self, arm):
        return self.rng.normal(self.true_rewards[arm], 1)
    
    def optimal_arm(self):
        return np.argmax(self.true_rewards)
    
    def optimal_reward(self):
        return np.max(self.true_rewards)


class UCBAgent:
    """
    UCB (Upper Confidence Bound) 智能体
    
    核心思想：乐观面对不确定性
    
    选择动作的准则：
    UCB(a) = Q(a) + c * sqrt(ln(t) / N(a))
    
    其中：
    - Q(a)：动作 a 的当前估计价值
    - N(a)：动作 a 被选择的次数
    - t：总步数
    - c：探索系数（通常取 1 或 2）
    
    直觉理解：
    - 第一项 Q(a)：利用当前知识
    - 第二项：对不确定性的"奖励"
      * N(a) 越小（选择越少），不确定性越大，bonus 越大
      * t 越大，整体探索压力增加，但增长缓慢（对数）
    
    优势：
    - 不需要手动调节 ε
    - 自动平衡探索与利用
    - 有理论保证（遗憾上界最优）
    """
    
    def __init__(self, n_arms, c=2.0):
        self.n_arms = n_arms
        self.c = c  # 探索系数
        
        self.q_values = np.zeros(n_arms)
        self.counts = np.zeros(n_arms, dtype=int)
        self.total_count = 0
    
    def select_action(self):
        """
        UCB 策略选择动作
        """
        self.total_count += 1
        
        # 如果有臂还没被选过，优先选它
        untried = np.where(self.counts == 0)[0]
        if len(untried) > 0:
            return untried[0]
        
        # 计算 UCB 值
        ucb_values = self.q_values + self.c * np.sqrt(
            np.log(self.total_count) / self.counts
        )
        
        return np.argmax(ucb_values)
    
    def update(self, arm, reward):
        """更新估计"""
        self.counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.counts[arm]


class EpsilonGreedyAgent:
    """ε-greedy 智能体（用于对比）"""
    
    def __init__(self, n_arms, epsilon=0.1, seed=None):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        
        self.q_values = np.zeros(n_arms)
        self.counts = np.zeros(n_arms, dtype=int)
    
    def select_action(self):
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_arms)
        else:
            return np.argmax(self.q_values)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.counts[arm]


def run_episode(env, agent, n_steps=1000):
    """运行一个回合"""
    rewards = np.zeros(n_steps)
    optimal_actions = np.zeros(n_steps, dtype=bool)
    optimal_arm = env.optimal_arm()
    
    for t in range(n_steps):
        action = agent.select_action()
        reward = env.pull(action)
        agent.update(action, reward)
        
        rewards[t] = reward
        optimal_actions[t] = (action == optimal_arm)
    
    return rewards, optimal_actions


def run_experiment(agent_class, agent_kwargs, n_arms=10, n_steps=1000, n_runs=2000, seed=42):
    """运行多次实验"""
    all_rewards = np.zeros((n_runs, n_steps))
    all_optimal = np.zeros((n_runs, n_steps), dtype=bool)
    
    for run in range(n_runs):
        env = BanditEnv(n_arms, seed=seed + run)
        agent = agent_class(n_arms=n_arms, **agent_kwargs)
        
        rewards, optimal = run_episode(env, agent, n_steps)
        all_rewards[run] = rewards
        all_optimal[run] = optimal
    
    return all_rewards.mean(axis=0), all_optimal.mean(axis=0)


def plot_comparison(results):
    """绘制对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    colors = ['red', 'blue', 'green']
    
    # 左图：平均奖励
    for (label, (rewards, _)), color in zip(results.items(), colors):
        axes[0].plot(rewards, label=label, color=color, alpha=0.8)
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('UCB vs ε-greedy: Average Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 右图：最优动作比例
    for (label, (_, optimal)), color in zip(results.items(), colors):
        axes[1].plot(optimal * 100, label=label, color=color, alpha=0.8)
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('% Optimal Action')
    axes[1].set_title('UCB vs ε-greedy: % Optimal Action')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('images/bandit_ucb_comparison.png', dpi=150)
    plt.close()
    print("结果已保存到 images/bandit_ucb_comparison.png")


def demonstrate_ucb_intuition():
    """演示 UCB 的直觉"""
    print()
    print("=" * 60)
    print("UCB 公式的直觉理解")
    print("=" * 60)
    print()
    print("假设有 2 个臂：")
    print("  臂 A：Q(A) = 0.8, N(A) = 100（选择很多，估计准确）")
    print("  臂 B：Q(B) = 0.6, N(B) = 5（选择很少，估计不确定）")
    print()
    print("当前总步数 t = 105")
    print()
    
    t = 105
    q_a, n_a = 0.8, 100
    q_b, n_b = 0.6, 5
    c = 2.0
    
    bonus_a = c * np.sqrt(np.log(t) / n_a)
    bonus_b = c * np.sqrt(np.log(t) / n_b)
    
    ucb_a = q_a + bonus_a
    ucb_b = q_b + bonus_b
    
    print(f"UCB(A) = {q_a:.2f} + {bonus_a:.3f} = {ucb_a:.3f}")
    print(f"UCB(B) = {q_b:.2f} + {bonus_b:.3f} = {ucb_b:.3f}")
    print()
    print(f"选择臂 {'A' if ucb_a > ucb_b else 'B'}！")
    print()
    print("洞察：虽然臂 A 估计价值更高，但臂 B 不确定性更大")
    print("UCB 会选择臂 B 探索，因为它的上置信界更高")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("UCB 上置信界探索：更聪明的探索策略")
    print("=" * 60)
    
    # 演示 UCB 直觉
    demonstrate_ucb_intuition()
    
    print("正在运行对比实验（2000次平均）...")
    
    results = {}
    
    # ε-greedy (ε=0.1)
    print("  测试 ε-greedy (ε=0.1)...")
    rewards, optimal = run_experiment(
        EpsilonGreedyAgent, {'epsilon': 0.1},
        n_arms=10, n_steps=1000, n_runs=2000
    )
    results['ε-greedy (ε=0.1)'] = (rewards, optimal)
    
    # UCB (c=1)
    print("  测试 UCB (c=1)...")
    rewards, optimal = run_experiment(
        UCBAgent, {'c': 1.0},
        n_arms=10, n_steps=1000, n_runs=2000
    )
    results['UCB (c=1)'] = (rewards, optimal)
    
    # UCB (c=2)
    print("  测试 UCB (c=2)...")
    rewards, optimal = run_experiment(
        UCBAgent, {'c': 2.0},
        n_arms=10, n_steps=1000, n_runs=2000
    )
    results['UCB (c=2)'] = (rewards, optimal)
    
    # 绘制对比图
    plot_comparison(results)
    
    # 输出统计
    print()
    print("=" * 60)
    print("结果分析")
    print("=" * 60)
    print()
    print(f"{'策略':<20} {'后100步平均奖励':>15} {'最优动作比例':>15}")
    print("-" * 55)
    for label, (rewards, optimal) in results.items():
        avg_reward = rewards[-100:].mean()
        opt_pct = optimal[-100:].mean() * 100
        print(f"{label:<20} {avg_reward:>15.3f} {opt_pct:>14.1f}%")
    
    print()
    print("=" * 60)
    print("关键发现")
    print("=" * 60)
    print()
    print("1. UCB 通常比 ε-greedy 更快找到最优臂")
    print("2. UCB 不需要手动调 ε，更智能")
    print("3. 探索系数 c 控制探索程度：")
    print("   - c 较小：更保守，接近贪婪")
    print("   - c 较大：更乐观，探索更多")
    print()
    print("下一节：从多臂老虎机到完整 MDP —— 引入状态！")
