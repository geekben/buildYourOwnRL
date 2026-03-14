"""
bandit_epsilon_greedy.py - ε-greedy 策略：平衡探索与利用

在 bandit.py 的基础上，引入一个新的概念：
- ε-greedy 策略：以 ε 的概率随机探索，以 1-ε 的概率贪婪利用

学习要点：
1. 探索与利用的权衡
2. ε 的选择如何影响学习效果
3. 与纯贪婪策略的对比
"""

import numpy as np
import matplotlib.pyplot as plt


class BanditEnv:
    """多臂老虎机环境（与 bandit.py 相同）"""
    
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


class EpsilonGreedyAgent:
    """
    ε-greedy 智能体
    
    策略：
    - 以 ε 的概率随机选择一个臂（探索）
    - 以 1-ε 的概率选择当前估计最好的臂（利用）
    
    关键洞察：
    - ε 太小：探索不足，可能错过最优臂
    - ε 太大：探索过多，浪费时间在已知不好的臂
    """
    
    def __init__(self, n_arms, epsilon=0.1, seed=None):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        
        self.q_values = np.zeros(n_arms)
        self.counts = np.zeros(n_arms, dtype=int)
    
    def select_action(self):
        """
        ε-greedy 策略选择动作
        """
        if self.rng.random() < self.epsilon:
            # 探索：随机选择一个臂
            return self.rng.integers(0, self.n_arms)
        else:
            # 利用：选择当前估计最好的臂
            return np.argmax(self.q_values)
    
    def update(self, arm, reward):
        """更新估计（增量式均值）"""
        self.counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.counts[arm]


class GreedyAgent:
    """纯贪婪智能体（用于对比）"""
    
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.q_values = np.zeros(n_arms)
        self.counts = np.zeros(n_arms, dtype=int)
    
    def select_action(self):
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
    """绘制不同 ε 值的对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    colors = ['red', 'green', 'blue', 'purple']
    
    # 左图：平均奖励
    for (label, (rewards, _)), color in zip(results.items(), colors):
        axes[0].plot(rewards, label=label, color=color, alpha=0.8)
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('ε-greedy: Average Reward Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 右图：最优动作比例
    for (label, (_, optimal)), color in zip(results.items(), colors):
        axes[1].plot(optimal * 100, label=label, color=color, alpha=0.8)
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('% Optimal Action')
    axes[1].set_title('ε-greedy: % Optimal Action Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('images/bandit_epsilon_greedy_comparison.png', dpi=150)
    plt.close()
    print("结果已保存到 images/bandit_epsilon_greedy_comparison.png")


if __name__ == "__main__":
    print("=" * 60)
    print("ε-greedy 策略：平衡探索与利用")
    print("=" * 60)
    print()
    print("核心思想：")
    print("  - 以 ε 的概率随机探索（尝试未知的臂）")
    print("  - 以 1-ε 的概率贪婪利用（选择当前最好的臂）")
    print()
    print("对比策略：")
    print("  - ε=0.0 : 纯贪婪（从不探索）")
    print("  - ε=0.01: 几乎不探索")
    print("  - ε=0.1 : 经典设置，10%时间探索")
    print("  - ε=0.5 : 过度探索")
    print()
    
    print("正在运行实验（2000次平均）...")
    
    results = {}
    
    # 测试不同的 ε 值
    epsilons = [0.0, 0.01, 0.1, 0.5]
    
    for eps in epsilons:
        print(f"  测试 ε={eps}...")
        if eps == 0.0:
            # 纯贪婪
            rewards, optimal = run_experiment(
                GreedyAgent, {}, 
                n_arms=10, n_steps=1000, n_runs=2000
            )
        else:
            rewards, optimal = run_experiment(
                EpsilonGreedyAgent, {'epsilon': eps},
                n_arms=10, n_steps=1000, n_runs=2000
            )
        
        label = f"ε={eps}" if eps > 0 else "Greedy (ε=0)"
        results[label] = (rewards, optimal)
    
    # 绘制对比图
    plot_comparison(results)
    
    # 输出统计
    print()
    print("=" * 60)
    print("结果分析")
    print("=" * 60)
    print()
    print(f"{'策略':<15} {'后100步平均奖励':>15} {'最优动作比例':>15}")
    print("-" * 50)
    for label, (rewards, optimal) in results.items():
        avg_reward = rewards[-100:].mean()
        opt_pct = optimal[-100:].mean() * 100
        print(f"{label:<15} {avg_reward:>15.3f} {opt_pct:>14.1f}%")
    
    print()
    print("=" * 60)
    print("关键发现")
    print("=" * 60)
    print("1. ε=0（贪婪）效果最差：探索不足，容易陷入次优")
    print("2. ε=0.1 效果最好：平衡了探索与利用")
    print("3. ε=0.5 效果下降：探索过多，浪费时间")
    print()
    print("思考：为什么 ε=0.01 后期表现不如 ε=0.1？")
    print("  → 因为探索太少，可能仍未找到最优臂！")
    print()
    print("下一节：UCB 策略如何更聪明地探索？")
