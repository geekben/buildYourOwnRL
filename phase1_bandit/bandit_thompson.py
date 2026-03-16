"""
bandit_thompson.py - Thompson Sampling：贝叶斯探索策略

在 bandit_ucb.py 的基础上，引入一个新的概念：
- Thompson Sampling：从后验分布采样，实现概率匹配探索

学习要点：
1. 贝叶斯思想在强化学习中的应用
2. 后验分布的更新
3. 从分布采样 vs 点估计
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class BanditEnv:
    """多臂老虎机环境（高斯奖励）"""
    
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


class ThompsonSamplingGaussian:
    """
    Thompson Sampling for Gaussian Rewards
    
    假设每个臂的奖励服从 N(μ, σ²)，其中 σ² 已知（设为 1）
    均值 μ 的先验分布为 N(μ₀, σ₀²)
    
    后验更新公式：
    - 观测到奖励 r 后
    - 后验精度 = 先验精度 + 数据精度
    - 后验均值 = 精度加权平均
    
    选择策略：从每个臂的后验分布中采样均值，选择采样值最大的臂
    
    直觉：
    - 如果某个臂被拉得少，后验分布方差大，采样值波动大
    - 如果某个臂被拉得多，后验分布方差小，采样值接近真实均值
    - 自然地实现了"探索不确定性高的臂"
    """
    
    def __init__(self, n_arms, prior_mean=0, prior_var=1, reward_var=1):
        self.n_arms = n_arms
        self.reward_var = reward_var  # 已知的奖励方差
        
        # 后验分布参数：N(posterior_mean, posterior_var)
        self.posterior_mean = np.full(n_arms, prior_mean, dtype=float)
        self.posterior_var = np.full(n_arms, prior_var, dtype=float)
        
        # 累计奖励和计数（用于更新）
        self.sum_rewards = np.zeros(n_arms)
        self.counts = np.zeros(n_arms, dtype=int)
        
        self.rng = np.random.default_rng()
    
    def select_action(self):
        """从每个臂的后验分布中采样，选择采样值最大的臂"""
        # 从后验分布采样
        samples = self.rng.normal(
            self.posterior_mean, 
            np.sqrt(self.posterior_var)
        )
        return np.argmax(samples)
    
    def update(self, arm, reward):
        """
        更新后验分布
        
        对于高斯分布，后验更新有闭式解：
        - 后验精度 τ_post = τ_prior + n * τ_reward
        - 后验均值 = (τ_prior * μ_prior + τ_reward * Σr) / τ_post
        
        这里使用递推形式更新
        """
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        
        n = self.counts[arm]
        
        # 后验精度
        prior_precision = 1.0 / self.reward_var  # 初始先验精度
        data_precision = n / self.reward_var     # 数据精度
        
        posterior_precision = prior_precision + data_precision
        self.posterior_var[arm] = 1.0 / posterior_precision
        
        # 后验均值
        prior_mean = 0  # 初始先验均值
        self.posterior_mean[arm] = (
            prior_precision * prior_mean + self.sum_rewards[arm] / self.reward_var
        ) / posterior_precision


class UCBAgent:
    """UCB 智能体（用于对比）"""
    
    def __init__(self, n_arms, c=2.0):
        self.n_arms = n_arms
        self.c = c
        
        self.q_values = np.zeros(n_arms)
        self.counts = np.zeros(n_arms, dtype=int)
        self.total_count = 0
    
    def select_action(self):
        self.total_count += 1
        
        untried = np.where(self.counts == 0)[0]
        if len(untried) > 0:
            return untried[0]
        
        ucb_values = self.q_values + self.c * np.sqrt(
            np.log(self.total_count) / self.counts
        )
        return np.argmax(ucb_values)
    
    def update(self, arm, reward):
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
    
    colors = ['red', 'blue', 'green', 'purple']
    
    # 左图：平均奖励
    for (label, (rewards, _)), color in zip(results.items(), colors):
        axes[0].plot(rewards, label=label, color=color, alpha=0.8)
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Thompson Sampling vs Other Methods')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 右图：最优动作比例
    for (label, (_, optimal)), color in zip(results.items(), colors):
        axes[1].plot(optimal * 100, label=label, color=color, alpha=0.8)
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('% Optimal Action')
    axes[1].set_title('Thompson Sampling vs Other Methods')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('images/bandit_thompson_comparison.png', dpi=150)
    plt.close()
    print("结果已保存到 images/bandit_thompson_comparison.png")


def visualize_posterior(agent, env, step):
    """可视化后验分布"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.linspace(-3, 3, 200)
    
    for arm in range(agent.n_arms):
        mean = agent.posterior_mean[arm]
        std = np.sqrt(agent.posterior_var[arm])
        y = stats.norm.pdf(x, mean, std)
        ax.plot(x, y, label=f'Arm {arm} (μ={mean:.2f}, σ={std:.2f})')
    
    # 标记真实值
    for arm, true_val in enumerate(env.true_rewards):
        ax.axvline(true_val, color='gray', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Reward Value')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Posterior Distributions after {step} steps')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'images/bandit_thompson_posterior_step{step}.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Thompson Sampling：贝叶斯探索策略")
    print("=" * 60)
    print()
    print("核心思想：")
    print("  - 维护每个臂奖励分布的后验概率")
    print("  - 从后验分布中采样，选择采样值最大的臂")
    print("  - 自然地平衡探索与利用")
    print()
    print("贝叶斯视角：")
    print("  - 先验：初始时假设所有臂的均值分布相同")
    print("  - 后验：每次拉臂后更新对均值的信念")
    print("  - 采样：从信念分布中采样，不确定性大的臂波动大")
    print()
    
    print("正在运行对比实验（2000次平均）...")
    
    results = {}
    
    # ε-greedy (ε=0.1)
    print("  测试 ε-greedy (ε=0.1)...")
    rewards, optimal = run_experiment(
        EpsilonGreedyAgent, {'epsilon': 0.1},
        n_arms=10, n_steps=1000, n_runs=2000
    )
    results['ε-greedy (ε=0.1)'] = (rewards, optimal)
    
    # UCB (c=2)
    print("  测试 UCB (c=2)...")
    rewards, optimal = run_experiment(
        UCBAgent, {'c': 2.0},
        n_arms=10, n_steps=1000, n_runs=2000
    )
    results['UCB (c=2)'] = (rewards, optimal)
    
    # Thompson Sampling
    print("  测试 Thompson Sampling...")
    rewards, optimal = run_experiment(
        ThompsonSamplingGaussian, {},
        n_arms=10, n_steps=1000, n_runs=2000
    )
    results['Thompson Sampling'] = (rewards, optimal)
    
    # 绘制对比图
    plot_comparison(results)
    
    # 输出统计
    print()
    print("=" * 60)
    print("结果分析")
    print("=" * 60)
    print()
    print(f"{'策略':<25} {'后100步平均奖励':>15} {'最优动作比例':>15}")
    print("-" * 60)
    for label, (rewards, optimal) in results.items():
        avg_reward = rewards[-100:].mean()
        opt_pct = optimal[-100:].mean() * 100
        print(f"{label:<25} {avg_reward:>15.3f} {opt_pct:>14.1f}%")
    
    # 可视化后验分布
    print()
    print("可视化后验分布演变...")
    env = BanditEnv(n_arms=5, seed=100)
    agent = ThompsonSamplingGaussian(n_arms=5)
    
    for t in range(100):
        action = agent.select_action()
        reward = env.pull(action)
        agent.update(action, reward)
        
        if t in [0, 9, 99]:
            visualize_posterior(agent, env, t + 1)
    
    print("后验分布图已保存到 images/bandit_thompson_posterior_step*.png")
    
    print()
    print("=" * 60)
    print("关键发现")
    print("=" * 60)
    print()
    print("1. Thompson Sampling 与 UCB 性能接近，都是理论最优策略")
    print("2. Thompson Sampling 的优势：")
    print("   - 不需要调节参数（UCB 需要调 c）")
    print("   - 自然地实现概率匹配")
    print("   - 容易扩展到其他奖励分布")
    print("3. 贝叶斯方法的核心：")
    print("   - 用分布而非点估计表示不确定性")
    print("   - 通过采样实现探索")
    print()
    print("下一节：从多臂老虎机到完整 MDP —— 引入状态！")
