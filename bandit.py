"""
bandit.py - 多臂老虎机：强化学习的 Hello World

这是强化学习最简单的问题：
- 没有"状态"的概念，只有"动作"
- 每个动作对应一个固定的奖励分布
- 目标：找到平均奖励最高的那个臂

学习要点：
1. 什么是探索（Exploration）与利用（Exploitation）的困境
2. 如何评估一个策略的好坏
3. 遗憾（Regret）的概念
"""

import numpy as np
import matplotlib.pyplot as plt


class BanditEnv:
    """
    多臂老虎机环境
    
    每个臂（动作）对应一个固定的奖励分布
    这里使用正态分布，均值为 true_reward，标准差为 1
    """
    
    def __init__(self, n_arms=10, seed=42):
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)
        
        # 每个臂的真实奖励值（从标准正态分布中采样）
        self.true_rewards = self.rng.normal(0, 1, n_arms)
        
    def pull(self, arm):
        """
        拉动某个臂，返回奖励
        
        Args:
            arm: 选择哪个臂（0 到 n_arms-1）
        
        Returns:
            reward: 从该臂的奖励分布中采样的值
        """
        # 奖励服从 N(true_reward[arm], 1)
        return self.rng.normal(self.true_rewards[arm], 1)
    
    def optimal_arm(self):
        """返回最优的臂（用于计算遗憾）"""
        return np.argmax(self.true_rewards)
    
    def optimal_reward(self):
        """返回最优臂的真实奖励值"""
        return np.max(self.true_rewards)


class GreedyAgent:
    """
    贪婪智能体
    
    策略：总是选择当前估计奖励最高的臂
    
    问题：如果初始估计不准确，可能会永远选错臂
    """
    
    def __init__(self, n_arms, initial_value=0):
        self.n_arms = n_arms
        self.initial_value = initial_value
        
        # 每个臂的估计奖励值
        self.q_values = np.full(n_arms, initial_value, dtype=float)
        # 每个臂被拉动的次数
        self.counts = np.zeros(n_arms, dtype=int)
    
    def select_action(self):
        """选择当前估计奖励最高的臂"""
        return np.argmax(self.q_values)
    
    def update(self, arm, reward):
        """
        更新对某个臂的估计
        
        使用增量式更新公式：
        Q_n+1 = Q_n + (R_n - Q_n) / (n+1)
        """
        self.counts[arm] += 1
        # 增量式均值更新
        self.q_values[arm] += (reward - self.q_values[arm]) / self.counts[arm]


def run_episode(env, agent, n_steps=1000):
    """
    运行一个回合
    
    Returns:
        rewards: 每步获得的奖励
        optimal_actions: 每步是否选择了最优臂
    """
    rewards = np.zeros(n_steps)
    optimal_actions = np.zeros(n_steps, dtype=bool)
    optimal_arm = env.optimal_arm()
    
    for t in range(n_steps):
        # 1. 智能体选择动作
        action = agent.select_action()
        
        # 2. 环境返回奖励
        reward = env.pull(action)
        
        # 3. 智能体更新估计
        agent.update(action, reward)
        
        # 记录
        rewards[t] = reward
        optimal_actions[t] = (action == optimal_arm)
    
    return rewards, optimal_actions


def run_experiment(n_arms=10, n_steps=1000, n_runs=2000, seed=42):
    """
    运行多次实验取平均
    
    这样可以消除单次实验的随机性
    """
    all_rewards = np.zeros((n_runs, n_steps))
    all_optimal = np.zeros((n_runs, n_steps), dtype=bool)
    
    for run in range(n_runs):
        # 每次运行使用不同的随机种子
        env = BanditEnv(n_arms, seed=seed + run)
        agent = GreedyAgent(n_arms)
        
        rewards, optimal = run_episode(env, agent, n_steps)
        all_rewards[run] = rewards
        all_optimal[run] = optimal
    
    return all_rewards.mean(axis=0), all_optimal.mean(axis=0)


def plot_results(avg_rewards, optimal_pct):
    """绘制实验结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 左图：平均奖励随时间变化
    axes[0].plot(avg_rewards)
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Greedy Agent: Average Reward')
    axes[0].grid(True, alpha=0.3)
    
    # 右图：选择最优臂的比例
    axes[1].plot(optimal_pct * 100)
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('% Optimal Action')
    axes[1].set_title('Greedy Agent: % Optimal Action')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('bandit_greedy_result.png', dpi=150)
    plt.close()
    print("结果已保存到 bandit_greedy_result.png")


if __name__ == "__main__":
    print("=" * 60)
    print("多臂老虎机 - 贪婪策略演示")
    print("=" * 60)
    print()
    print("问题描述：")
    print("  - 有 10 个老虎机臂，每个臂有不同的真实奖励值")
    print("  - 每次选择一个臂，获得一个随机奖励")
    print("  - 目标：最大化总奖励")
    print()
    print("贪婪策略：")
    print("  - 总是选择当前估计奖励最高的臂")
    print("  - 问题：初始估计不准，容易陷入次优选择")
    print()
    
    # 运行实验
    print("正在运行实验（2000次平均）...")
    avg_rewards, optimal_pct = run_experiment(n_arms=10, n_steps=1000, n_runs=2000)
    
    # 输出统计
    print(f"\n前100步平均奖励: {avg_rewards[:100].mean():.3f}")
    print(f"后100步平均奖励: {avg_rewards[-100:].mean():.3f}")
    print(f"最后100步选择最优臂的比例: {optimal_pct[-100:].mean()*100:.1f}%")
    
    # 绘图
    plot_results(avg_rewards, optimal_pct)
    
    print()
    print("=" * 60)
    print("思考：为什么贪婪策略效果不好？")
    print("=" * 60)
    print("因为贪婪策略从不探索未知的臂！")
    print("如果一开始某个次优臂运气好获得了高奖励，")
    print("贪婪策略就会一直选择它，错过真正最优的臂。")
    print()
    print("下一节：epsilon-greedy 策略如何解决这个问题？")
