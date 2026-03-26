"""
mdp_gridworld.py - 网格世界：从老虎机到马尔可夫决策过程

这是强化学习的第一个"有状态"问题：
- 引入"状态"的概念：智能体在不同位置
- 动作会影响状态转移，而不只是产生奖励
- 目标：从起点走到终点，避开障碍物

学习要点：
1. MDP 的五个元素：<S, A, P, R, γ>
   - S: 状态空间
   - A: 动作空间
   - P: 转移概率
   - R: 奖励函数
   - γ: 折扣因子
2. 与多臂老虎机的关键区别：状态转移
3. 折扣因子的作用：平衡即时奖励与长期奖励
"""

import numpy as np
import matplotlib.pyplot as plt


class GridWorld:
    """
    网格世界环境
    
    简单的 4x4 网格：
    - 起点：(0, 0)，奖励 0
    - 终点：(3, 3)，奖励 +1
    - 障碍物：(1, 1), (2, 2)，奖励 -1
    - 其他：奖励 -0.1（鼓励快速到达终点）
    
    动作：
    - 0: 上
    - 1: 下
    - 2: 左
    - 3: 右
    """
    
    def __init__(self, size=4, stochastic=False, slip_prob=0.2, seed=42):
        """
        Args:
            size: 网格大小
            stochastic: 是否使用随机转移
            slip_prob: 随机转移时的滑倒概率
            seed: 随机种子
        """
        self.size = size
        self.stochastic = stochastic
        self.slip_prob = slip_prob
        self.rng = np.random.default_rng(seed)
        
        # 定义特殊位置
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles = [(1, 1), (2, 2)]
        
        # 动作空间：上、下、左、右
        self.action_names = ['Up', 'Down', 'Left', 'Right']
        self.n_actions = 4
        
        # 构建状态空间
        self.n_states = size * size
        
        # 当前状态
        self.state = None
        self.reset()
    
    def reset(self):
        """重置环境到起点"""
        self.state = self.start
        return self.state
    
    def state_to_idx(self, state):
        """将坐标转换为状态索引"""
        return state[0] * self.size + state[1]
    
    def idx_to_state(self, idx):
        """将状态索引转换为坐标"""
        return (idx // self.size, idx % self.size)
    
    def get_transition_prob(self, state, action, next_state):
        """
        获取转移概率 P(s' | s, a)
        
        这是 MDP 的核心：状态转移的概率模型
        """
        if self.stochastic:
            # 随机环境：有概率滑倒到其他方向
            probs = self._get_stochastic_probs(state, action)
            return probs.get(next_state, 0)
        else:
            # 确定性环境：100% 转移到预期位置
            expected_next = self._get_next_state(state, action)
            return 1.0 if next_state == expected_next else 0
    
    def _get_next_state(self, state, action):
        """获取执行动作后的下一个状态（确定性转移）"""
        row, col = state
        
        if action == 0:  # Up
            next_row = max(0, row - 1)
            next_col = col
        elif action == 1:  # Down
            next_row = min(self.size - 1, row + 1)
            next_col = col
        elif action == 2:  # Left
            next_row = row
            next_col = max(0, col - 1)
        else:  # Right
            next_row = row
            next_col = min(self.size - 1, col + 1)
        
        next_state = (next_row, next_col)
        
        # 如果撞到障碍物，留在原地
        if next_state in self.obstacles:
            return state
        
        return next_state
    
    def _get_stochastic_probs(self, state, action):
        """获取随机转移的概率分布"""
        # 期望动作
        expected_next = self._get_next_state(state, action)
        
        # 获取其他可能的动作
        other_actions = [a for a in range(self.n_actions) if a != action]
        
        probs = {expected_next: 1 - self.slip_prob}
        
        # 滑倒概率均分到其他方向
        slip_each = self.slip_prob / len(other_actions)
        for a in other_actions:
            next_s = self._get_next_state(state, a)
            probs[next_s] = probs.get(next_s, 0) + slip_each
        
        return probs
    
    def get_reward(self, state, next_state):
        """
        获取奖励 R(s, s')
        
        奖励设计：
        - 到达终点：+1
        - 撞到障碍物：-1
        - 每步小惩罚：-0.1（鼓励效率）
        """
        if next_state == self.goal:
            return 1.0
        elif next_state in self.obstacles:
            return -1.0
        else:
            return -0.1
    
    def step(self, action):
        """
        执行动作，返回 (next_state, reward, done)
        
        这是 gym/gymnasium 的标准接口
        """
        if self.stochastic:
            # 随机转移
            probs = self._get_stochastic_probs(self.state, action)
            states = list(probs.keys())
            p_values = list(probs.values())
            # 随机选择一个状态索引，然后获取对应的状态
            idx = self.rng.choice(len(states), p=p_values)
            next_state = states[idx]
        else:
            # 确定性转移
            next_state = self._get_next_state(self.state, action)
        
        reward = self.get_reward(self.state, next_state)
        done = (next_state == self.goal)
        
        self.state = next_state
        
        return next_state, reward, done
    
    def render(self):
        """打印当前网格状态"""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        
        # 标记特殊位置
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'
        grid[self.goal[0]][self.goal[1]] = 'G'
        grid[self.start[0]][self.start[1]] = 'S'
        
        # 标记当前位置
        if self.state != self.start and self.state != self.goal:
            grid[self.state[0]][self.state[1]] = 'A'
        
        print("\n".join([" ".join(row) for row in grid]))
    
    def get_mdp_info(self):
        """返回 MDP 的完整信息（用于理解和调试）"""
        info = {
            '状态空间大小': self.n_states,
            '动作空间大小': self.n_actions,
            '状态空间': f'0 到 {self.n_states - 1}',
            '动作空间': self.action_names,
            '折扣因子': 'γ (discount factor，稍后介绍)',
            '转移概率': 'P(s\'|s,a) - 可以是确定性或随机的',
            '奖励函数': 'R(s,a,s\') - 到达终点 +1，撞障碍 -1，每步 -0.1'
        }
        return info


class RandomAgent:
    """
    随机策略智能体
    
    策略：等概率选择每个动作
    
    这是基准策略，用于对比后续更智能的策略
    """
    
    def __init__(self, n_actions, seed=42):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
    
    def select_action(self, state):
        """随机选择动作"""
        return self.rng.integers(0, self.n_actions)
    
    def update(self, state, action, reward, next_state):
        """随机策略不需要学习"""
        pass


def run_episode(env, agent, max_steps=100, verbose=False):
    """
    运行一个回合
    
    Returns:
        total_reward: 累计奖励
        steps: 使用步数
        trajectory: 轨迹
    """
    state = env.reset()
    total_reward = 0
    trajectory = [state]
    
    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        
        agent.update(state, action, reward, next_state)
        
        total_reward += reward
        trajectory.append(next_state)
        
        if verbose:
            print(f"\nStep {step + 1}: {env.action_names[action]}")
            env.render()
            print(f"Reward: {reward:.2f}, Total: {total_reward:.2f}")
        
        state = next_state
        
        if done:
            break
    
    return total_reward, step + 1, trajectory


def visualize_grid(env, trajectory=None, save_path=None):
    """可视化网格世界和可选的轨迹"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 绘制网格
    for i in range(env.size + 1):
        ax.axhline(i, color='gray', linewidth=0.5)
        ax.axvline(i, color='gray', linewidth=0.5)
    
    # 绘制特殊位置
    for obs in env.obstacles:
        ax.add_patch(plt.Rectangle((obs[1], env.size - 1 - obs[0]), 
                                    1, 1, color='red', alpha=0.5))
        ax.text(obs[1] + 0.5, env.size - 1 - obs[0] + 0.5, 'X', 
                ha='center', va='center', fontsize=20, fontweight='bold')
    
    # 起点和终点
    ax.add_patch(plt.Rectangle((env.start[1], env.size - 1 - env.start[0]), 
                                1, 1, color='blue', alpha=0.3))
    ax.text(env.start[1] + 0.5, env.size - 1 - env.start[0] + 0.5, 'S', 
            ha='center', va='center', fontsize=20, fontweight='bold', color='blue')
    
    ax.add_patch(plt.Rectangle((env.goal[1], env.size - 1 - env.goal[0]), 
                                1, 1, color='green', alpha=0.3))
    ax.text(env.goal[1] + 0.5, env.size - 1 - env.goal[0] + 0.5, 'G', 
            ha='center', va='center', fontsize=20, fontweight='bold', color='green')
    
    # 绘制轨迹
    if trajectory:
        rows = [env.size - 1 - s[0] for s in trajectory]
        cols = [s[1] + 0.5 for s in trajectory]
        ax.plot(cols, rows, 'b-o', linewidth=2, markersize=8, alpha=0.7)
        
        # 标记起点
        ax.plot(cols[0] + 0.5, rows[0], 'go', markersize=15, label='Start')
        # 标记终点
        ax.plot(cols[-1] + 0.5, rows[-1], 'ro', markersize=15, label='End')
    
    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_aspect('equal')
    ax.set_title('GridWorld: Find the Goal!', fontsize=14)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    
    plt.close()


def demonstrate_mdp_elements():
    """演示 MDP 的五个元素"""
    env = GridWorld(size=4, stochastic=True, slip_prob=0.2)
    
    print("\n" + "=" * 60)
    print("MDP 的五个元素")
    print("=" * 60)
    
    # 1. 状态空间 S
    print("\n1. 状态空间 S:")
    print(f"   所有可能的状态：{env.n_states} 个")
    print(f"   状态表示：，范围 0-{env.size-1}")
    print(f"   示例状态：(0,0)=起点, (3,3)=终点")
    
    # 2. 动作空间 A
    print("\n2. 动作空间 A:")
    print(f"   所有可选的动作：{env.action_names}")
    print(f"   动作数量：{env.n_actions}")
    
    # 3. 转移概率 P
    print("\n3. 转移概率 P(s'|s,a):")
    state = (0, 1)
    action = 3  # Right
    print(f"   当前状态：{state}, 动作：{env.action_names[action]}")
    print(f"   转移概率分布：")
    
    for next_row in range(env.size):
        for next_col in range(env.size):
            next_state = (next_row, next_col)
            prob = env.get_transition_prob(state, action, next_state)
            if prob > 0:
                print(f"     P({next_state} | {state}, {env.action_names[action]}) = {prob:.2f}")
    
    # 4. 奖励函数 R
    print("\n4. 奖励函数 R(s,a,s'):")
    print("   到达终点 (3,3)：+1.0")
    print("   撞到障碍物：-1.0")
    print("   每步移动：-0.1 (鼓励效率)")
    
    # 5. 折扣因子 γ
    print("\n5. 折扣因子 γ:")
    print("   作用：平衡即时奖励与未来奖励")
    print("   γ = 0: 只关心当前奖励")
    print("   γ = 1: 未来奖励和当前奖励同等重要")
    print("   γ = 0.9: 未来奖励的权重随时间衰减 (常用)")
    
    return env


def compare_deterministic_vs_stochastic():
    """比较确定性环境 vs 随机环境"""
    print("\n" + "=" * 60)
    print("确定性环境 vs 随机环境")
    print("=" * 60)
    
    n_runs = 1000
    max_steps = 50
    
    # 确定性环境
    env_det = GridWorld(stochastic=False)
    agent = RandomAgent(env_det.n_actions)
    
    det_success = 0
    det_rewards = []
    det_steps = []
    
    for _ in range(n_runs):
        reward, steps, _ = run_episode(env_det, agent, max_steps)
        det_rewards.append(reward)
        det_steps.append(steps)
        if steps < max_steps:
            det_success += 1
    
    # 随机环境
    env_sto = GridWorld(stochastic=True, slip_prob=0.2)
    
    sto_success = 0
    sto_rewards = []
    sto_steps = []
    
    for _ in range(n_runs):
        reward, steps, _ = run_episode(env_sto, agent, max_steps)
        sto_rewards.append(reward)
        sto_steps.append(steps)
        if steps < max_steps:
            sto_success += 1
    
    print(f"\n确定性环境 (1000 次随机策略):")
    print(f"  到达终点的比例: {det_success / n_runs * 100:.1f}%")
    print(f"  平均累计奖励: {np.mean(det_rewards):.2f}")
    print(f"  平均步数: {np.mean(det_steps):.1f}")
    
    print(f"\n随机环境 (slip_prob=0.2):")
    print(f"  到达终点的比例: {sto_success / n_runs * 100:.1f}%")
    print(f"  平均累计奖励: {np.mean(sto_rewards):.2f}")
    print(f"  平均步数: {np.mean(sto_steps):.1f}")
    
    print("\n观察：")
    print("  - 随机环境下更难到达终点（滑倒概率增加了不确定性）")
    print("  - 这就是为什么需要智能的策略！")


if __name__ == "__main__":
    print("=" * 60)
    print("网格世界 - MDP 入门")
    print("=" * 60)
    print()
    print("从多臂老虎机到网格世界，最大的变化是：")
    print("  多臂老虎机：没有状态，只有动作和奖励")
    print("  网格世界：有状态，动作会改变状态")
    print()
    
    # 演示 MDP 元素
    env = demonstrate_mdp_elements()
    
    # 比较确定性 vs 随机环境
    compare_deterministic_vs_stochastic()
    
    # 可视化
    print("\n" + "=" * 60)
    print("可视化网格世界")
    print("=" * 60)
    
    env_vis = GridWorld(stochastic=False)
    agent = RandomAgent(env_vis.n_actions)
    
    # 运行一个回合
    total_reward, steps, trajectory = run_episode(env_vis, agent, max_steps=50, verbose=False)
    
    print(f"\n随机策略执行结果：")
    print(f"  步数: {steps}")
    print(f"  累计奖励: {total_reward:.2f}")
    print(f"  是否到达终点: {'是' if steps < 50 else '否'}")
    
    # 绘制网格和轨迹
    visualize_grid(env_vis, trajectory, save_path='images/gridworld_demo.png')
    
    print("\n" + "=" * 60)
    print("下一步：如何找到最优策略？")
    print("=" * 60)
    print()
    print("随机策略显然不是最优的。")
    print("下一节将介绍：")
    print("  1. 值迭代 - 动态规划方法")
    print("  2. 策略迭代 - 另一种 DP 方法")
    print()
    print("关键思想：")
    print("  - 如果我们知道 MDP 的完整信息（P, R）")
    print("  - 就可以通过动态规划计算最优策略")
