# Thompson Sampling 深度理解笔记

## 目录
1. [核心概念](#核心概念)
2. [先验与后验分布](#先验与后验分布)
3. [精度 (Precision) 的理解](#精度-precision-的理解)
4. [贝叶斯共轭性质](#贝叶斯共轭性质)
5. [算法执行流程](#算法执行流程)
6. [代码实现分析](#代码实现分析)
7. [与其他算法对比](#与其他算法对比)

---

## 核心概念

### Thompson Sampling 是什么？

Thompson Sampling 是一种**概率性的探索-利用平衡策略**，通过以下机制实现：

1. **维护后验分布**：对每个臂的奖励分布保持一个贝叶斯信念
2. **采样与选择**：从后验分布采样，选择采样值最大的臂
3. **更新信念**：根据实际奖励更新该臂的后验分布
4. **循环迭代**：不断精化对所有臂的理解

```
┌─────────────────────────────────────┐
│  第 t 步：Thompson Sampling Loop    │
├─────────────────────────────────────┤
│ ① 维护所有臂的后验分布               │
│    N(μ_i, σ_i²), i=1,2,...,k      │
│                                     │
│ ② 从每个臂的后验分布采样一个值       │
│    sample_i ~ N(μ_i, σ_i²)        │
│                                     │
│ ③ 选择采样值最大的臂                │
│    a* = argmax(sample_i)           │
│                                     │
│ ④ 执行臂 a*，获得真实奖励 r       │
│                                     │
│ ⑤ 只更新臂 a* 的后验分布            │
│    根据新的奖励数据                  │
│                                     │
│ ⑥ 返回第 ① 步                      │
└─────────────────────────────────────┘
```

---

## 先验与后验分布

### 先验分布 (Prior)

**定义**：在看到**任何数据之前**对参数的信念。

**代码体现**：
```python
# 初始化时的先验
self.posterior_mean = np.full(n_arms, prior_mean=0, dtype=float)
self.posterior_var = np.full(n_arms, prior_var=1, dtype=float)
```

**直观理解**：
- 你走进一个赌场，面对 10 台老虎机
- 你没有任何关于这些机器的信息
- 你的先验假设："所有机器都差不多，我不确定"
- 用数学表示：**N(0, 1)** - 均值为 0，方差为 1 的高斯分布

| 参数 | 含义 | 初始值 |
|------|------|--------|
| **μ₀** | 先验均值 | 0 |
| **σ₀²** | 先验方差 | 1 |
| **τ₀** | 先验精度 | 1/1 = 1.0 |

### 后验分布 (Posterior)

**定义**：看到实际数据后，**更新的信念**。

**代码体现**：
```python
def update(self, arm, reward):
    self.counts[arm] += 1
    self.sum_rewards[arm] += reward
    
    n = self.counts[arm]
    
    # 后验精度 = 先验精度 + 数据精度
    posterior_precision = prior_precision + data_precision
    self.posterior_var[arm] = 1.0 / posterior_precision
    
    # 后验均值 = 精度加权平均
    self.posterior_mean[arm] = (
        prior_precision * prior_mean + self.sum_rewards[arm] / self.reward_var
    ) / posterior_precision
```

### 先验 → 后验的演变

```
第0步 (初始):
  先验: N(0, 1)        # 完全不确定  
  
拉一次，得到 +2:
  后验1: N(0.5, 0.5)   # 往上调，确定度提升  
  
再拉一次，得到 +1.5:
  后验2: N(0.875, 0.33) # 继续逼近真值  
  
拉十次，平均 +1.2:
  后验10: N(1.2, 0.09) # 已经很确定这个值了  

趋势:
  均值 μ:   0.0 → 0.5 → 0.875 → ... → 1.2 (逼近真实值)  
  方差 σ²:  1.0 → 0.5 → 0.33  → ... → 0.09 (越来越小)  
  精度 τ:   1.0 → 2.0 → 3.0   → ... → 11.0 (不断增长)  
```

---

## 精度 (Precision) 的理解

### 精度 ≠ 准确度

**精度的数学定义**：
$$\tau = \frac{1}{\sigma^2}$$

- **σ²（方差）**：数据的分散程度
- **τ（精度）**：信息的集中程度（方差的倒数）

### 直观对比

| 概念 | 符号 | 含义 | 值大 | 含义 |
|------|------|------|------|------|
| 方差 | σ² | 分散程度 | σ² ↑ | 越分散，越不确定 |
| 精度 | τ | 集中程度 | τ ↑ | 越集中，越确定 |

### 数值例子

```python
# 例子1：高置信度
σ² = 0.01  →  τ = 1/0.01 = 100   # 非常确定！

# 例子2：中等置信度
σ² = 1.0   →  τ = 1/1.0 = 1.0    # 中等不确定性

# 例子3：低置信度
σ² = 100   →  τ = 1/100 = 0.01   # 非常不确定
```

### 精度的信息论含义

**关键洞察**：**精度 ∝ 信息量**

在贝叶斯推断中：
- 高精度 = 高置信度信息 = 能显著改变我的信念
- 低精度 = 低置信度信息 = 对我的信念影响小

---

## 贝叶斯共轭性质

### 为什么用精度形式？

**传统方差形式**（复杂）：
```
σ_n² = (σ₀² · σ²) / (σ₀² + n·σ²)
```

**精度形式**（优雅）：
```
τ_n = τ₀ + n·τ  ✨ 直接相加！
```

### 完整推导过程

#### 第一步：高斯分布的两种形式

**标准形式**：
$$p(x|\mu) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**精度形式**（令 $\tau = 1/\sigma^2$）：
$$p(x|\mu) = \sqrt{\frac{\tau}{2\pi}} \exp\left(-\frac{\tau}{2}(x-\mu)^2\right)$$

#### 第二步：贝叶斯三要素

**① 先验 Prior**：
$$p(\mu) = \mathcal{N}(\mu_0, \sigma_0^2)$$

**② 似然 Likelihood**（n次观测）：
$$p(D|\mu) \propto \exp\left(-\frac{\tau}{2}\sum_{i=1}^{n}(x_i-\mu)^2\right)$$

**③ 后验 Posterior**（贝叶斯公式）：
$$p(\mu|D) \propto p(D|\mu) \cdot p(\mu)$$

#### 第三步：乘法合并

$$p(\mu|D) \propto \exp\left(-\frac{\tau}{2}\sum_{i=1}^{n}(x_i-\mu)^2 - \frac{\tau_0}{2}(\mu-\mu_0)^2\right)$$

#### 第四步：配方求后验

整理 μ 的系数：

$$\text{后验精度}: \quad \tau_n = \tau_0 + n\tau$$

$$\text{后验方差}: \quad \sigma_n^2 = \frac{1}{\tau_n}$$

$$\text{后验均值}: \quad \mu_n = \frac{\tau_0\mu_0 + \tau \sum_{i=1}^n x_i}{\tau_0 + n\tau}$$

### 共轭性的威力

```python
# 这就是为什么高斯+高斯是共轭的

# 第 n 次观测前
posterior_precision = prior_precision + (n-1) * data_precision
posterior_var = 1.0 / posterior_precision

# 第 n 次观测后（直接相加！）
posterior_precision_new = prior_precision + n * data_precision
posterior_var_new = 1.0 / posterior_precision_new

# 结果：O(1) 时间复杂度的更新！
```

### 代码实现对应推导

```python
# 精度定义
prior_precision = 1.0 / reward_var        # τ₀ = 1/σ²
data_precision = n / reward_var            # n·τ（n次观测的精度累积）

# 后验精度：τ_n = τ₀ + n·τ
posterior_precision = prior_precision + data_precision

# 后验方差：σ²_n = 1/τ_n
self.posterior_var[arm] = 1.0 / posterior_precision

# 后验均值：μ_n = (τ₀μ₀ + τ·Σx) / τ_n
self.posterior_mean[arm] = (
    prior_precision * prior_mean + self.sum_rewards[arm] / reward_var
) / posterior_precision
```

---

## 算法执行流程

### 伪代码

```
算法：Thompson Sampling for Gaussian Bandits

初始化：
  对每个臂 i:
    μ_i ← 0, σ²_i ← 1, count_i ← 0, sum_reward_i ← 0

循环 t = 1 to T:
  采样：
    对每个臂 i:
      sample_i ~ N(μ_i, σ²_i)
  
  选择：
    a* ← argmax_i(sample_i)
  
  执行与观测：
    reward ← 从臂 a* 获得奖励
  
  更新：
    count_{a*} ← count_{a*} + 1
    sum_reward_{a*} ← sum_reward_{a*} + reward
    n ← count_{a*}
    
    τ_n ← 1/σ²_noise + n/σ²_noise
    σ²_{a*} ← 1/τ_n
    μ_{a*} ← sum_reward_{a*} / τ_n
```

### 探索-利用的自然平衡

**为什么采样少的臂也会被选中？**

```
臂A（被拉过100次）:
  后验: N(μ=5.0, σ²=0.01)
  采样: 几乎总是 4.8~5.2 范围内
  
臂B（从未被拉过）:
  后验: N(μ=0.0, σ²=1.0)
  采样: 可能是 -3, -1, +2, +4, ... （波动大！）
  
结果：
  虽然 μ_A > μ_B
  但 σ_B 大，sample_B 有可能采样到很大的值
  → 有机会赢过 sample_A
  → 臂B被选中
  → 获得新数据更新臂B的分布
  → 臂B的σ逐渐缩小 ✨
```

**为什么优良臂越来越被选中？**

```
随着迭代增多：

臂A（真实最优）:
  σ²_A 不断缩小 → N(5.2, 0.001)
  采样集中在 5.1~5.3（总是最大）
  
臂B（较差）:
  σ²_B 也缩小，但 μ_B 止步于 2.5
  采样集中在 2.3~2.7（远小于A）
  
结果：
  臂A 被选中的概率 → 1（几乎确定性利用）
  最终收敛到最优臂！
```

---

## 代码实现分析

### 关键类：ThompsonSamplingGaussian

```python
class ThompsonSamplingGaussian:
    """Thompson Sampling with Gaussian Rewards"""
    
    def __init__(self, n_arms, prior_mean=0, prior_var=1, reward_var=1):
        self.n_arms = n_arms
        self.reward_var = reward_var  # 已知的奖励噪声方差
        
        # 后验分布参数
        self.posterior_mean = np.full(n_arms, prior_mean, dtype=float)
        self.posterior_var = np.full(n_arms, prior_var, dtype=float)
        
        # 累计统计
        self.sum_rewards = np.zeros(n_arms)
        self.counts = np.zeros(n_arms, dtype=int)
        
        self.rng = np.random.default_rng()
```

**成员变量含义**：

| 变量 | 形状 | 含义 |
|------|------|------|
| `posterior_mean` | (n_arms,) | 每个臂的后验均值 μ_i |
| `posterior_var` | (n_arms,) | 每个臂的后验方差 σ²_i |
| `sum_rewards` | (n_arms,) | 每个臂累计获得的奖励 |
| `counts` | (n_arms,) | 每个臂被拉过的次数 |

### select_action() - 采样与选择

```python
def select_action(self):
    """从每个臂的后验分布中采样，选择采样值最大的臂"""
    # 从后验分布采样
    samples = self.rng.normal(
        self.posterior_mean,              # μ_i
        np.sqrt(self.posterior_var)       # σ_i
    )
    return np.argmax(samples)             # 选择最大的
```

**关键点**：
- 从 **N(μ_i, σ²_i)** 采样，不是用点估计 μ_i
- 高方差 → 采样波动大 → 被选中的机会大 → 探索
- 低方差 → 采样值稳定 → 大概率是最大 → 利用

### update() - 后验更新

```python
def update(self, arm, reward):
    """更新选中臂的后验分布"""
    self.counts[arm] += 1
    self.sum_rewards[arm] += reward
    
    n = self.counts[arm]
    
    # 精度形式的优雅更新
    prior_precision = 1.0 / self.reward_var
    data_precision = n / self.reward_var
    
    posterior_precision = prior_precision + data_precision
    self.posterior_var[arm] = 1.0 / posterior_precision
    
    # 精度加权平均
    self.posterior_mean[arm] = (
        prior_precision * 0 + self.sum_rewards[arm] / self.reward_var
    ) / posterior_precision
```

**更新步骤**：
1. 累计观测数据 (count + reward)
2. 计算后验精度（直接相加）
3. 计算后验方差（取精度的倒数）
4. 计算后验均值（精度加权平均）

---

## 与其他算法对比

### 性能对比表

| 算法 | 探索机制 | 超参数 | 更新复杂度 | 理论保证 | 实现难度 |
|------|---------|--------|-----------|---------|---------|
| **ε-greedy** | 随机 | ε | O(1) | 无 | 很简单 |
| **UCB** | 置信上界 | c | O(1) | 有 | 简单 |
| **Thompson Sampling** | 后验采样 | 无 | O(1) | 有 | 中等 |

### 关键优势

```python
# Thompson Sampling vs UCB

# UCB 需要手调参数
ucb_agent = UCBAgent(n_arms=10, c=2.0)  # c 是啥？多少合适？

# Thompson Sampling 无需调参
thompson_agent = ThompsonSamplingGaussian(n_arms=10)  # 开箱即用！
```

### 算法优劣分析

| 算法 | 优点 | 缺点 |
|------|------|------|
| **ε-greedy** | ✅ 简单直观 | ❌ 盲目探索，浪费 |
| **UCB** | ✅ 理论最优 | ❌ 需要调 c，不够自适应 |
| **Thompson Sampling** | ✅ 无参数 ✅ 自适应 ✅ 优雅 | ⚠️ 理解门槛高 |

---

## 总结与关键洞察

### 三个核心要素

1. **分布而非点估计**
   - 用 N(μ, σ²) 表示不确定性，而不是只记录 q_value
   - 方差本身就是探索的信号

2. **精度的直接相加**
   - τ_n = τ₀ + nτ（共轭性的威力）
   - 使后验更新极其高效：O(1) 时间

3. **采样的智能探索**
   - 从后验采样，不确定的臂波动大
   - 自然实现"好奇心"：想更了解不确定的东西

### Thompson Sampling 的哲学

> **不是盲目地随机探索，而是根据当前的信念智能地采样。**

```
信念强（σ小） → 采样集中 → 大概率选中 → 利用
信念弱（σ大）  → 采样波动 → 有机会被选 → 探索

这种"信心度与选择概率正相关"的机制，
就是 Thompson Sampling 的精妙之处！
```

### 扩展方向

- **非高斯奖励**：用贝塔分布处理伯努利奖励
- **上下文老虎机**：加入状态信息
- **多目标**：同时优化多个指标
- **实时学习**：在线流数据中的应用

---

## 参考资源

### 理论基础
- Agrawal & Goyal (2013): "Thompson Sampling for 1-Dimensional Exponential Family Bandits"
- Thompson (1933): 原始论文《On the Likelihood that One Unknown Probability Exceeds Another in the Light of the Evidence of Two Samples》

### 相关文件
- `bandit_thompson.py`：实现代码
- `bandit_ucb.py`：UCB 对比
- `bandit_epsilon_greedy.py`：ε-greedy 对比

---

**最后更新**：2026-03-15  
**作者**：学习笔记  
**难度等级**：⭐⭐⭐ (中等偏难)
