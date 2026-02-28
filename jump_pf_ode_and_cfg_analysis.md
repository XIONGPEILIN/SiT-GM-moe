# 混合状态(Mixed Mode)彩色噪声分析与 PF-ODE 解决方案

本报告详细记录了在 SiT-GM-moe 框架下，针对 Jump-Flow 混合模型采样时出现的“彩色噪声”问题的诊断、数学理论分析、以及结合 Classifier-Free Guidance (CFG) 的最终代码落地实现。

## 1. 核心问题：数学证明为何原始的 Jump 会造成“彩色噪声”？

在原始代码中，Jump 过程在采样（`transport.py` 的 `sample_jump_flow`）时严格遵循了连续时间马尔可夫链（CTMC）的逻辑：
- 掷硬币决定是否跳跃：$m_i \sim \text{Bernoulli}(p_{\text{jump}, i})$
- 如果跳跃，多项式分布采样跳向何处：$y_i \sim \text{Multinomial}(J_{\theta, i})$

这里 $i$ 表示 VAE 潜空间图 $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$ 中的一个特定的空间像素/通道位置。

### 数学证明：空间相关性 (Spatial Correlation) 的崩塌

VAE 的潜空间 $\mathbf{X}$ 是用来表征一张连续、平滑的自然图像的。对于自然图像的潜空间表达，相邻的潜在像素 $i$ 和 $j$ 之间必须具有极其强烈的**正向空间协方差 (Spatial Covariance)**，即：
$$ \text{Cov}(X_i, X_j) \gg 0 \quad (\text{当 } ||i-j|| \text{ 很小时}) $$

现在我们来看原始离散 Jump 采样对这个空间相关性的破坏。
在极小时间步 $dt$ 内，系统的新状态 $X^{t+dt}_i$ 可以写作：
$$ X^{t+dt}_i = m_i \cdot y_i + (1 - m_i) \cdot X^t_i $$

因为我们在每一个空间维度 $i$ 上都是**独立地**进行 Bernoulli 和 Multinomial 采样的（$m_i \perp m_j$, $y_i \perp y_j$），所以对于发生跳跃的像素点，它们的新值协方差为：
$$ \text{Cov}(m_i y_i, m_j y_j) = \mathbb{E}[m_i m_j y_i y_j] - \mathbb{E}[m_i y_i]\mathbb{E}[m_j y_j] $$

由于各个维度的独立采样假设，上述协方差直接等于 $0$：
$$ \text{Cov}(X^{t+dt}_i, X^{t+dt}_j) \approx 0 \quad (\text{对于所有发生跳跃的相邻像素}) $$

**物理意义与视觉后果：**
协方差为 $0$ 在信号处理中意味着**自相关截面为脉冲函数 (Delta function)**，这正是**高斯白噪声 (White Noise)** 的数学定义。
换句话说，连续的自然画面本来具有极平滑的空间梯度。但原版的离散 Jump 采样在全图随机选择了部分像素，将它们独立地“瞬移”到了互不相关的坐标上。
当 VAE 解码器（Decoder）试图将这组空间上完全失去相关性的高频信号（白噪声）翻译回 RGB 空间时，由于解码器使用卷积核具有感受野拓展性，这些孤立的高频潜空间噪点被扩散放大，最终就变成了人眼看到的**五颜六色的彩色斑块 (Colorful Artifacts/Noise)**。

---

## 2. 数学破局：Jump 过程的 Probability Flow ODE 

为了在使用连续 VAE 潜空间的前提下解决上述的高频采样方差（协方差坍缩），同时**保证 Generator Matching (GM) 理论的严格合法性**，我们需要寻找离散 Jump 对应的 **Probability Flow ODE (PF-ODE)** 等效形式。

### 理论推导 (基于 Kramers-Moyal 截断)

KFE (Kolmogorov Forward Equation) 是 Generator Matching 理论中边缘分布 $p_t$ 演化的支配方程：
$$ \partial_t p_t = \mathcal{L}_t^*(p_t) $$

Jump Generator 定义为：
$$ \mathcal{L}_t^{\text{jump}} f(x) = \sum_{y} [f(y) - f(x)] \lambda_t(x) J_t(y|x) $$

根据 Kramers-Moyal 展开定理，马尔可夫生成器可以被展开为无限阶的偏微分算子。如果我们只保留其中决定确定性漂移的一阶微分项，将它作为连续空间的**近似流场**，我们就能得到 Jump Generator 对应的决定论速度场 $v_{\text{jump}}$：
$$ v_{\text{jump}}(x) = \sum_y (y-x) \lambda_t(x) J_t(y|x) = \lambda_t(x) \cdot (\mathbb{E}_{y \sim J_t}[y] - x) $$

### PF-ODE 代码实现 (Dual-Velocity Field)

在 `transport.py` 的 MIXED 采样模式中，混合采样变为了**两个欧拉速度场**的平滑积分（消除了强掷硬币）：
1. 计算分类概率重心的目标连续坐标：`expected_y = (J_theta * y_bins).sum(dim=-1)`
2. 用连续变分逼近跃迁发生概率近似：`p_jump = 1 - exp(-\lambda_t dt)`
3. 等效流积分：`x = x + p_jump * (expected_y - x)`

在此模式下，相邻像素 $i$ 和 $j$ 的期望 $E[y_i]$ 和 $E[y_j]$ 高度相关（因为神经网络输出的 $J_\theta$ 在空间上是平滑关联的）。因此 $\text{Cov}(X_i, X_j)$ 完美得以保留，彻底根除了彩色白噪声。

---

## 3. $J_\theta$ 预测的必要性：保留多模态物理属性

在切换为连续 PF-ODE 积分后，模型不直接输出单维坐标期望，而是依然保留 $J_\theta$（128个分类维度的 logits 输出），主要基于两点：

1. **捕捉多模态分布 (Multi-modal Distribution)**
   $J_\theta$ 使得模型能够保留不同跳跃目的地的**概率拓扑结构**（例如：猫的特征占 40%，狗的特征占 60%）。如果模型底层强行输出单一回归期望，物理意义上的多模态分布（峰值）就会坍缩。
2. **作为 CFG 引导的精确支点**
   只有在这个拥有 128 个显式维度的离散分布上执行运算，CFG 才能发挥“如同语言模型一般的精确控制力”。通过剥离极微小的无效类，推高目标类，精确地重塑后验概率密度。

---

## 4. Classifier-Free Guidance (CFG) 的分体式引导

我们对 `models.py` 的推断函数做了改良，对 Jump 过程的 CFG 采用了**分体施加法（Partial CFG Activation）**。

### 为什么不对跳跃强度 $\lambda_t$ 施加 CFG？
因为对 $\lambda_t$ 这类物理标量施加常规线性推导 $\lambda_t^{\text{CFG}} = \lambda_t^{\text{u}} + w(\lambda_t^{\text{c}} - \lambda_t^{\text{u}})$ 会导致：
- **物理破裂**：$\lambda_t$ 可能演化为负数（速率不能为负），进而导致欧拉积分方程崩溃。
- **轨迹破损**：会强制要求每一个极小帧在全图引发无意义的无限跳变（退化）。

### 现在的实现方案及其优势
- **对 $J_t$ 的 Logits 施加 CFG**：`half_J = uncond_J + cfg_scale * (cond_J - uncond_J)`
- **不对 $\lambda_t$ 施加 CFG**：保留条件原本的跳跃倾向（Intensity）。

在这样的架构下：
1. **`models.py` 内部**：完成了对 $J_\theta$ 概率面貌的 CFG “多峰重塑”。
2. **`transport.py` 运行时**：后端的 `(J_theta * y_bins).sum()` 直接接管带有强引导势能的分布，并无噪地汇入 PF-ODE 积分。
