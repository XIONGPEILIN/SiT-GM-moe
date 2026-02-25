# GM 严格索引版（逐行对照 `main.tex`）

## 核心约定：时间 t 的含义

本文档及所参考的论文 `main.tex` 遵循以下时间演化约定：
- **`t = 0`**: 对应 **纯噪声** (先验分布 `p_initial`)。
- **`t = 1`**: 对应 **纯数据** (目标数据分布 `p_data`)。

---

本文档是你要的“严格逐行对应 + 原文摘录版”，已修复公式显示问题。

## 说明

1. **公式显示修复**：将原始 LaTeX 转换为标准 Markdown Math (`$$`) 格式，并展开了自定义宏（如 `\pdata` -> `p_{data}`），确保在主流编辑器中正确渲染。
2. **双格式保留**：
    - **Rendered LaTeX**：可直接渲染的公式。
    - **ASCII-Canonical**：不依赖渲染器的规范写法（放在 `text` 代码块中）。
3. **严格对照**：保留 `Fxx` 编号与 `main.tex` 行号。

---

## A. 逐行公式索引（Formula Index）

### F01 路径分层采样定义
- Source lines: `main.tex:159-161`
- 用途：定义如何从条件路径得到边缘路径。

$$
z \sim p_{\text{data}}, x \sim p_t(dx|z) \quad \Rightarrow \quad x \sim p_t(dx)
$$

```text
z ~ p_data, x ~ p_t(.|z)  =>  x ~ p_t
```

### F02 Mixture 条件路径
- Source lines: `main.tex:180-183`
- 用途：路径构造模板 1。

$$
p_t(dx|z)=(1-\kappa_t) p_{\text{initial}}(dx) + \kappa_t \delta_{z}(dx)
$$

```text
p_t(.|z) = (1-kappa_t) p0 + kappa_t delta_z
```

### F03 Geometric-average 条件路径
- Source lines: `main.tex:184`
- 用途：路径构造模板 2。

$$
p_t(dx|z)= \mathbb{E}_{x_0} [ \delta_{\sigma_t x_0 + \alpha_t z} ] \Leftrightarrow x_t = \sigma_t x_0 + \alpha_t z
$$

```text
x_t = sigma_t * x_0 + alpha_t * z,  x_0 ~ p0
```

### F04 路径调度边界条件
- Source lines: `main.tex:186`
- 用途：保证 `t=0` 对应先验，`t=1` 对应数据。

$$
\kappa_0=\alpha_0=\sigma_1=0,\quad \kappa_1=\alpha_1=\sigma_0=1,\quad 0\leq \kappa_t\leq 1
$$

```text
kappa_0=0, kappa_1=1, alpha_0=0, alpha_1=1, sigma_0=1, sigma_1=0
```

### F05 KFE 定义
- Source lines: `main.tex:774-776`
- Label: `eq:kfe`
- 用途：路径正确性的约束方程。

$$
\partial_{t} \langle p_t, f \rangle = \langle p_t, \mathcal{L}_t f \rangle
$$

```text
d/dt <p_t, f> = <p_t, L_t f>
```

### F06 边缘生成器（条件生成器后验平均）
- Source lines: `main.tex:795-797`
- Label: `eq:marginal_generator_equation`

$$
\mathcal{L}_t f(x) = \mathbb{E}_{z \sim p_{1|t}(\cdot|x)} [ \mathcal{L}_t^z f(x) ]
$$

```text
L_t f(x) = E_{z ~ p(z|x,t)} [ L_t^z f(x) ]
```

### F07 `R^d` 统一生成器分解 (已修正)
- Source lines: `main.tex:631-638` (in `Table 1`)
- 用途: 将任意马尔科夫生成器分解为 flow, diffusion, jump 三个部分。

$$
\mathcal{L}_t f(x) = \nabla f(x)^T u_t(x) + \frac{1}{2} \text{Tr}(\nabla^2 f(x) \Sigma_t(x)) + \int \{f(y)-f(x)\} Q_t(dy;x)
$$

```text
L_t = flow + diffusion + jump
```
> **修正说明**: 原 `GM.md` 行号 `635-636` 索引不正确。此公式的完整定义位于 `main.tex` 的 `Table 1` (行号 631-638) 中。同时修正了扩散项，使用 `Tr` (Trace) 和协方差矩阵 `Sigma_t` 以匹配原文，比标量 `sigma^2` 更通用。

### F08 Bregman divergence
- Source lines: `main.tex:853-855`
- Label: `eq:bregman_divergences`

$$
D(a,b) = \phi(a) - [ \phi(b) + \langle a-b, \nabla \phi(b) \rangle ], \quad a,b \in \Omega
$$

```text
D(a,b) = phi(a) - phi(b) - <a-b, grad phi(b)>
```

### F09 GM loss（不可直接训练）
- Source lines: `main.tex:857-859`

$$
L_{\text{gm}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, x \sim p_t} [ D(F_t(x), F_t^\theta(x)) ]
$$

```text
L_gm = E_{t,x~p_t}[ D(F_t, F_theta) ]
```

### F10 CGM loss（实际训练目标）
- Source lines: `main.tex:863-865`
- Label: `eq:cgm_loss`

$$
L_{\text{cgm}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot|z)} [ D(F^z_t(x), F_t^\theta(x)) ]
$$

```text
L_cgm = E_{t,z,x~p_t(.|z)}[ D(F_t^z(x), F_t^theta(x)) ]
```

### F11 Markov superposition（flow+jump 合法性）
- Source lines: `main.tex:965`

$$
\alpha_t^1 \mathcal{L}_t + \alpha_t^2 \mathcal{L}_t', \quad \alpha_t^1, \alpha_t^2 \ge 0, \quad \alpha_t^1 + \alpha_t^2 = 1
$$

```text
L_ms = a1 * L1 + a2 * L2, a1>=0, a2>=0, a1+a2=1
```

### F12 CondOT 路径定义
- Source lines: `main.tex:2417`, `main.tex:2505`

$$
p_t(\cdot|z) = \mathcal{N}(tz, (1-t)^2)
$$

```text
p_t(.|z) = Normal(mean=t*z, var=(1-t)^2)
```

### F13 CondOT 的 flow 条件目标
- Source lines: `main.tex:2419`

$$
u_t(x|z) = \frac{z-x}{1-t}
$$

```text
u_target = (z - x) / (1 - t)
```

### F14 CondOT 的 jump 强度
- Source lines: `main.tex:2596-2597`
- Label: `eq:lambda_t_condot_jump`

$$
\lambda_t(x) = \frac{[k_t(x)]_{+}}{(1-t)^3}
$$

```text
lambda_t(x) = relu(k_t(x)) / (1-t)^3
```

### F15 CondOT 的 jump 分布 (已修正)
- Source lines: `main.tex:2598-2599` (Note: this line has confusing notation), see also `main.tex:827` for the clearer `eq:jump_model`.
- 用途: 定义在 `x` 点发生 jump 后，选择目标点 `y` 的概率分布。

$$
J_t(dy;x) \propto [-k_t(y)]_{+} \mathcal{N}(y; tz, (1-t)^2) dy
$$

```text
# Correct interpretation: The probability of jumping TO a point 'y'
# is proportional to the following density at 'y'.
J_target(y) ∝ relu(-k_t(y)) * Normal(y; t*z, (1-t)^2)
```

> **修正说明**: 原 `GM.md` 此处直接翻译了 `main.tex:2599` 的歧义写法，导致 jump 的“源”和“目标”变量混淆。现已更正为正确的形式：jump 目标的分布依赖于**目标点 `y`** 的属性。此修正也使公式与文档 `D节` 的伪代码达成一致。


### F16 CondOT 的 jump 核
- Source lines: `main.tex:2600-2601`
- Label: `eq:Q_t_condot_jump`

$$
Q_t(y;x) = \lambda_t(x) J_t(y;x)
$$

```text
Q_t = lambda_t * J_t
```

### F17 `k_t` 多项式定义
- Source lines: `main.tex:2599`

$$
k_t(x) = x^2 - (t+1)xz - (1-t)^2 + tz^2
$$

```text
k_t(x)=x^2-(t+1)*x*z-(1-t)^2+t*z^2
```

### F18 U-Net 输出语义（jump 图像）
- Source lines: `main.tex:2632`

```text
... using a U-Net architecture with b+1 channels where b describes the number of bins.
```

```text
jump head 输出 b+1 通道：b 个 bin logits + 1 个强度相关通道
```

### F19 jump 采样更新（逐维独立）
- Source lines: `main.tex:2634-2639`

$$
X_{t+h}^i = \begin{cases} X_t & \text{if } m=0 \\ \sim J_t^i(X_t) & \text{if } m=1 \end{cases}
$$

```text
if m_i=0 keep x_i; if m_i=1 sample x_i from J_i
```

### F20 1D jump 训练损失（infinitesimal KL）
- Source lines: `main.tex:2644-2646`

$$
Q_t(y;x) = J_t(y;x) \lambda_t(x), \quad Q_t^\theta(y;x) = J_t(y;x) \lambda_t^\theta(x)
$$
$$
D(Q_t(y;x), Q_t^\theta(y;x)) = \sum_{y \neq x} Q_t^\theta(y;x) - Q_t(y;x) \log Q_t^\theta(y;x)
$$

```text
D = sum_{y!=x} [ Q_theta(y;x) - Q(y;x) * log Q_theta(y;x) ]
```

### F21 多维 jump 损失分解
- Source lines: `main.tex:2649-2650`

$$
Q_t^i(y^i;x) = J_t^i(y^i;x) \lambda_t^i(x), \quad Q_t^{\theta,i}(y^i;x) = J_t^{\theta,i}(y^i;x) \lambda_t^{\theta,i}(x)
$$
$$
D(Q_t(y;x), Q_t^\theta(y;x)) = \sum_{i=1}^{d} D_0(Q_t^i(y^i;x), Q_t^{\theta,i}(y^i;x))
$$

```text
D_total = sum_i D0(Q_i, Q_i_theta)
```

### F22 Euler 采样器更新式 (附实现提示)
- Source lines: `main.tex:1167-1171`

$$
\lambda_t(X_t) = \int Q_t(dy;X_t)
$$
$$
\bar{X}_{t+h} \sim Q_t(\cdot;X_t) / \lambda_t(X_t), \quad \epsilon \sim \mathcal{N}(0, 1)
$$
$$
m \sim \text{Bernoulli}(h \lambda_t(X_t))
$$
$$
\tilde{X}_{t+h} = X_t + h u_t(X_t) + \sqrt{h} \sigma_t(X_t) \epsilon
$$
$$
X_{t+h} = m \bar{X}_{t+h} + (1-m) \tilde{X}_{t+h}
$$

```text
sample jump candidate + continuous candidate, then Bernoulli gate
```
> **实现提示 (重要)**: 上述 `m ~ Bernoulli(h * lambda_t(X_t))` 是最基础的 Euler-Maruyama 近似。论文附录 (`main.tex:2611-2624`) 指出，使用一个更精确的非线性 jump 调度器 `m ~ Bernoulli(1 - R_{t,t+h}(lambda_t(x)))` 对最终生成质量有“显著影响” (CIFAR-10 FID 12 -> 4.5)。为复现最佳效果，应参考附录实现该调度器 `R`。

### F23 `flow + jump` 联合损失可相加原则
- Source lines: `main.tex:1000`, `main.tex:1981-1984`
- 理论说明：在 Bregman 散度框架下，不同马尔科夫分量的损失函数可以直接 1:1 相加，无需引入超参数权重。

$$
L_{\text{cgm}} = \mathbb{E} [ D_{\text{flow}}(\dots) + D_{\text{jump}}(\dots) ]
$$

```text
L_total = L_flow + L_jump
```

### F24 flow matching 是 CGM(MSE) 特例
- Source lines: `main.tex:2796-2798`

$$
u_t(x) = \int u_t(x|z) \frac{p_t(x|z) p_{\text{data}}(z)}{p_t(x)} dz
$$

```text
flow target in conditional form + MSE => flow matching special case of CGM
```

---

## B. 训练路径到采样的逐步实现（严格按公式索引）

## B1 路径构建（Step 0-1）

1. 选先验 `p0`。
2. 定义条件路径 `p_t(.|z)`，并满足边界（F01, F04）。
3. 本文 image 实验使用 CondOT（F12）。
4. 从 CondOT 可直接采样训练点：

```text
x_t = t*z + (1-t)*eps, eps ~ N(0, I)
```

## B2 条件生成器目标（Step 2）

1. flow 目标（F13）：

```text
u_target(x_t,z,t) = (z-x_t)/(1-t)
```

2. jump 目标（F14-F17）：

```text
k_t(x)=x^2-(t+1)*x*z-(1-t)^2+t*z^2
lambda_target(x_t)=relu(k_t(x_t))/(1-t)^3
J_target(bin) ∝ relu(-k_t(bin)) * Normal(bin; t*z, (1-t)^2)
Q_target = lambda_target * J_target
```

## B3 网络参数化（Step 4）

1. flow head 预测 `u_theta`。
2. jump head 使用 U-Net `b+1` 通道（F18）：
- `b` 通道：bins logits（softmax 后为 `J_theta`）。
- `1` 通道：强度相关输出（用于 `lambda_theta`）。

## B4 损失构造（Step 3, Step 5）

1. 通用训练目标用 CGM（F10），Bregman 定义见 F08。
2. flow 分支常用 MSE（F24）。
3. jump 分支用 infinitesimal KL（F20, F21）。
4. `flow + jump` 无需调参，直接 1:1 相加（F23）：

```text
L_total = L_flow + L_jump
```

## B5 采样（Step 6）

1. 通用 Euler 更新见 F22。
2. flow+jump 合法组合见 F11。
3. 图像 jump 逐维更新见 F19。

---

## C. 逐部分原文摘录（verbatim, 带行号）

> 说明：以下摘录是你要求的“每个部分原文贴在下面”。为控制篇幅，仅贴训练闭环最关键段落；每段都附行号。

### C1 路径定义与边界
- Source: `main.tex:158-162`, `178-186`

```tex
[158] ... conditional probability path is a set of time-varying probability distributions (p_t(dx|z))_{0<=t<=1} ...
[159] \begin{align}
[160] z\sim \pdata&, x\sim p_{t}(dx|z) \quad \Rightarrow \quad x\sim p_t(dx)
[161] \end{align}
[162] ... p_0(dx|z)=\pinitial and p_1(dx|z)=\delta_{z} ...
[178] Two common constructions are mixtures ... and geometric averages ...
[179] \begin{align}
[180] p_t(dx|z)&=(1-\kappa_t)\cdot\pinitial(dx) + \kappa_t\cdot \delta_{z}(dx) ...
[184] p_t(dx|z)&= \E_{x_0}\brac{  \delta_{\sigma_t x_0 + \alpha_t z}} ... \Leftrightarrow x_t=\sigma_t x_0+\alpha_t z
[186] ... \kappa_0=\alpha_0=\sigma_1=0 and \kappa_1=\alpha_1=\sigma_0=1 and 0\leq \kappa_t\leq 1
```

### C2 KFE 与边缘生成器
- Source: `main.tex:774-777`, `795-797`

```tex
[774] \label{eq:kfe}
[775] \partial_{t}\action{p_t}{f}= \action{p_t}{\gL_t f}
[777] ... if a generator \gL_t ... satisfies the above equation, then X_t generates the probability path ...
[795] \label{eq:marginal_generator_equation}
[796] \mathcal{L}_tf(x) =&\mathbb{E}_{z\sim p_{1|t}(\cdot|x)}[\mathcal{L}_t^zf(x)]
[797] \end{align}
```

### C3 CGM 训练目标
- Source: `main.tex:853-865`

```tex
[853] \label{eq:bregman_divergences}
[854] D(a,b)=\phi(a) - [\phi(b) +\ip{a-b, \nabla \phi(b)}],\quad a,b\in \Omega
[858] L_{\text{gm}}(\theta)\defe\mathbb{E}_{t\sim\text{Unif}, x\sim p_t}\left[D(F_t(x),F_t^\theta(x))\right]
[863] \label{eq:cgm_loss}
[864] L_{\text{cgm}}(\theta) &\defe\mathbb{E}_{t\sim\text{Unif}, z\sim \pdata, x\sim p_t(\cdot|z)}\left[D(F^z_t(x),F_t^\theta(x))\right]
[865] \end{align}
```

### C4 CondOT 下 flow 目标
- Source: `main.tex:2417-2420`

```tex
[2417] ... CondOT probability path p_t(\cdot|z)=\mathcal{N}(tz,(1-t)^2).
[2418] \begin{align*}
[2419] u_t(x|z)=\frac{z-x}{1-t}
[2420] \end{align*}
```

### C5 CondOT 下 jump 目标
- Source: `main.tex:2593-2602`

```tex
[2593] ... jump intensity \lambda_t(x) and state-independent jump distribution J_t ...
[2595] \label{eq:lambda_t_condot_jump}
[2596] \lambda_t(x)
[2597] =&\frac{[k_t(x)]_{+}}{(1-t)^3}\\
[2598] \label{eq:J_t_condot_jump}
[2599] J_t(x;\tilde{x})=J_t(x)\propto& [-k_t(x)]_{+}\mathcal{N}(x,tz,(1-t)^2)\\\text{where }\quad k_t(x)=&x^2-(t+1)xz-(1-t)^2+tz^2\\
[2600] \label{eq:Q_t_condot_jump}
[2601] Q_t(y;x)=&\lambda_t(x)J_t(y;x)
[2602] \end{align}
```

### C6 U-Net 与 jump 损失
- Source: `main.tex:2632-2651`

```tex
[2632] ... using a U-Net architecture with b+1 channels where b describes the number of bins ...
[2633] \begin{align}
[2634] X_{t+h} =& (X_{t+h}^1,\dots, X_{t+h}^d)\\
[2635] m_i \sim& \text{Bernoulli}(1-R_{t,t+h}(\lambda_t(x)))\\
[2636] X_{t+h}^i=&\begin{cases}
[2637]     X_{t} & \text{if }m=0\\
[2638]     \sim J_t^i(X_{t}) & \text{if }m=1
[2639] \end{cases}
[2642] \paragraph{Loss function.} ... infinitesimal KL-divergence ...
[2644] Q_t(y;x)=&J_t(y;x)\lambda_t(x),\quad Q_t^\theta(y;x)=J_t(y;x)\lambda_t^\theta(x)\\
[2645] D(Q_t(y;x), Q_t^\theta(y;x))=&\sum\limits_{y\neq x}Q_t^\theta(y;x)-Q_t(y;x)\log Q_t^\theta(y;x)
[2649] Q_t^i(y^i;x)=&J_t^i(y^i;x)\lambda_t^i(x),\quad Q_t^{\theta,i}(y^i;x)=J_t^{\theta,i}(y^i;x)\lambda_t^{\theta,i}(x)\\
[2650] D(Q_t(y;x), Q_t^\theta(y;x))=&\sum\limits_{i=1}^{d}D_0(Q_t^i(y^i;x), Q_t^{\theta,i}(y^i;x))
```

### C7 flow+jump 组合与采样
- Source: `main.tex:965`, `1167-1171`, `1000`, `1981-1984`

```tex
[965] \alpha_t^1\mathcal{L}_t+\alpha_t^2\mathcal{L}_t' ... \alpha_t^1+\alpha_t^2=1
[1000] Loss function: We can simply take the sum of loss functions for each S_i.
[1167] Jump intensity \lambda_t(X_t)=\int Q_t(dy;X_t)
[1168] \bar{X}_{t+h}\sim Q_t(\cdot;X_t)/\lambda_t(X_t)
[1169] m\sim \text{Bernoulli}(h\lambda_t(X_t))
[1170] \tilde{X}_{t+h}=X_t+hu_t(X_t)+\sqrt{h}\sigma_t(X_t)\epsilon_t
[1171] X_{t+h}=m\bar{X}_{t+h}+(1-m)\tilde{X}_{t+h}
[1981] L_{\text{cgm}}(\theta) =&\mathbb{E}_{...}\left[D(F^z_t(x),F_t^\theta(x))\right]
[1983] ... [D_1(...) + D_2(...)]
```

---

## D. 可直接照着实现的 flow+jump 训练伪代码（对照 Fxx）

```text
for each step:
  sample z ~ p_data
  sample t ~ Uniform(0,1)
  sample eps ~ N(0, I)
  x_train = t*z + (1-t)*eps                         # F12 训练点采样

  u_target = (z - x_train)/(1-t)                     # F13

  # k(x, z, t) 函数定义 (F17)，用于计算不同点位的速率/分布
  k_func = lambda val: val**2 - (t+1)*val*z - (1-t)**2 + t*z**2 

  lambda_target = relu(k_func(x_train))/(1-t)**3     # F14, 使用当前点 x_train
  J_target(y) ∝ relu(-k_func(y))*N(y;tz,(1-t)**2)    # F15, 使用目标点 y (bins)
  Q_target = lambda_target * J_target                # F16

  u_theta, jump_head = net(x_train, t, cond)
  # jump_head: b+1 通道 (F18)
  J_theta = softmax(logits_b)
  lambda_theta = map_to_nonnegative(intensity_channel)
  Q_theta = lambda_theta * J_theta

  L_flow = mse(u_theta, u_target)                    # F24
  L_jump = sum_i D0(Q_target_i, Q_theta_i)           # F20, F21
  L_total = L_flow + L_jump                          # F23 (注意：1:1 直接相加，理论保证无需调参)

  backprop + optimizer step
```

---

## E. 论文未给出的实现参数（仍需你/代码库补齐）

1. optimizer 类型与超参（lr、beta、weight decay）。
2. batch size、总步数、EMA。
3. `w_f/w_j` 权重与调度。
4. U-Net 深度、每层通道、attention 分辨率、time embedding 细节。
5. mixed precision、梯度裁剪、数据增强。

---

## F. Full Raw Excerpts（逐段完整原文）

下面全部是从 main.tex 直接按行号抽取的原文，不做改写。

### 概率路径定义与边界 (158,186)
```text
   158	A fundamental paradigm of recent state-of-the-art generative models is that they prespecify a transformation of a simple distribution $\pinitial$ (e.g. a Gaussian) into $\pdata$ via probability paths. Specifically, a \textbf{conditional probability path} is a set of time-varying probability distributions $(p_t(dx|z))_{0\leq t\leq 1}$ depending on a data point $z\in S$. Together with the data distribution $\pdata$, this induces a corresponding \textbf{marginal probability path} via the hierarchical sampling procedure:
   159	\begin{align}
   160	z\sim \pdata&, x\sim p_{t}(dx|z) \quad \Rightarrow \quad x\sim p_t(dx)
   161	\end{align}
   162	i.e. first sample a data point $z\sim\pdata$ and then sample $x\sim p_t(dx|z)$ from the conditional path. As we will see, this makes training scalable. The conditional probability path is usually chosen such that $p_0(dx|z)=\pinitial$ and $p_1(dx|z)=\delta_{z}$ where $\delta_{z}$ is the Dirac delta distribution at $z$ (i.e. the trivial, deterministic distribution returning $z$ every draw). The associated marginal probability path interpolates between  $\pinitial$ and $\pdata$, leading to the first design principle of GM:
   163	\begin{center}\vspace{-10pt}			% Centering minipage
   164	    \colorbox{mygray} {		% Set's the color of minipage
   165	      \begin{minipage}{0.98\linewidth} 	% Starts minipage
   166	       \centering
   167	       \vspace{-0pt}
   168	\textbf{Principle 1}: Given a data distribution $\pdata$, choose a prior $\pinitial$ and a conditional probability path such that its marginal probability path $(p_t)_{0\leq t\leq 1}$ fulfills $\pinitial=p_0$ and $\pdata=p_1$.
   169	      \end{minipage}}			% End minipage
   170	      \vspace{-1em}
   171	\end{center}
   172	% \begin{align}
   173	% \label{eq:prob_path_condition}
   174	%     p_0 = \pinitial, \quad p_1=\pdata.
   175	% \end{align}
   176	% In addition, one usually requires that the path is smooth in $t$ (see \peter{?appendix?} for a rigorous statement of the smoothness assumption).
   177	%\brian{As opposed to $\cdot$, would explicit arguments $X$ or $x$ be easier to understand?}  
   178	Two common constructions are mixtures (for arbitrary $S$) and geometric averages (for $S=\mathbb{R}^d$):
   179	\begin{align}
   180	    p_t(dx|z)&=(1-\kappa_t)\cdot\pinitial(dx) + \kappa_t\cdot \delta_{z}(dx) \ \  \Leftrightarrow \ \   x_t \sim \begin{cases}
   181	        z & \text{with prob } \kappa_t\\
   182	        x_0 & \text{with prob } (1-\kappa_t)
   183	    \end{cases} \ \  \blacktriangleright\text{mixture}\\
   184	    p_t(dx|z)&= \E_{x_0}\brac{  \delta_{\sigma_t x_0 + \alpha_t z}} \quad \qquad \ \ \qquad \qquad \ \ \ \Leftrightarrow \ \  x_t = \sigma_t x_0 + \alpha_t z \quad \quad \   \blacktriangleright\text{geometric average}
   185	\end{align}
   186	where $x_t\sim p_t(\cdot|z), x_0\sim \pinitial, z\sim\pdata$,  and $\alpha_t,\sigma_t,\kappa_t\in \mathbb{R}_{\geq 0}$ are differentiable functions satisfying $\kappa_0=\alpha_0=\sigma_1=0$ and $\kappa_1=\alpha_1=\sigma_0=1$ and $0\leq \kappa_t\leq 1$.
```

### KFE 与边缘生成器 (774,803)
```text
   774	\label{eq:kfe}
   775	\partial_{t}\action{p_t}{f}= \action{p_t}{\gL_t f} \qquad \blacktriangleright\text{Kolmogorov Forward Equation (KFE)}
   776	\end{equation}
   777	Conversely, if a generator $\gL_t$ of a Markov process $X_t$ satisfies the above equation, then $X_t$ generates the probability path $(p_t)_{0\leq t \leq 1}$, i.e. initializing $X_0\sim p_0$ will imply that $X_t\sim p_t$ for all $0\leq t\leq 1$ (see \cref{appendix:kfe_is_sufficient_to_define_marginals}) \citep{rogers2000diffusions}. 
   778	Therefore, the key challenge of Generator Matching is:
   779	\begin{center}\vspace{-10pt}			% Centering minipage
   780	    \colorbox{mygray} {		% Set's the color of minipage
   781	      \begin{minipage}{1.0\linewidth} 	% Starts minipage
   782	       \centering
   783	       \vspace{-0pt}   
   784	\textbf{Principle 3*}: Given a marginal probability path $(p_t)_{0\leq t\leq 1}$, find a generator satisfying the KFE. 
   785	      \end{minipage}}			% End minipage
   786	      \vspace{-1em}
   787	\end{center}
   788	\textbf{Remark - Adjoint KFE.} We note that the above version of the KFE determines the evolution of expectations of test functions $f$. Whenever a probability density $\frac{dp_t}{d\nu}(x)$ exists, one can use the \emph{adjoint KFE} (see \cref{table:Markov_overview} for examples and \cref{appendix:adjoint_kfe}). In this form, the KFE generalizes many equations used to develop generative models such as the Fokker-Planck or the continuity equation \citep{song2020score, lipman2022flow}.
   789	
   790	We now show how to find a generator that generates a marginal probability path $p_t$ with conditional path $p_t(\cdot|z)$. Assume that for every data point $z\in S$, we found a generator $\mathcal{L}_t^z$ that generates $p_t(\cdot|z)$. We call $\mathcal{L}_t^z$ \textbf{conditional generator}. This allows us to construct a generator for the marginal path (\textbf{marginal generator}):
   791	\begin{proposition}
   792	\label{prop:marginal_generator}
   793	The marginal probability path $(p_t)_{0\leq t \leq 1}$ is generated by a Markov process $X_t$ with generator
   794	\begin{align}
   795	\label{eq:marginal_generator_equation}
   796	\mathcal{L}_tf(x) =&\mathbb{E}_{z\sim p_{1|t}(\cdot|x)}[\mathcal{L}_t^zf(x)]
   797	\end{align}
   798	where $p_{1|t}(dz|x)$ is the posterior distribution (i.e. the conditional distribution over data $z$ given an observation $x$). For $S=\mathbb{R}^d$ and the representation in \cref{eq:universal_representation}, we get a marginal representation of $\mathcal{L}_tf(x)$ given by:
   799	\begin{align*}
   800	\nabla f(x)^T \mathbb{E}_{z\sim p_{1|t}(\cdot|x)}[u_t(x|z)] +\frac{\nabla^2 f(x)}{2}\cdot \mathbb{E}_{z\sim p_{1|t}(\cdot|x)}[\sigma_t^2(x|z)]
   801	+ \int \brac{f(y)-f(x)}\mathbb{E}_{z\sim p_{1|t}(\cdot|x)}[Q_t(dy;x|z)]
   802	\end{align*}
   803	Generally, an identity as in \cref{eq:marginal_generator_equation} holds for any linear parameterization of the generator (see \cref{appendix:linear_parameterization}).
```

### CGM 训练目标 (853,865)
```text
   853	\label{eq:bregman_divergences}
   854	    D(a,b)=\phi(a) - [\phi(b) +\ip{a-b, \nabla \phi(b)}],\quad a,b\in \Omega
   855	\end{align}
   856	which are a general class of loss functions including many examples such as MSE or the KL-divergence (see \cref{appendix:bregman_examples}). We use $D$ to measure how well $F_t^\theta$ approximates $F_t$ via the \textbf{Generator Matching loss} defined as
   857	\begin{align}
   858	    L_{\text{gm}}(\theta)\defe\mathbb{E}_{t\sim\text{Unif}, x\sim p_t}\left[D(F_t(x),F_t^\theta(x))\right]&& \blacktriangleright\text{ Generator Matching}
   859	\end{align}
   860	Unfortunately, the above training objective is intractable as we do not know the marginal generator $\mathcal{L}_t$ and also no parameterization $F_t$ of the marginal generator. To make training tractable, let us set $F_t^z$ to be a linear parameterization of the conditional generator $\mathcal{L}_t^z$ with data point $z$ (see \cref{appendix:linear_parameterization}). For clarity, we reiterate that by construction, we know $F_t^\theta, F_t^z, p_t(\cdot|z),D$ as well as can draw $\text{data samples }z\sim \pdata$ but the shape of $F_t$ is unknown. By \cref{prop:marginal_generator}, we can assume that $F_t$ has the shape $F_t(x) = \int F^z_t(x)p_{1|t}(dz|x)$.
   861	This enables us to define the \textbf{conditional Generator Matching loss} as
   862	\begin{align}
   863	    \label{eq:cgm_loss}
   864	    L_{\text{cgm}}(\theta) &\defe\mathbb{E}_{t\sim\text{Unif}, z\sim \pdata, x\sim p_t(\cdot|z)}\left[D(F^z_t(x),F_t^\theta(x))\right]&& \blacktriangleright\text{ Conditional Generator Matching}
   865	\end{align}
```

### 组合模型命题（superposition） (961,971)
```text
   961	\begin{proposition}[Combining models] 
   962	\label{prop:markov_superposition}
   963	Let $p_t$ be a marginal probability path, then the following generators solve the KFE for $p_t$ and consequently define a generative model with $p_t$ as marginal:
   964	\begin{enumerate}
   965	\item \textbf{Markov superposition: } $\alpha_t^1\mathcal{L}_t+\alpha^2_t\mathcal{L}_t'$, where $\mathcal{L}_t,\mathcal{L}_t'$ are two generators of Markov processes solving the KFE for $p_t$, and $\alpha_t^{1},\alpha_t^{2}\geq 0$ satisfy $\alpha^1_t+\alpha^2_t=1$. We call this a \textbf{Markov superposition}. \vspace{-0.3em}
   966	\item \textbf{Divergence-free components: } $\mathcal{L}_t+\beta_t\mathcal{L}_t^\text{div}$, where $\mathcal{L}_t^\text{div}$ is a generator such that $\action{p_t}{\mathcal{L}_t^\text{div}f}=0$ for all $f\in \gT$, and $\beta_t\geq 0$. We call such $\mathcal{L}_t^\text{div}$ \textbf{divergence-free}. \vspace{-0.3em}
   967	%
   968	% \item \textbf{Predictor-corrector: }$\alpha^1_t\mathcal{L}_t-\alpha^2_t\bar{\mathcal{L}}_t$, where $\mathcal{L}_t$ is a generator solving the KFE for $p_t$ in forward-time and $\bar{\mathcal{L}}_t$ is a generator solving the KFE in backward time, and  $\alpha_t^{1},\alpha_t^{2}\geq 0$ with $\alpha^1_t+ \alpha^2_t=1$. 
   969	% %
   970	\item \textbf{Predictor-corrector: }$\alpha^1_t\mathcal{L}_t+\alpha^2_t\bar{\mathcal{L}}_t$, where $\mathcal{L}_t$ is a generator solving the KFE for $p_t$ in forward-time and $\bar{\mathcal{L}}_t$ is a generator solving the KFE in backward time, and  $\alpha_t^{1},\alpha_t^{2}\geq 0$ with $\alpha^1_t-\alpha^2_t=1$. \vspace{-0.3em}
   971	\end{enumerate}
```

### GM Recipe + Euler Sampler (1145,1175)
```text
  1145	    \begin{algorithm}[H]    \caption{\label{alg:gm_recipe}Generator Matching recipe for constructing Markov generative model (theory in black, implementation in \implpart{brown})}
  1146	    \theorypart{\textbf{Step 0: }Choose prior $\pinitial$}\\
  1147	    \theorypart{\textbf{Step 1:} Choose $p_t(dx|z)$ such that \\
  1148	    marginal $p_0=\pinitial$ and $p_1=\pdata$}\\
  1149	    \theorypart{\textbf{Step 2:} Find solution $\mathcal{L}_t^z$ to KFE}\\
  1150	    \theorypart{\textbf{Step 3:} Choose Bregman div. as loss}\\
  1151	\implpart{\textbf{Step 4:} Construct neural net $\mathcal{L}_t^\theta$}\\
  1152	    \implpart{\textbf{Step 5:} Minimize CGM loss using $\mathcal{L}_t^z$}\\
  1153	    \implpart{\textbf{Step 6:} Sample using Algorithm 2.}
  1154	\end{algorithm}
  1155	    % \includegraphics[width=\textwidth]{figure.png} % Replace "figure.png" with the path to your image file
  1156	    %     \caption{Caption for the figure}
  1157	    %     \label{fig:figure1}
  1158	    \end{minipage}\hfill  
  1159	%\resizebox{1.0\textwidth}{!}{
  1160	    \begin{minipage}{0.45\textwidth}
  1161	        \begin{algorithm}[H]
  1162	\caption{\label{alg:euler_sampling}Euler sampling for $S=\mathbb{R}^d$}\label{alg:two}
  1163	            \textbf{Given: } $u_t,\sigma_t$, $Q_t$, $\pinitial$, step size $h>0$\\
  1164	            \textbf{Init: }$X_0\sim \pinitial$
  1165	            \begin{algorithmic}[1]
  1166	                \For{$t$ in $\text{linspace}(0,1,1/h)$}
  1167	                \State Jump intensity $\lambda_t(X_t)=\int Q_t(dy;X_t)$
  1168	                    \State $\bar{X}_{t+h}\sim Q_t(\cdot;X_t)/\lambda_t(X_t)$
  1169	                    \State $m\sim \text{Bernoulli}(h\lambda_t(X_t))$, $\epsilon_t\sim \mathcal{N}(0,1)$
  1170	                    \State $\tilde{X}_{t+h}=X_{t}+hu_t(X_t)+\sqrt{h}\sigma_t(X_t)\epsilon_t$
  1171	                    \State $X_{t+h}=m\bar{X}_{t+h}+(1-m)\tilde{X}_{t+h}$
  1172	                \EndFor
  1173	            \end{algorithmic}
  1174	            \textbf{Return: }$X_1$
  1175	        \end{algorithm}
```

### CondOT flow 目标 (2416,2422)
```text
  2416	\subsection{Derivation of flow solution to  CondOT path}
  2417	Next, we consider the CondOT probability path $p_t(\cdot|z)=\mathcal{N}(tz,(1-t)^2)$. We consider a Markov process defined via a flow determined by the vector field
  2418	\begin{align*}
  2419	    u_t(x|z)=\frac{z-x}{1-t}
  2420	\end{align*}
  2421	as already introduced in \citep{lipman2022flow}. For completeness, we show that this generates the probability path (in \citep{lipman2022flow}, this is done more generally for Gaussian probability paths). Specifically, one has to show that it fulfils the continuity equation (the adjoint KFE for flows, see \cref{table:Markov_overview}), i.e.
  2422	\begin{align*}
```

### CondOT jump 总结公式 (2593,2603)
```text
  2593	\paragraph{Summary.} The CondOT probability path is generated by a jump process with jump intensity $\lambda_t(x)$ and state-independent jump distribution $J_t$ given by:
  2594	\begin{align}
  2595	\label{eq:lambda_t_condot_jump}
  2596	\lambda_t(x)
  2597	=&\frac{[k_t(x)]_{+}}{(1-t)^3}\\
  2598	\label{eq:J_t_condot_jump}
  2599	J_t(x;\tilde{x})=J_t(x)\propto& [-k_t(x)]_{+}\mathcal{N}(x,tz,(1-t)^2)\\\text{where }\quad k_t(x)=&x^2-(t+1)xz-(1-t)^2+tz^2\\
  2600	\label{eq:Q_t_condot_jump}
  2601	Q_t(y;x)=&\lambda_t(x)J_t(y;x)
  2602	\end{align}
  2603	Intuitively, we jump at $x_t$ only if $k(x_t)$ has a positive value. If we jump, we jump to a region of negative $k_t(x)$ proportional to $k_t(x)$ multiplied with the desired density.
```

### U-Net 与 jump 损失 (2632,2651)
```text
  2632	where $\lambda_t^i(x)\geq 0$ and $J_t^i(x)$ is a categorical distribution (using softmax) over a fixed set of bins in $[-1,1]$ (support of normalized images). On images, we implement this by using a U-Net architecture with $b+1$ channels where  $b$ describes the number of bins. During sampling, for each time update $t\mapsto t+h$, updates happen independently per dimension. Specifically,
  2633	\begin{align}
  2634	X_{t+h} =& (X_{t+h}^1,\dots, X_{t+h}^d)\\
  2635	m_i \sim& \text{Bernoulli}(1-R_{t,t+h}(\lambda_t(x)))\\
  2636	X_{t+h}^i=&\begin{cases}
  2637	    X_{t} & \text{if }m=0\\
  2638	    \sim J_t^i(X_{t}) & \text{if }m=1
  2639	\end{cases}
  2640	\end{align}
  2641	
  2642	\paragraph{Loss function.}As a loss function, we use an infinitesimal KL-divergence in $1d$ via
  2643	\begin{align}
  2644	    Q_t(y;x)=&J_t(y;x)\lambda_t(x),\quad Q_t^\theta(y;x)=J_t(y;x)\lambda_t^\theta(x)\\
  2645	    D(Q_t(y;x), Q_t^\theta(y;x))=&\sum\limits_{y\neq x}Q_t^\theta(y;x)-Q_t(y;x)\log Q_t^\theta(y;x)
  2646	\end{align}
  2647	where the sum of $y$'s is here over regularly spaced bin values in $[-1,1]$. We extend the above loss to the multi-dimensional case via
  2648	\begin{align}
  2649	    Q_t^i(y^i;x)=&J_t^i(y^i;x)\lambda_t^i(x),\quad Q_t^{\theta,i}(y^i;x)=J_t^{\theta,i}(y^i;x)\lambda_t^{\theta,i}(x)\\
  2650	    D(Q_t(y;x), Q_t^\theta(y;x))=&\sum\limits_{i=1}^{d}D_0(Q_t^i(y^i;x), Q_t^{\theta,i}(y^i;x))
  2651	\end{align}
```

### loss 可相加原则 (994,1001)
```text
   994	\begin{proposition}[Multimodal generative models - Informal version] 
   995	\label{prop:multimodal_model_informal}
   996	Let $q_t^1(\cdot|z_1),q_t^2(\cdot|z_2)$ be two conditional probability paths on state spaces $S_1,S_2$. Define the conditional factorized path on $S_1\times S_2$ as $p_t(\cdot|z_1,z_2)=q_t^1(\cdot|z_1)q_t^2(\cdot|z_2)$. Let $p_t(dx)$ be its marginal path.
   997	\begin{enumerate}
   998	    \item \textbf{Conditional generator:} To find a solution to the KFE for the conditional factorized path, we only have to find solutions to the KFE for each $S_1,S_2$. We can combine them component-wise. \vspace{-0.3em}
   999	    \item \textbf{Marginal generator: }The marginal generator of $p_t(dx)$ can be parameterized as follows: (1) parameterize a generator on each $S_i$ but make it values depend on all dimensions; (2) During sampling, update each component independently as one would do for each $S_i$ in the unimodal case. \vspace{-0.3em}
  1000	    \item \textbf{Loss function: }We can simply take the sum of loss functions for each $S_i$. \vspace{-0.3em}
  1001	\end{enumerate}
```

### 联合 loss 的和式写法 (1981,1984)
```text
  1981	L_{\text{cgm}}(\theta) =&\mathbb{E}_{t\sim\text{Unif}, z\sim \pdata, x_1\sim p_t(\cdot|z_1), x_2\sim p_t(\cdot|z_2)}\left[D(F^z_t(x),F_t^\theta(x))\right]\\
  1982	=&\mathbb{E}_{t\sim\text{Unif}, z\sim \pdata, x_1\sim p_t(\cdot|z_1), x_2\sim p_t(\cdot|z_2)}\left[D((F_t^{z_1}(x_1),F_t^{z_2}(x_2)),F_t^\theta(x))\right]\\
  1983	=&\mathbb{E}_{t\sim\text{Unif}, z\sim \pdata, x_1\sim p_t(\cdot|z_1), x_2\sim p_t(\cdot|z_2)}\left[D_1(F_t^{z_1}(x_1),F_{t,1}^\theta(x))+D_2(F_t^{z_2}(x_2),F_{t,2}^\theta(x))\right]
  1984	\end{align}
```

### flow matching 与 MSE 对应 (2794,2798)
```text
  2794	\citep{lipman2022flow, liu2022flow} are immediate instances of Generator Matching leveraging the flow-specific versions of the KFE given by the continuity equation (see \cref{table:Markov_overview} and \cref{subsec:flows} for a derivation). We briefly describe here how one can map the propositions from this work to their work. Specifically, flow matching restricts itself to generators of the form $\mathcal{L}_t^\theta f(x)=\nabla f(x)^Tu_t^\theta(x)$ for a vector field $u_t^\theta(x)$ parameterized by a neural network with parameters $\theta$. Given a conditional vector field $u_t(x|z)$ and a probability path $p_t(x|z)$, the corresponding marginal vector field in flow matching (see \citep[equation (8)]{lipman2022flow}) is given by
  2795	\begin{align}
  2796	    u_t(x) = \int u_t(x|z)\frac{p_t(x|z)\pdata(z)}{p_t(x)}dz
  2797	\end{align}
  2798	and corresponds to the marginal generator (see \cref{prop:marginal_generator}). The Bregman divergence used is the mean squared error (MSE) obtained by choosing $\phi(x)=\|x\|^2$ in \cref{eq:bregman_divergences}. The conditional flow matching loss \citep[Theorem 2 ]{lipman2022flow} is a special case of \cref{prop:conditional_generator_matching}. Therefore, Generator Matching can be seen as a generalization of the principles of flow matching to the space of Markov process generators for arbitrary state spaces.
```

### flow+jump 同架构与 mixed 采样 (1069,1073)
```text
  1069	We first study jump models as a novel model class in Euclidean space. We use the jump model defined in \cref{eq:jump_model} and extend it to multiple dimensions using \cref{prop:multimodal_model_informal}. The jump kernel is parameterized with a U-Net architecture (see \cref{appendix:details_jump_model} for details). We use the loss from \cref{table:Markov_overview}. In \cref{appendix:elbo_loss}, we show that this corresponds to an ELBO loss. We apply the model on CIFAR10 and the ImageNet32 (blurred faces) datasets. As jump models do not have yet an equivalent of classifier-free guidance, we focus on unconditional generation. A challenge for a fair comparison is that flow models can use higher-order ODE samplers, while sampling for jump models in $\mathbb{R}^d$ is only done with Euler sampling so far. Hence, we ablate over this choice. As one can see in \cref{fig:image_generation_examples}, the jump model can generate realistic images of high quality. In \cref{tab:image_generation_benchmarking}, we show quantitative results. While lacking behind current state-of-the art models, the jump model shows very promising results as a first version of an unexplored class of models.
  1070	
  1071	\textbf{Combining models - Markov superposition.} Next, we train a flow and jump model in the same architecture. We validate that the flow part achieves the state-of-the-art results as before. We then combine both models via a Markov superposition.
  1072	% (see \cref{appendix:details_markov_superposition} for details)
  1073	As one can see in \cref{tab:image_generation_benchmarking}, a Markov superposition of a flow and jump model boosts the performance of each other. For Euler sampling, we see significant improvements. We can also combine 2nd order samplers for flows with Euler sampling for jumps in a ``mixed'' sampling method (see \cref{tab:image_generation_benchmarking}) leading to improvements of the current SOTA by flow models. We anticipate that with further improvements of the jump model, the increased performance via Markov superposition will be even more pronounced.
```
