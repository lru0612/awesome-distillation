# Logit Fusion RL 后训练：实现规格说明

## 1. 项目简介

本文档规定了“Logit Fusion”（logit 融合）实现的需求。该混合后训练方法融合了 SFT（监督微调）与 RL（强化学习）：每步解码时将 Teacher 与 Student 的 logits 融合，然后用 Importance Sampling (IS) 校正的 REINFORCE 更新。

> 参考论文: [Learning from Mixed Rollouts: Logit Fusion as a Bridge Between Imitation and Exploration]

---

## 2. 环境与模型

- **Student 模型 ($\pi_S$)：** Qwen/Qwen3-8B（ Instruct，待优化的策略）。
- **Teacher 模型 ($\pi_T$)：** Qwen/Qwen3-8B（Instruct，只做推理，权重冻结，可以获取到题目答案作为上下文）。
- **数据集：** DeepMath-103K（须提取每条 prompt 的难度标量 $d$）。
- **框架：** PyTorch, HuggingFace transformers, accelerate/DeepSpeed（多 GPU 可选）。

---

## 3. 严格工程约束（必须遵守）

1. **禁止 vLLM 用于 rollout 生成**：vLLM 等高吞吐 engine 不支持同步多模型 token-by-token 推理。
2. **锁步解码（Lockstep Decoding）：** 必须用自定义 HuggingFace 解码循环实现，每步同时更新 Teacher/Student 的 KV-cache，保证 token-by-token 且全部模型均在 GPU。
3. **词表大小不一致（$V_S$ vs $V_T$）：** Teacher 常多出 padding。解决：令 $V_S$ 为 student 词表，融合 softmax/logits 前对 teacher logit 截断 `logits_T = logits_T[:, :V_S]`。
4. **EOS token 不一致：** 令 $e_S$、$e_T$ 分别为 Student, Teacher 的 EOS id，采样时任一采到则终止：`eos_token_id = [e_S, e_T]`。

---

## 4. rollout 生成：逐 token logit 融合

**每步解码（对 prompt $x$，历史上下文 $y_{<i}$）：**

1. 取 $\ell_T$（Teacher logits）、$\ell_S$（Student logits）。

2. 按上文对齐 teacher 词表区间。

3. 计算动态权重 $\alpha(x)$（见第 5 节）。

4. 计算融合 logits：

   $$
   \ell_\text{mix}(y_i|x, y_{<i}) = \alpha(x) \, \ell_T(y_i|x, y_{<i}) + (1 - \alpha(x)) \, \ell_S(y_i|x, y_{<i})
   $$

5. 按 $\pi_\text{mix} = \text{softmax}(\ell_\text{mix})$ 采样 $y_i$。

6. $y_i$ 添加到上下文，同时更新 Teacher/Student 的 KV-cache。

7. 保存训练所需：$x$、生成的序列 $y$、历史混合概率 $\pi_{\text{mix}_{\text{old}}}(y_i)$、reward 等。

---

## 5. 动态 Alpha 调度 ($\alpha$)

- $\alpha$ 控制 Teacher 影响力，需随 time step 衰减，并随 prompt 难度调整。

- 超参数：$\alpha_\text{init} = 0.5,\, K=5000$（衰减步数）。

- prompt 难度 $d$：[数据集提取，$d \in [d_{min}, d_{max}]$] (如果数据集有难度标签的话，optional)

- 步骤：

  - **难度缩放：**
    $$
    s(d(x)) = \frac{d - d_{min}}{d_{max} - d_{min}}
    $$

  - **基础衰减：**
    $$
    \alpha_\text{base} = \alpha_\text{init} \cdot \max\left(0, 1 - \frac{\text{current\_step}}{K}\right)
    $$

  - **最终 $\alpha$：**
    $$
    \alpha(x) = \alpha_\text{base} \cdot s(d(x))
    $$

---

## 6. RL 更新与目标函数

### ⚠️ 重要：**不能**使用标准 PPO 比率两端剪切！

- 使用**单比率 IS 目标**，每 token 设上界（Token-Level Capping）。

#### 6.1 Group Advantage 计算（GRPO 风格）

- 对每个 prompt $x$，生成 $G$ 个 rollout，得出 reward $R(x, y^{(j)})$。

- 每个 rollout 的每个 token，优势估计为：

  $$
  A_i^{(j)} = R(x, y^{(j)}) - \text{mean}\left( \{R(x, y^{(k)})\}_{k=1}^G \right)
  $$

- （注：优势对整个序列 tokens 等同。）

#### 6.2 Importance Sampling（IS）比率

- rollout 时，采样分布为历史混合分布，训练时为 Student。

  $$
  r_i(\theta) = \frac{\pi_S(y_i|x, y_{<i};\theta)}{\pi_{\text{mix}_{\text{old}}}(y_i|x, y_{<i})}
  $$

#### 6.3 损失函数

- 最小化负目标：

  $$
  \mathcal{J}(\theta) = \frac{1}{B} \sum_\text{batch} \sum_{i=1}^{|y|} \min(r_i(\theta), C_\text{cap}) \cdot A_i
  $$

  - $C_\text{cap} = 3.0$

*实现备注：*

- 先求 $\log \pi_S(\theta)$ 减去 $\log \pi_{\text{mix}_{\text{old}}}$，exp 得 $r_i(\theta)$，torch.clamp 上界 $C_\text{cap}$，乘以 $A_i$，对 batch 取均值。

---

## 7. 代码实现需求（Deliverables）

- **LockstepLogitFusionGenerator 类**：输入 2 个 HF 模型，按动态 $\alpha$ 逐 token 融合，并返回轨迹及 $\log \pi_{\text{mix}_{\text{old}}}$
- **RL Trainer 类**：实现 GRPO 风格平均回报，计算单比率 IS capped loss（无 PPO 剪切），对 Student 回传并更新