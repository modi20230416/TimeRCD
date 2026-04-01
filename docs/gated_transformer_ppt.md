# 门控增强 Transformer Blocks（PPT内容草案）

## 1. 动机（Why Gating）

- **原始 Transformer Block 的局限**
  - 每层输出对所有 token/特征维度一视同仁，缺少“按需放大/抑制”的显式机制。
  - 在时序场景中，噪声片段、弱相关通道会被残差路径持续传递，影响表示质量。
- **训练稳定性挑战**
  - 深层堆叠时，层间信号累积容易出现无效激活放大。
  - 仅靠 LayerNorm/Dropout 难以做到对“信息通道”的细粒度控制。
- **引入门控的目标**
  - 为每层增加可学习的动态筛选能力。
  - 提升有效信息通过率，抑制噪声传播。
  - 在几乎不改变整体架构的前提下增强模型表达能力与训练稳定性。

---

## 2. 方法与公式（What We Add）

在每个 Transformer 层输出后加入门控：

\[
H_{out}' = H_{out} \odot \sigma(H_{inp}W_{\theta})
\]

- \(H_{inp}\)：该层输入特征（layer input）
- \(H_{out}\)：该层原始输出（attention + MLP 后）
- \(W_{\theta}\)：可学习门控参数（线性映射）
- \(\sigma\)：Sigmoid，输出 \((0,1)\) 的软门值
- \(\odot\)：逐元素乘法

**解释：**
- 门值由输入条件化生成，属于 input-conditioned gating。
- 每个位置、每个隐藏维度都有独立门值，实现细粒度特征调制。

---

## 3. 理论依据（Why It Works）

- **条件计算（Conditional Computation）**
  - 门控让网络对不同输入激活不同“子通路”，提升表示效率。
- **噪声抑制与信号选择**
  - 当某些维度对当前样本不重要时，门值趋近 0，减少无效特征干扰。
- **优化与稳定性**
  - Sigmoid 门控将激活幅值限制在可控区间，降低层间异常放大风险。
  - 与残差结构配合，能在“保留主干表达”同时实现自适应调制。
- **参数效率较高**
  - 仅新增一个 \(d_{model}\rightarrow d_{model}\) 线性层，额外开销可控。

---

## 4. 核心代码（How It Is Implemented）

> 文件：`models/time_rcd/ts_encoder_bi_bias.py`

### 4.1 在 Encoder Layer 中新增门控参数

```python
self.gate_proj = nn.Linear(d_model, d_model, bias=True)
```

### 4.2 在 forward 中按公式应用门控

```python
layer_input = src
...
src = residual + self.dropout2(src)  # 原始 H_out

gate = torch.sigmoid(self.gate_proj(layer_input))
src = src * gate  # H_out'
return src
```

---

## 5. 可视化展示建议（PPT页可直接用）

- **页1：问题定义**
  - “为什么原始 Transformer 在时序任务中仍会受到噪声传播影响？”
- **页2：方法概览图**
  - 标出每层输出后新增 Gate 模块（Linear + Sigmoid + Element-wise Mul）。
- **页3：数学表达**
  - 重点展示 \(H_{out}' = H_{out} \odot \sigma(H_{inp}W_{\theta})\)。
- **页4：理论收益**
  - 表达能力↑、噪声抑制↑、训练稳定性↑、参数增量小。
- **页5：代码落地**
  - 展示 `gate_proj` 定义和 forward 中 3 行核心门控逻辑。

---

## 6. 实验指标建议（答辩时可补）

- 收敛稳定性：训练 loss 方差、梯度范数波动
- 检测性能：AUC / F1 / Precision / Recall
- 训练效率：达到同等指标的 epoch 数
- 消融实验：
  - Baseline（无门控）
  - Gate on Attention output
  - Gate on Block output（当前方案）
