# Dual-Stream Mamba Encoder（用于 PPT 展示）

## 背景
现有编码器采用 Transformer + RoPE + 二元注意力偏置：
\[
A = \mathrm{softmax}\left(\frac{Q R K^T}{\sqrt{d}} + U_{\text{intra}} + U_{\text{inter}}\right)V
\]
其中 `RoPE` 提供相对位置信息，`U_intra/U_inter` 区分同变量与跨变量注意力。

## 新编码器核心思想
为增强长序列建模能力，引入 **Dual-Stream Mamba Encoder**：

1. **同变量流（Intra-Variable Stream）**  
   - 将输入按变量维度拆分。  
   - 每个变量独立通过轻量级 Mamba 块，聚焦单变量时序动力学。

2. **跨变量流（Inter-Variable Stream）**  
   - 将全部变量拼接后作为全局序列输入标准 Mamba 流。  
   - 学习变量间依赖与全局交互。

3. **门控融合（Gated Fusion）**  
   \[
   H_{out} = G \odot H_{intra} + (1-G) \odot H_{inter}
   \]
   其中 `G = sigmoid(W[H_intra;H_inter])` 为可学习门控函数。

## 与原 Transformer 方案的对应关系
- **去除 RoPE**：Mamba 天然建模顺序信息，直接利用状态空间时序归纳偏置。  
- **替代二元偏置**：由双流 + 门控融合显式建模“变量内/变量间”两类依赖。  
- **复杂度优势**：相比全局自注意力的二次复杂度，更适合长窗口时间序列。

## 代码实现要点
- 新增 `LightweightMambaBlock`：采用归一化 + 门控投影 + 深度可分离时序卷积 + 全局状态汇聚。  
- 新增 `DualStreamMambaEncoder`：并行计算 `H_intra` 与 `H_inter`，末端门控融合。  
- `TimeSeriesEncoder` 新增参数 `encoder_type`：
  - `encoder_type="transformer"`：保持原有 Transformer/RoPE 兼容路径。
  - `encoder_type="mamba"`：启用新双流 Mamba 编码器（自动关闭 RoPE）。

## PPT 一页可讲清的算法流程
1. Patch 化时间序列并线性嵌入。  
2. 分叉为同变量流与跨变量流并行编码。  
3. 通过可学习门控动态加权融合。  
4. 投影回局部 embedding，供重建/异常头使用。
