# models/time_rcd/ts_encoder_mamba.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


try:
    from mamba_ssm import Mamba

    print("✅ 使用官方 mamba_ssm (CUDA 加速)")
except ImportError:
    try:
        from mamba_min.mamba import Mamba

        print("⚠️  使用 mamba-min (纯 PyTorch，速度较慢)")
    except ImportError:
        raise ImportError(
            "请安装 mamba_ssm 或 mamba-min:\n"
            "pip install mamba-ssm\n"
            "# 或\n"
            "pip install git+https://github.com/kyegomez/mamba-min"
        )


class RMSNorm(nn.Module):
    """Root Mean Square Normalization"""

    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class BidirectionalMambaBlock(nn.Module):
    """双向 Mamba 块：前向 + 后向处理，模拟非因果建模"""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.forward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.backward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        residual = x
        x = self.norm(x)
        # 前向 pass
        y_forward = self.forward_mamba(x)
        # 后向 pass（反转序列）
        y_backward = self.backward_mamba(torch.flip(x, dims=[1]))
        y_backward = torch.flip(y_backward, dims=[1])
        # 相加融合（也可拼接，但会增加维度）
        y = y_forward + y_backward
        return y + residual


class TimeSeriesEncoderMamba(nn.Module):
    """
    Mamba 架构的时间序列编码器，兼容原 TimeSeriesEncoder 接口。

    输入:
        time_series: (B, seq_len, num_features)
        mask: (B, seq_len) —— 用于 padding 掩码（当前未用于 Mamba，但保留接口）

    输出:
        local_embeddings: (B, seq_len, num_features, d_proj)
    """

    def __init__(
            self,
            d_model: int = 512,
            d_proj: int = 256,
            patch_size: int = 4,
            num_layers: int = 8,
            d_ff_dropout: float = 0.1,
            max_total_tokens: int = 8192,  # 保留参数，Mamba 不依赖此值
            use_rope: bool = True,  # 保留参数，Mamba 不使用 RoPE
            num_features: int = 1,
            activation: str = "gelu",  # 保留参数，Mamba 内部使用 SiLU
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_proj = d_proj
        self.num_layers = num_layers
        self.num_features = num_features

        # Patch embedding: 将每个 patch 映射到 d_model 维
        self.embedding_layer = nn.Linear(patch_size, d_model)

        # 可选：注入特征 ID 信息（增强多变量感知）
        if num_features > 1:
            self.feature_embed = nn.Embedding(num_features, d_model)
        else:
            self.feature_embed = None

        # 堆叠双向 Mamba 层
        self.mamba_layers = nn.ModuleList([
            BidirectionalMambaBlock(d_model=d_model)
            for _ in range(num_layers)
        ])

        # 输出投影层：从 d_model → patch_size * d_proj
        self.projection_layer = nn.Linear(d_model, patch_size * d_proj)
        self.dropout = nn.Dropout(d_ff_dropout)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, time_series: torch.Tensor, mask: torch.Tensor):
        """
        Forward pass.
        Args:
            time_series: (B, seq_len, num_features)
            mask: (B, seq_len) —— 1 表示有效，0 表示 padding
        Returns:
            local_embeddings: (B, seq_len, num_features, d_proj)
        """
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(-1)  # (B, T, 1)

        B, seq_len, num_features = time_series.shape
        assert num_features == self.num_features, f"Expected {self.num_features} features, got {num_features}"

        # Pad sequence to be divisible by patch_size
        padded_length = math.ceil(seq_len / self.patch_size) * self.patch_size
        if padded_length > seq_len:
            pad_amount = padded_length - seq_len
            time_series = F.pad(time_series, (0, 0, 0, pad_amount), value=0.0)
            mask = F.pad(mask, (0, pad_amount), value=0)

        # Reshape into patches
        num_patches = padded_length // self.patch_size
        patches = time_series.view(B, num_patches, self.patch_size, num_features)
        patches = patches.permute(0, 3, 1, 2).contiguous()  # (B, F, P, PS)
        patches = patches.view(B, num_features * num_patches, self.patch_size)  # (B, L, PS)

        # Embed patches
        embedded_patches = self.embedding_layer(patches)  # (B, L, d_model)

        # Inject feature ID if multi-variate
        if self.feature_embed is not None:
            device = time_series.device
            feature_id = torch.arange(num_features, device=device).repeat_interleave(num_patches)  # (L,)
            feature_id = feature_id.unsqueeze(0).expand(B, -1)  # (B, L)
            feature_emb = self.feature_embed(feature_id)  # (B, L, d_model)
            embedded_patches = embedded_patches + feature_emb

        # Apply Mamba layers
        output = embedded_patches
        for layer in self.mamba_layers:
            output = layer(output)

        # Project back to local embeddings
        patch_proj = self.projection_layer(output)  # (B, L, PS * d_proj)
        local_embeddings = patch_proj.view(B, num_features, num_patches, self.patch_size, self.d_proj)
        local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4)  # (B, P, PS, F, d_proj)
        local_embeddings = local_embeddings.contiguous().view(B, -1, num_features, self.d_proj)
        local_embeddings = local_embeddings[:, :seq_len, :, :]  # Trim padding

        return local_embeddings
