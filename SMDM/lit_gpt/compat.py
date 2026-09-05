"""Native PyTorch fallbacks for optional SMDM CUDA extensions."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from xformers.ops import SwiGLU as SwiGLU
except ImportError:
    class SwiGLU(nn.Module):
        # ponytail: use three Linear layers when xformers is unavailable; the
        # optimized packed kernel can be restored without changing checkpoint keys.
        def __init__(self, in_features, hidden_features, bias=False, _pack_weights=False, **kwargs):
            super().__init__()
            self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
            self.w3 = nn.Linear(hidden_features, in_features, bias=bias)

        def forward(self, x):
            return self.w3(F.silu(self.w1(x)) * self.w2(x))


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    """Match flash-attn's [B, T, H, D] interface using PyTorch SDPA."""
    q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
    if q.shape[1] != k.shape[1]:
        repeats = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    y = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=dropout_p,
        scale=softmax_scale or 1.0 / math.sqrt(q.shape[-1]),
        is_causal=causal,
    )
    return y.transpose(1, 2)


def apply_rotary_emb_func(x, cos, sin, interleaved=False, inplace=False):
    """Pure PyTorch rotary embedding for SMDM's [B, T, H, D] tensors."""
    rotary_dim = 2 * cos.shape[-1]
    x_ro, x_tail = x[..., :rotary_dim], x[..., rotary_dim:]
    cos = cos[: x.shape[1]].to(dtype=x.dtype)[None, :, None, :]
    sin = sin[: x.shape[1]].to(dtype=x.dtype)[None, :, None, :]
    if interleaved:
        x1, x2 = x_ro[..., ::2], x_ro[..., 1::2]
        rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    else:
        x1, x2 = x_ro.chunk(2, dim=-1)
        rotated = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return torch.cat((rotated, x_tail), dim=-1)
