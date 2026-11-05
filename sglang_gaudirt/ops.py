"""SGLang operator dispatch into GaudiRT device primitives."""

import torch


def rms_norm(self, x, residual=None, post_residual_addition=None):
    if (post_residual_addition is not None or self.variance_size_override is not None or
            self.cast_x_before_out_mul or self.fp32_residual or self.override_orig_dtype is not None or
            not self.has_weight):
        raise ValueError("GaudiRT RMSNorm does not support these options")
    if residual is None:
        return torch.ops.gaudirt.rms_norm.default(x, self.weight, self.variance_epsilon)
    return torch.ops.gaudirt.fused_add_rms_norm.default(x, residual, self.weight, self.variance_epsilon)


def silu_and_mul(self, x):
    return torch.ops.gaudirt.silu_and_mul.default(x)


def rotary_embedding(self, positions, query, key, offsets=None, fused_set_kv_buffer_arg=None):
    if offsets is not None or fused_set_kv_buffer_arg is not None or hasattr(self, "sin_cos_cache"):
        raise ValueError("GaudiRT RoPE does not support offsets or fused KV writes")
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    if backend.metadata is None or positions is not backend.positions:
        raise RuntimeError("GaudiRT RoPE requires the active batch's validated positions")
    self._match_cos_sin_cache_dtype(query)
    return torch.ops.gaudirt.rotary_embedding.default(
        query, key, self.cos_sin_cache, backend.metadata,
        self.head_size, self.rotary_dim, self.is_neox_style,
    )


def register_ops():
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.layers.utils.multi_platform import MultiPlatformOp

    for cls, fn in ((RMSNorm, rms_norm), (SiluAndMul, silu_and_mul),
                    (RotaryEmbedding, rotary_embedding)):
        MultiPlatformOp.register_oot_forward(cls, fn, "gaudirt")
