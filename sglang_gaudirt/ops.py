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
    return torch.ops.gaudirt._rotary_embedding_validated.default(
        query, key, self.cos_sin_cache, backend.rotary_positions, backend.rotary_header,
        backend.rotary_context, self.head_size, self.rotary_dim, self.is_neox_style,
    )


_collective_ops_registered = False


def _resolve_device_group(group_name: str):
    from sglang.srt.distributed.parallel_state import _groups

    entry = _groups.get(group_name)
    group = entry() if callable(entry) else entry
    if group is None:
        raise ValueError(f"GaudiRT collective: SGLang group {group_name!r} is not active")
    return group


def _register_collective_custom_ops():
    global _collective_ops_registered
    if _collective_ops_registered:
        return
    # Ensure sglang custom ops are defined on torch.ops.sglang before attaching PrivateUse1 kernels.
    import sglang.srt.distributed.parallel_state  # noqa: F401

    @torch.library.impl("sglang::inplace_all_reduce", "PrivateUse1")
    def _gaudirt_inplace_all_reduce(tensor: torch.Tensor, group_name: str) -> None:
        group = _resolve_device_group(group_name)
        if group.world_size > 1:
            torch.distributed.all_reduce(tensor, group=group.device_group)

    @torch.library.impl("sglang::outplace_all_reduce", "PrivateUse1")
    def _gaudirt_outplace_all_reduce(
        tensor: torch.Tensor, group_name: str, outplace_all_reduce_method: str
    ) -> torch.Tensor:
        del outplace_all_reduce_method
        group = _resolve_device_group(group_name)
        out = tensor.clone()
        if group.world_size > 1:
            torch.distributed.all_reduce(out, group=group.device_group)
        return out

    @torch.library.impl("sglang::reg_all_gather_into_tensor", "PrivateUse1")
    def _gaudirt_reg_all_gather_into_tensor(
        output: torch.Tensor, input: torch.Tensor, group_name: str
    ) -> None:
        group = _resolve_device_group(group_name)
        if group.world_size == 1:
            output.copy_(input)
        else:
            torch.distributed.all_gather_into_tensor(output, input, group=group.device_group)

    @torch.library.impl("sglang::reg_reduce_scatter_tensor", "PrivateUse1")
    def _gaudirt_reg_reduce_scatter_tensor(
        output: torch.Tensor, input: torch.Tensor, group_name: str
    ) -> None:
        group = _resolve_device_group(group_name)
        if group.world_size == 1:
            output.copy_(input)
        else:
            torch.distributed.reduce_scatter_tensor(output, input, group=group.device_group)

    @torch.library.impl("sglang::reg_all_to_all_single", "PrivateUse1")
    def _gaudirt_reg_all_to_all_single(
        output: torch.Tensor, input: torch.Tensor, group_name: str
    ) -> None:
        group = _resolve_device_group(group_name)
        if group.world_size == 1:
            output.copy_(input)
        else:
            torch.distributed.all_to_all_single(output, input, group=group.device_group)

    _collective_ops_registered = True


def register_ops():
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.layers.utils.multi_platform import MultiPlatformOp

    for cls, fn in ((RMSNorm, rms_norm), (SiluAndMul, silu_and_mul),
                    (RotaryEmbedding, rotary_embedding)):
        MultiPlatformOp.register_oot_forward(cls, fn, "gaudirt")
    _register_collective_custom_ops()
