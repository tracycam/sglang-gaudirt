"""Paged attention using SGLang's real request and KV pools."""

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class GaudiRTPagedAttnBackend(AttentionBackend):
    def __init__(self, model_runner):
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        pool = self.token_to_kv_pool
        if not isinstance(pool, MHATokenToKVPool) or pool.kv_cache_layout != "nhd":
            raise ValueError("GaudiRT attention requires an NHD MHA KV pool")
        if pool.dtype != torch.bfloat16 or pool.v_head_dim != pool.head_dim:
            raise ValueError("GaudiRT attention requires matching BF16 K/V heads")
        self.metadata = None
        self.forward_batch = None
        self.positions = None
        self.rotary_positions = None
        self.rotary_header = None
        self.rotary_context = None

    def init_forward_metadata(self, forward_batch):
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            prefixes = forward_batch.extend_prefix_lens_cpu
            lengths = forward_batch.extend_seq_lens_cpu
            if prefixes is None or lengths is None:
                raise ValueError("GaudiRT attention requires host extend lengths")
        elif forward_batch.forward_mode == ForwardMode.DECODE:
            if forward_batch.seq_lens_cpu is None or forward_batch.seq_lens_cpu.device.type != "cpu":
                raise ValueError("GaudiRT attention requires host decode sequence lengths")
            prefixes = [length - 1 for length in forward_batch.seq_lens_cpu.tolist()]
            lengths = [1] * len(prefixes)
        else:
            raise ValueError("GaudiRT attention supports extend and decode only")
        pool = self.token_to_kv_pool
        self.metadata = torch.classes.gaudirt.PagedMetadata.from_device(
            forward_batch.out_cache_loc,
            forward_batch.req_pool_indices,
            forward_batch.positions,
            self.req_to_token_pool.req_to_token,
            prefixes,
            lengths,
            pool.get_key_buffer(pool.start_layer).size(0),
        )
        self.forward_batch = forward_batch
        self.positions = forward_batch.positions
        self.rotary_positions = self.metadata.rotary_positions()
        self.rotary_header = self.metadata.rotary_header()
        self.rotary_context = self.metadata.context()
        torch._dynamo.mark_dynamic(self.rotary_positions, 0, min=1, max=4096)

    def _forward(self, q, k, v, layer, forward_batch, save_kv_cache, kwargs):
        if self.metadata is None or forward_batch is not self.forward_batch:
            raise RuntimeError("GaudiRT attention metadata is not initialized for this batch")
        if not save_kv_cache or kwargs or k is None or v is None:
            raise ValueError("GaudiRT attention requires Q/K/V and KV writes")
        if (layer.attn_type != AttentionType.DECODER or layer.is_cross_attention or
                layer.sliding_window_size != -1 or layer.logit_cap != 0 or
                layer.qk_head_dim != layer.v_head_dim or layer.k_scale is not None or
                layer.v_scale is not None):
            raise ValueError("GaudiRT attention supports unquantized causal full attention only")
        rows = q.shape[0]
        return torch.ops.gaudirt.paged_attn.default(
            q.reshape(rows, -1), k.reshape(rows, -1), v.reshape(rows, -1),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.metadata, layer.qk_head_dim, layer.scaling,
        )

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache, kwargs)

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache, kwargs)
