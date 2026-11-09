"""Paged attention using SGLang's real request and KV pools."""

import torch
import gaudirt

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from .kv_pool import GaudiRTMHATokenToKVPool


class GaudiRTPagedAttnBackend(AttentionBackend):
    def __init__(self, model_runner):
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        pool = self.token_to_kv_pool
        if not isinstance(pool, GaudiRTMHATokenToKVPool):
            raise ValueError("GaudiRT attention requires its head-major MHA KV pool")
        if pool.dtype != torch.bfloat16 or pool.v_head_dim != pool.head_dim:
            raise ValueError("GaudiRT attention requires matching BF16 K/V heads")
        if model_runner.ps.tp_size != 1:
            raise ValueError("GaudiRT attention currently supports tensor parallel size 1")
        self.query_heads = model_runner.model_config.num_attention_heads
        if self.query_heads % pool.head_num:
            raise ValueError("GaudiRT attention requires an integral GQA head ratio")
        self.vocabulary = model_runner.model_config.vocab_size
        if not isinstance(self.vocabulary, int) or self.vocabulary <= 0:
            raise ValueError("GaudiRT attention requires a positive model vocabulary")
        self.metadata = None
        self.forward_batch = None
        self.positions = None
        self.rotary_positions = None
        self.rotary_header = None
        self.rotary_context = None
        self.row_locations = None
        self.row_requests = None
        self.attention_plan = None
        self.last_token_indices = None

    def init_forward_metadata(self, forward_batch):
        self.metadata = self.forward_batch = self.positions = None
        self.rotary_positions = self.rotary_header = self.rotary_context = None
        self.row_locations = self.row_requests = self.attention_plan = None
        self.last_token_indices = None
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            prefixes = forward_batch.extend_prefix_lens_cpu
            lengths = forward_batch.extend_seq_lens_cpu
            if prefixes is None or lengths is None:
                raise ValueError("GaudiRT attention requires host extend lengths")
        elif forward_batch.forward_mode == ForwardMode.DECODE:
            if forward_batch.seq_lens_cpu is None or forward_batch.seq_lens_cpu.device.type != "cpu":
                raise ValueError("GaudiRT attention requires host decode sequence lengths")
            if forward_batch.positions.device.type != "cpu" or forward_batch.positions.dtype != torch.int64:
                raise ValueError("GaudiRT attention requires host decode positions before metadata preparation")
            prefixes = forward_batch.positions.tolist()
            lengths = [1] * len(prefixes)
        else:
            raise ValueError("GaudiRT attention supports extend and decode only")
        pool = self.token_to_kv_pool
        arguments = (forward_batch.out_cache_loc, forward_batch.req_pool_indices, forward_batch.positions,
                     self.req_to_token_pool.req_to_token, prefixes, lengths,
                     pool.get_key_buffer(pool.start_layer).size(0), forward_batch.input_ids, self.vocabulary)
        if forward_batch.forward_mode == ForwardMode.DECODE:
            self.metadata = torch.classes.gaudirt.PagedMetadata._from_scheduler_fused(
                *arguments, self.query_heads, pool.head_num, pool.head_dim
            )
            forward_batch.positions = self.metadata.model_positions()
        else:
            self.metadata = torch.classes.gaudirt.PagedMetadata._from_scheduler(*arguments)
        (tokens, self.last_token_indices, self.rotary_positions, self.rotary_header,
         self.row_locations, self.row_requests, self.attention_plan, self.rotary_context) = \
            self.metadata.forward_tensors(self.query_heads, pool.head_num, pool.head_dim)
        tokens._gaudirt_validated_vocab = self.vocabulary
        self.last_token_indices._gaudirt_validated_rows = tokens.numel()
        forward_batch.input_ids = tokens
        torch._dynamo.mark_static(self.last_token_indices, 0)
        torch._dynamo.mark_dynamic(forward_batch.input_ids, 0, min=1, max=4096)
        self.forward_batch = forward_batch
        self.positions = forward_batch.positions
        torch._dynamo.mark_dynamic(self.attention_plan, 0, min=1, max=536870912)
        for tensor in (self.rotary_positions, self.row_locations, self.row_requests):
            torch._dynamo.mark_dynamic(tensor, 0, min=1, max=4096)

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
        return torch.ops.gaudirt._paged_attn_validated.default(
            q if q.ndim == 2 else q.reshape(rows, -1),
            k if k.ndim == 2 else k.reshape(rows, -1),
            v if v.ndim == 2 else v.reshape(rows, -1),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.row_locations, self.row_requests, self.rotary_positions, self.attention_plan,
            self.rotary_header, layer.qk_head_dim, layer.scaling,
        )

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache, kwargs)

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache, kwargs)

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache, kwargs)
