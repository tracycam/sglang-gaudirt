"""Paged attention using SGLang's real request and KV pools."""

import weakref

import torch
import gaudirt

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from .kv_pool import GaudiRTMHATokenToKVPool


def _inclusive_sliding_window(window_size):
    if window_size is not None and int(window_size) >= 0:
        return int(window_size) + 1
    return -1


class GaudiRTPagedAttnBackend(AttentionBackend):
    def __init__(self, model_runner):
        self._model_runner_ref = weakref.ref(model_runner)
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        pool = self.token_to_kv_pool
        if not isinstance(pool, GaudiRTMHATokenToKVPool):
            raise ValueError("GaudiRT attention requires its head-major MHA KV pool")
        if pool.dtype != torch.bfloat16 or pool.v_head_dim != pool.head_dim:
            raise ValueError("GaudiRT attention requires matching BF16 K/V heads")
        tp_size = model_runner.ps.tp_size
        if tp_size not in (1, 2, 4, 8):
            raise ValueError("GaudiRT attention supports tensor parallel sizes 1, 2, 4, and 8")
        self.query_heads = model_runner.model_config.get_num_attention_heads(tp_size)
        if self.query_heads <= 0 or self.query_heads % pool.head_num:
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
        self.swa_window = _inclusive_sliding_window(getattr(model_runner, "sliding_window_size", None))
        self.swa_attention_plan = None
        self._swa_windows = None
        self.swa_attention_plans = {}
        self.last_token_indices = None

    def _ensure_swa_windows(self):
        if self._swa_windows is not None:
            return self._swa_windows
        mr = self._model_runner_ref()
        wins = set()
        if self.swa_window > 0:
            wins.add(self.swa_window)
        model = getattr(mr, "model", None) if mr is not None else None
        if model is None:
            return tuple(sorted(wins))
        for mod in model.modules():
            w = _inclusive_sliding_window(getattr(mod, "sliding_window_size", None))
            if w > 0:
                wins.add(w)
        self._swa_windows = tuple(sorted(wins))
        return self._swa_windows

    def init_forward_metadata(self, forward_batch):
        self.metadata = self.forward_batch = self.positions = None
        self.rotary_positions = self.rotary_header = self.rotary_context = None
        self.row_locations = self.row_requests = self.attention_plan = None
        self.swa_attention_plan = None
        self.swa_attention_plans = {}
        self.last_token_indices = None
        if forward_batch.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED, ForwardMode.SPLIT_PREFILL):
            prefixes = forward_batch.extend_prefix_lens_cpu
            lengths = forward_batch.extend_seq_lens_cpu
            if prefixes is None or lengths is None:
                raise ValueError("GaudiRT attention requires host extend lengths")
            if isinstance(prefixes, torch.Tensor):
                prefixes = prefixes.tolist()
            if isinstance(lengths, torch.Tensor):
                lengths = lengths.tolist()
        elif forward_batch.forward_mode == ForwardMode.DECODE:
            if forward_batch.seq_lens_cpu is None or forward_batch.seq_lens_cpu.device.type != "cpu":
                raise ValueError("GaudiRT attention requires host decode sequence lengths")
            if forward_batch.positions.device.type == "cpu":
                if forward_batch.positions.dtype != torch.int64:
                    raise ValueError("GaudiRT attention requires host int64 decode positions")
                prefixes = forward_batch.positions.tolist()
            else:
                prefixes = [int(length) - 1 for length in forward_batch.seq_lens_cpu.tolist()]
            lengths = [1] * len(prefixes)
        else:
            raise ValueError(
                f"GaudiRT attention does not support forward mode {forward_batch.forward_mode}"
            )
        pool = self.token_to_kv_pool
        has_tokens = forward_batch.input_ids is not None
        token_arg = forward_batch.input_ids if has_tokens else None
        vocab_arg = self.vocabulary if has_tokens else 0
        arguments = (forward_batch.out_cache_loc, forward_batch.req_pool_indices, forward_batch.positions,
                     self.req_to_token_pool.req_to_token, prefixes, lengths,
                     pool.get_key_buffer(pool.start_layer).size(0), token_arg, vocab_arg)
        use_fused = (forward_batch.forward_mode == ForwardMode.DECODE or
                     forward_batch.positions.device.type == "cpu")
        if use_fused:
            self.metadata = torch.classes.gaudirt.PagedMetadata._from_scheduler_fused(
                *arguments, self.query_heads, pool.head_num, pool.head_dim
            )
            forward_batch.positions = self.metadata.model_positions()
        else:
            self.metadata = torch.classes.gaudirt.PagedMetadata._from_scheduler(*arguments)
        if has_tokens:
            (tokens, self.last_token_indices, self.rotary_positions, self.rotary_header,
             self.row_locations, self.row_requests, self.attention_plan, self.rotary_context) = \
                self.metadata.forward_tensors(self.query_heads, pool.head_num, pool.head_dim)
            tokens._gaudirt_validated_vocab = self.vocabulary
            self.last_token_indices._gaudirt_validated_rows = tokens.numel()
            forward_batch.input_ids = tokens
        else:
            self.last_token_indices = self.metadata.last_token_indices()
            self.rotary_positions = self.metadata.rotary_positions()
            self.rotary_header = self.metadata.rotary_header()
            self.row_locations = self.metadata.row_locations()
            self.row_requests = self.metadata.row_requests()
            self.attention_plan = self.metadata.attention_plan(self.query_heads, pool.head_num, pool.head_dim)
            self.rotary_context = self.metadata.context()
            self.last_token_indices._gaudirt_validated_rows = self.metadata.rows()
        swa_windows = self._ensure_swa_windows()
        if swa_windows and hasattr(self.metadata, "attention_plan_for"):
            for w in swa_windows:
                self.swa_attention_plans[w] = self.metadata.attention_plan_for(
                    self.query_heads, pool.head_num, pool.head_dim, w
                )
            self.swa_attention_plan = self.swa_attention_plans.get(
                self.swa_window, next(iter(self.swa_attention_plans.values()), None)
            )
        self.forward_batch = forward_batch
        self.positions = forward_batch.positions

    def mark_capture_inputs(self):
        """Declare symbolic metadata only when the model graph is captured."""
        torch._dynamo.mark_static(self.last_token_indices, 0)
        if self.forward_batch.input_ids is not None:
            torch._dynamo.mark_dynamic(self.forward_batch.input_ids, 0)
        torch._dynamo.mark_dynamic(self.attention_plan, 0)
        for plan in self.swa_attention_plans.values():
            torch._dynamo.mark_dynamic(plan, 0)
        for tensor in (self.rotary_positions, self.row_locations, self.row_requests):
            torch._dynamo.mark_dynamic(tensor, 0)

    def _forward(self, q, k, v, layer, forward_batch, save_kv_cache, kwargs):
        if self.metadata is None or forward_batch is not self.forward_batch:
            raise RuntimeError("GaudiRT attention metadata is not initialized for this batch")
        if not save_kv_cache or kwargs or k is None or v is None:
            raise ValueError("GaudiRT attention requires Q/K/V and KV writes")
        if (layer.attn_type != AttentionType.DECODER or layer.is_cross_attention or
                layer.qk_head_dim != layer.v_head_dim or layer.k_scale is not None or
                layer.v_scale is not None):
            raise ValueError("GaudiRT attention supports unquantized causal decoder attention only")
        softcap = float(layer.logit_cap or 0.0)
        sliding_window = _inclusive_sliding_window(layer.sliding_window_size)
        plan = self.attention_plan
        if sliding_window > 0:
            if sliding_window in self.swa_attention_plans:
                plan = self.swa_attention_plans[sliding_window]
            elif self.swa_attention_plan is not None and self.swa_window == sliding_window:
                plan = self.swa_attention_plan
            elif hasattr(self.metadata, "attention_plan_for"):
                plan = self.metadata.attention_plan_for(
                    self.query_heads, self.token_to_kv_pool.head_num, layer.qk_head_dim, sliding_window
                )
        rows = q.shape[0]
        return torch.ops.gaudirt._paged_attn_validated.default(
            q if q.ndim == 2 else q.reshape(rows, -1),
            k if k.ndim == 2 else k.reshape(rows, -1),
            v if v.ndim == 2 else v.reshape(rows, -1),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.row_locations, self.row_requests, self.rotary_positions, plan,
            self.rotary_header, layer.qk_head_dim, layer.scaling, softcap, sliding_window,
        )

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache, kwargs)

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache, kwargs)

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache, kwargs)
