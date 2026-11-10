"""Single-device SGLang platform backed by GaudiRT."""

import os
from contextlib import contextmanager

import torch
import gaudirt

from sglang.srt.platforms.interface import PlatformEnum, SRTPlatform


def _hbm_budget():
    value = int(os.environ.get("GAUDIRT_HBM_BYTES", str(80 * 1024**3)))
    if value <= 0:
        raise ValueError("GAUDIRT_HBM_BYTES must be positive")
    return value


class GaudiRTSRTPlatform(SRTPlatform):
    _enum = PlatformEnum.OOT
    device_name = "gaudirt"
    device_type = "gaudirt"

    def __init__(self):
        self._registered = False

    def get_dispatch_key_name(self):
        return "gaudirt"

    def get_default_attention_backend(self):
        return "gaudirt"

    def validate_input_token_ids(self, input_ids, vocabulary_size):
        if vocabulary_size <= 0 or any(sequence is None for sequence in input_ids) or any(
            token < 0 or token >= vocabulary_size
            for sequence in input_ids for token in sequence
        ):
            raise ValueError("GaudiRT input token ID is outside the model vocabulary")

    def attention_accepts_flat_kv(self):
        return True

    def get_last_token_indices(self):
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        if backend.last_token_indices is None:
            raise RuntimeError("GaudiRT attention metadata has no last-token indices")
        return backend.last_token_indices

    def apply_server_args_defaults(self, server_args):
        if any(getattr(server_args, name) != 1 for name in (
            "tp_size", "pp_size", "dp_size", "nnodes",
            "tokenizer_worker_num", "detokenizer_worker_num",
        )):
            raise ValueError("GaudiRT currently requires one device and one tokenizer/detokenizer worker")
        if server_args.page_size not in (None, 1, 16, 64):
            raise ValueError("GaudiRT supports KV page sizes 1, 16, and 64")
        if server_args.page_size is None:
            server_args.page_size = 1
        server_args.disable_cuda_graph = True

    @contextmanager
    def rope_cache_initialization_context(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.device("cpu"):
                yield
        finally:
            torch.set_num_threads(threads)

    def prepare_rope_cache(self, cache):
        return gaudirt._reserve_rotary_cache(cache)

    def rope_cache_computation_device(self, cache_device):
        return torch.device("cpu")

    def get_compile_backend(self, mode=None):
        if mode is not None:
            raise ValueError("GaudiRT does not support compilation modes")
        return "gaudirt"

    def prepare_model_forward(self, forward):
        from gaudirt.compiler import compile

        return compile(forward, dynamic_arg_indices=(0, 1))

    def requires_graceful_worker_shutdown(self):
        return True

    def write_req_to_token_pool(
        self, pool, request_indices, prefix_lengths, extend_lengths, prefixes, locations,
        *, is_encoder_decoder=False,
    ):
        if is_encoder_decoder:
            raise ValueError("GaudiRT supports decoder-only request mappings")
        from .req_to_token import write_req_to_token_pool

        write_req_to_token_pool(pool, request_indices, prefix_lengths, extend_lengths, prefixes, locations)
        return True

    def paged_decode_last_locations(self, pool, request_indices, sequence_lengths):
        from .req_to_token import read_decode_last_locations

        return read_decode_last_locations(pool, request_indices, sequence_lengths)

    def prepare_decode_positions(self, seq_lens_cpu, device):
        from .positions import decode_positions

        return decode_positions(seq_lens_cpu, device)

    def gather_scheduler_values(self, buffer, indices):
        return torch.ops.gaudirt._index_select_validated(buffer, 0, indices)

    def store_scheduler_values(self, buffer, indices, values):
        torch.ops.gaudirt._index_put_validated_(buffer, indices, values.to(buffer.dtype))

    def defer_eager_decode_positions(self) -> bool:
        return True

    def prepare_extend_positions(self, prefix_lens_cpu, seq_lens_cpu, total, device):
        from .positions import extend_positions

        return extend_positions(prefix_lens_cpu, seq_lens_cpu, total, device)

    def get_device_total_memory(self, device_id=0):
        if device_id != 0:
            raise ValueError("GaudiRT supports device 0 only")
        return gaudirt.mem_get_info()[1] if gaudirt.is_initialized() else _hbm_budget()

    def get_current_memory_usage(self, device=None):
        if device not in (None, 0, "gaudirt:0", torch.device("gaudirt:0")):
            raise ValueError("GaudiRT supports device 0 only")
        return float(gaudirt.memory_allocated()) if gaudirt.is_initialized() else 0.0

    def get_available_memory(self, device_id=0):
        if device_id != 0:
            raise ValueError("GaudiRT supports device 0 only")
        return gaudirt.mem_get_info() if gaudirt.is_initialized() else (_hbm_budget(), _hbm_budget())

    def get_device(self, device_id=None):
        if device_id not in (None, 0):
            raise ValueError("GaudiRT supports device 0 only")
        return "gaudirt" if device_id is None else "gaudirt:0"

    def empty_cache(self):
        gaudirt.empty_cache()

    def synchronize(self):
        gaudirt.synchronize()

    def get_mha_kv_pool_cls(self):
        from .kv_pool import GaudiRTMHATokenToKVPool

        return GaudiRTMHATokenToKVPool

    def get_paged_allocator_cls(self):
        from .allocator import GaudiRTTokenToKVPoolAllocator

        return GaudiRTTokenToKVPoolAllocator

    def init_backend(self):
        if self._registered:
            return
        if not gaudirt.is_initialized():
            gaudirt.initialize(hbm_bytes=_hbm_budget())

        from sglang.srt.layers.attention.attention_registry import register_attention_backend
        from sglang.srt.server_args import add_attention_backend_choices
        from .attention import GaudiRTPagedAttnBackend
        from .ops import register_ops

        register_attention_backend("gaudirt")(GaudiRTPagedAttnBackend)
        add_attention_backend_choices(["gaudirt"])
        register_ops()
        self._registered = True
