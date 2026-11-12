"""Multi-device SGLang platform backed by GaudiRT."""

import os
from contextlib import contextmanager

import torch
import gaudirt

from sglang.srt.platforms.interface import PlatformEnum, SRTPlatform


def _install_graceful_worker_shutdown():
    import zmq
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.managers.io_struct import ShutdownReq, sock_send
    from sglang.srt.utils.network import get_zmq_socket

    def _shutdown_workers_gracefully(self):
        watchdog = self.tokenizer_manager._subprocess_watchdog
        workers = dict(zip(watchdog._names, watchdog._processes))
        tp_size = int(getattr(self.server_args, "tp_size", 1))
        sched_names = [f"scheduler_{i}" for i in range(tp_size)]
        expected = set(sched_names) | {"detokenizer"}
        if set(workers) != expected:
            raise RuntimeError(
                f"graceful engine shutdown expected {sorted(expected)}, got {sorted(workers)}"
            )

        def wait_for(name):
            worker = workers[name]
            worker.join(timeout=60)
            if worker.is_alive() or worker.exitcode != 0:
                raise RuntimeError(f"{name} did not exit cleanly: {worker.exitcode}")

        context = None
        socket = None
        try:
            self.tokenizer_manager._dispatch_to_scheduler(ShutdownReq())
            for name in sched_names:
                wait_for(name)
            context = zmq.Context()
            socket = get_zmq_socket(context, zmq.PUSH, self.port_args.detokenizer_ipc_name, False)
            sock_send(socket, ShutdownReq())
            wait_for("detokenizer")
        except Exception:
            for worker in workers.values():
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5)
            raise
        finally:
            if socket is not None:
                socket.close(linger=0)
            if context is not None:
                context.term()

    Engine._shutdown_workers_gracefully = _shutdown_workers_gracefully


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

    def get_torch_distributed_backend_str(self) -> str:
        return "gaudirt"

    def supports_fp8(self):
        return True

    def supports_compressed_w8a8_fp8(self, weight_quant, input_quant):
        from compressed_tensors.quantization import QuantizationStrategy, QuantizationType

        common = bool(
            weight_quant is not None and input_quant is not None and
            weight_quant.type == input_quant.type == QuantizationType.FLOAT and
            weight_quant.num_bits == input_quant.num_bits == 8 and
            weight_quant.symmetric and not weight_quant.dynamic and
            input_quant.dynamic and input_quant.symmetric
        )
        if not common:
            return False
        if weight_quant.strategy == QuantizationStrategy.BLOCK:
            return (tuple(weight_quant.block_structure or ()) == (128, 128) and
                    input_quant.strategy == QuantizationStrategy.GROUP and input_quant.group_size == 128)
        return (weight_quant.strategy in (QuantizationStrategy.CHANNEL, QuantizationStrategy.TENSOR) and
                input_quant.strategy == QuantizationStrategy.TOKEN)

    def prepare_w8a8_fp8_weight(self, layer, strategy, scale_name):
        from compressed_tensors.quantization import QuantizationStrategy

        from .fp8 import prepare_block_weight, prepare_channel_weight

        if strategy in (QuantizationStrategy.BLOCK, "block"):
            prepare_block_weight(layer, scale_name)
        elif strategy in (QuantizationStrategy.CHANNEL, QuantizationStrategy.TENSOR, "channel_or_tensor"):
            prepare_channel_weight(layer, scale_name)
        else:
            raise NotImplementedError("GaudiRT FP8: weight strategy is not implemented")
        return True

    def get_w8a8_block_fp8_linear(self):
        from .fp8 import block_fp8_linear

        return block_fp8_linear

    def get_w8a8_fp8_linear(self):
        from .fp8 import fp8_linear

        return fp8_linear

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
        if server_args.tp_size not in (1, 2, 4, 8):
            raise ValueError("GaudiRT supports tensor parallel sizes 1, 2, 4, and 8")
        if server_args.tp_size > gaudirt.device_count():
            raise ValueError(
                f"GaudiRT tp_size={server_args.tp_size} exceeds visible devices ({gaudirt.device_count()})"
            )
        if any(getattr(server_args, name) != 1 for name in (
            "pp_size", "dp_size", "nnodes",
            "tokenizer_worker_num", "detokenizer_worker_num",
        )):
            raise ValueError("GaudiRT currently requires pp=dp=nnodes=1 and one tokenizer/detokenizer worker")
        if server_args.page_size not in (None, 1, 16, 64):
            raise ValueError("GaudiRT supports KV page sizes 1, 16, and 64")
        if server_args.page_size is None:
            server_args.page_size = 1
        server_args.disable_cuda_graph = True
        server_args.disable_custom_all_reduce = True
        if hasattr(server_args, "pre_warm_nccl"):
            server_args.pre_warm_nccl = False

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
        from sglang.srt.model_executor.forward_context import get_attn_backend

        model = getattr(forward, "__self__", None)
        if isinstance(model, torch.nn.Module):
            for mod in model.modules():
                gatherer = getattr(mod, "_logits_gatherer", None)
                if gatherer is not None and getattr(gatherer, "_state", None) is getattr(gatherer, "_UNINIT", object()):
                    gatherer._state = None

        return compile(forward, dynamic_arg_indices=(0, 1),
                       prepare_capture=lambda: get_attn_backend().mark_capture_inputs())

    def requires_graceful_worker_shutdown(self):
        _install_graceful_worker_shutdown()
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

    def advance_decode_lengths(self, seq_lens, orig_seq_lens):
        return torch.ops.gaudirt._advance_decode_lengths(seq_lens, orig_seq_lens)

    def defer_eager_decode_positions(self) -> bool:
        return True

    def copy_eager_inputs(self) -> bool:
        return False

    def prepare_extend_positions(self, prefix_lens_cpu, seq_lens_cpu, total, device):
        from .positions import extend_positions

        return extend_positions(prefix_lens_cpu, seq_lens_cpu, total, device)

    def get_device_total_memory(self, device_id=0):
        idx = gaudirt._resolve_device_index(device_id)
        if gaudirt.is_initialized() and idx == gaudirt.current_device():
            return gaudirt.mem_get_info(idx)[1]
        return _hbm_budget()

    def get_current_memory_usage(self, device=None):
        idx = gaudirt._resolve_device_index(device)
        if gaudirt.is_initialized() and idx == gaudirt.current_device():
            return float(gaudirt.memory_allocated(idx))
        return 0.0

    def get_available_memory(self, device_id=0):
        idx = gaudirt._resolve_device_index(device_id)
        if gaudirt.is_initialized() and idx == gaudirt.current_device():
            return gaudirt.mem_get_info(idx)
        return (_hbm_budget(), _hbm_budget())

    def get_device(self, device_id=None):
        if device_id is None:
            return "gaudirt"
        idx = gaudirt._resolve_device_index(device_id)
        return f"gaudirt:{idx}"

    def set_device(self, device):
        gaudirt.set_device(device)

    def empty_cache(self):
        if gaudirt.is_initialized():
            gaudirt.empty_cache()

    def synchronize(self):
        if gaudirt.is_initialized():
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
        gaudirt.configure_default_runtime(hbm_bytes=_hbm_budget())

        from sglang.srt.layers.attention.attention_registry import register_attention_backend
        from sglang.srt.server_args import add_attention_backend_choices
        from .attention import GaudiRTPagedAttnBackend
        from .ops import register_ops

        register_attention_backend("gaudirt")(GaudiRTPagedAttnBackend)
        add_attention_backend_choices(["gaudirt"])
        register_ops()
        _install_graceful_worker_shutdown()
        self._registered = True
