"""SGLang MHA pool backed by GaudiRT's combined head-major storage."""

import torch
import gaudirt

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


class GaudiRTMHATokenToKVPool(MHATokenToKVPool):
    def _create_buffers(self):
        if (self.is_quantized_kv_cache or self.post_capture_active or
                self.dtype != torch.bfloat16 or self.v_head_dim != self.head_dim or
                self.enable_custom_mem_pool):
            raise ValueError("GaudiRT requires an unquantized BF16 MHA pool")
        self.kv_cache_layout = "gaudirt_head_major"
        self.k_scale_buffer = self.v_scale_buffer = None
        self.dq_k_buffer = self.dq_v_buffer = None
        self._combined_kv = gaudirt._C._allocate_paged_kv(
            self.layer_num, self.size + self.page_size, self.head_num, self.head_dim
        )
        # The views preserve SGLang's [slot, head, dim] interface while their
        # backing storage is [layer, K/V, head, slot, dim].
        self.k_buffer = [self._combined_kv[i, 0].transpose(0, 1) for i in range(self.layer_num)]
        self.v_buffer = [self._combined_kv[i, 1].transpose(0, 1) for i in range(self.layer_num)]
        self._kv_buffer_descs = []

    def get_kv_size_bytes(self):
        key_bytes = sum(t.numel() * t.element_size() for t in self.k_buffer)
        value_bytes = sum(t.numel() * t.element_size() for t in self.v_buffer)
        reserved_tail = self._combined_kv.untyped_storage().nbytes() - key_bytes - value_bytes
        return key_bytes + reserved_tail, value_bytes

    def _clear_buffers(self):
        super()._clear_buffers()
        self._combined_kv = None

    def _init_kv_copy_and_warmup(self):
        raise RuntimeError("GaudiRT KV slot moves are not implemented")

    def set_kv_buffer(self, *args, **kwargs):
        raise RuntimeError("GaudiRT KV writes must use paged attention")

    def move_kv_cache(self, *args, **kwargs):
        raise RuntimeError("GaudiRT KV slot moves are not implemented")

    def get_cpu_copy(self, *args, **kwargs):
        raise RuntimeError("GaudiRT KV offload is not implemented")

    def load_cpu_copy(self, *args, **kwargs):
        raise RuntimeError("GaudiRT KV offload is not implemented")

    def get_contiguous_buf_infos(self):
        raise RuntimeError("GaudiRT KV transfer is not implemented")
