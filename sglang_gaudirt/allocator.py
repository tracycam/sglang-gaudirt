"""Single-token pages for the SGLang KV allocator interface."""

from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator


class GaudiRTTokenToKVPoolAllocator(TokenToKVPoolAllocator):
    def __init__(self, size, page_size, dtype, device, kvcache, need_sort):
        if page_size != 1 or need_sort:
            raise ValueError("GaudiRT supports unsorted KV page size 1 only")
        super().__init__(size, dtype, device, kvcache, need_sort)
