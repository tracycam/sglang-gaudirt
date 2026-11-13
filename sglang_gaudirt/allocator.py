"""SGLang KV slot allocator backed by device page-index primitives."""

import torch
import gaudirt

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator


class GaudiRTTokenToKVPoolAllocator(TokenToKVPoolAllocator):
    def __init__(self, size, page_size, dtype, device, kvcache, need_sort):
        if not isinstance(page_size, int) or page_size < 1 or need_sort or size % page_size or size >= 2**31:
            raise ValueError("GaudiRT supports unsorted KV pages with positive integer page_size")
        if page_size == 1:
            super().__init__(size, dtype, device, kvcache, need_sort)
        else:
            BaseTokenToKVPoolAllocator.__init__(self, size, page_size, dtype, device, kvcache, need_sort)
            self.num_pages = size // page_size
            self.empty_last_slot = torch.empty((1,), dtype=torch.int64, device=device)
            self.clear()

    def clear(self):
        if self.page_size == 1:
            # The request table and attention metadata store 32-bit slot IDs.
            # Keep the allocator in that same representation so every decode
            # write does not require a separate device cast.
            self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int32).to(self.device)
            self.release_pages = torch.empty((0,), dtype=torch.int32, device=self.device)
            self.is_not_in_free_group = True
            self.free_group = []
            return
        # This is scheduler metadata constructed on reset, not model data.
        # Preparing it on the host avoids a shape-specific device recipe when
        # the configured pool capacity changes between processes.
        self.free_pages = torch.arange(1, self.num_pages + 1, dtype=torch.int64).to(self.device)
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)
        self.is_not_in_free_group = True
        self.free_group = []

    def available_size(self):
        if self.page_size == 1:
            return super().available_size()
        return len(self.free_pages) * self.page_size

    def _allocate(self, last_loc, prefixes, lengths):
        count = sum((prefix + length + self.page_size - 1) // self.page_size -
                    (prefix + self.page_size - 1) // self.page_size
                    for prefix, length in zip(prefixes, lengths))
        if count > len(self.free_pages):
            return None
        if last_loc.dtype != torch.int64:
            last_loc = last_loc.to(torch.int64)
        result, used = gaudirt._C._allocate_paged_slots(last_loc, self.free_pages, prefixes, lengths, self.page_size)
        if used != count:
            raise RuntimeError("GaudiRT paged allocator returned an inconsistent page count")
        self.free_pages = self.free_pages[used:]
        return result

    def alloc(self, need_size):
        if self.page_size == 1:
            return super().alloc(need_size)
        if need_size < 0 or need_size % self.page_size:
            raise ValueError("paged KV allocations must be page-aligned")
        if not need_size:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        return self._allocate(self.empty_last_slot, [0], [need_size])

    def alloc_extend(self, prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc,
                     extend_num_tokens, num_new_pages=None):
        if self.page_size == 1:
            raise RuntimeError("page-size-one allocation uses alloc")
        prefixes, ends = prefix_lens_cpu.tolist(), seq_lens_cpu.tolist()
        lengths = [end - prefix for prefix, end in zip(prefixes, ends)]
        if len(prefixes) != len(ends) or sum(lengths) != extend_num_tokens:
            raise ValueError("inconsistent paged extend lengths")
        return self._allocate(last_loc, prefixes, lengths)

    def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
        if self.page_size == 1:
            raise RuntimeError("page-size-one allocation uses alloc")
        ends = seq_lens_cpu.tolist()
        return self._allocate(last_loc, [end - 1 for end in ends], [1] * len(ends))

    def free(self, free_index):
        if self.page_size == 1:
            if free_index.dtype != torch.int32:
                free_index = free_index.to(torch.int32)
            return super().free(free_index)
        if not free_index.numel():
            return
        if not self.is_not_in_free_group:
            self.free_group.append(free_index)
            return
        if free_index.dtype != torch.int64:
            free_index = free_index.to(torch.int64)
        pages = gaudirt._C._release_paged_slots(free_index, self.page_size, self.num_pages)
        self.free_pages = torch.cat((pages, self.free_pages))
