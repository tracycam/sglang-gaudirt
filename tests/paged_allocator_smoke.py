"""Exercise SGLang page allocation, partial pages, decode, and grouped free."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import torch
import sglang
import gaudirt
from sglang_gaudirt.allocator import GaudiRTTokenToKVPoolAllocator


def main():
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=8 * 1024**2)
    for page_size in (16, 64):
        allocator = GaudiRTTokenToKVPoolAllocator(64 * page_size, page_size, torch.bfloat16,
                                                  "gaudirt:0", None, False)
        initial = allocator.alloc(2 * page_size)
        assert initial.cpu().tolist() == list(range(page_size, 3 * page_size))
        prefixes = torch.tensor([7, 0], dtype=torch.int64)
        ends = torch.tensor([page_size + 3, page_size + 1], dtype=torch.int64)
        last = torch.tensor([page_size + 6, -1], dtype=torch.int64).to("gaudirt:0")
        lengths = (ends - prefixes).tolist()
        extended = allocator.alloc_extend(prefixes.to("gaudirt:0"), prefixes, ends.to("gaudirt:0"),
                                           ends, last, sum(lengths))
        expected = (list(range(page_size + 7, 2 * page_size)) +
                    list(range(3 * page_size, 3 * page_size + 3)) +
                    list(range(4 * page_size, 5 * page_size + 1)))
        assert extended.cpu().tolist() == expected
        after = ends + 1
        last = torch.tensor([3 * page_size + 2, 5 * page_size], dtype=torch.int64).to("gaudirt:0")
        decoded = allocator.alloc_decode(after.to("gaudirt:0"), after, last)
        assert decoded.cpu().tolist() == [3 * page_size + 3, 5 * page_size + 1]
        assert allocator.available_size() == 59 * page_size
        allocator.free_group_begin()
        allocator.free(initial)
        allocator.free(extended)
        allocator.free(decoded)
        allocator.free_group_end()
        assert allocator.available_size() == 64 * page_size
        assert allocator.alloc(page_size).cpu().tolist() == list(range(page_size, 2 * page_size))
        print("SGLang paged allocator", page_size, "partial/extend/decode/grouped free passed", flush=True)
        del allocator, initial, extended, decoded, last
    gaudirt.empty_cache()
    gaudirt.shutdown()


if __name__ == "__main__":
    main()
