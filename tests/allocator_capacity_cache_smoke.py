"""Pool resets at different capacities do not compile range recipes."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
from sglang_gaudirt.allocator import GaudiRTTokenToKVPoolAllocator


def main():
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    for page_size in (1, 16, 64):
        for size in (2048, 8192, 32768):
            before = gaudirt.recipe_cache_stats()[1]
            pool = GaudiRTTokenToKVPoolAllocator(size, page_size, torch.bfloat16, "gaudirt", None, False)
            assert gaudirt.recipe_cache_stats()[1] == before, (page_size, size, "initialization")
            expected = torch.arange(1, size // page_size + 1,
                                    dtype=torch.int32 if page_size == 1 else torch.int64)
            assert torch.equal(pool.free_pages.cpu(), expected), (page_size, size)
            pool.clear()
            assert gaudirt.recipe_cache_stats()[1] == before, (page_size, size, "clear")
            assert torch.equal(pool.free_pages.cpu(), expected), (page_size, size, "clear")
    print("KV metadata: page sizes 1/16/64, capacities 2048/8192/32768; no recipe compilation", flush=True)


if __name__ == "__main__":
    main()
