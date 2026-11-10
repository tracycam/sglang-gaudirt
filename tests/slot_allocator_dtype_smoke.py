"""Page-one slots write to the INT32 request table without a cast launch."""

import ctypes
import os
import sys
from types import SimpleNamespace

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
from sglang_gaudirt.allocator import GaudiRTTokenToKVPoolAllocator
from sglang_gaudirt.req_to_token import write_req_to_token_pool


def main():
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    counter = ctypes.CDLL(None).gaudirt_test_launch_count
    counter.restype = ctypes.c_uint64
    allocator = GaudiRTTokenToKVPoolAllocator(64, 1, torch.bfloat16, "gaudirt", None, False)
    table = torch.zeros((2, 32), dtype=torch.int32, device="gaudirt")
    pool = SimpleNamespace(req_to_token=table)
    locations = allocator.alloc(1)
    assert locations.dtype == torch.int32
    before = counter()
    write_req_to_token_pool(pool, torch.tensor([1]), torch.tensor([5]), 1, None, locations)
    assert counter() == before, "request-table write must not launch a cast recipe"
    assert table.cpu()[1, 5].item() == 1
    allocator.free(locations)
    all_slots = allocator.alloc(64).cpu()
    assert all_slots.dtype == torch.int32 and torch.equal(all_slots.sort().values, torch.arange(1, 65, dtype=torch.int32))
    allocator.clear()
    assert allocator.free_pages.dtype == allocator.release_pages.dtype == torch.int32
    print("page-one INT32 slots write without a cast launch and free correctly", flush=True)


if __name__ == "__main__":
    main()
