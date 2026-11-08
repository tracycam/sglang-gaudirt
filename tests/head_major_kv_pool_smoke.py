"""Exercise the SGLang pool's logical views and shared descriptor backing."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
from sglang_gaudirt.kv_pool import GaudiRTMHATokenToKVPool


def main():
    gaudirt.initialize(hbm_bytes=192 * 1024**2, pinned_bytes=4 * 1024**2)
    pool = GaudiRTMHATokenToKVPool(64, 1, torch.bfloat16, 2, 64, 3, "gaudirt:0", False)
    assert pool.kv_cache_layout == "gaudirt_head_major"
    keys, values = pool.get_kv_buffer(1)
    assert tuple(keys.shape) == tuple(values.shape) == (65, 2, 64)
    assert not keys.is_contiguous()
    assert keys.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()
    assert values.data_ptr() - keys.data_ptr() == 65 * 128 * 2
    x = torch.arange(keys.numel(), dtype=torch.float32).reshape(keys.shape).to(torch.bfloat16)
    keys.copy_(x)
    assert torch.equal(keys.cpu(), x)
    assert torch.count_nonzero(values.cpu()).item() == 0
    assert torch.count_nonzero(pool.get_key_buffer(0).cpu()).item() == 0
    assert torch.count_nonzero(pool.get_key_buffer(2).cpu()).item() == 0
    assert sum(pool.get_kv_size_bytes()) == pool._combined_kv.untyped_storage().nbytes()
    del keys, values, pool
    gaudirt.synchronize()
    print("SGLang head-major KV pool views passed", flush=True)


if __name__ == "__main__":
    main()
