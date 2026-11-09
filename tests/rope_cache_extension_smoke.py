"""Check that startup can extend a device-resident RoPE table on the CPU."""

import os
import sys
from types import SimpleNamespace

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import torch
from sglang.srt.platforms import current_platform


def main():
    current_platform.init_backend()
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.runtime_context import get_context

    get_context().set_server_args(SimpleNamespace(rl_on_policy_target=None))
    rope = RotaryEmbedding(128, 128, 64, 10000, True, torch.bfloat16)
    assert rope.cos_sin_cache.untyped_storage().nbytes() >= 131072 * 128 * 2
    before = rope.cos_sin_cache.cpu().clone()
    rope._ensure_cos_sin_cache_length(129)
    assert rope.cos_sin_cache.untyped_storage().nbytes() >= 131072 * 128 * 2
    after = rope.cos_sin_cache.cpu()
    assert after.shape[0] > 129 and torch.equal(after[:64], before)
    with current_platform.rope_cache_initialization_context():
        inv = 1.0 / (10000 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
        angles = torch.outer(torch.arange(64, after.shape[0], dtype=torch.float32), inv)
        expected = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    assert torch.equal(after[64:], expected)
    print("device RoPE cache extended without changing existing rows", after.shape, flush=True)


if __name__ == "__main__":
    main()
