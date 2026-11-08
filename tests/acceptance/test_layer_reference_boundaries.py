"""Keep full-layer FP32 arithmetic free of internal BF16 approximations."""

import math
import os
import sys
from pathlib import Path
from unittest.mock import patch

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "GaudiRT" / "tests"))

import torch
import layer_fp64_audit as reference
import layer_fp64_multitree_audit as multitree


def main():
    generator = torch.Generator().manual_seed(440)
    heads, kv_heads, dim = 2, 1, 4
    width, intermediate = heads * dim, 12

    def value(*shape):
        return torch.randn(*shape, generator=generator).to(torch.bfloat16).float()

    weights = {"in_norm": value(width), "post_norm": value(width),
               "q": value(width, width), "k": value(kv_heads * dim, width),
               "v": value(kv_heads * dim, width), "out": value(width, width),
               "gate": value(intermediate, width), "up": value(intermediate, width),
               "down": value(width, intermediate)}
    x = value(1, width)
    attention = {"position": 1, "keys": value(2, kv_heads, dim), "values": value(2, kv_heads, dim)}
    geometry = (heads, kv_heads, dim, 1e-6, 1 / math.sqrt(dim), 10000)
    boundaries = []
    original = reference.rounded

    def track(value, dtype):
        boundaries.append(tuple(value.shape))
        return original(value, dtype)

    with patch.object(reference, "rounded", track):
        actual = reference.layer_step(x, weights, attention, geometry, torch.float32)
    assert boundaries == [(kv_heads, dim), (kv_heads, dim), (1, width)], boundaries
    assert torch.isfinite(actual).all() and torch.equal(actual, actual.to(torch.bfloat16).float())
    tree = multitree.step(x, weights, attention, geometry, torch.float32, 1)
    assert torch.equal(actual, tree)
    print("FP32 layer reference rounds only the external KV and layer outputs", flush=True)


if __name__ == "__main__":
    main()
