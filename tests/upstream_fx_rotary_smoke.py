"""Capture actual SGLang RoPE dispatch with QKV projection in one FX graph."""

import hashlib
import os
import sys
from types import SimpleNamespace

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import torch
import torch.nn.functional as F
import gaudirt
import gaudirt.compiler as compiler
import bucket_fx_rotary_smoke as reference
from sglang.srt.platforms import current_platform


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(45913)
    current_platform.init_backend()
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    from sglang.srt.runtime_context import get_context

    get_context().set_server_args(SimpleNamespace(rl_on_policy_target=None))
    rope = RotaryEmbedding(64, 64, 128, 10000, True, torch.bfloat16)
    assert rope.cos_sin_cache.device.type == "gaudirt"
    cache_cpu = rope.cos_sin_cache.cpu()
    weight_cpu = (torch.randn(256, 128) * 0.05).to(torch.bfloat16)
    weight = weight_cpu.to("gaudirt:0")

    def graph(x, weight, positions):
        q, k, v = F.linear(x, weight).split([128, 64, 64], dim=-1)
        q, k = rope(positions, q, k)
        return q, k, v

    captures = []
    original_backend = compiler.gaudirt_backend

    def count_backend(gm, example_inputs):
        captures.append(gm)
        return original_backend(gm, example_inputs)

    compiler.gaudirt_backend = count_backend
    compiled = compiler.compile(graph, dynamic_arg_indices=(0, 2))
    payload, errors, floors, absolutes = bytearray(), [], [], []
    backend = SimpleNamespace()
    for rows in (1, 3, 17, 137, 513):
        metadata, positions_cpu = reference.metadata(rows)
        positions = positions_cpu.to("gaudirt:0")
        rotary_positions = metadata.rotary_positions()
        torch._dynamo.mark_dynamic(rotary_positions, 0, min=1, max=4096)
        backend.metadata, backend.positions = metadata, positions
        backend.rotary_positions = rotary_positions
        backend.rotary_header, backend.rotary_context = metadata.rotary_header(), metadata.context()
        x_cpu = torch.randn(rows, 128, dtype=torch.bfloat16)
        with forward_context(ForwardContext(backend)):
            result = compiled(x_cpu.to("gaudirt:0"), weight, positions)
        data, error, floor, absolute = reference.check(result, x_cpu, weight_cpu, cache_cpu, positions_cpu)
        payload.extend(data); errors.append(error); floors.append(floor); absolutes.append(absolute)
        print("SGLang FX RoPE rows", rows, "captures", len(captures),
              "recipe stats", gaudirt.recipe_cache_stats(), flush=True)
    assert len(captures) == 1, f"SGLang RoPE captured {len(captures)} times"
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[:2] == [0, 5], counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[:2] == [5, 0], counts
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("SGLang compiled rotary stats", counts, "max row error", max(errors),
          "FP32 floor", max(floors), "max abs", max(absolutes),
          "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
