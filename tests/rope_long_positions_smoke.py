"""Check long-position RoPE inputs and outputs across fresh processes."""

import hashlib
import os
from types import SimpleNamespace

import torch
import upstream_attention_smoke as support
import rotary_runtime_smoke as rotary_ref
import gaudirt

from sglang.srt.platforms import current_platform


def main():
    assert support.GEOMETRY == "nanbeige"
    current_platform.init_backend()
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    from sglang.srt.runtime_context import get_context
    from sglang_gaudirt.attention import GaudiRTPagedAttnBackend

    get_context().set_server_args(SimpleNamespace(rl_on_policy_target=None))
    req_pool = ReqToTokenPool(support.REQUESTS, support.CONTEXT, "gaudirt:0", False)
    pool = MHATokenToKVPool(support.REQUESTS * support.CONTEXT, 1, torch.bfloat16,
                            support.KV_HEADS, support.DIM, 1, "gaudirt:0", False)
    mapping = torch.zeros((support.REQUESTS + 1, support.CONTEXT), dtype=torch.int32)
    mapping[1:] = torch.arange(1, support.SLOTS, dtype=torch.int32).reshape(
        support.REQUESTS, support.CONTEXT
    )
    req_pool.req_to_token.copy_(mapping)
    lengths, prefixes = [256, 257], [2, 3]
    rows = sum(lengths)
    requests = torch.repeat_interleave(torch.tensor([1, 3]), torch.tensor(lengths))
    positions_cpu = torch.cat([torch.arange(start, start + count)
                               for start, count in zip(prefixes, lengths)])
    locations = mapping[requests, positions_cpu].long()
    seq_lens_cpu = torch.tensor([a + b for a, b in zip(prefixes, lengths)])
    positions = positions_cpu.to("gaudirt:0")
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND, batch_size=2,
        input_ids=torch.zeros(rows, dtype=torch.int64, device="gaudirt:0"),
        req_pool_indices=torch.tensor([1, 3], dtype=torch.int64, device="gaudirt:0"),
        seq_lens=seq_lens_cpu.to("gaudirt:0"), out_cache_loc=locations.to("gaudirt:0"),
        seq_lens_sum=int(seq_lens_cpu.sum()), seq_lens_cpu=seq_lens_cpu,
        positions=positions, extend_prefix_lens_cpu=prefixes, extend_seq_lens_cpu=lengths,
    )
    backend = GaudiRTPagedAttnBackend(SimpleNamespace(req_to_token_pool=req_pool, token_to_kv_pool=pool))
    backend.init_forward_metadata(batch)
    rope = RotaryEmbedding(support.DIM, support.DIM, support.CONTEXT, 10000, True, torch.bfloat16)
    assert rope.cos_sin_cache.device.type == "gaudirt"
    initial_cache = rope.cos_sin_cache.cpu().clone()
    q_cpu = torch.randn(rows, support.Q_WIDTH, dtype=torch.bfloat16)
    k_cpu = torch.randn(rows, support.KV_WIDTH, dtype=torch.bfloat16)
    with forward_context(ForwardContext(backend)):
        query, key = rope(positions, q_cpu.to("gaudirt:0"), k_cpu.to("gaudirt:0"))
    cache_cpu = rope.cos_sin_cache.cpu()
    assert torch.equal(cache_cpu.view(torch.int16), initial_cache.view(torch.int16))
    output = []
    for device, source in ((query, q_cpu), (key, k_cpu)):
        actual = device.cpu()
        r64 = rotary_ref.reference(source, initial_cache, positions_cpu, support.DIM, support.DIM, True, True)
        r32 = rotary_ref.reference(source, initial_cache, positions_cpu, support.DIM, support.DIM, True, False)
        error, floor, absolute = support.check(device, r64, r32)
        print("RoPE error", error, "FP32 floor", floor, "max_abs", absolute, flush=True)
        output.append(actual)
    tensors = (initial_cache, q_cpu, k_cpu, *output)
    payload = b"".join(value.contiguous().view(torch.int16).numpy().tobytes() for value in tensors)
    print("RoPE table, inputs, and outputs SHA256", hashlib.sha256(payload).hexdigest(), flush=True)
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as dest:
            dest.write(payload)
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[0] == 0 and counts[1] > 0, counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[0] > 0 and counts[1] == 0, counts
    print("long-position RoPE device audit passed", counts, flush=True)


if __name__ == "__main__":
    main()
