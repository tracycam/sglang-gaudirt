"""Exercise two same-geometry SGLang attention layers with real pools."""

import atexit
import math
import os
import sys
from types import SimpleNamespace

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"


def assert_no_bridge():
    assert "habana_frameworks" not in sys.modules
    with open("/proc/self/maps", encoding="utf-8") as maps:
        assert not any("habana_pytorch" in line or "libhabana_pytorch" in line for line in maps)


assert_no_bridge()
atexit.register(assert_no_bridge)

import torch
import gaudirt
from fp64_reference import direct_bf16
from paged_attention_reference import isolated_errors
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang_gaudirt.attention import GaudiRTPagedAttnBackend
from sglang_gaudirt.kv_pool import GaudiRTMHATokenToKVPool


def main():
    torch.set_num_threads(4)
    heads, kv_heads, dim, rows, prefix, context = 4, 2, 128, 19, 3, 256
    gaudirt.initialize(hbm_bytes=768 * 1024 * 1024, pinned_bytes=8 * 1024 * 1024)
    req_pool = ReqToTokenPool(1, context, "gaudirt:0", False)
    pool = GaudiRTMHATokenToKVPool(512, 1, torch.bfloat16, kv_heads, dim, 2, "gaudirt:0", False)
    mapping = torch.stack((torch.zeros(context, dtype=torch.int32),
                           torch.arange(1, context + 1, dtype=torch.int32)))
    req_pool.req_to_token.copy_(mapping)
    runner = SimpleNamespace(req_to_token_pool=req_pool, token_to_kv_pool=pool,
                             ps=SimpleNamespace(tp_size=1),
                             model_config=SimpleNamespace(vocab_size=32000, num_attention_heads=heads,
                                                          get_num_attention_heads=lambda tp: heads // tp))
    backend = GaudiRTPagedAttnBackend(runner)
    positions = torch.arange(prefix, prefix + rows, dtype=torch.int64)
    locations = mapping[1, positions].long()
    seq_lens = torch.tensor([prefix + rows], dtype=torch.int64)
    batch = ForwardBatch(forward_mode=ForwardMode.EXTEND, batch_size=1,
        input_ids=torch.zeros(rows, dtype=torch.int64, device="gaudirt:0"),
        req_pool_indices=torch.tensor([1], dtype=torch.int64, device="gaudirt:0"),
        seq_lens=seq_lens.to("gaudirt:0"), out_cache_loc=locations.to("gaudirt:0"),
        seq_lens_sum=int(seq_lens.sum()), seq_lens_cpu=seq_lens,
        positions=positions.to("gaudirt:0"), extend_prefix_lens_cpu=[prefix], extend_seq_lens_cpu=[rows])
    backend.init_forward_metadata(batch)
    rng = torch.Generator().manual_seed(20261108)
    query = torch.randn(rows, heads * dim, generator=rng).to(torch.bfloat16)
    key = torch.randn(rows, kv_heads * dim, generator=rng).to(torch.bfloat16)
    value = torch.randn(rows, kv_heads * dim, generator=rng).to(torch.bfloat16)
    q, k, v = (tensor.to("gaudirt:0") for tensor in (query, key, value))
    layers = [RadixAttention(heads, dim, dim ** -0.5, kv_heads, layer_id=i) for i in range(2)]
    results = []
    for index, layer in enumerate(layers):
        before = gaudirt.recipe_cache_stats()
        result = backend.forward_extend(q, k, v, layer, batch)
        if index:
            assert gaudirt.recipe_cache_stats() == before, "same geometry recompiled"
        results.append(result)
    gaudirt.synchronize()
    assert torch.equal(results[0].cpu(), results[1].cpu())
    expected_key = torch.zeros((513, kv_heads, dim), dtype=torch.bfloat16)
    expected_value = torch.zeros_like(expected_key)
    expected_key[locations] = key.reshape(rows, kv_heads, dim)
    expected_value[locations] = value.reshape(rows, kv_heads, dim)
    for index in range(2):
        assert torch.equal(pool.get_key_buffer(index).cpu(), expected_key)
        assert torch.equal(pool.get_value_buffer(index).cpu(), expected_value)
    head_map = torch.arange(heads) // (heads // kv_heads)
    scale = torch.tensor(dim ** -0.5, dtype=torch.float32).item()
    ref64, ref_pv = [], []
    for row in range(rows):
        visible = mapping[1, :positions[row] + 1].long()
        ks, vs = expected_key[visible][:, head_map, :], expected_value[visible][:, head_map, :]
        q_row = query[row].reshape(heads, dim)
        s64 = torch.einsum("hd,thd->ht", q_row.double(), ks.double()) * scale
        ref64.append(torch.einsum("ht,thd->hd", s64.softmax(-1), vs.double()))
        s32 = torch.einsum("hd,thd->ht", q_row.float(), ks.float()) * scale
        p = s32.softmax(-1).to(torch.bfloat16).float()
        ref_pv.append(torch.einsum("ht,thd->hd", p, vs.float()))
    reference64 = direct_bf16(torch.stack(ref64).reshape(rows, -1))
    reference_pv = torch.stack(ref_pv).reshape(rows, -1).to(torch.bfloat16).double()
    errors = isolated_errors(results[0].cpu().double(), reference64, reference_pv)
    counts = gaudirt.recipe_cache_stats()
    if os.environ["GAUDIRT_TEST_CACHE"] == "warm":
        assert counts[0] > 0 and counts[1] == 0, counts
    else:
        assert counts[0] == 0 and counts[1] > 0, counts
    payload = b"".join(t.cpu().contiguous().view(torch.int16).numpy().tobytes() for t in results)
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("SGLang two-layer fused attention passed", errors, "cache", counts, flush=True)


if __name__ == "__main__":
    main()
