"""SGLang OOT dispatch, real pools, varlen prefill, and decode on one device."""

import hashlib
import math
import os
import sys
from types import SimpleNamespace

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import torch
from fp64_reference import direct_bf16
import paged_attention_smoke as attention_ref
import rotary_runtime_smoke as rotary_ref
import gaudirt
from sglang.srt.platforms import current_platform

torch.set_num_threads(4)
torch.random.default_generator.manual_seed(20261104)
GEOMETRY = os.environ.get("GAUDIRT_TEST_GEOMETRY", "small")
assert GEOMETRY in ("small", "nanbeige")
HEADS, KV_HEADS, DIM = (4, 2, 64) if GEOMETRY == "small" else (56, 8, 128)
Q_WIDTH, KV_WIDTH = HEADS * DIM, KV_HEADS * DIM
CONTEXT, REQUESTS = 512, 4
SLOTS = REQUESTS * CONTEXT + 1
SCALE = DIM ** -0.5
attention_ref.HEADS, attention_ref.KV_HEADS, attention_ref.DIM = HEADS, KV_HEADS, DIM
attention_ref.Q_WIDTH, attention_ref.KV_WIDTH = Q_WIDTH, KV_WIDTH
attention_ref.SCALE_FP32 = torch.tensor(SCALE, dtype=torch.float32).item()


def check(actual, ref64, ref32):
    actual = actual.cpu().double()
    assert torch.isfinite(actual).all() and torch.isfinite(ref64).all() and torch.isfinite(ref32).all()
    denom = torch.linalg.vector_norm(ref64, dim=-1).clamp_min(1e-30)
    error = (torch.linalg.vector_norm(actual - ref64, dim=-1) / denom).max().item()
    floor = (torch.linalg.vector_norm(ref32 - ref64, dim=-1) / denom).max().item()
    absolute = (actual - ref64).abs().max().item()
    assert error <= max(3e-3, 2 * floor) and error <= 1e-2, (error, floor, absolute)
    return error, floor, absolute


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()


def main():
    assert current_platform.device_type == "gaudirt"
    current_platform.init_backend()
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    from sglang.srt.runtime_context import get_context
    from sglang_gaudirt.attention import GaudiRTPagedAttnBackend

    get_context().set_server_args(SimpleNamespace(rl_on_policy_target=None))
    req_pool = ReqToTokenPool(REQUESTS, CONTEXT, "gaudirt:0", False)
    pool = MHATokenToKVPool(REQUESTS * CONTEXT, 1, torch.bfloat16, KV_HEADS, DIM,
                            2, "gaudirt:0", False)
    mapping = torch.zeros((REQUESTS + 1, CONTEXT), dtype=torch.int32)
    mapping[1:] = torch.arange(1, SLOTS, dtype=torch.int32).reshape(REQUESTS, CONTEXT)
    req_pool.req_to_token.copy_(mapping)
    runner = SimpleNamespace(req_to_token_pool=req_pool, token_to_kv_pool=pool,
                             model_config=SimpleNamespace(vocab_size=32000))
    backend = GaudiRTPagedAttnBackend(runner)
    layers = [RadixAttention(HEADS, DIM, SCALE, KV_HEADS, layer_id=i) for i in range(2)]
    rope = RotaryEmbedding(DIM, DIM, CONTEXT, 10000, True, torch.bfloat16)
    assert rope.cos_sin_cache.device.type == "gaudirt"
    initial_cache = rope.cos_sin_cache.cpu().clone()
    print("initial RoPE table hash", tensor_hash(initial_cache), flush=True)
    norm = RMSNorm(DIM, eps=1e-6, weight_dtype=torch.bfloat16)
    norm.weight = torch.nn.Parameter(torch.ones(DIM, dtype=torch.bfloat16, device="gaudirt:0"),
                                     requires_grad=False)
    silu = SiluAndMul()
    assert norm._forward_method.__func__.__module__ == "sglang_gaudirt.ops"
    assert rope._forward_method.__func__.__module__ == "sglang_gaudirt.ops"
    assert silu._forward_method.__func__.__module__ == "sglang_gaudirt.ops"
    payload, errors, floors, absolutes = bytearray(), [], [], []

    packed_cpu = torch.randn((7, 2 * DIM), dtype=torch.bfloat16)
    gate, up = packed_cpu.double().chunk(2, dim=-1)
    act_truth = direct_bf16(direct_bf16(gate / (1 + torch.exp(-gate))) * up)
    act_floor = ((gate.float() / (1 + torch.exp(-gate.float()))).to(torch.bfloat16).float()
                 * up.float()).to(torch.bfloat16).double()
    stats = check(silu(packed_cpu.to("gaudirt:0")), act_truth, act_floor)
    errors.append(stats[0]); floors.append(stats[1]); absolutes.append(stats[2])
    x_cpu = torch.randn((7, DIM), dtype=torch.bfloat16)
    residual_cpu = torch.randn((7, DIM), dtype=torch.bfloat16)
    normalized, residual = norm(x_cpu.to("gaudirt:0"), residual_cpu.to("gaudirt:0"))
    boundary = direct_bf16(x_cpu.double() + residual_cpu.double())
    eps = torch.tensor(1e-6, dtype=torch.float32).item()
    truth = direct_bf16(boundary * torch.rsqrt((boundary * boundary).mean(-1, keepdim=True) + eps))
    boundary32 = (x_cpu.float() + residual_cpu.float()).to(torch.bfloat16).float()
    floor = (boundary32 * torch.rsqrt((boundary32 * boundary32).mean(-1, keepdim=True) + eps)).to(torch.bfloat16).double()
    assert torch.equal(residual.cpu().double(), boundary)
    stats = check(normalized, truth, floor)
    errors.append(stats[0]); floors.append(stats[1]); absolutes.append(stats[2])

    for total in (19, 137, 513):
        base_kv = [(torch.randn(SLOTS, KV_HEADS, DIM, dtype=torch.bfloat16),
                    torch.randn(SLOTS, KV_HEADS, DIM, dtype=torch.bfloat16)) for _ in range(2)]
        for layer_id, (keys, values) in enumerate(base_kv):
            pool.get_key_buffer(layer_id).copy_(keys)
            pool.get_value_buffer(layer_id).copy_(values)
        lengths = [total // 2, total - total // 2]
        prefixes = [2, 3]
        for step in range(4):
            row_requests = torch.repeat_interleave(torch.tensor([1, 3]), torch.tensor(lengths))
            positions_cpu = torch.cat([torch.arange(start, start + count) for start, count in zip(prefixes, lengths)])
            locations = mapping[row_requests, positions_cpu].long()
            seq_lens_cpu = torch.tensor([start + count for start, count in zip(prefixes, lengths)])
            positions = positions_cpu.to("gaudirt:0")
            batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND if step == 0 else ForwardMode.DECODE,
                batch_size=2, input_ids=torch.zeros(len(positions_cpu), dtype=torch.int64, device="gaudirt:0"),
                req_pool_indices=torch.tensor([1, 3], dtype=torch.int64, device="gaudirt:0"),
                seq_lens=seq_lens_cpu.to("gaudirt:0"), out_cache_loc=locations.to("gaudirt:0"),
                seq_lens_sum=int(seq_lens_cpu.sum()), seq_lens_cpu=seq_lens_cpu,
                positions=positions, extend_prefix_lens_cpu=prefixes, extend_seq_lens_cpu=lengths,
            )
            backend.init_forward_metadata(batch)
            assert batch.input_ids.shape == (len(positions_cpu),)
            assert getattr(batch.input_ids, "_gaudirt_validated_vocab") == 32000
            expected_last = torch.cumsum(torch.tensor(lengths), dim=0) - 1
            assert torch.equal(backend.last_token_indices.cpu().long(), expected_last)
            q_cpu = torch.randn((len(positions_cpu), Q_WIDTH), dtype=torch.bfloat16)
            k_cpu = torch.randn((len(positions_cpu), KV_WIDTH), dtype=torch.bfloat16)
            v_cpu = torch.randn((len(positions_cpu), KV_WIDTH), dtype=torch.bfloat16)
            with forward_context(ForwardContext(backend)):
                assert current_platform.get_last_token_indices() is backend.last_token_indices
                q_rot, k_rot = rope(positions, q_cpu.to("gaudirt:0"), k_cpu.to("gaudirt:0"))
                cache_cpu = rope.cos_sin_cache.cpu()
                assert torch.equal(cache_cpu.view(torch.int16), initial_cache.view(torch.int16)), \
                    "RoPE table changed while moving to the device"
                for actual, source in ((q_rot, q_cpu), (k_rot, k_cpu)):
                    r64 = rotary_ref.reference(source, cache_cpu, positions_cpu, DIM, DIM, True, True)
                    r32 = rotary_ref.reference(source, cache_cpu, positions_cpu, DIM, DIM, True, False)
                    stats = check(actual, r64, r32)
                    errors.append(stats[0]); floors.append(stats[1]); absolutes.append(stats[2])
                q_rot_cpu, k_rot_cpu = q_rot.cpu(), k_rot.cpu()
                print("input and RoPE hashes", total, step,
                      *(tensor_hash(value) for value in (q_cpu, k_cpu, v_cpu, cache_cpu, q_rot_cpu, k_rot_cpu)),
                      flush=True)
                value = v_cpu.to("gaudirt:0")
                for layer_id, layer in enumerate(layers):
                    expected_k, expected_v = base_kv[layer_id]
                    expected_k[locations] = k_rot_cpu.reshape(-1, KV_HEADS, DIM)
                    expected_v[locations] = v_cpu.reshape(-1, KV_HEADS, DIM)
                    before = gaudirt.recipe_cache_stats()
                    output = layer(q_rot, k_rot, value, batch)
                    if layer_id == 1:
                        assert gaudirt.recipe_cache_stats() == before, "same-geometry layer recompiled"
                    actual_k, actual_v = pool.get_key_buffer(layer_id).cpu(), pool.get_value_buffer(layer_id).cpu()
                    assert torch.equal(actual_k.view(torch.int16), expected_k.view(torch.int16))
                    assert torch.equal(actual_v.view(torch.int16), expected_v.view(torch.int16))
                    r64 = attention_ref.expected(q_rot_cpu, expected_k, expected_v, row_requests, positions_cpu, mapping, True)
                    r32 = attention_ref.expected(q_rot_cpu, expected_k, expected_v, row_requests, positions_cpu, mapping, False)
                    stats = check(output, r64, r32)
                    errors.append(stats[0]); floors.append(stats[1]); absolutes.append(stats[2])
                    payload.extend(output.cpu().view(torch.int16).numpy().tobytes())
                    payload.extend(actual_k.view(torch.int16).numpy().tobytes())
                    payload.extend(actual_v.view(torch.int16).numpy().tobytes())
            print("prefill", total, "step", step, "rows", len(positions_cpu),
                  "recipe stats", gaudirt.recipe_cache_stats(), flush=True)
            prefixes = seq_lens_cpu.tolist()
            lengths = [1, 1]
    before_invalid = [(pool.get_key_buffer(i).cpu(), pool.get_value_buffer(i).cpu()) for i in range(2)]
    batch.input_ids = torch.full((batch.input_ids.numel(),), -1, dtype=torch.int64).to("gaudirt:0")
    try:
        backend.init_forward_metadata(batch)
    except RuntimeError as error:
        assert "token ID outside vocabulary" in str(error), str(error)
    else:
        raise AssertionError("out-of-vocabulary input was accepted")
    assert backend.metadata is None and backend.forward_batch is None
    for layer, (before_k, before_v) in enumerate(before_invalid):
        assert torch.equal(pool.get_key_buffer(layer).cpu(), before_k)
        assert torch.equal(pool.get_value_buffer(layer).cpu(), before_v)
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[0] == 0 and counts[1] > 0, counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[0] > 0 and counts[1] == 0, counts
    digest = hashlib.sha256(payload).hexdigest()
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("SGLang pool and dispatch stats:", counts, "max_row_relative_l2:", max(errors),
          "max_cpu_fp32_floor:", max(floors), "max_abs:", max(absolutes), "output and KV sha256:", digest,
          flush=True)
    print("SGLang real pools, rotary, normalization, activation, and attention passed", flush=True)


if __name__ == "__main__":
    main()
