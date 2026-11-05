"""Compile connected SGLang norm and activation dispatch with dynamic tokens."""

import hashlib
import os
import sys
from types import SimpleNamespace

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import torch
import torch.nn.functional as F
from fp64_reference import direct_bf16
import gaudirt
import gaudirt.compiler as compiler
from sglang.srt.platforms import current_platform

torch.set_num_threads(4)
torch.random.default_generator.manual_seed(3621)
EPS = torch.tensor(1e-6, dtype=torch.float32).item()


def reference(x, residual, gamma, up, down, fp64):
    dtype = torch.float64 if fp64 else torch.float32
    rounded = direct_bf16 if fp64 else lambda value: value.to(torch.bfloat16).float()
    x, residual, gamma, up, down = [value.to(dtype) for value in (x, residual, gamma, up, down)]
    residual = rounded(x + residual)
    x = rounded(residual * torch.rsqrt((residual * residual).mean(-1, keepdim=True) + EPS) * gamma)
    gate, up_result = rounded(x @ up.t()).chunk(2, dim=-1)
    x = rounded(rounded(gate / (1 + torch.exp(-gate))) * up_result)
    return rounded(x @ down.t()).double(), residual.double()


def check(outputs, inputs):
    result64, residual64 = reference(*inputs, True)
    result32, residual32 = reference(*inputs, False)
    actual, residual = [value.cpu().double() for value in outputs]
    assert torch.equal(residual, residual64) and torch.equal(residual32, residual64)
    assert torch.isfinite(actual).all() and torch.isfinite(result64).all() and torch.isfinite(result32).all()
    denom = torch.linalg.vector_norm(result64, dim=-1).clamp_min(1e-30)
    row_errors = torch.linalg.vector_norm(actual - result64, dim=-1) / denom
    error = row_errors.max().item()
    floor = (torch.linalg.vector_norm(result32 - result64, dim=-1) / denom).max().item()
    absolute = (actual - result64).abs().max().item()
    if not (error <= max(3e-3, 2 * floor) and error <= 1e-2):
        print("failing row", row_errors.argmax().item(), "error", error, "floor", floor,
              "max_abs", absolute, flush=True)
        if path := os.environ.get("GAUDIRT_DEBUG_CASE"):
            torch.save((inputs, tuple(value.cpu() for value in outputs)), path)
        raise AssertionError((error, floor, absolute))
    payload = b"".join(value.cpu().view(torch.int16).numpy().tobytes() for value in outputs)
    return payload, error, floor, absolute


def main():
    current_platform.init_backend()
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.runtime_context import get_context

    get_context().set_server_args(SimpleNamespace(rl_on_policy_target=None))

    class Dense(torch.nn.Module):
        def __init__(self, gamma, up, down):
            super().__init__()
            self.norm = RMSNorm(64, eps=1e-6, weight_dtype=torch.bfloat16)
            self.norm.weight = torch.nn.Parameter(gamma.to("gaudirt:0"), requires_grad=False)
            self.act = SiluAndMul()
            self.up = torch.nn.Parameter(up.to("gaudirt:0"), requires_grad=False)
            self.down = torch.nn.Parameter(down.to("gaudirt:0"), requires_grad=False)

        def forward(self, x, residual):
            normalized, updated = self.norm(x, residual)
            packed = F.linear(normalized, self.up)
            return F.linear(self.act(packed), self.down), updated

    parameters = [((torch.randn(64) * 0.1 + 1).to(torch.bfloat16),
                   (torch.randn(256, 64) * 0.1).to(torch.bfloat16),
                   (torch.randn(64, 128) * 0.1).to(torch.bfloat16)) for _ in range(2)]
    modules = [Dense(*args) for args in parameters]
    functions = [compiler.compile(module.forward, dynamic_arg_indices=(0, 1)) for module in modules]
    payload, errors, floors, absolutes = bytearray(), [], [], []
    for rows in (1, 3, 17, 137, 513, 2048, 4096):
        x_cpu = torch.randn(rows, 64, dtype=torch.bfloat16)
        residual_cpu = torch.randn(rows, 64, dtype=torch.bfloat16)
        x, residual = x_cpu.to("gaudirt:0"), residual_cpu.to("gaudirt:0")
        first = functions[0](x, residual)
        first_check = check(first, (x_cpu, residual_cpu, *parameters[0]))
        before = gaudirt.recipe_cache_stats()
        second = functions[1](*first)
        assert gaudirt.recipe_cache_stats() == before, "isomorphic SGLang layer recompiled"
        second_check = check(second, (first[0].cpu(), first[1].cpu(), *parameters[1]))
        for value in (first_check, second_check):
            payload.extend(value[0]); errors.append(value[1]); floors.append(value[2]); absolutes.append(value[3])
        print("SGLang FX rows", rows, "recipe stats", before, flush=True)
    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert stats[0] == 0 and stats[1] > 0, stats
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats[0] > 0 and stats[1] == 0, stats
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as result:
            result.write(payload)
    print("SGLang FX stats:", stats, "max_row_relative_l2:", max(errors),
          "max_cpu_fp32_floor:", max(floors), "max_abs:", max(absolutes),
          "output SHA256:", hashlib.sha256(payload).hexdigest(), flush=True)
    print("SGLang operator dispatch in two connected FX graphs passed", flush=True)


if __name__ == "__main__":
    main()
