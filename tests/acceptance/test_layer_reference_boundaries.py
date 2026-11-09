"""Check real BF16 tensor boundaries and pure arithmetic inside each operator."""

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

    boundaries.clear()
    with patch.object(reference, "rounded", track):
        same = reference.layer_step(x, weights, attention, geometry, torch.float32, same_boundary=True)
    assert boundaries == [(1, width), (1, width), (1, kv_heads * dim), (kv_heads, dim),
                          (heads, dim), (kv_heads, dim), (1, width), (1, width), (1, width),
                          (1, width), (1, intermediate), (1, intermediate), (1, intermediate),
                          (1, width), (1, width)], boundaries

    def bf(value):
        return value.to(torch.bfloat16).float()

    def rope(value, n, cosine, sine):
        first, second = value.reshape(n, dim).chunk(2, dim=-1)
        return torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1)

    norm = bf(x * torch.rsqrt((x * x).mean(-1, keepdim=True) + geometry[3]) * weights["in_norm"])
    q, k, v = (bf(norm @ weights[name].t()) for name in ("q", "k", "v"))
    cosine, sine = reference.cosine_sine(attention["position"], geometry[-1], dim, torch.float32)
    q, k = bf(rope(q, heads, cosine, sine)), bf(rope(k, kv_heads, cosine, sine))
    v = v.reshape(kv_heads, dim)
    keys = torch.cat((attention["keys"][:-1], k.unsqueeze(0)))
    values = torch.cat((attention["values"][:-1], v.unsqueeze(0)))
    head_map = torch.arange(heads) // (heads // kv_heads)
    scores = torch.einsum("hd,thd->ht", q, keys[:, head_map]) * geometry[4]
    attended = bf(torch.einsum("ht,thd->hd", scores.softmax(-1), values[:, head_map]).reshape(1, -1))
    residual = bf(x + bf(attended @ weights["out"].t()))
    post_norm = bf(residual * torch.rsqrt((residual * residual).mean(-1, keepdim=True) + geometry[3]) *
                   weights["post_norm"])
    gate, up = (bf(post_norm @ weights[name].t()) for name in ("gate", "up"))
    activation = bf(gate / (1 + torch.exp(-gate)) * up)
    expected = bf(residual + bf(activation @ weights["down"].t()))
    assert torch.equal(same, expected)
    assert torch.equal(same, multitree.step(x, weights, attention, geometry, torch.float32,
                                           1, same_boundary=True))
    boundaries.clear()
    with patch.object(reference, "rounded", track):
        reference.layer_step(x, weights, attention, geometry, torch.float32,
                             same_boundary=True, bf16_probability=True)
    assert len(boundaries) == 16 and boundaries[6] == (heads, 2), boundaries
    pure = torch.tensor([1.0, 1.0], dtype=torch.float64)
    bad = torch.tensor([1.01, 1.0], dtype=torch.float64)
    wide_floor = torch.tensor([1.1, 1.0], dtype=torch.float64)
    result = reference.frozen_layer_metrics(0, bad, pure, pure, wide_floor)
    assert result["relative_l2_pure"] < result["limit_pure"] and not result["pass"]
    rejected = multitree.assess_conditioned_kl(0.02, {"1": 0.001, "2": 0.002})
    assert rejected["ref_floor"] == 0.002 and not rejected["kl_pass"]
    assert not multitree.assess_conditioned_kl(0.02, {"1": 0.1})["kl_pass"]
    assert multitree.assess_conditioned_kl(0.009, {"1": 0.001})["kl_pass"]
    try:
        multitree.assess_conditioned_kl(float("nan"), {"1": 0.001})
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite conditioned KL accepted")
    print("BF16 operator boundaries and pure inner attention reference passed", flush=True)


if __name__ == "__main__":
    main()
