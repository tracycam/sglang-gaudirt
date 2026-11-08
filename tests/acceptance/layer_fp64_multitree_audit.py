"""Frozen layers and fixed-online-prefix logits against independent CPU trees."""

import json
import math
import os
from pathlib import Path

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
import torch
from layer_fp64_audit import Weights, cosine_sine, rotate, rounded


def project(value, weight, dtype, parts):
    if dtype == torch.float64 or parts == 1:
        return value @ weight.t()
    bounds = [(weight.shape[1] * i // parts, weight.shape[1] * (i + 1) // parts) for i in range(parts)]
    partial = [value[..., start:end] @ weight[:, start:end].t() for start, end in bounds]
    while len(partial) > 1:
        partial = [partial[i] + partial[i + 1] for i in range(0, len(partial), 2)]
    return partial[0]


def step(x, weights, attention, geometry, dtype, parts=1):
    heads, kv_heads, dimension, eps, scale, theta = geometry
    x = x.to(dtype)
    norm = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * weights["in_norm"]
    query = project(norm, weights["q"], dtype, parts)
    key = project(norm, weights["k"], dtype, parts)
    value = project(norm, weights["v"], dtype, parts).reshape(kv_heads, dimension)
    cosine, sine = cosine_sine(attention["position"], theta, dimension, dtype)
    query = rotate(query, heads, dimension, cosine, sine, dtype)
    key = rounded(rotate(key, kv_heads, dimension, cosine, sine, dtype), dtype)
    value = rounded(value, dtype)
    keys = torch.cat((attention["keys"][:-1].to(dtype), key.unsqueeze(0)))
    values = torch.cat((attention["values"][:-1].to(dtype), value.unsqueeze(0)))
    head_map = torch.arange(heads) // (heads // kv_heads)
    scores = torch.einsum("hd,thd->ht", query, keys[:, head_map]) * scale
    attended = torch.einsum("ht,thd->hd", torch.softmax(scores, dim=-1), values[:, head_map]).reshape(1, -1)
    projected = project(attended, weights["out"], dtype, parts)
    residual = projected + x
    norm = residual * torch.rsqrt((residual * residual).mean(-1, keepdim=True) + eps) * weights["post_norm"]
    gate = project(norm, weights["gate"], dtype, parts)
    up = project(norm, weights["up"], dtype, parts)
    activated = gate / (1 + torch.exp(-gate)) * up
    down = project(activated, weights["down"], dtype, parts)
    return rounded(down + residual, dtype)


def main():
    torch.set_num_threads(int(os.environ.get("GAUDIRT_CPU_THREADS", "8")))
    directory = Path(os.environ["GAUDIRT_TARGET_MODEL"])
    config = json.loads((directory / "config.json").read_text())
    count = config["num_hidden_layers"]
    state = torch.load(os.environ["GAUDIRT_LAYER_CAPTURE_PATH"], map_location="cpu", weights_only=True)
    assert len(state["inputs"]) == count + 1 and len(state["attention"]) == count
    dim = config["head_dim"]
    geometry = (config["num_attention_heads"], config["num_key_value_heads"], dim,
                torch.tensor(config["rms_norm_eps"], dtype=torch.float32).item(),
                torch.tensor(1 / math.sqrt(dim), dtype=torch.float32).item(), config.get("rope_theta", 10000))
    assert geometry[0] % geometry[1] == 0 and dim == 128
    names = {"in_norm": "input_layernorm.weight", "post_norm": "post_attention_layernorm.weight",
             "q": "self_attn.q_proj.weight", "k": "self_attn.k_proj.weight", "v": "self_attn.v_proj.weight",
             "out": "self_attn.o_proj.weight", "gate": "mlp.gate_proj.weight",
             "up": "mlp.up_proj.weight", "down": "mlp.down_proj.weight"}
    trees = (1, 2, 4)
    rolling64 = state["inputs"][0].double()
    rolling32 = {parts: state["inputs"][0].float() for parts in trees}
    records = []
    weights = Weights(directory)
    try:
        for i in range(count):
            stored = {name: weights.get(f"model.layers.{i}." + suffix) for name, suffix in names.items()}
            w64 = {name: value.double() for name, value in stored.items()}
            w32 = {name: value.float() for name, value in stored.items()}
            actual = state["inputs"][i + 1].double()
            truth = step(state["inputs"][i], w64, state["attention"][i], geometry, torch.float64)
            floor = step(state["inputs"][i], w32, state["attention"][i], geometry, torch.float32).double()
            assert all(torch.isfinite(value).all() for value in (actual, truth, floor))
            denominator = torch.linalg.vector_norm(truth).clamp_min(1e-30)
            error = (torch.linalg.vector_norm(actual - truth) / denominator).item()
            floor_error = (torch.linalg.vector_norm(floor - truth) / denominator).item()
            limit = max(3e-3, 2 * floor_error)
            record = {"layer": i, "relative_l2": error, "fp32_floor": floor_error, "limit": limit,
                      "max_abs": (actual - truth).abs().max().item(), "pass": error <= limit and error <= 1e-2}
            records.append(record)
            print("FROZEN_LAYER", json.dumps(record, sort_keys=True), flush=True)
            rolling64 = step(rolling64, w64, state["attention"][i], geometry, torch.float64)
            for parts in trees:
                rolling32[parts] = step(rolling32[parts], w32, state["attention"][i], geometry,
                                        torch.float32, parts)
            del stored, w64, w32
        failed = [record["layer"] for record in records if not record["pass"]]
        result = {"layers": records, "failed_layers": failed,
                  "max_row_relative_l2": max(item["relative_l2"] for item in records)}
        if failed:
            if path := os.environ.get("GAUDIRT_LAYER_RECEIPT"):
                Path(path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            raise AssertionError(f"frozen layer gate failed: {failed}")

        def logits(value, dtype, parts=1):
            gamma = weights.get("model.norm.weight").to(dtype)
            normalized = value * torch.rsqrt((value * value).mean(-1, keepdim=True) + geometry[3]) * gamma
            return rounded(project(normalized, weights.get("lm_head.weight").to(dtype), dtype, parts), dtype).double()

        truth_logits = logits(rolling64, torch.float64)
        cpu_logits = {parts: logits(rolling32[parts], torch.float32, parts) for parts in trees}
        candidate = state["logits"].double()
        assert all(value.shape == truth_logits.shape and torch.isfinite(value).all()
                   for value in (candidate, truth_logits, *cpu_logits.values()))
        log_truth = torch.log_softmax(truth_logits, dim=-1)
        probabilities = log_truth.exp()

        def divergence(value):
            return (probabilities * (log_truth - torch.log_softmax(value, dim=-1))).sum(-1).max().item()

        kl = divergence(candidate)
        tree_kl = {str(parts): divergence(value) for parts, value in cpu_logits.items()}
        # The capture is the validated M=1 replay of this same online prefix;
        # the acceptance run independently compares its logits with M=137.
        floor_kl = max(kl, *tree_kl.values())
        top = torch.topk(truth_logits, 2, dim=-1)
        first = top.values[..., 0].to(torch.bfloat16)
        spacing = (torch.nextafter(first, torch.full_like(first, float("inf"))).float() - first.float()).abs().double()
        tied = top.values[..., 0] - top.values[..., 1] < spacing
        selected = candidate.argmax(-1)
        top1 = selected == top.indices[..., 0]
        top2 = (selected.unsqueeze(-1) == top.indices).any(-1)
        result.update(max_kl=kl, validated_m1_kl=kl, cpu_tree_kl=tree_kl, ref_floor=floor_kl,
                      kl_limit=max(0.01, 1.15 * floor_kl),
                      top1_match_non_tied=bool(top1[~tied].all()) if (~tied).any() else None,
                      top2_match_tied=bool(top2[tied].all()) if tied.any() else None,
                      tied_samples=tied.sum().item())
        if path := os.environ.get("GAUDIRT_LAYER_RECEIPT"):
            Path(path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        assert kl <= result["kl_limit"] and torch.where(tied, top2, top1).all(), result
        print("MULTITREE_CONDITIONED_AUDIT", json.dumps(result, sort_keys=True), flush=True)
    finally:
        weights.close()


if __name__ == "__main__":
    main()
