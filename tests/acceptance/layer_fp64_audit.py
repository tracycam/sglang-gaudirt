"""Independently check frozen layers and one rolling online-prefix decode."""

import json
import math
import os
import sys
import atexit
from contextlib import ExitStack
from pathlib import Path

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"


def audit_legacy_bridge():
    assert not any(name == "habana_frameworks" or name.startswith("habana_frameworks.") for name in sys.modules)
    with open("/proc/self/maps", encoding="utf-8") as maps:
        mapped = maps.read().lower()
    assert not any(name in mapped for name in ("libhabana_pytorch", "gaudi-pytorch-bridge", "habana_frameworks"))


audit_legacy_bridge()


def exit_audit():
    try:
        audit_legacy_bridge()
    except AssertionError:
        print("legacy bridge was loaded during the reference process", file=sys.stderr, flush=True)
        os._exit(1)


atexit.register(exit_audit)

import torch
from safetensors import safe_open
from fp64_reference import direct_bf16


class Weights:
    def __init__(self, directory):
        self.directory = directory
        self.stack = ExitStack()
        index = directory / "model.safetensors.index.json"
        if index.exists():
            self.mapping = json.loads(index.read_text())["weight_map"]
            self.files = {name: self.stack.enter_context(
                safe_open(str(directory / name), framework="pt", device="cpu")
            ) for name in set(self.mapping.values())}
        else:
            name = "model.safetensors"
            handle = self.stack.enter_context(safe_open(str(directory / name), framework="pt", device="cpu"))
            self.files = {name: handle}
            self.mapping = {key: name for key in handle.keys()}

    def get(self, name):
        return self.files[self.mapping[name]].get_tensor(name)

    def close(self):
        self.stack.close()


def rounded(value, dtype):
    return direct_bf16(value) if dtype == torch.float64 else value.to(torch.bfloat16).float()


def cosine_sine(position, theta, dimension, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, dimension, 2, dtype=torch.float32) / dimension))
    angles = torch.tensor(position, dtype=torch.float32) * inv
    return tuple(value.to(torch.bfloat16).to(dtype) for value in (angles.cos(), angles.sin()))


def rotate(value, heads, dimension, cosine, sine, dtype):
    value = value.reshape(heads, dimension)
    a, b = value.chunk(2, dim=-1)
    return torch.cat((a * cosine - b * sine, b * cosine + a * sine), dim=-1)


def tiled_bf16_pv(scores, values, head_map, tile, dtype):
    """Independent FP64/FP32 tile reference with BF16 inputs to each P@V."""
    if not isinstance(tile, int) or isinstance(tile, bool) or tile <= 0:
        raise ValueError("BF16-PV reference requires a positive KV tile")
    merged_max = merged_sum = merged_output = None
    for start in range(0, scores.size(-1), tile):
        tile_scores = scores[:, start:start + tile]
        tile_max = tile_scores.amax(dim=-1)
        exponential = (tile_scores - tile_max[:, None]).exp()
        tile_sum = exponential.sum(dim=-1)
        probability = rounded(exponential / tile_sum[:, None], dtype)
        tile_values = values[start:start + tile, head_map]
        tile_output = torch.einsum("ht,thd->hd", probability, tile_values) * tile_sum[:, None]
        if merged_max is None:
            merged_max, merged_sum, merged_output = tile_max, tile_sum, tile_output
            continue
        maximum = torch.maximum(merged_max, tile_max)
        a, b = (merged_max - maximum).exp(), (tile_max - maximum).exp()
        merged_output = a[:, None] * merged_output + b[:, None] * tile_output
        merged_sum = a * merged_sum + b * tile_sum
        merged_max = maximum
    return merged_output / merged_sum[:, None]


def layer_step(x, weights, attention, geometry, dtype, same_boundary=False, project=None,
               bf16_probability=False, bf16_pv_tile=None):
    if bf16_probability and not same_boundary:
        raise ValueError("BF16-PV requires the same-boundary reference")
    if bf16_probability and bf16_pv_tile is None and attention["position"] + 1 > 512:
        raise ValueError("long BF16-PV references require the actual KV tile")
    heads, kv_heads, dimension, eps, scale, theta = geometry
    x = x.to(dtype)
    if project is None:
        project = lambda value, weight: value @ weight.t()

    def stored(value):
        return rounded(value, dtype) if same_boundary else value

    norm = stored(x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * weights["in_norm"])
    query = stored(project(norm, weights["q"]))
    key = stored(project(norm, weights["k"]))
    value = stored(project(norm, weights["v"]).reshape(kv_heads, dimension))
    cosine, sine = cosine_sine(attention["position"], theta, dimension, dtype)
    query = stored(rotate(query, heads, dimension, cosine, sine, dtype))
    # The KV pool is an external BF16 state boundary in both references.
    key = rounded(rotate(key, kv_heads, dimension, cosine, sine, dtype), dtype)
    if not same_boundary:
        value = rounded(value, dtype)
    keys = torch.cat((attention["keys"][:-1].to(dtype), key.unsqueeze(0)))
    values = torch.cat((attention["values"][:-1].to(dtype), value.unsqueeze(0)))
    head_map = torch.arange(heads) // (heads // kv_heads)
    scores = torch.einsum("hd,thd->ht", query, keys[:, head_map]) * scale
    # The main same-boundary reference keeps probabilities in the reference
    # dtype. The separate BF16-PV reference follows the actual KV tiling.
    if bf16_probability:
        if bf16_pv_tile is None or bf16_pv_tile == "full":
            probabilities = rounded(torch.softmax(scores, dim=-1), dtype)
            attended = torch.einsum("ht,thd->hd", probabilities, values[:, head_map])
        else:
            attended = tiled_bf16_pv(scores, values, head_map, bf16_pv_tile, dtype)
    else:
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.einsum("ht,thd->hd", probabilities, values[:, head_map])
    attended = stored(attended.reshape(1, -1))
    projected = stored(project(attended, weights["out"]))
    residual = stored(projected + x)
    norm = stored(residual * torch.rsqrt((residual * residual).mean(-1, keepdim=True) + eps) *
                  weights["post_norm"])
    gate = stored(project(norm, weights["gate"]))
    up = stored(project(norm, weights["up"]))
    activated = stored(gate / (1 + torch.exp(-gate)) * up)
    down = stored(project(activated, weights["down"]))
    return rounded(down + residual, dtype)


def frozen_layer_metrics(layer, actual, pure64, boundary64, boundary32, bf16_pv64):
    values = (actual, pure64, boundary64, boundary32, bf16_pv64)
    assert all(torch.isfinite(value).all() for value in values)

    def relative(value, reference):
        denominator = torch.linalg.vector_norm(reference).clamp_min(1e-30)
        return (torch.linalg.vector_norm(value - reference) / denominator).item()

    error_pure = relative(actual, pure64)
    error_fp64_probability = relative(actual, boundary64)
    error_boundary = relative(actual, bf16_pv64)
    boundary_vs_pure = relative(boundary32, pure64)
    arithmetic_difference = relative(boundary32, boundary64)
    limit = max(3e-3, 2 * boundary_vs_pure)
    result = {"layer": layer, "relative_l2_pure": error_pure,
            "fp32_same_boundary_vs_fp64_pure": boundary_vs_pure,
            "relative_l2_same_boundary": error_boundary,
            "fp32_vs_fp64_operator_boundary": arithmetic_difference,
            "max_abs_pure": (actual - pure64).abs().max().item(),
            "max_abs_same_boundary": (actual - bf16_pv64).abs().max().item(),
            "relative_l2_same_operators_fp64_probability": error_fp64_probability,
            "max_abs_same_operators_fp64_probability": (actual - boundary64).abs().max().item(),
            "limit_pure": limit, "limit_same_boundary": 0.003,
            "limit_same_operators_fp64_probability": limit,
            "pass": (error_pure <= limit and error_pure <= 0.01 and error_boundary <= 0.003 and
                     error_fp64_probability <= limit and error_fp64_probability <= 0.01)}
    return result


def main():
    torch.set_num_threads(int(os.environ.get("GAUDIRT_CPU_THREADS", "8")))
    directory = Path(os.environ["GAUDIRT_TARGET_MODEL"])
    config = json.loads((directory / "config.json").read_text())
    count = config["num_hidden_layers"]
    tile_arg = os.environ.get("GAUDIRT_AUDIT_PV_TILE")
    pv_tile = "full" if tile_arg == "full" else int(tile_arg) if tile_arg is not None else None
    if pv_tile is not None and pv_tile != "full" and pv_tile <= 0:
        raise ValueError("GAUDIRT_AUDIT_PV_TILE must be positive")
    state = torch.load(os.environ["GAUDIRT_LAYER_CAPTURE_PATH"], map_location="cpu", weights_only=True)
    assert len(state["inputs"]) == count + 1 and len(state["attention"]) == count
    dimension = config["head_dim"]
    geometry = (config["num_attention_heads"], config["num_key_value_heads"], dimension,
                torch.tensor(config["rms_norm_eps"], dtype=torch.float32).item(),
                torch.tensor(1 / math.sqrt(dimension), dtype=torch.float32).item(),
                config.get("rope_theta", 10000))
    assert geometry[0] % geometry[1] == 0 and geometry[2] == 128
    names = {"in_norm": "input_layernorm.weight", "post_norm": "post_attention_layernorm.weight",
             "q": "self_attn.q_proj.weight", "k": "self_attn.k_proj.weight",
             "v": "self_attn.v_proj.weight", "out": "self_attn.o_proj.weight",
             "gate": "mlp.gate_proj.weight", "up": "mlp.up_proj.weight", "down": "mlp.down_proj.weight"}
    weights = Weights(directory)
    rolling64 = state["inputs"][0].double()
    rolling32 = state["inputs"][0].float()
    layers = []
    try:
        for i in range(count):
            base = f"model.layers.{i}."
            stored = {name: weights.get(base + suffix) for name, suffix in names.items()}
            weight64 = {name: value.double() for name, value in stored.items()}
            weight32 = {name: value.float() for name, value in stored.items()}
            actual = state["inputs"][i + 1].double()
            truth = layer_step(state["inputs"][i], weight64, state["attention"][i], geometry, torch.float64)
            same64 = layer_step(state["inputs"][i], weight64, state["attention"][i], geometry,
                                torch.float64, same_boundary=True)
            same32 = layer_step(state["inputs"][i], weight32, state["attention"][i], geometry,
                                torch.float32, same_boundary=True).double()
            same_pv64 = layer_step(state["inputs"][i], weight64, state["attention"][i], geometry,
                                   torch.float64, same_boundary=True, bf16_probability=True,
                                   bf16_pv_tile=pv_tile)
            layers.append(frozen_layer_metrics(i, actual, truth, same64, same32, same_pv64))
            assert layers[-1]["pass"], layers[-1]
            rolling64 = layer_step(rolling64, weight64, state["attention"][i], geometry, torch.float64)
            rolling32 = layer_step(rolling32, weight32, state["attention"][i], geometry, torch.float32,
                                   same_boundary=True)
            print("FROZEN_LAYER", json.dumps(layers[-1], sort_keys=True), flush=True)
            del stored, weight64, weight32

        def logits(x, dtype, same_boundary=False):
            gamma = weights.get("model.norm.weight").to(dtype)
            normalized = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + geometry[3]) * gamma
            if same_boundary:
                normalized = rounded(normalized, dtype)
            return rounded(normalized @ weights.get("lm_head.weight").to(dtype).t(), dtype).double()

        truth_logits = logits(rolling64, torch.float64)
        floor_logits = logits(rolling32, torch.float32, same_boundary=True)
        candidate_logits = state["logits"].double()
        assert truth_logits.shape == floor_logits.shape == candidate_logits.shape
        assert all(torch.isfinite(value).all() for value in (truth_logits, floor_logits, candidate_logits))
        log_truth = torch.log_softmax(truth_logits, dim=-1)
        probabilities = log_truth.exp()

        def divergence(value):
            return (probabilities * (log_truth - torch.log_softmax(value, dim=-1))).sum(-1).max().item()

        kl, floor_kl = divergence(candidate_logits), divergence(floor_logits)
        top = torch.topk(truth_logits, 2, dim=-1)
        first = top.values[..., 0].to(torch.bfloat16)
        next_value = torch.nextafter(first, torch.full_like(first, float("inf")))
        spacing = (next_value.float() - first.float()).abs().double()
        tied = top.values[..., 0] - top.values[..., 1] < spacing
        selected = candidate_logits.argmax(-1)
        top1 = selected == top.indices[..., 0]
        top2 = (selected.unsqueeze(-1) == top.indices).any(-1)
        top1_match = bool(top1[~tied].all()) if (~tied).any() else None
        top2_match = bool(top2[tied].all()) if tied.any() else None
        assert kl <= 0.01, (kl, floor_kl)
        assert torch.where(tied, top2, top1).all(), (tied.tolist(), selected.tolist(), top.indices.tolist())
        result = {"layers": layers, "bf16_pv_tile": pv_tile,
                  "max_row_relative_l2": max(item["relative_l2_pure"] for item in layers),
                  "max_kl": kl, "ref_floor": floor_kl, "kl_limit": 0.01,
                  "top1_match_non_tied": top1_match, "top2_match_tied": top2_match,
                  "tied_samples": tied.sum().item()}
        print("LAYER_AND_CONDITIONED_AUDIT", json.dumps(result, sort_keys=True), flush=True)
        if path := os.environ.get("GAUDIRT_LAYER_RECEIPT"):
            Path(path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    finally:
        weights.close()


if __name__ == "__main__":
    main()
