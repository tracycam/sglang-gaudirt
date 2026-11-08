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


def layer_step(x, weights, attention, geometry, dtype):
    heads, kv_heads, dimension, eps, scale, theta = geometry
    x = x.to(dtype)
    norm = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * weights["in_norm"]
    query = norm @ weights["q"].t()
    key = norm @ weights["k"].t()
    value = (norm @ weights["v"].t()).reshape(kv_heads, dimension)
    cosine, sine = cosine_sine(attention["position"], theta, dimension, dtype)
    query = rotate(query, heads, dimension, cosine, sine, dtype)
    # The KV pool is an external BF16 state boundary. All other layer
    # arithmetic, including attention probabilities, stays in the reference dtype.
    key = rounded(rotate(key, kv_heads, dimension, cosine, sine, dtype), dtype)
    value = rounded(value, dtype)
    keys = torch.cat((attention["keys"][:-1].to(dtype), key.unsqueeze(0)))
    values = torch.cat((attention["values"][:-1].to(dtype), value.unsqueeze(0)))
    head_map = torch.arange(heads) // (heads // kv_heads)
    scores = torch.einsum("hd,thd->ht", query, keys[:, head_map]) * scale
    attended = torch.einsum("ht,thd->hd", torch.softmax(scores, dim=-1), values[:, head_map]).reshape(1, -1)
    projected = attended @ weights["out"].t()
    residual = projected + x
    norm = residual * torch.rsqrt((residual * residual).mean(-1, keepdim=True) + eps) * weights["post_norm"]
    gate = norm @ weights["gate"].t()
    up = norm @ weights["up"].t()
    activated = gate / (1 + torch.exp(-gate)) * up
    down = activated @ weights["down"].t()
    return rounded(down + residual, dtype)


def main():
    torch.set_num_threads(int(os.environ.get("GAUDIRT_CPU_THREADS", "8")))
    directory = Path(os.environ["GAUDIRT_TARGET_MODEL"])
    config = json.loads((directory / "config.json").read_text())
    count = config["num_hidden_layers"]
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
            floor = layer_step(state["inputs"][i], weight32, state["attention"][i], geometry, torch.float32).double()
            assert all(torch.isfinite(value).all() for value in (actual, truth, floor))
            denominator = torch.linalg.vector_norm(truth).clamp_min(1e-30)
            error = (torch.linalg.vector_norm(actual - truth) / denominator).item()
            floor_error = (torch.linalg.vector_norm(floor - truth) / denominator).item()
            max_abs = (actual - truth).abs().max().item()
            limit = max(3e-3, 2 * floor_error)
            layers.append({"layer": i, "relative_l2": error, "fp32_floor": floor_error,
                           "max_abs": max_abs, "limit": limit})
            assert error <= limit and error <= 1e-2, layers[-1]
            rolling64 = layer_step(rolling64, weight64, state["attention"][i], geometry, torch.float64)
            rolling32 = layer_step(rolling32, weight32, state["attention"][i], geometry, torch.float32)
            print("FROZEN_LAYER", json.dumps(layers[-1], sort_keys=True), flush=True)
            del stored, weight64, weight32

        def logits(x, dtype):
            gamma = weights.get("model.norm.weight").to(dtype)
            normalized = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + geometry[3]) * gamma
            return rounded(normalized @ weights.get("lm_head.weight").to(dtype).t(), dtype).double()

        truth_logits = logits(rolling64, torch.float64)
        floor_logits = logits(rolling32, torch.float32)
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
        assert kl <= max(0.01, 1.15 * floor_kl), (kl, floor_kl)
        assert torch.where(tied, top2, top1).all(), (tied.tolist(), selected.tolist(), top.indices.tolist())
        result = {"layers": layers, "max_row_relative_l2": max(item["relative_l2"] for item in layers),
                  "max_kl": kl, "ref_floor": floor_kl, "kl_limit": max(0.01, 1.15 * floor_kl),
                  "top1_match_non_tied": top1_match, "top2_match_tied": top2_match,
                  "tied_samples": tied.sum().item()}
        print("LAYER_AND_CONDITIONED_AUDIT", json.dumps(result, sort_keys=True), flush=True)
        if path := os.environ.get("GAUDIRT_LAYER_RECEIPT"):
            Path(path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    finally:
        weights.close()


if __name__ == "__main__":
    main()
