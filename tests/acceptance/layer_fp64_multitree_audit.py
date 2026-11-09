"""Frozen layers and fixed-online-prefix logits against independent CPU trees."""

import json
import math
import os
from pathlib import Path

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
import torch
from layer_fp64_audit import Weights, frozen_layer_metrics, layer_step, rounded


def assess_conditioned_kl(candidate, cpu_trees):
    if not cpu_trees or not all(math.isfinite(value) for value in (candidate, *cpu_trees.values())):
        raise ValueError("conditioned KL requires finite candidate and CPU references")
    return {"max_kl": candidate, "candidate_m1_kl": candidate, "cpu_tree_kl": cpu_trees,
            "ref_floor": max(cpu_trees.values()), "kl_limit": 0.01, "kl_pass": candidate <= 0.01}


def project(value, weight, dtype, parts):
    if dtype == torch.float64 or parts == 1:
        return value @ weight.t()
    bounds = [(weight.shape[1] * i // parts, weight.shape[1] * (i + 1) // parts) for i in range(parts)]
    partial = [value[..., start:end] @ weight[:, start:end].t() for start, end in bounds]
    while len(partial) > 1:
        partial = [partial[i] + partial[i + 1] for i in range(0, len(partial), 2)]
    return partial[0]


def step(x, weights, attention, geometry, dtype, parts=1, same_boundary=False, bf16_probability=False,
         bf16_pv_tile=None):
    return layer_step(x, weights, attention, geometry, dtype, same_boundary=same_boundary,
                      project=lambda value, weight: project(value, weight, dtype, parts),
                      bf16_probability=bf16_probability, bf16_pv_tile=bf16_pv_tile)


def main():
    torch.set_num_threads(int(os.environ.get("GAUDIRT_CPU_THREADS", "8")))
    directory = Path(os.environ["GAUDIRT_TARGET_MODEL"])
    config = json.loads((directory / "config.json").read_text())
    count = config["num_hidden_layers"]
    tile_arg = os.environ.get("GAUDIRT_AUDIT_PV_TILE")
    pv_tile = int(tile_arg) if tile_arg is not None else None
    if pv_tile is not None and pv_tile <= 0:
        raise ValueError("GAUDIRT_AUDIT_PV_TILE must be positive")
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
            same64 = step(state["inputs"][i], w64, state["attention"][i], geometry, torch.float64,
                          same_boundary=True)
            same32 = step(state["inputs"][i], w32, state["attention"][i], geometry, torch.float32,
                          same_boundary=True).double()
            same_pv64 = step(state["inputs"][i], w64, state["attention"][i], geometry, torch.float64,
                             same_boundary=True, bf16_probability=True, bf16_pv_tile=pv_tile)
            record = frozen_layer_metrics(i, actual, truth, same64, same32, same_pv64)
            records.append(record)
            print("FROZEN_LAYER", json.dumps(record, sort_keys=True), flush=True)
            rolling64 = step(rolling64, w64, state["attention"][i], geometry, torch.float64)
            for parts in trees:
                rolling32[parts] = step(rolling32[parts], w32, state["attention"][i], geometry,
                                        torch.float32, parts, same_boundary=True)
            del stored, w64, w32
        failed = [record["layer"] for record in records if not record["pass"]]
        result = {"layers": records, "failed_layers": failed, "bf16_pv_tile": pv_tile,
                  "max_row_relative_l2": max(item["relative_l2_pure"] for item in records),
                  "max_row_relative_l2_same_boundary": max(item["relative_l2_same_boundary"] for item in records)}
        if failed:
            if path := os.environ.get("GAUDIRT_LAYER_RECEIPT"):
                Path(path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            raise AssertionError(f"frozen layer gate failed: {failed}")

        def logits(value, dtype, parts=1, same_boundary=False):
            gamma = weights.get("model.norm.weight").to(dtype)
            normalized = value * torch.rsqrt((value * value).mean(-1, keepdim=True) + geometry[3]) * gamma
            if same_boundary:
                normalized = rounded(normalized, dtype)
            return rounded(project(normalized, weights.get("lm_head.weight").to(dtype), dtype, parts), dtype).double()

        truth_logits = logits(rolling64, torch.float64)
        cpu_logits = {parts: logits(rolling32[parts], torch.float32, parts, same_boundary=True) for parts in trees}
        candidate = state["logits"].double()
        assert all(value.shape == truth_logits.shape and torch.isfinite(value).all()
                   for value in (candidate, truth_logits, *cpu_logits.values()))
        log_truth = torch.log_softmax(truth_logits, dim=-1)
        probabilities = log_truth.exp()

        def divergence(value):
            return (probabilities * (log_truth - torch.log_softmax(value, dim=-1))).sum(-1).max().item()

        kl = divergence(candidate)
        tree_kl = {str(parts): divergence(value) for parts, value in cpu_logits.items()}
        # This capture is itself the M=1 candidate. Its own KL cannot be
        # used to raise its acceptance floor. The Nanbeige acceptance uses
        # the stricter absolute 0.01 bound even when a CPU tree differs more.
        top = torch.topk(truth_logits, 2, dim=-1)
        first = top.values[..., 0].to(torch.bfloat16)
        spacing = (torch.nextafter(first, torch.full_like(first, float("inf"))).float() - first.float()).abs().double()
        tied = top.values[..., 0] - top.values[..., 1] < spacing
        selected = candidate.argmax(-1)
        top1 = selected == top.indices[..., 0]
        top2 = (selected.unsqueeze(-1) == top.indices).any(-1)
        result.update(**assess_conditioned_kl(kl, tree_kl),
                      top1_match_non_tied=bool(top1[~tied].all()) if (~tied).any() else None,
                      top2_match_tied=bool(top2[tied].all()) if tied.any() else None,
                      tied_samples=tied.sum().item())
        if path := os.environ.get("GAUDIRT_LAYER_RECEIPT"):
            Path(path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        assert result["kl_pass"] and torch.where(tied, top2, top1).all(), result
        print("MULTITREE_CONDITIONED_AUDIT", json.dumps(result, sort_keys=True), flush=True)
    finally:
        weights.close()


if __name__ == "__main__":
    main()
