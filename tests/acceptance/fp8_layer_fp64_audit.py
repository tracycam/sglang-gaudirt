"""Audit frozen block-FP8 layers against original OCP W8A8 CPU references."""

import json
import math
import os
from pathlib import Path

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"

import torch
from layer_fp64_audit import (Weights, audit_legacy_bridge, frozen_layer_metrics,
                              layer_attention_step, layer_post_attention_step, layer_step, rounded)


def dequantize_weight(raw, scale, dtype):
    if (raw.dtype != torch.float8_e4m3fn or raw.ndim != 2 or raw.shape[0] % 128 or raw.shape[1] % 128
            or tuple(scale.shape) != (raw.shape[0] // 128, raw.shape[1] // 128)
            or not torch.isfinite(raw.float()).all() or not torch.isfinite(scale).all() or (scale < 0).any()):
        raise ValueError("invalid original OCP block weight or scale")
    n, k = raw.shape
    return (raw.to(dtype).reshape(n // 128, 128, k // 128, 128) *
            scale.to(dtype).reshape(n // 128, 1, k // 128, 1)).reshape(n, k)


def ocp_activation(value, dtype):
    # Linear inputs are BF16 tensors in the model. FP64 values use direct
    # BF16 rounding so that the CPU reference does not double-round.
    bf16 = rounded(value, dtype)
    group = bf16.float().reshape(-1, value.shape[-1] // 128, 128)
    scale = group.abs().amax(-1, keepdim=True) / 448.0
    denominator = torch.where(scale == 0, 1.0, scale)
    quantized = (group / denominator).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (quantized.to(dtype) * scale.to(dtype)).reshape(value.shape)


def quantized_project(value, weight):
    if value.shape[-1] % 128:
        raise ValueError("FP8 activation width must be a multiple of 128")
    return ocp_activation(value, value.dtype) @ weight.t()


def pure_quantized_project(value, weight):
    """Pure-layer FP64 path: no BF16 storage at internal linear inputs."""
    if value.dtype != torch.float64 or value.shape[-1] % 128 or not torch.isfinite(value).all():
        raise ValueError("pure FP8 reference requires finite FP64 inputs with aligned width")
    group = value.reshape(-1, value.shape[-1] // 128, 128)
    scale = group.abs().amax(-1, keepdim=True).float() / 448.0
    denominator = torch.where(scale == 0, 1.0, scale)
    magnitude = (group / denominator.double()).abs().clamp_max(448.0).contiguous()
    levels = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).double()
    upper = torch.searchsorted(levels, magnitude).clamp(1, 126)
    lower = upper - 1
    low_distance = magnitude - levels[lower]
    high_distance = levels[upper] - magnitude
    choose_upper = (high_distance < low_distance) | ((high_distance == low_distance) & ((lower & 1) != 0))
    quantized = torch.where(choose_upper, levels[upper], levels[lower])
    quantized = torch.where(torch.signbit(group), -quantized, quantized)
    return (quantized * scale.double()).reshape(value.shape) @ weight.t()


def unquantized_project(value, weight):
    return rounded(value, value.dtype) @ weight.t()


def row_error(actual, reference):
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    denominator = torch.linalg.vector_norm(reference, dim=-1)
    assert (denominator > 0).all()
    return (torch.linalg.vector_norm(actual - reference, dim=-1) / denominator).max().item()


def reference_gate_overlap(pure, same_boundary, pure_limit, boundary_limit):
    """Necessary triangle-inequality condition for both relative L2 gates."""
    assert torch.isfinite(pure).all() and torch.isfinite(same_boundary).all()
    pure_norm = torch.linalg.vector_norm(pure).item()
    boundary_norm = torch.linalg.vector_norm(same_boundary).item()
    separation = torch.linalg.vector_norm(pure - same_boundary).item()
    if not all(math.isfinite(number) for number in (pure_norm, boundary_norm, separation)):
        raise ValueError("reference norm or separation is nonfinite")
    allowed = pure_limit * pure_norm + boundary_limit * boundary_norm
    return {"reference_separation_l2": separation, "pure_reference_l2_norm": pure_norm,
            "same_boundary_reference_l2_norm": boundary_norm, "allowed_separation_l2": allowed,
            "gates_can_overlap": separation <= allowed}


def fp8_gates(stage_errors, projection_errors, layer_bf16_pv_error, layer_fp64_probability_error):
    if len(stage_errors) != 2 or len(projection_errors) != 7:
        raise ValueError("FP8 layer audit requires both stages and all seven projections")
    errors = (*stage_errors, *projection_errors.values(), layer_bf16_pv_error, layer_fp64_probability_error)
    if not all(math.isfinite(value) and value >= 0 for value in errors):
        raise ValueError("FP8 layer error is nonfinite or negative")
    return (max(stage_errors) <= 0.01 and max(projection_errors.values()) <= 0.01 and
            layer_bf16_pv_error <= 0.01 and layer_fp64_probability_error <= 0.02)


def frozen_projection_errors(layer, segments, weights):
    """Compare each exposed projection with OCP quantization of its captured input."""
    before, after = segments[2 * layer], segments[2 * layer + 2]
    if len(before) != (1 if layer == 0 else 4) or len(after) not in (3, 4):
        raise ValueError("unexpected FP8 segment projection layout")
    qkv, output, gate_up, down = before[-1], *after[:3]
    errors = {}

    def check(capture, names):
        source = capture["input"].double()
        actual = capture["output"].double()
        sizes = [weights[name].shape[0] for name in names]
        if actual.shape[-1] != sum(sizes):
            raise ValueError("captured FP8 projection width does not match checkpoint weights")
        for name, observed in zip(names, actual.split(sizes, dim=-1)):
            reference = rounded(quantized_project(source, weights[name]), torch.float64)
            errors[name] = row_error(observed, reference)

    check(qkv, ("q", "k", "v"))
    check(output, ("out",))
    check(gate_up, ("gate", "up"))
    check(down, ("down",))
    return errors


def main():
    audit_legacy_bridge()
    torch.set_num_threads(int(os.environ.get("GAUDIRT_CPU_THREADS", "8")))
    directory = Path(os.environ["GAUDIRT_TARGET_MODEL"])
    config = json.loads((directory / "config.json").read_text())
    quantization = config["quantization_config"]
    schemes = list(quantization["config_groups"].values())
    if (quantization["quant_method"] != "compressed-tensors" or len(schemes) != 1 or
            schemes[0]["weights"]["strategy"] != "block" or
            list(schemes[0]["weights"]["block_structure"]) != [128, 128] or
            schemes[0]["input_activations"]["strategy"] != "group" or
            schemes[0]["input_activations"]["group_size"] != 128 or
            not schemes[0]["input_activations"]["dynamic"]):
        raise ValueError("expected the declared OCP 128x128 block / dynamic group-128 W8A8 scheme")
    count = config["num_hidden_layers"]
    state = torch.load(os.environ["GAUDIRT_LAYER_CAPTURE_PATH"], map_location="cpu", weights_only=True)
    assert len(state["inputs"]) == count + 1 and len(state["attention"]) == count
    segments = state.get("fp8_segments")
    if not isinstance(segments, dict) or set(segments) != set(range(0, 2 * count + 1, 2)):
        raise ValueError("capture requires GAUDIRT_CAPTURE_FP8_STAGES=1 for every layer")
    dim = config["head_dim"]
    geometry = (config["num_attention_heads"], config["num_key_value_heads"], dim,
                torch.tensor(config["rms_norm_eps"], dtype=torch.float32).item(),
                torch.tensor(1 / math.sqrt(dim), dtype=torch.float32).item(), config.get("rope_theta", 10000))
    assert geometry[0] % geometry[1] == 0
    tile_arg = os.environ.get("GAUDIRT_AUDIT_PV_TILE")
    pv_tile = "full" if tile_arg == "full" else int(tile_arg) if tile_arg is not None else None
    names = {"in_norm": "input_layernorm.weight", "post_norm": "post_attention_layernorm.weight",
             "q": "self_attn.q_proj.weight", "k": "self_attn.k_proj.weight", "v": "self_attn.v_proj.weight",
             "out": "self_attn.o_proj.weight", "gate": "mlp.gate_proj.weight",
             "up": "mlp.up_proj.weight", "down": "mlp.down_proj.weight"}
    quantized_names = set(names) - {"in_norm", "post_norm"}
    rolling64 = state["inputs"][0].double()
    rolling32 = state["inputs"][0].float()
    records = []
    weights = Weights(directory)
    try:
        for i in range(count):
            base = f"model.layers.{i}."
            w64, w32 = {}, {}
            for name, suffix in names.items():
                raw = weights.get(base + suffix)
                if name in quantized_names:
                    scale = weights.get(base + suffix.removesuffix("weight") + "weight_scale").float()
                    w64[name] = dequantize_weight(raw, scale, torch.float64)
                    w32[name] = dequantize_weight(raw, scale, torch.float32)
                else:
                    w64[name], w32[name] = raw.double(), raw.float()
            x, attention = state["inputs"][i], state["attention"][i]
            actual = state["inputs"][i + 1].double()
            truth = layer_step(x, w64, attention, geometry, torch.float64, project=pure_quantized_project)
            same64 = layer_step(x, w64, attention, geometry, torch.float64, same_boundary=True,
                                project=quantized_project)
            same32 = layer_step(x, w32, attention, geometry, torch.float32, same_boundary=True,
                                project=quantized_project).double()
            same_pv64 = layer_step(x, w64, attention, geometry, torch.float64, same_boundary=True,
                                   project=quantized_project, bf16_probability=True, bf16_pv_tile=pv_tile)
            unquantized = layer_step(x, w64, attention, geometry, torch.float64, same_boundary=True,
                                     project=unquantized_project)
            record = frozen_layer_metrics(i, actual, truth, same64, same32, same_pv64)
            record["diagnostic_pure_formula_limit"] = record.pop("limit_pure")
            record["fp64_ocp_w8a8_l2"] = row_error(actual, same64)
            record["fp64_unquantized_activation_l2"] = row_error(actual, unquantized)
            record["reference_gate_overlap"] = reference_gate_overlap(truth, same_pv64, 0.01, 0.01)
            stage1 = layer_attention_step(x, w64, attention, geometry, torch.float64,
                                          same_boundary=True, project=quantized_project,
                                          bf16_probability=True, bf16_pv_tile=pv_tile)
            stage2 = layer_post_attention_step(x, attention["output"], w64, geometry, torch.float64,
                                               same_boundary=True, project=quantized_project)
            stage_errors = [row_error(attention["output"].double(), stage1), row_error(actual, stage2)]
            projection_errors = frozen_projection_errors(i, segments, w64)
            record.update({"stage_attention_l2": stage_errors[0], "stage_post_attention_l2": stage_errors[1],
                           "frozen_projection_l2": projection_errors, "limit_stage": 0.01,
                           "limit_projection": 0.01, "limit_same_boundary": 0.01,
                           "limit_same_operators_fp64_probability": 0.02,
                           "pure_reference_diagnostic_only": True})
            record["pass"] = fp8_gates(stage_errors, projection_errors,
                                       record["relative_l2_same_boundary"], record["fp64_ocp_w8a8_l2"])
            records.append(record)
            print("FROZEN_FP8_LAYER", json.dumps(record, sort_keys=True), flush=True)
            rolling64 = layer_step(rolling64, w64, attention, geometry, torch.float64,
                                   same_boundary=True, project=quantized_project,
                                   bf16_probability=True, bf16_pv_tile=pv_tile)
            rolling32 = layer_step(rolling32, w32, attention, geometry, torch.float32, same_boundary=True,
                                   project=quantized_project)
            del w64, w32

        def logits(value, dtype, same_boundary=False):
            gamma = weights.get("model.norm.weight").to(dtype)
            normalized = value * torch.rsqrt((value * value).mean(-1, keepdim=True) + geometry[3]) * gamma
            if same_boundary:
                normalized = rounded(normalized, dtype)
            return rounded(normalized @ weights.get("lm_head.weight").to(dtype).t(), dtype).double()

        truth_logits = logits(rolling64, torch.float64)
        floor_logits = logits(rolling32, torch.float32, same_boundary=True)
        candidate = state["logits"].double()
        assert truth_logits.shape == floor_logits.shape == candidate.shape
        assert all(torch.isfinite(value).all() for value in (truth_logits, floor_logits, candidate))
        log_truth = torch.log_softmax(truth_logits, dim=-1)
        probability = log_truth.exp()

        def divergence(value):
            return (probability * (log_truth - torch.log_softmax(value, dim=-1))).sum(-1).max().item()

        kl, floor_kl = divergence(candidate), divergence(floor_logits)
        top = torch.topk(truth_logits, 2, dim=-1)
        first = top.values[..., 0].to(torch.bfloat16)
        spacing = (torch.nextafter(first, torch.full_like(first, float("inf"))).float() - first.float()).abs().double()
        tied = top.values[..., 0] - top.values[..., 1] < spacing
        selected = candidate.argmax(-1)
        top1 = selected == top.indices[..., 0]
        top2 = (selected.unsqueeze(-1) == top.indices).any(-1)
        result = {"reference": "original_ocp_w8a8", "layers": records, "bf16_pv_tile": pv_tile,
                  "max_fp64_ocp_w8a8_l2": max(item["fp64_ocp_w8a8_l2"] for item in records),
                  "max_fp64_unquantized_activation_l2": max(item["fp64_unquantized_activation_l2"] for item in records),
                  "max_kl": kl, "ref_floor": floor_kl, "kl_limit": 0.01,
                  "top1_match_non_tied": bool(top1[~tied].all()) if (~tied).any() else None,
                  "top2_match_tied": bool(top2[tied].all()) if tied.any() else None,
                  "tied_samples": tied.sum().item()}
        if path := os.environ.get("GAUDIRT_LAYER_RECEIPT"):
            Path(path).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        assert all(item["pass"] for item in records), result
        assert kl <= 0.01 and torch.where(tied, top2, top1).all(), result
        print("FP8_LAYER_AND_CONDITIONED_AUDIT", json.dumps(result, sort_keys=True), flush=True)
    finally:
        weights.close()
        audit_legacy_bridge()


if __name__ == "__main__":
    main()
