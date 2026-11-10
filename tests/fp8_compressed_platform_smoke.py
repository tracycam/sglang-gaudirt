"""Exercise SGLang's generic block-FP8 scheme through the OOT platform."""

import json
import os
import sys
from pathlib import Path
from unittest import mock

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import torch
from compressed_tensors.quantization import QuantizationArgs
from gaudi_kernels.fp8_codec import reencode_e4m3fn_half
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
from sglang.srt.layers.quantization.compressed_tensors.schemes import CompressedTensorsW8A8Fp8
from sglang.srt.platforms import current_platform

import gaudirt
import gaudirt.compiler as compiler


def main():
    torch.random.default_generator.manual_seed(149)
    config = json.loads(Path(os.environ["GAUDIRT_FP8_CONFIG"]).read_text())["quantization_config"]
    group = config["config_groups"]["group_0"]
    weights = QuantizationArgs.model_validate(group["weights"])
    activations = QuantizationArgs.model_validate(group["input_activations"])
    assert current_platform.supports_compressed_w8a8_fp8(weights, activations)
    assert not current_platform.supports_compressed_w8a8_fp8(
        weights, activations.model_copy(update={"group_size": 64})
    )
    quant = CompressedTensorsConfig({}, [], config["format"], {}, [])
    with mock.patch.object(torch.cuda, "get_device_capability", side_effect=AssertionError("CUDA queried")):
        scheme = quant._get_scheme_from_parts(weights, activations)
    assert isinstance(scheme, CompressedTensorsW8A8Fp8)
    assert scheme.w8a8_block_fp8_linear is current_platform.get_w8a8_block_fp8_linear()

    n = k = 256
    original = torch.randn(n, k).clamp(-4, 4).to(torch.float8_e4m3fn)
    original_scale = torch.rand(n // 128, k // 128, dtype=torch.float32) + 0.1
    expected_weight, expected_scale, _ = reencode_e4m3fn_half(original, original_scale)
    expected_weight = expected_weight.view(torch.uint8).reshape(n, k // 128, 128).permute(1, 0, 2).contiguous()
    gaudirt.initialize(hbm_bytes=512 * 1024 * 1024)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(original.to("gaudirt:0"), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(original_scale.to("gaudirt:0"), requires_grad=False)
    scheme.process_weights_after_loading(layer)
    assert torch.equal(layer.weight.cpu().view(torch.uint8), expected_weight)
    assert torch.equal(layer.weight_scale.cpu(), expected_scale)
    assert layer.input_scale is None

    def apply(x, weight, scale):
        return scheme.w8a8_block_fp8_linear(x, weight, (128, 128), scale)

    compiled = compiler.compile(apply, dynamic_arg_indices=(0,))
    x_cpu = torch.randn(17, k).to(torch.bfloat16)
    output = compiled(x_cpu.to("gaudirt:0"), layer.weight, layer.weight_scale).cpu().double()
    grouped = x_cpu.float().reshape(17, k // 128, 128)
    scale = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0
    q = (grouped / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    ref_x = (q.double() * scale.double()).reshape(17, k)
    ref_w = (original.double().reshape(n // 128, 128, k // 128, 128) *
             original_scale.double().reshape(n // 128, 1, k // 128, 1)).reshape(n, k)
    ref = ref_x @ ref_w.T
    unquantized_ref = x_cpu.double() @ ref_w.T
    assert torch.isfinite(output).all() and torch.isfinite(ref).all() and torch.isfinite(unquantized_ref).all()
    error = (torch.linalg.vector_norm(output - ref, dim=-1) / torch.linalg.vector_norm(ref, dim=-1)).max().item()
    diagnostic = (torch.linalg.vector_norm(output - unquantized_ref, dim=-1) /
                  torch.linalg.vector_norm(unquantized_ref, dim=-1)).max().item()
    assert error <= 0.02, (error, diagnostic)
    print("compressed FP8 platform", "fp64_ocp_w8a8_l2", error,
          "fp64_unquantized_activation_l2", diagnostic, "cache", gaudirt.recipe_cache_stats(), flush=True)
    del compiled, layer
    gaudirt.empty_cache()
    gaudirt.shutdown()
    assert "habana_frameworks" not in sys.modules


if __name__ == "__main__":
    main()
