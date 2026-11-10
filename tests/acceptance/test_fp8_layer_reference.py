"""Check that FP8 layer references use original OCP codes and scales."""

import os
import sys
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "GaudiRT" / "tests"))

import torch
import fp8_layer_fp64_audit as reference


def main():
    generator = torch.Generator().manual_seed(56)
    raw = torch.randn(256, 256, generator=generator).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.125, 0.25], [0.5, 1.0]], dtype=torch.float32)
    expected_weight = raw.double() * scale.double().repeat_interleave(128, 0).repeat_interleave(128, 1)
    weight = reference.dequantize_weight(raw, scale, torch.float64)
    assert torch.equal(weight, expected_weight)

    value = torch.randn(3, 256, generator=generator).to(torch.bfloat16).double()
    value[0].zero_()
    group = value.float().reshape(3, 2, 128)
    activation_scale = group.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0
    quantized = (group / activation_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    expected_activation = (quantized.double() * activation_scale.double()).reshape(3, 256)
    assert torch.equal(reference.ocp_activation(value, torch.float64), expected_activation)
    assert torch.equal(reference.quantized_project(value, weight), expected_activation @ expected_weight.t())
    assert not torch.equal(reference.unquantized_project(value, weight),
                           reference.quantized_project(value, weight))
    try:
        reference.dequantize_weight(raw, torch.full((2, 2), float("inf")), torch.float64)
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite checkpoint scale accepted")
    print("independent OCP W8A8 and unquantized activation references passed", flush=True)


if __name__ == "__main__":
    main()
