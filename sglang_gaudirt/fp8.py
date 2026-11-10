"""Load-time preparation and dispatch for the supported OCP block-FP8 scheme."""

import torch
from torch.nn import Parameter

from gaudi_kernels.fp8_codec import reencode_e4m3fn_half


def prepare_block_weight(layer, scale_name):
    weight = layer.weight
    scales = getattr(layer, scale_name)
    if (weight.dtype != torch.float8_e4m3fn or weight.ndim != 2 or
            scales.dtype != torch.float32 or scales.ndim != 2):
        raise ValueError("GaudiRT FP8: expected an OCP matrix and FP32 block scales")
    n, k = weight.shape
    if n <= 0 or k <= 0 or n % 128 or k % 128 or tuple(scales.shape) != (n // 128, k // 128):
        raise ValueError("GaudiRT FP8: block matrix and scales must have aligned 128x128 geometry")
    device = weight.device
    if scales.device != device:
        raise ValueError("GaudiRT FP8: weight and scales must share a device")
    # Loading and packing run once. Retain only the packed one-byte weights
    # on the device; raw checkpoint tensors remain the independent reference.
    raw = weight.detach().cpu().contiguous()
    raw_scales = scales.detach().cpu().contiguous()
    encoded, adjusted, _ = reencode_e4m3fn_half(raw, raw_scales)
    packed = encoded.view(torch.uint8).reshape(n, k // 128, 128).permute(1, 0, 2).contiguous()
    packed = packed.view(torch.float8_e4m3fn)
    del weight, scales
    layer.weight = None
    setattr(layer, scale_name, None)
    layer.weight = Parameter(packed.to(device), requires_grad=False)
    setattr(layer, scale_name, Parameter(adjusted.to(device), requires_grad=False))
    getattr(layer, scale_name).format_ue8m0 = False
    layer.input_scale = None


def block_fp8_linear(input, weight, block_size, weight_scale, input_scale=None, bias=None):
    if tuple(block_size) != (128, 128) or input_scale is not None:
        raise ValueError("GaudiRT FP8: only dynamic activation and 128x128 weight blocks are supported")
    return torch.ops.gaudirt.block_fp8_linear.default(input, weight, weight_scale, bias)
