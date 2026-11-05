"""Build scheduler positions from the host lengths SGLang already retains."""

import torch


def _device_vector(values, dtype, device):
    return torch.tensor(values, dtype=dtype, device="cpu").to(device, non_blocking=True)


def decode_positions(seq_lens_cpu, device):
    if (not isinstance(seq_lens_cpu, torch.Tensor) or seq_lens_cpu.device.type != "cpu" or
            seq_lens_cpu.dim() != 1 or seq_lens_cpu.dtype not in (torch.int32, torch.int64)):
        raise ValueError("GaudiRT decode positions require host sequence lengths")
    lengths = seq_lens_cpu.tolist()
    if not lengths or len(lengths) > 4096 or any(length <= 0 for length in lengths):
        raise ValueError("GaudiRT decode sequence lengths are out of range")
    return _device_vector([length - 1 for length in lengths], torch.int64, device)


def extend_positions(prefix_lens_cpu, seq_lens_cpu, total, device):
    if (not isinstance(prefix_lens_cpu, list) or not isinstance(seq_lens_cpu, list) or
            not prefix_lens_cpu or len(prefix_lens_cpu) != len(seq_lens_cpu) or
            not isinstance(total, int) or not 0 < total <= 4096 or sum(seq_lens_cpu) != total or
            any(not isinstance(start, int) or not isinstance(length, int) or start < 0 or length <= 0
                for start, length in zip(prefix_lens_cpu, seq_lens_cpu))):
        raise ValueError("GaudiRT extend positions require valid host sequence lengths")
    positions, offsets = [], []
    for start, length in zip(prefix_lens_cpu, seq_lens_cpu):
        offsets.append(len(positions))
        positions.extend(range(start, start + length))
    return (_device_vector(positions, torch.int64, device),
            _device_vector(offsets, torch.int32, device))
