"""Check positions generated from the scheduler's host lengths."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import torch
from sglang.srt.platforms import current_platform


def main():
    current_platform.init_backend()
    for total in (1, 3, 19, 137, 513, 4096):
        lengths = [total] if total == 1 else [total // 2, total - total // 2]
        prefixes = [5 + index for index in range(len(lengths))]
        positions, offsets = current_platform.prepare_extend_positions(
            prefixes, lengths, total, "gaudirt:0"
        )
        expected = [position for start, length in zip(prefixes, lengths)
                    for position in range(start, start + length)]
        assert positions.cpu().tolist() == expected
        assert offsets.cpu().tolist() == ([0] if len(lengths) == 1 else [0, lengths[0]])
        seq_lens = torch.tensor([start + length for start, length in zip(prefixes, lengths)])
        decode = current_platform.prepare_decode_positions(seq_lens, "gaudirt:0")
        assert decode.cpu().tolist() == [length - 1 for length in seq_lens.tolist()]
    for total in (0, 4097):
        try:
            current_platform.prepare_extend_positions([0], [total], total, "gaudirt:0")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid token count accepted")
    print("SGLang extend/decode positions from host metadata passed", flush=True)


if __name__ == "__main__":
    main()
