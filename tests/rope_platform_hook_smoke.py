"""The RoPE initialization hook restores process-local CPU threading."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
from sglang.srt.platforms import current_platform


def main():
    torch.set_num_threads(4)
    with current_platform.rope_cache_initialization_context():
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == 4
    try:
        with current_platform.rope_cache_initialization_context():
            assert torch.get_num_threads() == 1
            raise ValueError("expected test exception")
    except ValueError as error:
        assert "expected test exception" in str(error)
    else:
        raise AssertionError("test exception was lost")
    assert torch.get_num_threads() == 4
    print("RoPE initialization hook restores CPU thread settings", flush=True)


if __name__ == "__main__":
    main()
