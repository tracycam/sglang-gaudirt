"""RoPE initialization uses CPU and restores device and thread settings."""

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
    with torch.device("meta"):
        with current_platform.rope_cache_initialization_context():
            assert torch.get_num_threads() == 1
            assert torch.arange(4, device=None).device.type == "cpu"
        assert torch.empty(0).device.type == "meta"
    assert torch.get_num_threads() == 4
    try:
        with torch.device("meta"):
            try:
                with current_platform.rope_cache_initialization_context():
                    assert torch.get_num_threads() == 1
                    assert torch.empty(0).device.type == "cpu"
                    raise ValueError("expected test exception")
            finally:
                assert torch.empty(0).device.type == "meta"
    except ValueError as error:
        assert "expected test exception" in str(error)
    else:
        raise AssertionError("test exception was lost")
    assert torch.get_num_threads() == 4
    print("RoPE initialization restores device and CPU thread settings", flush=True)


if __name__ == "__main__":
    main()
