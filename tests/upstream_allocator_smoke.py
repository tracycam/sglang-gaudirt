"""SGLang platform identity and token-slot allocator against GaudiRT."""

import os
import sys
from types import SimpleNamespace

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
from sglang.srt.platforms import current_platform
from sglang.srt.utils.common import get_device


def main():
    assert get_device() == "gaudirt"
    assert get_device(0) == "gaudirt:0"
    args = SimpleNamespace(page_size=None, disable_cuda_graph=False)
    current_platform.apply_server_args_defaults(args)
    assert args.page_size == 1 and args.disable_cuda_graph
    try:
        current_platform.apply_server_args_defaults(SimpleNamespace(page_size=16))
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported KV page size was accepted")
    current_platform.init_backend()
    allocator_cls = current_platform.get_paged_allocator_cls()
    allocator = allocator_cls(20, 1, torch.bfloat16, "gaudirt:0", None, False)
    assert allocator.available_size() == 20
    first, second = allocator.alloc(3), allocator.alloc(4)
    assert first.cpu().tolist() == [1, 2, 3]
    assert second.cpu().tolist() == [4, 5, 6, 7]
    assert allocator.available_size() == 13
    allocator.free(first)
    assert allocator.available_size() == 16
    rest = allocator.alloc(16)
    assert rest.cpu().tolist() == list(range(8, 21)) + [1, 2, 3]
    assert allocator.alloc(1) is None
    try:
        allocator_cls(20, 2, torch.bfloat16, "gaudirt:0", None, False)
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported KV allocator was accepted")
    assert current_platform.get_current_memory_usage() > 0
    assert current_platform.get_available_memory()[0] < current_platform.get_device_total_memory()
    print("SGLang platform identity and token-slot allocator passed", flush=True)


if __name__ == "__main__":
    main()
