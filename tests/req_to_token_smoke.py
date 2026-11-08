"""Check extend and decode mapping writes against SGLang's real request pool."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import torch
from sglang.srt.platforms import current_platform


def main():
    current_platform.init_backend()
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

    pool = ReqToTokenPool(4, 16, "gaudirt:0", False)
    pool.req_to_token.fill_(-17)
    requests = torch.tensor([1, 3], dtype=torch.int64)
    starts = torch.tensor([2, 1], dtype=torch.int64)
    lengths = torch.tensor([3, 2], dtype=torch.int64)
    prefixes = [torch.tensor([8, 9], device="gaudirt:0"),
                torch.tensor([11], device="gaudirt:0")]
    locations = torch.tensor([20, 21, 22, 30, 31], device="gaudirt:0")
    assert current_platform.write_req_to_token_pool(pool, requests, starts, lengths, prefixes, locations)
    expected = torch.full((5, 16), -17, dtype=torch.int32)
    expected[1, :5] = torch.tensor([8, 9, 20, 21, 22], dtype=torch.int32)
    expected[3, :3] = torch.tensor([11, 30, 31], dtype=torch.int32)
    assert torch.equal(pool.req_to_token.cpu(), expected)
    assert current_platform.write_req_to_token_pool(
        pool, requests, torch.tensor([5, 3]), 1, None,
        torch.tensor([23, 32], device="gaudirt:0"),
    )
    expected[1, 5] = 23
    expected[3, 3] = 32
    assert torch.equal(pool.req_to_token.cpu(), expected)
    last = current_platform.paged_decode_last_locations(pool, requests, torch.tensor([6, 4]))
    assert last.cpu().tolist() == [23, 32]
    for invalid in (torch.tensor([0, 4]), torch.tensor([6, 17])):
        try:
            current_platform.paged_decode_last_locations(pool, requests, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid decode mapping accepted")
    try:
        current_platform.write_req_to_token_pool(pool, requests, starts, 1, None, locations)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched locations accepted")
    assert torch.equal(pool.req_to_token.cpu(), expected)
    print("SGLang extend/decode request mappings and untouched slots passed", flush=True)


if __name__ == "__main__":
    main()
