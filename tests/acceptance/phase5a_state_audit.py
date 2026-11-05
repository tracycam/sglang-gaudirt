"""Compare every KV byte against the previous state plus this step's writes."""

import torch


def check_kv_step(previous, buffers, writes, locations):
    rows = locations.numel()
    assert rows and torch.unique(locations).numel() == rows, "duplicate KV write locations"
    assert len(buffers) == 2 * len(writes)
    if previous is None:
        previous = [torch.zeros_like(buffer) for buffer in buffers]
    for index, buffer in enumerate(buffers):
        expected = writes[index // 2][index % 2]
        assert torch.isfinite(expected).all(), "nonfinite KV write"
        expected = expected.reshape((rows,) + tuple(buffer.shape[1:]))
        previous[index].index_copy_(0, locations, expected)
        assert torch.equal(previous[index].view(torch.int16), buffer.view(torch.int16)), (
            "KV write differs, historical KV changed, or untouched KV changed"
        )
    return buffers
