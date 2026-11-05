"""Exercise KV auditing against independent slot corruption cases."""

import torch

from phase5a_state_audit import check_kv_step


def main():
    empty = [torch.zeros((8, 2, 4), dtype=torch.bfloat16) for _ in range(2)]
    locations = torch.tensor([2, 5])
    writes = [(torch.ones((2, 8), dtype=torch.bfloat16),
               torch.full((2, 8), 2, dtype=torch.bfloat16))]
    actual = [buffer.clone() for buffer in empty]
    for buffer, expected in zip(actual, writes[0]):
        buffer[locations] = expected.reshape(2, 2, 4)
    try:
        check_kv_step(None, actual, writes, torch.tensor([2, 2]))
    except AssertionError:
        pass
    else:
        raise AssertionError("missed duplicate KV locations")
    previous = check_kv_step(None, actual, writes, locations)
    new_location = torch.tensor([6])
    new_writes = [(torch.full((1, 8), 3, dtype=torch.bfloat16),
                   torch.full((1, 8), 4, dtype=torch.bfloat16))]
    good = [buffer.clone() for buffer in previous]
    for buffer, expected in zip(good, new_writes[0]):
        buffer[new_location] = expected.reshape(1, 2, 4)
    for label, slot in (("history", 2), ("untouched", 0), ("target", 6)):
        damaged = [buffer.clone() for buffer in good]
        damaged[0][slot, 0, 0] += 1
        try:
            check_kv_step([buffer.clone() for buffer in previous], damaged, new_writes, new_location)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"missed {label} corruption")
    check_kv_step([buffer.clone() for buffer in previous], good, new_writes, new_location)
    nonfinite = [(new_writes[0][0].clone(), new_writes[0][1])]
    nonfinite[0][0][0, 0] = float("nan")
    try:
        check_kv_step([buffer.clone() for buffer in previous], good, nonfinite, new_location)
    except AssertionError:
        pass
    else:
        raise AssertionError("missed nonfinite KV write")
    print("KV audit rejects duplicate, target, history, untouched and nonfinite corruption")


if __name__ == "__main__":
    main()
