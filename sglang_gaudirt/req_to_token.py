"""Write SGLang request mappings with stream-ordered device copies."""

import torch


def read_decode_last_locations(pool, request_indices, sequence_lengths):
    table = pool.req_to_token
    vectors = (request_indices, sequence_lengths)
    if (table.device.type != "gaudirt" or table.dtype != torch.int32 or table.dim() != 2 or
            not all(isinstance(x, torch.Tensor) and x.device.type == "cpu" and x.dim() == 1
                    and x.dtype in (torch.int32, torch.int64) for x in vectors)):
        raise ValueError("GaudiRT decode mapping requires a device table and host metadata")
    requests, lengths = request_indices.tolist(), sequence_lengths.tolist()
    if (not requests or len(requests) != len(lengths) or any(
        request < 0 or request >= table.size(0) or length <= 0 or length > table.size(1)
        for request, length in zip(requests, lengths)
    )):
        raise ValueError("GaudiRT decode mapping metadata is out of range")
    result = torch.empty(len(requests), dtype=torch.int32, device=table.device)
    for index, (request, length) in enumerate(zip(requests, lengths)):
        result[index:index + 1].copy_(table[request, length - 1:length], non_blocking=True)
    return result


def write_req_to_token_pool(pool, request_indices, prefix_lengths, extend_lengths, prefixes, locations):
    table = pool.req_to_token
    host_vectors = (request_indices, prefix_lengths)
    if not all(isinstance(x, torch.Tensor) and x.device.type == "cpu" and x.dim() == 1
               and x.dtype in (torch.int32, torch.int64) for x in host_vectors):
        raise ValueError("GaudiRT request mapping requires host request and prefix vectors")
    if (table.device.type != "gaudirt" or table.dtype != torch.int32 or table.dim() != 2 or
            locations.device.type != "gaudirt" or locations.dim() != 1 or
            locations.dtype not in (torch.int32, torch.int64)):
        raise ValueError("GaudiRT request mapping requires an int32 device table and integer locations")
    requests, starts = request_indices.tolist(), prefix_lengths.tolist()
    if isinstance(extend_lengths, int):
        lengths = [extend_lengths] * len(requests)
    elif (isinstance(extend_lengths, torch.Tensor) and extend_lengths.device.type == "cpu" and
          extend_lengths.dim() == 1 and extend_lengths.dtype in (torch.int32, torch.int64)):
        lengths = extend_lengths.tolist()
    else:
        raise ValueError("GaudiRT request mapping requires host extend lengths")
    if (not requests or len(requests) != len(starts) or len(requests) != len(lengths) or
            len(set(requests)) != len(requests) or sum(lengths) != locations.numel() or
            any(req < 0 or req >= table.size(0) or start < 0 or length <= 0 or
                start + length > table.size(1)
                for req, start, length in zip(requests, starts, lengths))):
        raise ValueError("GaudiRT request mapping metadata is out of range")
    if prefixes is not None and (len(prefixes) != len(requests) or any(
        prefix.numel() != start or (start and prefix.device.type != "gaudirt")
        for prefix, start in zip(prefixes, starts)
    )):
        raise ValueError("GaudiRT request mapping prefix lengths do not match")
    offset = 0
    for index, (request, start, length) in enumerate(zip(requests, starts, lengths)):
        row = table[request]
        if prefixes is not None and start:
            row[:start].copy_(prefixes[index], non_blocking=True)
        row[start:start + length].copy_(locations[offset:offset + length], non_blocking=True)
        offset += length
