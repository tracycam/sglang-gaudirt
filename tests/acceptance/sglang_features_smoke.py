"""Compare ordinary SGLang inference with chunking and prefix caching enabled."""

import copy
import hashlib
import json
import os
import operator
import struct
import sys
from pathlib import Path

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import gaudirt
import sglang
import torch
from sglang_gaudirt.platform import GaudiRTSRTPlatform
from phase5a_feature_prompts import feature_prompts

feature_writes = []
feature_values = []
capture_values = bool(os.environ.get("GAUDIRT_FEATURE_CAPTURE_DIR"))
if os.environ.get("GAUDIRT_FEATURE_STATE_AUDIT") == "1" or capture_values:
    import gaudirt.compiler as compiler

    original_backend = compiler.gaudirt_backend

    def audit_backend(graph, inputs):
        compiled = original_backend(graph, inputs)
        compiled.__self__.enable_trace()
        segments = list(compiled.__self__.named_children())
        for index, (_, segment) in enumerate(segments):
            if index % 2 != 1:
                if capture_values and index < len(segments) - 1:
                    previous = segment.forward
                    audit_segment = None
                    audit_compiled = []
                    if index == 0 and os.environ.get("GAUDIRT_FEATURE_CAPTURE_NORM") == "1":
                        audit_segment = copy.deepcopy(segment)
                        norms = [node for node in audit_segment.graph.nodes if node.op == "call_function" and
                                 node.target in (torch.ops.gaudirt.rms_norm.default,
                                                 torch.ops.gaudirt.fused_add_rms_norm.default)]
                        assert len(norms) == 1, audit_segment.graph
                        norm = norms[0]
                        if norm.target == torch.ops.gaudirt.fused_add_rms_norm.default:
                            with audit_segment.graph.inserting_after(norm):
                                norm = audit_segment.graph.call_function(operator.getitem, (norm, 0))
                        result = next(node for node in audit_segment.graph.nodes if node.op == "output")
                        original_outputs = result.args[0]
                        if not isinstance(original_outputs, (tuple, list)):
                            original_outputs = (original_outputs,)
                        result.args = ((*original_outputs, norm),)
                        audit_segment.graph.lint()

                    def layer(*args, previous=previous, index=index, audit_segment=audit_segment,
                              audit_compiled=audit_compiled):
                        output = previous(*args)
                        feature_values.append((index, output[3][-1].cpu()))
                        rows = output[3].size(0)
                        if rows >= 1024 and audit_segment is not None:
                            if not audit_compiled:
                                audit_compiled.append(compiler._program_backend(audit_segment, args))
                            audited = audit_compiled[0](*args)
                            norm = audited[-1]
                            for original, replay in zip(output, audited[:-1]):
                                if isinstance(original, torch.Tensor) and original.ndim and original.size(0) == rows:
                                    if os.environ.get("GAUDIRT_FEATURE_CAPTURE_NORM_ALL") == "1":
                                        assert torch.equal(original.cpu(), replay.cpu())
                                    for row in (2, 21):
                                        assert torch.equal(original[row].cpu(), replay[row].cpu())
                            for row in (2, 21):
                                feature_values.append((f"{index}_norm_row{row}", norm[row].cpu()))
                            if os.environ.get("GAUDIRT_FEATURE_CAPTURE_NORM_ALL") == "1":
                                feature_values.append((f"{index}_norm_all", norm.cpu()))
                        if rows >= 1024:
                            for value_index, value in enumerate(output):
                                if isinstance(value, torch.Tensor) and value.ndim and value.size(0) == rows:
                                    for row in (2, 21):
                                        feature_values.append((f"{index}_fx{value_index}_row{row}", value[row].cpu()))
                        return output

                    segment.forward = layer
                continue
            nodes = tuple(segment.graph.nodes)
            placeholders = [node for node in nodes if node.op == "placeholder"]
            attention = [node for node in nodes if node.op == "call_function" and
                         node.target == torch.ops.gaudirt._paged_attn_validated.default]
            assert len(attention) == 1
            indexes = [placeholders.index(node) for node in attention[0].args[:8]]
            previous = segment.forward

            def paged(*args, previous=previous, index=index, indexes=indexes):
                output = previous(*args)
                query_index, key_index, value_index, key_pool_index, value_pool_index, _, request_index, position_index = indexes
                if os.environ.get("GAUDIRT_FEATURE_STATE_AUDIT") == "1":
                    feature_writes.append((args[key_index].cpu(), args[value_index].cpu()))
                if capture_values:
                    for name, tensor_index in (("query", query_index), ("key", key_index), ("value", value_index)):
                        feature_values.append((f"{index}_{name}", args[tensor_index][-1].cpu()))
                        if args[tensor_index].size(0) >= 1024:
                            for row in (2, 21):
                                feature_values.append((f"{index}_{name}_row{row}", args[tensor_index][row].cpu()))
                    if os.environ.get("GAUDIRT_FEATURE_CAPTURE_KV") == "1" and args[query_index].size(0) > 1:
                        from sglang.srt.model_executor.forward_context import get_attn_backend

                        request = int(args[request_index][-1].cpu())
                        position = int(args[position_index][-1].cpu())
                        mapping = get_attn_backend().req_to_token_pool.req_to_token
                        slots = mapping.cpu()[request, :position + 1].long()
                        feature_values.append((f"{index}_keys", args[key_pool_index].cpu()[slots]))
                        feature_values.append((f"{index}_values", args[value_pool_index].cpu()[slots]))
                    feature_values.append((index, output[-1].cpu()))
                return output

            segment.forward = paged
        return compiled

    compiler.gaudirt_backend = audit_backend

original_prepare = GaudiRTSRTPlatform.prepare_model_forward


def audit_prepare(self, forward):
    compiled = original_prepare(self, forward)
    event_counts = None
    previous_events = None
    event_steps = 0
    previous_kv = None
    audit_steps = 0
    capture_steps = 0
    completed_prefills = 0
    request_index = -1
    audit_mode = os.environ.get("GAUDIRT_FEATURE_STATE_AUDIT") == "1"
    expected_audit_steps = int(os.environ.get("GAUDIRT_FEATURE_EXPECT_STEPS", "10"))
    if audit_mode:
        from phase5a_state_audit import check_kv_step

        mode = os.environ["GAUDIRT_AUDIT_MODE"]
        assert mode in ("cold", "warm")
        trace = open(os.environ["GAUDIRT_TRACE_PATH"], "wb" if mode == "cold" else "rb")
        digest = hashlib.sha256()

        def emit(tensor):
            raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            header = struct.pack("<Q", len(raw))
            digest.update(header)
            digest.update(raw)
            if mode == "cold":
                trace.write(header)
                trace.write(raw)
            else:
                assert trace.read(len(header)) == header and trace.read(len(raw)) == raw, "cold/warm bytes differ"
    if os.environ.get("GAUDIRT_TRACE_EVENTS") == "1":
        import ctypes

        probe = ctypes.CDLL(None)
        create = probe.gaudirt_test_event_create_count
        destroy = probe.gaudirt_test_event_destroy_count
        create.restype = destroy.restype = ctypes.c_uint64
        event_counts = lambda: (create(), destroy())

    def execute(*args, **kwargs):
        nonlocal previous_events, event_steps, previous_kv, audit_steps, capture_steps
        nonlocal completed_prefills, request_index
        before_events = event_counts() if event_counts else None
        if audit_mode:
            feature_writes.clear()
        if capture_values:
            feature_values.clear()
        if os.environ.get("GAUDIRT_TRACE_CONFIG_PATCH") == "1":
            import gaudirt.compiler as compiler

            patch_code = type(torch._dynamo.config).patch.__code__
            calls = [0, 0]
            configs = (torch._dynamo.config, torch.fx.experimental._config)

            def profile(frame, event, arg):
                if event == "call" and frame.f_code is patch_code:
                    caller = frame.f_back
                    if caller and caller.f_globals.get("__name__") == compiler.__name__ and caller.f_code.co_name == "execute":
                        for index, config in enumerate(configs):
                            if frame.f_locals.get("self") is config:
                                calls[index] += 1

            previous = sys.getprofile()
            sys.setprofile(profile)
            try:
                output = compiled(*args, **kwargs)
            finally:
                sys.setprofile(previous)
            print("FEATURE_CONFIG_PATCH_CALLS", calls, flush=True)
        else:
            output = compiled(*args, **kwargs)
        if os.environ.get("GAUDIRT_FEATURE_LOGITS_DIR") or os.environ.get("GAUDIRT_FEATURE_REPLAY_DIR"):
            from sglang.srt.model_executor.forward_batch_info import ForwardMode

            batch = args[2]
            if batch.forward_mode == ForwardMode.EXTEND:
                prefixes, lengths = batch.extend_prefix_lens_cpu, batch.extend_seq_lens_cpu
                assert len(prefixes) == len(lengths) == 1
                if prefixes[0] == 0:
                    request_index += 1
                end = prefixes[0] + lengths[0]
                prompt_length = int(os.environ.get("GAUDIRT_FEATURE_PREFIX", "160")) + int(
                    os.environ.get("GAUDIRT_FEATURE_SUFFIX", "9"))
                if end == prompt_length and completed_prefills < 2:
                    logits = output.next_token_logits.cpu().contiguous()
                    assert torch.isfinite(logits).all()
                    path = Path(os.environ["GAUDIRT_FEATURE_LOGITS_DIR"]) / f"request-{completed_prefills}.bin"
                    path.write_bytes(logits.view(torch.uint8).numpy().tobytes())
                    completed_prefills += 1
                if os.environ.get("GAUDIRT_FEATURE_REPLAY_DIR") and request_index == 3 and end == prompt_length:
                    logits = output.next_token_logits.cpu().contiguous()
                    assert torch.isfinite(logits).all()
                    torch.save(logits, Path(os.environ["GAUDIRT_FEATURE_REPLAY_DIR"]) / "chunk.pt")
            elif (os.environ.get("GAUDIRT_FEATURE_REPLAY_DIR") and
                  batch.forward_mode == ForwardMode.DECODE and request_index == 2):
                logits = output.next_token_logits.cpu().contiguous()
                assert torch.isfinite(logits).all()
                torch.save(logits, Path(os.environ["GAUDIRT_FEATURE_REPLAY_DIR"]) / "single.pt")
        if capture_values:
            capture_steps += 1
            path = Path(os.environ["GAUDIRT_FEATURE_CAPTURE_DIR"]) / f"step-{capture_steps}.pt"
            torch.save({"rows": args[0].numel(), "values": list(feature_values),
                        "logits": output.next_token_logits.cpu()}, path)
        if audit_mode:
            from sglang.srt.model_executor.forward_context import get_attn_backend

            pool = get_attn_backend().token_to_kv_pool
            locations = args[2].out_cache_loc.cpu().long()
            buffers = [tensor.cpu() for layer in range(pool.start_layer, pool.start_layer + pool.layer_num)
                       for tensor in (pool.get_key_buffer(layer), pool.get_value_buffer(layer))]
            assert len(feature_writes) == pool.layer_num
            previous_kv = check_kv_step(previous_kv, buffers, feature_writes, locations)
            logits = output.next_token_logits.cpu()
            assert torch.isfinite(logits).all()
            for value in (locations, logits, *buffers):
                emit(value)
            audit_steps += 1
            if audit_steps == expected_audit_steps:
                if mode == "warm":
                    assert not trace.read(1), "cold trace has extra records"
                trace.close()
                summary = {"mode": mode, "steps": audit_steps, "sha256": digest.hexdigest(),
                           "cache": gaudirt.recipe_cache_stats()}
                Path(os.environ["GAUDIRT_FEATURE_STATE_RECEIPT"]).write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n")
                print("FEATURE_STATE_AUDIT", json.dumps(summary), flush=True)
        if event_counts:
            after_events = event_counts()
            between = None if previous_events is None else [a - b for a, b in zip(before_events, previous_events)]
            forward = [a - b for a, b in zip(after_events, before_events)]
            if os.environ.get("GAUDIRT_EXPECT_STABLE_EVENTS") == "1" and event_steps >= 2:
                assert between == [0, 0] and forward == [0, 0], (between, forward)
            print("FEATURE_EVENT_DELTA", {"between_forwards": between,
                                          "forward": forward},
                  flush=True)
            previous_events = after_events
            event_steps += 1
        stats = gaudirt.recipe_cache_stats()
        if os.environ.get("GAUDIRT_EXPECT_WARM") == "1":
            assert stats[1] == 0, stats
        print("FEATURE_CACHE", list(stats), flush=True)
        return output

    return execute


GaudiRTSRTPlatform.prepare_model_forward = audit_prepare


def main():
    from sglang.srt.entrypoints.engine import Engine

    model = Path(os.environ["GAUDIRT_TARGET_MODEL"])
    config = json.loads((model / "config.json").read_text())
    assert config["num_hidden_layers"] == int(os.environ.get("GAUDIRT_EXPECT_LAYERS", "2"))
    enabled = os.environ["GAUDIRT_FEATURE_MODE"] == "enabled"
    assert enabled or os.environ["GAUDIRT_FEATURE_MODE"] == "plain"
    radix = os.environ.get("GAUDIRT_FEATURE_RADIX", "1" if enabled else "0")
    assert radix in ("0", "1")
    chunk_size = int(os.environ.get("GAUDIRT_FEATURE_CHUNK_SIZE", "64" if enabled else "-1"))
    page_size = int(os.environ.get("GAUDIRT_TEST_PAGE_SIZE", "1"))
    context = int(os.environ.get("GAUDIRT_FEATURE_CONTEXT", "1024"))
    vocabulary = config["vocab_size"]
    prompts = feature_prompts(vocabulary)
    logits_dir = Path(os.environ["GAUDIRT_FEATURE_LOGITS_DIR"])
    logits_dir.mkdir(parents=True, exist_ok=True)
    assert not list(logits_dir.glob("request-*.bin")), "logits directory must be empty"
    replay_dir = os.environ.get("GAUDIRT_FEATURE_REPLAY_DIR")
    if replay_dir:
        assert radix == "0" and chunk_size > 0, "same-prefix replay requires uncached chunked prefill"
        assert os.environ.get("GAUDIRT_FEATURE_STATE_AUDIT") != "1", "run state audit separately from replay"
        Path(replay_dir).mkdir(parents=True, exist_ok=True)
        assert not any((Path(replay_dir) / f"{name}.pt").exists() for name in ("single", "chunk"))
    contract = {"context_length": context, "page_size": page_size, "chunked_prefill_size": chunk_size,
                "prompt_lengths": [len(prompt) for prompt in prompts], "vocabulary": vocabulary}
    engine = Engine(
        model_path=str(model), skip_tokenizer_init=True, device="gaudirt", dtype="bfloat16",
        tp_size=1, max_total_tokens=int(os.environ.get("GAUDIRT_FEATURE_MAX_TOKENS", "1024")),
        max_running_requests=1, context_length=context,
        mem_fraction_static=0.9, model_loader_extra_config='{"enable_multithread_load":false}',
        page_size=page_size, disable_cuda_graph=True, disable_radix_cache=radix == "0",
        disable_overlap_schedule=True,
        chunked_prefill_size=chunk_size,
        log_level="warning",
    )
    try:
        results = [engine.generate(input_ids=prompt, sampling_params={
            "temperature": 0, "max_new_tokens": 4, "ignore_eos": True,
        }) for prompt in prompts]
        output = [{"tokens": result["output_ids"],
                   "cached_tokens": result["meta_info"]["cached_tokens"]} for result in results]
        logits = [(logits_dir / f"request-{index}.bin").read_bytes() for index in range(len(output))]
        assert all(logits), "missing first-step logits"
        receipt = {"contract": contract, "radix": radix == "1", "requests": output,
                   "first_logits_sha256": [hashlib.sha256(value).hexdigest() for value in logits]}
        Path(os.environ["GAUDIRT_FEATURE_RECEIPT"]).write_text(json.dumps(receipt, indent=2) + "\n")
        print("FEATURE_OUTPUT", json.dumps(output), flush=True)
        assert all(len(item["tokens"]) == 4 for item in output), output
        if radix == "1":
            assert output[1]["cached_tokens"] >= int(os.environ.get("GAUDIRT_FEATURE_MIN_CACHED", "128")), output
        else:
            assert output[1]["cached_tokens"] == 0, output
        if reference := os.environ.get("GAUDIRT_FEATURE_REFERENCE"):
            expected = json.loads(Path(reference).read_text())
            assert expected["contract"] == contract, "bit-exact comparison requires the same context, page and chunking"
            assert [item["tokens"] for item in output] == [item["tokens"] for item in expected["requests"]], output
            reference_logits = Path(os.environ["GAUDIRT_FEATURE_LOGITS_REFERENCE"])
            assert all(value == (reference_logits / f"request-{index}.bin").read_bytes()
                       for index, value in enumerate(logits)), "first-step logits differ"
        if replay_dir:
            replay_length = len(prompts[0]) - 1
            replay_prefix = [3 + (i * 53 + 701) % (vocabulary - 3) for i in range(replay_length)]
            first = engine.generate(input_ids=replay_prefix, sampling_params={
                "temperature": 0, "max_new_tokens": 2, "ignore_eos": True,
            })
            second = engine.generate(input_ids=replay_prefix + first["output_ids"][:1], sampling_params={
                "temperature": 0, "max_new_tokens": 1, "ignore_eos": True,
            })
            single = torch.load(Path(replay_dir) / "single.pt", weights_only=True)
            chunk = torch.load(Path(replay_dir) / "chunk.pt", weights_only=True)
            assert single.shape == chunk.shape and single.dtype == chunk.dtype
            maximum = (single.double() - chunk.double()).abs().max().item()
            exact = torch.equal(single.view(torch.uint8), chunk.view(torch.uint8))
            print("FEATURE_SAME_PREFIX_DIAGNOSTIC", json.dumps({"max_abs": maximum, "bit_exact": exact}),
                  flush=True)
            if os.environ.get("GAUDIRT_FEATURE_REPLAY_REQUIRE_EXACT") == "1":
                assert exact, ("same-path logits differ", maximum)
                assert first["output_ids"][1:] == second["output_ids"]
        if os.environ.get("GAUDIRT_FEATURE_STATE_AUDIT") == "1":
            summary = json.loads(Path(os.environ["GAUDIRT_FEATURE_STATE_RECEIPT"]).read_text())
            assert summary["steps"] == int(os.environ.get("GAUDIRT_FEATURE_EXPECT_STEPS", "10"))
            if os.environ["GAUDIRT_AUDIT_MODE"] == "warm":
                assert summary["cache"][1] == 0, summary
    finally:
        workers = list(zip(engine.tokenizer_manager._subprocess_watchdog._names,
                           engine.tokenizer_manager._subprocess_watchdog._processes))
        for name, process in workers:
            with open(f"/proc/{process.pid}/maps", encoding="utf-8") as maps:
                mapped = maps.read().lower()
            assert not any(x in mapped for x in ("libhabana_pytorch", "gaudi-pytorch-bridge", "habana_frameworks"))
        engine.shutdown()
        for name, process in workers:
            assert process.exitcode == 0, (name, process.exitcode)
            print("WORKER_EXIT", name, process.exitcode, flush=True)


if __name__ == "__main__":
    main()
