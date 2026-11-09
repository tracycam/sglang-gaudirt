"""Compare ordinary SGLang inference with chunking and prefix caching enabled."""

import hashlib
import json
import os
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
if os.environ.get("GAUDIRT_FEATURE_STATE_AUDIT") == "1":
    import gaudirt.compiler as compiler

    original_backend = compiler.gaudirt_backend

    def audit_backend(graph, inputs):
        compiled = original_backend(graph, inputs)
        segments = list(compiled.__self__.named_children())
        for index, (_, segment) in enumerate(segments):
            if index % 2 != 1:
                continue
            nodes = tuple(segment.graph.nodes)
            placeholders = [node for node in nodes if node.op == "placeholder"]
            attention = [node for node in nodes if node.op == "call_function" and
                         node.target == torch.ops.gaudirt._paged_attn_validated.default]
            assert len(attention) == 1
            key_index, value_index = [placeholders.index(node) for node in attention[0].args[1:3]]
            previous = segment.forward

            def paged(*args, previous=previous, key_index=key_index, value_index=value_index):
                output = previous(*args)
                feature_writes.append((args[key_index].cpu(), args[value_index].cpu()))
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
        nonlocal previous_events, event_steps, previous_kv, audit_steps
        before_events = event_counts() if event_counts else None
        if audit_mode:
            feature_writes.clear()
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
    page_size = int(os.environ.get("GAUDIRT_TEST_PAGE_SIZE", "1"))
    vocabulary = config["vocab_size"]
    prompts = feature_prompts(vocabulary)
    engine = Engine(
        model_path=str(model), skip_tokenizer_init=True, device="gaudirt", dtype="bfloat16",
        tp_size=1, max_total_tokens=int(os.environ.get("GAUDIRT_FEATURE_MAX_TOKENS", "1024")),
        max_running_requests=1, context_length=int(os.environ.get("GAUDIRT_FEATURE_CONTEXT", "1024")),
        mem_fraction_static=0.9, model_loader_extra_config='{"enable_multithread_load":false}',
        page_size=page_size, disable_cuda_graph=True, disable_radix_cache=not enabled,
        disable_overlap_schedule=True,
        chunked_prefill_size=int(os.environ.get("GAUDIRT_FEATURE_CHUNK_SIZE", "64")) if enabled else -1,
        log_level="warning",
    )
    try:
        results = [engine.generate(input_ids=prompt, sampling_params={
            "temperature": 0, "max_new_tokens": 4, "ignore_eos": True,
        }) for prompt in prompts]
        output = [{"tokens": result["output_ids"],
                   "cached_tokens": result["meta_info"]["cached_tokens"]} for result in results]
        assert all(len(item["tokens"]) == 4 for item in output), output
        if enabled:
            assert output[1]["cached_tokens"] >= int(os.environ.get("GAUDIRT_FEATURE_MIN_CACHED", "128")), output
        else:
            assert output[1]["cached_tokens"] == 0, output
        if reference := os.environ.get("GAUDIRT_FEATURE_REFERENCE"):
            expected = json.loads(Path(reference).read_text())
            assert [item["tokens"] for item in output] == [item["tokens"] for item in expected], output
        if os.environ.get("GAUDIRT_FEATURE_STATE_AUDIT") == "1":
            summary = json.loads(Path(os.environ["GAUDIRT_FEATURE_STATE_RECEIPT"]).read_text())
            assert summary["steps"] == int(os.environ.get("GAUDIRT_FEATURE_EXPECT_STEPS", "10"))
            if os.environ["GAUDIRT_AUDIT_MODE"] == "warm":
                assert summary["cache"][1] == 0, summary
        Path(os.environ["GAUDIRT_FEATURE_RECEIPT"]).write_text(json.dumps(output, indent=2) + "\n")
        print("FEATURE_OUTPUT", json.dumps(output), flush=True)
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
