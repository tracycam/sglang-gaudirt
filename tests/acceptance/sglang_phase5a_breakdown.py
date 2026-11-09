"""Separate instrumented host and compute-stream timing from service benchmarks."""

import json
import os
import sys
import time
from pathlib import Path

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import gaudirt
import gaudirt.compiler as compiler
import sglang
from sglang_gaudirt.platform import GaudiRTSRTPlatform
from sglang_gaudirt.attention import GaudiRTPagedAttnBackend

layers = int(os.environ["GAUDIRT_EXPECT_LAYERS"])
warm_output = int(os.environ.get("GAUDIRT_PROFILE_WARM_OUTPUT", "128"))
sample_steps = {warm_output + 1, warm_output + 2, 2 * warm_output + 3, 2 * warm_output + 4}
state = {"active": False, "segments": [], "events": [], "calls": [], "step": 0, "metadata": None,
         "whole_events": None, "metadata_events": None}
original_backend = compiler.gaudirt_backend
original_prepare = GaudiRTSRTPlatform.prepare_model_forward
original_metadata = GaudiRTPagedAttnBackend.init_forward_metadata


def profiling_metadata(self, batch):
    if state["step"] + 1 not in sample_steps:
        return original_metadata(self, batch)
    assert state["metadata_events"] is not None
    start, end = state["metadata_events"]
    start.record()
    before = time.perf_counter_ns()
    result = original_metadata(self, batch)
    after = time.perf_counter_ns()
    end.record()
    state["metadata"] = (start, end, (after - before) / 1e6)
    return result


def profiling_backend(graph, inputs):
    compiled = original_backend(graph, inputs)
    segments = list(compiled.__self__.named_children())
    assert len(segments) == 2 * layers + 1 and not state["segments"]
    state["segments"] = segments
    for index, (_, segment) in enumerate(segments):
        previous = segment.forward

        def timed(*args, previous=previous, index=index):
            if not state["active"]:
                return previous(*args)
            start, end = state["events"][index]
            before = time.perf_counter_ns()
            start.record()
            launched = time.perf_counter_ns()
            output = previous(*args)
            returned = time.perf_counter_ns()
            end.record()
            recorded = time.perf_counter_ns()
            state["calls"].append((index, (returned - launched) / 1e6,
                                   (launched - before + recorded - returned) / 1e6))
            return output

        segment.forward = timed
    return compiled


def profiling_prepare(self, forward):
    compiled = original_prepare(self, forward)

    def execute(*args, **kwargs):
        state["step"] += 1
        if state["step"] not in sample_steps:
            output = compiled(*args, **kwargs)
            if state["step"] == 1:
                state["events"] = [(gaudirt.Event(enable_timing=True), gaudirt.Event(enable_timing=True))
                                   for _ in state["segments"]]
                state["whole_events"] = (gaudirt.Event(enable_timing=True), gaudirt.Event(enable_timing=True))
                state["metadata_events"] = (gaudirt.Event(enable_timing=True), gaudirt.Event(enable_timing=True))
                for start, end in [*state["events"], state["whole_events"], state["metadata_events"]]:
                    start.record()
                    end.record()
                for start, end in [*state["events"], state["whole_events"], state["metadata_events"]]:
                    start.synchronize()
                    end.synchronize()
            return output
        if not state["events"]:
            raise RuntimeError("profiling events were not prepared during warmup")
        whole_start, whole_end = state["whole_events"]
        cache_before = gaudirt.recipe_cache_stats()
        state["calls"] = []
        state["active"] = True
        whole_start.record()
        before = time.perf_counter_ns()
        try:
            output = compiled(*args, **kwargs)
        finally:
            state["active"] = False
        submitted = time.perf_counter_ns()
        whole_end.record()
        whole_end.synchronize()
        calls = sorted(state["calls"])
        assert len(calls) == len(state["segments"]) and [item[0] for item in calls] == list(range(len(calls)))
        records = []
        for index, host_ms, event_host_ms in calls:
            start, end = state["events"][index]
            start.synchronize()
            end.synchronize()
            records.append({"segment": index, "kind": "attention" if index % 2 else "fx",
                            "host_call_ms": host_ms, "event_record_host_ms": event_host_ms,
                            "compute_stream_ms": start.elapsed_time(end)})
        whole_start.synchronize()
        assert state["metadata"] is not None
        metadata_start, metadata_end, metadata_host_ms = state["metadata"]
        metadata_start.synchronize()
        metadata_end.synchronize()
        cache_after = gaudirt.recipe_cache_stats()
        assert cache_after[1] == cache_before[1], (cache_before, cache_after)
        summary = {"step": state["step"], "rows": args[0].numel(),
                   "host_submit_ms": (submitted - before) / 1e6,
                   "compute_stream_ms": whole_start.elapsed_time(whole_end), "segments": records,
                   "metadata_host_ms": metadata_host_ms,
                   "metadata_compute_stream_ms": metadata_start.elapsed_time(metadata_end),
                   "cache_before": cache_before, "cache_after": cache_after}
        state["metadata"] = None
        print("MODEL_BREAKDOWN", json.dumps(summary, sort_keys=True), flush=True)
        return output

    return execute


compiler.gaudirt_backend = profiling_backend
GaudiRTSRTPlatform.prepare_model_forward = profiling_prepare
GaudiRTPagedAttnBackend.init_forward_metadata = profiling_metadata


def main():
    from sglang.srt.entrypoints.engine import Engine

    model = Path(os.environ["GAUDIRT_TARGET_MODEL"])
    config = json.loads((model / "config.json").read_text())
    assert config["num_hidden_layers"] == layers
    lengths = tuple(int(value) for value in os.environ.get("GAUDIRT_PROFILE_PROMPTS", "1024,4096").split(","))
    assert len(lengths) == 2 and warm_output >= 2
    if os.environ.get("GAUDIRT_TEXT_MODE") == "1":
        from transformers import AutoTokenizer
        from phase5a_prompt import prompt_builder

        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
        documents = [Path(path) for path in os.environ["GAUDIRT_PROMPT_DOCUMENTS"].split(os.pathsep)]
        ids = prompt_builder(tokenizer, documents)
    else:
        ids = lambda length: [3 + (i * 37) % (config["vocab_size"] - 3) for i in range(length)]
    engine = Engine(model_path=str(model), skip_tokenizer_init=True, device="gaudirt", dtype="bfloat16",
                    tp_size=1, max_total_tokens=int(os.environ.get("GAUDIRT_MAX_TOTAL_TOKENS", "4608")),
                    max_running_requests=1, context_length=int(os.environ.get("GAUDIRT_CONTEXT_LENGTH", "4608")),
                    mem_fraction_static=float(os.environ.get("GAUDIRT_STATIC_FRACTION", "0.9")),
                    disable_cuda_graph=True, disable_radix_cache=True, disable_overlap_schedule=True,
                    chunked_prefill_size=-1, stream_interval=1, log_level="warning")
    requests = []
    try:
        for length in lengths:
            prompt = ids(length)
            params = {"temperature": 0, "max_new_tokens": warm_output, "ignore_eos": True}
            assert len(engine.generate(input_ids=prompt, sampling_params=params)["output_ids"]) == warm_output
            before = time.perf_counter_ns()
            arrival = []
            result = []
            first_chunk_tokens = None
            for item in engine.generate(input_ids=prompt, sampling_params={
                    "temperature": 0, "max_new_tokens": 2, "ignore_eos": True}, stream=True):
                if len(item["output_ids"]) > len(result):
                    result = list(item["output_ids"])
                    if first_chunk_tokens is None:
                        first_chunk_tokens = len(result)
                    arrival.append(time.perf_counter_ns())
            assert len(result) == 2 and arrival
            split = first_chunk_tokens == 1 and len(arrival) == 2
            requests.append({"prompt_tokens": length, "total_ms": (arrival[-1] - before) / 1e6,
                             "first_chunk_tokens": first_chunk_tokens,
                             "first_token_ms": (arrival[0] - before) / 1e6 if split else None,
                             "next_token_ms": (arrival[1] - arrival[0]) / 1e6 if split else None})
        Path(os.environ["GAUDIRT_PROFILE_REQUESTS"]).write_text(json.dumps(requests, indent=2) + "\n")
        print("REQUEST_BREAKDOWN", json.dumps(requests), flush=True)
    finally:
        workers = list(zip(engine.tokenizer_manager._subprocess_watchdog._names,
                           engine.tokenizer_manager._subprocess_watchdog._processes))
        engine.shutdown()
        for name, process in workers:
            assert process.exitcode == 0, (name, process.exitcode)
            print("WORKER_EXIT", name, process.exitcode, flush=True)


if __name__ == "__main__":
    main()
