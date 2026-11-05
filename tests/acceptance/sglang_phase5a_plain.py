"""Replay audit-off SGLang generation and measure uninstrumented streaming."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
from sglang.srt.managers.scheduler import Scheduler

original_state = Scheduler.get_internal_state


def state_with_allocator(self, request):
    result = original_state(self, request)
    result.internal_state["gaudirt_measurement"] = {
        "peak_allocated_bytes": gaudirt.max_memory_allocated(),
        "allocated_bytes": gaudirt.memory_allocated(),
        "recipe_cache": gaudirt.recipe_cache_stats(),
    }
    return result


Scheduler.get_internal_state = state_with_allocator


def measurement(engine):
    state = engine.get_server_info()

    def find(value):
        if isinstance(value, dict):
            if "gaudirt_measurement" in value:
                return value["gaudirt_measurement"]
            for item in value.values():
                found = find(item)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = find(item)
                if found is not None:
                    return found
        return None

    result = find(state)
    assert result is not None, "scheduler allocator measurement unavailable"
    command = ["hl-smi", "-Q", "index,module_id,memory.total,memory.used,memory.free",
               "-f", "csv,noheader,nounits"]
    output = subprocess.check_output(command, text=True)
    devices = {int(parts[1]): {"index": int(parts[0]), "module_id": int(parts[1]),
                               "total_mib": int(parts[2]), "reported_used_mib": int(parts[3]),
                               "reported_free_mib": int(parts[4])}
               for line in output.splitlines()
               if len(parts := [part.strip() for part in line.split(",")]) == 5}
    result["hl_smi"] = devices[int(os.environ.get("GAUDIRT_MODULE_ID", "0"))]
    result["pool_reserved_bytes"] = int(os.environ.get("GAUDIRT_HBM_BYTES", str(80 * 1024**3)))
    return result


def main():
    from sglang.srt.entrypoints.engine import Engine

    model = Path(os.environ["GAUDIRT_TARGET_MODEL"])
    config = json.loads((model / "config.json").read_text())
    assert config["num_hidden_layers"] == int(os.environ["GAUDIRT_EXPECT_LAYERS"])
    vocabulary = config["vocab_size"]
    prompt_lengths = tuple(int(x) for x in os.environ.get("GAUDIRT_PROMPTS", "137,1053,4096").split(","))
    bench_lengths = tuple(int(x) for x in os.environ.get("GAUDIRT_BENCH_PROMPTS", "1024,4096").split(","))
    decode_steps = int(os.environ.get("GAUDIRT_DECODE_STEPS", "128"))
    bench_output = int(os.environ.get("GAUDIRT_BENCH_OUTPUT", "128"))
    tokenizer = None
    if os.environ.get("GAUDIRT_TEXT_MODE") == "1":
        from transformers import AutoTokenizer
        from phase5a_prompt import prompt_builder
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
        documents = [Path(path) for path in os.environ["GAUDIRT_PROMPT_DOCUMENTS"].split(os.pathsep)]
        ids = prompt_builder(tokenizer, documents)
    else:
        def ids(length):
            return [3 + (i * 37) % (vocabulary - 3) for i in range(length)]

    reference = json.loads(Path(os.environ["GAUDIRT_REFERENCE_TOKENS"]).read_text())
    engine = Engine(model_path=str(model), skip_tokenizer_init=True, device="gaudirt", dtype="bfloat16",
                    tp_size=1, max_total_tokens=int(os.environ.get("GAUDIRT_MAX_TOTAL_TOKENS", "4608")),
                    max_running_requests=1, context_length=int(os.environ.get("GAUDIRT_CONTEXT_LENGTH", "4608")),
                    mem_fraction_static=float(os.environ.get("GAUDIRT_STATIC_FRACTION", "0.9")),
                    disable_cuda_graph=True, disable_radix_cache=True, disable_overlap_schedule=True,
                    chunked_prefill_size=-1, stream_interval=1, log_level="warning")
    try:
        prefix = ids(136)
        first = engine.generate(input_ids=prefix, sampling_params={
            "temperature": 0, "max_new_tokens": 2, "ignore_eos": True,
        })
        second = engine.generate(input_ids=prefix + first["output_ids"][:1], sampling_params={
            "temperature": 0, "max_new_tokens": 1, "ignore_eos": True,
        })
        assert first["output_ids"] == reference[0]["outputs"]
        assert second["output_ids"] == reference[0]["replay_outputs"]
        for index, length in enumerate(prompt_lengths):
            result = engine.generate(input_ids=ids(length), sampling_params={
                "temperature": 0, "max_new_tokens": decode_steps + 1, "ignore_eos": True,
            })
            assert result["output_ids"] == reference[index + 1]["output_ids"], length
            print("AUDIT_OFF_MATCH", length, len(result["output_ids"]), flush=True)
        warm = measurement(engine)
        assert warm["recipe_cache"][1] == 0, warm
        timings = []
        for length in bench_lengths:
            prompt = ids(length)
            warmup_stream = engine.generate(input_ids=prompt, sampling_params={
                "temperature": 0, "max_new_tokens": bench_output, "ignore_eos": True,
            }, stream=True)
            warmup = list(warmup_stream)
            assert len(warmup[-1]["output_ids"]) == bench_output
            before_timed = measurement(engine)
            start = time.perf_counter()
            first_at = last_at = None
            output_ids = []
            for chunk in engine.generate(input_ids=prompt, sampling_params={
                "temperature": 0, "max_new_tokens": bench_output, "ignore_eos": True,
            }, stream=True):
                now = time.perf_counter()
                current = chunk["output_ids"]
                if len(current) > len(output_ids):
                    if first_at is None:
                        assert len(current) == 1, "first streaming response contains multiple tokens"
                        first_at = now
                    last_at = now
                    output_ids = list(current)
            assert len(output_ids) == bench_output and first_at is not None and last_at is not None
            after_timed = measurement(engine)
            assert after_timed["recipe_cache"][1] == before_timed["recipe_cache"][1], (
                before_timed["recipe_cache"], after_timed["recipe_cache"]
            )
            record = {"prompt_tokens": length, "output_tokens": len(output_ids),
                      "ttft_ms": (first_at - start) * 1000,
                      "tpot_ms": (last_at - first_at) * 1000 / (len(output_ids) - 1),
                      "allocator_and_device": after_timed}
            timings.append(record)
            print("UNINSTRUMENTED_TIMING", json.dumps(record, sort_keys=True), flush=True)
        result = {"audit_off_matches": True, "warm_before_bench": warm, "timings": timings}
        Path(os.environ["GAUDIRT_PLAIN_SUMMARY"]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    finally:
        watchdog = engine.tokenizer_manager._subprocess_watchdog
        workers = list(zip(watchdog._names, watchdog._processes))
        engine.shutdown()
        for name, process in workers:
            assert process.exitcode == 0, (name, process.exitcode)
            print("WORKER_EXIT", name, process.exitcode, flush=True)


if __name__ == "__main__":
    main()
