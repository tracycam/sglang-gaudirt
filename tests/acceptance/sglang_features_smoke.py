"""Compare ordinary SGLang inference with chunking and prefix caching enabled."""

import json
import os
import sys
from pathlib import Path

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import gaudirt
import sglang
from sglang_gaudirt.platform import GaudiRTSRTPlatform

original_prepare = GaudiRTSRTPlatform.prepare_model_forward


def audit_prepare(self, forward):
    compiled = original_prepare(self, forward)

    def execute(*args, **kwargs):
        if os.environ.get("GAUDIRT_TRACE_CONFIG_PATCH") == "1":
            import torch
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
    assert config["num_hidden_layers"] == 2
    enabled = os.environ["GAUDIRT_FEATURE_MODE"] == "enabled"
    assert enabled or os.environ["GAUDIRT_FEATURE_MODE"] == "plain"
    page_size = int(os.environ.get("GAUDIRT_TEST_PAGE_SIZE", "1"))
    vocabulary = config["vocab_size"]
    prefix = [3 + (i * 37) % (vocabulary - 3) for i in range(160)]
    prompts = [prefix + [3 + (j * 11 + i) % (vocabulary - 3) for i in range(9)]
               for j in (1, 2)]
    engine = Engine(
        model_path=str(model), skip_tokenizer_init=True, device="gaudirt", dtype="bfloat16",
        tp_size=1, max_total_tokens=1024, max_running_requests=1, context_length=1024,
        mem_fraction_static=0.9, model_loader_extra_config='{"enable_multithread_load":false}',
        page_size=page_size, disable_cuda_graph=True, disable_radix_cache=not enabled,
        disable_overlap_schedule=True, chunked_prefill_size=64 if enabled else -1,
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
            assert output[1]["cached_tokens"] >= 128, output
        else:
            assert output[1]["cached_tokens"] == 0, output
        if reference := os.environ.get("GAUDIRT_FEATURE_REFERENCE"):
            expected = json.loads(Path(reference).read_text())
            assert [item["tokens"] for item in output] == [item["tokens"] for item in expected], output
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
