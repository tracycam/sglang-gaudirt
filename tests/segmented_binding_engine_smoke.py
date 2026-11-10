"""Check bound execution against a saved real-weight two-layer token trace."""

import json
import os
import sys
from pathlib import Path

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import gaudirt.compiler as compiler

original_backend = compiler.gaudirt_backend


def observed_backend(graph, inputs):
    result = original_backend(graph, inputs)
    execution = getattr(result, "__self__", None)
    if execution is None or not isinstance(execution, compiler._SegmentExecution):
        return result

    class ObservedRunner:
        def __init__(self, runner):
            self.runner = runner
            self.printed = False

        def bind_static(self, slots, tensors):
            print("STATIC_BOUND", len(slots), flush=True)
            return self.runner.bind_static(slots, tensors)

        def run(self, values):
            return self.runner.run(values)

        def run_bound(self, values):
            if not self.printed:
                print("DYNAMIC_INPUTS", len(values), flush=True)
                self.printed = True
            return self.runner.run_bound(values)

    execution.runner = ObservedRunner(execution.runner)
    return result


compiler.gaudirt_backend = observed_backend


def main():
    from sglang.srt.entrypoints.engine import Engine

    path = Path(os.environ["GAUDIRT_TARGET_MODEL"])
    config = json.loads((path / "config.json").read_text())
    assert config["num_hidden_layers"] == 2
    reference = json.loads(Path(os.environ["GAUDIRT_REFERENCE_TOKENS"]).read_text())
    vocabulary = config["vocab_size"]

    def ids(length):
        return [3 + (i * 37) % (vocabulary - 3) for i in range(length)]

    engine = Engine(model_path=str(path), skip_tokenizer_init=True, device="gaudirt", dtype="bfloat16",
                    tp_size=1, max_total_tokens=4608, max_running_requests=1, context_length=4608,
                    disable_cuda_graph=True, disable_radix_cache=True, disable_overlap_schedule=True,
                    chunked_prefill_size=-1, log_level="warning")
    try:
        first = engine.generate(input_ids=ids(136), sampling_params={
            "temperature": 0, "max_new_tokens": 2, "ignore_eos": True})
        second = engine.generate(input_ids=ids(136) + first["output_ids"][:1], sampling_params={
            "temperature": 0, "max_new_tokens": 1, "ignore_eos": True})
        assert first["output_ids"] == reference[0]["outputs"]
        assert second["output_ids"] == reference[0]["replay_outputs"]
        for item in reference[1:]:
            result = engine.generate(input_ids=ids(item["prompt_tokens"]), sampling_params={
                "temperature": 0, "max_new_tokens": len(item["output_ids"]), "ignore_eos": True})
            assert result["output_ids"] == item["output_ids"]
        print("BOUND_ENGINE_TOKENS_MATCH", flush=True)
    finally:
        watchdog = engine.tokenizer_manager._subprocess_watchdog
        workers = list(zip(watchdog._names, watchdog._processes))
        engine.shutdown()
        for name, process in workers:
            assert process.exitcode == 0, (name, process.exitcode)
            print("WORKER_EXIT", name, process.exitcode, flush=True)


if __name__ == "__main__":
    main()
