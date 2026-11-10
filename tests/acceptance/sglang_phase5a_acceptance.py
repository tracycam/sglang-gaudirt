"""SGLang state and fixed-prefix numerical audit for one GaudiRT device.

The cold trace contains logits, physical write locations, the written KV values,
and full KV snapshots after each request. The warm run compares every byte.
Each forward also checks that no other physical KV slot changed.
"""

import copy
import hashlib
import json
import operator
import os
import struct
import sys
from pathlib import Path

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import torch
import sglang
import gaudirt
import gaudirt.compiler as compiler
from phase5a_state_audit import check_kv_step
from sglang_gaudirt.platform import GaudiRTSRTPlatform

layers = int(os.environ["GAUDIRT_EXPECT_LAYERS"])
prompt_lengths = tuple(int(x) for x in os.environ.get("GAUDIRT_PROMPTS", "137,1053,4096").split(",") if x)
decode_steps = int(os.environ.get("GAUDIRT_DECODE_STEPS", "128"))
prefix_length = int(os.environ.get("GAUDIRT_FIXED_PREFIX_LENGTH", "136"))
chunk_size = int(os.environ.get("GAUDIRT_AUDIT_CHUNK_SIZE", "-1"))
cross_tree = os.environ.get("GAUDIRT_AUDIT_CROSS_TREE") == "1"
assert prefix_length > 0 and (chunk_size == -1 or chunk_size > 0) and decode_steps >= 0
if cross_tree:
    assert chunk_size > 0 and os.environ.get("GAUDIRT_REPLAY_LAYER_CAPTURE_PATH"), (
        "cross-tree validation requires chunking and a separate replay capture")


def chunks(length):
    return 1 if chunk_size == -1 else (length + chunk_size - 1) // chunk_size


prefix_end = chunks(prefix_length)
single_step = prefix_end + 1
replay_end = single_step + chunks(prefix_length + 1)
expected_steps = replay_end
snapshot_steps = {replay_end}
for length in prompt_lengths:
    expected_steps += chunks(length) + decode_steps
    snapshot_steps.add(expected_steps)
state = {"active": False, "complete": False, "inputs": [], "attention": [], "intermediates": [],
         "replay_active": False, "replay_inputs": [], "replay_attention": [],
         "writes": [], "compile_deltas": [], "track_compile": False, "rows": 0, "bound_traced_steps": 0}
original_backend = compiler.gaudirt_backend
original_prepare = GaudiRTSRTPlatform.prepare_model_forward
captures = 0


def audit_backend(graph, inputs):
    global captures
    captures += 1
    assert captures == 1, "model forward was captured more than once"
    compiled = original_backend(graph, inputs)

    class AuditRunner:
        def __init__(self, runner):
            self.runner = runner

        def __getattr__(self, name):
            return getattr(self.runner, name)

        def run_bound_traced(self, values, indices):
            state["bound_traced_steps"] += 1
            return self.runner.run_bound_traced(values, indices)

    compiled.__self__.runner = AuditRunner(compiled.__self__.runner)
    compiled.__self__.enable_trace()
    segments = list(compiled.__self__.named_children())
    assert len(segments) == 2 * layers + 1, len(segments)
    for index, (_, segment) in enumerate(segments):
        previous = segment.forward
        if index % 2:
            nodes = tuple(segment.graph.nodes)
            placeholders = [node for node in nodes if node.op == "placeholder"]
            attention = [node for node in nodes if node.op == "call_function" and
                         node.target == torch.ops.gaudirt._paged_attn_validated.default]
            assert len(attention) == 1
            indexes = [placeholders.index(node) for node in attention[0].args[:10]]

            def paged(*args, previous=previous, indexes=indexes):
                output = previous(*args)
                query, key, value, key_pool, value_pool, locations, requests, positions, _, _ = (
                    args[i] for i in indexes
                )
                key_cpu, value_cpu = key.cpu(), value.cpu()
                state["writes"].append((key_cpu, value_cpu))
                if state["active"] or state["replay_active"]:
                    if state["active"]:
                        assert query.shape[0] == 1
                    position, request, location = (int(tensor.cpu()[-1])
                                                    for tensor in (positions, requests, locations))
                    from sglang.srt.model_executor.forward_context import get_attn_backend
                    mapping = get_attn_backend().req_to_token_pool.req_to_token
                    slots = mapping.cpu()[request, :position + 1].long()
                    assert int(slots[-1]) == location
                    keys, values = key_pool.cpu()[slots].clone(), value_pool.cpu()[slots].clone()
                    assert torch.equal(keys[-1].reshape(1, -1).view(torch.int16), key_cpu[-1:].view(torch.int16))
                    assert torch.equal(values[-1].reshape(1, -1).view(torch.int16), value_cpu[-1:].view(torch.int16))
                    attention_capture = {"position": position, "query": query[-1:].cpu(), "key": key_cpu[-1:],
                                         "value": value_cpu[-1:], "keys": keys, "values": values,
                                         "output": output[-1:].cpu()}
                    target = "attention" if state["active"] else "replay_attention"
                    state[target].append(attention_capture)
                return output

            segment.forward = paged
        elif index < len(segments) - 1:
            def layer_input(*args, previous=previous, index=index):
                output = previous(*args)
                if state["track_compile"]:
                    delta = compiled.__self__.trace_compile_delta(index)
                    state["compile_deltas"].append({"rows": state["rows"], "segment": index, "compiled": delta})
                    if index >= 4:
                        assert delta == 0, (state["rows"], index, delta)
                if state["active"]:
                    state["inputs"].append(output[3].cpu())
                if state["replay_active"]:
                    state["replay_inputs"].append(output[3][-1:].cpu())
                return output

            segment.forward = layer_input
        else:
            audit_segment = copy.deepcopy(segment)
            norms = [node for node in audit_segment.graph.nodes if node.op == "call_function" and
                     node.target == torch.ops.gaudirt.fused_add_rms_norm.default]
            assert norms
            with audit_segment.graph.inserting_after(norms[-1]):
                last_residual = audit_segment.graph.call_function(operator.getitem, (norms[-1], 1))
            result = next(node for node in audit_segment.graph.nodes if node.op == "output")
            result.args = ((result.args[0], last_residual),)
            audit_segment.graph.lint()
            audit_compiled = []

            def last_layer(*args, previous=previous, audit_segment=audit_segment, audit_compiled=audit_compiled):
                output = previous(*args)
                if state["active"] or state["replay_active"]:
                    if not audit_compiled:
                        audit_compiled.append(compiler._program_backend(audit_segment, args))
                    audited, residual = audit_compiled[0](*args)
                    assert torch.equal(output.cpu().view(torch.uint8), audited.cpu().view(torch.uint8))
                    if state["active"]:
                        state["inputs"].append(residual.cpu())
                    else:
                        state["replay_inputs"].append(residual[-1:].cpu())
                return output

            segment.forward = last_layer
    return compiled


def audit_prepare(self, forward):
    # Set this before Dynamo tracing: its guards include the CPU thread count.
    # These threads only affect the full-buffer checks in this audit process.
    torch.set_num_threads(int(os.environ.get("GAUDIRT_AUDIT_CPU_THREADS", "8")))
    compiled = original_prepare(self, forward)
    mode = os.environ["GAUDIRT_AUDIT_MODE"]
    assert mode in ("cold", "warm")
    trace = open(os.environ["GAUDIRT_TRACE_PATH"], "wb" if mode == "cold" else "rb")
    digest = hashlib.sha256()
    previous = fixed_prefix = single_logits = None
    seen_rows = set()
    step = 0

    def emit(raw):
        header = struct.pack("<Q", len(raw))
        digest.update(header)
        digest.update(raw)
        if mode == "cold":
            trace.write(header)
            trace.write(raw)
        else:
            assert trace.read(len(header)) == header, "cold/warm trace record length differs"
            assert trace.read(len(raw)) == raw, "cold/warm trace bytes differ"

    def bytes_of(tensor):
        return tensor.contiguous().view(torch.uint8).numpy().tobytes()

    def execute(*args, **kwargs):
        nonlocal previous, fixed_prefix, single_logits, step
        rows = args[0].numel()
        state.update(active=not state["complete"] and step + 1 == single_step,
                     replay_active=step + 1 == replay_end, writes=[], rows=rows,
                     track_compile=rows not in seen_rows)
        seen_rows.add(rows)
        output = compiled(*args, **kwargs)
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        pool = backend.token_to_kv_pool
        assert pool.layer_num == layers and len(state["writes"]) == layers
        locations = args[2].out_cache_loc.cpu().long()
        assert locations.ndim == 1 and locations.numel() == rows
        buffers = [tensor.cpu() for layer in range(pool.start_layer, pool.start_layer + pool.layer_num)
                   for tensor in (pool.get_key_buffer(layer), pool.get_value_buffer(layer))]
        logits = output.next_token_logits.cpu()
        assert torch.isfinite(logits).all()
        assert (locations >= 0).all() and (locations < buffers[0].shape[0]).all()
        # Starting from zeros, update only the expected slots and compare the
        # entire pool. By induction this checks targets, history and all
        # untouched bytes at every step, including their finite-value status.
        previous = check_kv_step(previous, buffers, state["writes"], locations)
        step += 1
        if step in (prefix_end, single_step, replay_end):
            if step == single_step:
                assert rows == 1
            request = int(backend.row_requests.cpu()[-1])
            slots = backend.req_to_token_pool.req_to_token.cpu()[request, :prefix_length].long()
            current = [buffer[slots].clone() for buffer in buffers]
            if step == prefix_end:
                fixed_prefix = current
            else:
                assert all(torch.equal(a.view(torch.int16), b.view(torch.int16))
                           for a, b in zip(fixed_prefix, current)), "same-prefix KV differs"
            if step == single_step:
                single_logits = logits.double()
            if step == replay_end:
                other = logits.double()
                relative = (torch.linalg.vector_norm(other - single_logits) /
                            torch.linalg.vector_norm(single_logits).clamp_min(1e-30)).item()
                maximum = (other - single_logits).abs().max().item()
                assert len(state["replay_inputs"]) == len(state["inputs"]) == layers + 1
                assert len(state["replay_attention"]) == len(state["attention"]) == layers
                if cross_tree:
                    torch.save({"inputs": state["replay_inputs"], "attention": state["replay_attention"],
                                "intermediates": [], "logits": logits},
                               os.environ["GAUDIRT_REPLAY_LAYER_CAPTURE_PATH"])
                else:
                    assert torch.equal(other, single_logits), (relative, maximum)
                    for index, (single, replay) in enumerate(zip(state["inputs"], state["replay_inputs"])):
                        assert torch.equal(single.view(torch.uint8), replay.view(torch.uint8)), ("layer input", index)
                    for index, (single, replay) in enumerate(zip(state["attention"], state["replay_attention"])):
                        assert torch.equal(single["output"].view(torch.uint8), replay["output"].view(torch.uint8)), (
                            "layer attention", index)
                print("MODEL_SAME_PREFIX", json.dumps({"relative_l2": relative, "max_abs": maximum,
                                                       "cross_tree": cross_tree}), flush=True)
        emit(struct.pack("<QQ", step, rows))
        emit(bytes_of(locations))
        emit(bytes_of(logits))
        for buffer in buffers:
            emit(bytes_of(buffer[locations]))
        if step in snapshot_steps:
            for buffer in buffers:
                emit(bytes_of(buffer))
        if state["active"]:
            assert len(state["inputs"]) == layers + 1 and len(state["attention"]) == layers
            capture = {"inputs": state["inputs"], "attention": state["attention"],
                       "intermediates": state["intermediates"], "logits": logits}
            torch.save(capture, os.environ["GAUDIRT_LAYER_CAPTURE_PATH"])
            state["complete"] = True
        state["active"] = False
        if step == expected_steps:
            assert state["complete"]
            assert state["bound_traced_steps"] > 0, "audit did not execute the bound runner"
            if mode == "warm":
                assert not trace.read(1), "cold trace has extra data"
                assert gaudirt.recipe_cache_stats()[1] == 0, gaudirt.recipe_cache_stats()
            trace.close()
            summary = {"mode": mode, "layers": layers, "steps": step, "model_captures": captures,
                       "cache": gaudirt.recipe_cache_stats(), "peak_hbm_bytes": gaudirt.max_memory_allocated(),
                       "trace_sha256": digest.hexdigest(), "compile_deltas": state["compile_deltas"],
                       "bound_traced_steps": state["bound_traced_steps"],
                       "same_prefix_check": "independent_fp64_required" if cross_tree else "bit_exact"}
            Path(os.environ["GAUDIRT_SUMMARY_PATH"]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print("FULL_AUDIT", json.dumps(summary, sort_keys=True), flush=True)
        return output

    return execute


compiler.gaudirt_backend = audit_backend
GaudiRTSRTPlatform.prepare_model_forward = audit_prepare


def main():
    from sglang.srt.entrypoints.engine import Engine

    model = Path(os.environ["GAUDIRT_TARGET_MODEL"])
    config = json.loads((model / "config.json").read_text())
    assert config["num_hidden_layers"] == layers
    vocabulary = config["vocab_size"]
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

    engine = Engine(model_path=str(model), skip_tokenizer_init=True, device="gaudirt", dtype="bfloat16",
                    tp_size=1, max_total_tokens=int(os.environ.get("GAUDIRT_MAX_TOTAL_TOKENS", "4608")),
                    max_running_requests=1, context_length=int(os.environ.get("GAUDIRT_CONTEXT_LENGTH", "4608")),
                    mem_fraction_static=float(os.environ.get("GAUDIRT_STATIC_FRACTION", "0.9")),
                    model_loader_extra_config=os.environ.get("GAUDIRT_MODEL_LOADER_CONFIG", "{}"),
                    page_size=int(os.environ.get("GAUDIRT_TEST_PAGE_SIZE", "1")),
                    disable_cuda_graph=True, disable_radix_cache=True, disable_overlap_schedule=True,
                    chunked_prefill_size=chunk_size, log_level="warning")
    results = []
    try:
        prefix = ids(prefix_length)
        first = engine.generate(input_ids=prefix, sampling_params={
            "temperature": 0, "max_new_tokens": 2, "ignore_eos": True,
        })
        second = engine.generate(input_ids=prefix + first["output_ids"][:1], sampling_params={
            "temperature": 0, "max_new_tokens": 1, "ignore_eos": True,
        })
        assert len(first["output_ids"]) == 2 and len(second["output_ids"]) == 1
        if not cross_tree:
            assert first["output_ids"][1:] == second["output_ids"]
        results.append({"prefix_tokens": prefix_length, "outputs": first["output_ids"],
                        "replay_outputs": second["output_ids"]})
        for length in prompt_lengths:
            result = engine.generate(input_ids=ids(length), sampling_params={
                "temperature": 0, "max_new_tokens": decode_steps + 1, "ignore_eos": True,
            })
            tokens = result["output_ids"]
            assert len(tokens) == decode_steps + 1
            record = {"prompt_tokens": length, "output_ids": tokens}
            if tokenizer is not None:
                record["text"] = tokenizer.decode(tokens, skip_special_tokens=False)
            results.append(record)
            print("PROMPT_COMPLETE", length, "output_tokens", len(tokens), flush=True)
        Path(os.environ["GAUDIRT_TOKEN_PATH"]).write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    finally:
        watchdog = engine.tokenizer_manager._subprocess_watchdog
        workers = list(zip(watchdog._names, watchdog._processes))
        gaudirt._audit_legacy_bridge()
        for name, process in workers:
            with open(f"/proc/{process.pid}/maps", encoding="utf-8") as maps:
                mapped = maps.read().lower()
            assert not any(part in mapped for part in ("libhabana_pytorch", "gaudi-pytorch-bridge", "habana_frameworks"))
            print("WORKER_MAP_AUDIT", name, process.pid, flush=True)
        engine.shutdown()
        for name, process in workers:
            assert process.exitcode == 0, (name, process.exitcode)
            print("WORKER_EXIT", name, process.exitcode, flush=True)
        if path := os.environ.get("GAUDIRT_SITE_AUDIT_LOG"):
            entries = [json.loads(line) for line in Path(path).read_text().splitlines()]
            for name, process in workers:
                stages = {item["stage"] for item in entries if item["pid"] == process.pid and
                          item["module_clean"] and item["maps_clean"]}
                assert {"entry", "exit"} <= stages, (name, process.pid, stages)


if __name__ == "__main__":
    main()
