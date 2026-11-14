"""Multi-device (TP=2) SGLang numerical, feature, and uninstrumented streaming gate."""

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import gaudirt
import torch
from sglang.srt.managers.scheduler import Scheduler
from sglang_gaudirt.platform import GaudiRTSRTPlatform

_MODE = os.environ.get("GAUDIRT_TP_GATE_MODE", "tp2_bench")
_CAPTURE_LOGITS_DIR = os.environ.get("GAUDIRT_TP_LOGITS_DIR", "")

_orig_get_internal_state = Scheduler.get_internal_state


def _state_with_tp_measurement(self, request):
    result = _orig_get_internal_state(self, request)
    sched_summary = None
    try:
        model_runner = getattr(getattr(self, "tp_worker", None), "model_runner", None)
        fwd = getattr(getattr(model_runner, "model", None), "forward", None)
        exec_obj = getattr(fwd, "_gaudirt_last_execution", None)
        if exec_obj is not None and hasattr(exec_obj, "runner"):
            raw_sched = list(exec_obj.runner.schedule_summary())
            sched_summary = {
                "total_ops": len(raw_sched),
                "programs": sum(1 for k, _, _ in raw_sched if k == "program"),
                "attentions": sum(1 for k, _, _ in raw_sched if k == "attention"),
                "gather_rows": sum(1 for k, _, _ in raw_sched if k == "gather_rows"),
                "collective_starts": sum(1 for k, _, _ in raw_sched if k == "collective_start"),
                "dual_stream_collectives": sum(1 for k, _, dual in raw_sched if k == "collective_start" and dual),
            }
    except Exception:
        sched_summary = None

    from sglang.srt.distributed import get_tp_group

    tp_group = get_tp_group()
    tp_rank = int(tp_group.rank_in_group)
    tp_size = int(tp_group.world_size)
    dev_idx = int(torch.gaudirt.current_device())
    local_m = {
        "tp_rank": tp_rank,
        "device_index": dev_idx,
        "physical_module_id": int(gaudirt._C.visible_module_for_device(dev_idx)),
        "peak_allocated_bytes": int(gaudirt.max_memory_allocated()),
        "allocated_bytes": int(gaudirt.memory_allocated()),
        "recipe_cache": list(gaudirt.recipe_cache_stats()),
        "schedule_summary": sched_summary,
    }
    cpu_group = tp_group.cpu_group if tp_size > 1 else None
    if tp_size > 1 and cpu_group is not None:
        gathered = [None] * tp_size
        torch.distributed.all_gather_object(gathered, local_m, group=cpu_group)
        result.internal_state["gaudirt_tp_measurements"] = gathered
    else:
        result.internal_state["gaudirt_tp_measurements"] = [local_m]
    return result


Scheduler.get_internal_state = _state_with_tp_measurement

# Only attach logit capture in non-benchmark numerical audit modes.
if _CAPTURE_LOGITS_DIR and (_MODE == "reference_tp1" or _MODE.endswith(("_cold", "_warm"))):
    _orig_prepare = GaudiRTSRTPlatform.prepare_model_forward

    def _capture_prepare(self, forward):
        compiled = _orig_prepare(self, forward)
        prefill_idx = 0

        def _wrapped(*args, **kwargs):
            nonlocal prefill_idx
            out = compiled(*args, **kwargs)
            from sglang.srt.model_executor.forward_batch_info import ForwardMode

            batch = args[2]
            if batch.forward_mode == ForwardMode.EXTEND and int(torch.gaudirt.current_device()) == 0:
                logits_cpu = out.next_token_logits[-1:].detach().cpu().contiguous()
                assert torch.isfinite(logits_cpu).all(), "non-finite prefill logits"
                out_dir = Path(_CAPTURE_LOGITS_DIR)
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(logits_cpu, out_dir / f"prefill_{prefill_idx}.pt")
                prefill_idx += 1
            if hasattr(compiled, "_gaudirt_last_execution"):
                _wrapped._gaudirt_last_execution = compiled._gaudirt_last_execution
            return out

        return _wrapped

    GaudiRTSRTPlatform.prepare_model_forward = _capture_prepare


def _query_hl_smi():
    cmd = [
        "hl-smi",
        "-Q",
        "index,module_id,memory.total,memory.used,memory.free",
        "-f",
        "csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    devices = {}
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5:
            mod_id = int(parts[1])
            devices[str(mod_id)] = {
                "index": int(parts[0]),
                "module_id": mod_id,
                "total_mib": int(parts[2]),
                "used_mib": int(parts[3]),
                "free_mib": int(parts[4]),
            }
    return devices


def _collect_rank_measurements(engine, expected_tp_size: int):
    info = engine.get_server_info()
    found_by_rank = {}

    def _walk(obj):
        if isinstance(obj, dict):
            if "gaudirt_tp_measurements" in obj:
                for m in obj["gaudirt_tp_measurements"]:
                    if isinstance(m, dict) and "tp_rank" in m:
                        found_by_rank[int(m["tp_rank"])] = m
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(info)
    assert len(found_by_rank) == expected_tp_size, (
        f"expected measurements from {expected_tp_size} TP ranks, got {sorted(found_by_rank.keys())}"
    )
    return [found_by_rank[r] for r in range(expected_tp_size)]


def _logit_metrics(ref_bf16: torch.Tensor, cand_bf16: torch.Tensor) -> dict:
    ref = ref_bf16.double().reshape(-1)
    cand = cand_bf16.double().reshape(-1)
    diff = cand - ref
    max_abs = float(diff.abs().max().item())
    rel_l2 = float(torch.linalg.vector_norm(diff).item() / max(torch.linalg.vector_norm(ref).item(), 1e-30))
    cos = float(torch.nn.functional.cosine_similarity(ref.unsqueeze(0), cand.unsqueeze(0)).item())
    log_p = torch.log_softmax(ref, dim=-1)
    log_q = torch.log_softmax(cand, dim=-1)
    p = log_p.exp()
    kl = float(torch.sum(p * (log_p - log_q)).clamp_min(0.0).item())
    bit_exact = bool(torch.equal(ref_bf16.view(torch.uint8), cand_bf16.view(torch.uint8)))
    return {
        "bit_exact": bit_exact,
        "max_abs": max_abs,
        "rel_l2": rel_l2,
        "cosine_similarity": cos,
        "kl_ref_to_cand": kl,
        "top1_ref": int(ref.argmax().item()),
        "top1_cand": int(cand.argmax().item()),
        "top1_match": int(ref.argmax().item()) == int(cand.argmax().item()),
    }


def _build_prompt_fn(model_dir: Path, vocabulary: int):
    tokenizer = None
    if os.environ.get("GAUDIRT_TEXT_MODE") == "1":
        from transformers import AutoTokenizer
        from phase5a_prompt import prompt_builder

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False
        )
        docs = [Path(p) for p in os.environ["GAUDIRT_PROMPT_DOCUMENTS"].split(os.pathsep)]
        ids_fn = prompt_builder(tokenizer, docs)
    else:
        def ids_fn(length: int):
            return [3 + (i * 37) % (vocabulary - 3) for i in range(length)]
    return ids_fn, tokenizer


def _audit_and_shutdown(engine):
    watchdog = engine.tokenizer_manager._subprocess_watchdog
    workers = list(zip(watchdog._names, watchdog._processes))
    gaudirt._audit_legacy_bridge()
    worker_pids = []
    for name, proc in workers:
        worker_pids.append({"name": name, "pid": proc.pid})
        with open(f"/proc/{proc.pid}/maps", encoding="utf-8") as maps_file:
            mapped = maps_file.read().lower()
        assert not any(
            bad in mapped for bad in ("libhabana_pytorch", "gaudi-pytorch-bridge", "habana_frameworks")
        ), f"legacy bridge mapped in worker {name} (pid={proc.pid})"
    try:
        engine.shutdown()
    except Exception:
        for _, proc in workers:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        raise
    for name, proc in workers:
        assert proc.exitcode == 0, f"worker {name} exited with {proc.exitcode}"
    if site_log := os.environ.get("GAUDIRT_SITE_AUDIT_LOG"):
        entries = [json.loads(line) for line in Path(site_log).read_text().splitlines() if line.strip()]
        for item in worker_pids:
            stages = {
                e["stage"]
                for e in entries
                if e["pid"] == item["pid"] and e["module_clean"] and e["maps_clean"]
            }
            assert {"entry", "exit"} <= stages, (item, stages)
    return worker_pids


def main():
    from sglang.srt.entrypoints.engine import Engine

    model = Path(os.environ["GAUDIRT_TARGET_MODEL"])
    config = json.loads((model / "config.json").read_text())
    expected_layers = int(os.environ["GAUDIRT_EXPECT_LAYERS"])
    assert config["num_hidden_layers"] == expected_layers, (
        config["num_hidden_layers"],
        expected_layers,
    )
    vocabulary = int(config["vocab_size"])
    tp_size = 1 if _MODE == "reference_tp1" else int(os.environ.get("GAUDIRT_TP_SIZE", "2"))
    summary_path = Path(os.environ["GAUDIRT_SUMMARY_PATH"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if _MODE.endswith("_features"):
        from phase5a_feature_prompts import feature_prompts

        prompts = feature_prompts(vocabulary)
        page_size = int(os.environ.get("GAUDIRT_TEST_PAGE_SIZE", "16"))
        chunk_size = int(os.environ.get("GAUDIRT_FEATURE_CHUNK_SIZE", "512"))
        context_len = int(os.environ.get("GAUDIRT_FEATURE_CONTEXT", "4608"))
        max_tokens = int(os.environ.get("GAUDIRT_FEATURE_MAX_TOKENS", "4608"))
        min_cached = int(os.environ.get("GAUDIRT_FEATURE_MIN_CACHED", "128"))

        engine = Engine(
            model_path=str(model),
            skip_tokenizer_init=True,
            device="gaudirt",
            dtype="bfloat16",
            tp_size=tp_size,
            max_total_tokens=max_tokens,
            max_running_requests=1,
            context_length=context_len,
            mem_fraction_static=float(os.environ.get("GAUDIRT_STATIC_FRACTION", "0.9")),
            page_size=page_size,
            disable_cuda_graph=True,
            disable_radix_cache=False,
            disable_overlap_schedule=True,
            chunked_prefill_size=chunk_size,
            log_level="warning",
        )
        try:
            outputs = [
                engine.generate(
                    input_ids=p,
                    sampling_params={"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
                )
                for p in prompts
            ]
            records = [
                {"tokens": r["output_ids"], "cached_tokens": int(r["meta_info"]["cached_tokens"])}
                for r in outputs
            ]
            assert all(len(item["tokens"]) == 8 for item in records), records
            assert records[1]["cached_tokens"] >= min_cached, records
            ranks = _collect_rank_measurements(engine, tp_size)
            hl_during = _query_hl_smi()
        finally:
            workers = _audit_and_shutdown(engine)
        hl_after = _query_hl_smi()
        receipt = {
            "mode": _MODE,
            "layers": expected_layers,
            "tp_size": tp_size,
            "page_size": page_size,
            "chunked_prefill_size": chunk_size,
            "radix_cache": True,
            "requests": records,
            "rank_measurements": ranks,
            "workers": workers,
            "hl_smi_during": hl_during,
            "hl_smi_after": hl_after,
        }
        summary_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print("TP2_FEATURES_PASS", json.dumps({"requests": records, "ranks": ranks}), flush=True)
        return

    ids_fn, tokenizer = _build_prompt_fn(model, vocabulary)
    prompt_lengths = tuple(
        int(x) for x in os.environ.get("GAUDIRT_PROMPTS", "137,1024").split(",") if x.strip()
    )
    decode_steps = int(os.environ.get("GAUDIRT_DECODE_STEPS", "32"))

    if _CAPTURE_LOGITS_DIR:
        cap_dir = Path(_CAPTURE_LOGITS_DIR)
        cap_dir.mkdir(parents=True, exist_ok=True)
        for stale in cap_dir.glob("prefill_*.pt"):
            stale.unlink()

    engine = Engine(
        model_path=str(model),
        skip_tokenizer_init=True,
        device="gaudirt",
        dtype="bfloat16",
        tp_size=tp_size,
        max_total_tokens=int(os.environ.get("GAUDIRT_MAX_TOTAL_TOKENS", "4608")),
        max_running_requests=1,
        context_length=int(os.environ.get("GAUDIRT_CONTEXT_LENGTH", "4608")),
        mem_fraction_static=float(os.environ.get("GAUDIRT_STATIC_FRACTION", "0.9")),
        disable_cuda_graph=True,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        chunked_prefill_size=-1,
        stream_interval=1,
        log_level="warning",
    )

    try:
        prefix_136 = ids_fn(136)
        first = engine.generate(
            input_ids=prefix_136,
            sampling_params={"temperature": 0, "max_new_tokens": 2, "ignore_eos": True},
        )
        second = engine.generate(
            input_ids=prefix_136 + first["output_ids"][:1],
            sampling_params={"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
        )
        assert len(first["output_ids"]) == 2 and len(second["output_ids"]) == 1

        generations = [
            {
                "prefix_tokens": 136,
                "outputs": first["output_ids"],
                "replay_outputs": second["output_ids"],
                "same_prefix_replay_match": first["output_ids"][1:] == second["output_ids"],
            }
        ]
        assert generations[0]["same_prefix_replay_match"], generations[0]

        for length in prompt_lengths:
            gen = engine.generate(
                input_ids=ids_fn(length),
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": decode_steps,
                    "ignore_eos": True,
                },
            )
            toks = gen["output_ids"]
            assert len(toks) == decode_steps, (length, len(toks))
            rec = {"prompt_tokens": length, "output_ids": toks}
            if tokenizer is not None:
                rec["text"] = tokenizer.decode(toks, skip_special_tokens=False)
            generations.append(rec)
            print("PROMPT_DONE", _MODE, length, len(toks), flush=True)

        rank_stats = _collect_rank_measurements(engine, tp_size)
        hl_during = _query_hl_smi()

        if _MODE.endswith("_warm"):
            for r_stat in rank_stats:
                assert r_stat["recipe_cache"][0] > 0 and r_stat["recipe_cache"][1] == 0, (
                    f"expected >0 disk loads and 0 warm compiles on rank {r_stat['tp_rank']}, got {r_stat['recipe_cache']}"
                )

        timings = []
        if _MODE.endswith("_bench"):
            bench_lengths = tuple(
                int(x) for x in os.environ.get("GAUDIRT_BENCH_PROMPTS", "1024,4096").split(",") if x.strip()
            )
            bench_output = int(os.environ.get("GAUDIRT_BENCH_OUTPUT", "128"))
            for length in bench_lengths:
                prompt = ids_fn(length)
                warmup = list(
                    engine.generate(
                        input_ids=prompt,
                        sampling_params={
                            "temperature": 0,
                            "max_new_tokens": bench_output,
                            "ignore_eos": True,
                        },
                        stream=True,
                    )
                )
                assert len(warmup[-1]["output_ids"]) == bench_output
                before_ranks = _collect_rank_measurements(engine, tp_size)

                t0 = time.perf_counter()
                first_at = last_at = None
                out_ids = []
                for chunk in engine.generate(
                    input_ids=prompt,
                    sampling_params={
                        "temperature": 0,
                        "max_new_tokens": bench_output,
                        "ignore_eos": True,
                    },
                    stream=True,
                ):
                    now = time.perf_counter()
                    cur = chunk["output_ids"]
                    if len(cur) > len(out_ids):
                        if first_at is None:
                            assert len(cur) == 1, f"expected 1 first token, got {len(cur)}"
                            first_at = now
                        last_at = now
                        out_ids = list(cur)
                assert len(out_ids) == bench_output and first_at is not None and last_at is not None
                after_ranks = _collect_rank_measurements(engine, tp_size)
                for b_stat, a_stat in zip(before_ranks, after_ranks):
                    assert (
                        a_stat["recipe_cache"][1] == b_stat["recipe_cache"][1]
                        and a_stat["recipe_cache"][2] == b_stat["recipe_cache"][2]
                    ), (
                        f"unexpected compilation during timed run on rank {a_stat['tp_rank']}: "
                        f"{b_stat['recipe_cache']} -> {a_stat['recipe_cache']}"
                    )
                rec = {
                    "prompt_tokens": length,
                    "output_tokens": len(out_ids),
                    "ttft_ms": (first_at - t0) * 1000.0,
                    "tpot_ms": (last_at - first_at) * 1000.0 / (len(out_ids) - 1),
                    "ranks": after_ranks,
                }
                if tokenizer is not None:
                    rec["sample_text_preview"] = tokenizer.decode(out_ids[:48], skip_special_tokens=False)
                timings.append(rec)
                print("TP2_UNINSTRUMENTED_TIMING", json.dumps(rec, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        workers = _audit_and_shutdown(engine)

    hl_after = _query_hl_smi()
    comparisons = {}

    if cold_summary := os.environ.get("GAUDIRT_COLD_SUMMARY"):
        cold_data = json.loads(Path(cold_summary).read_text())
        assert generations == cold_data["generations"], f"cold vs warm TP={tp_size} tokens differ"
        cold_logits_dir = Path(os.environ["GAUDIRT_COLD_LOGITS_DIR"])
        cur_logits_dir = Path(_CAPTURE_LOGITS_DIR)
        cold_files = sorted(cold_logits_dir.glob("prefill_*.pt"))
        cur_files = sorted(cur_logits_dir.glob("prefill_*.pt"))
        assert len(cold_files) == len(cur_files) and len(cold_files) > 0
        for cf, wf in zip(cold_files, cur_files):
            c_t = torch.load(cf, weights_only=True)
            w_t = torch.load(wf, weights_only=True)
            assert torch.equal(c_t.view(torch.uint8), w_t.view(torch.uint8)), (
                f"cold vs warm TP={tp_size} prefill logits differ on {cf.name}"
            )
        comparisons["cold_vs_warm_bit_exact"] = True

    if tp1_summary := os.environ.get("GAUDIRT_TP1_SUMMARY"):
        tp1_data = json.loads(Path(tp1_summary).read_text())
        token_matches = []
        for g1, g2 in zip(tp1_data["generations"][1:], generations[1:]):
            s1, s2 = g1["output_ids"], g2["output_ids"]
            prefix_agree = 0
            for a, b in zip(s1, s2):
                if a != b:
                    break
                prefix_agree += 1
            exact_ratio = sum(1 for a, b in zip(s1, s2) if a == b) / max(len(s1), 1)
            token_matches.append({
                "prompt_tokens": g2["prompt_tokens"],
                "exact_match": s1 == s2,
                "prefix_agree_tokens": prefix_agree,
                "position_match_ratio": exact_ratio,
            })
        logit_checks = []
        if tp1_logits_dir := os.environ.get("GAUDIRT_TP1_LOGITS_DIR"):
            ref_files = sorted(Path(tp1_logits_dir).glob("prefill_*.pt"))
            cand_files = sorted(Path(_CAPTURE_LOGITS_DIR).glob("prefill_*.pt"))
            assert len(ref_files) == len(cand_files) and len(ref_files) > 0
            for rf, cf in zip(ref_files, cand_files):
                m = _logit_metrics(torch.load(rf, weights_only=True), torch.load(cf, weights_only=True))
                m["step"] = cf.stem
                assert m["kl_ref_to_cand"] <= 0.01, f"TP=1 vs TP={tp_size} KL too high: {m}"
                logit_checks.append(m)
        comparisons[f"tp1_vs_tp{tp_size}"] = {
            "token_comparisons": token_matches,
            "prefill_logit_metrics": logit_checks,
        }

    receipt = {
        "mode": _MODE,
        "layers": expected_layers,
        "tp_size": tp_size,
        "pool_reserved_bytes_per_rank": int(os.environ.get("GAUDIRT_HBM_BYTES", str(80 * 1024**3))),
        "rank_measurements": rank_stats,
        "generations": generations,
        "timings": timings,
        "comparisons": comparisons,
        "workers": workers,
        "hl_smi_during": hl_during,
        "hl_smi_after": hl_after,
    }
    summary_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print("TP_GATE_COMPLETE", _MODE, f"tp={tp_size}", f"layers={expected_layers}", flush=True)


if __name__ == "__main__":
    main()
