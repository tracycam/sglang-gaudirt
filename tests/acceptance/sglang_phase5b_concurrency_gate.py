"""High-concurrency (B=8/16/32, ForwardMode.MIXED) functional & uninstrumented SGLang gate."""

import asyncio
import json
import os
import statistics
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
import sglang_gaudirt.ops as _gaudirt_ops

_MODE = os.environ.get("GAUDIRT_CONCURRENCY_MODE", "throughput_bench")

_MODE_COUNTS = {}
_BITMASK_STATS = {"calls": 0, "devices": []}
_ORIG_PREPARE_FORWARD = GaudiRTSRTPlatform.prepare_model_forward
_ORIG_REGISTER_GRAMMAR_MASK_OPS = _gaudirt_ops._register_grammar_mask_ops
_WRAPPED_GRAMMAR = False

if _MODE == "functional_mixed":
    def _wrap_prepare(self, forward):
        compiled = _ORIG_PREPARE_FORWARD(self, forward)

        def _wrapped(*args, **kwargs):
            batch = args[2]
            name = getattr(getattr(batch, "forward_mode", None), "name", str(batch.forward_mode))
            _MODE_COUNTS[name] = _MODE_COUNTS.get(name, 0) + 1
            out = compiled(*args, **kwargs)
            if hasattr(compiled, "_gaudirt_last_execution"):
                _wrapped._gaudirt_last_execution = compiled._gaudirt_last_execution
            return out

        return _wrapped

    def _wrap_grammar_ops():
        global _WRAPPED_GRAMMAR
        _ORIG_REGISTER_GRAMMAR_MASK_OPS()
        if _WRAPPED_GRAMMAR:
            return
        from sglang.srt.constrained import llguidance_backend, xgrammar_backend

        for cls, is_static in (
            (xgrammar_backend.XGrammarGrammar, False),
            (xgrammar_backend.XGrammarGrammarBackend, True),
            (llguidance_backend.GuidanceGrammar, True),
        ):
            orig = cls.apply_vocab_mask
            if getattr(orig, "_gaudirt_conc_counted", False):
                continue
            if is_static:
                @staticmethod
                def _counted(logits: torch.Tensor, vocab_mask: torch.Tensor, _orig=orig) -> None:
                    _BITMASK_STATS["calls"] += 1
                    if logits.device.type not in _BITMASK_STATS["devices"]:
                        _BITMASK_STATS["devices"].append(logits.device.type)
                    return _orig(logits, vocab_mask)

                _counted._gaudirt_conc_counted = True
                cls.apply_vocab_mask = _counted
            else:
                def _counted(self, logits: torch.Tensor, vocab_mask: torch.Tensor, _orig=orig) -> None:
                    _BITMASK_STATS["calls"] += 1
                    if logits.device.type not in _BITMASK_STATS["devices"]:
                        _BITMASK_STATS["devices"].append(logits.device.type)
                    return _orig(self, logits, vocab_mask)

                _counted._gaudirt_conc_counted = True
                cls.apply_vocab_mask = _counted
        _WRAPPED_GRAMMAR = True

    GaudiRTSRTPlatform.prepare_model_forward = _wrap_prepare
    _gaudirt_ops._register_grammar_mask_ops = _wrap_grammar_ops


_ORIG_GET_INTERNAL_STATE = Scheduler.get_internal_state


def _state_with_conc_metrics(self, request):
    result = _ORIG_GET_INTERNAL_STATE(self, request)
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
        "forward_mode_counts": dict(_MODE_COUNTS),
        "bitmask_calls": int(_BITMASK_STATS["calls"]),
        "bitmask_devices": list(_BITMASK_STATS["devices"]),
    }
    cpu_group = tp_group.cpu_group if tp_size > 1 else None
    if tp_size > 1 and cpu_group is not None:
        gathered = [None] * tp_size
        torch.distributed.all_gather_object(gathered, local_m, group=cpu_group)
        result.internal_state["gaudirt_conc_measurements"] = gathered
    else:
        result.internal_state["gaudirt_conc_measurements"] = [local_m]
    return result


Scheduler.get_internal_state = _state_with_conc_metrics

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "enum": ["alpha", "beta", "gamma"]},
        "count": {"type": "integer", "minimum": 0, "maximum": 99},
    },
    "required": ["name", "count"],
    "additionalProperties": False,
}
JSON_SCHEMA_STR = json.dumps(JSON_SCHEMA, sort_keys=True)


def _validate_json_output(text: str) -> dict:
    parsed = json.loads(text)
    assert isinstance(parsed, dict) and set(parsed.keys()) == {"name", "count"}, text
    assert parsed["name"] in ("alpha", "beta", "gamma"), parsed
    assert isinstance(parsed["count"], int) and 0 <= parsed["count"] <= 99, parsed
    return parsed


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


def _collect_rank_measurements(engine, tp_size: int):
    info = engine.get_server_info()
    found = {}

    def _walk(obj):
        if isinstance(obj, dict):
            if "gaudirt_conc_measurements" in obj:
                for item in obj["gaudirt_conc_measurements"]:
                    if isinstance(item, dict) and "tp_rank" in item:
                        found[int(item["tp_rank"])] = item
            for val in obj.values():
                _walk(val)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(info)
    assert len(found) == tp_size, f"expected {tp_size} rank measurements, got {sorted(found)}"
    return [found[r] for r in range(tp_size)]


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


def _make_ids(vocab_size: int, length: int, seed: int) -> list:
    return [3 + ((seed + i * 37) % (vocab_size - 4)) for i in range(length)]


def _run_functional_mixed(engine, model_dir: Path, tp_size: int, vocab_size: int, expected_layers: int,
                          summary_path: Path):
    chunk_size = int(os.environ.get("GAUDIRT_CONC_CHUNK_SIZE", "512"))
    page_size = int(os.environ.get("GAUDIRT_TEST_PAGE_SIZE", "16"))
    decode_steps = int(os.environ.get("GAUDIRT_FUNC_DECODE_STEPS", "32"))
    wave_concurrency = int(os.environ.get("GAUDIRT_FUNC_CONCURRENCY", "16"))

    # 1. Single-request baseline for greedy canary and shared-prefix seeding.
    canary_a = _make_ids(vocab_size, 137, 1001)
    canary_b = _make_ids(vocab_size, 640, 2003)
    ref_a = engine.generate(
        input_ids=canary_a,
        sampling_params={"temperature": 0, "max_new_tokens": decode_steps, "ignore_eos": True},
    )
    ref_b = engine.generate(
        input_ids=canary_b,
        sampling_params={"temperature": 0, "max_new_tokens": decode_steps, "ignore_eos": True},
    )
    shared_prefix = _make_ids(vocab_size, 320, 9001)
    prefix_seed = engine.generate(
        input_ids=shared_prefix + _make_ids(vocab_size, 24, 9101),
        sampling_params={"temperature": 0, "max_new_tokens": 16, "ignore_eos": True},
    )
    assert len(ref_a["output_ids"]) == decode_steps
    assert len(ref_b["output_ids"]) == decode_steps
    assert len(prefix_seed["output_ids"]) == 16

    async def _run_wave(wave_idx: int):
        tasks = []
        # Slot 0 & 1: uncached greedy prompts matching fresh canary pattern vs wave 0/1
        canary_uncached_a = _make_ids(vocab_size, 137, 3001 + wave_idx * 500)
        tasks.append((
            "canary_a_cached",
            asyncio.create_task(
                engine.async_generate(
                    input_ids=canary_a,
                    sampling_params={"temperature": 0, "max_new_tokens": decode_steps, "ignore_eos": True},
                )
            ),
        ))
        await asyncio.sleep(0.002)
        tasks.append((
            "canary_b_cached",
            asyncio.create_task(
                engine.async_generate(
                    input_ids=canary_b,
                    sampling_params={"temperature": 0, "max_new_tokens": decode_steps, "ignore_eos": True},
                )
            ),
        ))
        tasks.append((
            "shared_prefix",
            asyncio.create_task(
                engine.async_generate(
                    input_ids=shared_prefix + _make_ids(vocab_size, 24, 9201 + wave_idx * 17),
                    sampling_params={"temperature": 0, "max_new_tokens": 16, "ignore_eos": True},
                )
            ),
        ))
        # Remaining slots up to wave_concurrency: mix long chunked prefill (> chunk_size), JSON schema, stochastic, short greedy.
        for slot in range(3, wave_concurrency):
            if slot % 4 == 0:
                if expected_layers >= 10 and slot == 4:
                    sp_json = {
                        "temperature": 0,
                        "max_new_tokens": 128,
                        "json_schema": JSON_SCHEMA_STR,
                    }
                else:
                    sp_json = {
                        "temperature": 1.0 if expected_layers < 10 else 0.6,
                        "top_k": 64,
                        "top_p": 0.95,
                        "sampling_seed": 1 if expected_layers < 10 else (5000 + wave_idx * 100 + slot),
                        "max_new_tokens": 128,
                        "json_schema": JSON_SCHEMA_STR,
                    }
                tasks.append((
                    "json_schema",
                    asyncio.create_task(
                        engine.async_generate(
                            prompt='Compact JSON example: {"name":"alpha","count":1}\nOutput compact JSON:\n{',
                            sampling_params=sp_json,
                        )
                    ),
                ))
            elif slot % 4 == 1:
                prompt_len = chunk_size + 128 + (slot % 3) * 128
                tasks.append((
                    "long_chunk_greedy",
                    asyncio.create_task(
                        engine.async_generate(
                            input_ids=_make_ids(vocab_size, prompt_len, 7000 + wave_idx * 100 + slot),
                            sampling_params={"temperature": 0, "max_new_tokens": decode_steps, "ignore_eos": True},
                        )
                    ),
                ))
            elif slot % 4 == 2:
                tasks.append((
                    "stochastic",
                    asyncio.create_task(
                        engine.async_generate(
                            input_ids=_make_ids(vocab_size, 192 + slot * 8, 8000 + wave_idx * 100 + slot),
                            sampling_params={
                                "temperature": 0.8,
                                "top_k": 16,
                                "top_p": 0.9,
                                "min_p": 0.05,
                                "sampling_seed": 6000 + wave_idx * 100 + slot,
                                "max_new_tokens": decode_steps,
                                "ignore_eos": True,
                            },
                        )
                    ),
                ))
            else:
                tasks.append((
                    "short_greedy",
                    asyncio.create_task(
                        engine.async_generate(
                            input_ids=canary_uncached_a if slot == 3 else _make_ids(vocab_size, 160, 4000 + wave_idx * 100 + slot),
                            sampling_params={"temperature": 0, "max_new_tokens": decode_steps, "ignore_eos": True},
                        )
                    ),
                ))
            await asyncio.sleep(0.002)

        outputs = await asyncio.gather(*(t for _, t in tasks))
        return [(kind, out) for (kind, _), out in zip(tasks, outputs)]

    wave_summaries = []
    json_checked = 0
    radix_hits = []
    canary_checks = []
    for wave_idx in range(2):
        res_list = engine.loop.run_until_complete(_run_wave(wave_idx))
        for kind, out in res_list:
            ids = out["output_ids"]
            assert all(0 <= t < vocab_size for t in ids), (kind, ids[:10])
            if kind == "json_schema":
                _validate_json_output(out["text"])
                json_checked += 1
            elif kind == "shared_prefix":
                cached = int(out["meta_info"]["cached_tokens"])
                assert cached >= 128, (wave_idx, cached)
                radix_hits.append(cached)
            elif kind in ("canary_a_cached", "canary_b_cached"):
                ref_ids = ref_a["output_ids"] if kind == "canary_a_cached" else ref_b["output_ids"]
                prefix_agree = 0
                for a, b in zip(ref_ids, ids):
                    if a != b:
                        break
                    prefix_agree += 1
                ratio = sum(1 for a, b in zip(ref_ids, ids) if a == b) / max(len(ref_ids), 1)
                assert len(ids) == decode_steps
                min_prefix = 8 if expected_layers >= 10 else 4
                assert prefix_agree >= min_prefix or ratio >= 0.5, (
                    kind, wave_idx, prefix_agree, ratio, ref_ids[:8], ids[:8]
                )
                canary_checks.append({
                    "wave": wave_idx,
                    "kind": kind,
                    "exact_match": ref_ids == ids,
                    "prefix_agree_tokens": prefix_agree,
                    "match_ratio": ratio,
                })
            else:
                assert len(ids) == decode_steps, (kind, len(ids))
        wave_summaries.append({"wave": wave_idx, "completed_requests": len(res_list)})

    ranks = _collect_rank_measurements(engine, tp_size)
    for r in ranks:
        assert r["forward_mode_counts"].get("MIXED", 0) > 0, (
            f"expected ForwardMode.MIXED > 0 on rank {r['tp_rank']}, got {r['forward_mode_counts']}"
        )
        assert r["bitmask_calls"] > 0, f"expected bitmask_calls > 0 on rank {r['tp_rank']}, got {r}"
        assert r["bitmask_devices"] == ["gaudirt"], r

    hl_during = _query_hl_smi()
    return {
        "mode": _MODE,
        "layers": expected_layers,
        "tp_size": tp_size,
        "page_size": page_size,
        "chunked_prefill_size": chunk_size,
        "wave_concurrency": wave_concurrency,
        "total_concurrent_requests": sum(w["completed_requests"] for w in wave_summaries),
        "waves": wave_summaries,
        "json_requests_validated": json_checked,
        "radix_cached_tokens": radix_hits,
        "canary_checks": canary_checks,
        "rank_measurements": ranks,
        "hl_smi_during": hl_during,
    }


def _run_throughput_bench(engine, tp_size: int, vocab_size: int, expected_layers: int):
    conc_levels = [int(x) for x in os.environ.get("GAUDIRT_CONC_LEVELS", "1,8,16,32").split(",") if x.strip()]
    bench_output = int(os.environ.get("GAUDIRT_BENCH_OUTPUT", "64"))
    chunk_size = int(os.environ.get("GAUDIRT_CONC_CHUNK_SIZE", "512"))
    page_size = int(os.environ.get("GAUDIRT_TEST_PAGE_SIZE", "16"))
    prompt_lens = [int(x) for x in os.environ.get("GAUDIRT_CONC_PROMPT_LENS", "256,384,640").split(",") if x.strip()]
    req_multiplier = int(os.environ.get("GAUDIRT_CONC_REQ_MULTIPLIER", "2"))

    async def _drive_batch(
        concurrency: int,
        num_requests: int,
        seed_base: int,
        timed: bool,
        max_out_tokens: int = bench_output,
        stagger: bool = True,
        fixed_prompt_len=None,
    ):
        sem = asyncio.Semaphore(concurrency)
        per_req = []

        async def _one(req_idx: int):
            p_len = (
                fixed_prompt_len
                if fixed_prompt_len is not None
                else prompt_lens[req_idx % len(prompt_lens)]
            )
            prompt = _make_ids(vocab_size, p_len, seed_base + req_idx * 97)
            async with sem:
                req_start = time.perf_counter()
                first_at = None
                last_at = None
                out_count = 0
                async for chunk in await engine.async_generate(
                    input_ids=prompt,
                    sampling_params={
                        "temperature": 0,
                        "max_new_tokens": max_out_tokens,
                        "ignore_eos": True,
                    },
                    stream=True,
                ):
                    now = time.perf_counter()
                    cur = len(chunk["output_ids"])
                    if cur > out_count:
                        if first_at is None:
                            first_at = now
                        last_at = now
                        out_count = cur
                assert out_count == max_out_tokens and first_at is not None and last_at is not None, (
                    concurrency, req_idx, out_count
                )
                if timed:
                    per_req.append({
                        "prompt_tokens": p_len,
                        "output_tokens": out_count,
                        "ttft_ms": (first_at - req_start) * 1000.0,
                        "tpot_ms": (last_at - first_at) * 1000.0 / max(out_count - 1, 1),
                        "latency_ms": (last_at - req_start) * 1000.0,
                    })

        # Stagger initial arrivals slightly when requested so decode overlaps incoming chunked prefill.
        async def _staggered(req_idx: int):
            if stagger and req_idx > 0 and req_idx < concurrency:
                await asyncio.sleep(0.002 * req_idx)
            await _one(req_idx)

        wall_start = time.perf_counter()
        await asyncio.gather(*(_staggered(i) for i in range(num_requests)))
        wall_s = time.perf_counter() - wall_start
        return wall_s, per_req

    # Warm up all power-of-two batch/mixed buckets un-staggered (8 tokens) and staggered full passes
    # so zero compilations happen during timed sweeps.
    max_conc = max(conc_levels)
    pow2_buckets = [c for c in (1, 2, 4, 8, 16, 32) if c <= max_conc]
    warm_counts = sorted(set(pow2_buckets + conc_levels))
    for p_warm_idx, fixed_len in enumerate((None, 256, 640)):
        for idx, conc in enumerate(warm_counts):
            engine.loop.run_until_complete(
                _drive_batch(conc, conc, 50000 + p_warm_idx * 10000 + idx * 1000, False, 8, False, fixed_len)
            )
    for idx, conc in enumerate(sorted(set(conc_levels))):
        warm_reqs = max(conc * req_multiplier, 4 if conc == 1 else conc)
        engine.loop.run_until_complete(
            _drive_batch(conc, warm_reqs, 100000 + idx * 10000, False, bench_output, True)
        )

    sweeps = []
    for idx, conc in enumerate(conc_levels):
        num_requests = max(conc * req_multiplier, 4 if conc == 1 else conc)
        before_ranks = _collect_rank_measurements(engine, tp_size)
        wall_s, req_metrics = engine.loop.run_until_complete(
            _drive_batch(conc, num_requests, 500000 + idx * 10000, True)
        )
        after_ranks = _collect_rank_measurements(engine, tp_size)
        for b_stat, a_stat in zip(before_ranks, after_ranks):
            assert a_stat["recipe_cache"][1] == b_stat["recipe_cache"][1], (
                f"recompilation during timed concurrency B={conc} on rank {a_stat['tp_rank']}: "
                f"{b_stat['recipe_cache']} -> {a_stat['recipe_cache']}"
            )
            assert a_stat["recipe_cache"][2] == b_stat["recipe_cache"][2], (
                f"new resident recipe during timed concurrency B={conc} on rank {a_stat['tp_rank']}: "
                f"{b_stat['recipe_cache']} -> {a_stat['recipe_cache']}"
            )

        total_prompt = sum(r["prompt_tokens"] for r in req_metrics)
        total_out = sum(r["output_tokens"] for r in req_metrics)
        ttfts = sorted(r["ttft_ms"] for r in req_metrics)
        tpots = sorted(r["tpot_ms"] for r in req_metrics)
        p95_idx = min(len(ttfts) - 1, int(0.95 * (len(ttfts) - 1)))
        rec = {
            "concurrency": conc,
            "num_requests": num_requests,
            "total_prompt_tokens": total_prompt,
            "total_output_tokens": total_out,
            "wall_time_s": wall_s,
            "output_tokens_per_s": total_out / wall_s,
            "total_tokens_per_s": (total_prompt + total_out) / wall_s,
            "ttft_ms_p50": statistics.median(ttfts),
            "ttft_ms_p95": ttfts[p95_idx],
            "ttft_ms_mean": statistics.mean(ttfts),
            "tpot_ms_p50": statistics.median(tpots),
            "tpot_ms_mean": statistics.mean(tpots),
            "ranks": after_ranks,
        }
        sweeps.append(rec)
        print(
            "CONCURRENCY_BENCH",
            f"tp={tp_size}",
            f"B={conc}",
            json.dumps({
                "reqs": num_requests,
                "out_tok_s": round(rec["output_tokens_per_s"], 2),
                "total_tok_s": round(rec["total_tokens_per_s"], 2),
                "ttft_p50_ms": round(rec["ttft_ms_p50"], 2),
                "tpot_p50_ms": round(rec["tpot_ms_p50"], 2),
            }, sort_keys=True),
            flush=True,
        )

    hl_during = _query_hl_smi()
    return {
        "mode": _MODE,
        "layers": expected_layers,
        "tp_size": tp_size,
        "page_size": page_size,
        "chunked_prefill_size": chunk_size,
        "bench_output_tokens": bench_output,
        "prompt_lengths": prompt_lens,
        "radix_cache": False,
        "sweeps": sweeps,
        "hl_smi_during": hl_during,
    }


def main():
    from sglang.srt.entrypoints.engine import Engine

    model = Path(os.environ["GAUDIRT_TARGET_MODEL"])
    config = json.loads((model / "config.json").read_text())
    expected_layers = int(os.environ["GAUDIRT_EXPECT_LAYERS"])
    assert config["num_hidden_layers"] == expected_layers, (
        config["num_hidden_layers"],
        expected_layers,
    )
    vocab_size = int(config["vocab_size"])
    tp_size = int(os.environ.get("GAUDIRT_TP_SIZE", "2"))
    summary_path = Path(os.environ["GAUDIRT_SUMMARY_PATH"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    is_functional = _MODE == "functional_mixed"
    max_running = int(os.environ.get("GAUDIRT_MAX_RUNNING", "32"))
    max_total_tokens = int(os.environ.get("GAUDIRT_MAX_TOTAL_TOKENS", "36864"))
    context_len = int(os.environ.get("GAUDIRT_CONTEXT_LENGTH", "4096"))
    page_size = int(os.environ.get("GAUDIRT_TEST_PAGE_SIZE", "16"))
    chunk_size = int(os.environ.get("GAUDIRT_CONC_CHUNK_SIZE", "512"))
    assert max_total_tokens % page_size == 0

    engine_kwargs = {
        "model_path": str(model),
        "skip_tokenizer_init": not is_functional,
        "device": "gaudirt",
        "dtype": "bfloat16",
        "tp_size": tp_size,
        "max_total_tokens": max_total_tokens,
        "max_running_requests": max_running,
        "context_length": context_len,
        "mem_fraction_static": float(os.environ.get("GAUDIRT_STATIC_FRACTION", "0.85")),
        "page_size": page_size,
        "disable_cuda_graph": True,
        "disable_radix_cache": not is_functional,
        "disable_overlap_schedule": True,
        "chunked_prefill_size": chunk_size,
        "enable_mixed_chunk": True,
        "stream_interval": 1,
        "log_level": "warning",
    }
    if is_functional:
        engine_kwargs["grammar_backend"] = os.environ.get("GAUDIRT_GRAMMAR_BACKEND", "xgrammar")

    engine = Engine(**engine_kwargs)
    try:
        if is_functional:
            receipt = _run_functional_mixed(engine, model, tp_size, vocab_size, expected_layers, summary_path)
        elif _MODE == "throughput_bench":
            receipt = _run_throughput_bench(engine, tp_size, vocab_size, expected_layers)
        else:
            raise ValueError(f"Unsupported GAUDIRT_CONCURRENCY_MODE={_MODE}")
    finally:
        workers = _audit_and_shutdown(engine)

    receipt["workers"] = workers
    receipt["hl_smi_after"] = _query_hl_smi()
    summary_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print("CONCURRENCY_GATE_COMPLETE", _MODE, f"tp={tp_size}", f"layers={expected_layers}", flush=True)


if __name__ == "__main__":
    main()
