"""Multi-architecture TP=4 scaling verification (Qwen3 + Gemma2) on GaudiRT with zero models/*.py edits."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import gaudirt
import torch
from sglang.srt.managers.scheduler import Scheduler
from sglang_gaudirt.platform import GaudiRTSRTPlatform

_OBSERVED_MODES = set()
_CAPTURE_DIR = os.environ.get("GAUDIRT_TP4_LOGITS_DIR", "")

_orig_get_internal_state = Scheduler.get_internal_state
_orig_prepare_forward = GaudiRTSRTPlatform.prepare_model_forward


def _patched_internal_state(self, request):
    result = _orig_get_internal_state(self, request)
    from sglang.srt.distributed import get_tp_group

    tp_group = get_tp_group()
    tp_rank = int(tp_group.rank_in_group)
    tp_size = int(tp_group.world_size)
    dev_idx = int(torch.gaudirt.current_device())
    local_m = {
        "tp_rank": tp_rank,
        "device_index": dev_idx,
        "physical_module_id": int(gaudirt._C.visible_module_for_device(dev_idx)),
        "recipe_cache": list(gaudirt.recipe_cache_stats()),
        "observed_forward_modes": sorted(_OBSERVED_MODES),
    }
    cpu_group = tp_group.cpu_group if tp_size > 1 else None
    if tp_size > 1 and cpu_group is not None:
        gathered = [None] * tp_size
        torch.distributed.all_gather_object(gathered, local_m, group=cpu_group)
        result.internal_state["gaudirt_tp4_e2e_measurements"] = gathered
    else:
        result.internal_state["gaudirt_tp4_e2e_measurements"] = [local_m]
    return result


def _patched_prepare_forward(self, forward):
    compiled = _orig_prepare_forward(self, forward)

    def _wrapped(*args, **kwargs):
        batch = args[2]
        mode_name = getattr(getattr(batch, "forward_mode", None), "name", str(batch.forward_mode))
        _OBSERVED_MODES.add(mode_name)
        out = compiled(*args, **kwargs)
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        if (
            _CAPTURE_DIR
            and batch.forward_mode == ForwardMode.EXTEND
            and int(torch.gaudirt.current_device()) == 0
        ):
            cap_dir = Path(_CAPTURE_DIR)
            if not (cap_dir / "frozen").exists():
                logits_cpu = out.next_token_logits[-1:].detach().cpu().contiguous()
                assert torch.isfinite(logits_cpu).all(), "non-finite prefill logits"
                cap_dir.mkdir(parents=True, exist_ok=True)
                torch.save(logits_cpu, cap_dir / "prefill_0.pt")
        if hasattr(compiled, "_gaudirt_last_execution"):
            _wrapped._gaudirt_last_execution = compiled._gaudirt_last_execution
        return out

    return _wrapped


Scheduler.get_internal_state = _patched_internal_state
GaudiRTSRTPlatform.prepare_model_forward = _patched_prepare_forward


def _logit_metrics(ref_logits: torch.Tensor, cand_bf16: torch.Tensor) -> dict:
    ref = ref_logits.double().reshape(-1)
    cand = cand_bf16.double().reshape(-1)
    diff = cand - ref
    max_abs = float(diff.abs().max().item())
    rel_l2 = float(torch.linalg.vector_norm(diff).item() / max(torch.linalg.vector_norm(ref).item(), 1e-30))
    cos = float(torch.nn.functional.cosine_similarity(ref.unsqueeze(0), cand.unsqueeze(0)).item())
    log_p = torch.log_softmax(ref, dim=-1)
    log_q = torch.log_softmax(cand, dim=-1)
    p = log_p.exp()
    kl = float(torch.sum(p * (log_p - log_q)).clamp_min(0.0).item())
    return {
        "max_abs": max_abs,
        "rel_l2": rel_l2,
        "cosine_similarity": cos,
        "kl_ref_to_cand": kl,
        "top1_ref": int(ref.argmax().item()),
        "top1_cand": int(cand.argmax().item()),
        "top1_match": int(ref.argmax().item()) == int(cand.argmax().item()),
    }


def _collect_e2e_measurements(engine, tp_size: int):
    info = engine.get_server_info()
    found = {}

    def _walk(obj):
        if isinstance(obj, dict):
            if "gaudirt_tp4_e2e_measurements" in obj:
                for item in obj["gaudirt_tp4_e2e_measurements"]:
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
    for name, proc in workers:
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


def _build_qwen3_fixture(model_dir: Path, seed: int = 4103):
    from transformers import AutoModelForCausalLM, Qwen3Config

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=1024,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        rms_norm_eps=1e-6,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name:
                param.copy_(1.0 + 0.15 * torch.randn_like(param))
    model = model.to(torch.bfloat16)
    model.save_pretrained(str(model_dir), safe_serialization=True)
    return int(cfg.vocab_size)


def _build_gemma2_fixture(model_dir: Path, seed: int = 5207):
    from transformers import AutoModelForCausalLM, Gemma2Config

    torch.manual_seed(seed)
    cfg = Gemma2Config(
        vocab_size=1024,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=256,
        query_pre_attn_scalar=256,
        sliding_window=32,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
        hidden_act="gelu_pytorch_tanh",
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name:
                param.copy_(0.15 * torch.randn_like(param))
    model = model.to(torch.bfloat16)
    model.save_pretrained(str(model_dir), safe_serialization=True)
    return int(cfg.vocab_size)


def _cpu_ref_last_logits(model_dir: Path, prompt_ids: list) -> torch.Tensor:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False
    ).float().eval()
    with torch.no_grad():
        x = torch.tensor([prompt_ids], dtype=torch.long)
        out = model(input_ids=x, use_cache=False)
        logits = out.logits[0, -1, :].detach().cpu().contiguous()
    del model
    return logits


def _trigger_mixed_batching(engine, vocab_size: int, base_seed: int):
    p1 = [3 + ((base_seed + i * 17) % (vocab_size - 4)) for i in range(40)]
    p2 = [4 + ((base_seed + i * 29) % (vocab_size - 5)) for i in range(96)]
    p3 = [5 + ((base_seed + i * 43) % (vocab_size - 6)) for i in range(96)]

    async def _run_all():
        t1 = asyncio.create_task(engine.async_generate(
            input_ids=p1,
            sampling_params={"temperature": 0, "max_new_tokens": 64, "ignore_eos": True},
        ))
        await asyncio.sleep(0.005)
        t2 = asyncio.create_task(engine.async_generate(
            input_ids=p2,
            sampling_params={"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
        ))
        await asyncio.sleep(0.005)
        t3 = asyncio.create_task(engine.async_generate(
            input_ids=p3,
            sampling_params={"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
        ))
        r1, r2, r3 = await asyncio.gather(t1, t2, t3)
        return {"decode_lead": r1["output_ids"], "chunk_b": r2["output_ids"], "chunk_c": r3["output_ids"]}

    return engine.loop.run_until_complete(_run_all())


def _run_child_case(case_name: str, model_dir: Path, tp_size: int, prompt_len: int,
                    work_dir: Path, vocab_size: int) -> dict:
    from sglang.srt.entrypoints.engine import Engine

    logits_dir = work_dir / f"{case_name}_logits"
    ref_path = work_dir / f"{case_name}_ref_logits.pt"
    out_json = work_dir / f"{case_name}_result.json"

    prompt = [3 + (i * 37) % (vocab_size - 4) for i in range(prompt_len)]
    ref_logits = _cpu_ref_last_logits(model_dir, prompt)
    torch.save(ref_logits, ref_path)

    os.environ["GAUDIRT_TP4_LOGITS_DIR"] = str(logits_dir)
    os.environ["GAUDIRT_TP4_CAPTURE_LIMIT"] = "1"
    engine = Engine(
        model_path=str(model_dir),
        skip_tokenizer_init=True,
        device="gaudirt",
        dtype="bfloat16",
        tp_size=tp_size,
        max_total_tokens=2048,
        max_running_requests=4,
        context_length=2048,
        page_size=16,
        mem_fraction_static=0.5,
        disable_cuda_graph=True,
        disable_radix_cache=False,
        disable_overlap_schedule=True,
        chunked_prefill_size=32,
        enable_mixed_chunk=True,
        log_level="warning",
    )
    try:
        greedy_prompt_res = engine.generate(
            input_ids=prompt,
            sampling_params={"temperature": 0, "max_new_tokens": 16, "ignore_eos": True},
        )
        (logits_dir / "frozen").write_text("frozen\n")
        greedy_ids = greedy_prompt_res["output_ids"]
        assert len(greedy_ids) == 16 and all(0 <= t < vocab_size for t in greedy_ids)

        cap_file = logits_dir / "prefill_0.pt"
        assert cap_file.is_file(), f"missing captured prefill logits for {case_name}"
        cand_logits = torch.load(cap_file, weights_only=True)
        hf_metrics = _logit_metrics(ref_logits, cand_logits)
        assert hf_metrics["kl_ref_to_cand"] <= 0.01, (case_name, hf_metrics)
        assert hf_metrics["cosine_similarity"] >= 0.999, (case_name, hf_metrics)
        assert hf_metrics["top1_match"], (case_name, hf_metrics)

        prompt2 = [7 + (i * 41) % (vocab_size - 8) for i in range(prompt_len)]
        _ = engine.generate(
            input_ids=prompt2,
            sampling_params={"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
        )
        before_warm = _collect_e2e_measurements(engine, tp_size)
        prompt3 = [9 + (i * 53) % (vocab_size - 10) for i in range(prompt_len)]
        _ = engine.generate(
            input_ids=prompt3,
            sampling_params={"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
        )
        after_warm = _collect_e2e_measurements(engine, tp_size)
        for b_stat, a_stat in zip(before_warm, after_warm):
            assert b_stat["recipe_cache"][1] == a_stat["recipe_cache"][1], (
                f"{case_name} recompiled on warm identical shape: {b_stat['recipe_cache']} -> {a_stat['recipe_cache']}"
            )
            assert b_stat["recipe_cache"][2] == a_stat["recipe_cache"][2], (
                f"{case_name} recipe count grew on warm run: {b_stat['recipe_cache']} -> {a_stat['recipe_cache']}"
            )

        mixed_ok = False
        modes = set()
        for attempt in range(3):
            _trigger_mixed_batching(engine, vocab_size, base_seed=100 + attempt * 1000)
            mid_stats = _collect_e2e_measurements(engine, tp_size)
            modes = set().union(*(m["observed_forward_modes"] for m in mid_stats))
            if "MIXED" in modes:
                mixed_ok = True
                break
        assert mixed_ok, f"{case_name}: failed to observe ForwardMode.MIXED, saw {modes}"

        sample_prompt_a = [11 + (i * 61) % (vocab_size - 12) for i in range(32)]
        sample_prompt_b = [13 + (i * 67) % (vocab_size - 14) for i in range(32)]
        sp = {
            "temperature": 0.8,
            "top_k": 16,
            "top_p": 0.9,
            "min_p": 0.05,
            "sampling_seed": 1234,
            "max_new_tokens": 12,
            "ignore_eos": True,
        }
        s1 = engine.generate(input_ids=sample_prompt_a, sampling_params=sp)["output_ids"]
        s2 = engine.generate(input_ids=sample_prompt_a, sampling_params=sp)["output_ids"]
        sp_other = dict(sp, sampling_seed=5678)
        s3 = engine.generate(input_ids=sample_prompt_b, sampling_params=sp_other)["output_ids"]
        assert len(s1) == 12 and s1 == s2, f"{case_name}: seeded stochastic sampling not reproducible: {s1} vs {s2}"
        assert all(0 <= t < vocab_size for t in s1 + s3)

        final_stats = _collect_e2e_measurements(engine, tp_size)
    finally:
        _audit_and_shutdown(engine)

    result = {
        "case": case_name,
        "tp_size": tp_size,
        "prompt_len": prompt_len,
        "vocab_size": vocab_size,
        "hf_prefill_metrics": hf_metrics,
        "greedy_output_ids": greedy_ids,
        "sampler_seeded_ids": s1,
        "sampler_alt_seed_ids": s3,
        "sampler_seed_reproducible": True,
        "warm_compiles_before": [m["recipe_cache"][1] for m in before_warm],
        "warm_compiles_after": [m["recipe_cache"][1] for m in after_warm],
        "observed_forward_modes": sorted(set().union(*(m["observed_forward_modes"] for m in final_stats))),
        "physical_modules": [m["physical_module_id"] for m in final_stats],
        "prefill_logits_path": str(cap_file),
    }
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "MULTI_ARCH_TP4_CASE_PASS",
        case_name,
        json.dumps(
            {
                "kl": round(hf_metrics["kl_ref_to_cand"], 6),
                "cos": round(hf_metrics["cosine_similarity"], 6),
                "top1_match": hf_metrics["top1_match"],
                "modes": result["observed_forward_modes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def _spawn_case(case_name: str, model_dir: Path, tp_size: int, prompt_len: int,
                work_dir: Path, vocab_size: int, visible_modules: str):
    env = dict(os.environ)
    env["HABANA_VISIBLE_MODULES"] = visible_modules
    env["GAUDIRT_RECIPE_CACHE"] = str(work_dir / f"cache_{case_name}")
    cmd = [
        sys.executable,
        __file__,
        "--child",
        case_name,
        str(model_dir),
        str(tp_size),
        str(prompt_len),
        str(work_dir),
        str(vocab_size),
    ]
    subprocess.run(cmd, env=env, check=True)
    return json.loads((work_dir / f"{case_name}_result.json").read_text())


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--child":
        _, _, case_name, model_dir, tp_size, prompt_len, work_dir, vocab_size = sys.argv
        _run_child_case(case_name, Path(model_dir), int(tp_size), int(prompt_len), Path(work_dir), int(vocab_size))
        return

    receipt_path = Path(os.environ["GAUDIRT_MULTI_ARCH_TP4_RECEIPT"])
    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    visible = [x.strip() for x in os.environ.get("HABANA_VISIBLE_MODULES", "0,1,2,3").split(",") if x.strip()]
    assert len(visible) >= 4, f"multi_arch_tp4_smoke requires at least 4 visible modules, got {visible}"
    tp1_mod = visible[0]
    tp4_mods = ",".join(visible[:4])

    with tempfile.TemporaryDirectory(prefix="gaudirt_multi_arch_tp4_") as tmp:
        work_dir = Path(tmp)
        qwen_dir = work_dir / "qwen3_4layer"
        gemma_dir = work_dir / "gemma2_4layer"
        qwen_dir.mkdir(parents=True, exist_ok=True)
        gemma_dir.mkdir(parents=True, exist_ok=True)
        qwen_vocab = _build_qwen3_fixture(qwen_dir)
        gemma_vocab = _build_gemma2_fixture(gemma_dir)

        qwen_tp1 = _spawn_case("qwen3_4l_tp1", qwen_dir, 1, 48, work_dir, qwen_vocab, tp1_mod)
        qwen_tp4 = _spawn_case("qwen3_4l_tp4", qwen_dir, 4, 48, work_dir, qwen_vocab, tp4_mods)
        qwen_tp_kl = _logit_metrics(
            torch.load(qwen_tp1["prefill_logits_path"], weights_only=True),
            torch.load(qwen_tp4["prefill_logits_path"], weights_only=True),
        )
        assert qwen_tp_kl["kl_ref_to_cand"] <= 0.01, qwen_tp_kl
        assert qwen_tp1["greedy_output_ids"] == qwen_tp4["greedy_output_ids"], (
            qwen_tp1["greedy_output_ids"], qwen_tp4["greedy_output_ids"]
        )

        # Gemma2: 2 KV heads replicated across TP=4, prompt_len=96 crosses sliding_window=32 by 3x.
        gemma_tp1 = _spawn_case("gemma2_4l_tp1", gemma_dir, 1, 96, work_dir, gemma_vocab, tp1_mod)
        gemma_tp4 = _spawn_case("gemma2_4l_tp4", gemma_dir, 4, 96, work_dir, gemma_vocab, tp4_mods)
        gemma_tp_kl = _logit_metrics(
            torch.load(gemma_tp1["prefill_logits_path"], weights_only=True),
            torch.load(gemma_tp4["prefill_logits_path"], weights_only=True),
        )
        assert gemma_tp_kl["kl_ref_to_cand"] <= 0.01, gemma_tp_kl
        assert gemma_tp1["greedy_output_ids"] == gemma_tp4["greedy_output_ids"], (
            gemma_tp1["greedy_output_ids"], gemma_tp4["greedy_output_ids"]
        )

        all_cases = [qwen_tp1, qwen_tp4, gemma_tp1, gemma_tp4]
        for rec in all_cases:
            rec.pop("prefill_logits_path", None)

        receipt = {
            "upstream_models_py_untouched": True,
            "cases": all_cases,
            "tp_parity": {
                "qwen3_4l_tp4": {"greedy_exact_match": True, "prefill_tp1_vs_tp4": qwen_tp_kl},
                "gemma2_4l_tp4": {"greedy_exact_match": True, "prefill_tp1_vs_tp4": gemma_tp_kl},
            },
        }
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print("MULTI_ARCH_TP4_ALL_PASSED", flush=True)


if __name__ == "__main__":
    main()
