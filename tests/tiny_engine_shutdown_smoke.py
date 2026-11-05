"""Start an actual two-layer SGLang Llama with a small generated fixture."""

import json
import os
import sys
from pathlib import Path

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert os.environ.get("SGLANG_PLATFORM") == "gaudirt"
assert "habana_frameworks" not in sys.modules

import torch
import sglang
import gaudirt.compiler as compiler
from safetensors.torch import save_file

original_backend = compiler.gaudirt_backend
captures = 0


def single_capture(graph, inputs):
    global captures
    captures += 1
    if captures != 1:
        raise AssertionError("model forward captured more than once")
    return original_backend(graph, inputs)


compiler.gaudirt_backend = single_capture


def prepare(path):
    path.mkdir(parents=True, exist_ok=True)
    config = dict(
        architectures=["LlamaForCausalLM"], model_type="llama",
        hidden_size=128, intermediate_size=256, num_attention_heads=2,
        num_key_value_heads=1, head_dim=64, num_hidden_layers=2,
        vocab_size=128, max_position_embeddings=1024, rms_norm_eps=1e-6,
        hidden_act="silu", torch_dtype="bfloat16", tie_word_embeddings=False,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    (path / "config.json").write_text(json.dumps(config))
    generator = torch.Generator(device="cpu").manual_seed(24719)

    def weight(*shape):
        return (torch.randn(shape, generator=generator, dtype=torch.float32) * 0.04).to(torch.bfloat16)

    weights = {
        "model.embed_tokens.weight": weight(128, 128),
        "model.norm.weight": torch.ones(128, dtype=torch.bfloat16),
        "lm_head.weight": weight(128, 128),
    }
    for layer in range(2):
        name = f"model.layers.{layer}."
        weights[name + "input_layernorm.weight"] = torch.ones(128, dtype=torch.bfloat16)
        weights[name + "post_attention_layernorm.weight"] = torch.ones(128, dtype=torch.bfloat16)
        for suffix, shape in (
            ("self_attn.q_proj.weight", (128, 128)),
            ("self_attn.k_proj.weight", (64, 128)),
            ("self_attn.v_proj.weight", (64, 128)),
            ("self_attn.o_proj.weight", (128, 128)),
            ("mlp.gate_proj.weight", (256, 128)),
            ("mlp.up_proj.weight", (256, 128)),
            ("mlp.down_proj.weight", (128, 256)),
        ):
            weights[name + suffix] = weight(*shape)
    save_file(weights, str(path / "model.safetensors"))


def main():
    path = Path(os.environ["GAUDIRT_TINY_MODEL"])
    prepare(path)
    from sglang.srt.entrypoints.engine import Engine

    print("Starting SGLang tiny two-layer engine", flush=True)
    engine = Engine(
        model_path=str(path), skip_tokenizer_init=True, device="gaudirt",
        dtype="bfloat16", tp_size=1, max_total_tokens=512,
        max_running_requests=2, context_length=512, mem_fraction_static=0.5,
        disable_cuda_graph=True, disable_radix_cache=True,
        disable_overlap_schedule=True, chunked_prefill_size=-1, log_level="info",
    )
    try:
        for rows in (1, 19, 137):
            result = engine.generate(
                input_ids=[3 + i % 125 for i in range(rows)],
                sampling_params=dict(temperature=0, max_new_tokens=3, ignore_eos=True),
            )
            assert len(result["output_ids"]) == 3, result
            print("Tiny SGLang dynamic prompt passed", rows, result["output_ids"], flush=True)
    finally:
        watchdog = engine.tokenizer_manager._subprocess_watchdog
        workers = list(zip(watchdog._names, watchdog._processes))
        for name, process in workers:
            with open(f"/proc/{process.pid}/maps", encoding="utf-8") as maps:
                mapped = maps.read().lower()
            assert not any(x in mapped for x in ("libhabana_pytorch", "gaudi-pytorch-bridge", "habana_frameworks"))
            print("Worker audited before shutdown", name, process.pid, flush=True)
        engine.shutdown()
        for name, process in workers:
            assert process.exitcode == 0, (name, process.exitcode)
            print("Worker exited cleanly", name, process.pid, process.exitcode, flush=True)
        engine.shutdown()
        assert not any(name == "habana_frameworks" or name.startswith("habana_frameworks.")
                       for name in sys.modules)


if __name__ == "__main__":
    main()
