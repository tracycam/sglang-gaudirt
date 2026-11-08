"""Prepare an external two-layer configuration backed by original weight shards."""

import json
import sys
from pathlib import Path


def main(source, destination):
    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    config = json.loads((source / "config.json").read_text())
    assert config["num_hidden_layers"] >= 2
    config["num_hidden_layers"] = 2
    index = json.loads((source / "model.safetensors.index.json").read_text())
    prefixes = ("model.layers.0.", "model.layers.1.")
    globals_ = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    weights = {name: shard for name, shard in index["weight_map"].items()
               if name.startswith(prefixes) or name in globals_}
    suffixes = (
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    required = globals_ | {f"model.layers.{layer}.{suffix}" for layer in range(2)
                           for suffix in suffixes}
    assert required <= weights.keys(), sorted(required - weights.keys())
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights}, indent=2, sort_keys=True) + "\n"
    )
    for shard in sorted(set(weights.values())):
        (destination / shard).symlink_to((source / shard).resolve(strict=True))
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "added_tokens.json"):
        if (source / name).exists():
            (destination / name).symlink_to((source / name).resolve(strict=True))
    print(f"two-layer fixture: {len(weights)} original weights, {len(set(weights.values()))} shards")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: create_two_layer_fixture.py SOURCE_MODEL DESTINATION")
    main(*sys.argv[1:])
