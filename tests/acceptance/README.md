# Single-device SGLang acceptance tools

These scripts reproduce the fixed-prefix numerical check, per-step KV state
audit, cold/warm comparison, process isolation check, and uninstrumented
streaming measurement. The model, compiled recipes, logs, traces and JSON
receipts must be kept outside the source repositories. The full state audit
copies the entire KV pool after each forward and can take hours; it is not a
performance measurement.

Install the plugin and its matching GaudiRT build and SGLang checkout. Run from
the plugin repository root with `MODEL_DIR`, `ARTIFACTS`, `GAUDIRT_TESTS` (the
GaudiRT `tests` directory), `OTHER_PUBLIC_DOCUMENT`, and `GC_KERNEL_PATH` set to
existing locations. Use a fresh artifact directory, an empty recipe cache, and
an idle card.

```sh
set -eu
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 SGLANG_PLATFORM=gaudirt
export PYTHONPATH="$PWD/tests/acceptance/startup:$PWD/tests/acceptance:$GAUDIRT_TESTS${PYTHONPATH:+:$PYTHONPATH}"
export GAUDIRT_TARGET_MODEL="$MODEL_DIR" GAUDIRT_EXPECT_LAYERS=52
export GAUDIRT_PROMPTS=137,1053,4096 GAUDIRT_DECODE_STEPS=128
export GAUDIRT_MAX_TOTAL_TOKENS=4608 GAUDIRT_CONTEXT_LENGTH=4608
export GAUDIRT_HBM_BYTES=85899345920 GAUDIRT_STATIC_FRACTION=0.9
export GAUDIRT_TEXT_MODE=1
export GAUDIRT_PROMPT_DOCUMENTS="$MODEL_DIR/README.md:$OTHER_PUBLIC_DOCUMENT"
export GAUDIRT_RECIPE_CACHE="$ARTIFACTS/cache"
export GAUDIRT_TRACE_PATH="$ARTIFACTS/trace.bin"

for mode in cold warm; do
  GAUDIRT_AUDIT_MODE="$mode" \
  GAUDIRT_SITE_AUDIT_LOG="$ARTIFACTS/$mode-process.jsonl" \
  GAUDIRT_SUMMARY_PATH="$ARTIFACTS/$mode.json" \
  GAUDIRT_TOKEN_PATH="$ARTIFACTS/$mode-tokens.json" \
  GAUDIRT_LAYER_CAPTURE_PATH="$ARTIFACTS/$mode-capture.pt" \
  python tests/acceptance/sglang_phase5a_acceptance.py
done

GAUDIRT_LAYER_CAPTURE_PATH="$ARTIFACTS/cold-capture.pt" \
GAUDIRT_LAYER_RECEIPT="$ARTIFACTS/numerical.json" \
  python tests/acceptance/layer_fp64_multitree_audit.py

GAUDIRT_REFERENCE_TOKENS="$ARTIFACTS/cold-tokens.json" \
GAUDIRT_PLAIN_SUMMARY="$ARTIFACTS/plain.json" \
GAUDIRT_SITE_AUDIT_LOG="$ARTIFACTS/plain-process.jsonl" \
  python tests/acceptance/sglang_phase5a_plain.py

python tests/acceptance/verify_process_audit.py "$ARTIFACTS/process-audit.json" \
  "$ARTIFACTS/cold-process.jsonl" "$ARTIFACTS/warm-process.jsonl" \
  "$ARTIFACTS/plain-process.jsonl"
```

For a captured online prefix longer than 512 tokens, set
`GAUDIRT_AUDIT_PV_TILE` to the actual KV tile used by that attention path.
The BF16-PV reference rounds each tile's probability input before its P@V
and merges tiles in FP64; the main CPU FP32 reference keeps probabilities in
FP32. A long-prefix reference without an explicit tile fails immediately.

The plain measurement uses pretokenized `Engine.generate` inputs and excludes
tokenization and HTTP. Its process-isolation audit runs at process entry and
exit; no per-step state audit or profiler is enabled during timing. Each prompt
is warmed for the complete output length before its measured request. The CPU
fault test for the state audit runs with
`python tests/acceptance/test_phase5a_state_audit.py`.

For a block-FP8 checkpoint, install the matching kernel Python package as well
as the plugin's `fp8` extra. Set `GAUDIRT_CAPTURE_FP8_STAGES=1` for the state
capture, then use `fp8_layer_fp64_audit.py` in place of the BF16 layer audit.
The capture exposes each projection's input and output in a stateless replay
and verifies the original segment output is byte-identical. The audit checks
each frozen projection and both frozen layer stages against original OCP
weights and standard W8A8 activations, as well as the full-layer same-boundary
and conditioned KL gates. Pure FP64 and unquantized BF16 activation distances
are separate diagnostics. A failed layer or KL check exits nonzero and retains
the receipt. Run
`python tests/acceptance/test_fp8_layer_reference.py` for the CPU reference
contract before auditing a capture.

For the prefix-cache comparison, both arms use the same context, page size and
chunking. The second request shares 1024 tokens. The check compares both token
sequences and every byte of each request's first-step logits; a reference with
different geometry or chunking is rejected. These checks are outside timing
windows.

```sh
export GAUDIRT_FEATURE_PREFIX=1024 GAUDIRT_FEATURE_SUFFIX=137
export GAUDIRT_FEATURE_MODE=enabled GAUDIRT_TEST_PAGE_SIZE=16
export GAUDIRT_FEATURE_CONTEXT=32768 GAUDIRT_FEATURE_MAX_TOKENS=4608
export GAUDIRT_FEATURE_CHUNK_SIZE=1024 GAUDIRT_FEATURE_MIN_CACHED=1024
GAUDIRT_FEATURE_RADIX=0 GAUDIRT_FEATURE_LOGITS_DIR="$ARTIFACTS/off-logits" \
GAUDIRT_FEATURE_RECEIPT="$ARTIFACTS/features-off.json" \
GAUDIRT_SITE_AUDIT_LOG="$ARTIFACTS/features-off-process.jsonl" \
  python tests/acceptance/sglang_features_smoke.py

GAUDIRT_FEATURE_RADIX=1 GAUDIRT_FEATURE_LOGITS_DIR="$ARTIFACTS/on-logits" \
GAUDIRT_FEATURE_REFERENCE="$ARTIFACTS/features-off.json" \
GAUDIRT_FEATURE_LOGITS_REFERENCE="$ARTIFACTS/off-logits" \
GAUDIRT_FEATURE_RECEIPT="$ARTIFACTS/features-on.json" \
GAUDIRT_SITE_AUDIT_LOG="$ARTIFACTS/features-on-process.jsonl" \
  python tests/acceptance/sglang_features_smoke.py
```

For same-configuration cold/warm full-pool checking, use a fresh recipe cache
and set `GAUDIRT_FEATURE_STATE_AUDIT=1`, `GAUDIRT_FEATURE_EXPECT_STEPS=9`,
`GAUDIRT_AUDIT_MODE=cold` then `warm`, `GAUDIRT_TRACE_PATH` to the same trace,
and `GAUDIRT_FEATURE_STATE_RECEIPT` and `GAUDIRT_FEATURE_LOGITS_DIR` to distinct
paths per process. Set `GAUDIRT_EXPECT_WARM=1` only on the warm run. The warm
run compares every recorded KV byte and logit with the cold trace.

For the chunked same-prefix replay, run with `GAUDIRT_FEATURE_RADIX=0` and an
empty `GAUDIRT_FEATURE_REPLAY_DIR`. The script reports the M=1 and chunked
last-row difference; this diagnostic alone is not a numerical acceptance.
Set `GAUDIRT_FEATURE_REPLAY_REQUIRE_EXACT=1` only when both requests use the
same arithmetic path. Run separately from the full-pool trace. Different
arithmetic trees require the per-layer reference and conditioned KL gate.

To capture the chunked replay for the per-layer reference, run the main state
audit with `GAUDIRT_FIXED_PREFIX_LENGTH=1160`, `GAUDIRT_AUDIT_CHUNK_SIZE=1024`,
`GAUDIRT_PROMPTS=''` and `GAUDIRT_DECODE_STEPS=0`. For full-row Decode versus
512-key tiled prefill, also set `GAUDIRT_AUDIT_CROSS_TREE=1` and provide a
separate `GAUDIRT_REPLAY_LAYER_CAPTURE_PATH`. Run the independent FP64 audit on
the original capture with `GAUDIRT_AUDIT_PV_TILE=full`, and on the replay
capture with `GAUDIRT_AUDIT_PV_TILE=512`. Each must satisfy the full per-layer
and conditioned KL gates. The primary FP32 reference keeps attention
probabilities in FP32. Without cross-tree mode, the main audit still requires
byte-identical same-path outputs.

## Phase 5B Serving Closeout (`TP=4`, Concurrency, Structured Output & Multi-Arch)

Unit-level standalone smokes in `$GAUDIRT_TESTS` verify dynamic FX/eager kernels,
vocab bitmask, and `TP=4` AllGather+RankSum primitives before running SGLang
end-to-end gates:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 $GAUDIRT_TESTS/dynamic_eager_ops_smoke.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 $GAUDIRT_TESTS/vocab_bitmask_device_smoke.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 $GAUDIRT_TESTS/dynamic_gather_rows_fx_smoke.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 HABANA_VISIBLE_MODULES=0,1,2,3 python3 $GAUDIRT_TESTS/tp4_gather_reduce_smoke.py
```

### Structured Output (`xgrammar` / `llguidance`) and Multi-Arch `TP=4` Parity

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 SGLANG_PLATFORM=gaudirt \
HABANA_VISIBLE_MODULES=0,1 \
GAUDIRT_SMOLLM2_MODEL=$SMOLLM2_DIR \
GAUDIRT_NANBEIGE_2L_MODEL=$NANBEIGE_2L_DIR \
GAUDIRT_STRUCTURED_RECEIPT=$ARTIFACTS/structured_output_receipt.json \
python3 tests/acceptance/sglang_structured_output_smoke.py

TORCH_DEVICE_BACKEND_AUTOLOAD=0 SGLANG_PLATFORM=gaudirt \
HABANA_VISIBLE_MODULES=0,1,2,3 \
GAUDIRT_MULTI_ARCH_TP4_RECEIPT=$ARTIFACTS/multi_arch_tp4_receipt.json \
python3 tests/acceptance/sglang_multi_arch_tp4_smoke.py
```

### 52L `TP=1` vs `TP=4` Deterministic Text-Mode Gate

With W8A8 FP8 weights, pass natural-text documents via `GAUDIRT_TEXT_MODE=1` and
`GAUDIRT_PROMPT_DOCUMENTS` (see `P-BENCH-18` for BOS attention-sink sensitivity
on synthetic BOS-less token IDs) and set `GAUDIRT_HBM_BYTES=85899345920`:

```bash
COMMON_ENV="TORCH_DEVICE_BACKEND_AUTOLOAD=0 SGLANG_PLATFORM=gaudirt \
GAUDIRT_TARGET_MODEL=$MODEL_DIR GAUDIRT_EXPECT_LAYERS=52 \
GAUDIRT_HBM_BYTES=85899345920 GAUDIRT_TEXT_MODE=1 \
GAUDIRT_PROMPT_DOCUMENTS=$MODEL_DIR/README.md:$OTHER_PUBLIC_DOCUMENT"

env $COMMON_ENV HABANA_VISIBLE_MODULES=0 GAUDIRT_TP_GATE_MODE=reference_tp1 \
  GAUDIRT_TP_LOGITS_DIR=$ARTIFACTS/tp1_logits \
  GAUDIRT_SUMMARY_PATH=$ARTIFACTS/nanbeige52_tp1_ref.json \
  python3 tests/acceptance/sglang_phase5b_tp2_gate.py

env $COMMON_ENV HABANA_VISIBLE_MODULES=0,1,2,3 GAUDIRT_TP_SIZE=4 \
  GAUDIRT_TP_GATE_MODE=tp2_cold \
  GAUDIRT_TP_LOGITS_DIR=$ARTIFACTS/tp4_cold_logits \
  GAUDIRT_TP1_LOGITS_DIR=$ARTIFACTS/tp1_logits \
  GAUDIRT_TP1_SUMMARY=$ARTIFACTS/nanbeige52_tp1_ref.json \
  GAUDIRT_SUMMARY_PATH=$ARTIFACTS/nanbeige52_tp4_cold.json \
  python3 tests/acceptance/sglang_phase5b_tp2_gate.py

env $COMMON_ENV HABANA_VISIBLE_MODULES=0,1,2,3 GAUDIRT_TP_SIZE=4 \
  GAUDIRT_TP_GATE_MODE=tp2_warm \
  GAUDIRT_COLD_SUMMARY=$ARTIFACTS/nanbeige52_tp4_cold.json \
  GAUDIRT_COLD_LOGITS_DIR=$ARTIFACTS/tp4_cold_logits \
  GAUDIRT_TP_LOGITS_DIR=$ARTIFACTS/tp4_warm_logits \
  GAUDIRT_TP1_LOGITS_DIR=$ARTIFACTS/tp1_logits \
  GAUDIRT_TP1_SUMMARY=$ARTIFACTS/nanbeige52_tp1_ref.json \
  GAUDIRT_SUMMARY_PATH=$ARTIFACTS/nanbeige52_tp4_warm.json \
  python3 tests/acceptance/sglang_phase5b_tp2_gate.py

env $COMMON_ENV HABANA_VISIBLE_MODULES=0,1,2,3 GAUDIRT_TP_SIZE=4 \
  GAUDIRT_TP_GATE_MODE=tp2_features \
  GAUDIRT_SUMMARY_PATH=$ARTIFACTS/nanbeige52_tp4_features.json \
  python3 tests/acceptance/sglang_phase5b_tp2_gate.py

env $COMMON_ENV HABANA_VISIBLE_MODULES=0,1,2,3 GAUDIRT_TP_SIZE=4 \
  GAUDIRT_TP_GATE_MODE=tp2_bench \
  GAUDIRT_SUMMARY_PATH=$ARTIFACTS/nanbeige52_tp4_bench.json \
  python3 tests/acceptance/sglang_phase5b_tp2_gate.py
```

### Continuous Batching Mixed Functional & `B=1/8/16/32` Concurrency Gate

```bash
# TP=2 throughput benchmark
TORCH_DEVICE_BACKEND_AUTOLOAD=0 SGLANG_PLATFORM=gaudirt \
HABANA_VISIBLE_MODULES=0,1 GAUDIRT_TP_SIZE=2 \
GAUDIRT_TARGET_MODEL=$MODEL_DIR GAUDIRT_EXPECT_LAYERS=52 \
GAUDIRT_CONCURRENCY_MODE=throughput_bench \
GAUDIRT_SUMMARY_PATH=$ARTIFACTS/nanbeige52_tp2_concurrency_bench.json \
python3 tests/acceptance/sglang_phase5b_concurrency_gate.py

# TP=4 functional mixed-batch (canaries + xgrammar JSON/regex + radix reuse)
TORCH_DEVICE_BACKEND_AUTOLOAD=0 SGLANG_PLATFORM=gaudirt \
HABANA_VISIBLE_MODULES=0,1,2,3 GAUDIRT_TP_SIZE=4 \
GAUDIRT_TARGET_MODEL=$MODEL_DIR GAUDIRT_EXPECT_LAYERS=52 \
GAUDIRT_CONCURRENCY_MODE=functional_mixed \
GAUDIRT_SUMMARY_PATH=$ARTIFACTS/nanbeige52_tp4_concurrency_functional.json \
python3 tests/acceptance/sglang_phase5b_concurrency_gate.py

# TP=4 throughput benchmark (B=1,8,16,32 with zero steady-state recompiles)
TORCH_DEVICE_BACKEND_AUTOLOAD=0 SGLANG_PLATFORM=gaudirt \
HABANA_VISIBLE_MODULES=0,1,2,3 GAUDIRT_TP_SIZE=4 \
GAUDIRT_TARGET_MODEL=$MODEL_DIR GAUDIRT_EXPECT_LAYERS=52 \
GAUDIRT_CONCURRENCY_MODE=throughput_bench \
GAUDIRT_SUMMARY_PATH=$ARTIFACTS/nanbeige52_tp4_concurrency_bench.json \
python3 tests/acceptance/sglang_phase5b_concurrency_gate.py
```

