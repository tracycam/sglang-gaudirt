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
empty `GAUDIRT_FEATURE_REPLAY_DIR`. The script compares the M=1 logits with the
last row of the 1024+137 chunked replay byte for byte. Run this separately from
the full-pool trace. Unchunked versus chunked uses different arithmetic trees;
its validation is the per-layer reference and conditioned KL gate, not a
greedy-token equality assertion.
