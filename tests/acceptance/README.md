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

The plain measurement uses pretokenized `Engine.generate` inputs and excludes
tokenization and HTTP. Its process-isolation audit runs at process entry and
exit; no per-step state audit or profiler is enabled during timing. Each prompt
is warmed for the complete output length before its measured request. The CPU
fault test for the state audit runs with
`python tests/acceptance/test_phase5a_state_audit.py`.

For the 52-layer prefix-cache and chunked-prefill comparison, set the following
before the plain measurement so it writes the matching uncached token outputs;
these requests run outside the timing windows. Then run the enabled service
check with the same warm recipe cache:

```sh
export GAUDIRT_FEATURE_PREFIX=1024 GAUDIRT_FEATURE_SUFFIX=137
export GAUDIRT_FEATURE_REFERENCE_OUT="$ARTIFACTS/features-plain.json"
# Run sglang_phase5a_plain.py as above, then:
GAUDIRT_FEATURE_MODE=enabled GAUDIRT_TEST_PAGE_SIZE=16 \
GAUDIRT_FEATURE_CONTEXT=32768 GAUDIRT_FEATURE_MAX_TOKENS=4608 \
GAUDIRT_FEATURE_CHUNK_SIZE=1024 GAUDIRT_FEATURE_MIN_CACHED=1024 \
GAUDIRT_EXPECT_WARM=1 GAUDIRT_FEATURE_REFERENCE="$ARTIFACTS/features-plain.json" \
GAUDIRT_FEATURE_RECEIPT="$ARTIFACTS/features-enabled.json" \
GAUDIRT_SITE_AUDIT_LOG="$ARTIFACTS/features-process.jsonl" \
  python tests/acceptance/sglang_features_smoke.py
```
