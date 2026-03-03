#!/bin/bash
# Download and prepare AIME 2024, AIME 2025, HMMT 2025, Amo-Bench as eval test sets.
# Output format: JSONL with prompt, label, metadata (same as prepare_aime.py)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/data/siqizhu4/lhy/data}"
SEED="${SEED:-42}"

# Optional: use China mirror
# export HF_ENDPOINT=https://hf-mirror.com

echo "=== Preparing eval test sets (output: $OUTPUT_DIR) ==="

# AIME 2024 (30 problems)
python "$SCRIPT_DIR/prepare_aime.py" --mode eval --num-samples 30 \
    --output "$OUTPUT_DIR/aime2024_eval.jsonl" --seed "$SEED"

# AIME 2025 (30 problems)
python "$SCRIPT_DIR/prepare_aime2025.py" --mode eval --num-samples 30 \
    --output "$OUTPUT_DIR/aime2025_eval.jsonl" --seed "$SEED"

# HMMT 2025 (30 problems)
python "$SCRIPT_DIR/prepare_hmmt2025.py" --mode eval --num-samples 30 \
    --output "$OUTPUT_DIR/hmmt2025_eval.jsonl" --seed "$SEED"

# Amo-Bench (50 problems)
python "$SCRIPT_DIR/prepare_amo_bench.py" --mode eval --num-samples 50 \
    --output "$OUTPUT_DIR/amo_bench_eval.jsonl" --seed "$SEED"

echo ""
echo "=== Done. Eval files: ==="
ls -la "$OUTPUT_DIR"/*_eval.jsonl 2>/dev/null || true
