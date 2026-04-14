#!/usr/bin/env bash
# Download LongMemEval benchmark data.
# Run from the project root: bash benchmarks/longmemeval/download_data.sh

set -euo pipefail

DATA_DIR="benchmarks/longmemeval/data"
BASE_URL="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

mkdir -p "$DATA_DIR"

echo "Downloading LongMemEval data..."

# Oracle (evidence-only, fast iteration)
if [ ! -f "$DATA_DIR/longmemeval_oracle.json" ]; then
    echo "  -> longmemeval_oracle.json"
    wget -q -O "$DATA_DIR/longmemeval_oracle.json" "$BASE_URL/longmemeval_oracle.json"
else
    echo "  -> longmemeval_oracle.json (already exists)"
fi

# S (small, ~115k tokens / ~40 sessions per instance)
if [ ! -f "$DATA_DIR/longmemeval_s_cleaned.json" ]; then
    echo "  -> longmemeval_s_cleaned.json"
    wget -q -O "$DATA_DIR/longmemeval_s_cleaned.json" "$BASE_URL/longmemeval_s_cleaned.json"
else
    echo "  -> longmemeval_s_cleaned.json (already exists)"
fi

# M (medium, ~500 sessions per instance)
if [ ! -f "$DATA_DIR/longmemeval_m_cleaned.json" ]; then
    echo "  -> longmemeval_m_cleaned.json"
    wget -q -O "$DATA_DIR/longmemeval_m_cleaned.json" "$BASE_URL/longmemeval_m_cleaned.json"
else
    echo "  -> longmemeval_m_cleaned.json (already exists)"
fi

echo "Done. Data in $DATA_DIR/"
