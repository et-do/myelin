#!/usr/bin/env bash
# Download LoCoMo benchmark data.
# Run from the project root: bash benchmarks/locomo/download_data.sh

set -euo pipefail

DATA_DIR="benchmarks/locomo/data"
RAW_URL="https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

mkdir -p "$DATA_DIR"

if [ ! -f "$DATA_DIR/locomo10.json" ]; then
    echo "Downloading locomo10.json..."
    wget -q -O "$DATA_DIR/locomo10.json" "$RAW_URL"
    echo "Done. Data in $DATA_DIR/"
else
    echo "locomo10.json already exists — skipping download."
fi
