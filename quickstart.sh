#!/bin/bash

echo "==================================="
echo "QUICK START - LOCAL RAG SYSTEM"
echo "==================================="
echo ""

# Check if model exists
if [ ! -f "models/qwen2.5-7b-instruct-q4_k_m.gguf" ]; then
    echo "Model not found. Downloading..."
    ./scripts/download_model.sh
fi

# Activate venv
source venv/bin/activate

# Run example
echo ""
echo "Running example usage..."
python example_usage.py
