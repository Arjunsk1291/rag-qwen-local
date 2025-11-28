#!/bin/bash

MODEL_DIR="$HOME/rag-qwen-local/models"
MODEL_FILE="qwen2.5-7b-instruct-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"

echo "==================================="
echo "Downloading Qwen2.5-7B-Instruct (Q4_K_M)"
echo "==================================="
echo "Model: Q4_K_M quantization (4.37GB)"
echo "Optimized for: GTX 1660 Ti (6GB VRAM)"
echo ""

cd "$MODEL_DIR"

# Download with wget (supports resume)
wget -c --show-progress "$MODEL_URL"

# Verify download
if [ -f "$MODEL_FILE" ]; then
    FILE_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo ""
    echo "✓ Download completed successfully!"
    echo "  File: $MODEL_FILE"
    echo "  Size: $FILE_SIZE"
    echo ""
else
    echo "✗ Download failed!"
    exit 1
fi
