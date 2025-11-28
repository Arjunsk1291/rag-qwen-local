#!/bin/bash

MODEL_DIR="$HOME/rag-qwen-local/models"
MODEL_FILE="Qwen2.5-7B-Instruct-Q4_K_M.gguf"

echo "==================================="
echo "Downloading Qwen2.5-7B-Instruct (Q4_K_M)"
echo "==================================="
echo "Source: bartowski/Qwen2.5-7B-Instruct-GGUF"
echo "Model: Q4_K_M quantization (~4.68GB)"
echo "Optimized for: GTX 1660 Ti (6GB VRAM)"
echo ""

cd "$MODEL_DIR"

# Method 1: Using huggingface-cli (RECOMMENDED)
if command -v huggingface-cli &> /dev/null; then
    echo "Downloading with huggingface-cli..."
    huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
        --include "Qwen2.5-7B-Instruct-Q4_K_M.gguf" \
        --local-dir . \
        --local-dir-use-symlinks False
else
    echo "huggingface-cli not found. Installing..."
    pip install -U "huggingface_hub[cli]"
    
    echo "Retrying download..."
    huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
        --include "Qwen2.5-7B-Instruct-Q4_K_M.gguf" \
        --local-dir . \
        --local-dir-use-symlinks False
fi

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
    echo "Please check your internet connection and try again."
    exit 1
fi
