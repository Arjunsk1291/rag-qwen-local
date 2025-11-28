#!/bin/bash

echo "==================================="
echo "LAUNCHING DOCUMENT CHATBOT"
echo "==================================="
echo ""

cd ~/rag-qwen-local
source venv/bin/activate

# Check if model exists
if [ ! -f "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf" ]; then
    echo "Model not found. Downloading..."
    ./scripts/download_model.sh
fi

# Launch chatbot
python document_chatbot.py
