# 📚 Document Chatbot - Local RAG System

**Chat with your documents using a local AI model**  
Hardware-optimized for NVIDIA GTX 1660 Ti (6GB VRAM)

## 🚀 Features

- **100% Local**: No API keys, no cloud services, complete privacy
- **GPU Accelerated**: Optimized for NVIDIA GTX 1660 Ti
- **Multi-Format Support**: PDF, DOCX, TXT documents
- **Web Interface**: Beautiful Gradio UI for easy interaction
- **REST API**: FastAPI server for programmatic access
- **Vector Search**: Semantic search with ChromaDB
- **Production Ready**: Comprehensive error handling and logging

## 🖥️ System Configuration

- **Model**: Qwen2.5-7B-Instruct (Q4_K_M quantization - 4.68GB)
- **GPU**: NVIDIA GTX 1660 Ti (6GB VRAM) - 35 layers offloaded
- **CPU**: Intel i7 11th Gen
- **RAM**: 16GB
- **OS**: Ubuntu 22.04 LTS
- **Embedding**: all-MiniLM-L6-v2 (384 dimensions)

## 📦 Installation

### Prerequisites
```bash
# CUDA Toolkit
sudo apt install nvidia-cuda-toolkit

# Python 3.10
sudo apt install python3.10 python3.10-venv

# Build tools
sudo apt install build-essential cmake
```

### Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/rag-qwen-local.git
cd rag-qwen-local

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download model
./scripts/download_model.sh
```

## 🎯 Quick Start

### Option 1: Web Chatbot (Recommended)
```bash
./launch_chatbot.sh
# Opens at http://localhost:7860
```

### Option 2: REST API
```bash
./launch_api.sh
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Option 3: Python Script
```bash
python example_usage.py
```

## 📖 Usage Examples

### Web Interface
1. Click "🚀 Initialize System"
2. Upload a document (PDF, DOCX, or TXT)
3. Start asking questions about your document!

### Programmatic Usage
```python
from rag_system import LocalRAGSystem

# Initialize
rag = LocalRAGSystem(
    model_path="models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    n_gpu_layers=35
)

# Ingest document
rag.ingest_document("document.pdf")

# Query
result = rag.query("What is this document about?")
print(result['answer'])
```

### API Examples
```bash
# Upload document
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@document.pdf"

# Query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'

# Get stats
curl "http://localhost:8000/stats"
```

### Batch Ingestion
```bash
python scripts/ingest_documents.py documents/*.pdf
```

## 🏗️ Project Structure
```
rag-qwen-local/
├── models/
│   └── Qwen2.5-7B-Instruct-Q4_K_M.gguf
├── documents/              # Place your documents here
├── data/
│   └── chromadb/          # Vector database
├── scripts/
│   ├── download_model.sh
│   └── ingest_documents.py
├── tests/
│   └── test_system.py
├── rag_system.py          # Core RAG implementation
├── document_chatbot.py    # Web UI
├── api_server.py          # REST API
├── example_usage.py       # Usage examples
└── launch_chatbot.sh      # Quick launch
```

## 🔧 Configuration

### Adjust GPU Memory Usage

Edit `rag_system.py`:
```python
# For 4GB VRAM
n_gpu_layers=25

# For 8GB+ VRAM
n_gpu_layers=45
```

### Adjust Context Window
```python
n_ctx=4096  # Default
n_ctx=8192  # More context (uses more memory)
```

### Change Model Quantization

Available quantizations (from bartowski):
- Q8_0: Highest quality (~7.7GB)
- Q6_K: High quality (~5.9GB)
- Q5_K_M: Balanced (~5.1GB)
- **Q4_K_M: Recommended** (~4.68GB) ✓
- Q3_K_M: Smaller (~3.5GB)

## 🧪 Testing
```bash
# Run all tests
python tests/test_system.py

# Expected output:
# ✓ System initialization
# ✓ Embedding generation
# ✓ Text generation
# ✓ Vector database
# 🎉 ALL TESTS PASSED
```

## 📊 Performance

- **Initialization**: ~10-15 seconds
- **Document Ingestion**: ~1-2 seconds per page
- **Query Response**: ~2-5 seconds (depending on context)
- **Tokens per Second**: ~15-25 tokens/s on GTX 1660 Ti

## 🐛 Troubleshooting

### CUDA Errors
```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Reinstall llama-cpp-python
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir --force-reinstall
```

### Out of Memory
```bash
# Reduce GPU layers
# Edit rag_system.py: n_gpu_layers=25

# Or use smaller quantization
./scripts/download_model.sh  # Download Q3_K_M instead
```

### Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Qwen Team**: For the excellent base model
- **bartowski**: For GGUF quantizations
- **llama.cpp**: For the inference engine
- **ChromaDB**: For vector storage
- **Sentence Transformers**: For embeddings

## 📞 Support

- Issues: [GitHub Issues](https://github.com/YOUR_USERNAME/rag-qwen-local/issues)
- Discussions: [GitHub Discussions](https://github.com/YOUR_USERNAME/rag-qwen-local/discussions)

---

**Made with ❤️ for local AI enthusiasts**
