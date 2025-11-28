# Local RAG System with Qwen2.5-7B

**Hardware-Optimized Retrieval-Augmented Generation System**

## System Configuration
- **Model**: Qwen2.5-7B-Instruct (Q4_K_M quantization - 4.37GB)
- **GPU**: NVIDIA GTX 1660 Ti (6GB VRAM)
- **CPU**: Intel i7 11th Gen
- **RAM**: 16GB
- **OS**: Ubuntu 22.04 LTS

## Key Features
- GPU-accelerated inference with llama.cpp
- Vector search with ChromaDB
- Semantic embeddings with all-MiniLM-L6-v2
- Support for PDF, DOCX, and TXT documents
- REST API with FastAPI
- Optimized for constrained hardware

## Installation
See `INSTALL.md` for complete setup instructions.

## Quick Start
```bash
source venv/bin/activate
python example_usage.py
```
