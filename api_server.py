"""
FastAPI REST API for Local RAG System
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import shutil
from pathlib import Path

from rag_system import LocalRAGSystem

app = FastAPI(
    title="Local RAG API",
    description="Hardware-optimized RAG with Qwen2.5-7B",
    version="1.0.0"
)

# Global RAG instance
rag_system = None

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    max_tokens: int = 512

@app.on_event("startup")
async def startup():
    global rag_system
    print("Initializing RAG System...")
    rag_system = LocalRAGSystem(
        model_path="models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        n_gpu_layers=35,
        verbose=True
    )
    print("✓ API Ready")

@app.get("/")
async def root():
    return {
        "service": "Local RAG API",
        "status": "running",
        "model": "Qwen2.5-7B-Instruct-Q4_K_M"
    }

@app.get("/stats")
async def stats():
    return rag_system.get_stats()

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload and ingest document."""
    file_path = Path("documents") / file.filename
    
    with file_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        chunks = rag_system.ingest_document(str(file_path))
        return {
            "status": "success",
            "filename": file.filename,
            "chunks": chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(request: QueryRequest):
    """Query the RAG system."""
    try:
        result = rag_system.query(
            question=request.question,
            top_k=request.top_k,
            max_tokens=request.max_tokens
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
