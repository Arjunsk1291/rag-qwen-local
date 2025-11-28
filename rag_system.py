"""
Local RAG System with Qwen2.5-7B-Instruct (Q4_K_M)
Hardware: Intel i7 11th Gen, 16GB RAM, NVIDIA GTX 1660 Ti 6GB
Framework: llama-cpp-python with GGML_CUDA support
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Document processing
import pypdf
from docx import Document as DocxDocument


class LocalRAGSystem:
    """
    Production-ready RAG system for local deployment.
    
    Specifications:
    - Model: Qwen2.5-7B-Instruct-Q4_K_M (4.37GB)
    - Embedding: all-MiniLM-L6-v2 (384 dimensions)
    - GPU: NVIDIA GTX 1660 Ti (6GB VRAM)
    - Framework: llama-cpp-python with GGML_CUDA
    """
    
    def __init__(
        self,
        model_path: str = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_gpu_layers: int = 35,  # Optimized for GTX 1660 Ti 6GB
        n_ctx: int = 4096,
        n_batch: int = 512,
        verbose: bool = True
    ):
        self.verbose = verbose
        self._log("="*70)
        self._log("INITIALIZING LOCAL RAG SYSTEM")
        self._log("="*70)
        
        # Verify model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Run: ./scripts/download_model.sh"
            )
        
        # Initialize LLM with GPU support
        self._log(f"Loading Qwen2.5-7B-Instruct (Q4_K_M)...")
        self._log(f"  Model path: {model_path}")
        self._log(f"  GPU layers: {n_gpu_layers}")
        self._log(f"  Context window: {n_ctx}")
        
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=8,
            use_mlock=True,
            verbose=verbose
        )
        self._log("✓ LLM loaded successfully")
        
        # Initialize embedding model
        self._log(f"Loading embedding model: {embedding_model}...")
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        self._log(f"✓ Embeddings ready (dimension: {self.embedding_dim})")
        
        # Initialize vector database
        self._log("Initializing ChromaDB...")
        os.makedirs("./data/chromadb", exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path="./data/chromadb",
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )
        self._log(f"✓ Vector database ready ({self.collection.count()} documents)")
        
        self._log("="*70)
        self._log("RAG SYSTEM READY")
        self._log("="*70 + "\n")
    
    def _log(self, message: str):
        """Internal logging."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text.strip()
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX."""
        doc = DocxDocument(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text])
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = 500, 
        overlap: int = 50
    ) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def ingest_document(
        self, 
        file_path: str, 
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Ingest a document into the RAG system.
        
        Args:
            file_path: Path to document
            metadata: Optional metadata
            
        Returns:
            Number of chunks created
        """
        self._log(f"\n{'─'*70}")
        self._log(f"INGESTING: {file_path}")
        self._log('─'*70)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract text
        file_ext = Path(file_path).suffix.lower()
        if file_ext == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            text = self.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            text = self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        if not text:
            raise ValueError(f"No text extracted from {file_path}")
        
        self._log(f"Extracted: {len(text)} characters")
        
        # Chunk text
        chunks = self.chunk_text(text)
        self._log(f"Created: {len(chunks)} chunks")
        
        # Generate embeddings
        self._log("Generating embeddings with all-MiniLM-L6-v2...")
        embeddings = self.embedder.encode(
            chunks, 
            show_progress_bar=self.verbose,
            batch_size=32
        )
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        base_metadata = {
            'source': file_path,
            'filename': Path(file_path).name,
            'timestamp': datetime.now().isoformat(),
            **metadata
        }
        
        # Add to database
        self._log("Adding to ChromaDB...")
        doc_id = Path(file_path).stem
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=[base_metadata for _ in chunks],
            ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        )
        
        self._log(f"✓ Ingestion complete: {len(chunks)} chunks added")
        self._log('─'*70 + "\n")
        
        return len(chunks)
    
    def retrieve_context(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict]:
        """Retrieve relevant context chunks."""
        # Generate query embedding
        query_embedding = self.embedder.encode([query])[0]
        
        # Query vector database
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count())
        )
        
        # Format results
        contexts = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                contexts.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
        
        return contexts
    
    def generate_response(
        self,
        query: str,
        contexts: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """Generate response using Qwen with context."""
        
        # Build context string
        context_str = "\n\n".join([
            f"Context {i+1}:\n{ctx['text']}"
            for i, ctx in enumerate(contexts)
        ])
        
        # Qwen2.5 prompt format
        prompt = f"""<|im_start|>system
You are a helpful AI assistant. Answer questions based STRICTLY on the provided context.
If the answer is not in the context, say "I cannot answer this based on the provided documents."
Be concise and accurate.<|im_end|>
<|im_start|>user
Context:
{context_str}

Question: {query}<|im_end|>
<|im_start|>assistant
"""
        
        # Generate
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            stop=["<|im_end|>", "<|endoftext|>"],
            echo=False
        )
        
        return response['choices'][0]['text'].strip()
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        max_tokens: int = 512,
        show_context: bool = False
    ) -> Dict:
        """
        Complete RAG pipeline.
        
        Args:
            question: User question
            top_k: Number of context chunks
            max_tokens: Max response length
            show_context: Include context in output
            
        Returns:
            Dictionary with answer and metadata
        """
        self._log(f"\n{'─'*70}")
        self._log(f"QUERY: {question}")
        self._log('─'*70)
        
        # Retrieve
        self._log(f"Retrieving top {top_k} chunks...")
        contexts = self.retrieve_context(question, top_k=top_k)
        
        if not contexts:
            return {
                'question': question,
                'answer': "No relevant information found in the knowledge base.",
                'contexts': []
            }
        
        self._log(f"Found {len(contexts)} relevant chunks")
        
        # Generate
        self._log("Generating response with Qwen2.5-7B-Instruct...")
        answer = self.generate_response(question, contexts, max_tokens)
        
        result = {
            'question': question,
            'answer': answer,
            'num_contexts': len(contexts)
        }
        
        if show_context:
            result['contexts'] = contexts
        
        self._log("✓ Response generated")
        self._log('─'*70 + "\n")
        
        return result
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        return {
            'total_documents': self.collection.count(),
            'embedding_dimension': self.embedding_dim,
            'model': 'Qwen2.5-7B-Instruct',
            'quantization': 'Q4_K_M (4-bit)',
            'model_size': '4.37GB',
            'embedding_model': 'all-MiniLM-L6-v2',
            'hardware': {
                'gpu': 'NVIDIA GTX 1660 Ti',
                'vram': '6GB',
                'cpu': 'Intel i7 11th Gen',
                'ram': '16GB'
            }
        }


if __name__ == "__main__":
    # Quick test
    rag = LocalRAGSystem(verbose=True)
    
    print("\n" + "="*70)
    print("SYSTEM STATISTICS")
    print("="*70)
    stats = rag.get_stats()
    
    print(f"\nModel: {stats['model']} ({stats['quantization']})")
    print(f"Model Size: {stats['model_size']}")
    print(f"Embedding: {stats['embedding_model']}")
    print(f"Documents in DB: {stats['total_documents']}")
    print(f"\nHardware:")
    for key, value in stats['hardware'].items():
        print(f"  {key}: {value}")
    print("="*70 + "\n")
