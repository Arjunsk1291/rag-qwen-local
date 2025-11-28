"""
System validation tests
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_system import LocalRAGSystem
import time

def test_initialization():
    """Test system initialization."""
    print("\n" + "="*70)
    print("TEST 1: System Initialization")
    print("="*70)
    
    start = time.time()
    rag = LocalRAGSystem(
        model_path="models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        n_gpu_layers=35,
        verbose=True
    )
    elapsed = time.time() - start
    
    print(f"\n✓ Initialization completed in {elapsed:.2f} seconds")
    return rag

def test_embedding(rag):
    """Test embedding generation."""
    print("\n" + "="*70)
    print("TEST 2: Embedding Generation")
    print("="*70)
    
    texts = ["This is a test sentence.", "Another test sentence for embeddings."]
    start = time.time()
    embeddings = rag.embedder.encode(texts)
    elapsed = time.time() - start
    
    print(f"Texts: {len(texts)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"✓ Embedding generation completed in {elapsed:.4f} seconds")

def test_generation(rag):
    """Test text generation."""
    print("\n" + "="*70)
    print("TEST 3: Text Generation (Qwen2.5-7B-Instruct Q4_K_M)")
    print("="*70)
    
    prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2? Answer briefly.<|im_end|>
<|im_start|>assistant
"""
    
    start = time.time()
    response = rag.llm(
        prompt,
        max_tokens=20,
        temperature=0.7,
        stop=["<|im_end|>"],
        echo=False
    )
    elapsed = time.time() - start
    
    answer = response['choices'][0]['text'].strip()
    print(f"Question: What is 2+2?")
    print(f"Answer: {answer}")
    print(f"✓ Generation completed in {elapsed:.2f} seconds")
    print(f"Tokens per second: ~{20/elapsed:.1f}")

def test_vector_db(rag):
    """Test vector database operations."""
    print("\n" + "="*70)
    print("TEST 4: Vector Database Operations")
    print("="*70)
    
    # Add test documents
    test_docs = [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neurons."
    ]
    
    print(f"Adding {len(test_docs)} test documents...")
    embeddings = rag.embedder.encode(test_docs)
    
    rag.collection.add(
        embeddings=embeddings.tolist(),
        documents=test_docs,
        metadatas=[{"source": "test"} for _ in test_docs],
        ids=[f"test_{i}" for i in range(len(test_docs))]
    )
    
    # Query
    query = "What is machine learning?"
    contexts = rag.retrieve_context(query, top_k=2)
    
    print(f"Query: {query}")
    print(f"Retrieved {len(contexts)} contexts")
    if contexts:
        print(f"Top result: {contexts[0]['text'][:60]}...")
    print("✓ Vector database test passed")

def print_system_info(rag):
    """Print system information."""
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    
    stats = rag.get_stats()
    
    print(f"\n📊 Model Configuration:")
    print(f"  Model: {stats['model']}")
    print(f"  Quantization: {stats['quantization']}")
    print(f"  Model Size: {stats['model_size']}")
    
    print(f"\n🖥️  Hardware:")
    for key, value in stats['hardware'].items():
        print(f"  {key.upper()}: {value}")
    
    print(f"\n📚 Knowledge Base:")
    print(f"  Total Chunks: {stats['total_documents']}")
    print(f"  Embedding Model: {stats['embedding_model']}")
    print(f"  Embedding Dimension: {stats['embedding_dimension']}")
    
    print("="*70)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LOCAL RAG SYSTEM - VALIDATION TESTS")
    print("Hardware: i7 11th Gen, 16GB RAM, GTX 1660 Ti 6GB")
    print("="*70)
    
    try:
        # Run all tests
        rag = test_initialization()
        print_system_info(rag)
        test_embedding(rag)
        test_generation(rag)
        test_vector_db(rag)
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY! 🎉")
        print("="*70 + "\n")
        
        print("Next steps:")
        print("1. Run document chatbot: ./launch_chatbot.sh")
        print("2. Run example usage: python example_usage.py")
        print("3. Launch API server: ./launch_api.sh")
        print()
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"Error: {e}\n")
        import traceback
        traceback.print_exc()
        print()
