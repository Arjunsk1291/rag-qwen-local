"""
System validation tests
"""

import sys
sys.path.append('..')

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
    
    print(f"\n✓ Initialization: {elapsed:.2f}s")
    return rag

def test_embedding(rag):
    """Test embedding generation."""
    print("\n" + "="*70)
    print("TEST 2: Embedding Generation")
    print("="*70)
    
    texts = ["This is a test.", "Another test sentence."]
    start = time.time()
    embeddings = rag.embedder.encode(texts)
    elapsed = time.time() - start
    
    print(f"Shape: {embeddings.shape}")
    print(f"✓ Embedding: {elapsed:.4f}s")

def test_generation(rag):
    """Test text generation."""
    print("\n" + "="*70)
    print("TEST 3: Text Generation")
    print("="*70)
    
    prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
"""
    
    start = time.time()
    response = rag.llm(prompt, max_tokens=20, stop=["<|im_end|>"], echo=False)
    elapsed = time.time() - start
    
    answer = response['choices'][0]['text'].strip()
    print(f"Q: What is 2+2?")
    print(f"A: {answer}")
    print(f"✓ Generation: {elapsed:.2f}s")

if __name__ == "__main__":
    print("\nStarting tests...")
    try:
        rag = test_initialization()
        test_embedding(rag)
        test_generation(rag)
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
