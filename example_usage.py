"""
Example usage of Local RAG System
"""

from rag_system import LocalRAGSystem
from pathlib import Path

def main():
    print("\n" + "="*70)
    print("LOCAL RAG SYSTEM - EXAMPLE USAGE")
    print("="*70 + "\n")
    
    # Initialize
    rag = LocalRAGSystem(
        model_path="models/qwen2.5-7b-instruct-q4_k_m.gguf",
        n_gpu_layers=35,
        verbose=True
    )
    
    # Create sample document
    sample_doc = "documents/ai_overview.txt"
    Path("documents").mkdir(exist_ok=True)
    
    with open(sample_doc, 'w') as f:
        f.write("""
Artificial Intelligence and Machine Learning

Machine learning is a branch of artificial intelligence that enables 
computers to learn from data without explicit programming. It uses 
algorithms to identify patterns and make predictions.

Deep learning is a subset of machine learning that uses artificial 
neural networks with multiple layers. These networks can automatically 
learn hierarchical representations of data, making them powerful for 
tasks like image recognition and natural language processing.

Natural Language Processing (NLP) allows computers to understand and 
generate human language. Modern NLP systems use transformer architectures 
like BERT, GPT, and T5. These models are pre-trained on large text corpora 
and can be fine-tuned for specific tasks.

Computer vision enables machines to interpret visual information. 
Applications include object detection, facial recognition, medical image 
analysis, and autonomous vehicles. Convolutional neural networks (CNNs) 
are commonly used for computer vision tasks.

Reinforcement learning is a type of machine learning where agents learn 
to make decisions by interacting with an environment. The agent receives 
rewards or penalties and learns to maximize cumulative rewards over time.
        """)
    
    # Ingest document
    print("Ingesting sample document...")
    rag.ingest_document(sample_doc)
    
    # Example queries
    queries = [
        "What is machine learning?",
        "Explain deep learning",
        "What are transformers used for?",
        "What is reinforcement learning?"
    ]
    
    print("\n" + "="*70)
    print("EXAMPLE QUERIES")
    print("="*70)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'─'*70}")
        print(f"Query {i}: {query}")
        print('─'*70)
        
        result = rag.query(
            question=query,
            top_k=3,
            max_tokens=256
        )
        
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\n(Used {result['num_contexts']} context chunks)")
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
