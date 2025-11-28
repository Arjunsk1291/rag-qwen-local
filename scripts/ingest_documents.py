"""
Batch document ingestion script
"""

import sys
sys.path.append('..')

from rag_system import LocalRAGSystem
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into RAG system"
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='Document files to ingest'
    )
    parser.add_argument(
        '--model',
        default='models/Qwen2.5-7B-Instruct-Q4_K_M.gguf',
        help='Path to model file'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("BATCH DOCUMENT INGESTION")
    print("="*70 + "\n")
    
    # Initialize RAG
    rag = LocalRAGSystem(
        model_path=args.model,
        n_gpu_layers=35,
        verbose=True
    )
    
    # Ingest each file
    total_chunks = 0
    successful = 0
    failed = 0
    
    for file_path in args.files:
        try:
            chunks = rag.ingest_document(file_path)
            total_chunks += chunks
            successful += 1
        except Exception as e:
            print(f"✗ Failed to ingest {file_path}: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("INGESTION SUMMARY")
    print("="*70)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total Chunks: {total_chunks}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
