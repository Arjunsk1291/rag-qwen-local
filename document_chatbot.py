"""
Document Chatbot - Interactive Web Interface
Hardware-optimized for GTX 1660 Ti (6GB VRAM)
"""

import gradio as gr
from rag_system import LocalRAGSystem
from pathlib import Path
from typing import List, Dict

class DocumentChatbot:
    """Interactive document chatbot with web interface."""
    
    def __init__(self):
        """Initialize the chatbot."""
        self.rag = None
        
    def initialize_rag(self) -> str:
        """Initialize RAG system."""
        try:
            self.rag = LocalRAGSystem(
                model_path="models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                n_gpu_layers=35,
                verbose=False
            )
            stats = self.rag.get_stats()
            return f"""✅ System Initialized Successfully!

📊 System Statistics:
- Model: {stats['model']} ({stats['quantization']})
- Documents Loaded: {stats['total_documents']} chunks
- GPU: {stats['hardware']['gpu']} ({stats['hardware']['vram']})
- Embedding: {stats['embedding_model']}

Ready to upload documents and answer questions!"""
        except Exception as e:
            return f"❌ Error initializing system: {str(e)}"
    
    def upload_document(self, file) -> str:
        """Handle document upload."""
        if self.rag is None:
            return "❌ Please initialize the system first!"
        
        if file is None:
            return "❌ No file selected"
        
        try:
            file_path = file.name
            num_chunks = self.rag.ingest_document(file_path)
            stats = self.rag.get_stats()
            
            return f"""✅ Document Uploaded Successfully!

📄 File: {Path(file_path).name}
📦 Chunks: {num_chunks}
📊 Total in DB: {stats['total_documents']} chunks

You can now ask questions!"""
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def chat(self, message: str, history: List[Dict]) -> List[Dict]:
        """Handle chat messages with proper message format."""
        if self.rag is None:
            # Return history with error message
            history.append({
                "role": "user",
                "content": message
            })
            history.append({
                "role": "assistant",
                "content": "❌ Please initialize the system first by clicking '🚀 Initialize System' button!"
            })
            return history
        
        if not message.strip():
            return history
        
        # Add user message
        history.append({
            "role": "user",
            "content": message
        })
        
        try:
            # Query the RAG system
            result = self.rag.query(
                question=message,
                top_k=5,
                max_tokens=512
            )
            
            response = result['answer']
            
            # Add source info
            if result['num_contexts'] > 0:
                response += f"\n\n📚 *Retrieved from {result['num_contexts']} document chunks*"
            
            # Add assistant message
            history.append({
                "role": "assistant",
                "content": response
            })
            
        except Exception as e:
            # Add error message
            history.append({
                "role": "assistant",
                "content": f"❌ Error: {str(e)}"
            })
        
        return history
    
    def clear_chat(self):
        """Clear chat history."""
        return []
    
    def get_system_info(self) -> str:
        """Get current system information."""
        if self.rag is None:
            return "❌ System not initialized"
        
        stats = self.rag.get_stats()
        return f"""## 🖥️ System Information

**Model Configuration:**
- Model: {stats['model']}
- Quantization: {stats['quantization']}
- Size: {stats['model_size']}

**Hardware:**
- GPU: {stats['hardware']['gpu']}
- VRAM: {stats['hardware']['vram']}
- CPU: {stats['hardware']['cpu']}
- RAM: {stats['hardware']['ram']}

**Embeddings:**
- Model: {stats['embedding_model']}
- Dimension: {stats['embedding_dimension']}

**Knowledge Base:**
- Total Chunks: {stats['total_documents']}"""

def create_interface():
    """Create Gradio interface."""
    chatbot_instance = DocumentChatbot()
    
    with gr.Blocks(title="Document Chatbot") as demo:
        
        gr.Markdown("""
        # 📚 Document Chatbot
        ### Chat with your documents using local AI
        
        **Hardware:** NVIDIA GTX 1660 Ti | **Model:** Qwen2.5-7B-Instruct (Q4_K_M)
        """)
        
        with gr.Row():
            # Left column - Controls
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ System Setup")
                
                init_btn = gr.Button(
                    "🚀 Initialize System",
                    variant="primary",
                    size="lg"
                )
                init_output = gr.Textbox(
                    label="Status",
                    lines=8,
                    interactive=False,
                    show_label=False
                )
                
                gr.Markdown("---")
                gr.Markdown("## 📤 Upload Document")
                
                file_upload = gr.File(
                    label="Select File",
                    file_types=[".pdf", ".docx", ".txt"]
                )
                upload_btn = gr.Button("📥 Upload", size="lg")
                upload_output = gr.Textbox(
                    label="Upload Status",
                    lines=6,
                    interactive=False,
                    show_label=False
                )
                
                gr.Markdown("---")
                gr.Markdown("## ℹ️ System Info")
                info_btn = gr.Button("📊 Show Stats")
                info_output = gr.Markdown()
            
            # Right column - Chat
            with gr.Column(scale=2):
                gr.Markdown("## 💬 Chat Interface")
                
                chatbot = gr.Chatbot(
                    type="messages",  # Use messages format
                    height=500,
                    label="Conversation",
                    show_label=False
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Ask a question about your documents...",
                        scale=5,
                        show_label=False
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                clear_btn = gr.Button("🗑️ Clear Chat")
                
                gr.Markdown("""
                ### 💡 Quick Guide:
                1. Click **Initialize System** (takes ~15 seconds)
                2. **Upload** a PDF, DOCX, or TXT file
                3. **Ask questions** about the document
                4. The AI answers based **only** on uploaded content
                """)
        
        # Event handlers
        init_btn.click(
            fn=chatbot_instance.initialize_rag,
            outputs=init_output
        )
        
        upload_btn.click(
            fn=chatbot_instance.upload_document,
            inputs=file_upload,
            outputs=upload_output
        )
        
        # Chat handlers
        submit_btn.click(
            fn=chatbot_instance.chat,
            inputs=[msg, chatbot],
            outputs=chatbot
        ).then(
            lambda: "",  # Clear input after sending
            outputs=msg
        )
        
        msg.submit(
            fn=chatbot_instance.chat,
            inputs=[msg, chatbot],
            outputs=chatbot
        ).then(
            lambda: "",  # Clear input after sending
            outputs=msg
        )
        
        clear_btn.click(
            fn=chatbot_instance.clear_chat,
            outputs=chatbot
        )
        
        info_btn.click(
            fn=chatbot_instance.get_system_info,
            outputs=info_output
        )
    
    return demo

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 LAUNCHING DOCUMENT CHATBOT")
    print("="*70)
    print("\n📍 Hardware: Intel i7 11th Gen, 16GB RAM, GTX 1660 Ti 6GB")
    print("🤖 Model: Qwen2.5-7B-Instruct-Q4_K_M (4.68GB)")
    print("🔗 URL: http://localhost:7860")
    print("\n⌨️  Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
