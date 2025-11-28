"""
Document Chatbot - Interactive Web Interface
Hardware-optimized for GTX 1660 Ti (6GB VRAM)
"""

import gradio as gr
from rag_system import LocalRAGSystem
from pathlib import Path
import os
from typing import List, Tuple

class DocumentChatbot:
    """Interactive document chatbot with web interface."""
    
    def __init__(self):
        """Initialize the chatbot."""
        self.rag = None
        self.chat_history = []
        
    def initialize_rag(self) -> str:
        """Initialize RAG system."""
        try:
            self.rag = LocalRAGSystem(
                model_path="models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                n_gpu_layers=35,
                verbose=False
            )
            stats = self.rag.get_stats()
            return f"""
✅ System Initialized Successfully!

📊 **System Statistics:**
- Model: {stats['model']} ({stats['quantization']})
- Documents Loaded: {stats['total_documents']} chunks
- GPU: {stats['hardware']['gpu']} ({stats['hardware']['vram']})
- Embedding Model: {stats['embedding_model']}

You can now upload documents and start chatting!
            """
        except Exception as e:
            return f"❌ Error initializing system: {str(e)}"
    
    def upload_document(self, file) -> str:
        """Handle document upload."""
        if self.rag is None:
            return "❌ Please initialize the system first!"
        
        if file is None:
            return "❌ No file selected"
        
        try:
            # Get the file path
            file_path = file.name
            
            # Ingest document
            num_chunks = self.rag.ingest_document(file_path)
            
            stats = self.rag.get_stats()
            
            return f"""
✅ Document Uploaded Successfully!

📄 **File:** {Path(file_path).name}
📦 **Chunks Created:** {num_chunks}
📊 **Total Documents in System:** {stats['total_documents']} chunks

You can now ask questions about this document!
            """
        except Exception as e:
            return f"❌ Error uploading document: {str(e)}"
    
    def chat(
        self, 
        message: str, 
        history: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], str]:
        """Handle chat messages."""
        if self.rag is None:
            response = "❌ Please initialize the system first!"
            history.append((message, response))
            return history, ""
        
        if not message.strip():
            return history, message
        
        try:
            # Query the RAG system
            result = self.rag.query(
                question=message,
                top_k=5,
                max_tokens=512
            )
            
            response = result['answer']
            
            # Add sources information
            if result['num_contexts'] > 0:
                response += f"\n\n📚 *Retrieved from {result['num_contexts']} document chunks*"
            
            history.append((message, response))
            return history, ""
            
        except Exception as e:
            response = f"❌ Error: {str(e)}"
            history.append((message, response))
            return history, ""
    
    def clear_chat(self):
        """Clear chat history."""
        self.chat_history = []
        return []
    
    def get_system_info(self) -> str:
        """Get current system information."""
        if self.rag is None:
            return "❌ System not initialized"
        
        stats = self.rag.get_stats()
        return f"""
## 🖥️ System Information

**Model Configuration:**
- Model: {stats['model']}
- Quantization: {stats['quantization']}
- Model Size: {stats['model_size']}

**Hardware:**
- GPU: {stats['hardware']['gpu']}
- VRAM: {stats['hardware']['vram']}
- CPU: {stats['hardware']['cpu']}
- RAM: {stats['hardware']['ram']}

**Embeddings:**
- Model: {stats['embedding_model']}
- Dimension: {stats['embedding_dimension']}

**Knowledge Base:**
- Total Chunks: {stats['total_documents']}
        """

def create_interface():
    """Create Gradio interface."""
    chatbot_instance = DocumentChatbot()
    
    with gr.Blocks(
        title="Document Chatbot",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("""
        # 📚 Document Chatbot
        ### Chat with your documents using local AI
        
        **Hardware-optimized for NVIDIA GTX 1660 Ti**  
        Powered by Qwen2.5-7B-Instruct (Q4_K_M quantization)
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Setup")
                
                init_btn = gr.Button(
                    "🚀 Initialize System",
                    variant="primary",
                    size="lg"
                )
                init_output = gr.Textbox(
                    label="Initialization Status",
                    lines=10,
                    interactive=False
                )
                
                gr.Markdown("## 📤 Upload Documents")
                
                file_upload = gr.File(
                    label="Select Document",
                    file_types=[".pdf", ".docx", ".txt"]
                )
                upload_btn = gr.Button("Upload Document")
                upload_output = gr.Textbox(
                    label="Upload Status",
                    lines=6,
                    interactive=False
                )
                
                gr.Markdown("## ℹ️ System Info")
                info_btn = gr.Button("Show System Info")
                info_output = gr.Markdown()
            
            with gr.Column(scale=2):
                gr.Markdown("## 💬 Chat")
                
                chatbot = gr.Chatbot(
                    height=500,
                    label="Conversation",
                    show_label=False
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Message",
                        placeholder="Ask a question about your documents...",
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat")
                
                gr.Markdown("""
                ### 💡 Tips:
                - Initialize the system before uploading documents
                - Supports PDF, DOCX, and TXT files
                - Ask specific questions for best results
                - The AI answers strictly based on uploaded documents
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
        
        submit_btn.click(
            fn=chatbot_instance.chat,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            fn=chatbot_instance.chat,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
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
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
