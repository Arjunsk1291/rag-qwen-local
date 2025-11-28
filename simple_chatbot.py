"""
Simplified Document Chatbot - Guaranteed Working Version
"""

import gradio as gr
from rag_system import LocalRAGSystem

# Global RAG instance
rag = None

def init_system():
    """Initialize the RAG system."""
    global rag
    try:
        rag = LocalRAGSystem(
            model_path="models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            n_gpu_layers=35,
            verbose=False
        )
        return "✅ System initialized! You can now upload documents."
    except Exception as e:
        return f"❌ Error: {str(e)}"

def upload_doc(file):
    """Upload and process document."""
    global rag
    if rag is None:
        return "❌ Initialize system first!"
    
    try:
        chunks = rag.ingest_document(file.name)
        return f"✅ Uploaded! Created {chunks} chunks."
    except Exception as e:
        return f"❌ Error: {str(e)}"

def chat_fn(message, history):
    """Chat function."""
    global rag
    
    if rag is None:
        history.append((message, "❌ Initialize system first!"))
        return history
    
    try:
        result = rag.query(message, top_k=5, max_tokens=512)
        answer = result['answer']
        answer += f"\n\n📚 (Used {result['num_contexts']} chunks)"
        history.append((message, answer))
        return history
    except Exception as e:
        history.append((message, f"❌ Error: {str(e)}"))
        return history

# Create interface
with gr.Blocks() as demo:
    gr.Markdown("# 📚 Document Chatbot\n### Local AI powered by Qwen2.5-7B")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Setup")
            init_btn = gr.Button("🚀 Initialize", variant="primary")
            init_out = gr.Textbox(label="Status", lines=2)
            
            gr.Markdown("## Upload")
            file = gr.File(label="Document", file_types=[".pdf", ".txt", ".docx"])
            upload_btn = gr.Button("📥 Upload")
            upload_out = gr.Textbox(label="Status", lines=2)
        
        with gr.Column():
            gr.Markdown("## Chat")
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(label="Message", placeholder="Ask a question...")
            send = gr.Button("Send")
            clear = gr.Button("Clear")
    
    # Connect functions
    init_btn.click(init_system, outputs=init_out)
    upload_btn.click(upload_doc, inputs=file, outputs=upload_out)
    send.click(chat_fn, inputs=[msg, chatbot], outputs=chatbot)
    msg.submit(chat_fn, inputs=[msg, chatbot], outputs=chatbot)
    clear.click(lambda: [], outputs=chatbot)

if __name__ == "__main__":
    print("\n🚀 Launching Simple Document Chatbot...")
    print("📍 URL: http://localhost:7860\n")
    demo.launch(server_port=7860, inbrowser=True)
