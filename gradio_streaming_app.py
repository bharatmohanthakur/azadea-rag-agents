#!/usr/bin/env python3
"""
Gradio Application for RAG Server Streaming Endpoint
Consumes the /query/stream endpoint and displays responses in real-time
"""

import gradio as gr
import requests
import json
import time
from typing import Iterator, Tuple
import threading
import queue

# Configuration
API_BASE_URL = "http://localhost:8060"
STREAM_ENDPOINT = f"{API_BASE_URL}/query/stream"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

def check_server_health():
    """Check if the RAG server is running"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=2)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def stream_query(query: str, user_id: str = "gradio_user", history: list = None) -> Iterator[Tuple[str, str, str]]:
    """
    Stream query to RAG server and yield status, tokens, and metadata
    
    Yields:
        (status_message, accumulated_text, metadata_json)
    """
    if not query or not query.strip():
        yield "", "", ""
        return
    
    # Check server health first
    health = check_server_health()
    if not health:
        yield "❌ Error: RAG server is not available. Please ensure the server is running on port 8060.", "", ""
        return
    
    accumulated_text = ""
    current_status = ""
    metadata = {}
    
    try:
        # Prepare request
        payload = {
            "query": query.strip(),
            "user_id": user_id
        }
        
        # Stream the response
        response = requests.post(
            STREAM_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=120
        )
        
        if response.status_code != 200:
            error_msg = f"❌ Server error: {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f" - {error_detail.get('detail', 'Unknown error')}"
            except:
                error_msg += f" - {response.text[:200]}"
            yield error_msg, "", ""
            return
        
        # Process Server-Sent Events (SSE)
        for line in response.iter_lines():
            if not line:
                continue
            
            # SSE format: "data: {...}"
            line_str = line.decode('utf-8')
            if not line_str.startswith('data: '):
                continue
            
            try:
                data_str = line_str[6:]  # Remove "data: " prefix
                data = json.loads(data_str)
                
                msg_type = data.get('type', '')
                
                if msg_type == 'status':
                    # Status message (intermediate progress)
                    current_status = data.get('message', '')
                    yield current_status, accumulated_text, json.dumps(metadata, indent=2)
                
                elif msg_type == 'progress':
                    # Progress update with percentage
                    percentage = data.get('percentage', 0)
                    progress_msg = data.get('message', '')
                    current_status = f"{progress_msg} ({percentage}%)"
                    yield current_status, accumulated_text, json.dumps(metadata, indent=2)
                
                elif msg_type == 'source_found':
                    # Source found event
                    source_name = data.get('source', 'Unknown')
                    source_index = data.get('index', 0)
                    score = data.get('score', 0.0)
                    # Update status to show source
                    current_status = f"📚 Found source {source_index}: {source_name} (score: {score:.2f})"
                    yield current_status, accumulated_text, json.dumps(metadata, indent=2)
                
                elif msg_type == 'code_block_start':
                    # Code block start
                    language = data.get('language', '')
                    accumulated_text += f"\n\n```{language}\n"
                    yield current_status, accumulated_text, json.dumps(metadata, indent=2)
                
                elif msg_type == 'code':
                    # Code content
                    code_text = data.get('text', '')
                    accumulated_text += code_text
                    yield current_status, accumulated_text, json.dumps(metadata, indent=2)
                
                elif msg_type == 'code_block_end':
                    # Code block end
                    accumulated_text += "\n```\n"
                    yield current_status, accumulated_text, json.dumps(metadata, indent=2)
                
                elif msg_type == 'token':
                    # Token (word) - accumulate text
                    token = data.get('text', '')
                    accumulated_text += token
                    yield current_status, accumulated_text, json.dumps(metadata, indent=2)
                
                elif msg_type == 'done':
                    # Final metadata
                    metadata = data.get('metadata', {})
                    yield current_status, accumulated_text, json.dumps(metadata, indent=2)
                
                elif msg_type == 'error':
                    # Error message
                    error_msg = data.get('error', 'Unknown error')
                    yield f"❌ Error: {error_msg}", accumulated_text, json.dumps(metadata, indent=2)
                    break
                    
            except json.JSONDecodeError as e:
                # Skip malformed JSON
                continue
            except Exception as e:
                yield f"⚠️ Warning: {str(e)}", accumulated_text, json.dumps(metadata, indent=2)
        
        # Final yield with complete response
        yield current_status, accumulated_text, json.dumps(metadata, indent=2)
        
    except requests.exceptions.Timeout:
        yield "⏱️ Request timeout. The query may be taking longer than expected.", accumulated_text, json.dumps(metadata, indent=2)
    except requests.exceptions.ConnectionError:
        yield "❌ Connection error. Please ensure the RAG server is running on port 8060.", "", ""
    except Exception as e:
        yield f"❌ Error: {str(e)}", accumulated_text, json.dumps(metadata, indent=2)

def chat_interface(message: str, history: list, user_id: str) -> Tuple[list, str, str, str]:
    """
    Handle chat interface with streaming
    
    Returns:
        (updated_history, status, response_text, metadata)
    """
    if not message or not message.strip():
        return history, "", "", ""
    
    # Initialize response
    status_message = ""
    response_text = ""
    metadata_json = ""
    
    # Stream the response
    for status, text, metadata in stream_query(message, user_id, history):
        status_message = status
        response_text = text
        metadata_json = metadata
        
        # Update history with streaming response
        if history and len(history) > 0 and history[-1][0] == message:
            # Update existing entry
            history[-1][1] = response_text
        else:
            # Add new entry
            history.append([message, response_text])
        
        yield history, status_message, response_text, metadata_json
    
    # Final update
    if history and len(history) > 0:
        history[-1][1] = response_text
    yield history, status_message, response_text, metadata_json

def create_gradio_interface():
    """Create and launch Gradio interface"""
    
    # Custom CSS for better UI
    custom_css = """
    .status-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f0f0;
        margin-bottom: 10px;
        font-weight: 500;
    }
    .metadata-box {
        font-family: monospace;
        font-size: 12px;
        max-height: 300px;
        overflow-y: auto;
    }
    """
    
    with gr.Blocks(title="RAG Server Streaming Chat", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.Markdown("""
        # 🤖 RAG Server Streaming Chat Interface
        
        This interface connects to the RAG server's `/query/stream` endpoint and displays responses in real-time.
        
        **Features:**
        - ⚡ Real-time streaming responses (word-by-word)
        - 📊 Intermediate status updates (like Gemini/Claude)
        - 📈 Complete metadata display
        - 💬 Conversation history
        """)
        
        # Server status check
        with gr.Row():
            server_status = gr.Textbox(
                label="Server Status",
                value="Checking...",
                interactive=False,
                scale=3
            )
            refresh_btn = gr.Button("🔄 Refresh Status", scale=1)
        
        # User ID input
        with gr.Row():
            user_id_input = gr.Textbox(
                label="User ID",
                value="gradio_user",
                placeholder="Enter your user ID",
                scale=1
            )
        
        # Chat interface
        chatbot = gr.Chatbot(
            label="Conversation",
            height=400,
            show_label=True,
            avatar_images=(None, "🤖")
        )
        
        # Status display
        status_display = gr.Textbox(
            label="Current Status",
            value="",
            interactive=False,
            placeholder="Status messages will appear here..."
        )
        
        # Message input
        with gr.Row():
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Type your question here...",
                scale=4,
                lines=2
            )
            submit_btn = gr.Button("Send 📤", scale=1, variant="primary")
        
        # Metadata display
        with gr.Accordion("📊 Response Metadata", open=False):
            metadata_display = gr.Code(
                label="Metadata (JSON)",
                language="json",
                value="",
                lines=10
            )
        
        # Examples
        gr.Examples(
            examples=[
                ["hi"],
                ["What is the maternity leave policy?"],
                ["What are the working hours?"],
                ["Tell me about vacation policy"],
                ["What is the dress code?"]
            ],
            inputs=msg
        )
        
        # Event handlers
        def check_status():
            health = check_server_health()
            if health:
                status_text = f"✅ Connected - Qdrant: {health.get('qdrant', 'unknown')}, Graphiti: {health.get('graphiti', 'unknown')}"
            else:
                status_text = "❌ Not Connected - Server may be down"
            return status_text
        
        def submit_message(message, history, user_id):
            if not message or not message.strip():
                return history, "", ""
            
            # Add user message to history
            history = history or []
            history.append([message, ""])
            
            # Stream response
            status_msg = ""
            response_text = ""
            metadata = ""
            
            for status, text, meta in stream_query(message, user_id, history):
                status_msg = status
                response_text = text
                metadata = meta
                if history and len(history) > 0:
                    history[-1][1] = response_text
                yield history, status_msg, metadata
        
        # Bind events
        refresh_btn.click(
            fn=check_status,
            outputs=server_status
        )
        
        submit_btn.click(
            fn=submit_message,
            inputs=[msg, chatbot, user_id_input],
            outputs=[chatbot, status_display, metadata_display]
        ).then(
            lambda: "",  # Clear message box
            outputs=msg
        )
        
        msg.submit(
            fn=submit_message,
            inputs=[msg, chatbot, user_id_input],
            outputs=[chatbot, status_display, metadata_display]
        ).then(
            lambda: "",  # Clear message box
            outputs=msg
        )
        
        # Initialize server status
        demo.load(
            fn=check_status,
            outputs=server_status
        )
    
    return demo

if __name__ == "__main__":
    # Check server before starting
    print("🔍 Checking RAG server connection...")
    health = check_server_health()
    if health:
        print(f"✅ RAG server is running: {health}")
    else:
        print("⚠️  Warning: RAG server is not available. Please start the server first.")
        print("   Run: python rag_server.py")
    
    # Create and launch interface
    demo = create_gradio_interface()
    demo.queue()  # Enable queuing for better performance
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
