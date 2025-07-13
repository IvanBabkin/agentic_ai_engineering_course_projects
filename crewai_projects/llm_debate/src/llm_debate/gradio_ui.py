import gradio as gr
import time
import threading
from datetime import datetime
from .debate_streamer import DebateStreamer


def stream_debate_execution(motion):
    """Stream debate execution using the original DebateStreamer approach"""
    if not motion.strip():
        yield "❌ Please enter a debate topic", "❌ Error: No topic provided", "", ""
        return
    
    try:
        # Use the original DebateStreamer
        streamer = DebateStreamer()
        
        # Container for completion status
        execution_complete = {'done': False}
        
        def run_debate():
            """Run the debate in a separate thread"""
            try:
                streamer.execute_debate(motion)
                execution_complete['done'] = True
            except Exception as e:
                streamer.update_queue.put(('error', str(e)))
                execution_complete['done'] = True
        
        # Start debate execution in background thread
        debate_thread = threading.Thread(target=run_debate, daemon=True)
        debate_thread.start()
        
        # Stream updates as they come in
        debate_content = ""
        current_status = "🔄 Starting debate..."
        
        while not execution_complete['done'] or not streamer.update_queue.empty():
            # Get all pending updates
            updates = streamer.get_updates()
            
            for update_type, update_data in updates:
                if update_type == 'status':
                    current_status = f"🔄 {update_data}"
                    
                elif update_type == 'result':
                    # Format the debate result
                    position = update_data['position']
                    argument = update_data['argument']
                    content = update_data['content']
                    timestamp = update_data['timestamp'].strftime("%H:%M:%S")
                    
                    if position == 'JUDGE':
                        header = f"\n\n## 🏛️ **JUDGE DECISION** ({timestamp})\n\n"
                    else:
                        header = f"\n\n## 🎯 **{position} - Argument {argument}** ({timestamp})\n\n"
                    
                    debate_content += header + content + "\n"
                    
                elif update_type == 'complete':
                    current_status = "✅ Debate complete!"
                    
                elif update_type == 'error':
                    current_status = f"❌ Error: {update_data}"
                    debate_content += f"\n\n❌ **Error:** {update_data}"
            
            # Get current logs
            try:
                execution_logs = streamer.get_logs()
                api_logs = streamer.get_api_logs()
            except Exception as e:
                execution_logs = f"Error getting logs: {e}"
                api_logs = f"Error getting API logs: {e}"
            
            # Yield current state
            yield (
                debate_content or "*Debate starting...*",
                current_status,
                execution_logs or "No execution logs yet...",
                api_logs or "No API calls logged yet..."
            )
            
            time.sleep(0.1)  # Quick polling for responsiveness
        
        # Wait for thread to complete
        debate_thread.join(timeout=10)
        
        # Final update with all logs
        try:
            final_execution_logs = streamer.get_logs()
            final_api_logs = streamer.get_api_logs()
        except Exception as e:
            final_execution_logs = f"Error getting final logs: {e}"
            final_api_logs = f"Error getting final API logs: {e}"
        
        yield (
            debate_content,
            current_status,
            final_execution_logs or "No execution logs captured.",
            final_api_logs or "No API calls logged."
        )
            
    except Exception as e:
        error_msg = str(e)
        yield (
            f"❌ Error starting debate: {error_msg}",
            f"❌ Error: {error_msg}",
            f"Error: {error_msg}",
            f"Error: {error_msg}"
        )


def create_interface():
    """Create the debate interface with proper log displays"""
    
    with gr.Blocks(title="AI Debate Arena", theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.HTML("""
        <div style='text-align: center; padding: 20px;'>
            <h1>🎭 AI Debate Arena</h1>
            <p>Watch AI agents debate any topic with real-time streaming!</p>
        </div>
        """)
        
        # Input section
        with gr.Row():
            topic_input = gr.Textbox(
                label="🎯 Debate Topic",
                placeholder="Enter a topic for debate (e.g., 'Remote work is more productive than office work')",
                value="Dachshunds are the best dog breed",
                scale=3
            )
            start_btn = gr.Button("🚀 Start Debate", variant="primary", scale=1)
        
        # Main debate display
        debate_display = gr.Markdown(
            value="*Enter a topic above and click 'Start Debate' to begin...*",
            label="💬 Live Debate",
            height=400
        )
        
        # Status indicator
        status = gr.Markdown("**Status:** Ready to debate")
        
        # Log displays (restored from original)
        with gr.Row():
            # Execution logs
            with gr.Column(scale=1):
                logs_display = gr.Textbox(
                    value="Execution logs will appear here...",
                    label="📝 Execution Logs",
                    lines=15,
                    interactive=False,
                    max_lines=20
                )
            
            # API logs
            with gr.Column(scale=1):
                api_logs_display = gr.Markdown(
                    value="API calls will appear here...",
                    label="📡 API Call Logs",
                    height=400
                )
        
        # Event handlers
        start_btn.click(
            fn=stream_debate_execution,
            inputs=[topic_input],
            outputs=[debate_display, status, logs_display, api_logs_display]
        )
        
        topic_input.submit(
            fn=stream_debate_execution,
            inputs=[topic_input],
            outputs=[debate_display, status, logs_display, api_logs_display]
        )
    
    return demo


def launch_ui(share=False, server_port=7860, server_name="127.0.0.1"):
    """Launch the debate UI"""
    demo = create_interface()
    demo.launch(
        share=share,
        server_port=server_port,
        server_name=server_name,
        show_error=True
    )


if __name__ == "__main__":
    launch_ui()