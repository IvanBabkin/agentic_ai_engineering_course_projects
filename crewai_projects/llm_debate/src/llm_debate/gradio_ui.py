import gradio as gr
import time
import threading
import queue
from datetime import datetime
from .debate_streamer import DebateStreamer


def run_debate_in_background(streamer, motion, result_queue):
    """Run debate in background thread"""
    try:
        streamer.execute_debate(motion)
        result_queue.put(('success', 'Debate completed'))
    except Exception as e:
        result_queue.put(('error', str(e)))


def stream_debate(motion):
    """Stream debate execution with real-time updates"""
    
    # Initialize state
    debate_transcript = ""
    judge_decision = "*The judge will render their decision after the debate concludes.*"
    status = "**Status:** Starting debate..."
    
    # Create streamer and result queue
    streamer = DebateStreamer()
    result_queue = queue.Queue()
    
    try:
        # Initial yield
        yield debate_transcript, judge_decision, status, "", ""
        
        # Start debate in background thread
        debate_thread = threading.Thread(
            target=run_debate_in_background,
            args=(streamer, motion, result_queue)
        )
        debate_thread.start()
        
        # Monitor for updates
        last_update = time.time()
        
        while debate_thread.is_alive() or not result_queue.empty():
            # Process streamer updates
            updates = streamer.get_updates()
            
            for update_type, data in updates:
                if update_type == 'status':
                    status = f"**Status:** {data}"
                    
                elif update_type == 'result':
                    # Format and add to transcript
                    position = data['position']
                    round_num = data['round']
                    content = data['content']
                    
                    if position == 'JUDGE':
                        judge_decision = content
                    else:
                        heading = f"### Round {round_num}: {'Proponent' if position == 'FOR' else 'Opponent'} of '{motion}'"
                        debate_transcript += f"\n\n---\n\n{heading}\n\n{content}"
                
                elif update_type == 'complete':
                    status = "**Status:** ✅ Debate completed successfully!"
                    
                elif update_type == 'error':
                    status = f"**Status:** ❌ Error: {data}"
            
            # Check for thread completion
            if not debate_thread.is_alive():
                try:
                    result_type, message = result_queue.get_nowait()
                    if result_type == 'error':
                        status = f"**Status:** ❌ Error: {message}"
                    elif status.startswith("**Status:** ❌") == False:  # Don't override error status
                        status = "**Status:** ✅ Debate completed successfully!"
                except queue.Empty:
                    pass
            
            # Periodic UI updates (throttled)
            if time.time() - last_update > 0.2:
                yield debate_transcript, judge_decision, status, streamer.get_logs(), streamer.get_api_logs()
                last_update = time.time()
            
            time.sleep(0.1)
        
        # Wait for thread to complete
        debate_thread.join()
        
    except Exception as e:
        status = f"**Status:** ❌ Unexpected error: {str(e)}"
    
    # Final yield with both log types
    yield debate_transcript, judge_decision, status, streamer.get_logs(), streamer.get_api_logs()


def create_interface():
    """Create the debate interface"""
    
    with gr.Blocks(title="AI Debate Arena", theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.HTML("""
        <div style='text-align: center; padding: 20px;'>
            <h1>🎭 AI Debate Arena</h1>
            <p>Two AI agents debate any topic you choose!</p>
        </div>
        """)
        
        # Input section
        with gr.Row():
            topic_input = gr.Textbox(
                label="🎯 Debate Topic",
                placeholder="What should the AIs debate about?",
                value="Dachshunds are the best dog breed",
                scale=3
            )
            start_btn = gr.Button("🚀 Start Debate", variant="primary", scale=1)
        
        # Chat area
        gr.Markdown("## 💬 Live Debate")
        
        debate_display = gr.Markdown(
            value="*The debate will appear here as it unfolds...*",
            elem_id="debate_display"
        )
        
        # Judge section
        gr.Markdown("## ⚖️ Judge's Verdict")
        judge_display = gr.Markdown(
            value="*The judge will render their decision after the debate concludes.*",
            show_label=False
        )
        
        # Status indicator
        status = gr.Markdown("**Status:** Ready to debate", visible=True)
        
        # CrewAI logs (collapsible)
        with gr.Accordion("CrewAI Execution Logs", open=False):
            logs_display = gr.Textbox(
                value="Logs will appear here...",
                lines=20,
                max_lines=20,
                interactive=False,
                show_label=False
            )
        
        # API Call logs (collapsible)
        with gr.Accordion("API Call Logs", open=False):
            api_logs_display = gr.Markdown(
                value="API calls will appear here...",
                show_label=False
            )
        
        # Connect the interface
        start_btn.click(
            fn=stream_debate,
            inputs=[topic_input],
            outputs=[debate_display, judge_display, status, logs_display, api_logs_display]
        )
        
        topic_input.submit(
            fn=stream_debate,
            inputs=[topic_input],
            outputs=[debate_display, judge_display, status, logs_display, api_logs_display]
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