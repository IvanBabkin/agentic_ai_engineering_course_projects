import gradio as gr
import time
import threading
import re
from datetime import datetime
from pathlib import Path

from .crew import Debate
from .logging.log_capture import LogCapture
from .logging.response_capture import capture_streaming_responses
from .logging.streaming_capture import streaming_capture


def format_debate_results_streaming(result, motion):
    """Format debate results using CrewAI's built-in task outputs for streaming"""
    
    # Get all task outputs from CrewAI result
    task_outputs = result.tasks_output
    
    # Build formatted output
    output = []
    
    # Header
    output.append("# 🎯 Debate Results")
    output.append(f"**Motion:** {motion}")
    output.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("")
    output.append("---")
    output.append("")
    
    # Process task outputs based on their position
    # Tasks are in order: FOR1, AGAINST1, FOR2, AGAINST2, JUDGE
    round_num = 1
    
    for i, task_output in enumerate(task_outputs):
        # Determine task type based on position
        if i < len(task_outputs) - 1:  # Not the last task (judge)
            position = "FOR" if i % 2 == 0 else "AGAINST"
            
            # Start new round when we hit a FOR argument
            if position == "FOR":
                output.append(f"## 🥊 Round {round_num}")
                output.append("")
            
            # Add the argument
            emoji = "✅" if position == "FOR" else "❌"
            output.append(f"### {emoji} {position} the Motion")
            output.append(task_output.raw)
            output.append("")
            
            # End round after AGAINST argument
            if position == "AGAINST":
                output.append("---")
                output.append("")
                round_num += 1
        else:
            # Judge's decision (last task)
            output.append("## 🏛️ Judge's Final Decision")
            output.append("")
            output.append(task_output.raw)
            output.append("")
    
    return "\n".join(output)


def convert_markdown_to_plain_text(markdown_text):
    """Convert markdown formatted API logs to plain text for textbox display"""
    if not markdown_text or markdown_text == "No API calls logged yet...":
        return markdown_text
    
    # Convert markdown headers to plain text
    text = re.sub(r'^## (.+)$', r'=== \1 ===', markdown_text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$', r'--- \1 ---', text, flags=re.MULTILINE)
    
    # Convert markdown bold to plain text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    
    # Convert markdown code blocks to plain text
    text = re.sub(r'`(.+?)`', r'"\1"', text)
    
    # Clean up markdown separators
    text = re.sub(r'^---$', r'────────────────────────────────────────', text, flags=re.MULTILINE)
    
    return text


def stream_debate_execution(motion):
    """Stream debate execution using the improved crew approach"""
    if not motion.strip():
        yield "❌ Please enter a debate topic", "❌ Error: No topic provided", "", ""
        return
    
    try:
        # Clear any previous API logs
        streaming_capture.clear()
        
        # Create log capture instance
        log_capture = LogCapture()
        
        # Container for the result and execution status
        execution_result = {'result': None, 'done': False, 'error': None}
        
        def run_debate():
            """Run the debate in a separate thread"""
            try:
                # Start capturing CrewAI logs
                log_capture.start()
                
                # Execute the crew with API call capture
                with capture_streaming_responses():
                    result = Debate().crew().kickoff(inputs={'motion': motion})
                
                execution_result['result'] = result
                execution_result['done'] = True
                
            except Exception as e:
                execution_result['error'] = str(e)
                execution_result['done'] = True
            finally:
                # Stop capturing CrewAI logs
                log_capture.stop()
        
        # Start debate execution in background thread
        debate_thread = threading.Thread(target=run_debate, daemon=True)
        debate_thread.start()
        
        # Stream updates as they come in
        debate_content = "*Debate starting...*"
        current_status = "🔄 Starting debate..."
        
        while not execution_result['done']:
            # Check for streaming API updates
            streaming_updates = streaming_capture.get_streaming_updates()
            
            for update_type, *update_data in streaming_updates:
                if update_type == 'token':
                    response_id, token, partial_response, task = update_data
                    if task:
                        current_status = f"🔄 {task.description[:50]}..."
                
                elif update_type == 'complete':
                    response_id, final_response, task = update_data
                    if task:
                        current_status = f"✅ Completed: {task.description[:50]}..."
            
            # Get current logs
            try:
                execution_logs = log_capture.get_logs()
                api_logs_markdown = streaming_capture.get_formatted_logs()
                # Convert API logs from markdown to plain text for textbox
                api_logs = convert_markdown_to_plain_text(api_logs_markdown)
            except Exception as e:
                execution_logs = f"Error getting logs: {e}"
                api_logs = f"Error getting API logs: {e}"
            
            # Yield current state
            yield (
                debate_content,
                current_status,
                execution_logs or "No execution logs yet...",
                api_logs or "No API calls logged yet..."
            )
            
            time.sleep(0.2)  # Polling interval for responsiveness
        
        # Wait for thread to complete
        debate_thread.join(timeout=10)
        
        # Handle the final result
        if execution_result['error']:
            error_msg = execution_result['error']
            yield (
                f"❌ Error during debate: {error_msg}",
                f"❌ Error: {error_msg}",
                log_capture.get_logs() or f"Error: {error_msg}",
                convert_markdown_to_plain_text(streaming_capture.get_formatted_logs()) or f"Error: {error_msg}"
            )
        elif execution_result['result']:
            # Format the final debate results
            formatted_results = format_debate_results_streaming(execution_result['result'], motion)
            final_status = "✅ Debate complete!"
            
            # Get final logs
            final_execution_logs = log_capture.get_logs()
            final_api_logs_markdown = streaming_capture.get_formatted_logs()
            final_api_logs = convert_markdown_to_plain_text(final_api_logs_markdown)
            
            yield (
                formatted_results,
                final_status,
                final_execution_logs or "No execution logs captured.",
                final_api_logs or "No API calls logged."
            )
        else:
            yield (
                "❌ Debate execution failed without error details",
                "❌ Unknown error occurred",
                log_capture.get_logs() or "No logs captured",
                convert_markdown_to_plain_text(streaming_capture.get_formatted_logs()) or "No API calls logged"
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
    """Create the debate interface with collapsible, identical log displays"""
    
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
                value="Inception is the best movie of all time as of July 2025.",
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
        
        # Log displays - Collapsible and identical text boxes
        with gr.Row():
            # Execution logs in collapsible accordion
            with gr.Column(scale=1):
                with gr.Accordion("📝 Execution Logs", open=False):
                    logs_display = gr.Textbox(
                        value="Execution logs will appear here...",
                        lines=15,
                        interactive=False,
                        max_lines=20,
                        show_label=False,
                        container=False
                    )
            
            # API logs in collapsible accordion - now identical textbox
            with gr.Column(scale=1):
                with gr.Accordion("📡 API Call Logs", open=False):
                    api_logs_display = gr.Textbox(
                        value="API calls will appear here...",
                        lines=15,
                        interactive=False,
                        max_lines=20,
                        show_label=False,
                        container=False
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