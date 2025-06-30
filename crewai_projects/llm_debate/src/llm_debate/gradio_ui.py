import gradio as gr
from pathlib import Path
from llm_debate.crew import Debate
from crewai import Agent, Crew, Process, Task
import re
import time
import os
import io
import sys
import contextlib
import json
import threading
from datetime import datetime

# API Call Logger
class APICallLogger:
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()
        
    def clear(self):
        with self.lock:
            self.calls.clear()
    
    def add_call(self, model, messages, response_obj, **kwargs):
        with self.lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Format input messages - handle different message formats
            input_text = ""
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "unknown")
                        content = str(msg.get("content", ""))
                        # Truncate long content
                        if len(content) > 200:
                            content = content[:200] + "..."
                        input_text += f"**{role.upper()}:** {content}\n\n"
            elif isinstance(messages, str):
                content = messages[:200] + "..." if len(messages) > 200 else messages
                input_text = f"**PROMPT:** {content}\n\n"
            
            # Format response - handle different response formats from different providers
            response_text = ""
            if hasattr(response_obj, 'choices') and response_obj.choices:
                # Standard OpenAI/LiteLLM format
                if hasattr(response_obj.choices[0], 'message'):
                    content = str(response_obj.choices[0].message.content or "")
                elif hasattr(response_obj.choices[0], 'text'):
                    content = str(response_obj.choices[0].text or "")
                else:
                    content = str(response_obj.choices[0])
                response_text = content[:300] + "..." if len(content) > 300 else content
            elif hasattr(response_obj, 'content'):
                # Direct content format
                content = str(response_obj.content or "")
                response_text = content[:300] + "..." if len(content) > 300 else content
            elif hasattr(response_obj, 'text'):
                # Text format
                content = str(response_obj.text or "")
                response_text = content[:300] + "..." if len(content) > 300 else content
            elif isinstance(response_obj, str):
                # Direct string response
                response_text = response_obj[:300] + "..." if len(response_obj) > 300 else response_obj
            else:
                # Fallback - convert to string
                response_text = str(response_obj)[:300]
            
            # Extract provider from model name
            provider = "unknown"
            if model:
                if "gpt" in model.lower() or "openai" in model.lower():
                    provider = "OpenAI"
                elif "claude" in model.lower() or "anthropic" in model.lower():
                    provider = "Anthropic"
                elif "gemini" in model.lower() or "google" in model.lower():
                    provider = "Google"
                elif "llama" in model.lower():
                    provider = "Meta"
                else:
                    provider = model.split('/')[0] if '/' in model else "Unknown"
            
            # Format tools if present
            tools_text = ""
            if kwargs.get('tools'):
                tools_count = len(kwargs['tools'])
                tools_text = f"\n**Tools Available:** {tools_count} tools"
            
            call_info = {
                'timestamp': timestamp,
                'model': model or 'unknown',
                'provider': provider,
                'input': input_text.strip(),
                'response': response_text.strip(),
                'tools': tools_text
            }
            self.calls.append(call_info)
    
    def get_formatted_logs(self):
        with self.lock:
            if not self.calls:
                return "No API calls logged yet..."
            
            formatted = ""
            for i, call in enumerate(self.calls, 1):
                formatted += f"## 📞 API Call #{i} ({call['timestamp']})\n\n"
                formatted += f"**Provider:** `{call['provider']}`\n"
                formatted += f"**Model:** `{call['model']}`\n\n"
                
                if call['input']:
                    formatted += f"**Input:**\n{call['input']}\n\n"
                
                if call['response']:
                    formatted += f"**Response:**\n{call['response']}\n\n"
                
                if call['tools']:
                    formatted += f"{call['tools']}\n\n"
                
                formatted += "---\n\n"
            
            return formatted.strip()

# Global logger instance
api_logger = APICallLogger()

@contextlib.contextmanager
def capture_api_calls():
    """Context manager to capture LiteLLM API calls (works with all providers)"""
    original_litellm = None
    original_openai = None
    original_anthropic = None
    
    try:
        # Primary: Patch LiteLLM (used by CrewAI for all providers)
        import litellm
        original_litellm = litellm.completion
        
        def patched_litellm_completion(*args, **kwargs):
            # Extract info before calling
            model = kwargs.get('model', 'unknown')
            messages = kwargs.get('messages', [])
            
            # Call original function
            response = original_litellm(*args, **kwargs)
            
            # Create clean kwargs for logging (remove duplicates)
            log_kwargs = {k: v for k, v in kwargs.items() if k not in ['model', 'messages']}
            
            # Log the call
            api_logger.add_call(model, messages, response, **log_kwargs)
            
            return response
        
        # Apply LiteLLM patch
        litellm.completion = patched_litellm_completion
        
    except ImportError:
        # Fallback: Try patching OpenAI directly
        try:
            import openai
            original_openai = openai.chat.completions.create
            
            def patched_openai_create(*args, **kwargs):
                model = kwargs.get('model', 'unknown')
                messages = kwargs.get('messages', [])
                response = original_openai(*args, **kwargs)
                
                # Create clean kwargs for logging
                log_kwargs = {k: v for k, v in kwargs.items() if k not in ['model', 'messages']}
                api_logger.add_call(model, messages, response, **log_kwargs)
                return response
            
            openai.chat.completions.create = patched_openai_create
            
        except ImportError:
            pass
    
    try:
        # Additional: Patch Anthropic if available
        import anthropic
        original_anthropic = anthropic.Anthropic.messages.create
        
        def patched_anthropic_create(self, *args, **kwargs):
            model = kwargs.get('model', 'claude')
            messages = kwargs.get('messages', [])
            response = original_anthropic(self, *args, **kwargs)
            
            # Create clean kwargs for logging
            log_kwargs = {k: v for k, v in kwargs.items() if k not in ['model', 'messages']}
            api_logger.add_call(model, messages, response, **log_kwargs)
            return response
        
        anthropic.Anthropic.messages.create = patched_anthropic_create
        
    except ImportError:
        pass
    
    try:
        yield api_logger
        
    finally:
        # Restore original functions
        try:
            if original_litellm:
                import litellm
                litellm.completion = original_litellm
        except:
            pass
            
        try:
            if original_openai:
                import openai
                openai.chat.completions.create = original_openai
        except:
            pass
            
        try:
            if original_anthropic:
                import anthropic
                anthropic.Anthropic.messages.create = original_anthropic
        except:
            pass

def split_argument_into_chunks(text, max_sentences=3):
    """Split argument into conversation-sized chunks"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    
    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= max_sentences:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def simulate_conversation(propose_content, oppose_content):
    """Create a back-and-forth conversation from the arguments"""
    conversation = []
    
    # Split arguments into chunks
    for_chunks = split_argument_into_chunks(propose_content, 2)
    against_chunks = split_argument_into_chunks(oppose_content, 2)
    
    # Create alternating conversation
    max_chunks = max(len(for_chunks), len(against_chunks))
    
    for i in range(max_chunks):
        # FOR debater speaks
        if i < len(for_chunks):
            conversation.append(("Debater FOR", for_chunks[i]))
        
        # AGAINST debater responds
        if i < len(against_chunks):
            conversation.append(("Debater AGAINST", against_chunks[i]))
    
    return conversation

def clean_ansi(text):
    """Remove ANSI color codes and format as readable markdown"""
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned = ansi_escape.sub('', text)
    
    # Remove box drawing and excessive dashes
    cleaned = re.sub(r'[╭╮╯╰─│]+', '', cleaned)
    cleaned = re.sub(r'^-+\s*$', '', cleaned, flags=re.MULTILINE)
    
    # Format key sections
    cleaned = re.sub(r'(Crew Execution Started|Crew Execution Completed)', r'## 🚀 \1', cleaned)
    cleaned = re.sub(r'(Agent Started)', r'### 🤖 \1', cleaned)
    cleaned = re.sub(r'(Agent Final Answer)', r'### ✅ \1', cleaned)
    cleaned = re.sub(r'(Task Completed)', r'### 📋 \1', cleaned)
    
    # Clean up extra whitespace and empty lines
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    cleaned = re.sub(r'^\s+', '', cleaned, flags=re.MULTILINE)
    
    return cleaned.strip()

@contextlib.contextmanager
def capture_logs():
    """Capture stdout for CrewAI logs"""
    old_stdout = sys.stdout
    captured = io.StringIO()
    sys.stdout = captured
    try:
        yield captured
    finally:
        sys.stdout = old_stdout

def stream_debate(motion):
    """Stream the debate results using dynamic task workflow"""
    if not motion.strip():
        yield [], "Please enter a motion for the debate.", "**Status:** Waiting for topic", "", ""
        return
    
    try:
        api_logger.clear()
        debate_crew = Debate()
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Clear previous outputs
        for file in output_dir.glob("*.md"):
            file.unlink()
        
        chat_messages = []
        inputs = {'motion': motion}
        all_logs = ""
        
        with capture_api_calls():
            yield chat_messages, "*Debate starting...*", "**Status:** 🔄 Multi-round debate beginning...", all_logs, api_logger.get_formatted_logs()
            
            # Create and run the debate crew
            with capture_logs() as logs:
                crew = debate_crew.crew()
                result = crew.kickoff(inputs=inputs)
            
            all_logs += clean_ansi(logs.getvalue())
            
            # Process debate rounds dynamically
            for round_num in range(1, 4):
                # FOR argument
                for_file = output_dir / f"FOR_round_{round_num}.md"
                if for_file.exists():
                    with open(for_file, 'r', encoding='utf-8') as f:
                        for_content = f.read()
                    chat_messages.append({"role": "assistant", "content": f"**Round {round_num} - FOR**: {for_content}"})
                    yield chat_messages, f"*Round {round_num} continuing...*", f"**Status:** ✅ Round {round_num} FOR complete", all_logs, api_logger.get_formatted_logs()
                
                # AGAINST argument
                against_file = output_dir / f"AGAINST_round_{round_num}.md"
                if against_file.exists():
                    with open(against_file, 'r', encoding='utf-8') as f:
                        against_content = f.read()
                    chat_messages.append({"role": "user", "content": f"**Round {round_num} - AGAINST**: {against_content}"})
                    
                    if round_num < 3:
                        yield chat_messages, f"*Round {round_num + 1} starting...*", f"**Status:** ✅ Round {round_num} complete", all_logs, api_logger.get_formatted_logs()
                    else:
                        yield chat_messages, "*Judge deliberating...*", "**Status:** ✅ All rounds complete", all_logs, api_logger.get_formatted_logs()
            
            # Display judge decision
            decide_file = output_dir / "decide.md"
            if decide_file.exists():
                with open(decide_file, 'r', encoding='utf-8') as f:
                    judge_decision = f.read()
                final_judge = f"## ⚖️ Final Verdict\n\n{judge_decision}"
                yield chat_messages, final_judge, "**Status:** ✅ Debate complete!", all_logs, api_logger.get_formatted_logs()
            else:
                yield chat_messages, "**Error:** No judge decision available", "**Status:** ❌ Error reading decision", all_logs, api_logger.get_formatted_logs()
        
    except Exception as e:
        yield [], f"**Error:** {str(e)}", "**Status:** ❌ Error", all_logs, api_logger.get_formatted_logs()

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
        
        chat_display = gr.Chatbot(
            height=600,
            show_label=False,
            elem_id="debate_chat",
            bubble_full_width=False,
            layout="panel",
            type="messages"
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
            outputs=[chat_display, judge_display, status, logs_display, api_logs_display]
        )
        
        topic_input.submit(
            fn=stream_debate,
            inputs=[topic_input],
            outputs=[chat_display, judge_display, status, logs_display, api_logs_display]
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