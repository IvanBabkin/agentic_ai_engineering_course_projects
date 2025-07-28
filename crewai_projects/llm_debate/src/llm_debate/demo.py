#!/usr/bin/env python
import warnings
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

from llm_debate.crew import Debate
from llm_debate.logging.log_capture import LogCapture
from llm_debate.logging.response_capture import capture_streaming_responses
from llm_debate.logging.streaming_capture import streaming_capture

load_dotenv()
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def format_debate_results(result, motion):
    """Format debate results using CrewAI's built-in task outputs"""
    
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

def format_api_calls(motion):
    """Format API calls into a structured log"""
    output = []
    
    # Header
    output.append("# 🔌 API Calls Log")
    output.append(f"**Motion:** {motion}")
    output.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("")
    output.append("---")
    output.append("")
    
    # Get formatted API logs from streaming capture
    api_logs = streaming_capture.get_formatted_logs()
    
    if api_logs and api_logs.strip() != "No API calls logged yet...":
        output.append(api_logs)
    else:
        output.append("*No API calls were captured during execution*")
        output.append("")
        output.append("This could mean:")
        output.append("- The debate system didn't make any LLM API calls")
        output.append("- API call interception is not working properly")
        output.append("- Calls are being made through a different pathway")
    
    output.append("")
    output.append("---")
    output.append(f"**Total API Calls:** {len(streaming_capture.calls)}")
    
    return "\n".join(output)

def run():
    """Run the debate crew and save formatted results"""
    inputs = {
        'motion': 'Inception is the best movie of all time as of July 2025.',
    }
    
    # Create log capture instance (existing CrewAI logging - unchanged)
    log_capture = LogCapture()
    
    try:
        # Clear any previous API logs
        streaming_capture.clear()
        
        # Start capturing CrewAI logs (existing implementation - unchanged)
        log_capture.start()
        
        # Execute the crew with API call capture
        with capture_streaming_responses():
            result = Debate().crew().kickoff(inputs=inputs)
        
        # Stop capturing CrewAI logs (existing implementation - unchanged)
        log_capture.stop()
        
        # Get captured logs (existing implementation - unchanged)
        captured_logs = log_capture.get_logs()
        
        # Format outputs
        formatted_results = format_debate_results(result, inputs['motion'])
        formatted_api_calls = format_api_calls(inputs['motion'])
        
        # Save to output directory
        Path("output").mkdir(exist_ok=True)
        
        # Existing file outputs (unchanged)
        Path("output/debate_result.md").write_text(formatted_results, encoding='utf-8')
        Path("output/debate_logs.txt").write_text(captured_logs, encoding='utf-8')
        
        # New API calls file
        Path("output/api_calls.txt").write_text(formatted_api_calls, encoding='utf-8')
        
        # Enhanced output messages
        print("Complete debate results saved to output/debate_result.md")
        print("Terminal logs saved to output/debate_logs.txt")
        print("API calls saved to output/api_calls.txt")
        print(f"📊 Captured {len(streaming_capture.calls)} API calls")
        
    except Exception as e:
        # Stop capturing logs even on error
        log_capture.stop()
        raise Exception(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":
    run()
