#!/usr/bin/env python
import warnings
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

from llm_debate.crew import Debate

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

def run():
    """Run the debate crew and save formatted results"""
    inputs = {
        'motion': 'Inception is the best movie of all time as of July 2025.',
    }
    
    try:
        # Execute the crew
        result = Debate().crew().kickoff(inputs=inputs)
        
        # Format the complete debate results
        formatted_results = format_debate_results(result, inputs['motion'])
        
        # Save to output directory
        Path("output").mkdir(exist_ok=True)
        Path("output/debate_result.md").write_text(formatted_results, encoding='utf-8')
        print("Complete debate results saved to output/debate_result.md")
        
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":
    run()
