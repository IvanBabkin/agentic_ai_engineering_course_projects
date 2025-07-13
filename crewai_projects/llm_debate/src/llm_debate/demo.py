#!/usr/bin/env python
import warnings
from dotenv import load_dotenv
from pathlib import Path

from llm_debate.crew import Debate

load_dotenv()


warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew.
    """
    inputs = {
        'motion': 'Inception is the best movie of all time as of July 2025.',
    }
    
    try:
        result = Debate().crew().kickoff(inputs=inputs)
        
        # Save to output directory
        Path("output").mkdir(exist_ok=True)
        Path("output/debate_result.md").write_text(result.raw, encoding='utf-8')
        print("Results saved to output/debate_result.md")
        
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":
    run()
