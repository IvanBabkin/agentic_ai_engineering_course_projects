#!/usr/bin/env python
import warnings
from dotenv import load_dotenv

from llm_debate.crew import Debate

load_dotenv()


warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'motion': 'Claude is the best LLM',
    }
    
    try:
        result = Debate().crew().kickoff(inputs=inputs)
        print(result.raw)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

def execute_debate(self, motion):
    """Execute debate using CrewAI's native orchestration"""
    try:
        streaming_capture.clear()
        self.log_capture.start()
        
        with capture_streaming_responses():
            debate_crew = Debate()
            inputs = {'motion': motion}
            
            # Use CrewAI's built-in streaming execution
            result = debate_crew.crew().kickoff(inputs=inputs)
            self.update_queue.put(('complete', result.raw))
            
    except Exception as e:
        self.update_queue.put(('error', str(e)))
    finally:
        self.log_capture.stop()
