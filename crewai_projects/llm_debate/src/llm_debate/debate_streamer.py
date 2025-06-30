import threading
import queue
import time
from crewai import Task
from datetime import datetime
from .crew import Debate
from .logging.log_capture import LogCapture
from .logging.response_capture import capture_streaming_responses
from .logging.streaming_capture import streaming_capture
import asyncio
from typing import Dict, List, Any
from collections import deque
import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

class DebateStreamer:
    """Optimized debate execution system with improved performance"""
    
    def __init__(self):
        self.update_queue = queue.Queue()
        self.log_capture = LogCapture()
        self.results: Dict[str, str] = {}
        self.debate_crew = None  # Store crew instance
        
    def execute_debate(self, motion):
        """Execute debate tasks sequentially and stream results"""
        try:
            # Clear previous API call logs
            streaming_capture.clear()
            
            self.log_capture.start()
            
            # Use the API capture context manager for logging
            with capture_streaming_responses():
                # Initialize debate crew once and store it
                self.debate_crew = Debate()
                
                # Create individual tasks manually for sequential execution
                tasks = self._create_task_configs(motion)
                
                # Execute each task and stream results
                for task_info in tasks:
                    self._execute_and_stream_task(task_info, motion)
                
            self.update_queue.put(('complete', None))
            
        except Exception as e:
            self.update_queue.put(('error', str(e)))
        finally:
            self.log_capture.stop()
    
    def _create_task_configs(self, motion: str, num_arguments: int = 3) -> List[Dict[str, Any]]:
        """Create task configurations more efficiently"""
        configs = []
        
        for i in range(1, num_arguments + 1):
            # FOR argument
            configs.append({
                'position': 'FOR',
                'argument': i,
                'description': self._format_description('FOR', i, motion),
                'context': f'against_a{i-1}' if i > 1 else None
            })
            
            # AGAINST argument  
            configs.append({
                'position': 'AGAINST',
                'argument': i,
                'description': self._format_description('AGAINST', i, motion),
                'context': f'for_a{i}'
            })
        
        # Judge decision
        configs.append({
            'position': 'JUDGE',
            'argument': 'FINAL',
            'description': f"Review all arguments and decide which side won for: {motion}",
            'context': 'all_arguments'
        })
        
        return configs
    
    def _format_description(self, position: str, arg_num: int, motion: str) -> str:
        """Format task description more efficiently"""
        base = f"Argument {arg_num}: You are arguing {position} the motion: '{motion}'. "
        return base + (
            "Present your opening argument." if position == 'FOR' and arg_num == 1 else
            "Respond to your opponent's most recent argument and then present your own points."
        )
    
    def _execute_and_stream_task(self, task_info, motion):
        """Execute a single task and stream its result"""
        # Send status update
        position = task_info['position']
        argument_num = task_info['argument']
        
        if position == 'JUDGE':
            status = "Judge is deliberating..."
        else:
            status = f"Argument {argument_num}: {position} is responding..."
        
        self.update_queue.put(('status', status))
        
        # Build context from previous responses if needed
        context_text = self._build_context(task_info['context'])
        full_description = task_info['description']
        if context_text:
            full_description += f"\n\nPrevious arguments:\n{context_text}"
        
        # Get the appropriate agent based on position
        agent = self.debate_crew.judge() if position == 'JUDGE' else self.debate_crew.debater()
        
        # Create and execute task
        task = Task(
            description=full_description,
            expected_output=f"Your compelling argument {position.lower()} the motion in argument {argument_num}." if position != 'JUDGE' else "Your final decision on which side won the debate with detailed reasoning.",
            agent=agent
        )
        
        # Execute task and extract string content from TaskOutput
        task_output = task.execute_sync()
        
        # Extract the actual text content from TaskOutput object
        if hasattr(task_output, 'raw'):
            result = task_output.raw
        elif hasattr(task_output, 'content'):
            result = task_output.content
        elif hasattr(task_output, 'result'):
            result = task_output.result
        else:
            # Fallback - convert to string
            result = str(task_output)
        
        # Ensure result is a string
        if not isinstance(result, str):
            result = str(result)
        
        # Store result for context building
        self._store_result(task_info, result)
        
        # Stream the result
        self.update_queue.put(('result', {
            'position': position,
            'argument': argument_num,
            'content': result,
            'timestamp': datetime.now()
        }))
    
    def _build_context(self, context_key: str) -> str:
        """Optimized context building"""
        if not context_key or not self.results:
            return ""
        
        if context_key == 'all_arguments':
            # Use list comprehension and join for efficiency
            context_parts = []
            for i in range(1, 4):
                for position in ['for', 'against']:
                    key = f"{position}_a{i}"
                    if key in self.results:
                        position_label = position.upper()
                        context_parts.append(f"{position_label} (Argument {i}): {self.results[key]}")
            return "\n\n".join(context_parts)
        
        return self.results.get(context_key, "")
    
    def _store_result(self, task_info, result):
        """Store result for context building"""
        if not hasattr(self, 'results'):
            self.results = {}
        
        position = task_info['position'].lower()
        argument_num = task_info['argument']
        
        if position == 'judge':
            key = 'judge'
        else:
            key = f"{position}_a{argument_num}"
        
        self.results[key] = result
    
    def get_updates(self):
        """Get all pending updates from the queue"""
        updates = []
        while not self.update_queue.empty():
            try:
                updates.append(self.update_queue.get_nowait())
            except queue.Empty:
                break
        return updates
    
    def get_logs(self):
        """Get captured logs"""
        return self.log_capture.get_logs()
    
    def get_api_logs(self):
        """Get formatted API call logs"""
        return streaming_capture.get_formatted_logs()

class StreamingCapture:
    """Optimized streaming capture with better thread safety"""
    
    def __init__(self):
        self.calls = deque()  # More efficient for frequent append operations
        self.current_responses = {}
        self.response_queue = queue.Queue()
        self.lock = threading.RLock()  # Reentrant lock for nested operations
        
    def get_streaming_updates(self):
        """Efficiently drain queue with timeout"""
        updates = []
        try:
            while True:
                update = self.response_queue.get_nowait()
                updates.append(update)
        except queue.Empty:
            pass
        return updates
    
    def get_formatted_logs(self):
        """Optimized log formatting"""
        with self.lock:
            if not self.calls:
                return "No API calls logged yet..."
            
            # Use list comprehension and join for better performance
            log_parts = []
            for i, call in enumerate(self.calls, 1):
                parts = [
                    f"## 📞 API Call #{i} ({call['timestamp']})",
                    f"**Model:** `{call['model']}`"
                ]
                
                if call.get('task'):
                    parts.append(f"**Task:** `{call['task'].description}`")
                
                parts.append(f"**Response:** {call['response']}")
                log_parts.append("\n".join(parts))
            
            return "\n\n---\n\n".join(log_parts) 

def run():
    """Run the crew with improved error handling."""
    inputs = {'motion': 'Claude is the best LLM'}
    
    # Let CrewAI exceptions propagate naturally with their original context
    result = Debate().crew().kickoff(inputs=inputs)
    print(result.raw)

if __name__ == "__main__":
    run() 