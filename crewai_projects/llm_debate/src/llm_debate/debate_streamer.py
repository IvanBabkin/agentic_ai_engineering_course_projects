import threading
import queue
import time
from crewai import Agent, Task
from datetime import datetime
from .crew import Debate
from .logging.log_capture import LogCapture
from .logging.response_capture import capture_streaming_responses
from .logging.streaming_capture import streaming_capture


class DebateStreamer:
    """Simple debate execution system that streams results as tasks complete"""
    
    def __init__(self):
        self.update_queue = queue.Queue()
        self.log_capture = LogCapture()
        
    def execute_debate(self, motion):
        """Execute debate tasks sequentially and stream results"""
        try:
            # Clear previous API call logs
            streaming_capture.clear()
            
            self.log_capture.start()
            
            # Use the API capture context manager for logging
            with capture_streaming_responses():
                # Initialize debate crew
                debate_crew = Debate()
                
                # Create individual tasks manually for sequential execution
                tasks = self._create_individual_tasks(debate_crew, motion)
                
                # Execute each task and stream results
                for task_info in tasks:
                    self._execute_and_stream_task(task_info, motion)
                
            self.update_queue.put(('complete', None))
            
        except Exception as e:
            self.update_queue.put(('error', str(e)))
        finally:
            self.log_capture.stop()
    
    def _create_individual_tasks(self, crew, motion):
        """Create list of tasks to execute sequentially"""
        
        # Get agents
        debater = crew.debater()
        judge = crew.judge()
        
        tasks = []
        
        # Round 1
        tasks.append({
            'agent': debater,
            'description': f"Round 1: You are arguing FOR the motion: {motion}. Present your opening argument.",
            'position': 'FOR',
            'round': 1,
            'context': None
        })
        
        tasks.append({
            'agent': debater,
            'description': f"Round 1: You are arguing AGAINST the motion: {motion}. Respond to your opponent's most recent argument and then present your own points.",
            'position': 'AGAINST', 
            'round': 1,
            'context': 'for_r1'
        })
        
        # Round 2
        tasks.append({
            'agent': debater,
            'description': f"Round 2: You are arguing FOR the motion: {motion}. Respond to your opponent's most recent argument and then present your own points.",
            'position': 'FOR',
            'round': 2,
            'context': 'against_r1'
        })
        
        tasks.append({
            'agent': debater,
            'description': f"Round 2: You are arguing AGAINST the motion: {motion}. Respond to your opponent's most recent argument and then present your own points.",
            'position': 'AGAINST',
            'round': 2,
            'context': 'for_r2'
        })
        
        # Round 3
        tasks.append({
            'agent': debater,
            'description': f"Round 3: You are arguing FOR the motion: {motion}. Respond to your opponent's most recent argument and then present your own points.",
            'position': 'FOR',
            'round': 3,
            'context': 'against_r2'
        })
        
        tasks.append({
            'agent': debater,
            'description': f"Round 3: You are arguing AGAINST the motion: {motion}. Respond to your opponent's most recent argument and then present your own points.",
            'position': 'AGAINST',
            'round': 3,
            'context': 'for_r3'
        })
        
        # Judge decision
        tasks.append({
            'agent': judge,
            'description': f"Review all arguments from both sides across all rounds and decide which side is more convincing for the motion: {motion}. Consider the strength of arguments, rebuttals, and overall debate performance.",
            'position': 'JUDGE',
            'round': 'FINAL',
            'context': 'all_rounds'
        })
        
        return tasks
    
    def _execute_and_stream_task(self, task_info, motion):
        """Execute a single task and stream its result"""
        # Send status update
        position = task_info['position']
        round_num = task_info['round']
        
        if position == 'JUDGE':
            status = "Judge is deliberating..."
        else:
            status = f"Round {round_num}: {position} is responding..."
        
        self.update_queue.put(('status', status))
        
        # Build context from previous responses if needed
        context_text = self._build_context(task_info['context'])
        full_description = task_info['description']
        if context_text:
            full_description += f"\n\nPrevious arguments:\n{context_text}"
        
        # Create and execute task
        task = Task(
            description=full_description,
            expected_output=f"Your compelling argument {position.lower()} the motion in round {round_num}." if position != 'JUDGE' else "Your final decision on which side won the debate with detailed reasoning.",
            agent=task_info['agent']
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
            'round': round_num,
            'content': result,
            'timestamp': datetime.now()
        }))
    
    def _build_context(self, context_key):
        """Build context string from previous results"""
        if not context_key or not hasattr(self, 'results'):
            return ""
        
        if context_key == 'all_rounds':
            # For judge, include all previous arguments
            context_parts = []
            for key, result in self.results.items():
                if key != 'judge':
                    context_parts.append(f"{key.upper()}: {result}")
            return "\n\n".join(context_parts)
        elif context_key in self.results:
            return self.results[context_key]
        
        return ""
    
    def _store_result(self, task_info, result):
        """Store result for context building"""
        if not hasattr(self, 'results'):
            self.results = {}
        
        position = task_info['position'].lower()
        round_num = task_info['round']
        
        if position == 'judge':
            key = 'judge'
        else:
            key = f"{position}_r{round_num}"
        
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