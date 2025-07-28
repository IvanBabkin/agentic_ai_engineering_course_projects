import queue
import threading
from datetime import datetime
from crewai import Task


class StreamingCapture:
    """Captures and manages streaming API responses and provides formatted logs"""
    
    def __init__(self):
        self.calls = []
        self.current_responses = {}
        self.response_queue = queue.Queue()
        self.lock = threading.Lock()
        
    def clear(self):
        """Clear all captured calls and responses"""
        with self.lock:
            self.calls.clear()
            self.current_responses.clear()
            while not self.response_queue.empty():
                try:
                    self.response_queue.get_nowait()
                except queue.Empty:
                    break
    
    def add_streaming_token(self, model, token, response_id=None, task: Task = None):
        """Add a streaming token to the current response"""
        with self.lock:
            if response_id not in self.current_responses:
                self.current_responses[response_id] = ""
            self.current_responses[response_id] += token
            self.response_queue.put(('token', response_id, token,
                                   self.current_responses[response_id], task))
    
    def complete_response(self, model, response_id, final_response, task: Task = None):
        """Mark a response as complete and add it to the call log"""
        with self.lock:
            self.calls.append({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'model': model or 'unknown',
                'response': final_response[:300] + "..." if len(final_response) > 300 else final_response,
                'task': task
            })
            self.response_queue.put(('complete', response_id, final_response, task))
    
    def get_streaming_updates(self):
        """Get all pending streaming updates from the queue"""
        updates = []
        while not self.response_queue.empty():
            try:
                updates.append(self.response_queue.get_nowait())
            except queue.Empty:
                break
        return updates
    
    def get_formatted_logs(self):
        """Get formatted logs of all API calls"""
        with self.lock:
            if not self.calls:
                return "No API calls logged yet..."
            
            formatted = ""
            for i, call in enumerate(self.calls, 1):
                formatted += f"## 📞 API Call #{i} ({call['timestamp']})\n"
                formatted += f"**Model:** `{call['model']}`\n"
                if call.get('task'):
                    formatted += f"**Task:** `{call['task'].description}`\n"
                formatted += f"**Response:** {call['response']}\n\n---\n\n"
            return formatted.strip()


# Global instance to be used across the application
streaming_capture = StreamingCapture() 