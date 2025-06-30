import io
import sys
import re


class LogCapture:
    """Captures stdout logs for display in the UI"""
    
    def __init__(self):
        self.buffer = io.StringIO()
        self.old_stdout = None
        
    def start(self):
        """Start capturing stdout"""
        self.old_stdout = sys.stdout
        sys.stdout = self.buffer
        
    def stop(self):
        """Stop capturing stdout and restore original"""
        if self.old_stdout:
            sys.stdout = self.old_stdout
            
    def get_logs(self):
        """Get captured logs with ANSI codes cleaned"""
        content = self.buffer.getvalue()
        # Clean ANSI codes
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        cleaned = ansi_escape.sub('', content)
        return cleaned.strip() 