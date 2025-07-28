import contextlib
from .streaming_capture import streaming_capture
from crewai import Task


class ResponseWrapper:
    """Wrapper to capture streaming response while maintaining API compatibility"""
    
    def __init__(self, original_response, model, is_streaming=False, task: Task = None):
        self.original_response = original_response
        self.model = model
        self.is_streaming = is_streaming
        self.response_id = id(original_response)
        self.full_content = ""
        self.task = task
        
        # Eagerly capture non-streaming content
        if not self.is_streaming:
            if hasattr(original_response, 'choices') and original_response.choices:
                content = original_response.choices[0].message.content
                self.full_content = content
                # Mark as complete immediately for non-streaming
                streaming_capture.complete_response(self.model, self.response_id, self.full_content, self.task)
        
        # Copy attributes from original response for compatibility
        if hasattr(original_response, 'choices'):
            self.choices = original_response.choices
        if hasattr(original_response, 'model'):
            self.model = original_response.model
            
    def __iter__(self):
        """Handle streaming responses"""
        if self.is_streaming:
            for chunk in self.original_response:
                # Capture streaming tokens
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        token = delta.content
                        self.full_content += token
                        streaming_capture.add_streaming_token(self.model, token, self.response_id, self.task)
                yield chunk
            # Mark response as complete
            streaming_capture.complete_response(self.model, self.response_id, self.full_content, self.task)
        else:
            # Non-streaming response - ensure it's still yielded properly
            if hasattr(self.original_response, '__iter__'):
                for item in self.original_response:
                    yield item
            else:
                yield self.original_response
    
    def __getattr__(self, name):
        """Delegate all other attributes to original response"""
        return getattr(self.original_response, name)


class LightweightResponseWrapper:
    """Minimal wrapper for production use"""
    def __init__(self, original_response, task=None):
        self.original_response = original_response
        self.task = task
        
    def __iter__(self):
        for chunk in self.original_response:
            yield chunk
    
    def __getattr__(self, name):
        return getattr(self.original_response, name)


@contextlib.contextmanager
def capture_streaming_responses():
    """Capture streaming LLM responses by patching litellm.completion"""
    original_litellm = None
    
    try:
        import litellm
        original_litellm = litellm.completion
        
        def patched_litellm_completion(*args, **kwargs):
            model = kwargs.get('model', 'unknown')
            is_streaming = kwargs.get('stream', False)
            
            # Call original function
            response = original_litellm(*args, **kwargs)
            
            # Return wrapped response
            return ResponseWrapper(response, model, is_streaming)
        
        litellm.completion = patched_litellm_completion
        yield streaming_capture
        
    except ImportError:
        yield streaming_capture
    finally:
        if original_litellm:
            try:
                import litellm
                litellm.completion = original_litellm
            except:
                pass 