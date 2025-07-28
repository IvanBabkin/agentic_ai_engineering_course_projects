"""
LLM Debate application with simplified streaming.

This package provides a debate system where AI agents argue different sides
of a topic with real-time UI updates.
"""

from .logging.log_capture import LogCapture
from .logging.streaming_capture import streaming_capture
from .logging.response_capture import capture_streaming_responses

__all__ = [
    'LogCapture',
    'streaming_capture',
    'capture_streaming_responses'
]