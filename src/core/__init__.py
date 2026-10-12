"""Core modules for voice companion."""
from .audio import Speaker
from .config import Config
from .model import ModelInterface, parse_mood_tag, split_sentences
from .recording import record_until_silence

__all__ = [
    'Speaker',
    'Config',
    'ModelInterface',
    'parse_mood_tag',
    'split_sentences',
    'record_until_silence',
]
