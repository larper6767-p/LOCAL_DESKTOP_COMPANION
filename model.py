"""Model interaction with error recovery and streaming."""
import re
import logging
from typing import Dict, List, Optional, Tuple, Generator, Any

import ollama

logger = logging.getLogger(__name__)

# Sentence boundary detection
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


class ModelInterface:
    """Wrapper for Ollama with error recovery and token tracking."""
    
    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.total_tokens = {"prompt": 0, "completion": 0}
        self._validate_model()
    
    def _validate_model(self) -> None:
        """Check if model exists in Ollama."""
        try:
            result = ollama.list()
            # ollama.list() returns a ListResponse object with 'models' attribute
            available = [m.model for m in result.get("models", [])]
            if not any(self.model_name.startswith(name) for name in available):
                logger.warning(
                    f"Model '{self.model_name}' not found in Ollama. "
                    f"Available models: {', '.join(available)}"
                )
            else:
                logger.info(f"Model '{self.model_name}' validated")
        except Exception as e:
            logger.warning(f"Could not validate model (is Ollama running?): {e}")
    
    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream chat completion from Ollama.
        
        Yields:
            Chunk dicts from ollama.chat(stream=True)
        
        Raises:
            Exception: If streaming fails
        """
        temp = temperature if temperature is not None else self.temperature
        
        try:
            for chunk in ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                options={
                    "temperature": temp,
                    "num_ctx": 2048,  # Reduce context window to save GPU memory
                    "num_gpu": 1,     # Ensure using GPU
                }
            ):
                yield chunk
                
                # Track tokens if available
                if "prompt_eval_count" in chunk:
                    self.total_tokens["prompt"] += chunk["prompt_eval_count"]
                if "eval_count" in chunk:
                    self.total_tokens["completion"] += chunk["eval_count"]
        
        except Exception as e:
            logger.error(f"Model streaming failed: {e}")
            raise
    
    def get_token_stats(self) -> Dict[str, int]:
        """Return cumulative token usage."""
        return self.total_tokens.copy()
    
    def reset_token_stats(self) -> None:
        """Reset token counters."""
        self.total_tokens = {"prompt": 0, "completion": 0}


def parse_mood_tag(text: str) -> Tuple[Optional[str], str]:
    """Extract leading [mood] tag if present.
    
    Returns:
        (mood, text_without_tag)
    """
    text = text.lstrip()
    if text.startswith("[") and "]" in text:
        end = text.index("]")
        mood = text[1:end].strip().lower()
        remaining = text[end + 1:].lstrip()
        return mood, remaining
    return None, text


def split_sentences(text: str, min_length: int = 20) -> List[str]:
    """Split text into sentences for incremental TTS.
    
    Args:
        text: Input text
        min_length: Minimum length before splitting
    
    Returns:
        List of sentences (may include incomplete final sentence)
    """
    parts = SENTENCE_END.split(text)
    if len(parts) > 1 and len(" ".join(parts[:-1])) >= min_length:
        complete = " ".join(parts[:-1])
        remaining = parts[-1]
        return [complete, remaining]
    return [text]
