"""LLM interface: Cerebras cloud or Ollama local, with streaming, summarization, and retry."""
import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Generator, Any

import ollama
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

from body_tags import resolve_mood

load_dotenv()

logger = logging.getLogger(__name__)

SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")

# ── Graceful error messages in Marin's voice ─────────────────────────────────
_FALLBACK_MESSAGES = [
    "Ugh, my brain just glitched out. Try again?",
    "Okay that definitely didn't work. One more time?",
    "Something broke up there, sorry! What were you saying?",
]
_fallback_idx = 0


def next_fallback() -> str:
    global _fallback_idx
    msg = _FALLBACK_MESSAGES[_fallback_idx % len(_FALLBACK_MESSAGES)]
    _fallback_idx += 1
    return msg


class ModelInterface:
    """Wrapper for Cerebras or Ollama with streaming, token tracking, and summarization."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        use_cerebras: bool = True,
        max_tokens: int = 300,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.use_cerebras = use_cerebras
        self.max_tokens = max_tokens
        self.repeat_penalty = repeat_penalty
        self.stop = stop or []
        self.total_tokens: Dict[str, int] = {"prompt": 0, "completion": 0}

        if use_cerebras:
            api_key = os.environ.get("CEREBRAS_API_KEY")
            if not api_key:
                raise ValueError(
                    "CEREBRAS_API_KEY not found in environment. Add it to your .env file."
                )
            self.cerebras_client = Cerebras(api_key=api_key)
            logger.info(f"Cerebras API ready — model: {self.model_name}")
            # Cerebras allows at most 4 stop sequences
            if len(self.stop) > 4:
                logger.warning(
                    f"Cerebras only supports 4 stop sequences; trimming from {len(self.stop)} to 4"
                )
                self.stop = self.stop[:4]
        else:
            self.cerebras_client = None
            self._validate_ollama_model()

    # ── Public API ────────────────────────────────────────────────────────────

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream chat completion. Yields Ollama-compatible chunk dicts.

        Each chunk: {"message": {"content": str}, "done": bool}
        Raises on failure after logging — caller should catch and use next_fallback().
        """
        temp = temperature if temperature is not None else self.temperature

        if self.use_cerebras:
            yield from self._stream_cerebras(messages, temp)
        else:
            yield from self._stream_ollama(messages, temp)

    def summarize(
        self,
        messages: List[Dict[str, str]],
        character_name: str = "the assistant",
    ) -> str:
        """Summarize a block of conversation history into a compact paragraph.

        Used to compress old turns before they would be dropped.
        Returns the summary string, or an empty string on failure.
        """
        if not messages:
            return ""

        transcript = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages
        )
        prompt = (
            f"Summarize the following conversation between a user and {character_name} "
            "in 3–5 concise bullet points, preserving key facts, events, and emotional moments. "
            "Write only the bullet points, nothing else.\n\n"
            f"{transcript}"
        )
        summary_messages = [{"role": "user", "content": prompt}]

        # Use a low temperature for factual summarization
        try:
            full = ""
            for chunk in self.stream_chat(summary_messages, temperature=0.2):
                full += chunk["message"]["content"]
            summary = full.strip()
            logger.info(f"Summarized {len(messages)} messages into {len(summary)} chars")
            return summary
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return ""

    def get_token_stats(self) -> Dict[str, int]:
        return self.total_tokens.copy()

    def reset_token_stats(self) -> None:
        self.total_tokens = {"prompt": 0, "completion": 0}

    # ── Private ───────────────────────────────────────────────────────────────

    def _stream_cerebras(
        self, messages: List[Dict[str, str]], temp: float
    ) -> Generator[Dict[str, Any], None, None]:
        try:
            kwargs: Dict[str, Any] = dict(
                model=self.model_name,
                messages=messages,
                stream=True,
                temperature=temp,
                max_completion_tokens=self.max_tokens,
                # Disable Qwen-3 reasoning — we don't need chain-of-thought
                # for a voice companion and it causes empty delta.content until
                # the entire think block finishes.
                extra_body={"reasoning_effort": "low"},
            )
            if self.stop:
                kwargs["stop"] = self.stop

            stream = self.cerebras_client.chat.completions.create(**kwargs)

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # reasoning chunks arrive in delta.reasoning — skip them silently.
                # actual response text arrives in delta.content once thinking is done.
                content = (delta.content or "") if hasattr(delta, "content") else ""
                done = chunk.choices[0].finish_reason is not None
                if not content and not done:
                    continue  # pure reasoning token, nothing to yield yet
                yield {"message": {"content": content}, "done": done}

                if hasattr(chunk, "usage") and chunk.usage:
                    if hasattr(chunk.usage, "prompt_tokens"):
                        self.total_tokens["prompt"] = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, "completion_tokens"):
                        self.total_tokens["completion"] = chunk.usage.completion_tokens

        except Exception as e:
            logger.error(f"Cerebras streaming failed: {e}")
            raise

    def _stream_ollama(
        self, messages: List[Dict[str, str]], temp: float
    ) -> Generator[Dict[str, Any], None, None]:
        try:
            opts: Dict[str, Any] = {
                "temperature": temp,
                "num_ctx": 2048,
                "num_gpu": 1,
                "repeat_penalty": self.repeat_penalty,
            }
            if self.stop:
                opts["stop"] = self.stop

            for chunk in ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                options=opts,
            ):
                yield chunk
                if "prompt_eval_count" in chunk:
                    self.total_tokens["prompt"] += chunk["prompt_eval_count"]
                if "eval_count" in chunk:
                    self.total_tokens["completion"] += chunk["eval_count"]

        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise

    def _validate_ollama_model(self) -> None:
        try:
            result = ollama.list()
            available = [m.model for m in result.get("models", [])]
            if not any(self.model_name.startswith(n) for n in available):
                logger.warning(
                    f"Model '{self.model_name}' not in Ollama. "
                    f"Available: {', '.join(available)}"
                )
            else:
                logger.info(f"Ollama model '{self.model_name}' validated")
        except Exception as e:
            logger.warning(f"Could not validate Ollama model (is Ollama running?): {e}")


# ── Text helpers ──────────────────────────────────────────────────────────────

def parse_mood_tag(text: str) -> Tuple[Optional[str], str]:
    """Extract a leading mood tag.  Returns (mood_or_None, remaining_text).

    Handles both the legacy bare form (``[joyful]``) and the explicit form
    (``[mood:joyful]`` / alias ``[mood:happy]``) through ONE shared resolver
    (body_tags.resolve_mood), so they all resolve identically.  Unknown or
    disabled moods return None and the bracket is NOT stripped.
    """
    text = text.lstrip()
    if text.startswith("[") and "]" in text:
        end = text.index("]")
        raw = text[1:end].strip()
        if raw and raw.lower() != "clear":
            mood = resolve_mood(raw)
            if mood is not None:
                return mood, text[end + 1:].lstrip()
        return None, text
    return None, text


def split_sentences(text: str, min_length: int = 20) -> List[str]:
    """Split into [complete_part, incomplete_tail] for incremental TTS.

    Returns a one-element list when there is nothing to split yet.
    """
    parts = SENTENCE_END.split(text)
    if len(parts) > 1 and len(" ".join(parts[:-1])) >= min_length:
        return [" ".join(parts[:-1]), parts[-1]]
    return [text]
