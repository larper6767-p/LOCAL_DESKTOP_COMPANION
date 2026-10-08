"""Audio handling: TTS synthesis and playback with caching."""
import io
import os
import re
import time
import queue
import hashlib
import logging
import tempfile
import threading
from typing import Optional, Dict, Tuple

import pygame
from fishaudio import FishAudio
from fishaudio.utils import save

logger = logging.getLogger(__name__)


def clean_for_speech(text: str) -> str:
    """Remove formatting and *action text* that TTS would read out loud."""
    text = re.sub(r"\*[^*]+\*", "", text)  # *giggles* -> removed
    text = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = text.replace("*", "").replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


class TTSCache:
    """LRU cache for synthesized audio."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, bytes] = {}
        self.access_order = []
        self.lock = threading.Lock()
    
    def _key(self, text: str, mood: Optional[str], voice_id: str, model: str) -> str:
        """Generate cache key."""
        return hashlib.md5(f"{text}|{mood}|{voice_id}|{model}".encode()).hexdigest()
    
    def get(self, text: str, mood: Optional[str], voice_id: str, model: str) -> Optional[bytes]:
        """Retrieve from cache."""
        key = self._key(text, mood, voice_id, model)
        with self.lock:
            if key in self.cache:
                self.access_order.remove(key)
                self.access_order.append(key)
                logger.debug(f"TTS cache hit: {text[:30]}...")
                return self.cache[key]
        return None
    
    def put(self, text: str, mood: Optional[str], voice_id: str, model: str, data: bytes) -> None:
        """Store in cache."""
        key = self._key(text, mood, voice_id, model)
        with self.lock:
            if key in self.cache:
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Evict least recently used
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
                logger.debug("TTS cache evicted oldest entry")
            
            self.cache[key] = data
            self.access_order.append(key)


class Speaker:
    """Turns sentences into audio in the background and plays them in order.
    Supports graceful shutdown, caching, and volume control."""
    
    def __init__(self, client: FishAudio, voice_id: str, model: str,
                 emotion_template: Optional[str] = None,
                 volume: float = 0.7,
                 cache_enabled: bool = True,
                 cache_max_size: int = 100):
        pygame.mixer.init()
        pygame.mixer.music.set_volume(volume)
        
        self.client = client
        self.voice_id = voice_id
        self.model = model
        self.emotion_template = emotion_template
        self.enabled = True
        self.volume = volume
        
        self.cache = TTSCache(cache_max_size) if cache_enabled else None
        self.gen = 0
        self.sentences = queue.Queue()
        self.audio = queue.Queue()
        self.shutdown_event = threading.Event()
        
        self.tts_thread = threading.Thread(target=self._tts_loop, daemon=False)
        self.play_thread = threading.Thread(target=self._play_loop, daemon=False)
        self.tts_thread.start()
        self.play_thread.start()
        
        logger.info(f"Speaker initialized (volume={volume}, cache={cache_enabled})")
    
    def say(self, text: str, mood: Optional[str] = None) -> None:
        """Queue text for synthesis and playback."""
        text = clean_for_speech(text)
        if self.enabled and text:
            self.sentences.put((self.gen, text, mood))
            logger.debug(f"Queued: {text[:50]}...")
    
    def stop(self) -> None:
        """Interrupt current playback and clear queues."""
        self.gen += 1
        pygame.mixer.music.stop()
        # Drain queues
        while not self.sentences.empty():
            try:
                self.sentences.get_nowait()
            except queue.Empty:
                break
        while not self.audio.empty():
            try:
                self.audio.get_nowait()
            except queue.Empty:
                break
        logger.info("Speaker stopped and queues cleared")
    
    def set_volume(self, volume: float) -> None:
        """Set playback volume (0.0 to 1.0)."""
        self.volume = max(0.0, min(1.0, volume))
        pygame.mixer.music.set_volume(self.volume)
        logger.info(f"Volume set to {self.volume:.1%}")
    
    def shutdown(self, timeout: float = 10.0) -> None:
        """Gracefully shut down, waiting for queues to drain."""
        logger.info("Shutting down speaker...")
        self.shutdown_event.set()
        self.tts_thread.join(timeout=timeout)
        self.play_thread.join(timeout=timeout)
        if self.tts_thread.is_alive() or self.play_thread.is_alive():
            logger.warning("Speaker threads did not finish in time")
        else:
            logger.info("Speaker shut down cleanly")
    
    def _synth(self, text: str, mood: Optional[str]) -> bytes:
        """Synthesize text to audio, using cache if available."""
        # Check cache
        if self.cache:
            cached = self.cache.get(text, mood, self.voice_id, self.model)
            if cached:
                return cached
        
        # Apply emotion template
        if self.emotion_template and mood:
            text = self.emotion_template.format(mood=mood, text=text)
        
        # Synthesize
        logger.debug(f"Synthesizing: {text[:50]}...")
        audio = self.client.tts.convert(text=text, reference_id=self.voice_id, model=self.model)
        
        # Save to temp file and read back (Fish Audio SDK requirement)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name
        save(audio, path)
        with open(path, "rb") as f:
            data = f.read()
        os.remove(path)
        
        # Cache result
        if self.cache:
            self.cache.put(text, mood, self.voice_id, self.model, data)
        
        return data
    
    def _tts_loop(self) -> None:
        """Background thread: synthesize queued sentences."""
        while not self.shutdown_event.is_set():
            try:
                gen, text, mood = self.sentences.get(timeout=0.5)
            except queue.Empty:
                continue
            
            try:
                if gen == self.gen:
                    data = self._synth(text, mood)
                    self.audio.put((gen, data))
            except Exception as e:
                logger.error(f"TTS synthesis failed for '{text[:30]}...': {e}")
        
        logger.debug("TTS loop exiting")
    
    def _play_loop(self) -> None:
        """Background thread: play synthesized audio."""
        while not self.shutdown_event.is_set():
            try:
                gen, data = self.audio.get(timeout=0.5)
            except queue.Empty:
                continue
            
            try:
                if gen == self.gen:
                    pygame.mixer.music.load(io.BytesIO(data), "mp3")
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy() and gen == self.gen:
                        if self.shutdown_event.is_set():
                            break
                        time.sleep(0.05)
            except Exception as e:
                logger.error(f"Audio playback failed: {e}")
        
        logger.debug("Playback loop exiting")
