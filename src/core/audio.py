"""Audio: TTS synthesis, LRU cache, playback, emotion, speed control, system ducking."""
import io
import os
import re
import time
import queue
import hashlib
import logging
import tempfile
import threading
from typing import Callable, Optional, Dict, List

import pygame
from fishaudio import FishAudio
from fishaudio.utils import save

logger = logging.getLogger(__name__)

#  Windows audio ducking via pycaw 
try:
    from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume  # type: ignore
    _PYCAW_AVAILABLE = True
except ImportError:
    _PYCAW_AVAILABLE = False
    logger.debug("pycaw not installed — audio ducking disabled. pip install pycaw")


def _get_system_sessions():
    """Return all active non-companion audio sessions."""
    if not _PYCAW_AVAILABLE:
        return []
    try:
        sessions = AudioUtilities.GetAllSessions()
        return [s for s in sessions if s.Process and s.Process.name() != "python.exe"]
    except Exception:
        return []


def _duck(target_volume: float) -> None:
    """Lower all other application volumes."""
    if not _PYCAW_AVAILABLE:
        return
    try:
        for session in _get_system_sessions():
            volume = session._ctl.QueryInterface(ISimpleAudioVolume)
            volume.SetMasterVolume(target_volume, None)
    except Exception as e:
        logger.debug(f"Duck failed: {e}")


def _unduck() -> None:
    """Restore all application volumes to 1.0."""
    if not _PYCAW_AVAILABLE:
        return
    try:
        for session in _get_system_sessions():
            volume = session._ctl.QueryInterface(ISimpleAudioVolume)
            volume.SetMasterVolume(1.0, None)
    except Exception as e:
        logger.debug(f"Unduck failed: {e}")


# ── Lip-sync envelope decoding 

def decode_envelope(data: bytes, hz: float = 30.0):
    """Decode ``data`` (MP3/bytes) into a (envelope, frame_ms) loudness profile.

    Uses ``pygame.mixer.Sound`` + ``pygame.sndarray.samples`` and feeds the raw
    PCM through ``lipsync.build_envelope`` (RMS per ~33 ms frame, peak-
    normalised to 0..1).  Returns ``None`` on ANY failure so audio playback is
    never blocked — the caller simply falls back to the random-mouth flap.

    This runs on the TTS-play thread (off the GUI), so the heavy decode never
    touches the render thread.  numpy is imported lazily here so this module can
    be imported without numpy for the rest of the app.
    """
    if not data:
        return None
    try:
        import io as _io
        from lipsync import build_envelope

        snd = pygame.mixer.Sound(_io.BytesIO(data))
        samples = pygame.sndarray.samples(snd)
        # Derive the real sample rate from the decoded sample count and duration
        length_s = float(snd.get_length())
        if length_s > 0 and samples.shape[0] > 0:
            sample_rate = int(samples.shape[0] / length_s)
        else:
            sample_rate = 44100
        if sample_rate <= 0:
            sample_rate = 44100
        env, frame_ms = build_envelope(samples, sample_rate, hz)
        if not env:
            return None
        return list(env), float(frame_ms)
    except Exception as e:
        logger.debug(f"Lip-sync envelope decode failed (using flap): {e}")
        return None


# Text pre-processing 

def clean_for_speech(text: str) -> str:
    """Remove markdown / action text that TTS would read literally."""
    text = re.sub(r"\*[^*]+\*", "", text)                      # *giggles*
    text = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", text)           # __bold__
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)     # ## Heading
    text = re.sub(r"^\s*[-+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"`(.+?)`", r"\1", text)                     # `code`
    text = text.replace("*", "").replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


#  LRU cache

class TTSCache:
    """Thread-safe LRU cache for synthesised audio bytes."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, bytes] = {}
        self.order: list = []
        self.lock = threading.Lock()

    def _key(self, text: str, mood: Optional[str], voice_id: str, model: str, speed: float) -> str:
        raw = f"{text}|{mood}|{voice_id}|{model}|{speed:.2f}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, text: str, mood: Optional[str], voice_id: str, model: str, speed: float) -> Optional[bytes]:
        key = self._key(text, mood, voice_id, model, speed)
        with self.lock:
            if key in self.cache:
                self.order.remove(key)
                self.order.append(key)
                logger.debug(f"TTS cache hit: {text[:40]!r}")
                return self.cache[key]
        return None

    def put(self, text: str, mood: Optional[str], voice_id: str, model: str, speed: float, data: bytes) -> None:
        key = self._key(text, mood, voice_id, model, speed)
        with self.lock:
            if key in self.cache:
                self.order.remove(key)
            elif len(self.cache) >= self.max_size:
                oldest = self.order.pop(0)
                del self.cache[oldest]
            self.cache[key] = data
            self.order.append(key)


#  Speaker

class Speaker:
    """Synthesises and plays TTS audio with caching, emotion, speed, and ducking."""

    def __init__(
        self,
        client: FishAudio,
        voice_id: str,
        model: str,
        emotion_template: Optional[str] = None,
        volume: float = 0.7,
        speed: float = 1.0,
        cache_enabled: bool = True,
        cache_max_size: int = 100,
        ducking_enabled: bool = True,
        ducking_volume: float = 0.3,
        lip_sync_enabled: bool = True,
    ):
        pygame.mixer.init()
        pygame.mixer.music.set_volume(volume)

        self.client = client
        self.voice_id = voice_id
        self.model = model
        self.emotion_template = emotion_template
        self.enabled = True
        self.volume = volume
        self.speed = max(0.5, min(2.0, speed))
        self.ducking_enabled = ducking_enabled and _PYCAW_AVAILABLE
        self.ducking_volume = ducking_volume
        # Phase 4a: decode each clip's loudness envelope for real lip sync.  When
        # off, the mouth keeps its random flap.  Tunable at runtime.
        self._lip_sync_enabled = bool(lip_sync_enabled)

        self.cache = TTSCache(cache_max_size) if cache_enabled else None
        self.gen = 0                        # increments on stop() to cancel queued items
        self.sentences: queue.Queue = queue.Queue()
        self.audio: queue.Queue = queue.Queue()
        self.shutdown_event = threading.Event()

        # Optional hook fired from the playback thread at the moment a sentence's
        # audio actually STARTS (cached or freshly synthesised).  Used by the GUI
        # to reveal the bubble text, apply body-control tags on the Live2D
        # model, and install the lip-sync loudness envelope.  Signature:
        #     on_sentence_start(gen, text, mood, tags, envelope=None, frame_ms=None, t0=None)
        # ``envelope`` is a 0..1 RMS loudness profile (Phase 4a) or None when
        # decode failed / disabled; ``frame_ms`` is the ms per envelope frame.
        # ``t0`` (Phase 4b) is the monotonic time the sentence was queued (None
        # for untagged sentences) so the GUI can measure tag-to-audio latency.
        self.on_sentence_start: Optional[Callable[..., None]] = None

        self._tts_thread = threading.Thread(target=self._tts_loop, daemon=False, name="TTS-synth")
        self._play_thread = threading.Thread(target=self._play_loop, daemon=False, name="TTS-play")
        self._tts_thread.start()
        self._play_thread.start()

        logger.info(
            f"Speaker ready  voice={voice_id}  vol={volume}  speed={self.speed}  "
            f"duck={self.ducking_enabled}  cache={cache_enabled}"
        )

    # Public API 

    def say(self, text: str, mood: Optional[str] = None,
            tags: Optional[List] = None) -> None:
        text = clean_for_speech(text)
        if self.enabled and text:
            tag_list = list(tags or [])
            # Phase 4b: stamp when a TAGGED sentence is queued so the GUI can
            # later measure tag-to-audio-start latency at playback start.
            t0 = time.monotonic() if tag_list else None
            self.sentences.put((self.gen, text, mood, tag_list, t0))

    def stop(self) -> None:
        self.gen += 1
        pygame.mixer.music.stop()
        _unduck()
        self._drain(self.sentences)
        self._drain(self.audio)

    def is_playing(self) -> bool:
        """True while TTS audio is queued, synthesising, or playing.

        More accurate than pygame.mixer.music.get_busy() alone, which goes False
        between sentences while the next one is being synthesised.
        """
        try:
            music_busy = bool(pygame.mixer.music.get_busy())
        except pygame.error:
            music_busy = False
        return (
            music_busy
            or not self.sentences.empty()
            or not self.audio.empty()
        )

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, volume))
        pygame.mixer.music.set_volume(self.volume)
        logger.info(f"Volume → {self.volume:.0%}")

    def set_speed(self, speed: float) -> None:
        """Set TTS speed (0.5 = half speed, 2.0 = double speed)."""
        self.speed = max(0.5, min(2.0, speed))
        logger.info(f"TTS speed → {self.speed:.1f}x")

    def set_emotion_template(self, template: Optional[str]) -> None:
        """Update emotion template without restart."""
        self.emotion_template = template

    def set_ducking(self, enabled: bool, volume: Optional[float] = None) -> None:
        """Enable/disable ducking and optionally update duck volume."""
        self.ducking_enabled = enabled and _PYCAW_AVAILABLE
        if volume is not None:
            self.ducking_volume = max(0.0, min(1.0, volume))

    def shutdown(self, timeout: float = 10.0) -> None:
        logger.info("Speaker shutting down...")
        self.shutdown_event.set()
        self._tts_thread.join(timeout=timeout)
        self._play_thread.join(timeout=timeout)
        _unduck()
        if self._tts_thread.is_alive() or self._play_thread.is_alive():
            logger.warning("Speaker threads did not finish in time")

    # Private

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def _synth(self, text: str, mood: Optional[str]) -> bytes:
        """Synthesise text → MP3 bytes (with cache check)."""
        if self.cache:
            hit = self.cache.get(text, mood, self.voice_id, self.model, self.speed)
            if hit:
                return hit

        # Apply emotion template if configured
        tts_text = text
        if self.emotion_template and mood:
            tts_text = self.emotion_template.format(mood=mood, text=text)

        logger.debug(f"Synthesising ({mood or 'neutral'}): {tts_text[:60]!r}")

        # Build Fish Audio call kwargs — speed is passed if not 1.0
        tts_kwargs = dict(
            text=tts_text,
            reference_id=self.voice_id,
            model=self.model,
        )
        if abs(self.speed - 1.0) > 0.05:
            tts_kwargs["speed"] = self.speed  # type: ignore[assignment]

        audio_gen = self.client.tts.convert(**tts_kwargs)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name
        save(audio_gen, path)
        with open(path, "rb") as f:
            data = f.read()
        os.remove(path)

        if self.cache:
            self.cache.put(text, mood, self.voice_id, self.model, self.speed, data)

        return data

    def _tts_loop(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                gen, text, mood, tags, t0 = self.sentences.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if gen == self.gen:
                    data = self._synth(text, mood)
                    self.audio.put((gen, text, mood, tags, data, t0))
            except Exception as e:
                logger.error(f"TTS synth error: {e}")

    def _play_loop(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                gen, text, mood, tags, data, t0 = self.audio.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if gen != self.gen:
                    continue
                if self.ducking_enabled:
                    _duck(self.ducking_volume)
                # Phase 4a: decode the loudness envelope off the GUI thread before
                # playback.  Failure returns None and the mouth uses its flap.
                envelope, frame_ms = None, None
                if self._lip_sync_enabled:
                    result = decode_envelope(data)
                    if result is not None:
                        envelope, frame_ms = result
                pygame.mixer.music.load(io.BytesIO(data), "mp3")
                pygame.mixer.music.play()
                # Tags (and the lip-sync envelope) fire at the START of the
                # sentence's audio (cached or not), from the playback thread.
                # Interrupted generations are skipped by the gen check above and
                # the guard inside the call.
                if self.on_sentence_start is not None and gen == self.gen:
                    try:
                        self.on_sentence_start(
                            gen, text, mood, tags, envelope, frame_ms, t0)
                    except Exception:
                        logger.error("on_sentence_start callback error", exc_info=True)
                while pygame.mixer.music.get_busy() and gen == self.gen:
                    if self.shutdown_event.is_set():
                        break
                    time.sleep(0.05)
                if not pygame.mixer.music.get_busy():
                    _unduck()
            except Exception as e:
                logger.error(f"Playback error: {e}")
                _unduck()
