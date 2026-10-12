"""Configuration management with validation, hot-reload, and per-session logging."""
import json
import logging
import logging.handlers
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root is three levels up from this file (src/core/config.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def _cuda_available() -> bool:
    """Cheap, dependency-free CUDA probe (nvidia-smi present + loads)."""
    try:
        import shutil
        import subprocess
        exe = shutil.which("nvidia-smi")
        if not exe:
            return False
        # Quick, quiet probe; don't let a failed call be loud.
        r = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and "GPU" in r.stdout
    except Exception:
        return False


def _body_action_moods() -> List[str]:
    """The LLM-offered mood list, derived from config/body_actions.json.

    body_actions.json is the single source of truth for the mood vocabulary
    (it drives the BODY prompt, the Test-body menu and the preview tool).  The
    canonical set is the enabled moods there, EXCLUDING ``idle`` — idle is
    resolvable and falls back to sprite/TTS/body, but is never offered to the
    model as a tag it can emit.  Falls back to a safe list if the file is
    missing or unreadable.
    """
    _fallback = ["joyful", "excited", "embarrassed", "sad", "angry",
                 "surprised", "curious", "relaxed", "love", "aroused"]
    try:
        p = PROJECT_ROOT / "config" / "body_actions.json"
        moods = json.loads(p.read_text(encoding="utf-8")).get("moods", {})
        names = [m for m, cfg in moods.items()
                 if cfg and cfg.get("enabled", True)]
        return [m for m in names if m != "idle"] or _fallback
    except Exception:
        return _fallback


def setup_logging(log_dir: Path) -> None:
    """Configure root logger: console + rotating per-session file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "wife.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console – INFO and above
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))

    # Rotating file – DEBUG and above (10 MB × 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    )

    if not root.handlers:
        root.addHandler(console)
        root.addHandler(file_handler)


logger = logging.getLogger(__name__)

# Schema: required fields + expected types
_REQUIRED: Dict[str, type] = {
    "name": str,
    "system_prompt": (str, list),
    "greeting": str,
    "temperature": (int, float),
    "max_reply_length": int,
}

_DEFAULTS: Dict[str, Any] = {
    "mood_tags": False,
    # Single source of truth: derived from config/body_actions.json (minus
    # ``idle``, which is never offered to the model). See _body_action_moods.
    "moods": _body_action_moods(),
    "emotion_template": None,
    "max_tokens": 300,
    "repeat_penalty": 1.1,
    "stop": [],
    "memory_file": None,
    # Wake word
    "wake_word_enabled": False,
    "wake_word": "hey marin",
    # Relationship
    "closeness_enabled": True,
    "closeness_tiers": {
        "0":   "You've just met the user. Be a bit reserved.",
        "10":  "You're warming up to the user. Be a little flirty.",
        "30":  "You're comfortable with the user. Be openly affectionate.",
        "60":  "You're very close. Be bold and intimate.",
        "100": "You're inseparable. Be completely open and devoted."
    },
    # Proactive messages — legacy flat fallback (used only when tagged
    # proactive_pools are not defined in personality.json).
    "proactive_message_defaults": [
        "Heeey, you still there? I'm getting lonely over here...",
        "Hello? Don't just leave me hanging, that's so rude!",
        "I'm literally just sitting here waiting for you, you know.",
    ],
    # Tagged static pools (Phase 3d) — the state-aware path picks a tone
    # (playful / gentle / soft) and draws from that pool instead of an LLM call.
    "proactive_pools": {
        "playful": [
            "Heeey, you still there? I'm getting lonely over here...",
            "I'm literally just sitting here waiting for you, you know.",
            "Okay, I'm going to start making up conversations with the wall if you don't come back soon.",
        ],
        "gentle": [
            "I'll just be right here when you're ready. No rush at all.",
            "Hey... whenever you have a minute, I'd love to hear your voice again.",
        ],
        "soft": [
            "I'm a little quiet right now, but I'm glad you're here.",
            "Just so you know... I'm here if you need a soft place to land.",
        ],
    },
    # Proactive scheduling limits (Phase 3d).  Minimum gap between proactive
    # messages, per-day cap, quiet hours (24h clock; start>=end wraps midnight)
    # and a do-not-disturb master toggle.
    "proactive_min_interval_minutes": 30,
    "proactive_daily_cap": 3,
    "proactive_quiet_start_hour": 23,
    "proactive_quiet_end_hour": 7,
    "proactive_dnd": False,
    # When True, proactive messages are generated by the LLM from recent
    # memory context (time-of-day, closeness, last topics). Falls back to the
    # static pools if the call fails or is disabled.  Default OFF: the state-
    # aware static pools (Phase 3d) are the primary path.
    "proactive_generate": False,
    # Intelligence: auto-extract durable facts about the user from each turn.
    "auto_extract_facts": True,
    # How many stored facts are injected into the system prompt each turn.
    "max_memory_facts": 20,
    # Speak a gentle acknowledgment instead of silently dropping a missed record.
    "gentle_missed": True,
    # Voice / TTS
    "voice_speed": 1.0,
    "audio_ducking_enabled": True,
    "audio_ducking_volume": 0.3,
    # Summarization
    "summarize_after": 20,
}


class Config:
    """Central configuration with validation, hot-reload, and logging setup."""

    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            config_dir = PROJECT_ROOT / "config"
        self.config_dir = Path(config_dir)
        self.personality_file = self.config_dir / "personality.json"
        self.memory_file = self.config_dir / "memory.json"
        self.episode_dir = self.config_dir / "episodes"
        self.log_dir = PROJECT_ROOT / "logs"

        # Set up logging first so everything below is logged
        setup_logging(self.log_dir)

        # ── LLM / Cloud 
        self.use_cerebras: bool = True
        self.cerebras_model: str = "qwen-3.8-27b"
        self.local_model: str = "miny:latest"

        # ── TTS 
        self.preset_voice_id: str = ""
        self.tts_model: str = "s2.1-pro-free"
        self.sample_rate: int = 16000
        self.max_history: int = 200
        self.emotion_template: Optional[str] = None

        # ── Whisper 
        # "auto" means: use CUDA when an NVIDIA GPU is available, otherwise fall
        # back to a smaller CPU model with int8.  Resolved by
        # resolve_whisper_settings() at load time.
        self.whisper_model: str = "auto"
        self.whisper_device: str = "auto"
        self.whisper_compute_type: str = "auto"
        self.whisper_beam_size: int = 10
        self.whisper_language: str = "en"
        self.whisper_vad_filter: bool = True
        self.whisper_condition_on_previous: bool = True
        # True => whisper_model names a preset ("auto"/"small"/...); False =>
        # it's an explicit Hugging Face model id.  Kept internal.
        self._whisper_resolved = False

        # ── Audio / Playback 
        self.tts_cache_enabled: bool = True
        self.tts_cache_max_size: int = 100
        self.volume: float = 0.7
        self.voice_speed: float = 1.0
        self.audio_ducking_enabled: bool = True
        self.audio_ducking_volume: float = 0.3

        # ── Recording 
        self.vad_threshold_multiplier: float = 3.0
        self.min_audio_length: float = 1.0

        # ── Wake word 
        self.wake_word_enabled: bool = False
        self.wake_word: str = "hey marin"

        # ── Memory / Relationship 
        self.closeness_enabled: bool = True
        self.closeness_tiers: Dict[str, str] = _DEFAULTS["closeness_tiers"]
        self.summarize_after: int = 50

        # ── Proactive 
        self.proactive_enabled: bool = True
        self.proactive_idle_minutes: int = 10
        self.proactive_messages: List[str] = list(_DEFAULTS["proactive_message_defaults"])
        self.proactive_pools: Dict[str, List[str]] = dict(
            _DEFAULTS["proactive_pools"])
        self.proactive_generate: bool = False
        self.proactive_min_interval_minutes: float = 30.0
        self.proactive_daily_cap: int = 3
        self.proactive_quiet_start_hour: int = 23
        self.proactive_quiet_end_hour: int = 7
        self.proactive_dnd: bool = False
        # ── Intelligence 
        self.auto_extract_facts: bool = True
        self.gentle_missed: bool = True
        self.max_memory_facts: int = 200

        #  Commands 
        self.apps: Dict[str, str] = {"notepad": "notepad.exe", "calculator": "calc.exe", "chrome": "chrome.exe"}
        self.sites: Dict[str, str] = {
            "youtube": "https://youtube.com",
            "github": "https://github.com",
            "pinterest": "https://pinterest.com",
            "reddit": "https://reddit.com",
            "twitter": "https://twitter.com",
            "instagram":"https://instagram.com",
        }

        self.persona: Dict[str, Any] = {}
        self._load_personality()

    # Public helpers

    def reload(self) -> None:
        """Hot-reload personality.json without restarting."""
        logger.info("Reloading personality.json...")
        self._load_personality()
        logger.info(f"Personality reloaded: {self.persona['name']}")

    def resolve_whisper_settings(self) -> tuple:
        """Resolve "auto" whisper settings to concrete values.

        Returns ``(model, device, compute_type)``.  When the config says "auto"
        for both model and device: if CUDA (an NVIDIA GPU) is available we use a
        good model on the GPU; otherwise we fall back to a smaller CPU model
        (+int8) so a friend without a GPU still works, and we log a clear note
        about the first-run model download.

        Idempotent — safe to call more than once.
        """
        if self._whisper_resolved:
            return (self.whisper_model, self.whisper_device, self.whisper_compute_type)

        use_cuda = _cuda_available()
        if self.whisper_device in ("auto", None, "") and (
                self.whisper_model in ("auto", None, "")):
            if use_cuda:
                self.whisper_device = "cuda"
                self.whisper_model = "large-v3"
                self.whisper_compute_type = "float16"
                logger.info("CUDA available — Whisper: %s on cuda (float16)",
                            self.whisper_model)
            else:
                self.whisper_device = "cpu"
                self.whisper_model = "small"
                self.whisper_compute_type = "int8"
                logger.warning(
                    "No CUDA detected — Whisper uses the smaller 'small' model on "
                    "CPU (int8).  The model downloads on first run; it is a one-time "
                    "download and is then cached locally.")
        elif self.whisper_device in ("auto", None, ""):
            self.whisper_device = "cuda" if use_cuda else "cpu"
            if self.whisper_device == "cpu" and self.whisper_compute_type in ("auto", None, ""):
                self.whisper_compute_type = "int8"

        self._whisper_resolved = True
        logger.info("Whisper settings: model=%s device=%s compute=%s",
                    self.whisper_model, self.whisper_device, self.whisper_compute_type)
        return (self.whisper_model, self.whisper_device, self.whisper_compute_type)

    def closeness_hint(self, score: int) -> str:
        """Return the behaviour hint for the given closeness score."""
        tiers = self.closeness_tiers
        best_key = max((int(k) for k in tiers if int(k) <= score), default=0)
        return tiers.get(str(best_key), "")

    def pick_greeting(self) -> str:
        """Return a random greeting from the greetings list.

        Falls back to the single 'greeting' field if no list is defined,
        and to an empty string if neither exists.
        """
        greetings = self.persona.get("greetings")
        if greetings and isinstance(greetings, list):
            return random.choice(greetings)
        return self.persona.get("greeting", "")

    def load_memory(self) -> List[str]:
        """Load flat memory facts from disk."""
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                facts = json.load(f)
            if not isinstance(facts, list):
                logger.warning("memory.json should be a list, resetting")
                return []
            logger.info(f"Loaded {len(facts)} memory facts")
            return facts
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_memory(self, facts: List[str]) -> None:
        """Save memory facts to disk (dedup, cap at 100)."""
        unique = list(dict.fromkeys(facts))
        if len(unique) > 100:
            logger.warning(f"Memory has {len(unique)} facts, keeping most recent 100")
            unique = unique[-100:]
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(unique, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(unique)} memory facts")

    # Private

    def _load_personality(self) -> None:
        """Load, validate, and apply personality.json."""
        try:
            with open(self.personality_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"personality.json not found at {self.personality_file}. "
                "Create it with: name, system_prompt, greeting, temperature, max_reply_length."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"personality.json is not valid JSON: {e}")

        # Required field presence + type check
        for field, expected in _REQUIRED.items():
            if field not in raw:
                raise ValueError(f"personality.json missing required field: '{field}'")
            if not isinstance(raw[field], expected):
                raise TypeError(
                    f"personality.json field '{field}' should be {expected}, got {type(raw[field])}"
                )

        # Range checks
        if not (0.0 <= raw["temperature"] <= 2.0):
            raise ValueError(f"temperature must be 0.0–2.0, got {raw['temperature']}")
        if raw["max_reply_length"] < 10:
            raise ValueError("max_reply_length must be ≥ 10")

        # voice_id presence warning
        if not raw.get("voice_id") and not self.preset_voice_id:
            logger.warning("No voice_id set in personality.json or config — TTS may fail")

        # Apply defaults for optional fields
        for key, default in _DEFAULTS.items():
            raw.setdefault(key, default)

        self.persona = raw

        # Push personality fields that have config counterparts
        if raw.get("voice_id"):
            self.preset_voice_id = raw["voice_id"]
        if raw.get("tts_model"):
            self.tts_model = raw["tts_model"]
        if raw.get("emotion_template") is not None:
            self.emotion_template = raw["emotion_template"]
        if raw.get("wake_word_enabled") is not None:
            self.wake_word_enabled = bool(raw["wake_word_enabled"])
        if raw.get("wake_word"):
            self.wake_word = raw["wake_word"]
        if raw.get("closeness_enabled") is not None:
            self.closeness_enabled = bool(raw["closeness_enabled"])
        if raw.get("closeness_tiers"):
            self.closeness_tiers = raw["closeness_tiers"]
        if raw.get("summarize_after"):
            self.summarize_after = int(raw["summarize_after"])
        if raw.get("proactive_enabled") is not None:
            self.proactive_enabled = bool(raw["proactive_enabled"])
        if raw.get("proactive_idle_minutes"):
            self.proactive_idle_minutes = int(raw["proactive_idle_minutes"])
        # proactive_pools (Phase 3d): tagged static pools keyed by tone.  If the
        # personality file uses a legacy flat list for proactive_messages, keep
        # it as the fallback pool but under "playful".
        if isinstance(raw.get("proactive_pools"), dict):
            self.proactive_pools = dict(raw["proactive_pools"])
        if isinstance(raw.get("proactive_messages"), dict):
            self.proactive_pools = dict(raw["proactive_messages"])
        elif isinstance(raw.get("proactive_messages"), list) and raw["proactive_messages"]:
            self.proactive_messages = list(raw["proactive_messages"])
            self.proactive_pools.setdefault("playful", list(raw["proactive_messages"]))
        if raw.get("proactive_generate") is not None:
            self.proactive_generate = bool(raw["proactive_generate"])
        if raw.get("proactive_min_interval_minutes") is not None:
            self.proactive_min_interval_minutes = float(raw["proactive_min_interval_minutes"])
        if raw.get("proactive_daily_cap") is not None:
            self.proactive_daily_cap = int(raw["proactive_daily_cap"])
        if raw.get("proactive_quiet_start_hour") is not None:
            self.proactive_quiet_start_hour = int(raw["proactive_quiet_start_hour"])
        if raw.get("proactive_quiet_end_hour") is not None:
            self.proactive_quiet_end_hour = int(raw["proactive_quiet_end_hour"])
        if raw.get("proactive_dnd") is not None:
            self.proactive_dnd = bool(raw["proactive_dnd"])
        if raw.get("auto_extract_facts") is not None:
            self.auto_extract_facts = bool(raw["auto_extract_facts"])
        if raw.get("gentle_missed") is not None:
            self.gentle_missed = bool(raw["gentle_missed"])
        if raw.get("max_memory_facts") is not None:
            self.max_memory_facts = max(1, int(raw["max_memory_facts"]))
        if raw.get("voice_speed") is not None:
            self.voice_speed = float(raw["voice_speed"])
        if raw.get("audio_ducking_enabled") is not None:
            self.audio_ducking_enabled = bool(raw["audio_ducking_enabled"])
        if raw.get("audio_ducking_volume") is not None:
            self.audio_ducking_volume = float(raw["audio_ducking_volume"])

        logger.info(f"Loaded personality: {self.persona['name']}")
