"""Configuration management with validation."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Config:
    """Central configuration with validation and defaults."""
    
    def __init__(self, config_dir: Path = Path(".")):
        self.config_dir = config_dir
        self.personality_file = config_dir / "personality.json"
        self.memory_file = config_dir / "memory.json"
        
        # Model settings
        self.local_model = "vini:latest"
        self.preset_voice_id = "1d5d9bd251834c09ba653d8738baf64a"
        self.tts_model = "s2.1-pro-free"
        self.sample_rate = 16000
        self.max_history = 30
        self.emotion_template: Optional[str] = None
        
        # Audio settings
        self.tts_cache_enabled = True
        self.tts_cache_max_size = 100
        self.volume = 0.7  # 0.0 to 1.0
        
        # Recording settings
        self.vad_threshold_multiplier = 3.0
        self.min_audio_length = 0.5  # seconds
        
        # Commands
        self.apps = {"notepad": "notepad.exe", "calculator": "calc.exe"}
        self.sites = {"youtube": "https://youtube.com", "github": "https://github.com"}
        
        self.persona: Dict[str, Any] = {}
        self._load_personality()
    
    def _load_personality(self) -> None:
        """Load and validate personality.json."""
        try:
            with open(self.personality_file, "r", encoding="utf-8") as f:
                self.persona = json.load(f)
            
            # Validate required fields
            required = ["name", "system_prompt", "greeting", "temperature", "max_reply_length"]
            missing = [k for k in required if k not in self.persona]
            if missing:
                raise ValueError(f"personality.json missing required fields: {', '.join(missing)}")
            
            # Set defaults for optional fields
            self.persona.setdefault("mood_tags", False)
            
            logger.info(f"Loaded personality: {self.persona['name']}")
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"personality.json not found at {self.personality_file}. "
                "Please create it with required fields: name, system_prompt, greeting, "
                "temperature, max_reply_length"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"personality.json is not valid JSON: {e}")
    
    def load_memory(self) -> List[str]:
        """Load memory facts from disk."""
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
        """Save memory facts to disk."""
        # Deduplicate and limit
        unique_facts = list(dict.fromkeys(facts))  # preserves order
        if len(unique_facts) > 100:
            logger.warning(f"Memory has {len(unique_facts)} facts, keeping most recent 100")
            unique_facts = unique_facts[-100:]
        
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(unique_facts, f, indent=2)
        logger.info(f"Saved {len(unique_facts)} memory facts")
