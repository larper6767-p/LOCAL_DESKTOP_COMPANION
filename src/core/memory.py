"""Episodic memory: timestamped conversation log, relationship progression, closeness tiers."""
import json
import logging
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class EpisodeStore:
    """Persists daily conversation episode files under config/episodes/.

    Each file is named YYYY-MM-DD.json and contains a list of episode entries:
        {
            "ts":   "2026-10-09T14:32:01",
            "role": "user" | "assistant",
            "text": "...",
            "mood": "joyful" | null
        }

    Summaries are stored in config/episodes/summaries.json as a list of
        {"date": "YYYY-MM-DD", "summary": "..."}
    """

    def __init__(self, episode_dir: Path):
        self.episode_dir = Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self._summary_file = self.episode_dir / "summaries.json"
        self._lock = threading.Lock()

    # ── Write ─────────────────────────────────────────────────────────────────

    def record(self, role: str, text: str, mood: Optional[str] = None) -> None:
        """Append one turn to today's episode file."""
        entry: Dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "role": role,
            "text": text[:2000],  # hard cap per entry
            "mood": mood,
        }
        path = self._today_file()
        with self._lock:
            episodes = self._read_file(path)
            episodes.append(entry)
            self._write_file(path, episodes)

    def save_summary(self, day: str, summary: str) -> None:
        """Persist a generated summary for a given date string (YYYY-MM-DD)."""
        with self._lock:
            summaries = self._read_file(self._summary_file)
            # Replace existing entry for the same date or append
            summaries = [s for s in summaries if s.get("date") != day]
            summaries.append({"date": day, "summary": summary})
            self._write_file(self._summary_file, summaries)
        logger.info(f"Saved summary for {day}")

    # ── Read ──────────────────────────────────────────────────────────────────

    def today_messages(self) -> List[Dict[str, str]]:
        """Return today's entries as role/content dicts (compatible with LLM history)."""
        raw = self._read_file(self._today_file())
        return [{"role": e["role"], "content": e["text"]} for e in raw]

    def recent_summaries(self, n: int = 5) -> str:
        """Return the last n daily summaries as a formatted string."""
        summaries = self._read_file(self._summary_file)
        recent = summaries[-n:] if len(summaries) > n else summaries
        if not recent:
            return ""
        lines = [f"[{s['date']}] {s['summary']}" for s in recent]
        return "\n".join(lines)

    def get_all_episode_dates(self) -> List[str]:
        """Return list of YYYY-MM-DD strings for which episodes exist."""
        return sorted(
            p.stem for p in self.episode_dir.glob("????-??-??.json")
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _today_file(self) -> Path:
        return self.episode_dir / f"{date.today().isoformat()}.json"

    @staticmethod
    def _read_file(path: Path) -> list:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    @staticmethod
    def _write_file(path: Path, data: list) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class RelationshipTracker:
    """Tracks closeness score and persists it to disk.

    The score starts at 0 and increments after each successful conversation turn.
    Config-defined closeness_tiers map score thresholds → behaviour hints injected
    into the system prompt.
    """

    _SAVE_KEY = "closeness"

    def __init__(self, save_path: Path, tiers: Dict[str, str]):
        self.save_path = Path(save_path)
        self.tiers = tiers           # {"0": "...", "10": "...", ...}
        self._score: int = 0
        self._lock = threading.Lock()
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def score(self) -> int:
        return self._score

    def increment(self, amount: int = 1) -> None:
        with self._lock:
            self._score = min(self._score + amount, 200)  # soft cap
        self._save()

    def current_hint(self) -> str:
        """Return the behaviour hint for the current score."""
        best = max((int(k) for k in self.tiers if int(k) <= self._score), default=0)
        return self.tiers.get(str(best), "")

    def tier_name(self) -> str:
        """Human-readable tier for display (e.g. 'Acquaintance', 'Close')."""
        score = self._score
        if score < 10:
            return "Strangers"
        if score < 30:
            return "Acquaintances"
        if score < 60:
            return "Friends"
        if score < 100:
            return "Close"
        return "Devoted"

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._score = int(data.get(self._SAVE_KEY, 0))
            logger.info(f"Closeness score loaded: {self._score} ({self.tier_name()})")
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            self._score = 0

    def _save(self) -> None:
        try:
            # Merge with existing data so we don't clobber other keys
            existing: dict = {}
            try:
                with open(self.save_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            existing[self._SAVE_KEY] = self._score
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save closeness score: {e}")


class MemoryManager:
    """Combines flat facts, episodic log, and relationship tracking in one place.

    This is what VoiceCompanion interacts with directly.
    """

    def __init__(
        self,
        config_dir: Path,
        tiers: Dict[str, str],
        closeness_enabled: bool = True,
    ):
        self.config_dir = Path(config_dir)
        self.episode_store = EpisodeStore(self.config_dir / "episodes")
        self.relationship = RelationshipTracker(
            save_path=self.config_dir / "relationship.json",
            tiers=tiers,
        )
        self.closeness_enabled = closeness_enabled
        self._facts: List[str] = []

    # ── Facts ─────────────────────────────────────────────────────────────────

    def load_facts(self, facts: List[str]) -> None:
        self._facts = facts

    def get_facts(self) -> List[str]:
        return list(self._facts)

    def add_fact(self, fact: str) -> None:
        if fact not in self._facts:
            self._facts.append(fact)

    def clear_facts(self) -> None:
        self._facts.clear()

    # ── Episodes ──────────────────────────────────────────────────────────────

    def log_turn(self, role: str, text: str, mood: Optional[str] = None) -> None:
        self.episode_store.record(role, text, mood)

    def save_summary(self, day: str, summary: str) -> None:
        self.episode_store.save_summary(day, summary)

    def recent_context(self, n_summaries: int = 3) -> str:
        """A paragraph of recent memory to inject into the system prompt."""
        summaries = self.episode_store.recent_summaries(n_summaries)
        if not summaries:
            return ""
        return f"Recent conversation history:\n{summaries}"

    # ── Relationship ──────────────────────────────────────────────────────────

    def after_turn(self) -> None:
        """Call after each successful conversation turn."""
        if self.closeness_enabled:
            self.relationship.increment(1)

    def closeness_hint(self) -> str:
        return self.relationship.current_hint() if self.closeness_enabled else ""

    def closeness_score(self) -> int:
        return self.relationship.score

    def tier_name(self) -> str:
        return self.relationship.tier_name()

    # ── System prompt additions ───────────────────────────────────────────────

    def build_context_block(self, max_facts: Optional[int] = None) -> str:
        """Return a block to append to the system prompt.

        Injects the most recent `max_facts` facts (newest win) to keep the
        context window from bloating as the fact list grows. None = no cap.
        """
        lines: List[str] = []

        hint = self.closeness_hint()
        if hint:
            lines.append(f"[Relationship level — {self.tier_name()} ({self.relationship.score} pts)]: {hint}")

        if self._facts:
            facts = self._facts
            if max_facts is not None and len(facts) > max_facts:
                facts = facts[-max_facts:]
            lines.append("Things you know about the user: " + "; ".join(facts) + ".")

        recent = self.recent_context()
        if recent:
            lines.append(recent)

        return "\n\n".join(lines)
