
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from behavior_state import BehaviorSnapshot

# Default tone pools if personality.json provides none (last-resort fallback).
_DEFAULT_POOLS: Dict[str, List[str]] = {
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
}


@dataclass
class ProactiveLimits:
    """Scheduling limits (from config) — pure values, no clock reads."""

    min_interval_minutes: float = 30.0     # 0 disables the minimum-interval check
    daily_cap: int = 3                     # 0 disables the per-day cap
    quiet_start_hour: int = 23             # 24h hour, inclusive lower bound
    quiet_end_hour: int = 7                # 24h hour, exclusive upper bound
    dnd: bool = False                      # do-not-disturb master toggle
    ignored_min_interval_mult: float = 1.5  # widen the gap while ignored long

    @property
    def min_interval_s(self) -> float:
        return self.min_interval_minutes * 60.0


class ProactiveScheduler:
    """Enforces minimum-interval / daily-cap / quiet-hours / do-not-disturb.

    Pure: every decision takes an explicit ``now``; only in-memory state
    (last fire time and the running day's count) is kept here.
    """

    def __init__(self, limits: ProactiveLimits) -> None:
        self._limits = limits
        self._last_fire: Optional[float] = None   # wall clock (seconds since epoch)
        self._day = None                          # datetime.date of the current day
        self._daily_count = 0

    # helpers

    def _in_quiet_hours(self, now: datetime) -> bool:
        if self._limits.quiet_start_hour == self._limits.quiet_end_hour:
            return False  # equal => quiet hours disabled
        h = now.hour
        s = self._limits.quiet_start_hour
        e = self._limits.quiet_end_hour
        # start < end: quiet inside [s, e).  start > end: quiet wraps midnight.
        return (s <= h < e) if s < e else (h >= s or h < e)

    def _reset_day_if_needed(self, now: datetime) -> None:
        if self._day is None or now.date() != self._day:
            self._day = now.date()
            self._daily_count = 0

    # public API 

    def allowed(self, now: datetime, ignored_long: bool = False) -> bool:
        """Whether a proactive message may be emitted at ``now``.

        ``ignored_long`` widens the minimum gap but never *adds* messages, so
        the pet never gets more insistent the longer she is alone.
        """
        lim = self._limits
        if lim.dnd:
            return False
        if self._in_quiet_hours(now):
            return False
        if lim.daily_cap <= 0:
            cap_ok = True
        else:
            self._reset_day_if_needed(now)
            cap_ok = self._daily_count < lim.daily_cap
            if not cap_ok:
                return False
        if lim.min_interval_minutes <= 0 or self._last_fire is None:
            return True
        gap = lim.min_interval_s * (lim.ignored_min_interval_mult if ignored_long else 1.0)
        return (now.timestamp() - self._last_fire) >= gap

    def record(self, now: datetime) -> None:
        """Record that a proactive message was emitted at ``now``."""
        self._last_fire = now.timestamp()
        self._reset_day_if_needed(now)
        self._daily_count += 1


# tone / message selection

def select_tone(snap: Optional[BehaviorSnapshot], ignored_after_s: float = 300.0,
                idle_long_after_s: float = 900.0) -> str:
    """Pick a pool tag from how she currently feels (pure, no RNG).

    soft : sad / low valence
    gentle: ignored for a long while (gentle, never reproachful)
    playful: otherwise (Marin's default voice)
    """
    if snap is None:
        return "playful"
    if snap.valence <= -0.3:
        return "soft"
    if snap.seconds_since_interaction is not None and \
            snap.seconds_since_interaction >= idle_long_after_s:
        return "gentle"
    return "playful"


class ProactiveChooser:
    """Picks a message from the tagged pools, with ignored-long suppression.

    Deterministic given a seeded RNG.  ``pools`` maps a tone tag -> list of
    messages; any tone with no pool falls back to its nearest neighbour and
    finally to "playful".
    """

    def __init__(self, pools: Optional[Dict[str, List[str]]] = None,
                 rng: Optional[random.Random] = None,
                 ignored_after_s: float = 300.0,
                 idle_long_after_s: float = 900.0) -> None:
        self._pools = pools if pools is not None else dict(_DEFAULT_POOLS)
        self._rng = rng if rng is not None else random.Random()
        self._ignored_after_s = ignored_after_s
        self._idle_long_after_s = idle_long_after_s

    def tone(self, snap: Optional[BehaviorSnapshot]) -> str:
        return select_tone(snap, self._ignored_after_s, self._idle_long_after_s)

    def _resolve_pool(self, tone: str) -> List[str]:
        if tone in self._pools and self._pools[tone]:
            return self._pools[tone]
        # nearest-neighbour fallback for missing pools
        for fallback in ("playful", "gentle", "soft"):
            if fallback in self._pools and self._pools[fallback]:
                return self._pools[fallback]
        return []

    def choose(self, snap: Optional[BehaviorSnapshot]) -> Optional[str]:
        """Return a proactive message, or None if suppressed (ignored long).

        Ignored-long suppression: the longer she has been ignored, the less
        often she speaks at all (fewer messages, never more insistent).
        """
        if snap is not None and snap.seconds_since_interaction is not None:
            since = snap.seconds_since_interaction
            if since >= self._ignored_after_s:
                # fade her out the longer she's been ignored: from 40% chance at
                # ignored_after up to ~85% suppression at idle_long_after.
                span = max(self._idle_long_after_s - self._ignored_after_s, 1.0)
                suppress = min(0.4 + 0.45 * ((since - self._ignored_after_s) / span), 0.85)
                if self._rng.random() < suppress:
                    return None
        tone = self.tone(snap)
        pool = self._resolve_pool(tone)
        if not pool:
            return None
        return self._rng.choice(pool)


__all__ = [
    "ProactiveLimits",
    "ProactiveScheduler",
    "ProactiveChooser",
    "select_tone",
    "_DEFAULT_POOLS",
]
