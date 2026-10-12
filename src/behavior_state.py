"""Pure internal-state model for the Live2D behavior director .

This module is deliberately PURE and testable with no GL/Qt/live2d:
  - no clock reads — all time is passed in as ``dt`` to :meth:`BehaviorState.update`
  - no Qt, no OpenGL, no live2d imports
  - deterministic when given a seeded ``random.Random``
  - step-size independent: decay is exponential, so 10 x 0.1 s == 1 x 1.0 s
    (the raw axes and state_factor are independent of how large each dt is)

It tracks three slow-decaying axes (valence / arousal / engagement), applies
event inputs explicitly, and exposes the scalar outputs the behavior director
consumes: blink rate, sway amplitude + speed, gaze behaviour, breathing depth,
fidget rate, and a combined ``state_factor`` for the motion budget

    final_motion = liveliness x state_factor   (capped by the director's budget)

Inputs are applied explicitly (never polled internally):
  * :meth:`mood_impulse`  — one-shot pulse from the mood-tag table; full effect
                            when the mood CHANGES, diminishing on repeats (idle = 0)
  * :meth:`hotkey`        — F9 attention jolt
  * :meth:`set_listening` — user is speaking (recording.py state)
  * :meth:`set_speaking`  — the pet is speaking
  * :meth:`interact`      — click/drag on the pet
"""



from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "BehaviorState",
    "GazeSpec",
    "BehaviorSnapshot",
    "FeelingsBucket",
    "load_config",
    "DEFAULT_CONFIG_PATH",
]


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


# Exponential decay step toward a target given a half-life in seconds.
# z' = z + (target - z) * (1 - 2^(-dt/half_life)).
# This is exact for any dt, so results are independent of step size.
def _exp_toward(z: float, target: float, half_life: float, dt: float) -> float:
    hl = half_life if half_life > 0 else 1e-9
    factor = 1.0 - 2.0 ** (-dt / hl)
    return z + (target - z) * factor


# Ships matching defaults to config/behavior_state.json so the module is
# usable and fully testable without reading the file.
DEFAULTS: Dict[str, Any] = {
    "mood_impulses": {
        "idle":        {"dvalence": 0.0,  "darousal": 0.0},
        "joyful":      {"dvalence": 0.6,  "darousal": 0.3},
        "excited":     {"dvalence": 0.5,  "darousal": 0.8},
        "embarrassed": {"dvalence": -0.05, "darousal": 0.5},
        "sad":         {"dvalence": -0.8,  "darousal": -0.4},
        "angry":       {"dvalence": -0.5,  "darousal": 0.6},
        "surprised":   {"dvalence": 0.1,  "darousal": 0.9},
        "curious":     {"dvalence": 0.3,  "darousal": 0.5},
        "relaxed":     {"dvalence": 0.3,  "darousal": -0.5},
        "love":        {"dvalence": 0.9,  "darousal": 0.3},
        "aroused":     {"dvalence": 0.4,  "darousal": 0.7},
    },
    # Half-lives (seconds): time for an axis to move half-way to its target.
    # All are exponential toward baseline, so larger = slower. NOTE: will tune.
    "half_lives": {
        "valence":    240.0,   # ~4 min to get half-way back to neutral
        "arousal":    90.0,    # ~1.5 min
        "engagement": 120.0,   # ~2 min
    },
    "mood_repeat_scale": 0.5,  # each repeat of the SAME mood scales its impulse
    "state_floor": 0.35,       # state_factor is never below this (no freeze alone)
    "interaction": {
        "engagement_boost": 0.35,
        "arousal_boost": 0.15,
        "valence_boost": 0.05,
        "engagement_target_idle": 0.6,
        "engagement_floor_ignored": 0.25,
        "engagement_floor_idle_long": 0.12,
        "ignored_after_s": 300.0,
        "idle_long_after_s": 900.0,
    },
    "closeness": {"max": 100.0, "engagement_base": 0.25, "valence_base": 0.15},
    "listening": {"engagement_target": 1.0},
    "speaking": {"engagement_target": 1.0},
    "noise_scale": 0.02,
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "behavior_state.json"


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load behavior-state config from JSON (returning DEFAULTS if unreadable)."""
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        merged = dict(DEFAULTS)
        for k, v in raw.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        return merged
    except Exception:
        return DEFAULTS


@dataclass(frozen=True)
class GazeSpec:
    """Gaze behaviour the director should enact this frame."""

    mode: str          # "wander" | "look_ahead" | "attentive"
    intensity: float   # 0..1 how far / active the gaze is
    toward_viewer: bool  # intent to face the viewer (straight ahead)


class BehaviorState:
    """Slow-decaying valence / arousal / engagement model plus motion outputs.

    ``closeness`` (0..max from relationship.json) sets a small persistent
    baseline; transient mood is NOT persisted anywhere — it lives only in the
    live axes and decays.

    Two axes have a fixed baseline they drift back to: valence -> closeness
    valence baseline (usually ~0), arousal -> 0.5 (neutral). Engagement has no
    single baseline; it is driven toward a condition-dependent target
    (attentive when listening/speaking, idle target otherwise). Engagement is
    mainly meant to change the KIND of behaviour (attentive vs wandering gaze),
    not just the amount — see :meth:`gaze`.
    """

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        config: Optional[Dict[str, Any]] = None,
        closeness: float = 0.0,
    ) -> None:
        self._rng: random.Random = rng if rng is not None else random.Random()
        self._cfg: Dict[str, Any] = config if config is not None else DEFAULTS

        # closeness baseline (small, persistent — not a transient mood)
        cmax = float(self._cfg["closeness"].get("max", 100.0) or 100.0)
        frac = _clamp01(float(closeness) / cmax) if cmax else 0.0
        self._val_base = frac * float(self._cfg["closeness"].get("valence_base", 0.15))
        self._eng_base = frac * float(self._cfg["closeness"].get("engagement_base", 0.25))

        # live axes
        self.valence = 0.0          # [-1, 1], >0 happy
        self.arousal = 0.5          # [0, 1], >0 aroused
        self.engagement = 0.5       # [0, 1], >0 attentive

        self._t_since_interaction = 0.0
        self._listening = False
        self._speaking = False
        self._noise = 0.0           # deterministic per-update noise (seeded)
        self._current_mood: Optional[str] = None
        self._mood_repeat = 0

    # event inputs (applied explicitly)

    def mood_impulse(self, mood: str) -> None:
        """Apply a pulse from the mood→(dvalence, darousal) table.

        Full effect when the mood CHANGES; a repeat of the same mood is scaled
        down (diminishing returns) by ``mood_repeat_scale`` each time. Chaining
        back to a previous mood is treated as a change (full effect again).
        ``idle`` intentionally has a zero impulse.
        """
        imp = self._cfg["mood_impulses"].get(mood)
        if not imp:
            return
        dv = float(imp.get("dvalence", 0.0))
        da = float(imp.get("darousal", 0.0))
        if mood == self._current_mood:
            self._mood_repeat += 1
            scale = float(self._cfg.get("mood_repeat_scale", 0.5)) ** self._mood_repeat
        else:
            self._current_mood = mood
            self._mood_repeat = 0
            scale = 1.0
        self.valence = _clamp(self.valence + dv * scale, -1.0, 1.0)
        self.arousal = _clamp01(self.arousal + da * scale)

    def hotkey(self) -> None:
        """F9 attention jolt."""
        self.engagement = _clamp01(self.engagement + 0.2)
        self.arousal = _clamp01(self.arousal + 0.15)
        self._t_since_interaction = 0.0

    def interact(self) -> None:
        """Click/drag on the pet — an explicit interaction."""
        ic = self._cfg["interaction"]
        self.engagement = _clamp01(self.engagement + float(ic.get("engagement_boost", 0.35)))
        self.arousal = _clamp01(self.arousal + float(ic.get("arousal_boost", 0.15)))
        self.valence = _clamp(self.valence + float(ic.get("valence_boost", 0.05)), -1.0, 1.0)
        self._t_since_interaction = 0.0

    def set_listening(self, on: bool) -> None:
        """Set whether the user is speaking (from recording state)."""
        self._listening = bool(on)

    def set_speaking(self, on: bool) -> None:
        """Set whether the pet itself is speaking."""
        self._speaking = bool(on)

    def set_preview_axes(self, valence: float, arousal: float, engagement: float) -> None:
        """Directly set the live axes for the preview tool (visual verification).
        Pure — no clock, just writes the raw axes so sliders can demonstrate how
        a sad / low-arousal / low-engagement state visibly moves less."""
        self.valence = _clamp(valence, -1.0, 1.0)
        self.arousal = _clamp01(arousal)
        self.engagement = _clamp01(engagement)

    # time 
    def update(self, dt: float) -> None:
        """Advance all axes toward their targets by ``dt`` seconds.

        Draws a fresh seeded noise value for this step so every derived output
        is deterministic given the seed and the call sequence. Decay is
        exponential (half-life based), so the raw axes / state_factor are
        independent of how ``dt`` is split.
        """
        if dt <= 0.0:
            return
        self._noise = self._rng.uniform(-1.0, 1.0) * float(self._cfg.get("noise_scale", 0.02))

        hl = self._cfg["half_lives"]
        self.valence = _exp_toward(
            self.valence, self._val_base,
            float(hl.get("valence", 240.0)), dt)
        self.arousal = _exp_toward(
            self.arousal, 0.5,
            float(hl.get("arousal", 90.0)), dt)

        self._t_since_interaction += dt
        ic = self._cfg["interaction"]
        if self._listening:
            target = float(self._cfg["listening"].get("engagement_target", 1.0))
        elif self._speaking:
            target = float(self._cfg["speaking"].get("engagement_target", 1.0))
        else:
            ignored_after = float(ic.get("ignored_after_s", 300.0))
            idle_long_after = float(ic.get("idle_long_after_s", 900.0))
            if self._t_since_interaction < ignored_after:
                target = _clamp01(float(ic.get("engagement_target_idle", 0.6)) + self._eng_base)
            elif self._t_since_interaction < idle_long_after:
                target = _clamp01(float(ic.get("engagement_floor_ignored", 0.25)))
            else:
                target = _clamp01(float(ic.get("engagement_floor_idle_long", 0.12)))
        self.engagement = _exp_toward(
            self.engagement, target,
            float(hl.get("engagement", 120.0)), dt)

    # derived helpers 

    def valence_motion(self) -> float:
        """Map valence [-1, 1] -> [0, 1]; sad (negative) suppresses motion.

        Defined here (and documented in ARCHITECTURE.md) so the director and
        tests share one definition: happy -> 1.0, neutral -> 0.5, sad -> ~0.0.
        """
        return _clamp01(0.5 + 0.5 * self.valence)

    def _noised(self, value: float) -> float:
        return value + self._noise

    #  outputs for the director

    @property
    def listening(self) -> bool:
        return self._listening

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def seconds_since_interaction(self) -> float:
        return self._t_since_interaction

    def state_factor(self) -> float:
        """Combined 0..1 motion scalar. final_motion = liveliness x state_factor.

        Has a configurable floor (``state_floor``, default 0.35) so the pet never
        freezes down to nothing when alone, while a sad / low-arousal state still
        sits below the neutral value. Engagement mostly changes the KIND of
        behaviour (see :meth:`gaze`) rather than only this amount.
        """
        v = self.valence_motion()
        f = ((0.6 + 0.4 * v)
             * (0.5 + 0.5 * self._clamp_arousal01())
             * (0.4 + 0.6 * _clamp01(self.engagement)))
        floor = float(self._cfg.get("state_floor", 0.35))
        return _max_floor(f, floor)

    def _clamp_arousal01(self) -> float:
        return _clamp01(self.arousal)

    def blink_rate(self) -> float:
        """Blinks per second (~0.05–1.0). Lower when sad/calm, higher when lively."""
        base = 0.18
        rate = (base
                * (0.7 + 0.5 * self._clamp_arousal01())
                * (0.8 + 0.3 * _clamp01(self.engagement))
                * (0.55 + 0.5 * self.valence_motion()))
        return _clamp(self._noised(rate), 0.05, 1.0)

    def sway_amplitude(self) -> float:
        """Relative head-sway magnitude 0..1."""
        amp = ((0.3 + 0.7 * self._clamp_arousal01())
               * (0.4 + 0.6 * _clamp01(self.engagement))
               * self.valence_motion())
        return _clamp01(self._noised(amp))

    def sway_speed(self) -> float:
        """Relative sway speed 0..1."""
        s = _clamp01((0.3 + 0.7 * self._clamp_arousal01()) * self.valence_motion())
        return _clamp01(self._noised(s))

    def breathing_depth(self) -> float:
        """0..1 breathing depth (deeper/more visible when more aroused)."""
        depth = 0.5 + 0.4 * self._clamp_arousal01()
        return _clamp01(self._noised(depth))

    def fidget_rate(self) -> float:
        """0..1 frequency of small autonomous fidgets."""
        rate = ((0.4 + 0.6 * self._clamp_arousal01())
                * (0.5 + 0.5 * _clamp01(self.engagement))
                * self.valence_motion())
        return _clamp01(self._noised(rate))

    def gaze(self) -> GazeSpec:
        """Gaze behaviour for the director to enact.

        Engagement (plus speaking/listening) mainly switches the KIND of gaze:
        attentive/look-ahead toward the viewer when engaged or interacting,
        wandering when ignored for a long time.
        """
        if self._speaking:
            return GazeSpec("look_ahead", 0.5, True)
        if self._listening:
            return GazeSpec("attentive", 0.6, True)
        t = self._t_since_interaction
        ignored_after = float(self._cfg["interaction"].get("ignored_after_s", 300.0))
        idle_long_after = float(self._cfg["interaction"].get("idle_long_after_s", 900.0))
        if t >= idle_long_after:
            return GazeSpec("wander", 0.8, False)
        if t >= ignored_after:
            return GazeSpec("wander", 0.4 + 0.3 * self._clamp_arousal01(), False)
        return GazeSpec("look_ahead", 0.3 + 0.3 * _clamp01(self.engagement), True)

    def snapshot(self) -> "BehaviorSnapshot":
        """Immutable snapshot of the current live state (Phase 3b).

        The app publishes this each tick and the companion reads it once per
        turn on the worker thread, so the LLM-facing code never touches the
        mutable BehaviorState object itself.
        """
        return BehaviorSnapshot(
            valence=self.valence,
            arousal=self.arousal,
            engagement=self.engagement,
            seconds_since_interaction=self._t_since_interaction,
            speaking=self._speaking,
            listening=self._listening,
        )


def _max_floor(value: float, floor: float) -> float:
    return value if value > floor else floor


@dataclass(frozen=True)
class BehaviorSnapshot:
    """Immutable copy of the live behavior axes (Phase 3b).

    Published by the widget each tick and read once per turn by the companion
    from its worker thread, so the LLM-facing path never shares the mutable
    :class:`BehaviorState`.  All fields are floats/bools — safe to read from
    another thread because the object is frozen and never mutated after build.
    """

    valence: float = 0.0            # [-1, 1], >0 happy
    arousal: float = 0.5            # [0, 1], >0 aroused
    engagement: float = 0.5         # [0, 1], >0 attentive
    seconds_since_interaction: float = 0.0
    speaking: bool = False
    listening: bool = False


class FeelingsBucket:
    """Coarse, hysteresis-smoothed buckets of HER OWN feeling (Phase 3b).

    Turns a raw :class:`BehaviorSnapshot` into a short self-feeling phrase for
    the system prompt.  Deliberately describes only her own feeling (cheerful /
    calm / a little tired + energy), NEVER the user's behaviour — no words like
    ignored / wandered / away, because the pet cannot comment on how long the
    user was gone.

    Hysteresis: a bucket only flips once the value crosses the *outer* threshold
    in the new direction, then holds until it crosses back past the *inner*
    (release) threshold in the opposite direction.  This prevents the phrase
    from flapping turn to turn near a boundary.  The bucket state lives in this
    object (owned by the companion), not in BehaviorState.

    Pure: no clock, no Qt — deterministic given the snapshots fed in.
    """

    # valence buckets ('sad' | 'calm' | 'cheerful')
    _VAL_ENTER_SAD, _VAL_ENTER_HAPPY = -0.22, 0.22
    _VAL_RELEASE_SAD, _VAL_RELEASE_HAPPY = -0.05, 0.05

    # arousal buckets ('low' | 'mid' | 'high')
    _ARO_ENTER_LOW, _ARO_ENTER_HIGH = 0.34, 0.66
    _ARO_RELEASE_LOW, _ARO_RELEASE_HIGH = 0.44, 0.56

    def __init__(self) -> None:
        self._val_bucket: str = "calm"
        self._aro_bucket: str = "mid"

    # bucket updates (with hysteresis)

    def _valence_bucket(self, v: float) -> str:
        cur = self._val_bucket
        if cur == "cheerful":
            if v < self._VAL_RELEASE_HAPPY:
                cur = "calm"
        elif cur == "sad":
            if v > self._VAL_RELEASE_SAD:
                cur = "calm"
        else:  # calm
            if v >= self._VAL_ENTER_HAPPY:
                cur = "cheerful"
            elif v <= self._VAL_ENTER_SAD:
                cur = "sad"
        self._val_bucket = cur
        return cur

    def _arousal_bucket(self, a: float) -> str:
        cur = self._aro_bucket
        if cur == "high":
            if a < self._ARO_RELEASE_HIGH:
                cur = "mid"
        elif cur == "low":
            if a > self._ARO_RELEASE_LOW:
                cur = "mid"
        else:  # mid
            if a >= self._ARO_ENTER_HIGH:
                cur = "high"
            elif a <= self._ARO_ENTER_LOW:
                cur = "low"
        self._aro_bucket = cur
        return cur

    # phrase / mood (for the companion and tests)

    def phrase(self, snap: BehaviorSnapshot) -> str:
        """Short self-feeling phrase (e.g. 'cheerful and full of energy')."""
        val = self._valence_bucket(snap.valence)
        aro = self._arousal_bucket(snap.arousal)
        return _FEELING_PHRASES[(val, aro)]

    def suggested_mood(self, snap: BehaviorSnapshot) -> str:
        """Coarse mood the state would suggest (for the loop-stability test)."""
        v = snap.valence
        if v >= 0.3:
            return "joyful"
        if v <= -0.3:
            return "sad"
        return "relaxed" if self._arousal_bucket(snap.arousal) != "high" else "surprised"


_FEELING_PHRASES: dict[tuple[str, str], str] = {
    ("cheerful", "high"): "cheerful and full of energy",
    ("cheerful", "mid"):  "cheerful and steady",
    ("cheerful", "low"):  "cheerful and relaxed",
    ("calm", "high"):     "a bit restless",
    ("calm", "mid"):      "calm and steady",
    ("calm", "low"):      "calm and mellow",
    ("sad", "high"):      "wired but a little tired",
    ("sad", "mid"):       "a little tired",
    ("sad", "low"):       "a little tired and low on energy",
}
