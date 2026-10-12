"""Lip-sync loudness envelope (Phase 4a) — pure, testable, no Qt/GL/audio deps.

The heart of 4a: turn a clip's raw PCM samples into a low-rate loudness
envelope (default 30 values/sec, ~33 ms frames) and index it by playback
position in milliseconds.  Driving ParamMouthOpenY from this envelope makes the
mouth track the actual speech, instead of just a random 7 Hz flap.

Everything here is pure: a numpy array in -> a 0..1 envelope (+ frame_ms) out,
and a position -> mouth-open value out.  The decoder (pygame Sound + sndarray)
lives in core/audio.py; this module never imports pygame, so it is unit-testable
on a headless box with zero audio/GL/display.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

DEFAULT_HZ = 30.0  # frames per second (~33 ms per frame; 30-60 per the 4a spec)


def build_envelope(
    samples: Sequence,
    sample_rate: int,
    hz: float = DEFAULT_HZ,
) -> Tuple[List[float], float]:
    """Convert raw PCM samples into a normalised 0..1 RMS loudness envelope.

    ``samples`` is a 1D array of amplitudes, or a 2D array with channels on the
    last axis (e.g. what ``pygame.sndarray.samples`` returns).  Channels are
    mixed (averaged) before RMS is computed.  The envelope is normalised to its
    own peak so a quiet clip still spans 0..1 (only *relative* loudness drives
    the mouth, which is exactly what we want).

    Returns ``(envelope, frame_ms)`` where ``frame_ms`` is 1000/hz (the time in
    milliseconds each envelope value covers).
    """
    import numpy as np

    a = np.asarray(samples)
    if a.ndim < 1 or a.size == 0:
        return [], 1000.0 / hz
    if a.ndim == 2:
        a = a.mean(axis=1)          # mix channels -> 1D
    a = a.astype(np.float32)
    frame = max(int(sample_rate / hz), 1)
    n = len(a) // frame
    if n < 1:
        return [], 1000.0 / hz
    rms = np.sqrt((a[: frame * n] ** 2).reshape(n, frame).mean(axis=1))
    peak = float(rms.max()) if n else 0.0
    frame_ms = 1000.0 / hz
    if peak <= 1e-9:
        return [0.0] * n, frame_ms
    return [float(min(1.0, v / peak)) for v in rms], frame_ms


def index_envelope(
    env: Sequence[float],
    frame_ms: float,
    pos_ms: Optional[float],
    offset_ms: float = 0.0,
) -> Optional[float]:
    """Return the envelope value at playback position ``pos_ms`` + ``offset_ms``.

    ``offset_ms`` is the lip-sync timing correction (``lip_sync_offset_ms``);
    a positive value shifts the mouth *later* (audio leads the mouth), negative
    shifts it *earlier*.  Clamped to the envelope range; returns None when there
    is no envelope or no position (caller falls back to the random flap).
    """
    if not env or pos_ms is None:
        return None
    p = pos_ms + offset_ms
    if p <= 0.0:
        return env[0]
    idx = int(p / frame_ms) if frame_ms > 0 else 0
    if idx >= len(env):
        return env[-1]
    return env[idx]


def sample_mouth(
    env: Sequence[float],
    frame_ms: float,
    pos_ms: Optional[float],
    *,
    offset_ms: float = 0.0,
    prev: float = 0.0,
    attack: float = 0.6,
    release: float = 0.25,
    min_open: float = 0.12,
    gain: float = 1.0,
) -> Optional[float]:
    """Smooth the raw envelope into a mouth-open value (0..1) for one frame.

    Raises quiet frames so the mouth rests near ``min_open`` instead of sitting
    at 0 (natural closed-lips pose) and only opens on speech.  Attack (opening)
    is faster than release (closing) so syllables start crisply but don't
    flutter.  ``prev`` is the value from the previous frame.  Returns None when
    no envelope/position is available (caller keeps its own flap).
    """
    raw = index_envelope(env, frame_ms, pos_ms, offset_ms)
    if raw is None:
        return None
    target = min_open + (1.0 - min_open) * (max(0.0, raw) ** 1.5) * gain
    target = max(0.0, min(1.0, target))
    alpha = attack if target > prev else release
    return prev + (target - prev) * alpha


class LipSyncTracker:
    """Stateful helper: remember the active envelope and smooth per frame.

    The widget clears/updates the envelope via :meth:`set_envelope` and asks
    for a mouth value each frame via :meth:`value_at`.  Kept pure (no clock,
    no Qt) so it is trivially unit-testable.
    """

    def __init__(
        self,
        *,
        offset_ms: float = 0.0,
        attack: float = 0.6,
        release: float = 0.25,
        min_open: float = 0.12,
        gain: float = 1.0,
    ) -> None:
        self.offset_ms = offset_ms
        self.attack = attack
        self.release = release
        self.min_open = min_open
        self.gain = gain
        self._env: List[float] = []
        self._frame_ms: float = 1000.0 / DEFAULT_HZ
        self._last: float = 0.0
        self._has_envelope = False

    @property
    def has_envelope(self) -> bool:
        return self._has_envelope and bool(self._env)

    def set_envelope(self, env: Sequence[float], frame_ms: float) -> None:
        """Install a new clip's envelope (called from the GUI thread)."""
        self._env = [float(v) for v in env]
        self._frame_ms = float(frame_ms) if frame_ms > 0 else 1000.0 / DEFAULT_HZ
        self._has_envelope = True
        self._last = 0.0

    def clear(self) -> None:
        self._env = []
        self._has_envelope = False
        self._last = 0.0

    def value_at(self, pos_ms: Optional[float]) -> Optional[float]:
        """Smoothed mouth-open value (0..1) at ``pos_ms``, or None to use flap."""
        if not self.has_envelope:
            return None
        val = sample_mouth(
            self._env, self._frame_ms, pos_ms,
            offset_ms=self.offset_ms, prev=self._last,
            attack=self.attack, release=self.release,
            min_open=self.min_open, gain=self.gain,
        )
        if val is not None:
            self._last = val
        return val


__all__ = [
    "DEFAULT_HZ",
    "build_envelope",
    "index_envelope",
    "sample_mouth",
    "LipSyncTracker",
]
