
"""motion3_player.py — Cubism motion3.json player that bypasses the SDK motion
system entirely.

Evaluates the motion curves per-frame and calls model.SetParamById() directly,
so priority gating, group registration, and state machine quirks are irrelevant.

Supported segment types (per Cubism SDK spec):
    0  Linear
    1  Bezier  (two CP pairs before next keyframe)
    2  Stepped (value holds until next keyframe)
    3  InverseStepped (value jumps at next keyframe)

Usage:
    player = Motion3Player("path/to/motion.motion3.json")
    player.start(loops=1)          # loops=0 means infinite
    # each frame:
    player.update(dt, model)       # returns True while still playing
    player.stop()
"""

from __future__ import annotations

import json
import math
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Segment type constants 
_SEG_LINEAR          = 0
_SEG_BEZIER          = 1
_SEG_STEPPED         = 2
_SEG_INVERSE_STEPPED = 3


def _cubic_bezier(t: float, p0: float, p1: float, p2: float, p3: float) -> float:
    """Evaluate a cubic Bezier at parameter t ∈ [0,1]."""
    u = 1.0 - t
    return u*u*u*p0 + 3.0*u*u*t*p1 + 3.0*u*t*t*p2 + t*t*t*p3


def _solve_bezier_t(x: float, x0: float, x1: float, x2: float, x3: float,
                    tol: float = 1e-5) -> float:
    """Find the Bezier parameter t such that Bx(t) == x using bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = (lo + hi) * 0.5
        bx = _cubic_bezier(mid, x0, x1, x2, x3)
        if abs(bx - x) < tol:
            return mid
        if bx < x:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


class _Keyframe:
    __slots__ = ("time", "value", "seg_type",
                 "cp0x", "cp0y", "cp1x", "cp1y")

    def __init__(self, time: float, value: float, seg_type: int = _SEG_LINEAR,
                 cp0x: float = 0.0, cp0y: float = 0.0,
                 cp1x: float = 0.0, cp1y: float = 0.0):
        self.time = time
        self.value = value
        self.seg_type = seg_type
        self.cp0x = cp0x
        self.cp0y = cp0y
        self.cp1x = cp1x
        self.cp1y = cp1y


class _Curve:
    """One parameter curve from a motion3.json Curves entry."""

    def __init__(self, param_id: str, keyframes: list[_Keyframe]):
        self.param_id = param_id
        self.keyframes = keyframes

    def evaluate(self, t: float) -> float:
        kf = self.keyframes
        if not kf:
            return 0.0
        if t <= kf[0].time:
            return kf[0].value
        if t >= kf[-1].time:
            return kf[-1].value

        # Binary search for the segment containing t
        lo, hi = 0, len(kf) - 2
        while lo < hi:
            mid = (lo + hi) // 2
            if kf[mid + 1].time <= t:
                lo = mid + 1
            else:
                hi = mid
        a, b = kf[lo], kf[lo + 1]

        seg = a.seg_type
        if seg == _SEG_STEPPED:
            return a.value
        if seg == _SEG_INVERSE_STEPPED:
            return b.value

        dt = b.time - a.time
        if dt < 1e-9:
            return b.value
        u = (t - a.time) / dt     # normalised 0→1

        if seg == _SEG_LINEAR:
            return a.value + u * (b.value - a.value)

        # _SEG_BEZIER: control points are stored on the START keyframe
        # Cubism stores them in *time* space (absolute), normalise to 0-1
        # cx0/cy0 = first out-tangent CP; cx1/cy1 = second in-tangent CP
        tx = _solve_bezier_t(
            a.time + u * dt,
            a.time, a.cp0x, a.cp1x, b.time,
        )
        return _cubic_bezier(tx, a.value, a.cp0y, a.cp1y, b.value)


def _parse_segments(raw: list, fade_in: float = 0.0,
                    fade_out: float = 0.0, duration: float = 0.0) -> list[_Keyframe]:
    """Parse a Cubism Segments array into a list of _Keyframe objects.

    The array is a flat stream:
      time, value, [segment_type, [cp0x, cp0y, cp1x, cp1y]]
    where the *previous* keyframe stores the segment type and (for Bezier) the
    two control points that connect it to the *next* keyframe.

    Cubism layout (from SDK source):
      For Bezier:  t, v, 1, cp0x, cp0y, cp1x, cp1y
      For Linear:  t, v, 0
      For Stepped: t, v, 2
      For InvStep: t, v, 3
    Then the next point follows: t2, v2, seg_type2, ...
    The *last* point has no segment type (it is just t, v).
    """
    keyframes: list[_Keyframe] = []
    i = 0
    n = len(raw)

    while i < n:
        t = float(raw[i])
        v = float(raw[i + 1])
        i += 2

        if i >= n:
            # Last keyframe — no segment info
            keyframes.append(_Keyframe(t, v))
            break

        seg = int(raw[i])
        i += 1

        if seg == _SEG_BEZIER:
            cp0x = float(raw[i]);     cp0y = float(raw[i+1])
            cp1x = float(raw[i+2]);  cp1y = float(raw[i+3])
            i += 4
            keyframes.append(_Keyframe(t, v, _SEG_BEZIER, cp0x, cp0y, cp1x, cp1y))
        else:
            keyframes.append(_Keyframe(t, v, seg))

    return keyframes


def _load_motion(path: str | Path) -> tuple[list[_Curve], float, float, bool]:
    """Parse a motion3.json file.
    Returns (curves, duration, fps, loop).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = data.get("Meta", {})
    duration = float(meta.get("Duration", 1.0))
    fps      = float(meta.get("Fps", 30.0))
    loop     = bool(meta.get("Loop", False))

    curves: list[_Curve] = []
    for entry in data.get("Curves", []):
        if entry.get("Target") != "Parameter":
            continue
        param_id = entry.get("Id", "")
        raw_segs = entry.get("Segments", [])
        if not param_id or not raw_segs:
            continue
        kf = _parse_segments(raw_segs)
        curves.append(_Curve(param_id, kf))

    return curves, duration, fps, loop


class Motion3Player:
    """Plays a single .motion3.json on a Live2D model via direct param calls.

    Thread-safety: call only from the GL thread (inside paintGL).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._curves: list[_Curve] = []
        self._duration = 1.0
        self._fps = 30.0
        self._file_loops = False

        self._playing   = False
        self._elapsed   = 0.0
        self._loops_max = 1       # 0 = infinite
        self._loops_done = 0
        self._known_params: set[str] | None = None   # populated on first update

        self._load()

    def _load(self) -> None:
        try:
            self._curves, self._duration, self._fps, self._file_loops = \
                _load_motion(self._path)
            logger.info(f"Motion3Player: loaded {self._path.name} "
                        f"({len(self._curves)} curves, {self._duration:.1f}s, "
                        f"loop={self._file_loops})")
        except Exception as e:
            logger.error(f"Motion3Player: failed to load {self._path}: {e}")
            self._curves = []

    def start(self, loops: int = 1) -> None:
        """Start playback.  loops=0 means play indefinitely."""
        self._elapsed    = 0.0
        self._loops_done = 0
        self._loops_max  = loops
        self._playing    = True
        logger.info(f"Motion3Player: start (loops={'∞' if loops == 0 else loops})")

    def stop(self) -> None:
        self._playing = False
        logger.info("Motion3Player: stopped")

    @property
    def is_playing(self) -> bool:
        return self._playing

    def update(self, dt: float, model: object) -> bool:
        """Advance by dt seconds and write params to model.
        Returns True while still playing.
        """
        if not self._playing or not self._curves:
            return False

        self._elapsed += dt

        # Evaluate all curves at current time
        t = self._elapsed % self._duration if self._duration > 0 else 0.0
        self._write_params(t, model)

        # Check loop / end
        if self._elapsed >= self._duration:
            self._elapsed -= self._duration
            self._loops_done += 1
            if self._loops_max != 0 and self._loops_done >= self._loops_max:
                self._playing = False
                logger.info("Motion3Player: finished")
                return False

        return True

    def _write_params(self, t: float, model: object) -> None:
        """Evaluate all curves at time t and push to the model."""
        # Build the set of param IDs that actually exist on this model
        # on the first call (avoids per-frame try/except spam).
        if self._known_params is None:
            try:
                ids = set(model.GetParamIds())
                self._known_params = ids
                all_ids = {c.param_id for c in self._curves}
                missing = all_ids - ids
                if missing:
                    logger.info(f"Motion3Player: skipping missing params: {missing}")
            except Exception:
                self._known_params = set()

        for curve in self._curves:
            if curve.param_id not in self._known_params:
                continue
            val = curve.evaluate(t)
            try:
                # SetSaveParamById so values we own survive Model.Update()'s
                # LoadParameters(). See ARCHITECTURE.md Load/Save semantics.
                model.SetSaveParamById(curve.param_id, val, 1.0)
            except Exception:
                pass
