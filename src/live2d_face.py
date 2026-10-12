
"""Live2D face renderer widget for WIFE desktop pet.

Drop-in replacement for FaceLabel. Exposes the same interface:
    update_art(mood: str, mouth_open: bool, crossfade: bool = True)

Additional public API:
    play_motion(name: str, loops: int = 1)  — play a named motion (e.g. "scene1")
    stop_motion()                           — stop the current motion
    motion_is_playing() -> bool

Animation features
──────────────────
  Mood mapping      — JSON file maps mood name → {param: value} dict.
  Expressions       — .exp3.json files loaded at init and played by mood.
  Blink             — random 3-6 s interval, 3-phase state machine.
  Breathing         — sine on ParamBreath, ~4 s period.
  Head sway         — sine layered on ParamAngleZ, ~8 s period.
  Body drift        — slow sine on ParamBodyAngleX0/Z0.
  Random eye darts  — random gaze every 4-9 s.
  Random head poses — random AngleX/Y tilt every 6-14 s.
  Lip sync          — speaking flag → smoothed random ParamMouthOpenY.
  Eye tracking      — Drag(x, y) in widget-local pixels.
  Physics           — driven by model.Update() from physics3.json.

Motion playback
───────────────
  Motions are evaluated by Motion3Player (our own parser) which calls
  SetParamById directly — bypasses the SDK motion system entirely.
  Named motions are registered via add_motion(name, path).
  Scene1.motion3.json is registered as "scene1" at init.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from motion3_player import Motion3Player
from behavior_director import BehaviorDirector, MotionBudget, DIRECTOR_PARAMS
from lipsync import LipSyncTracker

logger = logging.getLogger(__name__)

# live2d-py import 
try:
    import live2d                              # type: ignore
    from live2d.utils.canvas import Canvas    # type: ignore
    _LIVE2D_OK = True
except Exception as _e:
    _LIVE2D_OK = False
    logger.warning(f"live2d-py not available: {_e}")

# Expression file map (filename → internal key) 
_EXP_FILES: dict[str, str] = {
    "脸红.exp3.json":  "blush",
    "生气.exp3.json":  "angry",
    "星星眼.exp3.json": "star_eyes",
    "爱心眼.exp3.json": "heart_eyes",
    "泪.exp3.json":    "tears",
    "白眼.exp3.json":  "eyeroll",
    "脸黑.exp3.json":  "dark",
    "血.exp3.json":    "blood",
    "右手.exp3.json":  "right_hand",
    "左手.exp3.json":  "left_hand",
}

# Mood → expression key (None = no expression, raw params only) 
_MOOD_EXPRESSION: dict[str, str | None] = {
    "idle":        None,
    "joyful":      "blush",
    "excited":     "star_eyes",
    "embarrassed": "blush",
    "sad":         "tears",
    "angry":       "angry",
    "surprised":   None,
    "curious":     None,
    "relaxed":     None,
    "love":        "heart_eyes",
    "aroused":     "blush",
}

# Confirmed writable parameters from CDI3 
#  NEVER write physics / hair / cloth / tail params — the physics engine owns them.
_SAFE_PARAMS: set[str] = {
    "ParamMouthOpenY", "ParamMouthForm", "ParamMouthX",
    "ParamCheekPuff", "ParamTongueOut",
    "ParamEyeLOpen", "ParamEyeROpen", "ParamEyeLSmile", "ParamEyeRSmile",
    "ParamEyeBallX", "ParamEyeBallY",
    "ParamBrowLY", "ParamBrowRY", "ParamBrowLForm", "ParamBrowRForm",
    "ParamAngleX", "ParamAngleY", "ParamAngleZ",
    "ParamBreath",
    "ParamBodyAngleX0", "ParamBodyAngleY0", "ParamBodyAngleZ0",
    "ParamSwitch1", "ParamSwitch2", "ParamSwitch3", "ParamSwitch4",
    "ParamSwitch5", "ParamSwitch6", "ParamSwitch7", "ParamSwitch8",
    "ParamSwitch9", "ParamSwitch10", "ParamSwitch11", "ParamSwitch21",
}

# Default baseline: all zero except eyes fully open
_BASELINE: dict[str, float] = {p: 0.0 for p in _SAFE_PARAMS}
_BASELINE["ParamEyeLOpen"] = 1.0   # 0-1.9 range per vtube.json; 1.0 = normal
_BASELINE["ParamEyeROpen"] = 1.0
_BASELINE["ParamMouthForm"] = 0.0  # range -2.0 to 1.0 per vtube.json


def configure_opengl_format() -> None:
    """Must be called BEFORE QApplication.

    Uses CompatibilityProfile because live2d-py's Canvas shader uses the
    legacy gl_FragColor built-in, which is removed in Core Profile.
    """
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setAlphaBufferSize(8)
    fmt.setSamples(4)
    fmt.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
    QSurfaceFormat.setDefaultFormat(fmt)


class Live2DWidget(QOpenGLWidget):
    """QOpenGLWidget Live2D renderer.  Same public interface as FaceLabel:
        update_art(mood, mouth_open, crossfade=True)
    """

    def __init__(self, model_json: str | Path, moods_json: str | Path,
                 parent=None, *, invert_eye_x: bool = False,
                 invert_head_x: bool = False, invert_head_y: bool = False,
                 idle_enabled: bool = True,
                 body_enabled: bool = True, wink_swap: bool = False,
                 look_invert_x: bool = False,
                 max_simultaneous_fx: int = 3,
                 behavior_director_enabled: bool = True,
                 behavior_state=None,
                 motion_budget: MotionBudget | None = None,
                 liveliness: float = 0.7,
                 lean_param: str = "ParamBodyAngleY0",
                 lean_sign: float = 1.0,
                 breath_pause_enabled: bool = True,
                 breath_pause_every: tuple = (40.0, 90.0),
                 breath_pause_hold: tuple = (1.0, 1.5),
                 lip_sync_enabled: bool = True,
                 lip_sync_offset_ms: float = 0.0) -> None:
        if not _LIVE2D_OK:
            raise RuntimeError("live2d-py is not installed or failed to import.")
        super().__init__(parent)

        self._model_json = str(model_json)
        self._model_dir  = Path(model_json).parent
        self._moods_path = Path(moods_json)

        # ── User-overridable display config (cannot be re-clobbered by config) ──
        self._invert_eye_x  = bool(invert_eye_x)
        self._invert_head_x = bool(invert_head_x)
        self._invert_head_y = bool(invert_head_y)
        self._idle_enabled  = bool(idle_enabled)

        # Body-control tuning (Phase 1c).  Left/right = the SCREEN side as the
        # viewer sees it; wink_swap / look_invert_x let the user flip the L/R
        # parameter mapping visually per model.
        self._body_enabled       = bool(body_enabled)
        self._wink_swap          = bool(wink_swap)
        self._look_invert_x      = bool(look_invert_x)
        self._max_simultaneous_fx = int(max_simultaneous_fx)

        # OpenGL / Qt transparency setup
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")

        # Model + canvas (created in initializeGL)
        self._model: Optional[object] = None
        self._canvas: Optional[object] = None
        self._initialized = False
        self._disposed = False

        # Parameter state
        # _current  = smoothed mood base (eased toward _target)
        # _target   = goal set by the mood map
        # _proc     = procedural override/offset layer (blink, mouth, breath,
        #             idle drift, head cursor, fx) written AFTER smoothing so
        #             it can never be smoothed away.
        self._current:      dict[str, float] = dict(_BASELINE)
        self._target:       dict[str, float] = dict(_BASELINE)
        self._proc:         dict[str, float] = {}
        # Frame-rate-independent smoothing; k units = 1/s (matches old ~0.15/frame at 30 fps)
        self._smooth_k = 4.9

        # Mood / expression state 
        self._mood_map:    dict[str, dict[str, float]] = {}
        self._current_mood = "idle"
        self._active_expression: str | None = None
        self._load_mood_map()

        # Lip sync
        self._speaking          = False
        self._mouth_current     = 0.0
        self._mouth_target      = 0.0
        self._mouth_noise_timer = 0.0
        # Phase 4a: real lip sync from a decoded loudness envelope.  When enabled
        # (and an envelope is installed), _tick_mouth tracks speech; otherwise it
        # falls back to the random flap.  offset tunes audio-vs-mouth timing.
        self._lip_sync_enabled  = bool(lip_sync_enabled)
        self._lip_sync_offset   = float(lip_sync_offset_ms)
        self._lip = LipSyncTracker(offset_ms=self._lip_sync_offset)
        self._mouth_form_base  = _BASELINE.get("ParamMouthForm", 0.0)
        self._mouth_tint_form  = 0.0  # small ParamMouthForm emotion tint

        # Blink state machine
        self._blink_state = "open"   # open | closing | closed | opening
        self._blink_t     = 0.0
        self._blink_next  = self._rand_blink_interval()

        # Continuous animation accumulators
        self._t_breath    = 0.0
        self._t_sway      = 0.0
        self._t_body      = 0.0
        self._last_frame  = time.monotonic()

        #  Random idle behaviour (eye darts, head poses) 
        # Eye dart: pick a random gaze direction for 0.3-0.8 s, then return
        self._dart_timer     = random.uniform(4.0, 8.0)   # seconds until next dart
        self._dart_target_x  = 0.0
        self._dart_target_y  = 0.0
        self._dart_hold      = 0.0   # seconds remaining in current dart
        self._dart_active    = False

        # Head pose: drift AngleX/Y to a gentle random pose and hold
        self._pose_timer     = random.uniform(6.0, 12.0)
        self._pose_target_x  = 0.0
        self._pose_target_y  = 0.0
        self._pose_cur_x     = 0.0
        self._pose_cur_y     = 0.0
        self._pose_hold      = 0.0
        self._pose_active    = False

        # Cursor tracking: target (raw) and eased (current) cursor offsets
        self._cursor_x       = 0.0   # normalized [-1, 1], from feed_cursor
        self._cursor_y       = 0.0
        self._cursor_cur_x   = 0.0   # eased, written into _proc
        self._cursor_cur_y   = 0.0
        # Dart eye-offset accumulators (eased, written into _proc)
        self._dart_cur_x     = 0.0
        self._dart_cur_y     = 0.0

        # Scale / offset
        self._scale    = 2.2
        self._offset_x = 0.0
        self._offset_y = 0.35

        # Motion3Player registry
        # name → Motion3Player; loaded after GL init
        self._motions: dict[str, Motion3Player] = {}
        self._active_player: Optional[Motion3Player] = None

        # Render timer 
        self._fps = 30
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(1000 // self._fps)

        #  Body-control effects 
        # Transient effects fired when each sentence's audio starts.  fx entries
        # override the mood's switch params while active, then auto-revert;
        # look/gesture/wink are procedural with a fade-in/out envelope.
        self._active_fx: list[dict] = []      # [{name, params, ends_at}]
        self._fx_overrides: dict[str, float] = {}   # param -> value this frame
        self._gestures: list[dict] = []       # [{kind, mode, params, start, ends_at}]

        # Behavior director 
        # The pure state model + director replace the old sine-based idle/blink
        # when ``behavior_director_enabled`` is on; ``behavior_state`` is shared
        # with the app (which feeds mood/interact/hotkey/listening inputs) and
        # the widget drives ``speaking`` each frame.  When disabled, the legacy
        # _tick_idle/_tick_idle_behaviour/_tick_blink path runs unchanged.
        self._behavior_director_enabled = bool(behavior_director_enabled)
        self._behavior_state = behavior_state
        self._director = BehaviorDirector(
            budget=motion_budget if motion_budget is not None else MotionBudget(),
            liveliness=float(liveliness),
            lean_param=lean_param,
            lean_sign=lean_sign,
            breath_pause_enabled=breath_pause_enabled,
            breath_pause_every=breath_pause_every,
            breath_pause_hold=breath_pause_hold,
        )
        self._director_liveliness = float(liveliness)
        self._body_log_state_timer = 0.0
        self._body_log_enabled = False
        # Phase 3b: optional callback fired each tick after the behavior state
        # advances, so the app (Pet) can publish an immutable snapshot.  Set by
        # Pet.__init__; None until then.
        self._snapshot_publisher = None

    # Public interface (matches FaceLabel)


    def update_art(self, mood: str, mouth_open: bool, crossfade: bool = True) -> None:
        """Set mood and speaking state.
        crossfade is accepted for API compatibility but ignored — the widget
        always eases parameters smoothly.
        """
        resolved = mood if mood in self._mood_map else "idle"
        if resolved != self._current_mood:
            self._apply_mood(resolved)
            self._current_mood = resolved
        # Direct speaking flag — no 7 Hz toggle needed; widget smooths it
        self._speaking = mouth_open

    def _resolve_expression_params(self, ops) -> dict:
        """Blend + clamp exp3 expansion ops against the model's real ranges.

        Phase 3c.  Blend semantics: ``Add`` = default + value, ``Overwrite`` =
        value, ``Multiply`` = default * value.  Each result is clamped to the
        real [min, max] read from the model.  We never call SetExpression — the
        resolved absolutes flow through the normal SetSaveParamById path.  An
        undefined param or unavailable model yields nothing.
        """
        out: dict = {}
        if self._model is None:
            return out
        for op in ops or []:
            pid = op.get("id")
            if not pid:
                continue
            value = float(op.get("value", 0.0))
            blend = (op.get("blend") or "Add").strip() or "Add"
            try:
                lo = self._model.GetParamMinById(pid)
                hi = self._model.GetParamMaxById(pid)
                default = self._model.GetParamDefaultById(pid)
            except Exception:
                lo = hi = default = None
            if blend == "Overwrite":
                v = value
            elif blend == "Multiply":
                v = default * value if default is not None else value
            else:  # Add
                v = default + value if default is not None else value
            if lo is not None and hi is not None:
                v = max(float(lo), min(float(hi), float(v)))
            out[pid] = float(v)
        return out

    def apply_body_actions(self, actions) -> None:
        """Apply body-control actions fired at a sentence's audio start.

        ``actions`` is a list of dicts produced by the body_tags resolver:
          {"kind": "clear"}
          {"kind": "fx",     "name", "params", "duration"}
          {"kind": "look",   "name", "params", "duration"}
          {"kind": "gesture","name", "params", "duration"}
          {"kind": "wink",   "side", "duration"}

        fx effects override the mood's switch params while active and auto-revert.
        look/gesture/wink are procedural with a fade-in/out envelope.  Guarded so
        that the inner-body call can always fire even if the widget isn't yet
        drawing (sentence audio may start before the first frame).
        """
        if self._model is None or not self._body_enabled:
            return
        now = time.monotonic()
        for a in actions:
            kind = a.get("kind")
            if kind == "clear":
                self._clear_body()
                continue
            if kind == "fx":
                self._active_fx = [f for f in self._active_fx
                                   if f["ends_at"] > now]
                if len(self._active_fx) >= self._max_simultaneous_fx:
                    continue                       # cap: drop new fx
                dur = max(float(a.get("duration", 4.0)), 0.2)
                # Phase 3c: resolve any exp3 expression ops (blended + clamped).
                params = self._resolve_expression_params(a.get("expression_ops", []) or [])
                params.update(dict(a.get("params", {}) or {}))   # explicit wins
                self._active_fx.append({
                    "name": a.get("name", ""),
                    "params": params,
                    "ends_at": now + dur,
                })
            elif kind in ("look", "gesture"):
                params = self._resolve_expression_params(a.get("expression_ops", []) or [])
                params.update(dict(a.get("params", {}) or {}))   # explicit wins
                if kind == "look" and self._look_invert_x:
                    for p in ("ParamAngleX", "ParamEyeBallX"):
                        if p in params:
                            params[p] = -params[p]
                dur = max(float(a.get("duration", 0.9)), 0.2)
                self._gestures.append({
                    "kind": kind, "mode": "add",
                    "params": params, "start": now, "ends_at": now + dur,
                })
            elif kind == "wink":
                side = a.get("side", "right")
                closed = "ParamEyeROpen" if side == "right" else "ParamEyeLOpen"
                if self._wink_swap:
                    closed = ("ParamEyeLOpen" if closed == "ParamEyeROpen"
                              else "ParamEyeROpen")
                dur = max(float(a.get("duration", 0.5)), 0.2)
                self._gestures.append({
                    "kind": "wink", "mode": "override",
                    "params": {closed: 0.0},
                    "start": now, "ends_at": now + dur,
                })

    def _clear_body(self) -> None:
        """Cancel all active body effects (fx + gestures) immediately."""
        self._active_fx = []
        self._gestures = []
        self._fx_overrides = {}

    #  Behavior director public controls (Phase 2b)

    def set_liveliness(self, value: float) -> None:
        """Set the 0..1 liveliness scalar on the behavior director."""
        self._director_liveliness = max(0.0, min(1.0, float(value)))
        self._director.set_liveliness(self._director_liveliness)

    def set_behavior_state(self, behavior_state) -> None:
        """Supply the shared BehaviorState the director reads each frame."""
        self._behavior_state = behavior_state

    def set_behavior_director(self, enabled: bool) -> None:
        """Switch between the behavior director and the legacy idle behaviour."""
        self._behavior_director_enabled = bool(enabled)

    def set_lean_param(self, lean_param: str, lean_sign: float = 1.0) -> None:
        """Pick which body parameter the lean uses (preview tuning)."""
        self._director._lean_param = lean_param if lean_param in DIRECTOR_PARAMS else "ParamBodyAngleY0"
        self._director._lean_sign = 1.0 if lean_sign >= 0 else -1.0

    def note_speech_start(self, text: str = "") -> None:
        """At the START of each spoken sentence (audio start, GUI thread): trigger
        a small speaking nod.  Amplitude is scaled by arousal and liveliness so an
        excited pet nods more, a sad/calm one less; emphasis (``!``/``?``) bumps it.
        """
        if not self._behavior_director_enabled or self._behavior_state is None:
            return
        s = self._behavior_state
        arousal = getattr(s, "arousal", 0.5)
        liv = self._director_liveliness
        amp = 3.0 * (0.5 + 0.6 * max(0.0, min(1.0, arousal))) * liv
        if any(c in text for c in "!?"):
            amp *= 1.4
        self._director.nod(amp)

    def set_mouth_envelope(self, env, frame_ms: float) -> None:
        """Phase 4a: install a sentence's decoded loudness envelope (GUI thread).

        Called at the sentence's audio start via on_body.  ``env`` is a 0..1 RMS
        profile from ``lipsync.build_envelope``; ``frame_ms`` is the ms per frame.
        The envelope is only used when lip sync is enabled AND the pet is
        speaking; otherwise _tick_mouth falls back to the random flap.
        """
        if not self._lip_sync_enabled:
            return
        if not env or frame_ms <= 0:
            self._lip.clear()
            return
        # Keep the offset in the tracker synced with any live change
        self._lip.offset_ms = self._lip_sync_offset
        self._lip.set_envelope(env, float(frame_ms))

    def clear_mouth_envelope(self) -> None:
        """Phase 4a: drop any leftover envelope so a stale clip never reapplies."""
        self._lip.clear()

    def _current_wink_side(self) -> str | None:
        """Screen-side ('left'/'right'/None) of any active wink gesture."""
        if self._model is None or not self._body_enabled:
            return None
        now = time.monotonic()
        for g in self._gestures:
            if g.get("kind") == "wink" and g["ends_at"] > now:
                # Determine the side the wink closed.  The gesture only stores the
                # closed param; recover the original side via wink_swap mapping.
                closed = next(iter(g["params"].keys()), "")
                is_right = ("ParamEyeROpen" if not self._wink_swap
                            else "ParamEyeLOpen") == closed
                return "right" if is_right else "left"
        return None

    def _apply_director_frame(self, frame) -> None:
        """Fold a BehaviorDirector DirectorFrame into the procedural layer.

        Additive head/eye/body drift is ADDED on top of the mood base (and any
        prior procedural offsets, e.g. from a gesture), so the layered priority
        ``mood < idle drift < gesture`` is preserved: a gesture that wrote an
        absolute value for a param earlier in the frame wins because additive
        drift is folded on top of it.  Breathing overrides; the blink writes the
        eye-openness override (None = that eye is owned elsewhere, e.g. a wink).

        Only params in DIRECTOR_PARAMS may ever be written here — a structural
        guarantee that no physics (hair/ear/tail/cloth) param leaks through.
        """
        for p, v in frame.additive.items():
            if p in DIRECTOR_PARAMS:
                self._proc[p] = self._proc.get(p, 0.0) + v
        for p, v in frame.override.items():
            if p in DIRECTOR_PARAMS:
                self._proc[p] = v
        l, r = frame.blink
        if l is not None and "ParamEyeLOpen" in DIRECTOR_PARAMS:
            self._proc["ParamEyeLOpen"] = l
        if r is not None and "ParamEyeROpen" in DIRECTOR_PARAMS:
            self._proc["ParamEyeROpen"] = r

    def _body_log_state(self, dt: float, frame) -> None:
        """Periodically log the director state so visual-tuning can be verified
        (a sad / low-arousal state should visibly move slower and less).
        Uses body_log at a low cadence to avoid log spam."""
        self._body_log_state_timer += dt
        if not getattr(self, "_body_log_enabled", False):
            return
        if self._body_log_state_timer < 6.0:
            return
        self._body_log_state_timer = 0.0
        s = self._behavior_state
        if s is None:
            return
        logger.info(
            "[body-state] val=%.3f aro=%.3f eng=%.3f liv=%.2f "
            "sf=%.3f motion=%.3f blink=%.2f/h gaze=%s spoken=%d listen=%d",
            s.valence, s.arousal, s.engagement,
            self._director_liveliness,
            s.state_factor(),
            self._director_liveliness * s.state_factor(),
            s.blink_rate(), frame.gaze,
            1 if s.speaking else 0,
            1 if s.listening else 0,
        )

    def _tick_body(self, dt: float, now: float) -> None:
        """Advance active body effects and fold them into the procedural layer.

        fx switch overrides are recomputed into _fx_overrides (consulted by
        _flush_params) so they take precedence over the mood base while active
        and auto-revert when they expire.  look/gesture/wink are written into
        _proc with a fade-in/out envelope so they never pop.
        """
        if self._model is None or not self._body_enabled:
            return
        # 1. Expire finished fx and rebuild the switch overrides for this frame.
        self._active_fx = [f for f in self._active_fx if f["ends_at"] > now]
        self._fx_overrides = {}
        for f in self._active_fx:
            for p, v in f["params"].items():
                self._fx_overrides[p] = v
        # 2. Transient gestures with a fade envelope (in over ~0.12s, out ~0.12s).
        alive = []
        for g in self._gestures:
            if g["ends_at"] <= now:
                continue
            alive.append(g)
            total = g["ends_at"] - g["start"]
            if total <= 0.0:
                env = 1.0
            else:
                env = min(1.0, (now - g["start"]) / 0.12,
                          (g["ends_at"] - now) / 0.12)
            env = max(0.0, min(1.0, env))
            if g["mode"] == "override":
                for p, v in g["params"].items():
                    self._proc[p] = v * env
            else:
                for p, v in g["params"].items():
                    self._proc[p] = self._proc.get(p, 0.0) + v * env
        self._gestures = alive

   
    # Scale / offset
    

    def set_model_scale(self, scale: float) -> None:
        self._scale = max(0.1, min(20.0, scale))
        if self._model:
            self._model.SetScale(self._scale)

    def set_model_offset(self, x: float, y: float) -> None:
        self._offset_x = max(-5.0, min(5.0, x))
        self._offset_y = max(-5.0, min(5.0, y))
        if self._model:
            self._model.SetOffset(self._offset_x, self._offset_y)

    def get_model_scale(self) -> float:
        return self._scale

    def get_model_offset(self) -> tuple[float, float]:
        return (self._offset_x, self._offset_y)

    def set_fps(self, fps: int) -> None:
        self._fps = max(10, min(60, fps))
        self._timer.start(1000 // self._fps)

    
    # Motion playback (Motion3Player — bypasses SDK motion system)
   

    def add_motion(self, name: str, path: str | Path) -> None:
        """Register a .motion3.json file under a name for later play_motion()."""
        try:
            self._motions[name] = Motion3Player(path)
            logger.info(f"Motion registered: '{name}' ← {Path(path).name}")
        except Exception as e:
            logger.warning(f"Could not register motion '{name}': {e}")

    def play_motion(self, name: str, loops: int = 1) -> bool:
        """Start playing a registered motion.  loops=0 = infinite.
        Returns True if the motion was found and started.
        """
        if name not in self._motions:
            logger.warning(f"Motion '{name}' not registered")
            return False
        if self._active_player is not None and self._active_player.is_playing:
            self._active_player.stop()
        self._active_player = self._motions[name]
        self._active_player.start(loops=loops)
        return True

    def stop_motion(self) -> None:
        if self._active_player is not None:
            self._active_player.stop()
            self._active_player = None

    def motion_is_playing(self) -> bool:
        return self._active_player is not None and self._active_player.is_playing

   
    # Idle-loop toggle (persistent user override, NOT re-clobbered by config)
   

    @property
    def idle_enabled(self) -> bool:
        """Whether procedural idle animations (breath/sway/body/darts/poses)
        are running. Persistent user override; survives restarts."""
        return self._idle_enabled

    def set_idle_enabled(self, enabled: bool) -> None:
        """Turn the procedural idle loop on/off.  This is the authoritative
        user override — it is applied after model load and never overwritten
        by config/live2d.json."""
        self.apply_idle_override(bool(enabled))

    def apply_idle_override(self, enabled: bool) -> None:
        """Apply the idle-loop setting to the renderer.  Called once from
        initializeGL (after load) and again whenever the user toggles it."""
        self._idle_enabled = bool(enabled)
        logger.info(f"Live2D idle loop {'enabled' if self._idle_enabled else 'disabled'} "
                    f"(persistent override)")

    # Mood map
   

    def _load_mood_map(self) -> None:
        try:
            raw = json.loads(self._moods_path.read_text(encoding="utf-8"))
            self._mood_map = {
                k: v for k, v in raw.items()
                if not k.startswith("_") and isinstance(v, dict)
            }
            logger.info(f"Live2D moods: {list(self._mood_map.keys())}")
        except Exception as e:
            logger.error(f"Failed to load live2d_moods.json: {e}")
            self._mood_map = {"idle": {}}

    def reload_mood_map(self) -> None:
        self._load_mood_map()

    def known_moods(self) -> list[str]:
        return list(self._mood_map.keys())

    def _apply_mood(self, mood: str) -> None:
        """Set parameter targets and trigger expression for the given mood."""
        # 1. Reset all safe params to baseline first so switch params clear
        new_target = dict(_BASELINE)
        # 2. Apply mood overrides
        for param, val in self._mood_map.get(mood, {}).items():
            if param in _SAFE_PARAMS:
                new_target[param] = float(val)
        self._target = new_target

        # Phase 4a: small ParamMouthForm emotion tint derived from the mood's own
        # mouth-form value (relative to the baseline).  Applied while lip-syncing
        # so an excited/sad mouth reads slightly differently, then fades back.
        mood_form = new_target.get("ParamMouthForm", self._mouth_form_base)
        self._mouth_tint_form = 0.5 * (mood_form - self._mouth_form_base)

        # 3. Trigger expression file if mapped
        self._play_expression_for_mood(mood)

    def _play_expression_for_mood(self, mood: str) -> None:
        """Play the .exp3.json expression associated with this mood, or reset."""
        if not self._initialized or self._model is None:
            return
        exp_key = _MOOD_EXPRESSION.get(mood, None)
        if exp_key == self._active_expression:
            return   # already playing the right one

        if exp_key is not None:
            try:
                self._model.SetExpression(exp_key)
                self._active_expression = exp_key
            except Exception as e:
                logger.debug(f"SetExpression({exp_key}) failed: {e}")
                self._active_expression = None
        else:
            try:
                self._model.ResetExpression()
                self._active_expression = None
            except Exception as e:
                logger.debug(f"ResetExpression failed: {e}")

    # GL lifecycle
    

    def initializeGL(self) -> None:
        try:
            live2d.init()
            live2d.glInit()

            self._model = live2d.Model()
            self._model.LoadModelJson(self._model_json)

            # Disable built-in effects — we drive everything manually
            try:
                self._model.SetAutoBlink(False)
                self._model.SetAutoBreath(False)
            except AttributeError:
                pass

            # Register expression files
            self._register_expressions()

            # Register motions via Motion3Player (bypasses SDK motion system)
            self._register_motions()

            # Canvas — must be created after GL context exists
            self._canvas = Canvas()

            w, h = self.width(), self.height()
            self._model.Resize(w, h)
            self._canvas.SetSize(w, h)

            self._model.SetScale(self._scale)
            self._model.SetOffset(self._offset_x, self._offset_y)

            # Push baseline so first frame is clean
            self._flush_params_direct(_BASELINE)

            self._log_param_ranges()

            # Apply user's idle-loop override AFTER model load so config can't
            # re-clobber it (this is the persistent pet-state setting).
            self.apply_idle_override(self._idle_enabled)

            self._initialized = True
            logger.info(f"Live2D model loaded: {self._model_json}")

        except Exception as e:
            logger.error(f"Live2D initializeGL failed: {e}", exc_info=True)
            self._initialized = False

    def _register_expressions(self) -> None:
        """Load each .exp3.json into the model."""
        for filename, key in _EXP_FILES.items():
            path = self._model_dir / filename
            if path.exists():
                try:
                    self._model.LoadExtraExpression(key, str(path))
                    logger.info(f"Expression registered: {key} ← {filename}")
                except Exception as e:
                    logger.warning(f"Could not load expression {filename}: {e}")
            else:
                logger.debug(f"Expression file not found: {filename}")

    def _register_motions(self) -> None:
        """Register Motion3Player instances for each known motion file."""
        scene1 = self._model_dir / "Scene1.motion3.json"
        if scene1.exists():
            self.add_motion("scene1", scene1)

    def _log_param_ranges(self) -> None:
        """Log real min/max/default for switch + continuous params.

        live2d-py 1.0.0 exposes GetParamMinById / GetParamMaxById /
        GetParamDefaultById (the non-Id variants do not exist).  We wrap in
        try/except so a missing param never breaks init.
        """
        if self._model is None:
            return
        switches = [f"ParamSwitch{i}" for i in range(1, 12)] + ["ParamSwitch21"]
        continuous = [
            "ParamAngleX", "ParamAngleY", "ParamAngleZ",
            "ParamEyeLOpen", "ParamEyeROpen", "ParamMouthOpenY", "ParamBreath",
            "ParamEyeBallX", "ParamEyeBallY", "ParamBodyAngleX0", "ParamBodyAngleZ0",
        ]
        for p in switches + continuous:
            try:
                lo = self._model.GetParamMinById(p)
                hi = self._model.GetParamMaxById(p)
                df = self._model.GetParamDefaultById(p)
                logger.info(f"  param {p:>16}: min={lo:8.3f} max={hi:8.3f} default={df:8.3f}")
            except Exception as e:
                logger.info(f"  param {p:>16}: <unavailable: {e}> ")

    def resizeGL(self, w: int, h: int) -> None:
        if not self._initialized:
            return
        try:
            dpr = self.devicePixelRatio()
            self._model.Resize(int(w * dpr), int(h * dpr))
            self._canvas.SetSize(int(w * dpr), int(h * dpr))
        except Exception as e:
            logger.error(f"Live2D resizeGL: {e}")

    def paintGL(self) -> None:
        if not self._initialized or self._model is None:
            return
        try:
            now = time.monotonic()
            dt  = min(now - self._last_frame, 0.1)
            self._last_frame = now

            # Advance active motion player (writes params directly to model)
            if self._active_player is not None:
                still_playing = self._active_player.update(dt, self._model)
                if not still_playing:
                    self._active_player = None

            # Procedural layer is rebuilt fresh every frame.  Clearing it here
            # (rather than accumulating) means each tick below ADDS its current
            # contribution on top of the others *within this frame only* — it can
            # never grow across frames.
            self._proc.clear()

            # 1. Ease mood base toward targets (frame-rate independent).
            #    Runs FIRST so procedural overrides/offsets below are applied
            #    on top and can never be smoothed back toward a stale target.
            self._tick_smooth(dt)
            # 2. Procedural layer — these write into _proc via _flush_params.
            if (self._behavior_director_enabled
                    and self._behavior_state is not None):
                # Feed the active wink side BEFORE the director reads it (it
                # suppresses blink only for the winked eye), then advance the
                # state model + director and fold the resulting frame in.
                self._director.set_wink_side(self._current_wink_side())
                self._behavior_state.update(dt)
                self._behavior_state.set_speaking(self._speaking)
                if self._snapshot_publisher is not None:
                    self._snapshot_publisher()
                frame = self._director.update(dt, self._behavior_state)
                self._apply_director_frame(frame)
                self._body_log_state(dt, frame)
            else:
                if self._idle_enabled:
                    self._tick_idle(dt)
                    self._tick_idle_behaviour(dt)
                self._tick_blink(dt)
            self._tick_mouth(dt)
            self._tick_cursor(dt)
            # Body effects (fx/look/gesture/wink) layered last so they win over
            # the director/breath/idle for the params they own, then flushed.
            self._tick_body(dt, now)
            # Merge mood base + procedural layer and push to the model.
            self._flush_params()

            # Capture what we wrote for the heartbeat so we can compare
            # against read-back AFTER Update() (LoadParameters would discard
            # SetParamById writes; SetSaveParamById must survive).
            hb_wrote_breath   = self._proc.get("ParamBreath", 0.0)
            hb_wrote_anglez   = self._proc.get("ParamAngleZ", 0.0) + self._current.get("ParamAngleZ", 0.0)

            self._model.Update()

            # ── Heartbeat: wrote vs read_back, ONE warning if writes discarded ─
            self._hb_frames = getattr(self, "_hb_frames", 0) + 1
            hb_every = max(1, int(self._fps * 3))
            if self._hb_frames % hb_every == 0:
                try:
                    rb_breath = self._model.GetParamValueById("ParamBreath")
                    rb_anglez = self._model.GetParamValueById("ParamAngleZ")
                except Exception:
                    rb_breath = rb_anglez = float("nan")
                drift = max(abs(rb_breath - hb_wrote_breath),
                            abs(rb_anglez - hb_wrote_anglez))
                if drift > 0.05:
                    self._hb_bad_s = getattr(self, "_hb_bad_s", 0.0) + dt
                else:
                    self._hb_bad_s = 0.0
                self._hb_last_drift = drift
                logger.info(
                    "L2D beat: frames=%d idle=%s mood=%s | breath wrote=%.3f rb=%.3f | "
                    "angleZ wrote=%.3f rb=%.3f | drift=%.3f",
                    self._hb_frames, self._idle_enabled, self._current_mood,
                    hb_wrote_breath, rb_breath, hb_wrote_anglez, rb_anglez, drift,
                )
                if self._hb_bad_s >= 3.0 and not getattr(self, "_hb_warned", False):
                    logger.warning(
                        "L2D heartbeat: writes appear to be DISCARDED "
                        "(read_back differs from wrote by >%.2f for %.1fs) — "
                        "check SetSaveParamById usage. drift=%.3f",
                        0.05, self._hb_bad_s, drift,
                    )
                    self._hb_warned = True
            # Heartbeat 

            def _draw():
                live2d.clearBuffer(0.0, 0.0, 0.0, 0.0)
                self._model.Draw()

            self._canvas.Draw(_draw)

        except Exception as e:
            logger.error(f"Live2D paintGL: {e}")

    def closeEvent(self, event) -> None:
        self.dispose()
        super().closeEvent(event)

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._timer.stop()
        try:
            self._model = None
            live2d.dispose()
        except Exception as e:
            logger.warning(f"Live2D dispose: {e}")
        logger.info("Live2D disposed")

   
    # Parameter management
   

    def _tick_smooth(self, dt: float) -> None:
        """Frame-rate-independent easing of mood params toward their targets.

        alpha = 1 - exp(-k*dt) gives identical convergence at any fps; k matches
        the old ~0.15 lerp per 1/30 s frame (k ≈ 4.9/s).

        Runs BEFORE procedural writes (blink/mouth/breath/cursor) so those are
        applied on top and never eased back toward a stale mood target.
        """
        a = 1.0 - math.exp(-self._smooth_k * dt)
        for p, tgt in self._target.items():
            cur = self._current.get(p, 0.0)
            self._current[p] = cur + (tgt - cur) * a

    def _flush_params(self) -> None:
        """Write (mood base + procedural layer) to the model.

        Uses SetSaveParamById so the values we own survive Model.Update()'s
        LoadParameters() (which would otherwise restore the saved values and
        discard every SetParamById call).  Procedural keys in `_proc` either
        override (blink/mouth/breath) or add onto (head/eye offsets) the
        smoothed mood base in `_current`.
        """
        if self._model is None:
            return
        always_proc = {"ParamEyeLOpen", "ParamEyeROpen", "ParamMouthOpenY",
                       "ParamBreath", "ParamMouthForm"}
        add_proc = {
            "ParamAngleX", "ParamAngleY", "ParamAngleZ",
            "ParamBodyAngleX0", "ParamBodyAngleZ0",
            "ParamEyeBallX", "ParamEyeBallY",
        }
        fx = getattr(self, "_fx_overrides", {})
        for p, v in self._current.items():
            try:
                if p in fx:
                    # Active fx (blush/tears switch) — takes precedence over the
                    # smoothed mood base while active; auto-reverts on expiry.
                    self._model.SetSaveParamById(p, self._fx_overrides[p], 1.0)
                elif p in always_proc and p in self._proc:
                    self._model.SetSaveParamById(p, self._proc[p], 1.0)
                elif p in add_proc and p in self._proc:
                    self._model.SetSaveParamById(p, v + self._proc[p], 1.0)
                else:
                    self._model.SetSaveParamById(p, v, 1.0)
            except Exception:
                pass

    def _tick_cursor(self, dt: float) -> None:
        """Apply cursor-driven head/eye offsets in the procedural layer.

        Per 简.vtube.json: ParamAngleX = left/right, ParamAngleY = up/down.
        Horizontal cursor motion → ParamAngleX, vertical → ParamAngleY.
        Eye-ball X is separately invertible via config (do not assume sign).
        Values are ADDED on top of any idle pose/dart offsets already in `_proc`.
        """
        ease = 1.0 - math.exp(-10.0 * dt)
        self._cursor_cur_x += (self._cursor_x - self._cursor_cur_x) * ease
        self._cursor_cur_y += (self._cursor_y - self._cursor_cur_y) * ease
        hx = self._cursor_cur_x * self._HEAD_X_SCALE * (-1.0 if self._invert_head_x else 1.0)
        hy = self._cursor_cur_y * self._HEAD_Y_SCALE * (-1.0 if self._invert_head_y else 1.0)
        ex = self._cursor_cur_x * self._EYE_X_SCALE * (-1.0 if self._invert_eye_x else 1.0)
        self._proc["ParamAngleX"]   = self._proc.get("ParamAngleX", 0.0) + hx
        self._proc["ParamAngleY"]   = self._proc.get("ParamAngleY", 0.0) + hy
        self._proc["ParamEyeBallX"] = self._proc.get("ParamEyeBallX", 0.0) + ex

    def _flush_params_direct(self, params: dict[str, float]) -> None:
        """Write a param dict directly (bypasses smoothing — init only).

        Uses SetSaveParamById so init values also survive Update()."""
        if self._model is None:
            return
        for p, v in params.items():
            try:
                self._model.SetSaveParamById(p, v, 1.0)
            except Exception:
                pass

    
    # Motion end stub
  

    def _tick_idle_behaviour(self, dt: float) -> None:
        """Eye darts and random head poses layered on top of the base idle.

        These are ADDITIVE offsets written into the procedural layer: the mood's
        own eye/head values remain in `_current` and _flush_params sums them.
        The eased accumulators always move toward their (0 when inactive)
        targets so poses/darts naturally return to centre.
        """

        #  Eye dart
        if self._dart_active:
            self._dart_hold -= dt
            if self._dart_hold <= 0.0:
                self._dart_active = False
                self._dart_target_x = 0.0
                self._dart_target_y = 0.0
                self._dart_timer = random.uniform(4.0, 9.0)
        else:
            self._dart_timer -= dt
            if self._dart_timer <= 0.0:
                self._dart_active = True
                self._dart_hold   = random.uniform(0.3, 0.9)
                # Pick a random gaze direction: mostly small, occasionally wide
                mag = random.choice([0.3, 0.3, 0.3, 0.6, 0.8])
                angle = random.uniform(0, math.pi * 2)
                # ParamEyeBallX is inverted per vtube.json (input 1 → output -1)
                self._dart_target_x = -round(math.cos(angle) * mag, 2)
                self._dart_target_y =  round(math.sin(angle) * mag * 0.5, 2)
        dspd = min(1.0, 0.25 * dt * 60.0)
        self._dart_cur_x += (self._dart_target_x - self._dart_cur_x) * dspd
        self._dart_cur_y += (self._dart_target_y - self._dart_cur_y) * dspd

        # Random head pose 
        if self._pose_active:
            self._pose_hold -= dt
            if self._pose_hold <= 0.0:
                self._pose_active   = False
                self._pose_target_x = 0.0
                self._pose_target_y = 0.0
                self._pose_timer    = random.uniform(5.0, 14.0)
        else:
            self._pose_timer -= dt
            if self._pose_timer <= 0.0:
                self._pose_active   = True
                self._pose_hold     = random.uniform(1.5, 4.0)
                # Gentle tilt — stays within a natural range
                self._pose_target_x = random.uniform(-8.0, 8.0)
                self._pose_target_y = random.uniform(-5.0, 5.0)
        pspd = min(1.0, 0.04 * dt * 60.0)
        self._pose_cur_x += (self._pose_target_x - self._pose_cur_x) * pspd
        self._pose_cur_y += (self._pose_target_y - self._pose_cur_y) * pspd

        # Fold into procedural layer (additive on the mood base).
        self._proc["ParamEyeBallX"] = self._proc.get("ParamEyeBallX", 0.0) + self._dart_cur_x
        self._proc["ParamEyeBallY"] = self._proc.get("ParamEyeBallY", 0.0) + self._dart_cur_y
        self._proc["ParamAngleX"]   = self._proc.get("ParamAngleX", 0.0) + self._pose_cur_x
        self._proc["ParamAngleY"]   = self._proc.get("ParamAngleY", 0.0) + self._pose_cur_y

    # Idle animations (breath / sway / body drift)
    

    def _tick_idle(self, dt: float) -> None:
        TAU = math.pi * 2

        # Breathing: ParamBreath 0→1, period 4 s (procedural override) 
        self._t_breath = (self._t_breath + dt) % TAU
        breath = 0.5 + 0.5 * math.sin(self._t_breath * TAU / 4.0)
        self._proc["ParamBreath"] = breath

        # Head sway: additive on AngleZ, period 8 s, ±3°
        self._t_sway = (self._t_sway + dt) % TAU
        sway = 3.0 * math.sin(self._t_sway * TAU / 8.0)
        # Additive: mood's AngleZ stays in _current; _flush_params sums them.
        self._proc["ParamAngleZ"] = sway

        # Body drift: slow sine on BodyAngleX0/Z0, period 12/9 s, ±2°/±1.5° (additive on mood base)
        self._t_body = (self._t_body + dt) % TAU
        body_x = 2.0 * math.sin(self._t_body * TAU / 12.0)
        body_z = 1.5 * math.sin(self._t_body * TAU / 9.0)
        self._proc["ParamBodyAngleX0"] = body_x
        self._proc["ParamBodyAngleZ0"] = body_z

   
    # Blink

    _BLINK_CLOSE = 0.12   # s to close
    _BLINK_HOLD  = 0.08   # s held closed
    _BLINK_OPEN  = 0.15   # s to re-open

    # Cursor → param scaling. Normalized[-1,1] cursor → degrees/units.
    _HEAD_X_SCALE = 12.0    # horizontal cursor extent → ParamAngleX degrees
    _HEAD_Y_SCALE = 10.0    # vertical cursor extent → ParamAngleY degrees
    _EYE_X_SCALE  = 0.5     # horizontal cursor → ParamEyeBallX units

    @staticmethod
    def _rand_blink_interval() -> float:
        return random.uniform(3.0, 6.0)

    def _tick_blink(self, dt: float) -> None:
        if self._blink_state == "open":
            self._blink_next -= dt
            if self._blink_next <= 0.0:
                self._blink_state = "closing"
                self._blink_t = 0.0
            return

        self._blink_t += dt

        if self._blink_state == "closing":
            p = min(self._blink_t / self._BLINK_CLOSE, 1.0)
            v = 1.0 - p
            self._proc["ParamEyeLOpen"] = v
            self._proc["ParamEyeROpen"] = v
            if p >= 1.0:
                self._blink_state = "closed"
                self._blink_t = 0.0

        elif self._blink_state == "closed":
            # Reach exactly 0.0 (fully closed) — written as procedural override
            self._proc["ParamEyeLOpen"] = 0.0
            self._proc["ParamEyeROpen"] = 0.0
            if self._blink_t >= self._BLINK_HOLD:
                self._blink_state = "opening"
                self._blink_t = 0.0

        elif self._blink_state == "opening":
            p = min(self._blink_t / self._BLINK_OPEN, 1.0)
            self._proc["ParamEyeLOpen"] = p
            self._proc["ParamEyeROpen"] = p
            if p >= 1.0:
                self._blink_state = "open"
                self._blink_t = 0.0
                self._blink_next = self._rand_blink_interval()

    
    # Lip sync
  

    _MOUTH_ATTACK  = 0.30
    _MOUTH_RELEASE = 0.08

    @staticmethod
    def _get_music_pos():
        """Best-effort playback position (ms) from pygame, or None if unavailable.

        Lips are indexed against the SAME clock the audio plays from, so the
        mouth stays in sync with what you hear.  When position can't be read
        (no mixer / no stream) we fall back to the random flap.
        """
        try:
            import pygame
            if pygame.mixer.get_init() is None:
                return None
            pos = pygame.mixer.music.get_pos()
            return float(pos) if pos >= 0 else None
        except Exception:
            return None

    def _tick_mouth(self, dt: float) -> None:
        # Phase 4a: when lip sync is on and a decoded envelope is installed for
        # this clip, drive the mouth from the actual loudness at the current
        # playback position.  value_at applies the attack/release smoothing and
        # the min_open floor, so no extra easing is needed here.
        if (self._speaking and self._lip_sync_enabled
                and self._lip.has_envelope):
            pos = self._get_music_pos()
            if pos is not None:
                val = self._lip.value_at(pos)
                if val is not None:
                    self._mouth_current = val
                    self._proc["ParamMouthOpenY"] = self._mouth_current
                    # Optional small ParamMouthForm emotion tint (mood-based)
                    if self._mouth_tint_form:
                        self._proc["ParamMouthForm"] = (
                            self._mouth_form_base + self._mouth_tint_form)
                    return

        # Fallback: random flap (also the path when not speaking / no envelope).
        if self._speaking:
            self._mouth_noise_timer += dt
            if self._mouth_noise_timer > 0.075:
                self._mouth_noise_timer = 0.0
                self._mouth_target = random.uniform(0.5, 0.92)
        else:
            self._mouth_target = 0.0

        alpha = self._MOUTH_ATTACK if self._mouth_target > self._mouth_current else self._MOUTH_RELEASE
        self._mouth_current += (self._mouth_target - self._mouth_current) * alpha
        # Procedural override — written straight to the model by _flush_params
        self._proc["ParamMouthOpenY"] = self._mouth_current

   
    # Eye tracking / hit test
   

    def feed_cursor(self, local_x: float, local_y: float) -> None:
        """Feed widget-local cursor position for head/eye tracking.

        Stores a normalized [-1, 1] offset relative to the widget centre; the
        easing and parametric scaling happen later in _tick_cursor so the model
        gl translates smoothly even if mouse events are sparse.
        """
        w = max(1, self.width())
        h = max(1, self.height())
        self._cursor_x = ((local_x - w / 2.0) / (w / 2.0))
        self._cursor_y = ((local_y - h / 2.0) / (h / 2.0))

    def hit_test(self, local_x: float, local_y: float) -> bool:
        if self._model is None or not self._initialized:
            return True
        try:
            return len(self._model.HitPart(local_x, local_y, False)) > 0
        except Exception:
            return True
