"""Desktop pet GUI — wraps VoiceCompanion with a fully-featured PySide6 window.

New in this version:
  • Typing indicator  — pulsing "..." animation while Cerebras is generating
  • Mood crossfade    — smooth opacity transition between mood images
  • Click zones       — left-click face = talk, right-click = context menu
  • Bubble flip       — speech bubble moves below the pet if near the top of screen
  • Tray notifications— system balloon when she speaks proactively
  • Proactive messages— QTimer fires Marin's message after N minutes of idle
  • InterruptMonitor  — she stops mid-sentence when you start talking
  • /speed /reload    — forwarded from tray / right-click menu
  • Live2D renderer   — QOpenGLWidget face with mood parameters, blink, breath,
                        lip sync; falls back to sprite FaceLabel if unavailable
  • Adjustable layout — Ctrl+wheel resizes face; Alt+wheel zooms model;
                        Alt/Shift+drag pans model; right-click presets

Run:
    python src/app.py
    python src/app.py --hotkey f9 --size 400
    python src/app.py --art characters/marin --config-dir config
"""
import re
import sys
import os
import json
import time
import logging
import argparse
import threading
import datetime
import random
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional

# pythonw safety (item 5)
# pythonw.exe has no console; stdout/stderr are None, which crashes anything
# that tries to print (progress bars, logger StreamHandler, etc.).
ROOT = Path(__file__).resolve().parent.parent

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _log_crash(exc_type, exc_value, exc_tb) -> None:
    """Append a timestamped traceback to logs/crash.log, then call the default hook."""
    try:
        log_dir = ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "crash.log", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"{datetime.datetime.now().isoformat()}  UNCAUGHT EXCEPTION\n")
            f.writelines(traceback.format_exception(exc_type, exc_value, exc_tb))
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _log_crash
threading.excepthook = lambda args: _log_crash(
    args.exc_type, args.exc_value, args.exc_traceback
)
# ───────────────

import pygame
import keyboard
from PySide6.QtCore import Qt, QObject, Signal, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QRect, QSize
from PySide6.QtGui import QPixmap, QPainter, QColor, QIcon, QPen, QMovie, QGuiApplication, QWheelEvent
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QSystemTrayIcon, QMenu,
    QInputDialog, QMessageBox, QGraphicsOpacityEffect,
)

from she import VoiceCompanion
from core.config import Config
from core.model import parse_mood_tag, split_sentences
from core.recording import record_until_silence, InterruptMonitor
from body_tags import StreamingTagParser, actions as body_actions, build_body_prompt

# Phase 2b: pure behavior state model + motion budget (no GL/Qt — safe to import
# unconditionally; only used when the Live2D renderer is active).
from behavior_state import BehaviorState, FeelingsBucket
from behavior_director import MotionBudget
from proactive import ProactiveChooser, ProactiveLimits, ProactiveScheduler
from body_stats import BodyStats

logger = logging.getLogger(__name__)  # noqa: E402

# ── Live2D availability check 
# Import is attempted here so the flag is available before Pet is constructed.
# The actual widget import happens lazily in Pet.__init__ so a missing package
# does not abort startup.
_LIVE2D_AVAILABLE = False
try:
    import live2d  # type: ignore   # just probe; real import is in live2d_face
    from live2d_face import Live2DWidget, configure_opengl_format
    _LIVE2D_AVAILABLE = True
except Exception as _live2d_err:
    logging.getLogger(__name__).warning(
        f"live2d-py unavailable — sprite fallback active: {_live2d_err}"
    )
    print("[MARIN] live2d renderer unavailable — using sprite mode. "
          "If you expected the 3D model, run diagnose.bat and send "
          "diagnose_out.txt to the developer.")
    Live2DWidget = None                # type: ignore
    configure_opengl_format = lambda: None  # type: ignore

# ── CLI args 
ap = argparse.ArgumentParser()
ap.add_argument("--config-dir", default=None)
ap.add_argument("--hotkey", default="f9")
ap.add_argument("--art", default=None)
ap.add_argument("--size", type=int, default=None,
                help="Override starting face size (pixels, default from config/live2d.json or 400)")
args = ap.parse_args()

# ── Live2D config (size + framing) 
LIVE2D_CFG_FILE = ROOT / "config" / "live2d.json"
STATE_FILE      = ROOT / "config" / "pet_state.json"
ART_DIR         = ROOT / "characters" / "marin"           # overridden in main()

def _load_live2d_cfg() -> dict:
    defaults = {
        "face_renderer": "sprites",
        "face": {
            "size": 400, "size_min": 200, "size_max": 900,
            "scale": 2.2, "offset_x": 0.0, "offset_y": 0.35, "fps": 30,
        },
        "invert_eye_x": False,
        "invert_head_x": False,
        "invert_head_y": False,
        "idle_loop": True,
        "body_control": True,
        "body_log": False,
        "state_in_prompt": False,
        "wink_swap": False,
        "look_invert_x": False,
        "behavior_director": True,
        "liveliness": 0.7,
        "lean_param": "ParamBodyAngleY0",
        "lean_sign": 1.0,
        "breath_pause_enabled": True,
        "breath_pause_every": [40.0, 90.0],
        "breath_pause_hold": [1.0, 1.5],
        "lip_sync_enabled": True,
        "lip_sync_offset_ms": 0.0,
        "motion_budget": {"max_head_angle": 8.0, "max_eye_offset": 0.5,
                          "max_body_drift": 2.0},
    }
    try:
        raw = json.loads(LIVE2D_CFG_FILE.read_text(encoding="utf-8"))
        face = raw.get("face", {})
        defaults["face_renderer"] = raw.get("face_renderer", defaults["face_renderer"])
        for k, v in face.items():
            defaults["face"][k] = v
        for k in ("invert_eye_x", "invert_head_x", "invert_head_y", "idle_loop",
                  "body_control", "body_log", "state_in_prompt", "wink_swap", "look_invert_x",
                  "behavior_director", "liveliness", "breath_pause_enabled",
                  "lip_sync_enabled"):
            if k in raw:
                defaults[k] = bool(raw[k]) if k not in ("liveliness",) else float(raw[k])
        defaults["lean_param"] = raw.get("lean_param", defaults["lean_param"])
        defaults["lean_sign"] = float(raw.get("lean_sign", defaults["lean_sign"]))
        defaults["breath_pause_every"] = raw.get("breath_pause_every", defaults["breath_pause_every"])
        defaults["breath_pause_hold"] = raw.get("breath_pause_hold", defaults["breath_pause_hold"])
        defaults["lip_sync_offset_ms"] = float(raw.get("lip_sync_offset_ms", defaults["lip_sync_offset_ms"]))
        if isinstance(raw.get("motion_budget"), dict):
            defaults["motion_budget"] = {
                **defaults["motion_budget"], **raw["motion_budget"]}
    except Exception:
        pass
    return defaults

_live2d_cfg = _load_live2d_cfg()
_FACE_RENDERER = _live2d_cfg["face_renderer"]   # "live2d" | "sprites"

# Body-control features (config/live2d.json).  body_control gates whether the
# streaming reply is parsed into sentences with tags; body_log logs every
# received/executed body action to logs/wife.log.
_BODY_CONTROL = bool(_live2d_cfg.get("body_control", True))
_BODY_LOG = bool(_live2d_cfg.get("body_log", False))

# Phase 4b: running body-control statistics, logged as [body] lines.
_BODY_STATS = BodyStats()

# Phase 3b: state-aware personality.  When on (and the BODY rule holds), a
# short "Right now you feel …" line is appended to the very end of the system
# message, derived only from her own feeling (never the user's behaviour).
# Default OFF — opt-in.
_STATE_IN_PROMPT = bool(_live2d_cfg.get("state_in_prompt", False))

# Tuning for the body-control L/R mapping (used in Phase 1c to map screen-side
# to model params on the Live2D model).  Left/right = the screen side as the
# viewer sees it; invert the X axis to compensate for models whose forward is
# flipped, and swap the wink eye params if the character's sides are reversed.
_WINK_SWAP = bool(_live2d_cfg.get("wink_swap", False))
_LOOK_INVERT_X = bool(_live2d_cfg.get("look_invert_x", False))

# Phase 2b: behavior-director switch (fall back to the old idle behaviour when
# off) and the motion budget (config numbers, subtle by default).  liveliness is
# a plain 0..1 scalar; presets/menu come in 2e.
_BEHAVIOR_DIRECTOR = bool(_live2d_cfg.get("behavior_director", True))
_LIVELINESS = float(_live2d_cfg.get("liveliness", 0.7))
_MOTION_BUDGET_CFG = _live2d_cfg.get("motion_budget", {}) or {}

# Phase 2c: which body param the lean uses + direction (config value; the preview
# tool lets you try the alternatives visually) and the optional breath pauses.
_LEAN_PARAM = _live2d_cfg.get("lean_param", "ParamBodyAngleY0")
_LEAN_SIGN = float(_live2d_cfg.get("lean_sign", 1.0))
_BREATH_PAUSE_ENABLED = bool(_live2d_cfg.get("breath_pause_enabled", True))
_BREATH_PAUSE_EVERY = _live2d_cfg.get("breath_pause_every", [40.0, 90.0])
_BREATH_PAUSE_HOLD = _live2d_cfg.get("breath_pause_hold", [1.0, 1.5])

# Phase 4a: real lip sync.  lip_sync_enabled gates envelope decoding; the offset
# tunes audio-vs-mouth timing (positive = mouth lags audio) in milliseconds.
_LIP_SYNC_ENABLED = bool(_live2d_cfg.get("lip_sync_enabled", True))
_LIP_SYNC_OFFSET_MS = float(_live2d_cfg.get("lip_sync_offset_ms", 0.0))

RELATIONSHIP_FILE = ROOT / "config" / "relationship.json"


def _closeness_baseline() -> float:
    """Small persistent baseline from config/relationship.json for the state model.
    Transient mood is NOT persisted; this only seeds valence/engagement base."""
    try:
        return float(json.loads(RELATIONSHIP_FILE.read_text(encoding="utf-8")).get("closeness", 0.0) or 0.0)
    except Exception:
        return 0.0

# Sentences shorter than this are merged into the next one (so exclamations like
# "Oh!" are not separate TTS calls) in the body-control streaming path.
_MIN_SENTENCE_CHARS = 10


def _resolve_body_actions(tags) -> list:
    """Convert body Tags (from body_tags) into concrete action dicts.

    The Live2D widget consumes these dicts (kind/name/params/duration/side).
    Mood tags are skipped — the mood is already applied separately via on_mood.
    """
    ba = body_actions()
    acts = []
    for t in tags:
        if t.kind == "clear":
            acts.append({"kind": "clear"})
        elif t.kind == "mood":
            continue
        elif t.kind == "fx":
            cfg = ba.fx.get(t.name)
            if cfg:
                acts.append({"kind": "fx", "name": t.name,
                             "params": cfg.get("params", {}) or {},
                             "expression_ops": cfg.get("expression_ops", []) or [],
                             "duration": cfg.get("duration", 4.0)})
        elif t.kind == "look":
            cfg = ba.look.get(t.name)
            if cfg:
                acts.append({"kind": "look", "name": t.name,
                             "params": cfg.get("params", {}) or {},
                             "expression_ops": cfg.get("expression_ops", []) or [],
                             "duration": cfg.get("duration", 0.8)})
        elif t.kind == "gesture":
            cfg = ba.gestures.get(t.name)
            if cfg:
                acts.append({"kind": "gesture", "name": t.name,
                             "params": cfg.get("params", {}) or {},
                             "expression_ops": cfg.get("expression_ops", []) or [],
                             "duration": cfg.get("duration", 0.9)})
        elif t.kind == "wink":
            wc = ba.wink_config()
            if wc:
                acts.append({"kind": "wink", "side": t.name,
                             "duration": wc.get("duration", 0.5)})
    return acts

# Determine starting size: --size CLI > saved state > config
def _resolve_initial_size() -> int:
    if args.size is not None:
        return max(200, min(900, args.size))
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        s = int(state.get("face_size", 0))
        if s >= 200:
            return s
    except Exception:
        pass
    return int(_live2d_cfg["face"].get("size", 400))

_INITIAL_SIZE = _resolve_initial_size()

# ── Module-level size constants (LEGACY — kept for FaceLabel/placeholder use
#    but NOT used directly in Pet geometry; Pet._apply_layout() owns that) ─────
SIZE     = _INITIAL_SIZE
WIDTH    = int(SIZE * 1.2)
BUBBLE_H = int(SIZE * 0.7)
FACE_TOP = BUBBLE_H + 6

MOOD_COLORS = {
    "idle":        "#b8a6ff",
    "joyful":      "#ffd56b",
    "excited":     "#ff9ecb",
    "embarrassed": "#ff8f8f",
    "sad":         "#8fb8ff",
    "angry":       "#ff6b6b",
    "surprised":   "#9be7ff",
    "curious":     "#b9f29a",
    "relaxed":     "#c9b8ff",
}


# Art helpers 

def placeholder(mood: str, mouth_open: bool, size: int = None) -> QPixmap:
    s = size if size is not None else SIZE
    pm = QPixmap(s, s)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    pen_w = max(2, s // 55)
    p.setPen(QPen(QColor("#2a2150"), pen_w))
    p.setBrush(QColor(MOOD_COLORS.get(mood, MOOD_COLORS["idle"])))
    m = s // 22
    p.drawEllipse(m, m, s - m * 2, s - m * 2)
    p.setBrush(QColor("#2a2150"))
    ey = int(s * 0.39)
    ew = s // 11
    eh = int(s / 7.86)
    p.drawEllipse(int(s * 0.32), ey, ew, eh)
    p.drawEllipse(int(s * 0.59), ey, ew, eh)
    mx = int(s * 0.41)
    mw = int(s // 5.5)
    if mouth_open:
        p.drawEllipse(mx, int(s * 0.61), mw, int(s / 7.33))
    else:
        p.setBrush(Qt.NoBrush)
        arc_y = int(s * (0.66 if mood in ("sad", "angry") else 0.59))
        arc_span = 180 * 16 if mood in ("sad", "angry") else -180 * 16
        p.drawArc(mx - 10, arc_y, int(s * 0.27), int(s / 7.33), 0 if mood in ("sad", "angry") else 180 * 16, arc_span)
    p.end()
    return pm


def load_art(mood: str, mouth_open: bool):
    """Return (path_str_or_None, is_gif)."""
    names = []
    if mouth_open:
        names += [f"{mood}_open.gif", f"{mood}_open.png"]
    names += [f"{mood}.gif", f"{mood}.png", "idle.gif", "idle.png"]
    for n in names:
        f = ART_DIR / n
        if f.exists():
            return str(f), n.endswith(".gif")
    return None, False


def load_pixmap(mood: str, mouth_open: bool, size: int = None) -> QPixmap:
    s = size if size is not None else SIZE
    path, is_gif = load_art(mood, mouth_open)
    if path is None:
        return placeholder(mood, mouth_open, s)
    if is_gif:
        movie = QMovie(path)
        movie.jumpToFrame(0)
        px = movie.currentPixmap()
        return px.scaledToHeight(s, Qt.SmoothTransformation) if not px.isNull() else placeholder(mood, mouth_open, s)
    px = QPixmap(path)
    return px.scaledToHeight(s, Qt.SmoothTransformation) if not px.isNull() else placeholder(mood, mouth_open, s)


# ── Signal bus 

class Bus(QObject):
    start    = Signal()           # LLM started responding
    text     = Signal(str)        # incremental text chunk
    mood     = Signal(str)        # mood tag detected
    body     = Signal(object)     # fired at a sentence's audio start: (gen, text, tags)
    end      = Signal()           # LLM finished
    status   = Signal(str, bool)  # (message, keep_showing)
    proact   = Signal(str)        # proactive message from idle timer
    typing   = Signal(bool)       # True = show typing indicator
    listening = Signal(bool)      # True = user is speaking (recording active)


# ── Extended companion ────────────────────────────────────────────────────────

class PetCompanion(VoiceCompanion):
    def __init__(self, config: Config, bus: Bus):
        super().__init__(config)
        self.bus = bus
        self.turn_lock = threading.Lock()
        self.interrupt_monitor = InterruptMonitor(
            sample_rate=config.sample_rate,
            threshold_multiplier=3.5,
        )
        # Phase 3b: coarse, hysteresis-smoothed buckets of HER OWN feeling, held
        # here (not in BehaviorState).  ``_state_getter`` is wired from main() to
        # return pet._state_snapshot (the immutable per-tick snapshot), never the
        # live mutable BehaviorState.
        self._feelings = FeelingsBucket()
        self._state_getter: Optional[Callable[[], Optional[object]]] = None
        # Phase 3d: state-aware proactive messages.  The chooser picks a tone
        # pool from her feeling snapshot; the scheduler enforces the approved
        # limits (minimum interval, daily cap, quiet hours, do-not-disturb).
        self._proactive_chooser = ProactiveChooser(pools=self.config.proactive_pools)
        self._proactive_scheduler = ProactiveScheduler(ProactiveLimits(
            min_interval_minutes=self.config.proactive_min_interval_minutes,
            daily_cap=self.config.proactive_daily_cap,
            quiet_start_hour=self.config.proactive_quiet_start_hour,
            quiet_end_hour=self.config.proactive_quiet_end_hour,
            dnd=self.config.proactive_dnd,
        ))

    def _state_line(self) -> Optional[str]:
        """The short self-feeling line, or None when the state feature is off.

        Gated by the same rule as the BODY section (Live2D renderer active +
        live2d available + body_control) AND ``state_in_prompt``.  Describes
        only her own feeling (calm / cheerful / a little tired + energy) — never
        the user's behaviour, never ignored/wandered/away.
        """
        if not (_FACE_RENDERER == "live2d" and _LIVE2D_AVAILABLE
                and _BODY_CONTROL and _STATE_IN_PROMPT):
            return None
        if self._state_getter is None:
            return None
        try:
            snap = self._state_getter()
        except Exception:
            return None
        if snap is None:
            return None
        phrase = self._feelings.phrase(snap)
        return (f"Right now you feel {phrase}. Do not mention or complain about "
                "this state, about being ignored, or about how long the user "
                "was away unless they bring it up.")

    def system_message(self) -> Dict[str, str]:
        msg = super().system_message()
        # BODY section is only injected when the Live2D renderer is actually
        # enabled (so a sprite/user, or a headless run, never gets it).
        if _FACE_RENDERER == "live2d" and _LIVE2D_AVAILABLE and _BODY_CONTROL:
            msg["content"] += build_body_prompt(body_actions())
        # Phase 3b: the state line goes at the VERY END of the system message.
        line = self._state_line()
        if line is not None:
            msg["content"] += "\n\n" + line
            if _BODY_LOG:
                logger.info("[body] state -> prompt: %s", line)
        return msg

    def start_turn(self):
        """Called by hotkey / face click — begins a voice turn."""
        self.speaker.stop()
        if not self.turn_lock.locked():
            threading.Thread(target=self._run_turn, daemon=True).start()

    def _run_turn(self):
        if not self.turn_lock.acquire(blocking=False):
            return
        b = self.bus
        try:
            b.status.emit("Listening…", True)
            b.listening.emit(True)   # Phase 2b: user is speaking (recording active)
            try:
                audio = record_until_silence(
                    sample_rate=self.config.sample_rate,
                    max_seconds=self.max_record_seconds,
                    threshold_multiplier=self.config.vad_threshold_multiplier,
                )
            finally:
                b.listening.emit(False)  # silence detected / recording finished
            if audio is None or len(audio) / self.config.sample_rate < self.config.min_audio_length:
                b.status.emit("(didn't catch that)", False)
                self._missed_ack()
                return

            text = self.transcribe(audio)
            if not text:
                b.status.emit("(didn't catch that)", False)
                self._missed_ack()
                return

            reply = self.try_command(text)
            if reply:
                b.status.emit(reply, False)
                self.speaker.say(reply)
                self.memory.log_turn("user", text)
                self.memory.log_turn("assistant", reply)
                self.last_exchange = (text, reply)
                return

            b.typing.emit(True)
            self.history.append({"role": "user", "content": text})
            self._maybe_summarize()
            messages = [self.system_message()] + self.history[-self.config.max_history:]

            try:
                reply_text, mood = self.stream_reply(messages)
            except Exception:
                # stream_reply already emitted and spoke the fallback; just clean up
                self.history.pop()
                return

            # item 15: if interrupted before any text was generated, discard the turn
            if not reply_text:
                self.history.pop()
                return

            self.history.append({"role": "assistant", "content": reply_text})
            self._trim_history()   # item 15: use safe trim instead of raw slice
            self.memory.log_turn("user", text)
            self.memory.log_turn("assistant", reply_text, mood)
            self.memory.after_turn()
            self.last_exchange = (text, reply_text)
            self._extract_facts(text, reply_text)

        except Exception as e:
            b.status.emit(f"Error: {e}", False)
        finally:
            b.typing.emit(False)
            self.turn_lock.release()

    def stream_reply(self, messages):
        """Like VoiceCompanion.stream_reply but also drives the GUI bus.

        When body control is active (Live2D renderer + config body_control) the
        stream is parsed by StreamingTagParser into clean sentences carrying
        inline tags; otherwise the legacy leading-mood path runs unchanged."""
        from core.model import next_fallback
        mood = None
        interrupted = False   # item 14
        b = self.bus
        b.start.emit()

        # Part 3-A: start the interrupt monitor now; keep it alive until TTS
        # drains completely (not just until LLM streaming ends).
        # A 500 ms grace period at start prevents her own speaker output from
        # immediately triggering the threshold (feedback false-positive).
        self.interrupt_monitor.start()
        _grace_until = [0.0]   # mutable cell for lambda; set after first audio

        import time as _time

        body_on = _BODY_CONTROL and _FACE_RENDERER == "live2d" and _LIVE2D_AVAILABLE
        if _BODY_LOG:
            if body_on:
                logger.info("[body] BODY section + tag parsing ON "
                            "(renderer=%s, live2d_available=%s, body_control=%s)",
                            _FACE_RENDERER, _LIVE2D_AVAILABLE, _BODY_CONTROL)
            else:
                why = ("live2d renderer is active" if _FACE_RENDERER == "live2d"
                       else f"renderer={_FACE_RENDERER!r}")
                if not _LIVE2D_AVAILABLE and _FACE_RENDERER == "live2d":
                    why += ", live2d module unavailable"
                if not _BODY_CONTROL:
                    why += ", body_control disabled in config"
                logger.info("[body] BODY section + tag parsing OFF (%s)", why)

        try:
            if body_on:
                full, mood = self._stream_body(messages, b, _grace_until, _time)
                return full, mood

            full = buf = head = ""
            mood_done = False
            for chunk in self.model.stream_chat(messages):
                # Only check interrupt after grace period has elapsed
                if (self.interrupt_monitor.triggered
                        and _time.monotonic() > _grace_until[0]):
                    self.speaker.stop()
                    interrupted = True
                    break

                piece = chunk["message"]["content"]
                if not mood_done:
                    head += piece
                    parsed, remaining = parse_mood_tag(head)
                    if parsed is not None:
                        mood = parsed
                        piece, mood_done = remaining, True
                        b.mood.emit(mood)
                    elif len(head.lstrip()) > 30:
                        piece, mood_done = head, True
                    else:
                        continue

                full += piece
                buf += piece
                b.text.emit(piece)

                parts = split_sentences(buf, min_length=20)
                if len(parts) == 2:
                    complete, buf = parts
                    self.speaker.say(complete, mood)
                    # Start grace period on first queued sentence
                    if _grace_until[0] == 0.0:
                        _grace_until[0] = _time.monotonic() + 0.5

            # item 14: don't speak leftover buffer if the user interrupted
            if not mood_done and not interrupted:
                full = buf = head
                b.text.emit(head)
            if buf and not interrupted:
                self.speaker.say(buf, mood)
                if _grace_until[0] == 0.0:
                    _grace_until[0] = _time.monotonic() + 0.5

            return full.strip(), mood

        except Exception as e:
            fallback = next_fallback()
            b.text.emit(fallback)
            self.speaker.say(fallback)
            raise

        finally:
            # Part 3-A: poll until all TTS has finished, then stop the monitor.
            # This is done in a daemon thread so the finally block returns fast.
            def _wait_and_stop():
                import time as _t
                deadline = _t.monotonic() + 30.0   # hard cap: 30 s max wait
                while _t.monotonic() < deadline:
                    if not self.speaker.is_playing():
                        break
                    if self.interrupt_monitor.triggered:
                        self.speaker.stop()
                        break
                    _t.sleep(0.05)
                self.interrupt_monitor.stop()
                self.interrupt_monitor.reset()

            threading.Thread(target=_wait_and_stop, daemon=True).start()
            b.end.emit()

    def _stream_body(self, messages, b, _grace_until, _time):
        """Body-control streaming path: parse tags into sentences and speak each
        with its tags.  Returns ``(clean_full_text, reply_mood)``.

        Bubble text is revealed at audio start (via on_sentence_start → on_body),
        NOT here.  Very short sentences (< _MIN_SENTENCE_CHARS) are merged into the
        next one so "Oh!" / "Hmm." are not separate TTS calls; their tags stay
        attached to the first sentence and fire at the merged audio start."""
        from body_tags import StreamingTagParser
        parser = StreamingTagParser()
        full_parts: List[str] = []
        reply_mood = None          # first mood seen = reply default (like legacy)
        mood_shown = None          # last mood emitted to the GUI bus
        pending = None             # a too-short sentence awaiting merge
        # Phase 4b: per-reply honoured tags by kind (across all spoken frames).
        reply_kinds: Dict[str, int] = {}

        def _speak(text, mood, tags):
            if _grace_until[0] == 0.0:
                _grace_until[0] = _time.monotonic() + 0.5
            self.speaker.say(text, mood, tags=tags)

        def _count_tags(tags) -> None:
            for t in tags:
                reply_kinds[t.kind] = reply_kinds.get(t.kind, 0) + 1

        def _consume(f):
            nonlocal reply_mood, mood_shown, pending
            if f.mood is not None:
                if reply_mood is None:
                    reply_mood = f.mood
                if f.mood != mood_shown:
                    mood_shown = f.mood
                    b.mood.emit(f.mood)
            if pending is not None:
                # Merge the held short sentence into this one.  Tags stay attached
                # to the first sentence; both fire together at the merged audio start.
                text = pending.text + " " + f.text
                tags = pending.tags + f.tags
                _count_tags(tags)
                full_parts[-1] = text      # replace the held entry
                pending = None
                _speak(text, f.mood, tags)
                return
            if len(f.text) < _MIN_SENTENCE_CHARS:
                pending = f
                full_parts.append(f.text)
                return
            _count_tags(f.tags)
            full_parts.append(f.text)
            _speak(f.text, f.mood, f.tags)

        for chunk in self.model.stream_chat(messages):
            if (self.interrupt_monitor.triggered
                    and _time.monotonic() > _grace_until[0]):
                self.speaker.stop()
                return " ".join(full_parts).strip(), reply_mood
            piece = chunk["message"]["content"]
            for f in parser.feed(piece):
                _consume(f)
        for f in parser.finish():
            _consume(f)
        # A trailing short sentence with nothing to merge into still gets spoken.
        if pending is not None:
            _count_tags(pending.tags)
            _speak(pending.text, pending.mood, pending.tags)
        # Phase 4b: log per-reply tag stats (tags by kind + dropped counts).
        if _BODY_LOG:
            pstats = parser.stats()
            logger.info(
                "[body] reply stats: honoured=%s unknown_dropped=%d cap_dropped=%d",
                reply_kinds,
                pstats["unknown_dropped"],
                pstats["cap_dropped"],
            )
        return " ".join(full_parts).strip(), reply_mood

    def say_proactive(self, message: str):
        """Speak a proactive message and push it to the GUI."""
        self.bus.proact.emit(message)
        self.speaker.say(message)
        self.memory.log_turn("assistant", message)

    def _choose_proactive(self) -> Optional[str]:
        """State-aware pick (Phase 3d): choose a tone pool from her feeling and
        draw a message from it.  Returns None if suppressed (ignored long —
        fewer messages, never more insistent; no reproach)."""
        snap = None
        if self._state_getter is not None:
            try:
                snap = self._state_getter()
            except Exception:
                snap = None
        return self._proactive_chooser.choose(snap)

    def generate_proactive(self) -> str:
        """Produce a proactive message.

        Primary path (Phase 3d): a static, state-aware pick from the tagged
        pools in personality.json — no LLM call per message.  Ignored-long
        suppression returns "" (the timer reschedules without speaking).
        Only when ``proactive_generate`` is explicitly enabled does it fall
        back to an LLM call with time-of-day + memory context, still falling
        back to the state-aware pools on failure.
        """
        if not self.config.proactive_generate:
            return self._choose_proactive() or ""

        try:
            hour = datetime.datetime.now().hour
            if hour < 5:
                time_hint = "It's very late (past midnight)."
            elif hour < 12:
                time_hint = "It's still morning."
            elif hour < 18:
                time_hint = "It's afternoon."
            else:
                time_hint = "It's evening."

            ctx = self.memory.build_context_block()
            name = self.config.persona["name"]
            prompt = [
                {"role": "system", "content": (
                    f"You are {name}, a companion who notices her user has gone "
                    "quiet after a while. Write ONE short spoken line (1–2 "
                    "sentences) of proactive dialogue. It should feel natural, "
                    "in-character, and aware of the context below. Reply with "
                    "the dialogue only — no labels, no asterisks, no quotes.\n\n"
                    f"Time context: {time_hint}\n\n"
                    f"Memory/relationship context:\n{ctx or '(none yet)'}"
                )},
                {"role": "user", "content": "(The user has been silent for a while. Say something to them in character.)"},
            ]
            full = ""
            for chunk in self.model.stream_chat(prompt, temperature=0.9):
                full += chunk["message"]["content"]
            line = full.strip().strip('"')
            if 10 <= len(line) <= 200:
                return line
        except Exception as e:
            logging.getLogger(__name__).warning(f"Proactive generation failed: {e}")

        return self._choose_proactive() or ""

    @staticmethod
    def _safe_error() -> str:
        from core.model import next_fallback
        return next_fallback()


# ── Mood face label with crossfade ────────────────────────────────────────────

class FaceLabel(QLabel):
    """QLabel that cross-fades between mood frames.

    Fixes vs. old version:
    - Uses QGraphicsOpacityEffect on the overlay instead of setWindowOpacity
      (setWindowOpacity is a no-op on child widgets).
    - Dedup key: update_art is a no-op when mood+mouth_open didn't change,
      so GIFs keep playing and lip-sync doesn't thrash file I/O.
    - Aborts any in-flight fade before starting a new one.
    - Preserves GIF aspect ratio via QSize rather than scaledToHeight.
    """

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)

        self._overlay = QLabel(parent)
        self._overlay.setAlignment(Qt.AlignCenter)
        self._overlay.setGeometry(self.geometry())
        self._overlay.hide()

        # QGraphicsOpacityEffect is the correct way to fade a child widget
        self._fx = QGraphicsOpacityEffect(self._overlay)
        self._fx.setOpacity(1.0)
        self._overlay.setGraphicsEffect(self._fx)

        self._anim: QPropertyAnimation | None = None
        self._current_movie: QMovie | None = None
        self._overlay_movie: QMovie | None = None
        self._key = None          # (path_or_tag, ...) — dedup guard

    def setGeometry(self, *args):
        super().setGeometry(*args)
        self._overlay.setGeometry(*args)

    # ── Public ────────────────────────────────────────────────────────────────

    def update_art(self, mood: str, mouth_open: bool, crossfade: bool = True):
        path, is_gif = load_art(mood, mouth_open)
        # Determine current face size from parent Pet if available
        _size = SIZE
        try:
            _size = self.parent().face_size  # type: ignore[union-attr]
        except Exception:
            pass

        # Dedup: identical key → do nothing (keeps GIFs running, avoids thrash)
        new_key = (path, mood, mouth_open, _size) if path else ("placeholder", mood, mouth_open, _size)
        if new_key == self._key:
            return
        self._key = new_key

        # Abort any animation that is still running
        self._abort_fade()

        # Populate overlay with new art
        if path is None:
            self._overlay.setPixmap(placeholder(mood, mouth_open, _size))
            if self._overlay_movie:
                self._overlay_movie.stop()
                self._overlay_movie = None
        elif is_gif:
            if self._overlay_movie:
                self._overlay_movie.stop()
            self._overlay_movie = self._scaled_movie(path, _size)
            self._overlay.setMovie(self._overlay_movie)
            self._overlay_movie.start()
        else:
            if self._overlay_movie:
                self._overlay_movie.stop()
                self._overlay_movie = None
            px = QPixmap(path).scaledToHeight(_size, Qt.SmoothTransformation)
            self._overlay.setPixmap(px)

        if not crossfade:
            self._swap_immediately()
            return

        # Crossfade: animate opacity 0 → 1 on the overlay via QGraphicsOpacityEffect
        self._fx.setOpacity(0.0)
        self._overlay.raise_()
        self._overlay.show()

        self._anim = QPropertyAnimation(self._fx, b"opacity", self)
        self._anim.setDuration(250)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.InOutQuad)
        self._anim.finished.connect(self._swap_immediately)
        self._anim.start()

    # ── Private ───────────────────────────────────────────────────────────────

    def _abort_fade(self) -> None:
        """Stop any in-flight crossfade cleanly."""
        if self._anim is not None:
            try:
                self._anim.finished.disconnect(self._swap_immediately)
            except (RuntimeError, TypeError):
                pass
            self._anim.stop()
            self._anim = None
        self._overlay.hide()

    @staticmethod
    def _scaled_movie(path: str, size: int = None) -> QMovie:
        """Return a QMovie scaled to size height with correct aspect ratio."""
        s = size if size is not None else SIZE
        movie = QMovie(path)
        movie.jumpToFrame(0)
        img_size = movie.currentImage().size()
        if img_size.height() > 0:
            w = img_size.width() * s // img_size.height()
            movie.setScaledSize(QSize(w, s))
        return movie

    def _swap_immediately(self) -> None:
        """Copy overlay state into the base label and hide the overlay."""
        if self._overlay_movie:
            if self._current_movie and self._current_movie is not self._overlay_movie:
                self._current_movie.stop()
            self._current_movie = self._overlay_movie
            self._overlay_movie = None
            super().setMovie(self._current_movie)
        else:
            if self._current_movie:
                self._current_movie.stop()
                self._current_movie = None
            px = self._overlay.pixmap()
            super().setPixmap(px if px is not None else QPixmap())
        self._overlay.hide()
        self._anim = None


# ── Typing indicator ──────────────────────────────────────────────────────────

class TypingIndicator(QLabel):
    """Animated '...' dots that pulse while the LLM is generating."""

    _FRAMES = ["   ", ".  ", ".. ", "..."]

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "background: rgba(38,31,77,200); color: #b8a6ff; border-radius: 10px;"
            "font-size: 22px; font-weight: bold; padding: 6px 14px;"
        )
        self._frame = 0
        self._timer = QTimer(self, interval=350, timeout=self._tick)
        self.hide()

    def show_indicator(self):
        self._frame = 0
        self.setText(self._FRAMES[0])
        self.adjustSize()
        self.show()
        self._timer.start()

    def hide_indicator(self):
        self._timer.stop()
        self.hide()

    def _tick(self):
        self._frame = (self._frame + 1) % len(self._FRAMES)
        self.setText(self._FRAMES[self._frame])


# ── Status indicator (mic / speaker pill)

class StatusIndicator(QLabel):
    """Small pill in the corner of the pet: 🎤 listening, 🔊 talking, hidden otherwise."""

    _STYLES = {
        "listening": (
            "background: rgba(220,50,50,210); color: white; border-radius: 10px;"
            "font-size: 14px; padding: 3px 8px;"
        ),
        "talking": (
            "background: rgba(100,60,200,210); color: white; border-radius: 10px;"
            "font-size: 14px; padding: 3px 8px;"
        ),
    }
    _LABELS = {"listening": "🎤", "talking": "🔊"}

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self._state = "idle"
        self._alpha = 255
        self._rising = False
        self._pulse_timer = QTimer(self, interval=40, timeout=self._pulse_tick)
        self.hide()

    def set_state(self, state: str) -> None:
        """state: 'listening' | 'talking' | 'idle'"""
        self._state = state
        if state == "idle":
            self._pulse_timer.stop()
            self.hide()
            return
        self.setText(self._LABELS[state])
        self.setStyleSheet(self._STYLES[state])
        self.adjustSize()
        self._alpha = 255
        self._rising = False
        self._pulse_timer.start()
        self.show()
        self.raise_()

    def _pulse_tick(self) -> None:
        """Fade in/out between 140 and 255 alpha for a gentle pulse."""
        step = 12
        if self._rising:
            self._alpha = min(255, self._alpha + step)
            if self._alpha >= 255:
                self._rising = False
        else:
            self._alpha = max(140, self._alpha - step)
            if self._alpha <= 140:
                self._rising = True
        self.setWindowOpacity(self._alpha / 255)



class Pet(QWidget):
    def __init__(self, on_click):
        super().__init__()
        self.on_click = on_click
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # ── Dynamic size state (replaces module-level constants) ──────────────
        _cfg_face = _live2d_cfg.get("face", {})
        self.face_size    = _INITIAL_SIZE
        self.face_size_min = int(_cfg_face.get("size_min", 200))
        self.face_size_max = int(_cfg_face.get("size_max", 900))

        # Model framing (Live2D scale/offset, also saved to state)
        self._model_scale    = float(_cfg_face.get("scale", 2.2))
        self._model_offset_x = float(_cfg_face.get("offset_x", 0.0))
        self._model_offset_y = float(_cfg_face.get("offset_y", 0.35))

        # Persisted idle-loop override.  Loaded from pet_state.json FIRST so it
        # wins over the config default (config cannot re-clobber a user toggle).
        self._idle_enabled = self._load_idle_override(default=bool(_live2d_cfg.get("idle_loop", True)))

        # Phase 2e: persisted liveliness (0..1) + behavior-director on/off.  Both
        # are user overrides that survive restarts and the config cannot re-clobber.
        self._liveliness = self._load_liveliness_override(default=_LIVELINESS)
        self._behavior_director_user = self._load_behavior_director_override(default=_BEHAVIOR_DIRECTOR)

        # Live2D mood map for on_mood acceptance check
        self._live2d_moods: set[str] = set()

        # Debounce timer for saving state after resize
        self._save_state_timer = QTimer(self, singleShot=True, timeout=self._save_state_now)

        # Speech bubble — text gets set before first layout
        self.bubble = QLabel(self)
        self.bubble.setWordWrap(True)
        self.bubble.hide()

        # Typing indicator (shown while LLM generates, before first token)
        self.typing_label = TypingIndicator(self)

        # Status indicator — mic / speaker pill bottom-right of face
        self.status_indicator = StatusIndicator(self)

        # ── Face widget — Live2D or sprite fallback ───────────────────────────
        self.face = None
        self._use_live2d = False
        # Phase 2b: shared BehaviorState for the behavior director.  Closeness
        # from config/relationship.json seeds a small persistent baseline; the
        # app feeds mood/interact/hotkey/listening inputs and the widget drives
        # it per-frame (speaking state, director writes).
        self.behavior_state = BehaviorState(closeness=_closeness_baseline())
        # Phase 3b: immutable snapshot of the state, published each tick by the
        # widget's director loop and read once per turn by the companion (never
        # the live mutable BehaviorState).  Initialised now so reads are safe
        # even before the first frame.
        self._state_snapshot = self.behavior_state.snapshot()

        def _publish_state_snapshot():
            if self.behavior_state is not None:
                self._state_snapshot = self.behavior_state.snapshot()
        self._publish_state_snapshot = _publish_state_snapshot
        if _FACE_RENDERER == "live2d" and _LIVE2D_AVAILABLE:
            try:
                model_json  = ROOT / "characters" / "marin" / "live2d" / "简.model3.json"
                moods_json  = ROOT / "config" / "live2d_moods.json"
                self.face   = Live2DWidget(
                    model_json, moods_json, parent=self,
                    invert_eye_x=_live2d_cfg.get("invert_eye_x", False),
                    invert_head_x=_live2d_cfg.get("invert_head_x", False),
                    invert_head_y=_live2d_cfg.get("invert_head_y", False),
                    idle_enabled=self._idle_enabled,
                    body_enabled=_BODY_CONTROL,
                    wink_swap=_WINK_SWAP,
                    look_invert_x=_LOOK_INVERT_X,
                    max_simultaneous_fx=body_actions().max_simultaneous_fx,
                    behavior_director_enabled=self._behavior_director_user,
                    behavior_state=self.behavior_state,
                    motion_budget=MotionBudget(
                        max_head_angle=float(_MOTION_BUDGET_CFG.get("max_head_angle", 8.0)),
                        max_eye_offset=float(_MOTION_BUDGET_CFG.get("max_eye_offset", 0.5)),
                        max_body_drift=float(_MOTION_BUDGET_CFG.get("max_body_drift", 2.0)),
                    ),
                    liveliness=self._liveliness,
                    lean_param=_LEAN_PARAM,
                    lean_sign=_LEAN_SIGN,
                    breath_pause_enabled=_BREATH_PAUSE_ENABLED,
                    breath_pause_every=tuple(_BREATH_PAUSE_EVERY),
                    breath_pause_hold=tuple(_BREATH_PAUSE_HOLD),
                    lip_sync_enabled=_LIP_SYNC_ENABLED,
                    lip_sync_offset_ms=_LIP_SYNC_OFFSET_MS,
                )
                self.face._body_log_enabled = _BODY_LOG
                self.face._snapshot_publisher = self._publish_state_snapshot
                self.face.set_model_scale(self._model_scale)
                self.face.set_model_offset(self._model_offset_x, self._model_offset_y)
                fps = int(_cfg_face.get("fps", 30))
                self.face.set_fps(fps)
                self._use_live2d = True
                self._live2d_moods = set(self.face.known_moods())
            except Exception as e:
                logging.getLogger(__name__).error(
                    f"Live2D widget init failed, falling back to sprites: {e}"
                )
                print(f"[MARIN] Live2D model failed to load ({e}). "
                      "Using sprite mode. If you expected the 3D model, run "
                      "diagnose.bat and send diagnose_out.txt to the developer.")
                self.face = None

        if self.face is None:
            self.face = FaceLabel(self)

        self.mood        = "idle"
        self.mouth_open  = False
        self.speaking    = False
        self.bubble_text = ""
        self._pending_body = []        # staged tags awaiting 1c model application
        self._bubble_above = True  # flips when near screen top

        # ── Drag state ────────────────────────────────────────────────────────
        self._press_pos   = QPoint()
        self._origin      = QPoint()
        self._moved       = False
        # Alt/Shift drag for model panning
        self._pan_press_offset_x = 0.0
        self._pan_press_offset_y = 0.0
        self._pan_press_cursor   = QPoint()
        self._panning            = False

        # ── Timers ────────────────────────────────────────────────────────────
        self.hide_timer  = QTimer(self, singleShot=True, timeout=self.bubble.hide)
        self.idle_timer  = QTimer(self, singleShot=True, timeout=self._go_idle)
        self.tick        = QTimer(self, interval=140, timeout=self._on_tick)
        self.tick.start()

        # ── Apply initial layout (sizes and positions all widgets) ────────────
        self._apply_layout(initial=True)

        self.face.update_art("idle", False, crossfade=False)

    # ── Layout engine (THE single source of truth for all geometry) ──────────

    def _apply_layout(self, initial: bool = False) -> None:
        """Recompute all geometry from self.face_size and reposition every widget.

        Must be called from the GUI thread.  Keeps the face's bottom-center
        fixed on screen (when not initial) so the pet grows upward / outward.
        """
        s  = self.face_size
        w  = int(s * 1.2)
        bh = int(s * 0.7)
        ft = bh + 6          # FACE_TOP — y of the face inside the window
        # Total window height: bubble strip above + face + padding + bubble strip below
        win_h = ft + s + 4 + bh

        if not initial:
            # Anchor: bottom-center of face in screen coords
            old_center_x = self.x() + self.width()  // 2
            old_face_bot  = self.y() + self._face_top + self._face_size
        else:
            old_center_x = None
            old_face_bot  = None

        # Store layout values on self for other methods
        self._face_size = s
        self._win_width  = w
        self._bubble_h   = bh
        self._face_top   = ft
        self._win_height = win_h

        # Also update the module-level globals so FaceLabel helper fns still work
        global SIZE, WIDTH, BUBBLE_H, FACE_TOP
        SIZE     = s
        WIDTH    = w
        BUBBLE_H = bh
        FACE_TOP = ft

        # Window geometry
        if initial:
            self.setFixedSize(w, win_h)
        else:
            # Move so bottom-center stays fixed
            new_x = old_center_x - w // 2
            new_y = old_face_bot - ft - s
            self.setFixedSize(w, win_h)
            self.move(new_x, new_y)
            # Clamp to available screen area
            self._clamp_to_screen()

        # Face widget
        face_x = (w - s) // 2
        self.face.setGeometry(face_x, ft, s, s)

        # Bubble
        self.bubble.setFixedWidth(w - 20)
        self.bubble.setMaximumHeight(bh - 10)
        self.bubble.setStyleSheet(
            "background: rgba(38,31,77,235); color: #f2eaff; border-radius: 14px;"
            "padding: 10px; font-size: 15px;"
        )

        # Indicator / typing — pin them
        self._place_indicator()
        if self.bubble.isVisible():
            self._place_bubble()

    def _clamp_to_screen(self) -> None:
        screens = QGuiApplication.screens()
        if not screens:
            return
        # Use the screen that contains most of the window
        best = None
        best_area = -1
        wr = QRect(self.pos(), self.size())
        for sc in screens:
            inter = wr.intersected(sc.availableGeometry())
            area  = inter.width() * inter.height()
            if area > best_area:
                best_area = area
                best = sc
        if best is None:
            best = screens[0]
        ag = best.availableGeometry()
        x  = max(ag.left(), min(self.x(), ag.right()  - self.width()))
        y  = max(ag.top(),  min(self.y(), ag.bottom() - self.height()))
        if x != self.x() or y != self.y():
            self.move(x, y)

    # ── Position persistence ──────────────────────────────────────────────────

    def _load_idle_override(self, default: bool = True) -> bool:
        """Read the persisted idle-loop override from pet_state.json.

        This is a user override that survives restarts and is NOT overwritten
        by config/live2d.json."""
        try:
            s = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return bool(s.get("idle_enabled", default))
        except Exception:
            return default

    def set_idle_enabled(self, enabled: bool) -> None:
        """Toggle the Live2D idle loop and persist the override immediately."""
        self._idle_enabled = bool(enabled)
        if self._use_live2d and self.face is not None:
            self.face.set_idle_enabled(self._idle_enabled)
        self.save_position()

    def _load_liveliness_override(self, default: float = 0.7) -> float:
        """Read the persisted liveliness (0..1) from pet_state.json.  A user
        override that survives restarts; when absent falls back to the config."""
        try:
            s = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            v = float(s["liveliness"])
            return max(0.0, min(1.0, v))
        except Exception:
            return default

    def _load_behavior_director_override(self, default: bool = True) -> bool:
        """Read the persisted behavior-director on/off from pet_state.json.  A
        user override that survives restarts; when absent falls back to the config."""
        try:
            s = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return bool(s["behavior_director"])
        except Exception:
            return default

    def set_liveliness(self, value: float) -> None:
        """Set the liveliness scalar and persist it.  Also applies to the widget."""
        self._liveliness = max(0.0, min(1.0, float(value)))
        if self._use_live2d and self.face is not None:
            self.face.set_liveliness(self._liveliness)
        self.save_position()

    def set_behavior_director_enabled(self, enabled: bool) -> None:
        """Toggle the behavior director (vs legacy idle) and persist it."""
        self._behavior_director_user = bool(enabled)
        if self._use_live2d and self.face is not None:
            self.face.set_behavior_director(self._behavior_director_user)
        self.save_position()

    def restore_position(self, screen):
        # Validate saved position against all screens before using it
        try:
            s = json.loads(STATE_FILE.read_text())
            x, y = int(s["x"]), int(s["y"])
            pet_rect = QRect(x, y + self._face_top, self._win_width, self._face_size)
            screens = QGuiApplication.screens()
            on_screen = any(pet_rect.intersects(sc.availableGeometry()) for sc in screens)
            if on_screen:
                self.move(x, y)
                # Restore framing values if present
                if "model_scale" in s:
                    self._model_scale    = float(s["model_scale"])
                    self._model_offset_x = float(s.get("model_offset_x", 0.0))
                    self._model_offset_y = float(s.get("model_offset_y", 0.35))
                    if self._use_live2d:
                        self.face.set_model_scale(self._model_scale)
                        self.face.set_model_offset(self._model_offset_x, self._model_offset_y)
            # Phase 2e: always apply persisted liveliness + director toggle, even
            # when the saved window position is off-screen (same fix as idle).
            if self._use_live2d and self.face is not None:
                self.face.set_liveliness(self._liveliness)
                self.face.set_behavior_director(self._behavior_director_user)
            return
        except Exception:
            pass
        g = screen.availableGeometry()
        self.move(g.right() - self._win_width - 20,
                  g.bottom() - (self._face_top + self._face_size) - 20)

    def save_position(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {
            "x": self.x(), "y": self.y(),
            "face_size": self.face_size,
            "idle_enabled": self._idle_enabled,
        }
        if self._use_live2d:
            data["model_scale"]    = self._model_scale
            data["model_offset_x"] = self._model_offset_x
            data["model_offset_y"] = self._model_offset_y
            data["liveliness"]          = round(self._liveliness, 2)
            data["behavior_director"]   = self._behavior_director_user
        STATE_FILE.write_text(json.dumps(data))

    def _save_state_debounced(self) -> None:
        """Debounce save — called after every resize/pan; persists after 500 ms idle."""
        self._save_state_timer.start(500)

    def _save_state_now(self) -> None:
        self.save_position()

    def _place_indicator(self):
        """Pin the status pill to the bottom-right corner of the face."""
        face_right  = (self._win_width - self._face_size) // 2 + self._face_size
        face_bottom = self._face_top + self._face_size
        iw = self.status_indicator.sizeHint().width() or 36
        ih = self.status_indicator.sizeHint().height() or 26
        self.status_indicator.setGeometry(face_right - iw - 6, face_bottom - ih - 6, iw, ih)

    # ── Bubble placement ──────────────────────────────────────────────────────

    def _place_bubble(self):
        """Flip bubble above/below pet depending on screen position."""
        screen_top = QGuiApplication.primaryScreen().availableGeometry().top()
        near_top = (self.y() - screen_top) < self._bubble_h + 20

        bh = max(40, self.bubble.sizeHint().height())
        self.bubble.setFixedHeight(bh)

        if near_top:
            # Place bubble BELOW the face — inside the extra transparent strip
            self.bubble.move(10, self._face_top + self._face_size + 4)
        else:
            # Place bubble ABOVE the face (original behaviour)
            self.bubble.move(10, self._face_top - bh)

        self.typing_label.move(10, self.bubble.y())

    def show_bubble(self, text: str):
        self.bubble.setText(text)
        visible = bool(text)
        self.bubble.setVisible(visible)
        if visible:
            self.bubble.adjustSize()
            self._place_bubble()

    # ── Slot handlers (called from GUI thread via signals) ────────────────────

    def on_start(self):
        self.hide_timer.stop()
        self.bubble_text = ""
        self.show_bubble("")
        self.status_indicator.set_state("idle")  # done listening

    def on_text(self, s: str):
        self.typing_label.hide_indicator()   # item 12: dots gone on first token
        self.bubble_text += s
        self.show_bubble(self.bubble_text)

    def on_mood(self, m: str):
        known = (
            m in MOOD_COLORS
            or (ART_DIR / f"{m}.png").exists()
            or (ART_DIR / f"{m}.gif").exists()
            or (self._use_live2d and m in self._live2d_moods)
        )
        self.mood = m if known else "idle"
        self.face.update_art(self.mood, self.mouth_open, crossfade=True)

    def on_body(self, payload):
        """Sentence audio started: reveal its bubble text and apply body tags.

        Runs in the GUI thread (queued connection from the Speaker playback
        thread).  payload = (gen, text, tags) or, with Phase 4a lip sync, also
        (envelope, frame_ms).  The bubble text is revealed here (at audio start)
        rather than at generation time, the tags are applied to the Live2D model
        (look/gesture/wink/fx/clear), and the lip-sync envelope is installed so
        the mouth tracks the actual speech."""
        if len(payload) >= 6:
            gen, text, tags, envelope, frame_ms, t0 = payload
        else:
            gen, text, tags = payload
            envelope, frame_ms, t0 = None, None, None
        # 2c: each spoken sentence's audio start triggers a small speaking nod
        # (scaled by arousal + liveliness inside the widget).  Fires in the GUI
        # thread at playback start, including for the plain (no-tag) path.
        if self._use_live2d and hasattr(self.face, "note_speech_start"):
            self.face.note_speech_start(text or "")
        # Phase 4a: install the decoded loudness envelope so _tick_mouth tracks
        # speech instead of the random flap.  Only when live2d + lip sync is on.
        if (self._use_live2d and envelope
                and hasattr(self.face, "set_mouth_envelope")):
            self.face.set_mouth_envelope(envelope, frame_ms)
        elif (self._use_live2d and hasattr(self.face, "clear_mouth_envelope")):
            # No envelope for this clip (decode failed/idle) — drop any leftover
            # so the previous clip's timing never reapplies.
            self.face.clear_mouth_envelope()
        if _BODY_CONTROL:
            # Bubble text reveals when the sentence's audio starts.
            if text:
                self.typing_label.hide_indicator()
                self.bubble_text += text
                self.show_bubble(self.bubble_text)
        if tags:
            # Phase 4b: record tag-to-audio-start latency + per-kind action counts.
            if t0 is not None:
                _BODY_STATS.record_latency(time.monotonic() - t0)
            kinds: Dict[str, int] = {}
            for t in tags:
                kinds[t.kind] = kinds.get(t.kind, 0) + 1
            _BODY_STATS.record_reply(kinds)
            if _BODY_LOG:
                label = ", ".join(f"{t.kind}:{t.name or ''}" for t in tags)
                logger.info(f"[body] applying: {label}")
                if t0 is not None:
                    logger.info("[body] tag->audio latency: %.0f ms",
                                (time.monotonic() - t0) * 1000.0)
            # Phase 2b: mood tags are impulses into the behavior state (fire at
            # audio start, not at generation time, per the plan).  The mood
            # expression itself is applied separately via on_mood/update_art.
            if self.behavior_state is not None:
                for t in tags:
                    if t.kind == "mood":
                        self.behavior_state.mood_impulse(t.name)
            if self._use_live2d and hasattr(self.face, "apply_body_actions"):
                self.face.apply_body_actions(_resolve_body_actions(tags))
            self._pending_body = list(tags)

    def on_listening(self, on: bool):
        """Phase 2b: user speaking flag from recording state (bus queued to the
        GUI thread).  Drives the behavior state's 'listening' (attentive gaze)."""
        if self.behavior_state is not None:
            self.behavior_state.set_listening(on)

    def on_end(self):
        self.hide_timer.start(max(5000, len(self.bubble_text) * 70))

    def on_status(self, msg: str, keep: bool):
        self.on_start()
        self.on_text(msg)
        if not keep:
            self.hide_timer.start(3000)
        # Show mic indicator while waiting for input
        if "listen" in msg.lower():
            self.status_indicator.set_state("listening")
            self._place_indicator()
        else:
            self.status_indicator.set_state("idle")

    def on_typing(self, active: bool):
        if active:
            self.typing_label.show_indicator()
            self._place_bubble()
        else:
            self.typing_label.hide_indicator()

    def on_proactive(self, msg: str):
        """Show proactive message in bubble (already spoken by speaker)."""
        self.on_start()
        self.on_text(msg)
        self.on_end()

    # ── Animation tick ────────────────────────────────────────────────────────

    @staticmethod
    def _audio_busy() -> bool:
        """Part 3-C: delegate to Speaker.is_playing() which covers the gap between
        sentences (synth queue non-empty but pygame not yet busy).
        Falls back to False if the companion ref isn't set yet or mixer is uninit.
        """
        try:
            if _companion_ref is not None:
                return _companion_ref.speaker.is_playing()
            return bool(pygame.mixer.music.get_busy())
        except Exception:
            return False

    def _on_tick(self):
        if self._audio_busy():
            self.speaking = True
            self.idle_timer.stop()
            if self._use_live2d:
                # Live2D drives its own smooth mouth; just signal speaking state
                if not self.mouth_open:
                    self.mouth_open = True
                    self.face.update_art(self.mood, True, crossfade=False)
            else:
                # Sprite mode: toggle mouth art at 140 ms for lip-flap effect
                self.mouth_open = not self.mouth_open
                self.face.update_art(self.mood, self.mouth_open, crossfade=False)
            if self.status_indicator._state != "talking":
                self.status_indicator.set_state("talking")
                self._place_indicator()
        else:
            if self.mouth_open:
                self.mouth_open = False
                self.face.update_art(self.mood, False, crossfade=False)
            if self.speaking:
                self.speaking = False
                self.idle_timer.start(4000)
                self.status_indicator.set_state("idle")

    def _go_idle(self):
        self.mood = "idle"
        self.face.update_art("idle", False, crossfade=True)

    # ── Mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press_pos = e.globalPosition().toPoint()
            self._origin    = self.pos()
            self._moved     = False
            # Phase 2b: any pointer contact with the pet is an interaction.
            if self.behavior_state is not None:
                self.behavior_state.interact()
            # Check for Alt/Shift pan modifier
            mods = e.modifiers()
            if (mods & Qt.AltModifier) or (mods & Qt.ShiftModifier):
                self._panning = True
                self._pan_press_cursor   = e.globalPosition().toPoint()
                self._pan_press_offset_x = self._model_offset_x
                self._pan_press_offset_y = self._model_offset_y
            else:
                self._panning = False
        elif e.button() == Qt.RightButton:
            # Right-click on pet body — show quick menu
            self._show_context_menu(e.globalPosition().toPoint())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            if self._panning and self._use_live2d:
                # Alt/Shift drag → pan the model inside the square
                delta = e.globalPosition().toPoint() - self._pan_press_cursor
                # 1 pixel ≈ 0.005 Live2D units at scale 2.2 and size 400;
                # calibrate: full face width (~400px) ≈ 2 Live2D units → 0.005 per px
                px_per_unit = self._face_size / (2.0 * self._model_scale)
                px_per_unit = max(1.0, px_per_unit)
                new_x = self._pan_press_offset_x + delta.x() / px_per_unit
                new_y = self._pan_press_offset_y - delta.y() / px_per_unit   # Y is inverted
                new_x = max(-3.0, min(3.0, new_x))
                new_y = max(-3.0, min(3.0, new_y))
                self._model_offset_x = new_x
                self._model_offset_y = new_y
                self.face.set_model_offset(new_x, new_y)
                self._save_state_debounced()
                self._moved = True
            else:
                delta = e.globalPosition().toPoint() - self._press_pos
                self._moved = self._moved or delta.manhattanLength() > 5
                self.move(self._origin + delta)
            # Feed cursor to Live2D eye tracking
            if self._use_live2d and not self._panning:
                local = e.position()
                self.face.feed_cursor(local.x() - self.face.x(),
                                      local.y() - self.face.y())

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        self._panning = False
        if self._moved:
            # Persist position immediately so a hard kill doesn't lose it
            self.save_position()
            return
        # Only trigger voice turn when clicking the face sprite, not the bubble area
        local_y = e.position().toPoint().y()
        local_x = e.position().toPoint().x()
        face_top = self._face_top
        face_bot = self._face_top + self._face_size
        if face_top <= local_y <= face_bot:
            # Live2D hit test (checks transparent areas)
            if self._use_live2d:
                lx = local_x - self.face.x()
                ly = local_y - self.face.y()
                if self.face.hit_test(lx, ly):
                    self.on_click()
            else:
                self.on_click()

    def wheelEvent(self, e: QWheelEvent) -> None:
        """Ctrl+wheel → resize face.  Alt+wheel → zoom model."""
        delta_y = e.angleDelta().y()
        if delta_y == 0:
            return
        mods = e.modifiers()
        step = 1 if delta_y > 0 else -1

        if mods & Qt.ControlModifier:
            # Resize the face square by ~5% per notch
            new_size = int(self.face_size * (1.0 + step * 0.05))
            new_size = max(self.face_size_min, min(self.face_size_max, new_size))
            if new_size != self.face_size:
                self.face_size = new_size
                self._apply_layout()
                self._save_state_debounced()
            e.accept()

        elif mods & Qt.AltModifier:
            # Zoom model inside the square
            if self._use_live2d:
                new_scale = self._model_scale * (1.0 + step * 0.05)
                new_scale = max(0.5, min(10.0, new_scale))
                self._model_scale = new_scale
                self.face.set_model_scale(new_scale)
                self._save_state_debounced()
            e.accept()

        else:
            e.ignore()

    # ── Context menu (right-click) ────────────────────────────────────────────

    def _show_context_menu(self, global_pos: QPoint):
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background: #1e1740; color: #f2eaff; border: 1px solid #4a3f8a; "
            "border-radius: 8px; padding: 4px; font-size: 14px; }"
            "QMenu::item:selected { background: #3a2f7a; }"
        )
        menu.addAction("🎤 Talk", self.on_click)
        menu.addSeparator()

        # ── Scene1 gag animation ──────────────────────────────────────────────
        if self._use_live2d and "scene1" in self.face._motions:
            loops = int(_live2d_cfg.get("scene1_loops", 1))
            if self.face.motion_is_playing():
                menu.addAction("⏹  Stop animation", self.face.stop_motion)
            else:
                menu.addAction("✨ Play Scene1", lambda: self.face.play_motion("scene1", loops))
            menu.addSeparator()

        # ── Test body action (fire any enabled fx/look/gesture/wink directly) ──
        if (self._use_live2d and _BODY_CONTROL
                and hasattr(self.face, "apply_body_actions")):
            test_menu = menu.addMenu("🎭 Test body action")
            ba = body_actions()
            for key, cfg in ba.enabled_capped_actions().items():
                act = test_menu.addAction(self._body_action_label(key, cfg))
                act.triggered.connect(
                    lambda _, k=key, c=cfg: self._fire_body_action(k, c))
            if not test_menu.actions():
                test_menu.addAction("(no enabled actions)")
                test_menu.setEnabled(False)
            menu.addSeparator()

        # ── Face size submenu ─────────────────────────────────────────────────
        size_menu = menu.addMenu("📐 Face size")
        for label, val in [("Small (250)", 250), ("Medium (400)", 400),
                            ("Large (600)", 600), ("Huge (800)", 800)]:
            act = size_menu.addAction(label)
            act.triggered.connect(lambda _, v=val: self._set_face_size(v))
        size_menu.addSeparator()
        if self._use_live2d:
            size_menu.addAction("↩  Reset framing", self._reset_framing)

        vol_menu = menu.addMenu("🔊 Volume")
        for label, val in [("25%", 0.25), ("50%", 0.5), ("75%", 0.75), ("100%", 1.0)]:
            act = vol_menu.addAction(label)
            act.triggered.connect(lambda _, v=val: _companion_ref.speaker.set_volume(v))

        spd_menu = menu.addMenu("⏩ Speed")
        for label, val in [("Slow (0.75×)", 0.75), ("Normal (1.0×)", 1.0), ("Fast (1.5×)", 1.5)]:
            act = spd_menu.addAction(label)
            act.triggered.connect(lambda _, v=val: _companion_ref.speaker.set_speed(v))

        # ── Phase 2e: behavior director on/off + liveliness presets ────────────
        if self._use_live2d:
            bd_act = menu.addAction("Behavior director: on/off")
            bd_act.setCheckable(True)
            bd_act.setChecked(self._behavior_director_user)
            bd_act.toggled.connect(self.set_behavior_director_enabled)

            liv_menu = menu.addMenu(f"⏲  Liveliness  ({self._liveliness:.2f})")
            presets = [("Still (0.00)", 0.0), ("Calm (0.40)", 0.4),
                       ("Normal (0.70)", 0.7), ("Lively (1.00)", 1.0)]
            group = QActionGroup(self)
            group.setExclusive(True)
            for label, val in presets:
                act = liv_menu.addAction(label)
                act.setCheckable(True)
                act.setChecked(abs(self._liveliness - val) < 0.001)
                act.triggered.connect(lambda _, v=val: self.set_liveliness(v))
                group.addAction(act)

        menu.addSeparator()
        menu.addAction("🔄 Reload personality", self._do_reload)
        menu.addAction("📊 Stats", self._do_stats)
        menu.addSeparator()
        menu.addAction("❌ Quit", QApplication.quit)

        menu.exec(global_pos)

    @staticmethod
    def _body_action_label(key: str, cfg: dict) -> str:
        """Human label for a body action menu entry (e.g. "fx:blush" → "Blush")."""
        kind, _, name = key.partition(":")
        icons = {"fx": "💢", "look": "👀", "gesture": "🤲", "wink": "😉"}
        icon = icons.get(kind, "•")
        pretty = name.replace("_", " ").capitalize()
        if kind == "wink" and name not in ("left", "right"):
            side = cfg.get("side", "")
            pretty = f"Wink ({side})" if side else "Wink"
        return f"{icon} {pretty}"

    def _fire_body_action(self, key: str, cfg: dict) -> None:
        """Fire a single body action directly (test menu) — no LLM needed.

        Builds a one-shot action list from ``cfg`` (already shaped by
        enabled_capped_actions) and pushes it through the same path the tags use.
        """
        if not (self._use_live2d and getattr(self.face, "_body_enabled", True)):
            return
        tags = []
        if cfg.get("kind") in ("fx", "look", "gesture"):
            from body_tags import Tag
            tags = [Tag(kind=cfg["kind"], name=cfg["name"], raw=f"[{key}]")]
        elif cfg.get("kind") == "wink":
            from body_tags import Tag
            side = cfg.get("side") or body_actions().wink_default_side()
            tags = [Tag(kind="wink", name=side, raw=f"[wink:{side}]")]
        if tags:
            if _BODY_LOG:
                logger.info("[body] test-menu firing: %s -> %s", key, tags)
            # on_body expects a (gen, text, tags) payload; text is empty so the
            # bubble is untouched and only the tags reach the model.
            self.on_body((None, "", tags))

    def _set_face_size(self, new_size: int) -> None:
        new_size = max(self.face_size_min, min(self.face_size_max, new_size))
        if new_size != self.face_size:
            self.face_size = new_size
            self._apply_layout()
            self._save_state_debounced()

    def _reset_framing(self) -> None:
        """Reset model scale and offset to config defaults."""
        cfg_face = _live2d_cfg.get("face", {})
        self._model_scale    = float(cfg_face.get("scale",    2.2))
        self._model_offset_x = float(cfg_face.get("offset_x", 0.0))
        self._model_offset_y = float(cfg_face.get("offset_y", 0.35))
        if self._use_live2d:
            self.face.set_model_scale(self._model_scale)
            self.face.set_model_offset(self._model_offset_x, self._model_offset_y)
        self._save_state_debounced()

    def _do_reload(self):
        try:
            _companion_ref.config.reload()
            _companion_ref.apply_config()   # Part 3-B: push new values into model+speaker
            self.on_status("Personality reloaded!", False)
        except Exception as e:
            self.on_status(f"Reload failed: {e}", False)

    def _do_stats(self):
        c = _companion_ref
        score = c.memory.closeness_score()
        tier = c.memory.tier_name()
        tokens = c.model.get_token_stats()
        lines = [
            f"Closeness: {score} pts  ({tier})",
            f"Facts: {len(c.memory.get_facts())}",
            f"History: {len(c.history)} messages",
            f"Tokens: {tokens['prompt']} prompt / {tokens['completion']} completion",
            f"Backend: {'Cerebras' if c.config.use_cerebras else 'Ollama'} / {c.model.model_name}",
        ]
        # Phase 4b: body-control statistics summary.
        if _BODY_CONTROL and _FACE_RENDERER == "live2d":
            s = _BODY_STATS.summary()
            lat = s["tag_to_audio_latency_ms"]
            body_lines = [
                "",
                f"Body tags: {s['replies_with_tags']} replies / {s['total_tags']} tags",
                f"  by kind: {s['tags_by_kind']}",
                f"  unknown dropped: {s['unknown_dropped']}  cap dropped: {s['cap_dropped']}",
                f"  tag->audio latency(ms): last={lat['last']:.0f} avg={lat['avg']:.0f} "
                f"min={lat['min']:.0f} max={lat['max']:.0f} (n={lat['samples']})",
            ]
            lines.extend(body_lines)
        QMessageBox.information(self, "Stats", "\n".join(lines))


# ── Proactive idle timer ──────────────────────────────────────────────────────

class ProactiveTimer:
    """Fires a random proactive message after N minutes of user silence."""

    def __init__(self, companion: PetCompanion, interval_minutes: int):
        self.companion = companion
        self._interval_ms = interval_minutes * 60 * 1000
        self._timer = QTimer(singleShot=True)
        self._timer.timeout.connect(self._fire)

    def reset(self):
        """Call this after each user turn to restart the idle countdown."""
        self._timer.stop()
        if self.companion.config.proactive_enabled:
            self._timer.start(self._interval_ms)

    def stop(self):
        self._timer.stop()

    def _is_ignored_long(self) -> bool:
        """True when she has been ignored long enough to widen the gap."""
        c = self.companion
        if c._state_getter is None:
            return False
        try:
            snap = c._state_getter()
        except Exception:
            return False
        if snap is None:
            return False
        s = getattr(snap, "seconds_since_interaction", None)
        return s is not None and s >= 300.0

    def _fire(self):
        if not self.companion.config.proactive_pools and not self.companion.config.proactive_generate:
            self.reset()
            return
        # item 16: don't fire while a voice turn is active — reschedule instead
        if self.companion.turn_lock.locked():
            self.reset()
            return
        # Phase 3d: scheduling limits (min interval, daily cap, quiet hours,
        # do-not-disturb, and ignored-long widens the gap so she speaks LESS
        # the longer she's alone).  When not allowed, just re-arm the countdown.
        now = datetime.datetime.now()
        if not self.companion._proactive_scheduler.allowed(
                now, ignored_long=self._is_ignored_long()):
            self.reset()
            return
        msg = self.companion.generate_proactive()
        if not msg:  # suppressed (ignored long) or nothing picked
            self.reset()
            return
        # Run on a thread so it doesn't block the GUI
        threading.Thread(
            target=self.companion.say_proactive,
            args=(msg,),
            daemon=True,
        ).start()
        self.companion._proactive_scheduler.record(now)
        self.reset()  # schedule next proactive after another interval


# ── Module-level companion ref (needed by right-click menu lambdas) ───────────
_companion_ref: PetCompanion = None  # set in main()


# ── Wake-word background thread (Part 3-D) ───────────────────────────────────

class WakeWordThread:
    """Runs the wake-word loop in the background; calls companion.start_turn()
    when triggered.  Pauses while a turn is active or speaker is playing so
    her own voice doesn't re-trigger the detector.
    """

    def __init__(self, companion: PetCompanion):
        self.companion = companion
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="WakeWord")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        import time as _t
        det = self.companion.wake_detector
        if det is None:
            return
        logger.info("Wake-word thread started")
        while not self._stop.is_set():
            # Pause while a turn is running or TTS is playing
            if (self.companion.turn_lock.locked()
                    or self.companion.speaker.is_playing()):
                _t.sleep(0.2)
                continue
            det.wait_for_wake()
            if not self._stop.is_set():
                logger.info("Wake word triggered — starting voice turn")
                self.companion.start_turn()
                _t.sleep(1.0)  # brief cooldown before listening again


# ── Main ──────────────────────────────────────────────────────────────────────

# Human-friendly key prompts for first-run setup (friend build).  Keys are
# written to ROOT/.env and are never logged.
_KEY_FIELDS = [
    ("FISH_API_KEY", "Fish Audio (the voice): get a key at https://fish.audio"),
    ("CEREBRAS_API_KEY", "Cerebras (the brain): get a key at https://cloud.cerebras.ai"),
]


def _load_env() -> dict:
    """Load ROOT/.env into os.environ and return the current key values."""
    try:
        from dotenv import load_dotenv, set_key
        load_dotenv(ROOT / ".env")
        load_dotenv()
        return {name: os.getenv(name, "").strip() for name, _ in _KEY_FIELDS}
    except Exception:
        return {name: os.getenv(name, "").strip() for name, _ in _KEY_FIELDS}


def _ensure_env_keys() -> None:
    """First-run setup: if a key is missing, ask on the console and write .env.

    Never prints a key value.  Only prompts when at least one key is missing or
    still a placeholder, so repeat launches skip straight to startup.
    """
    try:
        from dotenv import set_key
    except Exception:
        return  # dotenv missing — let the SDKs raise naturally

    values = _load_env()
    missing = [
        (name, msg) for name, msg in _KEY_FIELDS
        if not values.get(name) or "your_" in values.get(name, "") or "here" in values.get(name, "")
    ]
    if not missing:
        return

    print("\n=== First-time setup ===")
    print("You need two free API keys for Marin to work. Enter them now")
    print("(they are saved to .env in this folder and never printed back).\n")
    for name, msg in missing:
        print(f"  {msg}")
        try:
            val = input(f"  {name} = ").strip()
        except EOFError:
            val = ""
        if val:
            set_key(str(ROOT / ".env"), name, val)
            print("  saved.")
        else:
            print("  skipped (she'll warn you if it's still missing).")
    print()


def main():
    global ART_DIR, _companion_ref

    # First-run: prompt for missing API keys / write .env (before anything loads
    # the SDKs, which read these from the environment).
    _ensure_env_keys()

    # MUST be called before QApplication to get alpha buffer on the GL surface
    configure_opengl_format()

    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)

    print("Loading Whisper, voice, and model (this may take a few seconds)…")
    config = Config(config_dir=Path(args.config_dir) if args.config_dir else None)
    bus = Bus()
    companion = PetCompanion(config, bus)
    _companion_ref = companion

    # ── Restore persisted audio state (volume / mute) ─────────────────────────
    _restore_muted = False
    try:
        _saved = json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {}
        if "volume" in _saved:
            companion.speaker.set_volume(float(_saved["volume"]))
        _restore_muted = bool(_saved.get("muted"))
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not restore audio state: {e}")

    name = re.sub(r"\W+", "_", config.persona["name"].lower())
    # item 6: anchor to ROOT, not CWD
    ART_DIR = Path(args.art) if args.art else ROOT / "characters" / name

    pet = Pet(on_click=companion.start_turn)

    # Phase 3b: give the companion read-only access to Pet's immutable per-tick
    # behavior snapshot (never the live BehaviorState).  Ordered after Pet is
    # built so pet._state_snapshot exists; reads happen on the worker thread.
    companion._state_getter = lambda: getattr(pet, "_state_snapshot", None)

    # Part 3-D: start wake-word thread if enabled
    wake_thread = None
    if companion.wake_detector is not None:
        wake_thread = WakeWordThread(companion)
        wake_thread.start()

    # Wire all signals → GUI slots
    bus.start.connect(pet.on_start)
    bus.text.connect(pet.on_text)
    bus.mood.connect(pet.on_mood)
    bus.end.connect(pet.on_end)
    bus.status.connect(pet.on_status)
    bus.typing.connect(pet.on_typing)
    bus.proact.connect(pet.on_proactive)
    bus.listening.connect(pet.on_listening)

    # Body control: each sentence's clean text + inline tags travel with it in
    # the Speaker queues and are emitted when that sentence's audio actually
    # STARTS (playback thread) → bus → GUI slot.  The bubble text is revealed at
    # audio start, and the tags drive the Live2D model.  The gen check in Speaker
    # discards interrupted generations.
    companion.speaker.on_sentence_start = (
        lambda gen, text, mood, tags, envelope=None, frame_ms=None, t0=None:
            bus.body.emit((gen, text, tags, envelope, frame_ms, t0))
    )
    bus.body.connect(pet.on_body)

    pet.restore_position(app.primaryScreen())
    pet.show()

    # ── Proactive idle timer ──────────────────────────────────────────────────
    proactive = ProactiveTimer(companion, config.proactive_idle_minutes)

    # Reset proactive timer after each completed turn
    def _on_turn_end():
        proactive.reset()

    bus.end.connect(_on_turn_end)
    proactive.reset()  # start counting from launch

    # ── System tray ──────────────────────────────────────────────────────────
    tray = QSystemTrayIcon(QIcon(load_pixmap("idle", False, pet.face_size)), app)

    # Show tray balloon for proactive messages
    def _tray_proact(msg: str):
        tray.showMessage(
            config.persona["name"],
            msg,
            QSystemTrayIcon.MessageIcon.Information,
            4000,
        )

    bus.proact.connect(_tray_proact)

    tray_menu = QMenu()
    tray_menu.addAction("Show / Hide", lambda: pet.setVisible(not pet.isVisible()))
    mute_act = QAction("Mute voice", checkable=True)
    mute_act.toggled.connect(lambda on: setattr(companion.speaker, "enabled", not on))
    mute_act.setChecked(_restore_muted)
    if _restore_muted:
        companion.speaker.enabled = False
    tray_menu.addAction(mute_act)
    tray_menu.addSeparator()
    tray_menu.addAction("Quit", app.quit)
    tray.setContextMenu(tray_menu)
    tray.show()
    app._keep = (tray, tray_menu)

    # ── Hotkey ────────────────────────────────────────────────────────────────
    def _start_turn_hotkey():
        # Phase 2b: F9 is an attention jolt into the behavior state.  Runs on
        # the keyboard thread; behavior_state is pure (no Qt) so this is safe.
        if pet.behavior_state is not None:
            pet.behavior_state.hotkey()
        companion.start_turn()

    try:
        keyboard.add_hotkey(args.hotkey, _start_turn_hotkey)
    except Exception as e:
        pet.on_status(f"Hotkey '{args.hotkey}' unavailable: {e}", False)

    # Scene1 animation hotkey (configurable, default ctrl+shift+s)
    _scene1_hk = _live2d_cfg.get("scene1_hotkey", "")
    _scene1_loops = int(_live2d_cfg.get("scene1_loops", 1))
    if _scene1_hk and pet._use_live2d:
        try:
            def _play_scene1():
                if pet.face.motion_is_playing():
                    pet.face.stop_motion()
                else:
                    pet.face.play_motion("scene1", _scene1_loops)
            keyboard.add_hotkey(_scene1_hk, _play_scene1)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Scene1 hotkey '{_scene1_hk}' unavailable: {e}")

    # ── Shutdown ──────────────────────────────────────────────────────────────
    def on_quit():
        proactive.stop()
        if wake_thread:
            wake_thread.stop()
        pet.save_position()
        # Persist volume / mute alongside pet geometry
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {}
            state["volume"] = round(companion.speaker.volume, 2)
            state["muted"] = not companion.speaker.enabled
            STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
        except Exception:
            pass
        keyboard.unhook_all()
        companion.speaker.stop()
        # Dispose Live2D before the GL context is destroyed
        if pet._use_live2d:
            try:
                pet.face.dispose()
            except Exception:
                pass
        companion.shutdown()

    app.aboutToQuit.connect(on_quit)

    # ── Greeting ─────────────────────────────────────────────────────────────
    greeting = config.pick_greeting()
    if greeting:
        def _greet():
            pet.on_start()
            pet.on_text(greeting)
            pet.on_end()
            companion.speaker.say(greeting)
            companion.memory.log_turn("assistant", greeting)
            proactive.reset()

        QTimer.singleShot(500, _greet)

    print(
        f"{config.persona['name']} is ready.  "
        f"Hotkey: {args.hotkey}  |  left-click face  |  right-click for menu  |  tray to quit."
    )
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
