
"""body_tags — shared parsing for the AI body-control tags.

Single source of truth is ``config/body_actions.json``.  This module:

* resolves mood names (bare ``[joyful]``, ``[mood:joyful]`` and alias
  ``[mood:happy]``) through ONE code path,
* extracts inline tags from text (including text that arrives split across
  streamed chunks and/or has tags mid-sentence),
* honours the per-reply cap (fx/look/gesture/wink only — mood tags are always
  honoured) and the per-action ``enabled`` flag,
* strips tags from the text that is sent to TTS / the bubble / history /
  memory (the AI's raw words must never leak ``[mood:...]`` to the speaker),
* provides a last-resort filter that removes any leftover short bracketed
  ``[...:...]`` and ``[clear]`` so a missed tag is never spoken aloud.

``core.model.parse_mood_tag`` delegates its resolution here so the legacy
leading-tag path and the new parser agree identically.
"""



from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple

_here = Path(__file__).resolve().parent
_DEFAULT_ACTIONS_PATH = _here.parent / "config" / "body_actions.json"
# Default Live2D model dir used to resolve "expression" exp3 references.
_DEFAULT_MODEL_DIR = _here.parent / "characters" / "marin" / "live2d"

# Sentence terminator used by the app (kept in sync with core.model.SENTENCE_END).
SENTENCE_END: Pattern[str] = re.compile(r"(?<=[.!?…])\s+")

# Match any bracketed token (a candidate tag before we inspect its content).
_TAG_RE: Pattern[str] = re.compile(r"\[([^\[\]]*)\]")

# Last-resort leftover filter: a SHORT LOWERCASE word:word bracket is a missed
# tag (e.g. "[fx:blush]").  Deliberately excludes capitals and spaces so genuine
# prose such as "[Note: something]", "[1]", "list[0]" is never stripped.
_LEFTOVER_TAG_RE: Pattern[str] = re.compile(r"\[[a-z]+:[a-z0-9_]+\]")
# An UNCLOSED tag-prefix at the very end of the text (e.g. "[fx:bl" when a
# stream ends mid-tag) must be dropped too so it is never spoken.
_LEFTOVER_UNCLOSED_RE: Pattern[str] = re.compile(r"\s*\[[a-z]+:[a-z0-9_]*\s*$")

# Tag kinds that count toward the per-reply honour cap.
_CAPPED_KINDS = {"fx", "look", "gesture", "wink"}
# All recognised tag kinds (mood is special: never capped).
_KINDS = {"mood", "fx", "look", "gesture", "wink"}

# Defaults when config/body_actions.json is missing (e.g. headless tests).
_DEFAULT_ACTIONS: Dict = {
    "mood_aliases": {
        "happy": "joyful", "shy": "embarrassed", "mad": "angry", "glad": "joyful",
    },
    "moods": {m: {"enabled": True} for m in (
        "idle", "joyful", "excited", "embarrassed", "sad", "angry",
        "surprised", "curious", "relaxed", "love", "aroused")},
    "fx": {
        "blush":      {"enabled": True, "duration": 4.0, "params": {"ParamSwitch2": 1.0}},
        "tears":      {"enabled": True, "duration": 4.0, "params": {"ParamSwitch9": 1.0}},
        "face_dark":  {"enabled": True, "duration": 2.5, "params": {"ParamSwitch1": 1.0}},
        "blood":      {"enabled": True, "duration": 4.0, "params": {"ParamSwitch3": 1.0}},
        # ParamSwitch10/11 (hands) stay DISABLED until the user describes them.
        "hand_right": {"enabled": False, "duration": 2.5, "params": {"ParamSwitch10": 1.0}},
        "hand_left":  {"enabled": False, "duration": 2.5, "params": {"ParamSwitch11": 1.0}},
        # Phase 3c: expression-driven fx (expanded ONCE from .exp3.json at load,
        # blended+clamped at apply time, never via SetExpression).
        "star_eyes":  {"enabled": True, "duration": 4.0, "expression": "星星眼"},
        "heart_eyes": {"enabled": True, "duration": 4.0, "expression": "爱心眼"},
        "white_eyes": {"enabled": True, "duration": 3.0, "expression": "白眼"},
    },
    "look": {d: {"enabled": True, "duration": 0.8, "params": {}} for d in ("left", "right", "up", "down")},
    "gestures": {
        "nod": {"enabled": True, "duration": 0.8, "params": {}},
        "shake": {"enabled": True, "duration": 0.8, "params": {}},
        "tilt": {"enabled": True, "duration": 0.8, "params": {}},
        "wink": {"enabled": True, "duration": 0.5,
                  "sides": ["left", "right"], "default_side": "right", "params": {}},
    },
    "limits": {"max_tags_per_reply": 4, "max_simultaneous_fx": 3},
}


# Phase 3c: exp3 expression expansion
# An action may carry an optional "expression" key naming an .exp3.json file
# (by id, e.g. "星星眼", or a filename with/without .exp3.json).  These files are
# expanded ONCE into plain parameter ops at config load (not per apply).  The
# apply path afterwards blends each op against the model's real default and
# clamps to its real range.  SetExpression is never called.

def _resolve_exp3_paths(expression: str, model_dir: Optional[Path] = None) -> List[Path]:
    base = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
    candidates: List[Path] = []
    p = Path(expression)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(base / expression)
        candidates.append(base / f"{expression}.exp3.json")
        # if they passed just an id or a name with trailing dots, try the stem
        candidates.append(base / f"{Path(expression).stem}.json")
    return candidates


def _load_exp3_ops(expression: str, model_dir: Optional[Path] = None) -> List[Dict]:
    """Read an exp3 file once and return its plain param ops.

    Returns a list of ``{"id", "value", "blend"}``.  Any failure (missing file,
    bad shape) yields an empty list, so a typo'd expression is a silent no-op
    rather than a crash at load.
    """
    if not expression:
        return []
    for cand in _resolve_exp3_paths(expression, model_dir):
        try:
            if not cand.exists():
                continue
            data = json.loads(cand.read_text(encoding="utf-8"))
            params = data.get("Parameters") or []
            ops = []
            for prm in params:
                if not isinstance(prm, dict) or not prm.get("Id"):
                    continue
                ops.append({
                    "id": str(prm["Id"]),
                    "value": float(prm.get("Value", 0.0)),
                    "blend": str(prm.get("Blend") or "Add"),
                })
            if ops:
                return ops
        except Exception:
            continue
    return []


def _expand_expression_actions(pool: Dict, model_dir: Optional[Path] = None) -> None:
    """Inline-expand each action that carries an "expression" key into
    ``expression_ops`` (plain param ops) so the apply path never re-reads JSON."""
    for cfg in pool.values():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("expression"):
            cfg["expression_ops"] = _load_exp3_ops(cfg["expression"], model_dir)


# Config loading

class BodyActions:
    """Loaded body_actions.json with helpers for resolving actions."""

    def __init__(self, data: Optional[Dict] = None, model_dir: Optional[Path] = None):
        d = data if data is not None else _DEFAULT_ACTIONS
        self.raw: Dict = d
        self.aliases: Dict[str, str] = d.get("mood_aliases", {})
        self.moods: Dict = d.get("moods", {})
        self.fx: Dict = d.get("fx", {})
        self.look: Dict = d.get("look", {})
        self.gestures: Dict = d.get("gestures", {})
        lim = d.get("limits", {})
        self.max_tags_per_reply: int = int(lim.get("max_tags_per_reply", 4))
        self.max_simultaneous_fx: int = int(lim.get("max_simultaneous_fx", 3))
        # Phase 3c: expand any "expression" refs into plain param ops once here.
        _expand_expression_actions(self.fx, model_dir)
        _expand_expression_actions(self.look, model_dir)
        _expand_expression_actions(self.gestures, model_dir)

    def mood_names(self) -> List[str]:
        """All enabled real mood names (canonical)."""
        return [m for m, cfg in self.moods.items() if cfg.get("enabled", True)]

    def resolve_mood(self, name: str) -> Optional[str]:
        """Resolve a mood token to its canonical real name, or None if unknown/disabled.

        Accepts both ``joyful`` and the explicit ``mood:joyful`` (and aliases such
        as ``happy`` / ``mood:happy``) through the same path.
        """
        key = name.strip().lower()
        if key.startswith("mood:"):
            key = key[len("mood:"):].strip()
        if not key:
            return None
        key = self.aliases.get(key, key)
        cfg = self.moods.get(key)
        if cfg is None or not cfg.get("enabled", True):
            return None
        return key

    def wink_config(self) -> Optional[Dict]:
        """The gesture 'wink' config, or None if not enabled."""
        cfg = self.gestures.get("wink")
        if cfg is None or not cfg.get("enabled", True):
            return None
        return cfg

    def wink_sides(self) -> List[str]:
        """Valid wink sides; falls back to ['left','right']."""
        cfg = self.wink_config()
        if not cfg:
            return []
        return list(cfg.get("sides") or ["left", "right"])

    def wink_default_side(self) -> str:
        cfg = self.wink_config()
        if not cfg:
            return "right"
        sides = self.wink_sides()
        default = cfg.get("default_side", "right")
        return default if default in sides else (sides[0] if sides else "right")

    def resolve_wink(self, side: Optional[str] = None) -> Optional[str]:
        """Resolve a wink side to a canonical screen side ('left'/'right').

        Returns None if wink is disabled.  A missing/unknown side falls back to
        the default side.  Side is the SCREEN side as the viewer sees it."""
        if self.wink_config() is None:
            return None
        sides = self.wink_sides()
        if side:
            s = side.strip().lower()
            if s in sides:
                return s
        return self.wink_default_side()

    def enabled_capped_actions(self) -> Dict[str, Dict]:
        """All enabled non-mood actions (fx/look/gesture/wink) for menus/preview."""
        out: Dict[str, Dict] = {}
        for kind, pool in (("fx", self.fx), ("look", self.look),
                           ("gesture", self.gestures)):
            for name, cfg in pool.items():
                if name == "wink":
                    continue              # wink exposed separately with its sides
                if cfg.get("enabled", True):
                    out[f"{kind}:{name}"] = dict(cfg, kind=kind, name=name)
        # wink with sides (screen side as the viewer sees it)
        wcfg = self.wink_config()
        if wcfg is not None:
            for side in self.wink_sides():
                out[f"wink:{side}"] = dict(wcfg, kind="wink", name=side, side=side)
            out["wink"] = dict(wcfg, kind="wink",
                               name=self.wink_default_side(),
                               side=self.wink_default_side())
        return out


_ACTIONS: Optional[BodyActions] = None


def load_actions(path: Optional[Path] = None) -> BodyActions:
    """Load (and cache) BodyActions from body_actions.json."""
    global _ACTIONS
    if path is None:
        path = _DEFAULT_ACTIONS_PATH
    if _ACTIONS is not None and path == _DEFAULT_ACTIONS_PATH:
        return _ACTIONS
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = _DEFAULT_ACTIONS
    ba = BodyActions(data)
    if path == _DEFAULT_ACTIONS_PATH:
        _ACTIONS = ba
    return ba


def actions() -> BodyActions:
    return load_actions()


def resolve_mood(name: str) -> Optional[str]:
    """Global helper: resolve a mood token to its canonical real name."""
    return actions().resolve_mood(name)


# BODY prompt section 

def build_body_prompt(ba: Optional[BodyActions] = None) -> str:
    """Generate the 'BODY' section of the system prompt from body_actions.json.

    Include this ONLY when the Live2D renderer is active (the caller decides).
    The vocabulary (moods, fx, look, gestures) and the per-reply limit all come
    from body_actions.json so the prompt and the parser can never drift apart.

    Phrasing deliberately avoids the word ``tag`` (it is an internal mechanism):
    the model is told to put a short direction marker at the start of a sentence
    and encouraged to use it sparingly.
    """
    ba = ba or actions()
    moods  = ", ".join(ba.mood_names()) or "(none enabled)"
    fx     = ", ".join(ba.fx.keys()) or "(none enabled)"
    looks  = ", ".join(ba.look.keys()) or "(none enabled)"
    gests  = ", ".join(ba.gestures.keys()) or "(none enabled)"
    mx     = ba.max_tags_per_reply
    return (
        "\n--BODY CONTROL--\n"
        "Sometimes add a short direction marker in square brackets at the START of a "
        "sentence, before the words, to convey tone and subtle body movement. Keep "
        f"it sparing: 0 to 2 per reply is typical, never more than {mx}. Use ONLY "
        "the markers listed below. The marker applies to the sentence it precedes.\n"
        f"- tone marker (one of): {moods}\n"
        f"- effect marker (one of): {fx}\n"
        f"- look marker (one of): {looks}\n"
        f"- gesture marker (one of): {gests}\n"
        "Examples:\n"
        '[mood:joyful] Oh hi, you\'re back!\n'
        '[fx:blush] Really? That\'s so nice to hear.\n'
        '[mood:sad] I missed you a lot.\n'
        '[look:left] I was thinking about that earlier.\n'
        "Never mention or explain these markers to the user; just use them."
    )


# A single parsed tag 

class Tag:
    __slots__ = ("kind", "name", "raw")

    def __init__(self, kind: str, name: str, raw: str):
        self.kind = kind          # mood | fx | look | gesture | wink | clear
        self.name = name          # resolved name ('' for clear)
        self.raw = raw            # original bracketed text (for logging)

    def __repr__(self) -> str:    # pragma: no cover - debugging aid
        return f"Tag({self.kind}:{self.name})"


def _parse_bracket(raw: str, ba: BodyActions) -> Optional[Tag]:
    """Interpret the inside of one ``[ ... ]`` as a Tag, or None if not usable.

    * empty -> None
    * ``clear`` -> Tag('clear')
    * ``kind:name`` -> kind-specific Tag (unknown kind dropped)
    * bare name -> mood Tag via the shared resolver (so [joyful], [mood:joyful]
      and alias [happy] all resolve identically)
    """
    s = raw.strip()
    if not s:
        return None
    low = s.lower()
    if low == "clear":
        return Tag("clear", "", raw)
    if ":" in s:
        kind, _, val = s.partition(":")
        kind = kind.strip().lower()
        val = val.strip()
        if kind not in _KINDS:
            return None                       # unknown kind -> dropped silently
        if kind == "mood":
            real = ba.resolve_mood(val)
            return Tag("mood", real, raw) if real else None
        if kind == "wink":
            side = ba.resolve_wink(val)
            return Tag("wink", side, raw) if side else None
        # fx / look / gesture
        if kind == "gesture":
            pool = ba.gestures
        elif kind == "look":
            pool = ba.look
        else:  # fx
            pool = ba.fx
        # gesture:wink is shorthand for a wink with the default side
        if kind == "gesture" and val.lower() == "wink":
            side = ba.resolve_wink(None)
            return Tag("wink", side, raw) if side else None
        cfg = pool.get(val.lower())
        if cfg is None or not cfg.get("enabled", True):
            return None                       # unknown/disabled -> dropped
        return Tag(kind, val.lower(), raw)
    # Bare name -> legacy mood path, shared resolver (but bare [wink] is a wink).
    if s.lower() == "wink":
        side = ba.resolve_wink(None)
        return Tag("wink", side, raw) if side else None
    real = ba.resolve_mood(s)
    return Tag("mood", real, raw) if real else None


# Extracting tags from a complete string 

class Extracted:
    __slots__ = ("clean", "tags")

    def __init__(self, clean: str, tags: List[Tag]):
        self.clean = clean        # text with tags removed (safe for TTS)
        self.tags = tags          # honoured tags in order


def _strip_leftover(text: str) -> str:
    """Last-resort pre-TTS filter: remove any leftover short lowercase
    ``word:word`` bracket, any unclosed trailing tag-prefix, and any ``[clear]``
    so a missed tag is never spoken aloud.  Genuine prose brackets
    ("[Note: something]", "[1]", "list[0]") are intentionally left untouched."""
    text = _LEFTOVER_TAG_RE.sub("", text)
    text = _LEFTOVER_UNCLOSED_RE.sub("", text)
    return (text.replace("[clear]", "").replace("[Clear]", "")
                .replace("[CLEAR]", ""))


def extract_tags(text: str, ba: Optional[BodyActions] = None,
                 cap_remaining: Optional[int] = None,
                 counters: Optional[Dict[str, int]] = None) -> Tuple[str, List[Tag], int]:
    """Strip tags from ``text`` and return ``(clean, honoured_tags, remaining_cap)``.

    ``cap_remaining`` is the number of non-mood tags still allowed across the
    reply; mood tags never count toward it.  Returns the new remaining count.

    ``counters`` (optional) is an in/out dict bumped for Phase 4b stats:
      * ``unknown``     — a tag-like short lowercase ``word:word`` bracket that
                          is not a recognised tag (it stays verbatim here and is
                          stripped by the leftover filter before TTS).
      * ``cap_dropped`` — a non-mood tag dropped because the per-reply cap was
                          exhausted.
    """
    if ba is None:
        ba = actions()
    if cap_remaining is None:
        cap_remaining = ba.max_tags_per_reply

    honours: List[Tag] = []
    cap = cap_remaining

    def _repl(m: re.Match) -> str:
        nonlocal cap
        tag = _parse_bracket(m.group(1), ba)
        if tag is None:
            # Not a usable tag.  If it still *looks* like a missed tag (short
            # lowercase word:word), count it as an unknown tag dropped — a
            # tag-like bracket that never reaches the model or speech.
            if counters is not None and _LEFTOVER_TAG_RE.match(f"[{m.group(1)}]"):
                counters["unknown"] = counters.get("unknown", 0) + 1
            return m.group(0)     # not a usable tag -> keep verbatim
        if tag.kind == "mood":
            honours.append(tag)
            return ""
        if tag.kind == "clear":
            honours.append(tag)
            return ""
        # fx / look / gesture / wink : capped
        if cap > 0:
            honours.append(tag)
            cap -= 1
            return ""
        # cap exhausted -> silently drop (do not emit; do not speak)
        if counters is not None:
            counters["cap_dropped"] = counters.get("cap_dropped", 0) + 1
        return ""

    clean = _TAG_RE.sub(_repl, text)
    clean = _strip_leftover(clean)
    return clean.strip(), honours, cap


# Streaming tag parser 

class SentenceFrame:
    """One completed sentence ready for TTS, with its attached tags."""
    __slots__ = ("text", "tags", "mood")

    def __init__(self, text: str, tags: List[Tag], mood: Optional[str]):
        self.text = text    # clean (tag-free) sentence text
        self.tags = tags    # all honoured tags for this sentence's audio start
        self.mood = mood    # running reply mood at this sentence (or None)

    def __repr__(self) -> str:     # pragma: no cover - debugging aid
        return f"SentenceFrame({self.text!r}, mood={self.mood}, tags={self.tags})"


class StreamingTagParser:
    """Streaming, latency-aware tag parser.

    * A completed sentence goes to TTS immediately.
    * We hold back ONLY from an open ``[`` until ``]`` or 40 chars, so a tag
      split across chunks (``[mo`` + ``od:joyful] Hello!``) is never truncated
      and never sent to TTS as stray text.
    * A tag that follows sentence-ending punctuation attaches to the NEXT
      sentence (it lands in the next segment after the split).
    * A tag-only fragment is never sent to TTS as its own sentence.
    * The leading mood tag is the reply default; a later [mood:X] applies to
      its sentence and all following sentences until the next mood tag.
    * Mood tags never count toward the 4-tag cap; fx/look/gesture/wink do.
    """

    # Hold-back cap (chars) for an unclosed '[' before we flush as plain text.
    _MAX_BRACKET_HOLD = 40

    def __init__(self, ba: Optional[BodyActions] = None):
        self._ba = ba if ba is not None else actions()
        self._buf = ""
        self._cap_remaining = self._ba.max_tags_per_reply
        self._current_mood: Optional[str] = None
        # Non-mood tags carried from a tag-only / leading fragment so they attach
        # to the next real sentence (or the last one if the reply ends tag-only).
        self._pending: List[Tag] = []
        # The most recently emitted frame, so a trailing tag can attach to it.
        self._last_frame: Optional[SentenceFrame] = None
        # Phase 4b: per-reply tag stats (bumped by extract_tags).
        self._counters: Dict[str, int] = {}

    # -- state introspection (for tests / logging) --
    @property
    def current_mood(self) -> Optional[str]:
        return self._current_mood

    @property
    def cap_remaining(self) -> int:
        return self._cap_remaining

    def stats(self) -> Dict[str, int]:
        """Phase 4b: unknown-dropped and cap-dropped counts for this reply."""
        return {
            "unknown_dropped": self._counters.get("unknown", 0),
            "cap_dropped": self._counters.get("cap_dropped", 0),
        }

    def _next_complete(self) -> Optional[str]:
        """Return the next complete sentence (incl. terminator) or None if held."""
        m = SENTENCE_END.search(self._buf)
        if not m:
            return None
        end = m.start()                      # end of punctuation (before space)
        text_so_far = self._buf[:end]
        last_open = text_so_far.rfind("[")
        last_close = text_so_far.rfind("]")
        if last_open > last_close and (end - last_open) < self._MAX_BRACKET_HOLD:
            return None                     # inside a not-yet-closed tag: hold
        return self._buf[:m.end()]

    def feed(self, chunk: str) -> List[SentenceFrame]:
        """Feed a streamed chunk; returns any complete sentences ready for TTS."""
        self._buf += chunk
        frames: List[SentenceFrame] = []
        while True:
            sentence = self._next_complete()
            if sentence is None:
                break
            self._buf = self._buf[len(sentence):]
            self._buf = self._buf.lstrip("\n ")
            frame = self._render_sentence(sentence)
            if frame is not None:
                frames.append(frame)
                self._last_frame = frame
        return frames

    def _render_sentence(self, sentence: str) -> Optional[SentenceFrame]:
        """Strip tags from one completed sentence segment and assemble a frame.

        A tag-only fragment returns None (tags go to _pending, never spoken).
        """
        clean, tags, self._cap_remaining = extract_tags(
            sentence, self._ba, self._cap_remaining, self._counters)

        # Merge tags carried from before with this sentence's own tags.
        all_tags = self._pending + tags
        self._pending = []

        for t in all_tags:
            if t.kind == "mood":
                self._current_mood = t.name
            # fx/look/gesture/wink/clear: carried as execution actions.

        if not clean:
            # Tag-only fragment: keep tags for the next real sentence.
            self._pending = all_tags
            return None

        # Mood semantics: leading tag is the reply default; later [mood:X]
        # applies to this + following sentences until the next mood tag.
        frame_mood = self._current_mood
        return SentenceFrame(clean, all_tags, frame_mood)

    def finish(self) -> List[SentenceFrame]:
        """Flush any remaining buffered text and pending trailing tags.

        * A trailing tag with no following sentence fires at the START of the
          last emitted sentence (we chose this over dropping it, so the AI's
          closing mood/gesture is preserved).
        * An unclosed '[' that never closed within the hold cap is flushed as
          plain text; the leftover-strip filter removes it before TTS.
        """
        tail = self._buf.strip()
        self._buf = ""
        frames: List[SentenceFrame] = []
        if tail:
            frame = self._render_sentence(tail)
            if frame is not None:
                frames.append(frame)
                self._last_frame = frame
        # Any tags left over (trailing tag-only / tags after last sentence):
        # attach them to the last emitted sentence's tag set.
        if self._pending:
            last = frames[-1] if frames else self._last_frame
            if last is not None:
                for t in self._pending:
                    if t.kind == "mood":
                        self._current_mood = t.name
                combined = last.tags + self._pending
                last.tags = combined
                if self._current_mood:
                    last.mood = self._current_mood
            self._pending = []
        return frames
