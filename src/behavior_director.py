"""Pure behaviour director for the Live2D pet .

This is the "body" half of the brain/body split.  It is deliberately PURE and
testable — no GL, no Qt, no live2d, no clock.  All time is passed in as ``dt``,
and it uses a seeded ``random.Random`` so behaviour is deterministic in tests.

It consumes the outputs of :class:`behavior_state.BehaviorState` (blink_rate,
sway_amplitude, sway_speed, breathing_depth, fidget_rate, gaze, state_factor,
valence_motion) plus a ``liveliness`` scalar and produces a
:class:`DirectorFrame` of per-parameter write intents the Live2D widget applies
to the model.

Design rules :

* **Per-parameter priority / ownership.**  For any safe param, at most one
  owner writes it per frame, and that is decided by the documented priority:
  ``mood baseline < idle drift (additive) < cursor/attention (additive)
  < gesture overrides (faded)``; blink is a short reflex that a wink can
  suppress for its eye; fx switches are independent.  The director resolves the
  additive (head/eye/body drift + cursor/attention) and the override (breath)
  values; gestures/fx follow the existing body-tags ownership in the widget.
* **Never write physics parameters** (hair/ear/tail/cloth).  ``Ear_smile`` is
  the only direct ear parameter; ear/tail life comes from head motion.  This is
  enforced structurally: the director only ever emits params that are in its
  allowed set, and a test asserts no physics param ever appears.
* **Closed blink reaches exactly 0.0** and a **wink suppresses blink only for
  its eye**.
* **Liveliness scaling:** ``final_motion = liveliness x state_factor``, capped by
  the motion budget (max combined head angle in degrees, max eye offset, max
  body drift — config numbers, subtle by default).
* **Organic motion:** smoothed noise instead of plain sine sums; blink can be
  correlated with larger gaze shifts and there can be an occasional double
  blink.
* **Engagement mostly changes the KIND of behaviour** (gaze toward viewer vs
  wandering), not just the amount — the director enacts the `gaze` it is given.

Physics params are the ones NEVER written.  This is the authoritative (and only)
list the director may touch:
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional

__all__ = [
    "PHYSICS_PARAMS",
    "DIRECTOR_PARAMS",
    "MotionBudget",
    "DirectorFrame",
    "BehaviorDirector",
    "BlinkController",
]


#  Physics params the director must NEVER write
# These are the hair / ear / tail / cloth / accessory parameters the physics
# engine (简.physics3.json) produces.  The director is structurally forbidden
# from emitting them: DIRECTOR_PARAMS is disjoint from this set and a test
# asserts no physics param ever leaks into a frame.  Ear_smile is the only
# direct ear parameter on this model, but even it is NOT written by the director
# — ear life comes from head motion (physics runs on top of our head writes).
PHYSICS_PARAMS: frozenset[str] = frozenset({
    # hair (front / back / nailed / ponytail) + cloth / skirt / sleeves
    "HairFront_BS", "HairFront_BS2", "HairFront_BS3", "HairFront_BS4",
    "HairFront_BSY", "HairFront_BSY2", "HairFront_BSY3",
    "HairFront_IS1", "HairFront_IS2", "HairFront_IS3", "HairFront_IS4",
    "HairFront_ISY", "HairFront_ISY2",
    "HairFront_LS", "HairFront_LS2", "HairFront_LS3", "HairFront_LS4",
    "HairFront_LSY", "HairFront_LSY2",
    "HairFront2_LS1", "HairFront2_LS2", "HairFront2_LS3", "HairFront2_LS4",
    "HairFront2_LSY1", "HairFront2_LSY2",
    "HairPinytail_S1", "HairPinytail_S2", "HairPinytail_S3", "HairPinytail_S4",
    "HairPinytail_S5", "HairPinytail_S6",
    "HairPinytail_SY1", "HairPinytail_SY2", "HairPinytail_SY3", "HairPinytail_SY4",
    "HanfFront_LS1", "HanfFront_LS2", "HanfFront_LS3", "HanfFront_LS4", "HanfFront_LS5",
    "HanfFront_RS1", "HanfFront_RS2", "HanfFront_RS3", "HanfFront_RS4", "HanfFront_RS5",
    "HanfFront_RS6", "SillyHair_S1", "SillyHair_S2", "SillyHair_S3",
    "SillyHair_SY1", "SillyHair_SY2",
    # ears (incl. Ear_smile — the only direct ear param; still physics-owned)
    "Ear_SL1", "Ear_SL2", "Ear_SL3", "Ear_SR1", "Ear_SR2", "Ear_SR3",
    "Ear_eyeL", "Ear_eyeR", "Ear_smile",
    # eyes / lashes / orbit (art-mesh driven by physics, not the director)
    "EyeBall_X", "EyeBall_Y", "EyeCry_SL1", "EyeCry_SL2", "EyeCry_SR1", "EyeCry_SR2",
    "EyeLash_SL", "EyeLash_SR", "EyeLight_SL1", "EyeLight_SL2", "EyeLight_SR1", "EyeLight_SR2",
    "Eye_SL1", "Eye_SL2", "Eye_SL3", "Eye_SR1", "Eye_SR2", "Eye_SR3",
    "Orbit_SL1", "Orbit_SL2", "Orbit_SR1", "Orbit_SR2",
    # chest / clothes / hands / legs / tongue / skirt bows
    "ChestDecorate_S1", "ChestDecorate_S2", "ChestDecorate_SY1", "ChestDecorate_SY2",
    "Chest_S1", "Chest_S2", "Chest_S3", "Chest_SY", "Chest_SY2", "Chest_SY3",
    "ClothesHand_LS1", "ClothesHand_LS2", "ClothesHand_LS3", "ClothesHand_LSY1", "ClothesHand_LSY2",
    "ClothesHand_RS1", "ClothesHand_RS2", "ClothesHand_RS3", "ClothesHand_RSY1", "ClothesHand_RSY2",
    "Close2_S1", "Close2_SY1", "Close_S1", "Close_S2", "Close_SY1", "Close_SY2",
    "Leg_S1", "Leg_S5", "Leg_S6",
    "BowSkirtBack_S1", "BowSkirtBack_S2", "BowSkirtBack_S3", "BowSkirtBack_SY1",
    "ParamTongue_S1", "ParamTongue_S2",
    # per-art-mesh rotation params produced by physics
    "AnimeAngry", "AnimeAngry2",
    "ParamMouthAngleX", "ParamMouthX2", "ParamMouthX_EYE",
}) | frozenset(
    f"Param_Angle_Rotation_{i}_ArtMesh{mesh}"
    for mesh in (219, 220) for i in range(1, 22)
)

# The only params the director is allowed to emit.  Everything else — including
# all physics/hair/ear/tail/cloth params — is off limits.  Note there is NO
# ParamEar_smile here because it is not part of the motion-budget behaviour.
DIRECTOR_PARAMS: frozenset[str] = frozenset({
    # head (additive on mood base)
    "ParamAngleX", "ParamAngleY", "ParamAngleZ",
    # eyes (gaze additive; openness override via blink)
    "ParamEyeBallX", "ParamEyeBallY",
    "ParamEyeLOpen", "ParamEyeROpen",
    # breathing override
    "ParamBreath",
    # body drift (additive, not physics — gentle upper-body sway)
    "ParamBodyAngleX0", "ParamBodyAngleZ0",
    # forward-lean candidates (configurable ``lean_param``; body rotations, not
    # physics — the user picks which one actually looks like leaning in).
    "ParamBodyAngleY0", "ParamChestAngleY",
})


@dataclass(frozen=True)
class MotionBudget:
    """Motion caps, subtle by default.  Numbers come from config/live2d.json."""

    max_head_angle: float = 8.0     # max combined head offset, degrees
    max_eye_offset: float = 0.5     # max eye-ball offset, units
    max_body_drift: float = 2.0     # max body-drift offset, degrees


@dataclass
class DirectorFrame:
    """Per-frame write intents produced by the director.

    ``additive``  — values to ADD on top of the smoothed mood base (head, eyes,
                    body drift).  Already includes idle drift + cursor/attention.
    ``override``  — absolute values to SET (breath).
    ``blink``     — (left_open, right_open) eye-openness override from the blink
                    reflex; ``None`` means "the blink machine does not own this
                    eye this frame" (so a wink/fx/gesture can).  When closed it
                    is exactly ``0.0``.
    ``gaze``      — the gaze mode we are enacting (for logging).
    ``state_log`` — a small dict of the driving state for body_log lines.
    """
    additive: Dict[str, float] = field(default_factory=dict)
    override: Dict[str, float] = field(default_factory=dict)
    blink: tuple[Optional[float], Optional[float]] = (None, None)
    gaze: str = "look_ahead"
    state_log: Dict[str, float] = field(default_factory=dict)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _noise1(rng: random.Random) -> float:
    """Smooth-valued noise in [-1, 1] (sum of two uniforms → soft centre)."""
    return (rng.uniform(-1.0, 1.0) + rng.uniform(-1.0, 1.0)) * 0.5


# Blink reflex

class BlinkController:
    """A short blink reflex driving eye openness.

    * reaches **exactly** ``0.0`` when closed,
    * a ``wink`` suppresses the blink for exactly one eye (the other continues),
    * occasionally does a double blink,
    * interval scales with the desired blink rate.

    Pure: ``update(dt, blink_rate, wink_side) -> (left, right)`` where the return
    is (L_open, R_open) in [0, 1] or ``None`` for an eye the blink does not own
    (e.g. the winked eye, which the gesture layer owns instead).
    """

    CLOSE_T = 0.12
    HOLD_T  = 0.08
    OPEN_T  = 0.15
    MIN_GAP = 0.4    # minimum gap between any two blinks (no fluttering)

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._state = "open"
        self._t = 0.0
        self._next = self._interval(0.18)
        self._double = False

    @staticmethod
    def _interval(rate: float) -> float:
        base = max(rate, 0.05)
        # map a "blinks per second" to a seconds-between gap
        return max(BlinkController.MIN_GAP, 6.0 / (base * 5.0 + 0.5))

    def _decide(self, rate: float) -> None:
        # occasional double blink (small chance)
        self._double = self._rng.random() < 0.12

    def update(self, dt: float, blink_rate: float, wink_side: Optional[str] = None):
        """Advance the blink machine.

        Returns (L, R): each is a float openness 0..1 if the blink owns that eye
        this frame, or ``None`` if it does not (the winked eye — let the gesture
        layer own it).  When the eyes are fully open and idle the blink machine
        still "owns" them at 1.0 so nothing else drifts them off during a normal
        blink cycle; a wink side suppresses ownership of that eye only.
        """
        # A wink suppresses the blink for exactly its eye this whole frame.
        suppress_l = wink_side == "left"
        suppress_r = wink_side == "right"

        def own(openness: float, suppressed: bool) -> Optional[float]:
            return None if suppressed else _clamp(openness, 0.0, 1.0)

        if self._state == "open":
            self._next -= dt
            if self._next <= 0.0:
                self._state = "closing"
                self._t = 0.0
                self._decide(blink_rate)
            return own(1.0, suppress_l), own(1.0, suppress_r)

        self._t += dt

        if self._state == "closing":
            p = min(self._t / self.CLOSE_T, 1.0)
            v = 1.0 - p
            if p >= 1.0:
                self._state = "closed"
                self._t = 0.0
            return own(v, suppress_l), own(v, suppress_r)

        if self._state == "closed":
            # exactly 0.0
            if self._t >= self.HOLD_T:
                self._state = "opening"
                self._t = 0.0
            return own(0.0, suppress_l), own(0.0, suppress_r)

        # opening
        p = min(self._t / self.OPEN_T, 1.0)
        if p >= 1.0:
            self._state = "open"
            self._t = 0.0
            if self._double:
                self._next = self.MIN_GAP   # second blink, but never flutters
                self._double = False
            else:
                self._next = self._interval(blink_rate)
        return own(p, suppress_l), own(p, suppress_r)

    def reset(self) -> None:
        self._state = "open"
        self._t = 0.0
        self._double = False


 #The director 

class BehaviorDirector:
    """Maps behavior-state outputs + liveliness into per-parameter write intents.

    Uses smoothed noise (an eased low-frequency walk rather than raw sines) for
    organic sway/breath/fidget motion.  ``liveliness`` is a plain 0..1 scalar;
    per the plan ``final_motion = liveliness x state_factor`` is capped by the
    motion budget.
    """

    NOD_MIN_GAP = 1.5   # minimum seconds between nod events
    NOD_DUR     = 0.6    # duration of one nod (down-and-up), seconds

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        budget: Optional[MotionBudget] = None,
        liveliness: float = 0.7,
        lean_param: str = "ParamBodyAngleY0",
        lean_sign: float = 1.0,
        breath_pause_enabled: bool = True,
        breath_pause_every: tuple = (40.0, 90.0),
        breath_pause_hold: tuple = (1.0, 1.5),
    ) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._budget = budget if budget is not None else MotionBudget()
        self.liveliness = _clamp(liveliness, 0.0, 1.0)
        self._blink = BlinkController(self._rng)
        # eased noise walkers (per drive, per channel) — smoothed low-freq motion
        self._w = {"sway_x": 0.0, "sway_y": 0.0, "sway_z": 0.0,
                   "breath": 0.0, "body_x": 0.0, "body_z": 0.0,
                   "eye_x": 0.0, "eye_y": 0.0, "fidget": 0.0}
        self._t = 0.0
        self._active_wink_side: Optional[str] = None

        # ── lean (2c) — which body param looks like leaning in is configurable ──
        self._lean_param = lean_param if lean_param in DIRECTOR_PARAMS else "ParamBodyAngleY0"
        self._lean_sign = 1.0 if lean_sign >= 0 else -1.0
        self._lean_current = 0.0

        # ── nod events (2c) — triggered, not a continuous walker 
        self._nod_peak = 0.0
        self._nod_t = 1e9          # > NOD_DUR => not in a nod
        self._last_nod = -1e9
        self._listen_nod_timer = 0.0
        self._next_listen_nod = self._rng.uniform(3.0, 7.0)

        # ── breath pause / sigh (2d) 
        self._bp_enabled = bool(breath_pause_enabled)
        self._bp_every = tuple(float(x) for x in breath_pause_every)
        self._bp_hold = tuple(float(x) for x in breath_pause_hold)
        self._bp_t = 0.0
        self._bp_next = self._new_breath_pause_gap()
        self._bp_hold_rem = 0.0
        self._sigh_peak = 0.0
        self._sigh_t = 0.0
        self._sigh_dur = 2.5
        self._was_speaking = False

    # public 

    def set_liveliness(self, value: float) -> None:
        self.liveliness = _clamp(value, 0.0, 1.0)

    def update(self, dt: float, state) -> DirectorFrame:
        """Produce a frame of write intents for ``dt`` seconds.

        ``state`` is a BehaviorState instance exposing: blink_rate(),
        sway_amplitude(), sway_speed(), breathing_depth(), fidget_rate(),
        gaze(), state_factor(), listening/speaking.
        """
        if dt <= 0.0:
            return DirectorFrame()

        self._t += dt
        rate = state.blink_rate()
        liv = self.liveliness
        sf = state.state_factor()
        motion = _clamp(liv * sf, 0.0, 1.0)   # final_motion, before budget caps
        g = state.gaze()

        # 1. Organic idle drift via eased noise, scaled by the state.
        sway_amp = state.sway_amplitude()
        sway_spd = max(0.2, state.sway_speed())
        breath = state.breathing_depth()

        # walk the noise oscillators toward new targets (smoothed)
        self._walk(dt)

        # 2c ── lean toward the viewer while listening (eases in/out) 
        lean_target = (self._lean_sign * self._budget.max_body_drift * 0.5 * motion
                       if state.listening else 0.0)
        self._lean_current += (lean_target - self._lean_current) * (1.0 - math.exp(-2.0 * dt))

        # 2c ── autonomous small listening nods (every 3–7 s)
        if state.listening:
            self._listen_nod_timer += dt
            if self._listen_nod_timer >= self._next_listen_nod:
                self._listen_nod_timer = 0.0
                self._next_listen_nod = self._rng.uniform(3.0, 7.0)
                self.nod(self._lean_sign * 2.5 * motion)   # small slow listening nod

        # Advance any active nod trajectory (a timed down-and-up on ParamAngleY).
        nod_val = 0.0
        if self._nod_t < self.NOD_DUR:
            self._nod_t += dt
            p = min(self._nod_t / self.NOD_DUR, 1.0)
            nod_val = self._nod_peak * math.sin(math.pi * p)

        head_amp = self._budget.max_head_angle
        head_x = self._w["sway_x"] * sway_amp * head_amp * motion * sway_spd
        head_y = self._w["sway_y"] * sway_amp * head_amp * 0.6 * motion * sway_spd
        head_z = self._w["sway_z"] * 2.0 * sway_amp * motion * sway_spd

        eye_amp = self._budget.max_eye_offset
        # gaze direction: toward viewer (small saccades) vs wandering
        if g.toward_viewer:
            ex = self._w["eye_x"] * 0.35 * eye_amp * motion
            ey = self._w["eye_y"] * 0.25 * eye_amp * motion
        else:
            ex = self._w["eye_x"] * eye_amp * (0.5 + 0.5 * g.intensity) * motion
            ey = self._w["eye_y"] * eye_amp * 0.5 * motion

        body_x = self._w["body_x"] * self._budget.max_body_drift * motion
        body_z = self._w["body_z"] * self._budget.max_body_drift * 0.6 * motion

        # 2d ── breathing override with an occasional pause
        breath_hold = False
        if self._bp_enabled:
            self._bp_t += dt
            if self._bp_hold_rem > 0.0:
                self._bp_hold_rem -= dt
                breath_hold = True
            elif self._bp_t >= self._bp_next:
                self._bp_t = 0.0
                self._bp_next = self._new_breath_pause_gap()
                self._bp_hold_rem = self._rng.uniform(*self._bp_hold)
                breath_hold = True
        if breath_hold:
            breath_val = 0.5   # hold at rest during a pause
        else:
            breath_val = 0.5 + 0.5 * self._w["breath"] * 0.7 * breath * motion

        # Optional slow sigh right after a long reply when arousal is low.
        speaking = bool(state.speaking)
        if speaking and not self._was_speaking:
            self._sigh_peak = 0.0
            self._sigh_t = 0.0
        elif (not speaking) and self._was_speaking and state.arousal < 0.45:
            self._sigh_peak = 0.25
            self._sigh_t = 0.0
        self._was_speaking = speaking
        sigh_val = 0.0
        if self._sigh_peak > 0.0 and self._sigh_t < self._sigh_dur:
            self._sigh_t += dt
            p = min(self._sigh_t / self._sigh_dur, 1.0)
            sigh_val = self._sigh_peak * math.sin(math.pi * p)
            if p >= 1.0:
                self._sigh_peak = 0.0
        if not breath_hold:
            breath_val = _clamp(breath_val + sigh_val * motion, 0.0, 1.0)

        # 3. Blink reflex; a wink side from an active wink gesture (the widget
        #    feeds it in via set_wink_side each frame).
        wink = self._active_wink_side
        l, r = self._blink.update(dt, rate, wink)

        self._active_wink_side = None   # consumed this frame

        additive = {
            "ParamAngleX": _clamp(head_x, -self._budget.max_head_angle, self._budget.max_head_angle),
            "ParamAngleY": _clamp(head_y + nod_val, -self._budget.max_head_angle, self._budget.max_head_angle),
            "ParamAngleZ": head_z,
            "ParamBodyAngleX0": body_x,
            "ParamBodyAngleZ0": body_z,
            "ParamEyeBallX": _clamp(ex, -eye_amp, eye_amp),
            "ParamEyeBallY": _clamp(ey, -eye_amp, eye_amp),
        }
        if abs(self._lean_current) > 1e-4:
            additive[self._lean_param] = additive.get(self._lean_param, 0.0) + self._lean_current
        frame = DirectorFrame(
            additive=additive,
            override={"ParamBreath": _clamp(breath_val, 0.0, 1.0)},
            blink=(l, r),
            gaze=g.mode,
            state_log={
                "liveliness": liv, "state_factor": sf, "motion": motion,
                "blink_rate": rate, "arousal": state.arousal,
                "engagement": state.engagement, "valence": state.valence,
            },
        )
        return frame

    def nod(self, amplitude: float) -> None:
        """Trigger a nod event: a brief head down-and-up on ParamAngleY.

        Nods are E V E N T S, not a continuous walker.  Fired by the GUI at the
        start of each spoken sentence (scaled by arousal + liveliness) and
        autonomously (small) while listening.  A nod is ignored if one is already
        running or if another fired within ``NOD_MIN_GAP`` seconds so they never
        stack or flutter.  Amplitude is capped by the head motion budget.
        """
        if self._nod_t < self.NOD_DUR:
            return                          # a nod is already in progress
        if self._t - self._last_nod < self.NOD_MIN_GAP:
            return                          # too soon after the last nod
        self._last_nod = self._t
        self._nod_t = 0.0
        cap = max(self._budget.max_head_angle * 0.5, 1.0)
        self._nod_peak = _clamp(amplitude, -cap, cap)

    def _new_breath_pause_gap(self) -> float:
        return self._rng.uniform(self._bp_every[0], self._bp_every[1])

    def set_wink_side(self, side: Optional[str]) -> None:
        """Feed the current wink gesture side (the widget calls this each frame
        while a wink gesture is active so the blink suppresses only that eye)."""
        self._active_wink_side = side if side in ("left", "right") else None

    # noise walkers

    def _walk(self, dt: float) -> None:
        """Ease each noise walker toward a new small random target occasionally."""
        ease = 1.0 - math.exp(-1.5 * dt)
        for key in self._w:
            target = _noise1(self._rng) * 1.0
            self._w[key] += (target - self._w[key]) * ease
