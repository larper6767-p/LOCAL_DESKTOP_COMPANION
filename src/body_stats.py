""" body-control logging statistics (pure, testable).

Accumulates aggregates for the ``[body]`` log lines in logs/wife.log:

* tags per reply (how many honoured tags each reply carried, by kind)
* unknown tags dropped (tag-like ``word:word`` brackets never honoured)
* cap-dropped tags (non-mood tags beyond the per-reply cap, silently dropped)
* tag-to-audio-start latency (between enqueue and the sentence's audio start)

Pure module: no Qt, no GUI, no wall-clock reads at construction.  Latency
values are measured with ``time.monotonic()`` by the caller (in the GUI thread at
audio start) and passed in here, so everything is deterministic for tests.
"""





from __future__ import annotations

from typing import Dict, Optional


class BodyStats:
    """Aggregates body-control statistics across many replies."""

    def __init__(self) -> None:
        self._replies = 0                  # replies carrying at least one tag
        self._total_tags = 0               # total honoured tags
        self._reply_tag_counts: Dict[int, int] = {}   # tag-count -> how often
        self._tags_by_kind: Dict[str, int] = {}       # kind -> count
        self._unknown_dropped = 0          # tag-like brackets never honoured
        self._cap_dropped = 0              # non-mood tags dropped by the cap
        self._latency: list = []           # tag->audio-start latencies (s)

    def record_reply(self, tags_by_kind: Dict[str, int]) -> None:
        """Record one reply's honoured tags, given per-kind counts."""
        total = sum(v for v in tags_by_kind.values() if v > 0)
        if total <= 0:
            return
        self._replies += 1
        self._total_tags += total
        self._reply_tag_counts[total] = self._reply_tag_counts.get(total, 0) + 1
        for k, n in tags_by_kind.items():
            if n > 0:
                self._tags_by_kind[k] = self._tags_by_kind.get(k, 0) + n

    def record_unknown_dropped(self, n: int = 1) -> None:
        if n > 0:
            self._unknown_dropped += n

    def record_cap_dropped(self, n: int = 1) -> None:
        if n > 0:
            self._cap_dropped += n

    def record_latency(self, seconds: Optional[float]) -> None:
        if seconds is not None and seconds >= 0:
            self._latency.append(seconds)

    # accessors

    @property
    def replies(self) -> int:
        return self._replies

    @property
    def total_tags(self) -> int:
        return self._total_tags

    @property
    def unknown_dropped(self) -> int:
        return self._unknown_dropped

    @property
    def cap_dropped(self) -> int:
        return self._cap_dropped

    @property
    def tags_by_kind(self) -> Dict[str, int]:
        return dict(self._tags_by_kind)

    def _latency_summary(self) -> Dict[str, float]:
        if not self._latency:
            return {"samples": 0, "last": 0.0, "min": 0.0, "avg": 0.0, "max": 0.0}
        lo = min(self._latency)
        hi = max(self._latency)
        avg = sum(self._latency) / len(self._latency)
        return {"samples": len(self._latency), "last": self._latency[-1],
                "min": lo, "avg": avg, "max": hi}

    def summary(self) -> Dict:
        """Snapshot useful for a body_log line / debug dump."""
        return {
            "replies_with_tags": self._replies,
            "total_tags": self._total_tags,
            "tags_by_kind": dict(self._tags_by_kind),
            "unknown_dropped": self._unknown_dropped,
            "cap_dropped": self._cap_dropped,
            "tag_to_audio_latency_ms": self._latency_summary(),
        }

    def reset(self) -> None:
        self.__init__()


__all__ = ["BodyStats"]
