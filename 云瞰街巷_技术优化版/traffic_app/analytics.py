"""Frame-based traffic statistics; this module has no GUI or ML dependencies.

Timestamps are seconds in the source timeline (video presentation time, or
elapsed capture time for a camera), never wall-clock time. Counts describe
detected tracks and observed region dwell, not road capacity or violations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import math
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    track_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.class_id, int) or self.class_id < 0:
            raise ValueError("class_id must be a non-negative integer")
        if not isinstance(self.class_name, str) or not self.class_name:
            raise ValueError("class_name must be non-empty")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if len(self.bbox) != 4 or not all(math.isfinite(v) for v in self.bbox):
            raise ValueError("bbox must contain four finite xyxy coordinates")
        x1, y1, x2, y2 = self.bbox
        if x2 < x1 or y2 < y1:
            raise ValueError("bbox must use ordered xyxy coordinates")
        if self.track_id is not None and (
            not isinstance(self.track_id, int) or self.track_id < 0
        ):
            raise ValueError("track_id must be a non-negative integer or None")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnalyticsConfig:
    line_y: float = 0.5
    line_hysteresis: float = 0.02
    roi: tuple[float, float, float, float] | None = None
    dwell_threshold_seconds: float = 5.0
    track_ttl_seconds: float = 30.0
    # Any straight segment the user draws on the ground. A line on the ground
    # plane stays a line under perspective, so a segment drawn in the image is
    # a real cross-section on the ground — unlike a fixed horizontal row.
    section: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.line_y) or not 0 <= self.line_y <= 1:
            raise ValueError("line_y must be normalized to [0, 1]")
        if not math.isfinite(self.line_hysteresis) or not 0 <= self.line_hysteresis <= 0.5:
            raise ValueError("line_hysteresis must be between 0 and 0.5")
        if self.roi is not None:
            if len(self.roi) != 4 or not all(
                math.isfinite(v) and 0 <= v <= 1 for v in self.roi
            ):
                raise ValueError("roi must contain four normalized xyxy coordinates")
            x1, y1, x2, y2 = self.roi
            if x1 >= x2 or y1 >= y2:
                raise ValueError("roi must have positive width and height")
        if self.section is not None:
            if len(self.section) != 4 or not all(
                math.isfinite(v) and 0 <= v <= 1 for v in self.section
            ):
                raise ValueError("section must contain four normalized coordinates")
            if math.dist(self.section[:2], self.section[2:]) < 1e-3:
                raise ValueError("section endpoints must not coincide")
        if not math.isfinite(self.dwell_threshold_seconds) or self.dwell_threshold_seconds < 0:
            raise ValueError("dwell_threshold_seconds must be finite and non-negative")
        if not math.isfinite(self.track_ttl_seconds) or self.track_ttl_seconds <= 0:
            raise ValueError("track_ttl_seconds must be finite and positive")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnalyticsEvent:
    event_type: str
    timestamp: float
    frame_index: int
    track_id: int
    class_name: str
    direction: str | None = None
    duration_seconds: float | None = None
    x: float | None = None
    y: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FrameStats:
    timestamp: float
    frame_index: int
    current_count: int
    tracked_current_count: int
    cumulative_unique_count: int
    counts_by_class: dict[str, int]
    cumulative_by_class: dict[str, int]
    line_counts: dict[str, int]
    dwell_current_count: int
    events: tuple[AnalyticsEvent, ...] = ()
    roi_current_count: int = 0
    dwell_seconds_by_track: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class _TrackState:
    last_seen: float
    side: int = 0
    roi_entered_at: float | None = None
    dwell_reported: bool = False


class TrafficAnalytics:
    """One independent session, with persistent ID ledgers and expiring state.

    An ID is counted once per session and at most once in each crossing
    direction. A disappearing track's motion/dwell state expires; the compact
    ID ledgers remain so a reused ID cannot inflate cumulative counts.
    Start a new instance (or reset) when changing sources or seeking backwards.
    """

    def __init__(self, config: AnalyticsConfig | None = None) -> None:
        self.config = config or AnalyticsConfig()
        self.reset()

    def reset(self) -> None:
        self._seen_ids: set[int] = set()
        self._crossed: dict[str, set[int]] = {"up": set(), "down": set()}
        self._cumulative_classes: Counter[str] = Counter()
        self._states: dict[int, _TrackState] = {}
        self._last_timestamp: float | None = None
        self._first_timestamp: float | None = None
        self._last_frame_index: int | None = None
        self._frame_count = 0
        self._detection_count = 0
        self._dwell_event_count = 0
        self._peak_current_count = 0
        self._last_stats: FrameStats | None = None

    @property
    def active_track_count(self) -> int:
        return len(self._states)

    @property
    def last_stats(self) -> FrameStats | None:
        return self._last_stats

    def update(
        self,
        detections: Iterable[Detection],
        timestamp: float,
        frame_index: int,
        width: int,
        height: int,
    ) -> FrameStats:
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite source seconds >= 0")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("source timestamp moved backwards; reset for a new source or seek")
        if not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("frame_index must increase; reset for a new source or seek")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError("frame width and height must be positive integers")

        items = list(detections)
        if not all(isinstance(d, Detection) for d in items):
            raise TypeError("detections must contain Detection instances")

        # Duplicate tracker IDs occasionally occur in malformed output: keep
        # their highest-confidence detection, so one object cannot count twice.
        tracked: dict[int, Detection] = {}
        untracked: list[Detection] = []
        for detection in items:
            if detection.track_id is None:
                untracked.append(detection)
            elif detection.track_id not in tracked or (
                detection.confidence > tracked[detection.track_id].confidence
            ):
                tracked[detection.track_id] = detection
        items = untracked + list(tracked.values())

        # Dwell is continuous observation: even one missing analyzed frame ends
        # a visit. Motion state stays until TTL to tolerate short tracking gaps.
        present_ids = set(tracked)
        for track_id in list(self._states):
            state = self._states[track_id]
            if timestamp - state.last_seen > self.config.track_ttl_seconds:
                del self._states[track_id]
            elif track_id not in present_ids:
                state.roi_entered_at = None
                state.dwell_reported = False

        events: list[AnalyticsEvent] = []
        roi_current_count = 0
        dwell_seconds: dict[int, float] = {}
        dwell_current_count = 0
        for track_id, detection in tracked.items():
            if track_id not in self._seen_ids:
                self._seen_ids.add(track_id)
                self._cumulative_classes[detection.class_name] += 1
            state = self._states.setdefault(track_id, _TrackState(last_seen=timestamp))
            state.last_seen = timestamp
            x1, _y1, x2, y2 = detection.bbox
            x = ((x1 + x2) / 2) / width
            y = y2 / height

            if self.config.section is not None:
                ax, ay, bx, by = self.config.section
                ex, ey = bx - ax, by - ay
                length = math.hypot(ex, ey)
                # Signed distance to the segment's infinite line; the sign says
                # which side of the section the feet are on.
                signed = ((x - ax) * ey - (y - ay) * ex) / length
                band = self.config.line_hysteresis
                side = -1 if signed < -band else (1 if signed > band else 0)
            else:
                lower = self.config.line_y - self.config.line_hysteresis
                upper = self.config.line_y + self.config.line_hysteresis
                side = -1 if y < lower else (1 if y > upper else 0)
            if side:
                if state.side and side != state.side:
                    direction = "down" if side > state.side else "up"
                    if track_id not in self._crossed[direction]:
                        self._crossed[direction].add(track_id)
                        events.append(AnalyticsEvent(
                            event_type="line_crossing", timestamp=timestamp,
                            frame_index=frame_index, track_id=track_id,
                            class_name=detection.class_name, direction=direction,
                            x=x, y=y,
                        ))
                state.side = side

            inside = self.config.roi is not None and (
                self.config.roi[0] <= x <= self.config.roi[2]
                and self.config.roi[1] <= y <= self.config.roi[3]
            )
            if inside:
                roi_current_count += 1
                if state.roi_entered_at is None:
                    state.roi_entered_at = timestamp
                duration = max(0.0, timestamp - state.roi_entered_at)
                dwell_seconds[track_id] = duration
                if duration >= self.config.dwell_threshold_seconds:
                    dwell_current_count += 1
                    if not state.dwell_reported:
                        events.append(AnalyticsEvent(
                            event_type="roi_dwell", timestamp=timestamp,
                            frame_index=frame_index, track_id=track_id,
                            class_name=detection.class_name,
                            duration_seconds=duration, x=x, y=y,
                        ))
                        self._dwell_event_count += 1
                        state.dwell_reported = True
            else:
                state.roi_entered_at = None
                state.dwell_reported = False

        self._last_timestamp = timestamp
        if self._first_timestamp is None:
            self._first_timestamp = timestamp
        self._last_frame_index = frame_index
        self._frame_count += 1
        self._detection_count += len(items)
        self._peak_current_count = max(self._peak_current_count, len(items))
        self._last_stats = FrameStats(
            timestamp=timestamp, frame_index=frame_index,
            current_count=len(items), tracked_current_count=len(tracked),
            cumulative_unique_count=len(self._seen_ids),
            counts_by_class=dict(sorted(Counter(d.class_name for d in items).items())),
            cumulative_by_class=dict(sorted(self._cumulative_classes.items())),
            line_counts={key: len(value) for key, value in self._crossed.items()},
            dwell_current_count=dwell_current_count,
            events=tuple(events), roi_current_count=roi_current_count,
            dwell_seconds_by_track=dwell_seconds,
        )
        return self._last_stats

    def summary(self) -> dict:
        return {
            "analyzed_frames": self._frame_count,
            "detection_observations": self._detection_count,
            "cumulative_unique_count": len(self._seen_ids),
            "cumulative_by_class": dict(sorted(self._cumulative_classes.items())),
            "line_counts": {key: len(value) for key, value in self._crossed.items()},
            "roi_dwell_events": self._dwell_event_count,
            "peak_current_count": self._peak_current_count,
            "source_start_seconds": self._first_timestamp,
            "source_end_seconds": self._last_timestamp,
            "source_duration_seconds": (
                self._last_timestamp - self._first_timestamp
                if self._last_timestamp is not None and self._first_timestamp is not None
                else 0.0
            ),
            "counting_notes": {
                "current_count": "Objects detected in the current analyzed frame; includes untracked detections.",
                "cumulative_unique_count": "Distinct tracker IDs in this source session; tracker errors can affect counts.",
                "line_counts": "Bottom-center crosses horizontal line plus hysteresis; each ID once per direction.",
                "roi_dwell_events": "Continuous observed presence in ROI beyond threshold; not parking or violation identification.",
                "timestamp": "Seconds in source timeline, not wall-clock time.",
            },
        }
