"""Incremental session exports and bounded, restart-safe session history."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Iterable, Mapping
import uuid

from .analytics import AnalyticsConfig, Detection, FrameStats


SCHEMA_VERSION = 2
HISTORY_LIMIT = 100


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def atomic_write_json(path: str | Path, payload: dict | list) -> None:
    """Replace JSON only after a complete, flushed file exists alongside it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=destination.parent, prefix=f".{destination.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2,
                      default=_json_default, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_session_summary(path: str | Path) -> dict:
    """Read an existing export directly; this never opens/reprocesses a video."""
    summary_path = Path(path)
    if summary_path.is_dir():
        summary_path = summary_path / "summary.json"
    with summary_path.open("r", encoding="utf-8-sig") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError("Session summary must be a JSON object")
    return result


def _index_entry(summary: dict, session_dir: Path) -> dict:
    return {
        "session_id": summary.get("session_id", session_dir.name),
        "source": summary.get("source", ""),
        "started_at_utc": summary.get("started_at_utc", ""),
        "finished_at_utc": summary.get("finished_at_utc", ""),
        "status": summary.get("status", "unknown"),
        "session_dir": str(session_dir.resolve()),
        "summary_path": str((session_dir / "summary.json").resolve()),
        "analyzed_frames": summary.get("analyzed_frames", 0),
        "cumulative_unique_count": summary.get("cumulative_unique_count", 0),
        "line_counts": summary.get("line_counts", {"up": 0, "down": 0}),
        "peak_current_count": summary.get("peak_current_count", 0),
    }


def list_history(root: str | Path, limit: int = HISTORY_LIMIT) -> list[dict]:
    """Return up to 100 lightweight summaries, newest first.

    A damaged/missing history index is rebuilt from completed summary files.
    The CSV exports and original videos are never loaded for history display.
    """
    if not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    limit = min(limit, HISTORY_LIMIT)
    root_path = Path(root).resolve()
    index_path = root_path / "history.json"
    entries: list[dict] = []
    try:
        with index_path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        candidates = payload.get("sessions", []) if isinstance(payload, dict) else payload
        if not isinstance(candidates, list):
            raise ValueError("Invalid history index")
        for candidate in candidates:
            if not isinstance(candidate, dict) or not candidate.get("summary_path"):
                continue
            summary_path = Path(candidate["summary_path"]).resolve()
            if not summary_path.is_relative_to(root_path) or not summary_path.is_file():
                continue
            try:
                summary = read_session_summary(summary_path)
            except (OSError, ValueError, TypeError):
                continue
            entries.append(_index_entry(summary, summary_path.parent))
    except (OSError, ValueError, TypeError):
        for summary_path in root_path.glob("*/summary.json"):
            try:
                entries.append(_index_entry(read_session_summary(summary_path), summary_path.parent))
            except (OSError, ValueError, TypeError):
                continue

    entries.sort(key=lambda item: (item.get("started_at_utc", ""), item["session_id"]), reverse=True)
    # Also de-duplicate in case an older app wrote duplicate index entries.
    unique = {item["session_id"]: item for item in entries}
    return list(unique.values())[:limit]


class SessionRecorder:
    """One source session with streamed CSVs and atomic metadata.

    Call finalize on completion, cancellation, or inference error. Repeated
    finalization is harmless. Each instance reserves a unique directory, even
    if several sessions start in the same second.
    """

    FRAME_FIELDS = (
        "frame_index", "source_time_seconds", "width", "height", "processing_ms",
        "current_count", "tracked_current_count", "cumulative_unique_count",
        "counts_by_class_json", "cumulative_by_class_json", "line_up", "line_down",
        "roi_current_count", "dwell_current_count", "dwell_seconds_by_track_json",
        "conf", "iou", "imgsz", "density_threshold",
    )
    FRAME_PARAMETER_FIELDS = ("conf", "iou", "imgsz", "density_threshold")
    DETECTION_FIELDS = (
        "frame_index", "source_time_seconds", "track_id", "class_id", "class_name",
        "confidence", "x1", "y1", "x2", "y2", "bottom_center_x_normalized",
        "bottom_center_y_normalized",
    )
    EVENT_FIELDS = (
        "frame_index", "source_time_seconds", "event_type", "track_id", "class_name",
        "direction", "duration_seconds", "x_normalized", "y_normalized",
    )

    def __init__(
        self, root: str | Path, source: str | int | Path,
        config: AnalyticsConfig | dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.started_at_utc = _utc_now()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.session_id = f"session_{stamp}_{uuid.uuid4().hex[:12]}"
        self.session_dir = self.root / self.session_id
        self.session_dir.mkdir(exist_ok=False)
        self.source = str(source)
        self.config = config.to_dict() if isinstance(config, AnalyticsConfig) else (config or {})
        self.metadata = dict(metadata or {})
        self._handles: list = []
        self._writers: dict = {}
        self._finalized = False
        self._recorded_frames = 0
        self._last_frame_index: int | None = None
        self._last_source_timestamp: float | None = None
        self._last_flush_time = time.monotonic()
        self._summary: dict | None = None
        try:
            for name, fields in (
                ("frames", self.FRAME_FIELDS),
                ("detections", self.DETECTION_FIELDS),
                ("events", self.EVENT_FIELDS),
            ):
                handle = (self.session_dir / f"{name}.csv").open(
                    "w", newline="", encoding="utf-8-sig"
                )
                self._handles.append(handle)
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                self._writers[name] = writer
            atomic_write_json(self.session_dir / "session.json", self._session_payload("recording"))
            self.flush()
        except Exception:
            self.close()
            raise

    @property
    def summary_path(self) -> Path:
        return self.session_dir / "summary.json"

    def _session_payload(self, status: str) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "started_at_utc": self.started_at_utc,
            "source": self.source,
            "status": status,
            "config": self.config,
            "metadata": self.metadata,
            "timestamp_unit": "seconds in source timeline",
            "coordinate_unit": "xyxy pixels; event/ROI/line positions normalized to frame size",
            "frame_parameter_notes": (
                "Per-frame model-call parameters; imgsz is the requested size. "
                "Blank parameter cells mean the caller did not supply a snapshot."
            ),
            "files": {key: f"{key}.csv" for key in ("frames", "detections", "events")},
        }

    def record_frame(
        self, stats: FrameStats, detections: Iterable[Detection],
        width: int, height: int, processing_ms: float | None = None,
        *, inference_params: Mapping[str, int | float] | None = None,
    ) -> None:
        """Append results and the parameters used for this exact frame.

        ``inference_params`` is optional for existing recorder callers. Values
        must come from the frame's immutable inference snapshot, rather than
        reading the GUI's possibly newer configuration after inference.
        """
        if self._finalized or not self._handles or any(handle.closed for handle in self._handles):
            raise RuntimeError("Session recorder is closed")
        if width <= 0 or height <= 0:
            raise ValueError("Frame dimensions must be positive")
        if self._last_frame_index is not None and stats.frame_index <= self._last_frame_index:
            raise ValueError("Recorded frame_index must increase")
        if self._last_source_timestamp is not None and stats.timestamp < self._last_source_timestamp:
            raise ValueError("Recorded source timestamp must not move backwards")
        parameters = dict(inference_params or {})
        unknown = parameters.keys() - set(self.FRAME_PARAMETER_FIELDS)
        if unknown:
            raise ValueError(f"Unknown frame parameters: {', '.join(sorted(unknown))}")
        encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        self._writers["frames"].writerow({
            "frame_index": stats.frame_index, "source_time_seconds": stats.timestamp,
            "width": width, "height": height,
            "processing_ms": "" if processing_ms is None else processing_ms,
            "current_count": stats.current_count,
            "tracked_current_count": stats.tracked_current_count,
            "cumulative_unique_count": stats.cumulative_unique_count,
            "counts_by_class_json": encode(stats.counts_by_class),
            "cumulative_by_class_json": encode(stats.cumulative_by_class),
            "line_up": stats.line_counts.get("up", 0),
            "line_down": stats.line_counts.get("down", 0),
            "roi_current_count": stats.roi_current_count,
            "dwell_current_count": stats.dwell_current_count,
            "dwell_seconds_by_track_json": encode(stats.dwell_seconds_by_track),
            **{name: parameters.get(name, "") for name in self.FRAME_PARAMETER_FIELDS},
        })
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            self._writers["detections"].writerow({
                "frame_index": stats.frame_index, "source_time_seconds": stats.timestamp,
                "track_id": "" if detection.track_id is None else detection.track_id,
                "class_id": detection.class_id, "class_name": detection.class_name,
                "confidence": detection.confidence,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "bottom_center_x_normalized": ((x1 + x2) / 2) / width,
                "bottom_center_y_normalized": y2 / height,
            })
        for event in stats.events:
            self._writers["events"].writerow({
                "frame_index": event.frame_index, "source_time_seconds": event.timestamp,
                "event_type": event.event_type, "track_id": event.track_id,
                "class_name": event.class_name, "direction": event.direction or "",
                "duration_seconds": "" if event.duration_seconds is None else event.duration_seconds,
                "x_normalized": event.x, "y_normalized": event.y,
            })
        self._recorded_frames += 1
        self._last_frame_index = stats.frame_index
        self._last_source_timestamp = stats.timestamp
        # Keep inference responsive while limiting the amount of buffered data.
        if self._recorded_frames % 30 == 0 or time.monotonic() - self._last_flush_time >= 1:
            self.flush()

    def flush(self) -> None:
        for handle in self._handles:
            if not handle.closed:
                handle.flush()
        self._last_flush_time = time.monotonic()

    def close(self) -> None:
        """Release files; finalize is required to publish a completed summary."""
        for handle in self._handles:
            if not handle.closed:
                handle.flush()
                handle.close()

    def finalize(self, summary: dict, metadata: dict | None = None) -> Path:
        if self._finalized:
            return self.session_dir
        self.close()
        combined_metadata = {**self.metadata, **(metadata or {})}
        status = str(combined_metadata.pop("status", summary.get("status", "completed")))
        finished_at = _utc_now()
        self.metadata = combined_metadata
        self._summary = {
            **summary,
            **self._session_payload(status),
            "finished_at_utc": finished_at,
            "recorded_frames": self._recorded_frames,
            "session_dir": str(self.session_dir),
        }
        atomic_write_json(self.summary_path, self._summary)
        session_payload = self._session_payload(status)
        session_payload["finished_at_utc"] = finished_at
        session_payload["summary_file"] = "summary.json"
        atomic_write_json(self.session_dir / "session.json", session_payload)
        entries = list_history(self.root)
        entries = [entry for entry in entries if entry["session_id"] != self.session_id]
        entries.insert(0, _index_entry(self._summary, self.session_dir))
        atomic_write_json(self.root / "history.json", {
            "schema_version": SCHEMA_VERSION, "sessions": entries[:HISTORY_LIMIT],
        })
        self._finalized = True
        return self.session_dir

    def __enter__(self) -> SessionRecorder:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False
