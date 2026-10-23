import csv
import json
from pathlib import Path
import tempfile
import unittest

from traffic_app.analytics import AnalyticsConfig, Detection, TrafficAnalytics
from traffic_app.storage import SessionRecorder, atomic_write_json, list_history, read_session_summary


class SessionStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_export_has_source_timestamps_events_unicode_and_bom(self):
        config = AnalyticsConfig(roi=(0, 0, 1, 1), dwell_threshold_seconds=2)
        analytics = TrafficAnalytics(config)
        recorder = SessionRecorder(self.root, "测试视频.mp4", config)
        for frame, timestamp, y in [(0, 10, 40), (1, 12, 60)]:
            items = [Detection(0, "小汽车", .95, (10, 10, 30, y), 1)]
            stats = analytics.update(items, timestamp, frame, 100, 100)
            recorder.record_frame(stats, items, 100, 100, processing_ms=2.5)
        session_path = recorder.finalize(analytics.summary(), {"status": "stopped"})
        self.assertEqual(session_path, recorder.session_dir)
        for name in ("frames", "detections", "events"):
            self.assertTrue((session_path / f"{name}.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
        with (session_path / "frames.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["source_time_seconds"] for row in rows], ["10", "12"])
        # Original record_frame callers remain valid; missing parameters are
        # explicitly unknown rather than invented from the current config.
        for row in rows:
            self.assertTrue(all(row[name] == "" for name in SessionRecorder.FRAME_PARAMETER_FIELDS))
        self.assertEqual(json.loads(rows[1]["counts_by_class_json"]), {"小汽车": 1})
        with (session_path / "events.csv").open(encoding="utf-8-sig", newline="") as handle:
            events = list(csv.DictReader(handle))
        self.assertEqual({item["event_type"] for item in events}, {"line_crossing", "roi_dwell"})
        summary = read_session_summary(session_path)
        self.assertEqual(summary["status"], "stopped")
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["source_duration_seconds"], 2)
        self.assertEqual(summary["recorded_frames"], 2)
        session = json.loads((session_path / "session.json").read_text(encoding="utf-8"))
        self.assertEqual(session["status"], "stopped")
        self.assertEqual(list_history(self.root)[0]["cumulative_unique_count"], 1)

    def test_empty_and_repeated_finalize_and_unique_directories(self):
        first = SessionRecorder(self.root, 0)
        second = SessionRecorder(self.root, 0)
        self.assertNotEqual(first.session_dir, second.session_dir)
        first.finalize(TrafficAnalytics().summary())
        summary_bytes = first.summary_path.read_bytes()
        first.finalize({"changed": True})
        self.assertEqual(summary_bytes, first.summary_path.read_bytes())
        second.finalize(TrafficAnalytics().summary())
        self.assertEqual(len(list_history(self.root)), 2)
        with self.assertRaises(RuntimeError):
            first.record_frame(TrafficAnalytics().update([], 0, 0, 10, 10), [], 10, 10)

    def test_corrupt_history_recovers_from_summaries_without_video(self):
        recorder = SessionRecorder(self.root, "does_not_exist.mp4")
        recorder.finalize(TrafficAnalytics().summary())
        (self.root / "history.json").write_text("{broken", encoding="utf-8")
        restored = list_history(self.root)
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0]["source"], "does_not_exist.mp4")

    def test_engine_failure_status_is_preserved_in_saved_summary_and_history(self):
        recorder = SessionRecorder(self.root, "broken.mp4")
        recorder.finalize({"status": "failed", "error": "decode error"})
        self.assertEqual(read_session_summary(recorder.session_dir)["status"], "failed")
        self.assertEqual(list_history(self.root)[0]["status"], "failed")

    def test_history_is_bounded_and_atomic_when_more_than_100_sessions_exist(self):
        for index in range(105):
            directory = self.root / f"old_{index:03d}"
            atomic_write_json(directory / "summary.json", {
                "session_id": directory.name, "started_at_utc": f"2026-10-{index:03d}",
                "cumulative_unique_count": index,
            })
        self.assertEqual(len(list_history(self.root)), 100)
        recorder = SessionRecorder(self.root, "new.mp4")
        recorder.finalize(TrafficAnalytics().summary())
        index = json.loads((self.root / "history.json").read_text(encoding="utf-8"))
        self.assertEqual(len(index["sessions"]), 100)
        self.assertEqual(len(list_history(self.root, 200)), 100)
        self.assertEqual(list_history(self.root, 0), [])
        self.assertFalse(list(self.root.rglob("*.tmp")))

    def test_failed_json_write_preserves_previous_file_and_cleans_temp(self):
        path = self.root / "test.json"
        atomic_write_json(path, {"original": "保留"})
        with self.assertRaises(ValueError):
            atomic_write_json(path, {"invalid": float("nan")})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"original": "保留"})
        self.assertFalse(list(self.root.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
