"""Thread, cancellation, timing, and lossless-export regression checks."""

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication

from traffic_app import __version__
from traffic_app.engine import DetectionWorker, EngineConfig, _file_sha256, _video_timestamp


class _Array:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _Boxes:
    xyxy = _Array([[5, 4, 15, 12]])
    cls = _Array([0])
    conf = _Array([0.9])

    def __init__(self, tracked):
        self.id = _Array([1]) if tracked else None

    def __len__(self):
        return 1


class _Result:
    names = {0: "car"}

    def __init__(self, tracked):
        self.boxes = _Boxes(tracked)


class _Model:
    def __init__(self, fail_at=None, delay=0):
        self.thread_ids = []
        self.calls = []
        self.fail_at = fail_at
        self.delay = delay
        self.on_track = None
        self.on_predict = None

    def predict(self, frame, **options):
        self.thread_ids.append(threading.get_ident())
        self.calls.append(("predict", options))
        if self.on_predict:
            self.on_predict()
        return [_Result(False)]

    def track(self, frame, **options):
        self.thread_ids.append(threading.get_ident())
        self.calls.append(("track", options))
        if self.on_track:
            self.on_track()
        if self.fail_at and sum(name == "track" for name, _ in self.calls) == self.fail_at:
            raise RuntimeError("synthetic inference failure")
        if self.delay:
            time.sleep(self.delay)
        return [_Result(True)]


class _Capture:
    def __init__(self, count=12, fps=10):
        self.count = count
        self.fps = fps
        self.position = 0
        self.released = False
        self.settings = []

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.count
        if prop == cv2.CAP_PROP_POS_MSEC:
            return max(0, self.position - 1) / self.fps * 1000
        return 0

    def set(self, prop, value):
        self.settings.append((prop, value))
        return True

    def read(self):
        if self.position >= self.count:
            return False, None
        self.position += 1
        return True, np.zeros((32, 48, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="街巷测试_")
        self.root = Path(self.temporary.name)
        self.model_path = self.root / "模型.pt"
        self.model_path.touch()

    def tearDown(self):
        self.temporary.cleanup()

    def worker(self, model=None, capture=None, **overrides):
        self.model = model or _Model()
        self.capture = capture or _Capture()
        self.creation_threads = []
        self.sources = []
        model_instance = self.model
        capture_instance = self.capture
        creation_threads = self.creation_threads
        sources = self.sources

        def create_model(path):
            creation_threads.append(threading.get_ident())
            return model_instance

        def create_capture(source):
            sources.append(source)
            return capture_instance

        config = EngineConfig(source="中文路径视频.mp4", model_path=str(self.model_path),
                              record_dir=str(self.root / "结果"), **overrides)
        worker = DetectionWorker(config, model_factory=create_model, capture_factory=create_capture)
        worker._select_device = lambda _: "cpu"
        return worker

    def run_worker(self, worker):
        worker.start()
        self.assertTrue(worker.wait(5000), "worker failed to terminate")
        self.app.processEvents()
        return worker.summary

    def exported(self, worker):
        session_dir = Path(worker.summary["session_dir"])
        return json.loads((session_dir / "summary.json").read_text(encoding="utf-8"))

    def frame_rows(self, worker):
        with (Path(worker.summary["session_dir"]) / "frames.csv").open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_all_frames_recorded_with_bounded_preview_and_source_timing(self):
        worker = self.worker(display_fps=1)
        summary = self.run_worker(worker)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["analyzed_frames"], 12)
        self.assertEqual(sum(name == "track" for name, _ in self.model.calls), 12)
        self.assertEqual(summary["cumulative_unique_count"], 1)
        self.assertTrue(self.capture.released)
        packet = worker.latest_packet()
        self.assertEqual(packet.frame_index, 12)
        self.assertAlmostEqual(packet.source_time_sec, 1.1)
        self.assertIsNone(worker.latest_packet())
        with (Path(summary["session_dir"]) / "frames.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 12)
        self.assertEqual([float(row["source_time_seconds"]) for row in rows], [n / 10 for n in range(12)])
        self.assertEqual(self.exported(worker)["status"], "completed")

    def test_model_creation_warmup_and_inference_are_off_main_thread_and_cpu_full_precision(self):
        main_thread = threading.get_ident()
        model_bytes = b"Model provenance\x00" * 100000
        self.model_path.write_bytes(model_bytes)
        worker = self.worker(max_frames=2, half=True)
        hash_threads = []

        def hash_on_worker(path):
            hash_threads.append(threading.get_ident())
            return _file_sha256(path)

        with patch("traffic_app.engine._file_sha256", side_effect=hash_on_worker):
            self.run_worker(worker)
        self.assertTrue(self.creation_threads)
        self.assertEqual(len(hash_threads), 1)
        self.assertTrue(all(value != main_thread for value in self.creation_threads + self.model.thread_ids + hash_threads))
        self.assertTrue(all(options["half"] is False for _, options in self.model.calls))
        self.assertEqual(self.model.calls[0][0], "predict")
        exported = self.exported(worker)
        self.assertEqual(exported["status"], "limited")
        self.assertEqual(exported["inference_runtime"], {"device": "cpu", "half": False})
        self.assertEqual(exported["metadata"]["inference_runtime"], exported["inference_runtime"])
        expected_hash = hashlib.sha256(model_bytes).hexdigest()
        self.assertEqual(exported["model_sha256"], expected_hash)
        self.assertEqual(exported["metadata"]["model_sha256"], expected_hash)
        self.assertEqual(exported["app_version"], __version__)
        session_json = json.loads((Path(worker.summary["session_dir"]) / "session.json").read_text(encoding="utf-8"))
        self.assertEqual(session_json["metadata"]["model_sha256"], expected_hash)
        self.assertEqual(session_json["metadata"]["app_version"], __version__)

    def test_hot_parameters_are_snapshotted_at_frame_boundaries(self):
        model = _Model()
        worker = self.worker(model=model, max_frames=3)
        switched = False

        def switch():
            nonlocal switched
            if not switched:
                worker.update_config(conf=.7, iou=.6, imgsz=320, density_threshold=2)
                switched = True

        model.on_track = switch
        self.run_worker(worker)
        calls = [options for name, options in model.calls if name == "track"]
        self.assertEqual(calls[0]["conf"], .5)
        self.assertEqual(calls[1]["conf"], .7)
        self.assertEqual(calls[1]["imgsz"], 320)
        rows = self.frame_rows(worker)
        self.assertEqual(len(rows), 3)
        for row, options in zip(rows, calls):
            self.assertEqual(float(row["conf"]), options["conf"])
            self.assertEqual(float(row["iou"]), options["iou"])
            self.assertEqual(int(row["imgsz"]), options["imgsz"])
        self.assertEqual([int(row["density_threshold"]) for row in rows], [10, 2, 2])
        self.assertEqual([float(row["source_time_seconds"]) for row in rows], [0, .1, .2])
        self.assertEqual([int(row["frame_index"]) for row in rows], [1, 2, 3])
        with self.assertRaises(ValueError):
            worker.update_config(source=3)

    def test_failed_frame_does_not_fabricate_a_successful_parameter_snapshot(self):
        model = _Model(fail_at=3)
        worker = self.worker(model=model)
        frame = 0

        def switch():
            nonlocal frame
            frame += 1
            if frame == 1:
                worker.update_config(conf=.7, iou=.6, imgsz=320, density_threshold=2)
            elif frame == 2:
                worker.update_config(conf=.9, iou=.8, imgsz=960, density_threshold=3)

        model.on_track = switch
        summary = self.run_worker(worker)
        self.assertEqual(summary["status"], "failed")
        rows = self.frame_rows(worker)
        self.assertEqual([float(row["conf"]) for row in rows], [.5, .7])
        self.assertEqual([int(row["density_threshold"]) for row in rows], [10, 2])
        exported = self.exported(worker)
        self.assertEqual(exported["recorded_frames"], 2)
        self.assertEqual(exported["config_final"]["conf"], .9)
        self.assertEqual(exported["status"], "failed")

    def test_stop_keeps_the_last_inflight_frame_snapshot_even_after_a_hot_update(self):
        model = _Model()
        worker = self.worker(model=model)
        frame = 0

        def switch_and_stop():
            nonlocal frame
            frame += 1
            if frame == 1:
                worker.update_config(conf=.7, iou=.6, imgsz=320, density_threshold=2)
            elif frame == 3:
                worker.update_config(conf=.9, density_threshold=4)
                worker.stop()

        model.on_track = switch_and_stop
        summary = self.run_worker(worker)
        self.assertEqual(summary["status"], "stopped")
        rows = self.frame_rows(worker)
        self.assertEqual([float(row["conf"]) for row in rows], [.5, .7, .7])
        self.assertEqual([int(row["density_threshold"]) for row in rows], [10, 2, 2])
        self.assertEqual(self.exported(worker)["recorded_frames"], 3)
        self.assertEqual(self.exported(worker)["status"], "stopped")
        self.assertEqual(summary["config_final"]["conf"], .9)

    def test_inference_failure_still_releases_and_finalizes_partial_results(self):
        worker = self.worker(model=_Model(fail_at=3))
        summary = self.run_worker(worker)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["processed_frames"], 2)
        self.assertIn("synthetic inference failure", summary["error"])
        self.assertTrue(self.capture.released)
        self.assertEqual(self.exported(worker)["status"], "failed")

    def test_model_load_failure_keeps_its_hash_and_version_in_failed_export(self):
        self.model_path.write_bytes(b"unloadable local weight")
        worker = self.worker()

        def cannot_load(path):
            raise RuntimeError("synthetic model load failure")

        worker._model_factory = cannot_load
        summary = self.run_worker(worker)
        expected_hash = hashlib.sha256(b"unloadable local weight").hexdigest()
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["processed_frames"], 0)
        self.assertEqual(summary["model_sha256"], expected_hash)
        self.assertEqual(self.exported(worker)["metadata"]["model_sha256"], expected_hash)
        self.assertEqual(self.exported(worker)["app_version"], __version__)

    def test_stop_preserves_partial_exports_without_success_status(self):
        model = _Model(delay=.01)
        worker = self.worker(model=model, capture=_Capture(count=1000))
        calls = 0

        def stop_on_third():
            nonlocal calls
            calls += 1
            if calls == 3:
                worker.stop()

        model.on_track = stop_on_third
        summary = self.run_worker(worker)
        self.assertEqual(summary["status"], "stopped")
        self.assertEqual(summary["analyzed_frames"], 3)
        self.assertTrue(self.capture.released)
        self.assertEqual(self.exported(worker)["status"], "stopped")

    def test_real_camera_index_is_not_coerced_to_string(self):
        worker = self.worker(max_frames=1)
        worker._config = replace(worker.config, source=3)
        self.run_worker(worker)
        self.assertEqual(self.sources, [3])
        self.assertIn((cv2.CAP_PROP_BUFFERSIZE, 1), self.capture.settings)

    def test_unicode_image_decode_does_not_invent_track_ids(self):
        image_path = self.root / "测试图像.png"
        _, encoded = cv2.imencode(".png", np.zeros((32, 48, 3), dtype=np.uint8))
        encoded.tofile(str(image_path))
        worker = self.worker()
        worker._config = replace(worker.config, source=str(image_path))
        summary = self.run_worker(worker)
        self.assertEqual(summary["analyzed_frames"], 1)
        self.assertEqual(summary["cumulative_unique_count"], 0)
        self.assertEqual([name for name, _ in self.model.calls], ["predict", "predict"])

    def test_unicode_video_is_decoded_sequentially_with_real_opencv(self):
        video_path = self.root / "中文 录像.avi"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (48, 32))
        self.assertTrue(writer.isOpened(), "test codec unavailable")
        try:
            for value in range(7):
                writer.write(np.full((32, 48, 3), value * 20, dtype=np.uint8))
        finally:
            writer.release()
        worker = self.worker()
        worker._config = replace(worker.config, source=str(video_path))
        worker._capture_factory = None
        summary = self.run_worker(worker)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["analyzed_frames"], 7)
        self.assertEqual(summary["source_frame_count"], 7)
        self.assertEqual(summary["source_fps"], 5)
        self.assertAlmostEqual(worker.latest_packet().source_time_sec, 1.2)

    def test_independent_sessions_each_construct_model_and_reset_track_counts(self):
        workers = [self.worker(max_frames=1), self.worker(max_frames=1)]
        for worker in workers:
            summary = self.run_worker(worker)
            self.assertEqual(summary["cumulative_unique_count"], 1)
        self.assertNotEqual(workers[0].summary["session_dir"], workers[1].summary["session_dir"])

    def test_timestamp_fallback_uses_source_fps_and_never_moves_backwards(self):
        capture = _Capture()
        capture.position = 1
        self.assertAlmostEqual(_video_timestamp(capture, 4, 10, .2), .3)
        self.assertEqual(_video_timestamp(capture, 4, 10, .8), .8)

    def test_completed_worker_cannot_restart_or_overwrite_previous_summary(self):
        worker = self.worker(max_frames=1)
        original = self.run_worker(worker)
        with self.assertRaises(RuntimeError):
            worker.start()
        self.assertEqual(worker.summary, original)
        self.assertEqual(self.exported(worker)["status"], "limited")

    def test_stop_during_model_creation_skips_warmup_and_persists_stopped_session(self):
        model = _Model()
        worker = self.worker(model=model)

        def model_factory(path):
            worker.stop()
            return model

        worker._model_factory = model_factory
        summary = self.run_worker(worker)
        self.assertEqual(summary["status"], "stopped")
        self.assertEqual(model.calls, [])
        self.assertEqual(summary["analyzed_frames"], 0)
        self.assertEqual(self.exported(worker)["status"], "stopped")

    def test_stop_during_warmup_skips_input_and_all_source_inference(self):
        model = _Model()
        worker = self.worker(model=model)
        model.on_predict = worker.stop
        summary = self.run_worker(worker)
        self.assertEqual(summary["status"], "stopped")
        self.assertEqual([name for name, _ in model.calls], ["predict"])
        self.assertEqual(self.sources, [])
        self.assertEqual(summary["analyzed_frames"], 0)

    def test_stop_during_read_does_not_start_another_inference(self):
        worker = self.worker()
        original_read = self.capture.read

        def stopped_read():
            worker.stop()
            return original_read()

        self.capture.read = stopped_read
        summary = self.run_worker(worker)
        self.assertEqual(summary["status"], "stopped")
        self.assertEqual(summary["analyzed_frames"], 0)
        self.assertEqual([name for name, _ in self.model.calls], ["predict"])
        self.assertTrue(self.capture.released)

    def test_stopped_session_does_not_prevent_a_new_session(self):
        worker = self.worker()
        worker.stop()
        self.assertEqual(self.run_worker(worker)["status"], "stopped")
        next_worker = self.worker(max_frames=1)
        summary = self.run_worker(next_worker)
        self.assertEqual(summary["status"], "limited")
        self.assertEqual(summary["cumulative_unique_count"], 1)


if __name__ == "__main__":
    unittest.main()
