"""Offscreen UI lifecycle checks without loading a model or opening a camera."""

from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QObject, QPoint, Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QPlainTextEdit, QPushButton

from traffic_app.analytics import FrameStats
from traffic_app.engine import FramePacket
from traffic_app import ui


# Discovery imports all test modules before running their setUpClass methods.
# Establish QApplication now so engine tests can share it via instance().
_APP = QApplication.instance() or QApplication([])
_APP.setQuitOnLastWindowClosed(False)


class FakeWorker(QObject):
    status = Signal(str)
    error = Signal(str)
    session_finished = Signal(dict)
    finished = Signal()
    created = []

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.running = False
        self.stop_calls = 0
        self.deleted = False
        self.packet = None
        self.updates = []
        self.created.append(self)

    def start(self):
        self.running = True

    def isRunning(self):
        return self.running

    def stop(self):
        self.stop_calls += 1

    def latest_packet(self):
        packet, self.packet = self.packet, None
        return packet

    def update_config(self, **changes):
        self.updates.append(changes)

    def deleteLater(self):
        # Keep it available for deliberately delayed old-session signal tests.
        self.deleted = True

    def finish(self, status="completed"):
        summary = {"status": status, "processed_frames": 3, "device": "cpu"}
        self.session_finished.emit(summary)
        self.running = False
        self.finished.emit()


class FakePreview(QObject):
    ready = Signal(object)
    failure = Signal(str)
    finished = Signal()
    created = []

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.source = source
        self.running = False
        self.interrupted = False
        self.deleted = False
        self.created.append(self)

    def start(self):
        self.running = True

    def isRunning(self):
        return self.running

    def requestInterruption(self):
        self.interrupted = True

    def wait(self, *args):
        raise AssertionError("UI must never wait synchronously for a slow preview")

    def deleteLater(self):
        self.deleted = True

    def finish(self):
        self.running = False
        self.finished.emit()


class UiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="街巷界面测试_")
        self.root = Path(self.temporary.name)
        self.worker_patch = patch.object(ui, "DetectionWorker", FakeWorker)
        self.preview_patch = patch.object(ui, "PreviewLoader", FakePreview)
        self.worker_patch.start()
        self.preview_patch.start()
        FakeWorker.created = []
        FakePreview.created = []
        self.window = ui.MainWindow(self.root)
        self.window.timer.stop()
        self.window.model_path = self.root / "model.pt"
        self.window.model_path.touch()
        self.window.show()
        _APP.processEvents()

    def tearDown(self):
        for preview in list(self.window._previews):
            if preview.running:
                preview.finish()
        if self.window.worker and self.window.worker.running:
            self.window.worker.finish("stopped")
        _APP.processEvents()
        self.window.close()
        self.window.deleteLater()
        _APP.processEvents()
        self.worker_patch.stop()
        self.preview_patch.stop()
        self.temporary.cleanup()

    def start_camera(self, index=0):
        self.window.set_source(index)
        self.window.start_analysis()
        return self.window.worker

    def test_window_starts_with_safe_idle_controls(self):
        self.assertTrue(self.window.isVisible())
        self.assertTrue(self.window.start_btn.isEnabled())
        self.assertFalse(self.window.stop_btn.isEnabled())
        self.assertIsNone(self.window.worker)

    def test_person_mode_switches_labels_and_model(self):
        # Switch away first, so the test does not depend on which mode is default.
        self.window.mode.setCurrentIndex(self.window.mode.findData("vehicle"))
        index = self.window.mode.findData("person")
        self.window.mode.setCurrentIndex(index)
        self.assertEqual(self.window.metric_labels["current"].text(), "画面内人/车")
        self.assertIn("人/车", self.window.chart.caption)
        self.assertEqual(self.window.model_path.name, "yolo11s.pt")
        self.assertEqual(self.window.imgsz.currentData(), 1024)
        index = self.window.mode.findData("vehicle")
        self.window.mode.setCurrentIndex(index)
        self.assertEqual(self.window.metric_labels["current"].text(), "画面内车辆")
        self.assertEqual(self.window.model_path.name, "best.pt")

    def test_space_portrait_is_wired_into_the_workflow(self):
        self.assertEqual(self.window.mode.currentData(), "person")
        self.assertFalse(self.window.portrait_btn.isEnabled())
        self.assertIsNone(self.window.last_session_dir)
        # The observer's box is reused as the heat-map mask, so colour never
        # spills onto facades.
        self.window.set_source(0)
        self.window.viewer.roi = (0.2, 0.4, 0.8, 0.9)
        self.window.start_analysis()
        self.assertEqual(self.window.analysis_mask,
                         [(0.2, 0.4), (0.8, 0.4), (0.8, 0.9), (0.2, 0.9)])

    def test_calibration_starts_with_clean_points(self):
        import numpy as np
        self.window.viewer.set_frame(np.zeros((120, 200, 3), dtype=np.uint8))
        self.window.viewer.calibration_points = [(0.1, 0.1)]
        self.window.start_calibration()
        self.assertTrue(self.window.viewer.picking_calibration)
        self.assertEqual(self.window.viewer.calibration_points, [])
        self.window.clear_calibration()
        self.assertFalse(self.window.viewer.picking_calibration)

    def test_source_and_missing_model_errors_do_not_create_a_worker(self):
        self.window.start_analysis()
        self.assertIn("请先选择", self.window.status_label.text())
        self.window.set_source(0)
        self.window.model_path = self.root / "missing.pt"
        self.window.start_analysis()
        self.assertIn("未找到模型", self.window.status_label.text())
        self.assertEqual(FakeWorker.created, [])
        self.assertTrue(self.window.start_btn.isEnabled())

    def test_bad_capture_start_does_not_lock_the_window(self):
        self.window.set_source(0)
        self.window.capture_start.setText("invalid date")
        self.window.start_analysis()
        self.assertIn("无法开始", self.window.status_label.text())
        self.assertEqual(FakeWorker.created, [])
        self.assertTrue(self.window.start_btn.isEnabled())

    def test_start_preserves_camera_index_and_locks_only_session_parameters(self):
        worker = self.start_camera(3)
        self.assertEqual(worker.config.source, 3)
        self.assertFalse(self.window.start_btn.isEnabled())
        self.assertTrue(self.window.stop_btn.isEnabled())
        self.assertTrue(all(not control.isEnabled() for control in self.window.locked_controls))
        self.assertTrue(all(control.isEnabled() for control in (
            self.window.conf, self.window.iou, self.window.imgsz, self.window.threshold)))
        self.window.conf.setValue(.7)
        self.assertEqual(worker.updates[-1]["conf"], .7)
        self.window.start_analysis()
        self.assertEqual(len(FakeWorker.created), 1)
        self.window.set_source(7)
        self.assertEqual(self.window.source, 3)

    def test_switching_sources_is_nonblocking_and_old_preview_cannot_overwrite(self):
        self.window.set_source("slow first.mp4")
        first = self.window.preview_worker
        began = time.perf_counter()
        self.window.set_source("second.mp4")
        self.assertLess(time.perf_counter() - began, .5)
        second = self.window.preview_worker
        self.assertTrue(first.interrupted)
        self.assertIn(first, self.window._previews)
        with patch.object(self.window.viewer, "set_frame") as display:
            image = np.zeros((32, 48, 3), dtype=np.uint8)
            first.ready.emit(image)
            self.assertEqual(display.call_count, 0)
            second.ready.emit(image)
            self.assertEqual(display.call_count, 1)
            self.window.set_source(2)
            second.ready.emit(image)
            self.assertEqual(display.call_count, 1)
        first.failure.emit("stale failure")
        self.assertNotIn("stale failure", self.window.status_label.text())

    def test_preview_cannot_overwrite_an_active_analysis(self):
        self.window.set_source("first.mp4")
        preview = self.window.preview_worker
        self.window.start_analysis()
        self.assertTrue(preview.interrupted)
        with patch.object(self.window.viewer, "set_frame") as display:
            preview.ready.emit(np.zeros((32, 48, 3), dtype=np.uint8))
            self.assertEqual(display.call_count, 0)

    def test_roi_drag_is_normalized_and_start_clears_partial_drag(self):
        self.window.viewer.set_frame(np.zeros((200, 400, 3), dtype=np.uint8))
        self.window.select_roi()
        rect = self.window.viewer.frame_rect()
        start = QPoint(round(rect.left()+rect.width()*.2), round(rect.top()+rect.height()*.25))
        end = QPoint(round(rect.left()+rect.width()*.8), round(rect.top()+rect.height()*.75))
        QTest.mousePress(self.window.viewer, Qt.LeftButton, pos=start)
        QTest.mouseMove(self.window.viewer, end)
        QTest.mouseRelease(self.window.viewer, Qt.LeftButton, pos=end)
        for observed, expected in zip(self.window.viewer.roi, (.2, .25, .8, .75)):
            self.assertAlmostEqual(observed, expected, delta=.01)
        self.window.select_roi()
        QTest.mousePress(self.window.viewer, Qt.LeftButton, pos=start)
        self.assertIsNotNone(self.window.viewer._start)
        self.window.source = 0
        self.window.start_analysis()
        self.assertIsNone(self.window.viewer._start)
        self.assertIsNone(self.window.viewer._end)
        self.assertFalse(self.window.viewer.selecting)
        self.assertEqual(self.window.worker.config.analytics_config.roi, self.window.viewer.roi)

    def test_stop_does_not_unlock_controls_until_thread_finishes(self):
        worker = self.start_camera()
        self.window.stop_analysis()
        self.assertEqual(worker.stop_calls, 1)
        self.assertFalse(self.window.stop_btn.isEnabled())
        self.assertFalse(self.window.start_btn.isEnabled())
        self.assertTrue(worker.isRunning())
        worker.finish("stopped")
        self.assertTrue(self.window.start_btn.isEnabled())
        self.assertFalse(self.window.stop_btn.isEnabled())
        self.assertEqual(self.window.last_summary["status"], "stopped")

    def test_close_keeps_running_worker_alive_and_closes_after_finished(self):
        worker = self.start_camera()
        event = QCloseEvent()
        self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertTrue(self.window.closing)
        self.assertTrue(self.window.isVisible())
        self.assertEqual(worker.stop_calls, 1)
        self.assertTrue(worker.isRunning())
        worker.finish("stopped")
        _APP.processEvents()
        self.assertFalse(self.window.isVisible())

    def test_close_with_slow_preview_is_asynchronous(self):
        self.window.set_source("slow.mp4")
        preview = self.window.preview_worker
        event = QCloseEvent()
        self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertTrue(preview.interrupted)
        self.assertTrue(self.window.isVisible())
        preview.finish()
        _APP.processEvents()
        self.assertFalse(self.window.isVisible())

    def test_old_session_signals_cannot_unlock_or_replace_a_new_session(self):
        old_worker = self.start_camera()
        old_worker.finish()
        self.window.start_analysis()
        new_worker = self.window.worker
        self.assertIsNot(new_worker, old_worker)
        self.assertTrue(old_worker.deleted)
        self.assertFalse(self.window.start_btn.isEnabled())
        old_worker.session_finished.emit({"status": "failed", "processed_frames": 99})
        old_worker.finished.emit()
        self.assertIsNone(self.window.last_summary)
        self.assertFalse(self.window.start_btn.isEnabled())
        self.assertTrue(new_worker.isRunning())
        new_worker.finish()
        self.assertTrue(self.window.start_btn.isEnabled())
        self.assertEqual(self.window.last_summary["status"], "completed")

    def test_polling_displays_one_latest_packet_and_source_timeline(self):
        worker = self.start_camera()
        stats = FrameStats(timestamp=1.25, frame_index=12, current_count=3,
            tracked_current_count=3, cumulative_unique_count=7,
            counts_by_class={"car": 3}, cumulative_by_class={"car": 7},
            line_counts={"up": 1, "down": 2}, dwell_current_count=1)
        worker.packet = FramePacket(image=np.zeros((32,48,3),dtype=np.uint8), stats=stats,
            detections=(), source_time_sec=1.25, processing_ms=20, frame_index=12,
            processing_fps=50, source_fps=30, total_frames=24, progress=.5,
            density_exceeded=False)
        self.window.poll_frame()
        self.assertEqual(self.window.metric_values["current"].text(), "3")
        self.assertEqual(self.window.metric_values["unique"].text(), "7")
        self.assertEqual(self.window.progress.value(), 500)
        self.assertEqual(list(self.window.chart.points), [(1.25, 3)])
        self.window.poll_frame()
        self.assertEqual(list(self.window.chart.points), [(1.25, 3)])

    def test_history_displays_saved_chinese_summary_and_raw_json_without_reprocessing(self):
        summary = {'status': 'completed', 'source': 3, 'site_id': '西号巷',
                   'device': 'cpu', 'analyzed_frames': 12, 'recorded_frames': 12,
                   'source_start_seconds': 0., 'source_end_seconds': 1.833333,
                   'peak_current_count': 9, 'cumulative_unique_count': 11,
                   'line_counts': {'up': 2, 'down': 1}, 'roi_dwell_events': 0}
        path = self.root / 'saved_session'
        entry = {**summary, 'session_dir': str(path)}
        def inspect_dialog(dialog):
            details = dialog.findChild(QPlainTextEdit)
            self.assertIn('会话累计轨迹 ID：11', details.toPlainText())
            self.assertIn('源时间范围：0.000 ～ 1.833 秒', details.toPlainText())
            self.assertIn('摄像头 3', details.toPlainText())
            mode = dialog.findChild(QComboBox)
            mode.setCurrentIndex(1)
            self.assertIn('"cumulative_unique_count": 11', details.toPlainText())
            button = dialog.findChild(QPushButton)
            button.click()
            return 0
        with patch.object(ui, 'list_history', return_value=[entry]), \
             patch.object(ui, 'read_session_summary', return_value=summary) as read, \
             patch.object(QDialog, 'exec', inspect_dialog), \
             patch.object(self.window, 'open_folder') as open_folder:
            self.window.show_history()
            self.assertEqual(read.call_count, 2)
            open_folder.assert_called_once_with(str(path))
        self.assertEqual(FakeWorker.created, [])

    def test_history_missing_values_are_not_invented(self):
        text = ui.readable_session_summary({'status': 'failed', 'error': '无法读取输入'})
        self.assertIn('已分析帧数：未记录', text)
        self.assertIn('上行 / 下行过线：未记录 / 未记录', text)
        self.assertIn('错误：无法读取输入', text)
        self.assertNotIn('源时间范围', text)

    def test_clear_roi_cancels_an_incomplete_drag(self):
        self.window.viewer.roi = (.1, .1, .6, .6)
        self.window.viewer._start = (.2, .2)
        self.window.viewer._end = (.3, .3)
        self.window.viewer.selecting = True
        self.window.clear_roi()
        self.assertIsNone(self.window.viewer.roi)
        self.assertIsNone(self.window.viewer._start)
        self.assertIsNone(self.window.viewer._end)
        self.assertFalse(self.window.viewer.selecting)

    def test_five_clicks_calibrate_and_save(self):
        import numpy as np
        from PySide6.QtCore import QPoint

        self.window.viewer.set_frame(np.zeros((300, 400, 3), dtype=np.uint8))
        _APP.processEvents()
        self.window.site.setText("测试点")
        self.window.start_calibration()
        rect = self.window.viewer.frame_rect()
        points = [(0.15, 0.85), (0.85, 0.85), (0.85, 0.25), (0.15, 0.25), (0.5, 0.55)]
        with patch.object(ui.QInputDialog, "getDouble", side_effect=[(6.0, True), (4.0, True)]):
            for x, y in points:
                position = QPoint(int(rect.left() + x * rect.width()),
                                  int(rect.top() + y * rect.height()))
                QTest.mouseClick(self.window.viewer, Qt.LeftButton, Qt.NoModifier, position)
                _APP.processEvents()
        self.assertIsNotNone(self.window.calibration)
        self.assertIsNotNone(self.window.calibration_path)
        self.assertTrue(self.window.calibration_path.is_file())
        self.assertIn("中心校核误差", self.window.calib_label.text())
        self.assertFalse(self.window.viewer.picking_calibration)


if __name__ == "__main__":
    unittest.main()

