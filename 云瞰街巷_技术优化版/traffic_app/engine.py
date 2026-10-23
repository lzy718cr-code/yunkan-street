"""Thread-owned YOLO inference with complete recording and bounded UI delivery.

Only ``run`` touches YOLO or OpenCV capture handles. The GUI polls
``latest_packet``; it never receives one queued Qt signal per source frame.
Video files are read sequentially, including frames omitted from the preview.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import threading
import time
import traceback
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from . import __version__
from .analytics import AnalyticsConfig, Detection, FrameStats, TrafficAnalytics


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True, slots=True)
class EngineConfig:
    source: str | int
    model_path: str
    device: str = "auto"
    conf: float = 0.5
    iou: float = 0.45
    imgsz: int = 640
    density_threshold: int = 10
    record_dir: str | None = None
    site_id: str = ""
    capture_start: str | None = None
    max_frames: int | None = None
    display_fps: float = 15.0
    half: bool = False
    analytics_config: AnalyticsConfig | None = None
    classes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.source, bool) or not isinstance(self.source, (str, int)):
            raise ValueError("source 必须为文件路径或真实摄像头整数索引")
        if isinstance(self.source, int) and self.source < 0:
            raise ValueError("摄像头索引不能小于 0")
        if isinstance(self.source, str) and not self.source.strip():
            raise ValueError("请选择输入文件")
        if not isinstance(self.model_path, str) or not self.model_path:
            raise ValueError("请选择本地模型文件")
        for name in ("conf", "iou"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} 必须在 (0, 1] 范围内")
        if isinstance(self.imgsz, bool) or not isinstance(self.imgsz, int) or not 32 <= self.imgsz <= 2048:
            raise ValueError("imgsz 必须为 32 至 2048 的整数")
        if isinstance(self.density_threshold, bool) or not isinstance(self.density_threshold, int) or self.density_threshold < 1:
            raise ValueError("数量提示阈值必须为正整数")
        if self.max_frames is not None and (
            isinstance(self.max_frames, bool) or not isinstance(self.max_frames, int) or self.max_frames < 1
        ):
            raise ValueError("max_frames 必须为正整数或 None")
        if not math.isfinite(self.display_fps) or not 1 <= self.display_fps <= 60:
            raise ValueError("预览帧率必须在 1 至 60 范围内")
        if self.classes is not None:
            if (not isinstance(self.classes, tuple) or not self.classes
                    or not all(isinstance(name, str) and name for name in self.classes)):
                raise ValueError("classes 必须是非空类别名元组或 None")


@dataclass(frozen=True, slots=True)
class FramePacket:
    image: np.ndarray
    stats: FrameStats
    detections: tuple[Detection, ...]
    source_time_sec: float
    processing_ms: float
    frame_index: int
    processing_fps: float
    source_fps: float
    total_frames: int | None
    progress: float | None
    density_exceeded: bool

    @property
    def annotated_frame(self) -> np.ndarray:
        return self.image

    @property
    def frame(self) -> np.ndarray:
        return self.image

    @property
    def fps(self) -> float:
        return self.processing_fps

    @property
    def metrics(self) -> dict:
        return self.stats.to_dict()


class DetectionWorker(QThread):
    """One worker is one session, including its own model and tracker state.

    Connect ``session_finished`` for data and Qt's built-in ``finished`` for
    lifecycle handling. After stop(), keep the object alive until wait() or
    ``finished`` confirms the thread has actually exited.
    """

    status = Signal(str)
    error = Signal(str)
    session_finished = Signal(dict)

    _HOT_PARAMETERS = frozenset({"conf", "iou", "imgsz", "density_threshold"})

    def __init__(
        self,
        config: EngineConfig,
        analytics: TrafficAnalytics | None = None,
        recorder=None,
        parent=None,
        *,
        model_factory: Callable | None = None,
        capture_factory: Callable | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._config_lock = threading.Lock()
        self._packet_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._packet: FramePacket | None = None
        self._stop_event = threading.Event()
        self._analytics = analytics or TrafficAnalytics(config.analytics_config)
        self._recorder = recorder
        self._model_factory = model_factory
        self._capture_factory = capture_factory
        self._device = "cpu"
        self._inference_runtime: dict | None = None
        self._model_sha256: str | None = None
        self._last_publish_time = 0.0
        self._last_published_index = 0
        self._pending = None
        self._processing_times: deque[float] = deque(maxlen=30)
        self._total_processing_ms = 0.0
        self._summary: dict = {}
        self._started_once = False
        self._start_requested = False
        self._poster_saved = False

    @property
    def config(self) -> EngineConfig:
        with self._config_lock:
            return self._config

    @property
    def summary(self) -> dict:
        # Published after the session has finished. During inference poll packets.
        return dict(self._summary)

    def update_config(self, **changes) -> EngineConfig:
        """Atomically update inference knobs; applied on the next frame boundary."""
        unexpected = set(changes) - self._HOT_PARAMETERS
        if unexpected:
            raise ValueError(f"以下配置需开始新会话后修改：{', '.join(sorted(unexpected))}")
        with self._config_lock:
            updated = replace(self._config, **changes)
            self._config = updated
            return updated

    def latest_packet(self, clear: bool = True) -> FramePacket | None:
        with self._packet_lock:
            packet = self._packet
            if clear:
                self._packet = None
            return packet

    def stop(self) -> None:
        self._stop_event.set()
        self.requestInterruption()

    def start(self, priority=QThread.Priority.InheritPriority) -> None:
        """Prevent a completed worker from overwriting its closed session.

        QThread itself permits reuse; this worker deliberately owns a single
        recorder and tracker session. Repeated start while running is harmless.
        """
        with self._lifecycle_lock:
            if self._start_requested:
                if self.isRunning():
                    return
                raise RuntimeError("本次分析已结束，请创建新会话后再开始")
            self._start_requested = True
            try:
                super().start(priority)
            except Exception:
                self._start_requested = False
                raise

    def _stopping(self) -> bool:
        return self._stop_event.is_set() or self.isInterruptionRequested()

    def _select_device(self, requested: str) -> str:
        # Heavy imports, CUDA context initialization, and model work occur here,
        # called exclusively from run(), never from the UI constructor.
        import torch

        device = str(requested).strip().lower()
        if device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            return "cpu"
        if device.isdecimal():
            device = f"cuda:{device}"
        if device == "cuda":
            device = "cuda:0"
        if device.startswith("cuda:"):
            try:
                index = int(device.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError("CUDA 设备应形如 cuda:0") from exc
            if not torch.cuda.is_available() or not 0 <= index < torch.cuda.device_count():
                raise RuntimeError(f"{device} 不可用，请选择自动或 CPU")
            return device
        raise ValueError("设备应为 auto、cpu 或 cuda:0")

    def _create_model(self, model_path: str):
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"本地模型不存在：{path}")
        # Hash bytes on this worker before loading, so exports can identify the
        # selected weight even if loading or a later inference fails.
        self._model_sha256 = _file_sha256(path)
        if self._model_factory is not None:
            return self._model_factory(str(path))
        from ultralytics import YOLO

        # A separate instance prevents tracker IDs from leaking across sources.
        return YOLO(str(path))

    def _prediction_options(self, config: EngineConfig) -> dict:
        return {
            "conf": config.conf,
            "iou": config.iou,
            "imgsz": config.imgsz,
            "device": self._device,
            "half": bool(config.half and self._device.startswith("cuda")),
            "verbose": False,
            "save": False,
        }

    def _open_capture(self, source: str | int):
        if self._capture_factory is not None:
            capture = self._capture_factory(source)
        else:
            # Passing the original integer is essential: source="1" is a file,
            # whereas source=1 denotes physical camera index 1.
            if isinstance(source, int) and os.name == "nt":
                # DirectShow usually avoids MSMF's extra capture buffering.
                capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                if not capture.isOpened():
                    capture.release()
                    capture = cv2.VideoCapture(source)
            else:
                capture = cv2.VideoCapture(source)
            if not capture.isOpened() and isinstance(source, str) and os.name == "nt":
                capture.release()
                short_path = _windows_short_path(source)
                capture = cv2.VideoCapture(short_path or source)
        if not capture.isOpened():
            capture.release()
            label = f"摄像头 {source}" if isinstance(source, int) else source
            raise OSError(f"无法打开输入：{label}")
        if isinstance(source, int):
            # Best effort; unsupported backends may ignore buffer size. No
            # camera queue is created by Python, and UI delivery stays bounded.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    @staticmethod
    def _read_image(path: str) -> np.ndarray:
        # cv2.imread may fail on Windows Unicode paths; imdecode uses bytes.
        file_path = Path(path).expanduser()
        if not file_path.is_file():
            raise FileNotFoundError(f"图片不存在：{file_path}")
        data = np.fromfile(str(file_path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise OSError(f"图片无法解码：{file_path}")
        return image

    def _extract_detections(self, result) -> tuple[Detection, ...]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return ()
        coordinates = boxes.xyxy.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        ids = None if boxes.id is None else boxes.id.detach().cpu().numpy()
        names = result.names
        detections = []
        for index, (bbox, class_value, confidence) in enumerate(zip(coordinates, classes, confidences)):
            class_id = int(class_value)
            if isinstance(names, dict):
                class_name = str(names.get(class_id, class_id))
            else:
                class_name = str(names[class_id])
            track_id = None if ids is None else int(ids[index])
            if self.config.classes is not None and class_name not in self.config.classes:
                continue
            detections.append(Detection(
                class_id=class_id, class_name=class_name,
                confidence=float(confidence), bbox=tuple(float(x) for x in bbox),
                track_id=track_id,
            ))
        return tuple(detections)

    def _process_frame(self, model, frame, frame_index, timestamp, fps, total_frames, is_image=False):
        if frame is None or frame.size == 0:
            raise ValueError(f"第 {frame_index} 帧为空")
        config = self.config
        began = time.perf_counter()
        options = self._prediction_options(config)
        if is_image:
            # Still images have no temporal identity or traffic-flow evidence.
            results = model.predict(frame, **options)
        else:
            results = model.track(frame, tracker="bytetrack.yaml", persist=True, **options)
        detections = self._extract_detections(results[0])
        height, width = frame.shape[:2]
        stats = self._analytics.update(detections, timestamp, frame_index, int(width), int(height))
        processing_ms = (time.perf_counter() - began) * 1000.0
        self._processing_times.append(processing_ms)
        self._total_processing_ms += processing_ms
        if self._recorder is not None:
            # Every inferred frame is persisted, regardless of preview cadence.
            self._recorder.record_frame(
                stats, detections, int(width), int(height), processing_ms=processing_ms,
                inference_params={name: getattr(config, name) for name in self._HOT_PARAMETERS},
            )
            # One clean frame per session, so the heat map and the trajectory
            # overlay can be rebuilt later without the source video.
            if not self._poster_saved and not is_image:
                self._save_poster(frame)
        self._pending = (frame, detections, stats, timestamp, processing_ms, fps, total_frames, config)
        self._publish_pending()

    def _save_poster(self, frame) -> None:
        """Store one unannotated frame next to the CSVs.

        The offline portrait is rebuilt from the recording, so it needs a
        background image. Writing it here means the source video can be moved,
        renamed or deleted and the figures still render.
        """
        self._poster_saved = True
        try:
            session_dir = getattr(self._recorder, "session_dir", None)
            if session_dir is None:
                return
            poster = frame
            height, width = poster.shape[:2]
            if width > 1600:
                poster = cv2.resize(poster, (1600, int(height * 1600 / width)),
                                    interpolation=cv2.INTER_AREA)
            ok, buffer = cv2.imencode(".jpg", poster, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                buffer.tofile(str(Path(session_dir) / "poster.jpg"))
        except Exception:
            # A missing poster only costs the figures, never the recording.
            pass

    def _publish_pending(self, force=False) -> None:
        if self._pending is None:
            return
        frame, detections, stats, timestamp, processing_ms, fps, total_frames, config = self._pending
        if stats.frame_index == self._last_published_index:
            return
        now = time.perf_counter()
        if not force and now - self._last_publish_time < 1.0 / config.display_fps:
            return
        image = _annotate(frame, detections, stats, self._analytics.config)
        average_ms = sum(self._processing_times) / max(1, len(self._processing_times))
        packet = FramePacket(
            image=image, stats=stats, detections=detections,
            source_time_sec=timestamp, processing_ms=processing_ms,
            frame_index=stats.frame_index, processing_fps=1000.0 / max(average_ms, 0.001),
            source_fps=fps, total_frames=total_frames,
            progress=min(1.0, stats.frame_index / total_frames) if total_frames else None,
            density_exceeded=stats.current_count > config.density_threshold,
        )
        with self._packet_lock:
            self._packet = packet
        self._last_publish_time = now
        self._last_published_index = stats.frame_index

    def run(self) -> None:
        capture = None
        model = None
        source_fps = 0.0
        total_frames = None
        frame_index = 0
        final_status = "completed"
        error_message = ""
        started_at = datetime.now(timezone.utc).isoformat()
        started_clock = time.perf_counter()
        config = self.config
        try:
            if self._started_once:
                raise RuntimeError("请为每次检测创建新的 DetectionWorker，以保证会话与跟踪状态独立")
            self._started_once = True
            self._analytics.reset()
            if self._stopping():
                final_status = "stopped"
                return
            if self._recorder is None and config.record_dir:
                from .storage import SessionRecorder
                self._recorder = SessionRecorder(config.record_dir, config.source, asdict(config))
            self.status.emit("正在加载本地模型…")
            self._device = self._select_device(config.device)
            self._inference_runtime = {
                "device": self._device,
                "half": bool(config.half and self._device.startswith("cuda")),
            }
            model = self._create_model(config.model_path)
            if self._stopping():
                final_status = "stopped"
                return
            self.status.emit(f"正在预热模型（{self._device}）…")
            # Warm up via predict rather than track, so no synthetic frame ever
            # enters the tracker or the statistical/exported source timeline.
            model.predict(np.zeros((config.imgsz, config.imgsz, 3), dtype=np.uint8),
                          **self._prediction_options(config))
            if self._stopping():
                final_status = "stopped"
                return
            is_image = isinstance(config.source, str) and Path(config.source).suffix.lower() in IMAGE_SUFFIXES
            if is_image:
                total_frames = 1
                self.status.emit(f"正在检测图片（{self._device}）")
                image = self._read_image(config.source)
                self._process_frame(model, image, 1, 0.0, 0.0, 1, is_image=True)
                frame_index = 1
            else:
                self.status.emit(f"正在打开输入（{self._device}）…")
                capture = self._open_capture(config.source)
                is_camera = isinstance(config.source, int)
                raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
                source_fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else 0.0
                raw_total = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                total_frames = int(raw_total) if not is_camera and math.isfinite(raw_total) and raw_total > 0 else None
                if not is_camera and not source_fps:
                    self.status.emit("源文件未报告有效帧率，无法建立可靠视频时间轴；请转换视频后重试")
                    raise ValueError("视频帧率未知，不能可靠计算时间、过线和驻留统计")
                self.status.emit(f"正在检测{'摄像头' if is_camera else '视频'}（{self._device}）")
                capture_clock = time.perf_counter()
                previous_time = 0.0
                while not self._stopping():
                    ok, frame = capture.read()
                    if not ok:
                        if self._stopping():
                            final_status = "stopped"
                            break
                        if is_camera:
                            raise OSError("摄像头读取中断，请检查设备连接后重新开始")
                        if frame_index == 0:
                            raise OSError("视频没有可解码画面")
                        if total_frames and frame_index < total_frames:
                            raise OSError(f"视频解码提前结束：已处理 {frame_index}/{total_frames} 帧，已保存可用结果")
                        break
                    if self._stopping():
                        final_status = "stopped"
                        break
                    frame_index += 1
                    if is_camera:
                        timestamp = time.perf_counter() - capture_clock
                    else:
                        timestamp = _video_timestamp(capture, frame_index, source_fps, previous_time)
                    previous_time = timestamp
                    self._process_frame(model, frame, frame_index, timestamp, source_fps, total_frames)
                    if config.max_frames is not None and frame_index >= config.max_frames:
                        final_status = "limited"
                        break
            if self._stopping():
                final_status = "stopped"
        except Exception as exc:
            final_status = "failed"
            error_message = f"{type(exc).__name__}: {exc}"
            self.error.emit(error_message)
            # Keep the detailed traceback in the session summary rather than a
            # modal message, so diagnostics remain available after packaging.
            self._summary["traceback"] = traceback.format_exc()
        finally:
            try:
                if capture is not None:
                    capture.release()
                self._publish_pending(force=True)
            except Exception as exc:
                final_status = "failed"
                error_message = (error_message + "; " if error_message else "") + f"输入或预览清理失败：{exc}"
                self.error.emit(error_message)
            summary = self._analytics.summary()
            analyzed_frames = summary.get("analyzed_frames", 0)
            summary.update(self._summary)
            summary.update({
                "status": final_status, "error": error_message,
                "source": config.source, "site_id": config.site_id,
                "capture_start": config.capture_start,
                "model_path": config.model_path, "device": self._device,
                "model_sha256": self._model_sha256, "app_version": __version__,
                "inference_runtime": self._inference_runtime,
                "source_fps": source_fps, "source_frame_count": total_frames,
                "processed_frames": analyzed_frames,
                "average_processing_ms": self._total_processing_ms / analyzed_frames if analyzed_frames else 0.0,
                "average_processing_fps": analyzed_frames * 1000.0 / self._total_processing_ms if self._total_processing_ms else 0.0,
                "processing_time_notes": "Model inference plus statistics; excludes model loading, capture, export, and preview drawing.",
                "analytics_config": self._analytics.config.to_dict(),
                "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started_clock,
                "config_initial": asdict(config), "config_final": asdict(self.config),
            })
            if self._recorder is not None:
                try:
                    session_dir = self._recorder.finalize(summary, metadata={
                        "engine": "YOLO / ByteTrack", "status": final_status,
                        "error": error_message,
                        "model_sha256": self._model_sha256, "app_version": __version__,
                        "inference_runtime": self._inference_runtime,
                    })
                    summary["session_dir"] = str(session_dir)
                except Exception as exc:
                    summary["status"] = "failed"
                    summary["error"] = (error_message + "; " if error_message else "") + f"结果保存失败：{exc}"
                    self.error.emit(summary["error"])
                    close = getattr(self._recorder, "close", None)
                    if close is not None:
                        try:
                            close()
                        except Exception:
                            pass
            self._summary = summary
            # Release the model on the inference thread; avoid empty_cache(),
            # which would disturb other independent CUDA processes/sessions.
            del model
            self._pending = None
            self.session_finished.emit(dict(summary))


def _file_sha256(path: Path) -> str:
    """Identify a local model without loading its contents into memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_timestamp(capture, frame_index: int, fps: float, previous: float) -> float:
    reported = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
    fallback = (frame_index - 1) / fps
    if math.isfinite(reported) and reported >= 0 and (frame_index == 1 or reported > previous):
        return reported
    return max(previous, fallback)


def _windows_short_path(path: str) -> str | None:
    """Fallback for Windows codecs that cannot open a Unicode filename."""
    try:
        import ctypes
        from ctypes import wintypes
        function = ctypes.windll.kernel32.GetShortPathNameW
        function.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
        function.restype = wintypes.DWORD
        absolute = str(Path(path).resolve())
        required = function(absolute, None, 0)
        if not required:
            return None
        buffer = ctypes.create_unicode_buffer(required)
        return buffer.value if function(absolute, buffer, required) else None
    except (AttributeError, OSError):
        return None


def _annotate(frame, detections: tuple[Detection, ...], stats: FrameStats, config: AnalyticsConfig) -> np.ndarray:
    image = frame.copy()
    height, width = image.shape[:2]
    thickness = max(1, round(min(height, width) / 500))
    colors = ((91, 201, 88), (255, 185, 68), (73, 121, 255), (204, 111, 220))
    line_y = int(config.line_y * height)
    cv2.line(image, (0, line_y), (width - 1, line_y), (230, 180, 35), thickness)
    if config.roi:
        x1, y1, x2, y2 = config.roi
        cv2.rectangle(image, (int(x1 * width), int(y1 * height)),
                      (int(x2 * width), int(y2 * height)), (30, 190, 255), thickness)
    for detection in detections:
        x1, y1, x2, y2 = (int(round(x)) for x in detection.bbox)
        color = colors[detection.class_id % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        identifier = f" #{detection.track_id}" if detection.track_id is not None else ""
        label = f"{detection.class_name}{identifier} {detection.confidence:.2f}"
        dwell = stats.dwell_seconds_by_track.get(detection.track_id, 0.0)
        if dwell >= config.dwell_threshold_seconds:
            label += f" dwell {dwell:.1f}s"
        cv2.putText(image, label, (max(0, x1), max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, thickness, cv2.LINE_AA)
    return image
