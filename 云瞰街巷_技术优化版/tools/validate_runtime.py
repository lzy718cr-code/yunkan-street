"""Validate local weights, CPU/CUDA runtime, and lossless session exports.

The generated video consists of repeated source photographs. It is a software
test fixture, not a real traffic sequence or an accuracy evaluation dataset.

Run from this project using .venv/Scripts/python.exe tools/validate_runtime.py.
No package installation, model download, or network input is requested.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

os.environ["YOLO_AUTOINSTALL"] = "false"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication
import torch
from ultralytics import YOLO

from traffic_app.analytics import AnalyticsConfig, Detection, TrafficAnalytics
from traffic_app.engine import DetectionWorker, EngineConfig
from traffic_app.storage import atomic_write_json, list_history, read_session_summary


SETTINGS = {"imgsz": 640, "conf": .5, "iou": .45, "half": False}
FIXTURE_NOTE = (
    "由原始样本照片重复拼成，仅用于软件运行、导出与会话重置验证；"
    "不是实际连续交通视频，不用于准确率、交通流量或驻留效果评价。"
    "部分原始样本可能自带标注框，不能作为独立模型精度评估集。"
)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法解码样本：{path}")
    return image


def sample_images(directory: Path | None) -> list[Path]:
    if directory is None:
        candidates = sorted((ROOT.parent / ".material_review" / "raw").rglob("典型数据集样本"))
        if not candidates:
            raise FileNotFoundError("未找到原始典型数据集样本目录，请指定 --sample-dir")
        directory = candidates[0]
    images = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) < 6:
        raise ValueError(f"预期至少6张原始样本，实际{len(images)}张：{directory}")
    return images[:6]


def extract_detections(result) -> tuple[Detection, ...]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return ()
    coordinates = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    ids = None if boxes.id is None else boxes.id.detach().cpu().numpy()
    names = result.names
    items = []
    for index, (bbox, cls, conf) in enumerate(zip(coordinates, classes, confidences)):
        class_id = int(cls)
        name = str(names.get(class_id, class_id) if isinstance(names, dict) else names[class_id])
        items.append(Detection(class_id, name, float(conf), tuple(float(value) for value in bbox),
                               None if ids is None else int(ids[index])))
    return tuple(items)


def benchmark(model_path: Path, paths: list[Path], iterations: int, device: str, half: bool = False) -> dict:
    precision = "FP16" if half else "FP32"
    settings = {**SETTINGS, "half": half}
    print(f"开始 {device} {precision} 基准：2次预热 + {iterations}次测量", flush=True)
    frames = [read_image(path) for path in paths]
    model = YOLO(str(model_path))
    options = {**settings, "device": device, "verbose": False, "save": False}
    # The first track call initializes ByteTrack; exclude it from measurements.
    for _ in range(2):
        model.track(frames[0], tracker="bytetrack.yaml", persist=True, **options)
    for tracker in getattr(model.predictor, "trackers", ()):
        tracker.reset()
    analytics = TrafficAnalytics()
    rows = []
    for index in range(iterations):
        frame = frames[index % len(frames)]
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        began = time.perf_counter()
        result = model.track(frame, tracker="bytetrack.yaml", persist=True, **options)[0]
        detections = extract_detections(result)
        stats = analytics.update(detections, index / 10, index, int(frame.shape[1]), int(frame.shape[0]))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        milliseconds = (time.perf_counter() - began) * 1000
        rows.append({
            "iteration": index + 1, "sample": str(paths[index % len(paths)]),
            "source_shape_hwc": list(frame.shape), "track_and_stats_ms": milliseconds,
            "current_count": stats.current_count,
            "tracked_current_count": stats.tracked_current_count,
            "model_reported_ms": dict(result.speed),
        })
    timings = [row["track_and_stats_ms"] for row in rows]
    report = {
        "device": device, "precision": precision, "settings": settings,
        "warmup_iterations": 2, "measured_iterations": iterations,
        "mean_track_and_stats_ms": statistics.mean(timings),
        "median_track_and_stats_ms": statistics.median(timings),
        "throughput_frames_per_second": 1000 / statistics.mean(timings),
        "timing_includes": "preprocess + YOLO inference + postprocess + ByteTrack + tensor extraction + analytics; CUDA synchronized",
        "timing_excludes": "model loading, warmup, source decoding, CSV export, preview drawing",
        "accuracy_notes": "Unlabelled software runtime check only; no mAP or real-world accuracy claim.",
        "rows": rows,
    }
    print(f"完成 {device} {precision}: {report['mean_track_and_stats_ms']:.2f} ms/frame, {report['throughput_frames_per_second']:.2f} FPS", flush=True)
    del model
    return report


def make_fixture(paths: list[Path], output: Path) -> dict:
    width, height, fps = 640, 360, 6
    frame_count = 12
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"无法创建软件验证视频：{output}")
    try:
        for index in range(frame_count):
            source = read_image(paths[(index // 2) % len(paths)])
            scale = min(width / source.shape[1], height / source.shape[0])
            resized = cv2.resize(source, (round(source.shape[1] * scale), round(source.shape[0] * scale)))
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            top, left = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
            canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
            cv2.rectangle(canvas, (0, 0), (width, 23), (0, 0, 0), -1)
            cv2.putText(canvas, "SOFTWARE VALIDATION / STILL PHOTOS / NOT REAL TRAFFIC VIDEO", (5, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
    descriptor = {"path": str(output), "frame_count": frame_count, "fps": fps,
                  "resolution": [width, height], "note": FIXTURE_NOTE,
                  "source_images": [str(path) for path in paths]}
    atomic_write_json(output.with_suffix(".json"), descriptor)
    output.with_suffix(".txt").write_text(FIXTURE_NOTE + "\n", encoding="utf-8")
    return descriptor


def _csv_rows(path: Path) -> list[dict]:
    if not path.read_bytes().startswith(b"\xef\xbb\xbf"):
        raise AssertionError(f"CSV缺少UTF-8 BOM：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def engine_validation(model_path: Path, fixture: dict, output: Path, device: str) -> dict:
    app = QCoreApplication.instance() or QCoreApplication([])
    export_root = output / "engine_sessions"
    analytics_config = AnalyticsConfig(roi=(.1, .1, .9, .9), dwell_threshold_seconds=.3)
    # Intentionally reuse the analytics object: run() must clear all state.
    shared_analytics = TrafficAnalytics(analytics_config)
    sessions = []
    errors = []
    for session_number in range(1, 3):
        config = EngineConfig(source=fixture["path"], model_path=str(model_path),
                              device=device, conf=.5, iou=.45, imgsz=640, half=False,
                              display_fps=1, record_dir=str(export_root),
                              site_id="SOFTWARE_VALIDATION_ONLY", analytics_config=analytics_config)
        worker = DetectionWorker(config, analytics=shared_analytics)
        worker.error.connect(errors.append)
        print(f"开始完整 engine 会话 {session_number}/2（12帧，预览1FPS）", flush=True)
        worker.start()
        # Bounded wait calls let Qt deliver completion/errors without blocking
        # indefinitely. CPU inference has a generous overall 5-minute budget.
        deadline = time.monotonic() + 300
        while not worker.wait(100):
            app.processEvents()
            if time.monotonic() > deadline:
                worker.stop()
                while not worker.wait(100):
                    app.processEvents()
                raise TimeoutError("验证会话超时，已请求停止并等待线程退出")
        app.processEvents()
        summary = worker.summary
        if summary.get("status") != "completed":
            raise AssertionError(f"engine未完整完成：{summary.get('error', summary)}")
        session_dir = Path(summary["session_dir"])
        saved = read_session_summary(session_dir)
        metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
        frames = _csv_rows(session_dir / "frames.csv")
        detections = _csv_rows(session_dir / "detections.csv")
        events = _csv_rows(session_dir / "events.csv")
        checks = {
            "all_frames_analyzed": summary["analyzed_frames"] == fixture["frame_count"],
            "all_frames_recorded": len(frames) == fixture["frame_count"] == saved["recorded_frames"],
            "frame_indices_complete": [int(row["frame_index"]) for row in frames] == list(range(1, fixture["frame_count"] + 1)),
            "source_timestamps_match_fps": all(abs(float(row["source_time_seconds"]) - index / fixture["fps"]) < 1e-5
                                               for index, row in enumerate(frames)),
            "summary_matches_last_csv_cumulative": saved["cumulative_unique_count"] == int(frames[-1]["cumulative_unique_count"]),
            "detection_rows_match_frame_counts": len(detections) == sum(int(row["current_count"]) for row in frames),
            "line_events_match_summary": sum(row["event_type"] == "line_crossing" for row in events) == sum(saved["line_counts"].values()),
            "roi_events_match_summary": sum(row["event_type"] == "roi_dwell" for row in events) == saved["roi_dwell_events"],
            "status_consistent": saved["status"] == metadata["status"] == "completed",
            "source_metadata_consistent": saved["source"] == metadata["source"] == fixture["path"],
            "device_matches_request": saved["device"] == device,
            "first_frame_ids_reset": int(frames[0]["cumulative_unique_count"]) == int(frames[0]["tracked_current_count"]),
            "first_frame_no_crossing_or_dwell": int(frames[0]["line_up"]) == int(frames[0]["line_down"]) == int(frames[0]["dwell_current_count"]) == 0,
        }
        if not all(checks.values()):
            raise AssertionError(f"engine导出核验失败：{checks}")
        sessions.append({"session_number": session_number, "session_dir": str(session_dir),
                         "checks": checks, "analyzed_frames": saved["analyzed_frames"],
                         "detection_rows": len(detections), "event_rows": len(events),
                         "cumulative_unique_count": saved["cumulative_unique_count"],
                         "average_processing_ms": saved["average_processing_ms"]})
        print(f"完成会话 {session_number}：{len(frames)}帧CSV、{len(detections)}条检测、全部一致性检查通过", flush=True)
    history = list_history(export_root)
    checks = {
        "different_session_directories": sessions[0]["session_dir"] != sessions[1]["session_dir"],
        "both_sessions_readable_from_history": len(history) >= 2,
        "fresh_sessions_have_same_cumulative_count": sessions[0]["cumulative_unique_count"] == sessions[1]["cumulative_unique_count"],
        "no_engine_errors": not errors,
    }
    if not all(checks.values()):
        raise AssertionError(f"会话重开核验失败：{checks}")
    return {"fixture_note": FIXTURE_NOTE, "device": device, "sessions": sessions, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "validation")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    parser.add_argument("--skip-engine", action="store_true")
    args = parser.parse_args()
    if args.iterations < 3:
        parser.error("至少测量3次，推荐5次")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Real local model runtime and export verification; not model accuracy validation.",
        "settings": SETTINGS, "benchmarks": [], "errors": [],
        "environment": {
            "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "cuda_architectures": torch.cuda.get_arch_list(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "opencv": cv2.__version__, "ultralytics": importlib.metadata.version("ultralytics"),
            "numpy": np.__version__, "PySide6": importlib.metadata.version("PySide6"),
            "lap": importlib.metadata.version("lap"),
            "torch_cpu_threads": torch.get_num_threads(),
            "autoinstall_disabled": os.environ["YOLO_AUTOINSTALL"].lower() == "false",
        },
    }
    model_path = args.model.resolve()
    paths = sample_images(args.sample_dir)
    report["model"] = {"path": str(model_path), "bytes": model_path.stat().st_size,
                       "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
    report["samples"] = [str(path) for path in paths]
    runs = [(device, False) for device in args.devices]
    runs.extend((device, True) for device in args.devices if device.startswith("cuda"))
    for device, half in runs:
        if device.startswith("cuda") and not torch.cuda.is_available():
            report["errors"].append({"stage": "benchmark", "device": device, "error": "CUDA unavailable"})
            continue
        try:
            result = benchmark(model_path, paths, args.iterations, device, half)
            report["benchmarks"].append(result)
            suffix = "fp16" if half else "fp32"
            atomic_write_json(output / f"benchmark_{device.replace(':', '_')}_{suffix}.json", result)
        except Exception:
            report["errors"].append({"stage": "benchmark", "device": device, "traceback": traceback.format_exc()})
            print(report["errors"][-1]["traceback"], flush=True)
        atomic_write_json(output / "runtime_report.json", report)
    if not args.skip_engine:
        try:
            fixture = make_fixture(paths, output / "synthetic_photos_software_validation.avi")
            report["synthetic_fixture"] = fixture
            engine_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            report["engine_validation"] = engine_validation(model_path, fixture, output, engine_device)
        except Exception:
            report["errors"].append({"stage": "engine_validation", "traceback": traceback.format_exc()})
            print(report["errors"][-1]["traceback"], flush=True)
    report["passed"] = not report["errors"]
    atomic_write_json(output / "runtime_report.json", report)
    print(f"验证报告：{output / 'runtime_report.json'}；passed={report['passed']}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
