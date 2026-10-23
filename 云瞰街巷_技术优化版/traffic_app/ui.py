from __future__ import annotations

from collections import deque
from dataclasses import asdict
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QRectF, QPointF, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
    QMainWindow, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from .analytics import AnalyticsConfig
from .calibration import Calibration, CalibrationError
from .engine import DetectionWorker, EngineConfig, IMAGE_SUFFIXES
from .storage import list_history, read_session_summary


THEME = """
QMainWindow, QDialog { background: #0b1220; color: #e3eaf5; }
QWidget { color: #dce7f5; font-family: 'Microsoft YaHei UI'; font-size: 12px; }
QFrame#panel { background: #121e30; border: 1px solid #26364d; border-radius: 10px; }
QFrame#metric { background: #142238; border: 1px solid #253950; border-radius: 8px; }
QFrame#guide { background: #14283f; border: 1px solid #35526f; border-radius: 8px; }
QLabel#title { font-size: 23px; font-weight: 700; color: #eef8ff; }
QLabel#muted { color: #8fa5bf; }
QLabel#metricValue { color: #48d7c3; font-size: 27px; font-weight: 700; }
QPushButton { background: #20324b; border: 1px solid #354e6c; padding: 8px 12px; border-radius: 6px; }
QPushButton:hover { background: #2a4260; }
QPushButton:disabled { color: #6b7a8c; background: #152134; border-color: #263448; }
QPushButton#primary { color: #052822; background: #50d8bf; font-weight: 700; border: 0; }
QPushButton#primary:hover { background: #79e5d2; }
QPushButton#stop { color: #ffb7b7; border-color: #774348; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QPlainTextEdit {
  background: #0d1728; border: 1px solid #33445c; border-radius: 5px; padding: 6px;
  selection-background-color: #286b7b;
}
QComboBox QAbstractItemView { background: #142238; color: #e3eaf5; }
QProgressBar { background: #0d1728; border: 1px solid #283e55; border-radius: 4px; height: 10px; text-align: center; }
QProgressBar::chunk { background: #3dbba9; border-radius: 3px; }
QScrollArea { border: 0; background: transparent; }
QToolTip { color: #e6f2ff; background: #20324b; border: 1px solid #466078; }
"""


def application_root() -> Path:
    return Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent.parent


def default_model_path() -> Path:
    external = application_root() / 'models' / 'best.pt'
    if external.exists():
        return external
    return Path(getattr(sys, '_MEIPASS', application_root())) / 'models' / 'best.pt'


def default_space_model_path() -> Path:
    external = application_root() / 'models' / 'yolo11s.pt'
    if external.exists():
        return external
    return Path(getattr(sys, '_MEIPASS', application_root())) / 'models' / 'yolo11s.pt'


def default_data_root() -> Path:
    override = os.environ.get('CLOUDLANE_DATA_DIR')
    if override:
        return Path(override).resolve()
    candidate = application_root() / 'data'
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / '.write_probe'
        probe.write_text('', encoding='utf-8')
        probe.unlink()
        return candidate
    except OSError:
        return Path.home() / 'CloudlaneData'


def demo_clip_path() -> Path | None:
    """The bundled sample clip, whether running from source or from the EXE."""
    for base in (application_root() / 'demo', Path(getattr(sys, '_MEIPASS', application_root())) / 'demo'):
        candidate = base / 'demo_60s.mp4'
        if candidate.is_file():
            return candidate
    return None


def readable_session_summary(summary: dict) -> str:
    """Describe only saved results; absent values remain explicitly unknown."""
    status_labels = {'completed': '分析完成', 'stopped': '手动停止',
                     'limited': '达到验证帧数上限', 'failed': '分析失败'}
    source = summary.get('source', '未记录')
    if isinstance(source, int):
        source = f'摄像头 {source}'
    lines = [
        f"状态：{status_labels.get(summary.get('status'), summary.get('status', '未记录'))}",
        f"点位：{summary.get('site_id') or '未填写'}",
        f"来源：{source}",
        f"开始时间（UTC）：{summary.get('started_at_utc', summary.get('started_at', '未记录'))}",
        f"设备：{summary.get('device', '未记录')}",
        f"已分析帧数：{summary.get('analyzed_frames', summary.get('processed_frames', '未记录'))}",
        f"已保存帧数：{summary.get('recorded_frames', '未记录')}",
    ]
    start, end = summary.get('source_start_seconds'), summary.get('source_end_seconds')
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        lines.append(f'源时间范围：{start:.3f} ～ {end:.3f} 秒')
    crossings = summary.get('line_counts') or {}
    lines += [
        '', '计数结果',
        f"当帧数量峰值：{summary.get('peak_current_count', '未记录')}",
        f"会话累计轨迹 ID：{summary.get('cumulative_unique_count', '未记录')}",
        f"上行 / 下行过线：{crossings.get('up', '未记录')} / {crossings.get('down', '未记录')}",
        f"ROI 连续驻留事件：{summary.get('roi_dwell_events', '未记录')}",
    ]
    config = summary.get('config_final') or {}
    if config:
        lines += ['', '会话结束时的设置',
                  f"置信度：{config.get('conf', '未记录')}　IoU：{config.get('iou', '未记录')}",
                  f"输入尺寸：{config.get('imgsz', '未记录')}　数量提示阈值：{config.get('density_threshold', '未记录')}"]
    if summary.get('error'):
        lines += ['', f"错误：{summary['error']}"]
    lines += ['', '口径说明：累计轨迹 ID 不等同于真实车辆去重；区域驻留不证明停车或违法。',
              '逐帧数量、检测框及事件保存在 CSV；修改参数的记录请查 frames.csv。']
    return '\n'.join(lines)


class PreviewLoader(QThread):
    ready = Signal(object)
    failure = Signal(str)

    def __init__(self, source: str, parent=None):
        super().__init__(parent)
        self.source = source

    def run(self):
        cap = None
        try:
            if self.isInterruptionRequested():
                return
            if Path(self.source).suffix.lower() in IMAGE_SUFFIXES:
                image = cv2.imdecode(np.fromfile(self.source, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000,
                ])
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(self.source)
                ok, image = cap.read()
                if not ok:
                    image = None
            if self.isInterruptionRequested():
                return
            if image is None:
                raise OSError('无法读取预览，请检查文件格式或文件是否完整。')
            self.ready.emit(image)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failure.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()


class ImageView(QWidget):
    roi_selected = Signal(object)
    calibration_ready = Signal(object)
    section_ready = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 270)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pixmap = QPixmap()
        self.line_y = .5
        self.show_line = True
        self.section = None
        self.picking_section = False
        self._section_start = None
        self.roi = None
        self.selecting = False
        self.picking_calibration = False
        self.calibration_points: list[tuple[float, float]] = []
        self._start = None
        self._end = None

    def set_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        self.pixmap = QPixmap.fromImage(QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy())
        self.update()

    def frame_rect(self):
        if self.pixmap.isNull():
            return QRectF()
        scale = min(self.width() / self.pixmap.width(), self.height() / self.pixmap.height())
        width, height = self.pixmap.width() * scale, self.pixmap.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#09111e'))
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.frame_rect()
        if rect.isEmpty():
            painter.setPen(QColor('#7890ad'))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             '还没有画面\n\n点右侧「载入示例片段」直接体验\n或点「打开图片 / 视频」用自己的素材')
            return
        painter.drawPixmap(rect.toRect(), self.pixmap)
        if self.show_line:
            painter.setPen(QPen(QColor('#66cdeb'), 2, Qt.DashLine))
            painter.drawLine(QPointF(rect.left(), rect.top() + self.line_y*rect.height()),
                             QPointF(rect.right(), rect.top() + self.line_y*rect.height()))
        if self.section:
            ax, ay, bx, by = self.section
            start = QPointF(rect.left() + ax*rect.width(), rect.top() + ay*rect.height())
            end = QPointF(rect.left() + bx*rect.width(), rect.top() + by*rect.height())
            painter.setPen(QPen(QColor('#48d7c3'), 3))
            painter.drawLine(start, end)
            painter.setBrush(QColor('#f5c860'))
            painter.setPen(QPen(QColor('#f5c860'), 2))
            for point, label in ((start, 'A'), (end, 'B')):
                painter.drawEllipse(point, 6, 6)
                painter.drawText(QPointF(point.x() + 9, point.y() - 9), label)
        if self._section_start is not None:
            painter.setPen(QPen(QColor('#f5c860'), 2, Qt.DashLine))
            point = QPointF(rect.left() + self._section_start[0]*rect.width(),
                            rect.top() + self._section_start[1]*rect.height())
            painter.drawEllipse(point, 7, 7)
        roi = self.roi
        if self._start is not None and self._end is not None:
            roi = (min(self._start[0], self._end[0]), min(self._start[1], self._end[1]),
                   max(self._start[0], self._end[0]), max(self._start[1], self._end[1]))
        if roi:
            x1,y1,x2,y2 = roi
            painter.setPen(QPen(QColor('#ffc169'), 2))
            painter.setBrush(QColor(255, 193, 105, 20))
            painter.drawRect(QRectF(rect.left()+x1*rect.width(), rect.top()+y1*rect.height(),
                                    (x2-x1)*rect.width(), (y2-y1)*rect.height()))
        if self.calibration_points:
            painter.setBrush(QColor(255, 209, 102, 90))
            painter.setPen(QPen(QColor('#ffd166'), 2))
            for index, (x, y) in enumerate(self.calibration_points, 1):
                center = QPointF(rect.left() + x * rect.width(), rect.top() + y * rect.height())
                painter.drawEllipse(center, 6, 6)
                painter.drawText(QPointF(center.x() + 9, center.y() - 9), str(index))

    def normalized_position(self, pos):
        rect = self.frame_rect()
        if rect.isEmpty():
            return None
        return (max(0.,min(1.,(pos.x()-rect.left())/rect.width())),
                max(0.,min(1.,(pos.y()-rect.top())/rect.height())))

    def mousePressEvent(self, event):
        if self.picking_section and event.button() == Qt.LeftButton and self.frame_rect().contains(event.position()):
            position = self.normalized_position(event.position())
            if position is not None:
                if self._section_start is None:
                    self._section_start = position
                    self.update()
                else:
                    start = self._section_start
                    self._section_start = None
                    self.picking_section = False
                    self.setCursor(Qt.ArrowCursor)
                    if math.hypot(position[0]-start[0], position[1]-start[1]) > .02:
                        self.section = (start[0], start[1], position[0], position[1])
                        self.section_ready.emit(self.section)
                    self.update()
            return
        if self.picking_calibration and event.button() == Qt.LeftButton and self.frame_rect().contains(event.position()):
            position = self.normalized_position(event.position())
            if position is not None and len(self.calibration_points) < 5:
                self.calibration_points.append(position)
                self.update()
                if len(self.calibration_points) == 5:
                    self.picking_calibration = False
                    self.setCursor(Qt.ArrowCursor)
                    self.calibration_ready.emit(list(self.calibration_points))
            return
        if self.selecting and event.button() == Qt.LeftButton and self.frame_rect().contains(event.position()):
            self._start = self._end = self.normalized_position(event.position())

    def mouseMoveEvent(self, event):
        if self._start is not None:
            self._end = self.normalized_position(event.position())
            self.update()

    def mouseReleaseEvent(self, event):
        if not self.selecting or self._start is None:
            return
        end = self.normalized_position(event.position())
        x1,y1 = self._start
        x2,y2 = end
        self._start = self._end = None
        if abs(x2-x1) > .01 and abs(y2-y1) > .01:
            self.roi = (min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2))
            self.selecting = False
            self.setCursor(Qt.ArrowCursor)
            self.roi_selected.emit(self.roi)
        self.update()


class CountChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = deque(maxlen=300)
        self.threshold = 10
        self.caption = '画面内车辆数 · 源视频时间（秒）'
        self.setMinimumHeight(155)

    def add(self, timestamp, count):
        self.points.append((timestamp, count))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor('#0d1828'))
        p.setRenderHint(QPainter.Antialiasing)
        plot = QRectF(40, 18, max(20,self.width()-58), max(20,self.height()-48))
        ymax = max([5,self.threshold*1.2]+[c*1.2 for _,c in self.points])
        p.setFont(QFont('Microsoft YaHei UI', 8))
        for i in range(5):
            y = plot.bottom()-plot.height()*i/4
            p.setPen(QColor('#20334a'))
            p.drawLine(QPointF(plot.left(),y),QPointF(plot.right(),y))
            p.setPen(QColor('#8da1bc'))
            p.drawText(QRectF(0,y-7,33,15),Qt.AlignRight,str(round(ymax*i/4)))
        if self.threshold <= ymax:
            y = plot.bottom()-self.threshold/ymax*plot.height()
            p.setPen(QPen(QColor('#ba6f6f'),1,Qt.DashLine))
            p.drawLine(QPointF(plot.left(),y),QPointF(plot.right(),y))
        if self.points:
            first, last = self.points[0][0], self.points[-1][0]
            span = max(1.,last-first)
            prev = None
            for stamp,count in self.points:
                point = QPointF(plot.left()+(stamp-first)/span*plot.width(),plot.bottom()-count/ymax*plot.height())
                if prev is not None:
                    p.setPen(QPen(QColor('#fc9691' if count > self.threshold else '#4cd6bf'),2))
                    p.drawLine(prev,point)
                prev = point
            p.setPen(QColor('#91a7c2'))
            p.drawText(QRectF(plot.left(),plot.bottom()+7,110,18),f'{first:.1f} s')
            p.drawText(QRectF(plot.right()-110,plot.bottom()+7,110,18),Qt.AlignRight,f'{last:.1f} s')
        p.setPen(QColor('#91a7c2'))
        p.drawText(QRectF(0,1,self.width(),16),Qt.AlignCenter,self.caption)


class PortraitWorker(QThread):
    """Rebuild the space-use portrait for a finished session, off the GUI thread."""

    portrait_ready = Signal(object)
    portrait_failed = Signal(str)

    def __init__(self, session_dir, mask_polygon=None, playback_seconds: float = 15.0, parent=None):
        super().__init__(parent)
        self.session_dir = Path(session_dir)
        self.mask_polygon = mask_polygon
        self.playback_seconds = playback_seconds

    def run(self):
        try:
            from .portrait import build_portrait
            calibration = None
            calibration_path = self.session_dir / 'calibration.json'
            if calibration_path.is_file():
                calibration = Calibration.load(calibration_path)
            portrait = build_portrait(
                self.session_dir, calibration=calibration,
                mask_polygon=self.mask_polygon,
                playback_seconds=self.playback_seconds)
            self.portrait_ready.emit(portrait)
        except Exception as exc:
            logging.exception("空间画像生成失败")
            self.portrait_failed.emit(str(exc))


class PortraitDialog(QDialog):
    """Show the portrait: one figure plus the numbers a reader will ask for."""

    def __init__(self, portrait, parent=None):
        super().__init__(parent)
        self.portrait = portrait
        self.setWindowTitle('空间使用画像')
        self.resize(1140, 640)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        image = portrait.files.get('heatmap')
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(460, 260)
        self.preview.setStyleSheet('background:#0d1728; border:1px solid #26364d; border-radius:8px;')
        self._pixmap = QPixmap(str(image)) if image and Path(image).is_file() else QPixmap()
        if self._pixmap.isNull():
            self.preview.setText('本次会话没有可用的封面帧，未能生成热力图。')
        layout.addWidget(self.preview, 3)

        right = QWidget()
        column = QVBoxLayout(right)
        column.setContentsMargins(0, 0, 0, 0)
        heading = QLabel('本次观测得到了什么')
        heading.setObjectName('title')
        heading.setStyleSheet('font-size:18px;')
        column.addWidget(heading)
        table = QFormLayout()
        table.setSpacing(6)
        for label, value in portrait.headline():
            name = QLabel(label)
            name.setObjectName('muted')
            amount = QLabel(value)
            amount.setStyleSheet('color:#48d7c3; font-weight:700;')
            table.addRow(name, amount)
        column.addLayout(table)
        column.addWidget(QLabel(''))
        for warning in portrait.warnings:
            note = QLabel('· ' + warning)
            note.setWordWrap(True)
            note.setObjectName('muted')
            column.addWidget(note)
        caveat = QLabel('口径：停留是本会话内连续可见时长，不等于停留意愿；'
                        '未标定时所有距离按画面宽比例，不做米制换算；轨迹 ID 存在断裂，'
                        '计数是筛查值而非人流量。')
        caveat.setWordWrap(True)
        caveat.setObjectName('muted')
        column.addWidget(caveat)
        column.addStretch()
        location = QLabel(f'结果目录：{portrait.out_dir}')
        location.setObjectName('muted')
        location.setWordWrap(True)
        column.addWidget(location)
        open_btn = QPushButton('打开结果目录')
        open_btn.clicked.connect(lambda: MainWindow.open_folder(portrait.out_dir))
        close_btn = QPushButton('关闭')
        close_btn.setObjectName('primary')
        close_btn.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addWidget(open_btn)
        row.addWidget(close_btn)
        column.addLayout(row)
        layout.addWidget(right, 2)

    def _fit_preview(self):
        # Scale to the space the label actually got; scaling to a fixed size
        # clips the caption whenever the dialog is a little narrower.
        if self._pixmap.isNull():
            return
        self.preview.setPixmap(self._pixmap.scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def showEvent(self, event):
        super().showEvent(event)
        self._fit_preview()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_preview()


class MainWindow(QMainWindow):
    session_completed = Signal(dict)

    def __init__(self, data_root=None):
        super().__init__()
        self.setWindowTitle('云瞰街巷 3.0 · 小空间人群使用画像')
        self.resize(1280, 870)
        self.setMinimumSize(1010, 690)
        self.data_root = Path(data_root) if data_root else default_data_root()
        self.sessions_root = self.data_root/'sessions'
        self.sessions_root.mkdir(parents=True,exist_ok=True)
        self.worker = None
        self.preview_worker = None
        self._previews = set()
        self._preview_generation = 0
        self.source = None
        self.last_summary = None
        self.latest_packet = None
        self.calibration = None
        self.calibration_path = None
        self.closing = False
        self.test_max_frames = None
        self.portrait_worker = None
        self.portrait = None
        self.last_session_dir = None
        self.analysis_mask = None
        self.suppress_dialogs = False
        self.intro_seen = False
        try:
            from PySide6.QtCore import QSettings
            raw = QSettings('Cloudlane', '云瞰街巷').value('intro_seen', False)
            self.intro_seen = str(raw).lower() in ('true', '1')
        except Exception:
            pass
        self._build()
        if self.intro_seen:
            self.guide.setVisible(False)
        self.timer = QTimer(self)
        self.timer.setInterval(67)
        self.timer.timeout.connect(self.poll_frame)
        self.timer.start()

    def _build(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18,14,18,14)
        layout.setSpacing(12)
        top = QHBoxLayout()
        self.header_actions = top
        title = QLabel('云瞰街巷  /  小空间人群使用画像'); title.setObjectName('title')
        top.addWidget(title); top.addStretch()
        history = QPushButton('历史结果'); history.clicked.connect(self.show_history)
        folder = QPushButton('数据目录'); folder.clicked.connect(lambda:self.open_folder(self.data_root))
        top.addWidget(history); top.addWidget(folder)
        layout.addLayout(top)
        subtitle=QLabel('本地处理 · 逐帧记录 · 轨迹与停留画像 · 不重训模型'); subtitle.setObjectName('muted')
        layout.addWidget(subtitle)
        self.guide=QFrame();self.guide.setObjectName('guide')
        guide_box=QHBoxLayout(self.guide);guide_box.setContentsMargins(14,10,14,10)
        guide_text=QLabel('第一次使用？三步走：① 点右侧「载入示例片段」　② 点顶部「开始分析」　'
                          '③ 分析结束后自动弹出空间画像，里面有热力图和九个指标。')
        guide_text.setWordWrap(True)
        guide_box.addWidget(guide_text,1)
        guide_close=QPushButton('知道了');guide_close.clicked.connect(self.dismiss_guide)
        guide_box.addWidget(guide_close)
        layout.addWidget(self.guide)
        metrics=QHBoxLayout(); self.metric_values={}; self.metric_labels={}; self.metric_frames={}
        for key,label in [('current','画面内人/车'),('unique','轨迹数'),('cross','过线  ↑ / ↓'),('dwell','停留中'),('fps','处理速度 FPS')]:
            frame=QFrame();frame.setObjectName('metric');box=QVBoxLayout(frame)
            small=QLabel(label);small.setObjectName('muted');value=QLabel('—');value.setObjectName('metricValue')
            box.addWidget(small);box.addWidget(value);metrics.addWidget(frame);self.metric_values[key]=value
            self.metric_labels[key]=small;self.metric_frames[key]=frame
        self.metric_values['unique'].setToolTip('本次会话出现过的轨迹 ID 数。跟踪断裂或换 ID 可能影响统计，不等同于真实人数去重。')
        self.metric_values['cross'].setToolTip('按画面上下方向，轨迹底部中心跨越虚线；同一ID每个方向仅累计一次。')
        self.metric_values['dwell'].setToolTip('当前仍在观察区域内、且已超过驻留时长的目标数。')
        layout.addLayout(metrics)
        split=QSplitter(Qt.Horizontal)
        left=QWidget();box=QVBoxLayout(left);box.setContentsMargins(0,0,0,0);box.setSpacing(8)
        self.viewer=ImageView();self.viewer.roi_selected.connect(self.roi_changed)
        self.viewer.section_ready.connect(self.section_picked)
        box.addWidget(self.viewer,4)
        self.chart=CountChart();box.addWidget(self.chart,1)
        self.progress=QProgressBar();self.progress.setRange(0,1000);self.progress.setValue(0);self.progress.setTextVisible(False)
        box.addWidget(self.progress)
        self.detail=QLabel('第 1 步：点「载入示例片段」，或「打开图片 / 视频」');self.detail.setObjectName('muted');box.addWidget(self.detail)
        split.addWidget(left)
        panel=QFrame();panel.setObjectName('panel');panel.setMinimumWidth(305);panel.setMaximumWidth(380)
        controls=QVBoxLayout(panel);controls.setContentsMargins(10,10,10,10)
        self.file_btn=QPushButton('打开图片 / 视频');self.file_btn.clicked.connect(self.choose_file);controls.addWidget(self.file_btn)
        self.demo_btn=QPushButton('载入示例片段（校园街角）');self.demo_btn.clicked.connect(self.load_demo)
        self.demo_btn.setVisible(demo_clip_path() is not None)
        controls.addWidget(self.demo_btn)
        self.source_label=QLabel('尚未选择文件');self.source_label.setWordWrap(True);self.source_label.setObjectName('muted');controls.addWidget(self.source_label)
        row=QHBoxLayout();self.camera_index=QSpinBox();self.camera_index.setRange(0,20);self.camera_index.setPrefix('索引 ')
        self.camera_btn=QPushButton('使用摄像头');self.camera_btn.clicked.connect(self.choose_camera);row.addWidget(self.camera_index);row.addWidget(self.camera_btn);controls.addLayout(row)
        form=QFormLayout();form.setSpacing(6);self.form=form
        self.mode=QComboBox();self.mode.addItem('人行观测','person');self.mode.addItem('车辆观测','vehicle');form.addRow('场景',self.mode)
        self.site=QLineEdit();self.site.setPlaceholderText('例如 西号巷 / 点位01');form.addRow('点位',self.site)
        self.capture_start=QLineEdit();self.capture_start.setPlaceholderText('可选：2026-10-18 08:00:00');form.addRow('采集起点',self.capture_start)
        self.device=QComboBox();self.device.addItem('自动 · 优先 NVIDIA GPU','auto');self.device.addItem('仅 CPU','cpu');self.device.addItem('NVIDIA GPU 0','cuda:0');form.addRow('设备',self.device)
        self.conf=QDoubleSpinBox();self.conf.setRange(.01,.99);self.conf.setSingleStep(.05);self.conf.setValue(.5);form.addRow('置信度',self.conf)
        self.iou=QDoubleSpinBox();self.iou.setRange(.05,.95);self.iou.setSingleStep(.05);self.iou.setValue(.45);form.addRow('IoU',self.iou)
        self.imgsz=QComboBox()
        for size in [416,640,960,1024,1280]:self.imgsz.addItem(str(size),size)
        self.imgsz.setCurrentIndex(1);form.addRow('输入尺寸',self.imgsz)
        self.threshold=QSpinBox();self.threshold.setRange(1,9999);self.threshold.setValue(10);form.addRow('数量提示阈值',self.threshold)
        self.line=QSpinBox();self.line.setRange(5,95);self.line.setValue(50);self.line.setSuffix(' %');form.addRow('过线位置',self.line)
        self.dwell=QDoubleSpinBox();self.dwell.setRange(.1,3600);self.dwell.setValue(5);self.dwell.setSuffix(' 秒');form.addRow('驻留时长',self.dwell)
        controls.addLayout(form)
        self.section_box=QWidget();section_layout=QVBoxLayout(self.section_box)
        section_layout.setContentsMargins(0,0,0,0);section_layout.setSpacing(4)
        section_row=QHBoxLayout()
        self.section_btn=QPushButton('画断面线');self.section_btn.clicked.connect(self.start_section)
        self.section_clear=QPushButton('清除线');self.section_clear.clicked.connect(self.clear_section)
        section_row.addWidget(self.section_btn);section_row.addWidget(self.section_clear)
        section_layout.addLayout(section_row)
        self.section_label=QLabel('在画面里点两下，画一条地面断面线；人穿过它才算过线。')
        self.section_label.setWordWrap(True);self.section_label.setObjectName('muted')
        section_layout.addWidget(self.section_label)
        form.insertRow(1,'断面线',self.section_box)
        row=QHBoxLayout();self.roi_btn=QPushButton('框选驻留区域');self.roi_btn.clicked.connect(self.select_roi)
        self.roi_clear=QPushButton('清除');self.roi_clear.clicked.connect(self.clear_roi)
        self.calib_btn=QPushButton('四点标定');self.calib_btn.clicked.connect(self.start_calibration)
        self.calib_clear=QPushButton('清除标定');self.calib_clear.clicked.connect(self.clear_calibration)
        row.addWidget(self.roi_btn);row.addWidget(self.roi_clear);controls.addLayout(row)
        self.roi_label=QLabel('未设置区域，区域统计不启用');self.roi_label.setWordWrap(True);self.roi_label.setObjectName('muted');controls.addWidget(self.roi_label)
        self.calib_label=QLabel('未标定：按画面比例计算');self.calib_label.setWordWrap(True);self.calib_label.setObjectName('muted');controls.addWidget(self.calib_label)
        self.model_path=default_model_path();self.model_btn=QPushButton('更换本地模型');self.model_btn.clicked.connect(self.choose_model);controls.addWidget(self.model_btn)
        self.start_btn=QPushButton('开始分析');self.start_btn.setObjectName('primary');self.start_btn.clicked.connect(self.start_analysis)
        self.stop_btn=QPushButton('停止并保存');self.stop_btn.setObjectName('stop');self.stop_btn.clicked.connect(self.stop_analysis);self.stop_btn.setEnabled(False)
        self.portrait_btn=QPushButton('空间画像');self.portrait_btn.setToolTip('对最近一次会话重算热力图、轨迹叠加和指标')
        self.portrait_btn.clicked.connect(lambda:self.run_portrait());self.portrait_btn.setEnabled(False)
        self.header_actions.insertWidget(2,self.calib_btn)
        self.header_actions.insertWidget(3,self.calib_clear)
        self.header_actions.insertWidget(4,self.start_btn)
        self.header_actions.insertWidget(5,self.stop_btn)
        self.header_actions.insertWidget(6,self.portrait_btn)
        self.chart_btn=QPushButton('保存当前曲线 PNG');self.chart_btn.clicked.connect(self.export_chart);controls.addWidget(self.chart_btn)
        note=QLabel('数量提示用于观察，不代表拥堵等级。\n区域驻留不等同于停车或违法判断。');note.setWordWrap(True);note.setObjectName('muted');controls.addWidget(note)
        controls.addStretch()
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(panel);scroll.setMinimumWidth(315)
        split.addWidget(scroll);split.setSizes([870,330]);layout.addWidget(split,1)
        self.status_label=QLabel('就绪 · 请先打开文件');self.status_label.setWordWrap(True);layout.addWidget(self.status_label)
        self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumBlockCount(200);self.log.setMaximumHeight(58);layout.addWidget(self.log)
        self.setCentralWidget(root)
        self.locked_controls=[self.file_btn,self.demo_btn,self.camera_index,self.camera_btn,self.mode,self.site,self.capture_start,self.device,self.line,self.dwell,self.roi_btn,self.roi_clear,self.calib_btn,self.calib_clear,self.model_btn,self.section_btn,self.section_clear]
        for control in [self.conf,self.iou,self.threshold]:control.valueChanged.connect(self.hot_update)
        self.imgsz.currentIndexChanged.connect(self.hot_update)
        self.line.valueChanged.connect(self.update_line)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.viewer.calibration_ready.connect(self.calibration_picked)
        self.mode_changed()

    def write_status(self,text):
        self.status_label.setText(text);self.log.appendPlainText(text)

    def dismiss_guide(self):
        self.guide.setVisible(False)
        try:
            from PySide6.QtCore import QSettings
            QSettings('Cloudlane', '云瞰街巷').setValue('intro_seen', True)
        except Exception:
            pass

    def _set_row_visible(self, widget, visible):
        """Hide a whole form row, label included."""
        widget.setVisible(visible)
        label = self.form.labelForField(widget)
        if label is not None:
            label.setVisible(visible)

    def start_section(self):
        if self.viewer.pixmap.isNull():
            self.write_status('请先打开图片或视频预览，再画断面线。')
            return
        self.viewer._section_start = None
        self.viewer.picking_section = True
        self.viewer.setCursor(Qt.CrossCursor)
        self.section_label.setText('在画面里点第一下（A 点），再点第二下（B 点）。')

    def clear_section(self):
        self.viewer.section = None
        self.viewer._section_start = None
        self.viewer.picking_section = False
        self.viewer.setCursor(Qt.ArrowCursor)
        self.viewer.update()
        self.section_label.setText('未画断面线，不统计穿越。')
        if self.mode.currentData() == 'person':
            self.metric_frames['cross'].setVisible(False)

    def section_picked(self, section):
        self.viewer.section = section
        self.viewer.update()
        self.metric_labels['cross'].setText('断面穿越  A→B / B→A')
        self.metric_frames['cross'].setVisible(True)
        self.section_label.setText('断面线已设置。画面顶部「断面穿越」的两个数分别是两个方向的穿越量，相加即总量。')

    def choose_file(self):
        name,_=QFileDialog.getOpenFileName(self,'打开观测资料','','图片和视频 (*.png *.jpg *.jpeg *.bmp *.webp *.mp4 *.avi *.mov *.mkv *.m4v);;所有文件 (*)')
        if name:self.set_source(name)

    def set_source(self,source):
        if self.worker is not None and self.worker.isRunning():return
        self._preview_generation += 1
        generation = self._preview_generation
        for preview in self._previews:
            preview.requestInterruption()
        self.clear_roi()
        self.clear_calibration()
        self.viewer._start = self.viewer._end = None
        self.viewer.pixmap = QPixmap()
        self.viewer.update()
        self.source=source;self.viewer.roi=None;self.roi_changed(None)
        self.source_label.setText(Path(source).name if isinstance(source,str) else f'摄像头 {source}')
        self.source_label.setToolTip(str(source) if isinstance(source,str) else f'摄像头 {source}')
        if isinstance(source,str):
            preview = PreviewLoader(source,self)
            self.preview_worker = preview
            self._previews.add(preview)
            def ready(image):
                if generation == self._preview_generation and not self.closing and not (self.worker and self.worker.isRunning()):
                    self.viewer.set_frame(image)
            def failure(message):
                if generation == self._preview_generation and not self.closing:
                    self.write_status(message)
            def finished():
                self._previews.discard(preview)
                if self.preview_worker is preview:
                    self.preview_worker = None
                preview.deleteLater()
                if self.closing:
                    QTimer.singleShot(0,self.close)
            preview.ready.connect(ready)
            preview.failure.connect(failure)
            preview.finished.connect(finished)
            preview.start()
        self.write_status('输入已就绪。可设置点位、计数线和观察区域。')

    def choose_camera(self):self.set_source(self.camera_index.value())

    def load_demo(self):
        path = demo_clip_path()
        if path is None:
            self.write_status('未找到随程序分发的示例片段。')
            return
        person_index = self.mode.findData('person')
        if person_index >= 0:
            self.mode.setCurrentIndex(person_index)
        if not self.site.text().strip():
            self.site.setText('校园街角 便利店门前')
        self.set_source(str(path))
        self.write_status('示例片段已载入。点击“开始分析”，结束后会自动生成空间画像。')

    def run_portrait(self):
        if self.portrait_worker is not None and self.portrait_worker.isRunning():
            return
        session_dir = self.last_session_dir
        if not session_dir:
            self.write_status('请先完成一次分析，再生成空间画像。')
            return
        self.portrait_btn.setEnabled(False)
        self.write_status('正在从逐帧记录重建轨迹、热力图与指标…')
        worker = PortraitWorker(session_dir, mask_polygon=self.analysis_mask, parent=self)
        worker.portrait_ready.connect(self.portrait_done)
        worker.portrait_failed.connect(self.portrait_failed)
        self.portrait_worker = worker
        worker.start()

    def portrait_done(self, portrait):
        self.portrait = portrait
        metrics = portrait.metrics
        self.portrait_btn.setEnabled(True)
        for warning in portrait.warnings:
            self.write_status(warning)
        self.write_status(
            f"空间画像完成：{metrics['tracks_after_stitch']} 条轨迹 · "
            f"{metrics['person_seconds']:.0f} 人·秒 · 平均停留 {metrics['mean_dwell_seconds']:.1f} 秒 · "
            f"近距人车交织 {(metrics.get('near_miss') or {}).get('events', 0)} 次")
        session = self.last_summary or {}
        if getattr(self, 'suppress_dialogs', False) or session.get('status') == 'limited':
            return
        PortraitDialog(portrait, self).exec()

    def portrait_failed(self, message):
        self.portrait_btn.setEnabled(True)
        self.write_status(f'空间画像未生成：{message}')

    def choose_model(self):
        path,_=QFileDialog.getOpenFileName(self,'选择本地检测模型',str(self.model_path.parent),'PyTorch模型 (*.pt)')
        if path:self.model_path=Path(path);self.write_status(f'已选择模型：{self.model_path.name}')

    def mode_changed(self, *_):
        person = self.mode.currentData() == 'person'
        self.model_path = default_space_model_path() if person else default_model_path()
        size = 1024 if person else 640
        index = self.imgsz.findData(size)
        if index >= 0:
            self.imgsz.setCurrentIndex(index)
        self.metric_labels['current'].setText('画面内人/车' if person else '画面内车辆')
        self.metric_values['current'].setToolTip(
            '行人、自行车、电动车合计；分类数量见下方明细，远处小目标可能漏检。' if person
            else '当前帧检出的车辆数，不是累计车流。')
        self.chart.caption = ('画面内人/车数 · 源视频时间（秒）' if person
                              else '画面内车辆数 · 源视频时间（秒）')
        self.chart.update()
        # The crossing line counts vehicles passing a section; for a space-use
        # portrait it only adds a row of zeros, so it is hidden with the scene.
        self._set_row_visible(self.line, not person)
        self.metric_labels['cross'].setText('过线  ↑ / ↓' if not person else '断面穿越  A→B / B→A')
        self.metric_frames['cross'].setVisible((not person) or self.viewer.section is not None)
        self.section_box.setVisible(person)
        self.viewer.show_line = not person
        self.viewer.update()
        missing = not self.model_path.is_file()
        self.write_status(
            ('人行观测：使用通用行人/非机动车权重（COCO），默认 1024 输入。' if person else '车辆观测：使用四类机动车权重。')
            + ('未找到对应权重，请点击“更换本地模型”。' if missing else ''))

    def start_calibration(self):
        if self.viewer.pixmap.isNull():
            self.write_status('请先打开图片或视频预览，再做四点标定。')
            return
        self.viewer.calibration_points = []
        self.viewer.picking_calibration = True
        self.viewer.setCursor(Qt.CrossCursor)
        self.calib_label.setText('依次点选：左下 → 右下 → 右上 → 左上 → 矩形中心（第 5 点用于校核）')

    def clear_calibration(self):
        self.calibration = None
        self.calibration_path = None
        self.viewer.calibration_points = []
        self.viewer.picking_calibration = False
        self.viewer.setCursor(Qt.ArrowCursor)
        self.viewer.update()
        self.calib_label.setText('未标定：按画面比例计算')

    def calibration_picked(self, points):
        width, accepted = QInputDialog.getDouble(self, '地面标定', '矩形实际宽度（米）', 6.0, 0.5, 100.0, 2)
        if not accepted:
            self.viewer.calibration_points = []
            self.viewer.update()
            return
        height, accepted = QInputDialog.getDouble(self, '地面标定', '矩形实际高度（米）', 4.0, 0.5, 100.0, 2)
        if not accepted:
            self.viewer.calibration_points = []
            self.viewer.update()
            return
        pixmap = self.viewer.pixmap
        corners = points[:4]
        image_points = [(x * pixmap.width(), y * pixmap.height()) for x, y in corners]
        world_points = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
        check_image = check_world = None
        if len(points) >= 5:
            check_image = (points[4][0] * pixmap.width(), points[4][1] * pixmap.height())
            check_world = (width / 2.0, height / 2.0)
        try:
            calibration = Calibration.from_points(
                image_points, world_points,
                notes=f'{self.site.text().strip() or "未命名点位"} {width:g}x{height:g} m',
                check_image=check_image, check_world=check_world)
        except CalibrationError as exc:
            self.write_status(f'标定失败：{exc}')
            self.viewer.calibration_points = []
            self.viewer.update()
            return
        folder = self.data_root / 'calibrations'
        folder.mkdir(parents=True, exist_ok=True)
        name = (self.site.text().strip() or 'site').replace(' ', '_')
        path = folder / f'{name}_{datetime.now():%Y%m%d_%H%M%S}.json'
        calibration.save(path)
        self.calibration = calibration
        self.calibration_path = path
        check = calibration.check_error_m
        if check is None:
            quality = f'标定 {width:g}×{height:g} m'
        else:
            quality = f'标定 {width:g}×{height:g} m，中心校核误差 {check * 100:.1f} cm'
        self.calib_label.setText(f'{quality}\n{path.name}')
        self.write_status('标定完成：分析结束后会在会话目录写入 calibration.json。')

    def update_line(self):self.viewer.line_y=self.line.value()/100;self.viewer.update()

    def select_roi(self):
        if self.viewer.pixmap.isNull():self.write_status('请先打开图片或视频预览，再框选区域。');return
        self.viewer.selecting=True;self.viewer.setCursor(Qt.CrossCursor);self.write_status('在画面中按住鼠标左键，拖动框选观察区域。')

    def clear_roi(self):
        self.viewer.roi=None;self.viewer.selecting=False
        self.viewer._start=self.viewer._end=None
        self.viewer.setCursor(Qt.ArrowCursor);self.roi_changed(None);self.viewer.update()

    def roi_changed(self,roi):
        self.roi_label.setText('已设置观察区域，按轨迹底部中心连续驻留计时' if roi else '未设置区域，区域统计不启用')

    def hot_update(self):
        self.chart.threshold=self.threshold.value();self.chart.update()
        if self.worker is not None and self.worker.isRunning():
            self.worker.update_config(conf=self.conf.value(),iou=self.iou.value(),imgsz=self.imgsz.currentData(),density_threshold=self.threshold.value())

    def set_running(self,running):
        for control in self.locked_controls:control.setEnabled(not running)
        self.start_btn.setEnabled(not running);self.stop_btn.setEnabled(running)
        self.portrait_btn.setEnabled((not running) and bool(self.last_session_dir))

    def start_analysis(self):
        if self.worker is not None and self.worker.isRunning():return
        self.guide.setVisible(False)
        if self.source is None:self.write_status('请先选择图片、视频或摄像头。');return
        if not self.model_path.is_file():self.write_status('未找到模型，请点击“更换本地模型”选择 best.pt。');return
        try:
            start_text=self.capture_start.text().strip() or None
            if start_text:
                from datetime import datetime
                datetime.fromisoformat(start_text)
            config=EngineConfig(source=self.source,model_path=str(self.model_path),device=self.device.currentData(),
                conf=self.conf.value(),iou=self.iou.value(),imgsz=self.imgsz.currentData(),density_threshold=self.threshold.value(),
                record_dir=str(self.sessions_root),site_id=self.site.text().strip(),capture_start=start_text,
                classes=(('person','bicycle','motorcycle') if self.mode.currentData()=='person' else None),
                max_frames=self.test_max_frames,analytics_config=AnalyticsConfig(line_y=self.line.value()/100,
                    roi=self.viewer.roi,dwell_threshold_seconds=self.dwell.value(),
                    section=(self.viewer.section if self.mode.currentData()=='person' else None)))
            # The heat map is limited to the box the observer drew, so colour
            # never bleeds onto facades, trees or the sky.
            roi = self.viewer.roi
            self.analysis_mask = ([(roi[0], roi[1]), (roi[2], roi[1]),
                                   (roi[2], roi[3]), (roi[0], roi[3])] if roi else None)
            if self.worker is not None:
                self.worker.deleteLater()
            self._preview_generation += 1
            for preview in self._previews:
                preview.requestInterruption()
            self.worker=DetectionWorker(config,parent=self)
            self.worker.status.connect(self.write_status);self.worker.error.connect(lambda message:self.write_status('分析失败：'+message))
            self.worker.session_finished.connect(self.analysis_completed);self.worker.finished.connect(self.thread_finished)
            self.chart.points.clear();self.chart.update();self.progress.setValue(0);self.last_summary=None;self.latest_packet=None
            self.viewer.selecting=False;self.viewer.setCursor(Qt.ArrowCursor)
            self.viewer._start = self.viewer._end = None
            for value in self.metric_values.values():value.setText('—')
            self.set_running(True);self.worker.start()
            self.write_status('正在加载模型。加载、预热和推理均在后台进行。')
        except Exception as exc:self.write_status(f'无法开始：{exc}');self.set_running(False)

    def stop_analysis(self):
        if self.worker is not None and self.worker.isRunning():
            self.stop_btn.setEnabled(False);self.worker.stop();self.write_status('正在结束当前帧并保存结果，请稍候…')

    def poll_frame(self):
        if self.worker is None:return
        packet=self.worker.latest_packet()
        if packet is None:return
        self.latest_packet=packet;self.viewer.set_frame(packet.image)
        s=packet.stats
        self.metric_values['current'].setText(str(s.current_count));self.metric_values['unique'].setText(str(s.cumulative_unique_count))
        self.metric_values['cross'].setText(f"{s.line_counts.get('up',0)} / {s.line_counts.get('down',0)}")
        self.metric_values['dwell'].setText(str(s.dwell_current_count));self.metric_values['fps'].setText(f'{packet.processing_fps:.1f}')
        self.metric_values['current'].setStyleSheet('color: #ff9e9e' if packet.density_exceeded else 'color: #48d7c3')
        self.chart.add(packet.source_time_sec,s.current_count)
        self.progress.setValue(int(packet.progress*1000) if packet.progress is not None else 0)
        classes=' · '.join(f'{key}: {value}' for key,value in s.counts_by_class.items()) or '无检出目标'
        self.detail.setText(f'源时间 {packet.source_time_sec:.2f} 秒  |  第 {packet.frame_index} 帧  |  {packet.processing_ms:.1f} ms/帧  |  {classes}')

    def analysis_completed(self,summary):
        if self.sender() is not self.worker:return
        self.poll_frame();self.last_summary=summary
        session_dir = summary.get('session_dir')
        if session_dir and self.calibration_path and Path(self.calibration_path).is_file():
            try:
                shutil.copyfile(self.calibration_path, Path(session_dir) / 'calibration.json')
                self.write_status('标定文件已写入会话目录，可用于离线轨迹分析。')
            except OSError as exc:
                self.write_status(f'标定文件复制失败：{exc}')
        if session_dir and Path(session_dir).is_dir():
            self.last_session_dir = Path(session_dir)
            self.portrait_btn.setEnabled(True)
            # The portrait is the point of a session, so build it without
            # asking the user to press a second button. Scheduled for the next
            # event-loop turn rather than on a delay, so a scripted run that
            # closes on completion still exercises it.
            QTimer.singleShot(0, self.run_portrait)
        status=summary.get('status');labels={'completed':'分析完成','stopped':'已停止并保存','limited':'验证片段已完成','failed':'分析失败，已有结果已尽力保存'}
        self.write_status(f"{labels.get(status,status)} · {summary.get('processed_frames',0)} 帧 · 设备 {summary.get('device','未知')}")
        self.session_completed.emit(summary)

    def thread_finished(self):
        if self.sender() is not self.worker:return
        self.poll_frame();self.set_running(False)
        if self.closing:QTimer.singleShot(0,self.close)

    def export_chart(self):
        folder=self.data_root/'charts';folder.mkdir(parents=True,exist_ok=True)
        from datetime import datetime
        path=folder/('数量曲线_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.png')
        if self.chart.grab().save(str(path)):self.write_status(f'曲线已保存：{path}')

    @staticmethod
    def open_folder(path):QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve())))

    def show_history(self):
        dialog=QDialog(self);dialog.setWindowTitle('历史结果 · 直接查看已保存摘要');dialog.resize(880,560)
        layout=QVBoxLayout(dialog);items=QListWidget();items.setSelectionMode(QAbstractItemView.SingleSelection)
        entries=list_history(self.sessions_root)
        for item in entries:
            source=item.get('source','')
            source=f'摄像头 {source}' if isinstance(source,int) else Path(str(source)).name
            status={'completed':'已完成','stopped':'已停止','limited':'验证完成','failed':'失败'}.get(item.get('status'),item.get('status',''))
            label=f"{item.get('started_at_utc',item.get('created_at_utc',''))}  |  {source}  |  {status}"
            items.addItem(label)
        layout.addWidget(items,1);details=QPlainTextEdit();details.setReadOnly(True);layout.addWidget(details,2)
        mode=QComboBox();mode.addItem('中文结果摘要',False);mode.addItem('原始摘要 JSON',True)
        actions=QHBoxLayout();actions.addWidget(mode)
        open_btn=QPushButton('打开该次 CSV 数据目录');open_btn.setEnabled(bool(entries));actions.addWidget(open_btn)
        layout.addLayout(actions)
        def show(row):
            if row<0:return
            item=entries[row];path=Path(item.get('session_dir',''))
            try:
                summary=read_session_summary(path)
                details.setPlainText(json.dumps(summary,ensure_ascii=False,indent=2) if mode.currentData() else readable_session_summary(summary))
            except Exception as exc:details.setPlainText(str(exc))
        items.currentRowChanged.connect(show)
        mode.currentIndexChanged.connect(lambda:show(items.currentRow()))
        open_btn.clicked.connect(lambda:self.open_folder(entries[items.currentRow()].get('session_dir',self.sessions_root)) if items.currentRow()>=0 else None)
        if entries:items.setCurrentRow(0)
        else:details.setPlainText('暂无历史结果。完成一次分析后，会自动保存摘要和逐帧CSV。')
        dialog.exec()

    def closeEvent(self,event):
        if self.worker is not None and self.worker.isRunning():
            self.closing=True;self.worker.stop();self.write_status('正在安全停止分析并保存结果，完成后自动关闭。')
            event.ignore();return
        if self.portrait_worker is not None and self.portrait_worker.isRunning():
            self.closing=True
            self.portrait_worker.finished.connect(lambda: QTimer.singleShot(0,self.close))
            self.write_status('正在完成空间画像，完成后自动关闭。')
            event.ignore();return
        running_previews = [preview for preview in self._previews if preview.isRunning()]
        if running_previews:
            self.closing = True
            for preview in running_previews:
                preview.requestInterruption()
            event.ignore();return
        event.accept()
