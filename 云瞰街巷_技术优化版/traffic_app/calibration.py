"""Ground-plane calibration for micro public space observation.

Maps image pixels to a metric ground plane with a homography estimated from
four or more surveyed control points. Every distance produced downstream is
only as good as those control points, so the reprojection error is reported
with the matrix and a session can be rejected instead of silently trusted.

Conventions
-----------
* image points: (x, y) pixels, origin top-left, y down.
* world points: (x, y) metres on the surveyed ground plane, y north/up.
* a homography is stored row-major as three rows of three floats.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


class CalibrationError(ValueError):
    """Raised when control points cannot support a trustworthy homography."""


def _as_points(points, label: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise CalibrationError(f"{label} 必须是 N×2 的坐标点")
    if array.shape[0] < 4:
        raise CalibrationError(f"{label} 至少需要 4 个点")
    if not np.isfinite(array).all():
        raise CalibrationError(f"{label} 含非有限数值")
    if len(np.unique(array, axis=0)) != array.shape[0]:
        raise CalibrationError(f"{label} 存在重复点")
    return array


def _normalizer(points: np.ndarray) -> np.ndarray:
    """Hartley normalization: centroid at origin, mean distance sqrt(2)."""
    centroid = points.mean(axis=0)
    distance = np.sqrt(((points - centroid) ** 2).sum(axis=1)).mean()
    if distance <= 1e-12:
        raise CalibrationError("控制点退化，无法归一化")
    scale = math.sqrt(2.0) / distance
    return np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ])


def _dlt(image_points: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    """Normalized direct linear transform for a plane-to-plane homography."""
    source = _normalizer(image_points)
    target = _normalizer(world_points)
    def apply(matrix, pts):
        hom = np.column_stack([pts, np.ones(len(pts))]) @ matrix.T
        return hom[:, :2] / hom[:, 2:3]

    src = apply(source, image_points)
    dst = apply(target, world_points)
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    matrix = np.asarray(rows, dtype=float)
    _, singular, vt = np.linalg.svd(matrix)
    # A valid configuration has exactly one null direction: rank 8 whether the
    # system is 8x9 (four points) or 2n x 9 (more points).
    rank = int((singular > max(singular[0], 1e-12) * 1e-10).sum())
    if rank < 8:
        raise CalibrationError("控制点接近共线或共点，单应矩阵不稳定")
    homography = vt[-1].reshape(3, 3)
    homography = np.linalg.inv(target) @ homography @ source
    homography /= homography[2, 2]
    if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-12:
        raise CalibrationError("单应矩阵不可逆")
    return homography


def _project(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    hom = np.column_stack([points, np.ones(len(points))]) @ homography.T
    return hom[:, :2] / hom[:, 2:3]


@dataclass(frozen=True)
class Calibration:
    """A surveyed pixel-to-metre mapping with its measured error."""

    image_points: tuple[tuple[float, float], ...]
    world_points: tuple[tuple[float, float], ...]
    homography: tuple[tuple[float, float, float], ...]
    reprojection_rmse: float
    max_reprojection_error: float
    check_error_m: float | None = None
    notes: str = ""

    @classmethod
    def from_points(cls, image_points, world_points, notes: str = "",
                    check_image=None, check_world=None) -> "Calibration":
        image = _as_points(image_points, "图像点")
        world = _as_points(world_points, "世界点")
        if image.shape != world.shape:
            raise CalibrationError("图像点与世界点数量必须一致")
        # A tiny source area means the four points collapse in the image.
        ax, ay = image[1] - image[0]
        bx, by = image[2] - image[0]
        area = 0.5 * abs(ax * by - ay * bx)
        if area < 25.0:
            raise CalibrationError("控制点在画面中覆盖面积过小（<25 像素²）")
        matrix = _dlt(image, world)
        residuals = np.linalg.norm(_project(matrix, image) - world, axis=1)
        check_error = None
        if check_image is not None and check_world is not None:
            predicted = _project(matrix, np.asarray([check_image], dtype=float))[0]
            check_error = float(np.linalg.norm(predicted - np.asarray(check_world, dtype=float)))
        return cls(
            image_points=tuple((float(x), float(y)) for x, y in image),
            world_points=tuple((float(x), float(y)) for x, y in world),
            homography=tuple(tuple(float(v) for v in row) for row in matrix),
            reprojection_rmse=float(np.sqrt((residuals ** 2).mean())),
            max_reprojection_error=float(residuals.max()),
            check_error_m=check_error,
            notes=notes,
        )

    def to_world(self, x: float, y: float) -> tuple[float, float]:
        matrix = np.asarray(self.homography, dtype=float)
        hom = matrix @ np.array([float(x), float(y), 1.0])
        if abs(hom[2]) < 1e-12:
            raise CalibrationError("该像素点映射到无穷远，标定无法覆盖")
        return float(hom[0] / hom[2]), float(hom[1] / hom[2])

    def contains_image(self, x: float, y: float) -> bool:
        """True when a pixel lies inside the surveyed control polygon.

        Outside that polygon the homography is an extrapolation: it is still
        mathematically defined but no longer checked by any control point, so
        callers should drop or flag those observations.
        """
        polygon = np.asarray(self.image_points, dtype=float)
        inside = False
        count = len(polygon)
        previous = count - 1
        for current in range(count):
            xi, yi = polygon[current]
            xj, yj = polygon[previous]
            if (yi > y) != (yj > y):
                crossing = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < crossing:
                    inside = not inside
            previous = current
        return inside

    def to_dict(self) -> dict:
        return {
            "image_points": [list(p) for p in self.image_points],
            "world_points": [list(p) for p in self.world_points],
            "homography": [list(row) for row in self.homography],
            "reprojection_rmse_m": self.reprojection_rmse,
            "max_reprojection_error_m": self.max_reprojection_error,
            "check_error_m": self.check_error_m,
            "notes": self.notes,
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(
            image_points=tuple(tuple(float(v) for v in p) for p in payload["image_points"]),
            world_points=tuple(tuple(float(v) for v in p) for p in payload["world_points"]),
            homography=tuple(tuple(float(v) for v in row) for row in payload["homography"]),
            reprojection_rmse=float(payload["reprojection_rmse_m"]),
            max_reprojection_error=float(payload["max_reprojection_error_m"]),
            check_error_m=(float(payload["check_error_m"])
                           if payload.get("check_error_m") is not None else None),
            notes=str(payload.get("notes", "")),
        )


def grid_control_points(origin_x: float, origin_y: float, width: float, height: float,
                        image_quad, notes: str = "") -> Calibration:
    """Convenience wrapper for a rectangular patch measured on site.

    ``image_quad`` must list the four image corners in the same order as the
    world rectangle: (origin), (+width), (+width, +height), (+height).
    """
    world = [
        (origin_x, origin_y),
        (origin_x + width, origin_y),
        (origin_x + width, origin_y + height),
        (origin_x, origin_y + height),
    ]
    return Calibration.from_points(image_quad, world, notes=notes)
