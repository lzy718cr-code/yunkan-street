import csv
import tempfile
from pathlib import Path
import unittest

from traffic_app.calibration import Calibration
from traffic_app.space import analyse_session, load_tracks, stitch_tracks, summarise


FIELDS = ["frame_index", "source_time_seconds", "track_id", "class_id", "class_name",
          "confidence", "x1", "y1", "x2", "y2",
          "bottom_center_x_normalized", "bottom_center_y_normalized"]


def write_session(folder: Path, rows):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "frames.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_index", "source_time_seconds",
                                                    "width", "height"])
        writer.writeheader()
        writer.writerow({"frame_index": 1, "source_time_seconds": 0,
                         "width": 1000, "height": 500})
    with (folder / "detections.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def detection(frame, timestamp, track_id, x, y, name="person"):
    return {"frame_index": frame, "source_time_seconds": timestamp,
            "track_id": track_id, "class_id": 0, "class_name": name,
            "confidence": 0.9, "x1": x - 10, "y1": y - 40, "x2": x + 10, "y2": y,
            "bottom_center_x_normalized": x / 1000, "bottom_center_y_normalized": y / 500}


class SpaceAnalysisTest(unittest.TestCase):
    def test_loads_and_orders_tracks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_session(root, [detection(i, i / 10, 7, 100 + i, 400) for i in range(1, 11)])
            tracks, unit, _stats = load_tracks(root, class_names=["person"])
        self.assertEqual(unit, "frame_fraction")
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0].points), 10)
        self.assertAlmostEqual(tracks[0].duration, 0.9)

    def test_calibrated_coordinates_are_metres(self):
        calibration = Calibration.from_points(
            [(0, 0), (1000, 0), (1000, 500), (0, 500)],
            [(0, 0), (10, 0), (10, 5), (0, 5)])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_session(root, [detection(1, 0.0, 1, 500, 250),
                                 detection(2, 0.1, 1, 500, 250)])
            tracks, unit, _stats = load_tracks(root, calibration, ["person"])
        self.assertEqual(unit, "metre")
        self.assertAlmostEqual(tracks[0].points[0].x, 5.0, places=3)
        self.assertAlmostEqual(tracks[0].points[0].y, 2.5, places=3)

    def test_points_outside_calibration_are_dropped(self):
        calibration = Calibration.from_points(
            [(100, 100), (900, 100), (900, 400), (100, 400)],
            [(0, 0), (8, 0), (8, 3), (0, 3)])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_session(root, [detection(1, 0.0, 1, 500, 250),
                                 detection(2, 0.1, 1, 502, 250),
                                 detection(3, 0.2, 2, 50, 250)])
            tracks, _unit, stats = load_tracks(root, calibration, ["person"])
        self.assertEqual(stats["detections_outside_calibration"], 1)
        self.assertEqual([track.track_id for track in tracks], [1])

    def test_stitches_two_fragments(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [detection(i, i / 10, 1, 300, 400) for i in range(1, 6)]
            rows += [detection(i, i / 10, 2, 302, 401) for i in range(8, 13)]
            write_session(root, rows)
            tracks, _unit, _stats = load_tracks(root, class_names=["person"])
        merged, mapping = stitch_tracks(tracks, max_gap_seconds=2.0, max_distance=0.05)
        self.assertEqual(len(merged), 1)
        self.assertEqual(mapping[2], 1)

    def test_summary_reports_person_seconds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_session(root, [detection(i, i / 2, 1, 300, 400) for i in range(0, 5)])
            tracks, unit, _stats = load_tracks(root, class_names=["person"])
        merged, mapping = stitch_tracks(tracks)
        summary = summarise(tracks, unit, cell_size=0.02, merged=merged, mapping=mapping)
        self.assertEqual(summary["tracks_raw"], 1)
        self.assertAlmostEqual(summary["person_seconds"], 2.0)
        self.assertGreaterEqual(summary["occupied_cells"], 1)

    def test_full_analysis_writes_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "session"
            write_session(root, [detection(i, i / 5, 1, 300 + i, 400) for i in range(1, 16)])
            out = Path(folder) / "out"
            metrics = analyse_session(root, out, class_names=["person"], cell_size=0.02)
            self.assertTrue((out / "tracks.csv").is_file())
            self.assertTrue((out / "plan_view.png").is_file())
            self.assertTrue((out / "metrics.json").is_file())
            self.assertEqual(metrics["coordinate_unit"], "frame_fraction")


if __name__ == "__main__":
    unittest.main()

