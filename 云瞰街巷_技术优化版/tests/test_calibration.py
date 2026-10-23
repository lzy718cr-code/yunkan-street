import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from traffic_app.calibration import Calibration, CalibrationError


class CalibrationTest(unittest.TestCase):
    def test_recovers_a_known_homography(self):
        world = np.array([[0.0, 0.0], [12.0, 0.0], [12.0, 7.0], [0.0, 7.0],
                          [6.0, 3.5]])
        matrix = np.array([[74.0, 9.0, 130.0],
                           [4.0, 58.0, 90.0],
                           [0.012, 0.018, 1.0]])
        homogeneous = np.column_stack([world, np.ones(len(world))]) @ matrix.T
        image = homogeneous[:, :2] / homogeneous[:, 2:3]
        calibration = Calibration.from_points(image, world)
        self.assertLess(calibration.reprojection_rmse, 1e-6)
        for point, expected in zip(image, world):
            recovered = calibration.to_world(*point)
            self.assertAlmostEqual(recovered[0], expected[0], places=6)
            self.assertAlmostEqual(recovered[1], expected[1], places=6)

    def test_rejects_collinear_points(self):
        image = [(0, 0), (10, 10), (20, 20), (30, 30)]
        world = [(0, 0), (1, 0), (2, 0), (3, 0)]
        with self.assertRaises(CalibrationError):
            Calibration.from_points(image, world)

    def test_rejects_too_few_points(self):
        with self.assertRaises(CalibrationError):
            Calibration.from_points([(0, 0), (10, 0), (10, 10)],
                                    [(0, 0), (1, 0), (1, 1)])

    def test_check_point_reports_centre_error(self):
        world = np.array([[0.0, 0.0], [12.0, 0.0], [12.0, 7.0], [0.0, 7.0],
                          [6.0, 3.5]])
        matrix = np.array([[74.0, 9.0, 130.0],
                           [4.0, 58.0, 90.0],
                           [0.012, 0.018, 1.0]])
        homogeneous = np.column_stack([world, np.ones(len(world))]) @ matrix.T
        image = homogeneous[:, :2] / homogeneous[:, 2:3]
        calibration = Calibration.from_points(image[:4], world[:4],
                                              check_image=image[4], check_world=world[4])
        self.assertIsNotNone(calibration.check_error_m)
        self.assertLess(calibration.check_error_m, 1e-6)
        shifted = Calibration.from_points(image[:4], world[:4],
                                          check_image=image[4] + np.array([8.0, 0.0]),
                                          check_world=world[4])
        self.assertGreater(shifted.check_error_m, 0.05)

    def test_round_trip_json(self):
        calibration = Calibration.from_points(
            [(20, 40), (300, 40), (300, 260), (20, 260)],
            [(0, 0), (8, 0), (8, 6), (0, 6)], notes="test")
        with tempfile.TemporaryDirectory() as folder:
            path = calibration.save(Path(folder) / "calibration.json")
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["notes"], "test")
            again = Calibration.load(path)
        self.assertAlmostEqual(again.reprojection_rmse, calibration.reprojection_rmse)
        self.assertAlmostEqual(again.to_world(160, 150)[0],
                               calibration.to_world(160, 150)[0])


if __name__ == "__main__":
    unittest.main()
