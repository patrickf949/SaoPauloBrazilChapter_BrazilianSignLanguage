"""
Unit tests for landmark-based estimators: AnglesEstimator, DistancesEstimator, DifferencesEstimator, and LandmarkEstimator.

This test suite verifies the correctness of the following estimator classes:
- AnglesEstimator: Computes angles between triplets of pose or hand landmarks.
- DistancesEstimator: Computes distances between pairs of pose or hand landmarks.
- DifferencesEstimator: Computes frame-to-frame movement vectors for pose or hand landmarks.
- LandmarkEstimator: Extracts specific landmark coordinates as feature vectors.

Test coverage includes:
- Basic functionality: correct output format and values
- Annotated versions: returning results with proper feature names
- Edge case handling: invalid landmark types or modes

Mock data is used to simulate landmark inputs and computation results.

Usage:
-------
Run the tests with:

    python -m unittest test_estimators.py

To check code coverage:

    coverage run -m unittest discover
    coverage report -m
    coverage html  # (optional) for visual report in browser

Dependencies:
-------------
- Python standard library: unittest, types
- Third-party: numpy, coverage (optional)

"""

import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from types import SimpleNamespace
from models.landmark.dataset.base_estimator import LandmarkEstimator
from models.landmark.dataset.distances_estimator import DistancesEstimator
from models.landmark.dataset.frame2frame_differences_estimator import (
    DifferencesEstimator,
)
from models.landmark.dataset.angles_estimator import AnglesEstimator


class TestEstimators(unittest.TestCase):
    def setUp(self):
        self.landmarks = [
            SimpleNamespace(x=1.0, y=2.0, z=3.0),
            SimpleNamespace(x=2.0, y=3.0, z=4.0),
            SimpleNamespace(x=3.0, y=4.0, z=5.0),
        ]
        self.prev_landmarks = [
            SimpleNamespace(x=0.0, y=1.0, z=2.0),
            SimpleNamespace(x=1.0, y=2.0, z=3.0),
            SimpleNamespace(x=2.0, y=3.0, z=4.0),
        ]
        self.next_landmarks = self.landmarks

        self.angle_cfg = {"angle1": (0, 1, 2)}
        self.dist_cfg = {"dist1": (0, 2)}
        self.diff_cfg = {"diff1": 1}
        self.landmark_cfg = {"pos1": 0, "pos2": 1}

        self.landmark_estimator = LandmarkEstimator(
            self.landmark_cfg, self.landmark_cfg
        )

        self.angle_estimator = AnglesEstimator(self.angle_cfg, self.angle_cfg)
        self.dist_estimator = DistancesEstimator(self.dist_cfg, self.dist_cfg)
        self.diff_estimator = DifferencesEstimator(self.diff_cfg, self.diff_cfg)

    def test_angle_compute(self):
        angles = self.angle_estimator.compute(self.landmarks, "pose", "2D", "func")
        self.assertEqual(len(angles), 2)
        assert_almost_equal(angles[1], -1)

    def test_angle_annotated(self):
        result = self.angle_estimator.compute_annotated(
            self.landmarks, "2D", "hand", "func"
        )
        self.assertIn("angle1", result)
        assert_almost_equal(result["angle1"][1], -1)

    def test_distance_compute(self):
        dists = self.dist_estimator.compute(self.landmarks, "pose", "3D")
        self.assertEqual(len(dists), 1)
        self.assertEqual(dists[0], 1.0)

    def test_distance_annotated(self):
        result = self.dist_estimator.compute_annotated(self.landmarks, "hand", "3D")
        self.assertIn("dist1", result)
        self.assertEqual(result["dist1"], 1.0)

    def test_difference_compute(self):
        diffs = self.diff_estimator.compute(
            self.prev_landmarks, self.next_landmarks, "pose", "3D", "diff"
        )
        self.assertEqual(len(diffs), 3)
        assert_array_almost_equal(list(diffs), [1.0, 1.0, 1.0])

        diffs = self.diff_estimator.compute(
            self.prev_landmarks, self.next_landmarks, "pose", "3D", "normalized_diff"
        )
        self.assertEqual(len(diffs), 3)
        assert_array_almost_equal(list(diffs), [0.5, 0.5, 0.5])

    def test_difference_annotated(self):
        result = self.diff_estimator.compute_annotated(
            self.prev_landmarks, self.next_landmarks, "hand", "2D", "diff"
        )
        self.assertIn("diff1", result)
        self.assertEqual(result["diff1"], (1.0, 1.0))
        result = self.diff_estimator.compute_annotated(
            self.prev_landmarks, self.next_landmarks, "hand", "2D", "normalized_diff"
        )
        self.assertIn("diff1", result)
        self.assertEqual(result["diff1"], (0.5, 0.5))

    def test_landmark_compute(self):
        result = self.landmark_estimator.compute(self.landmarks, "hand", "3D")
        expected = np.array([1.0, 2.0, 3.0, 2.0, 3.0, 4.0])
        assert_array_almost_equal(result, expected)

        result = self.landmark_estimator.compute(self.landmarks, "hand", "2D")
        expected = np.array([1.0, 2.0, 2.0, 3.0])
        assert_array_almost_equal(result, expected)

    def test_landmark_annotated(self):
        result = self.landmark_estimator.compute_annotated(self.landmarks, "pose", "3D")
        expected = {"pos1": [1.0, 2.0, 3.0], "pos2": [2.0, 3.0, 4.0]}
        self.assertEqual(result, expected)
        result = self.landmark_estimator.compute_annotated(self.landmarks, "hand", "2D")
        expected = {"pos1": [1.0, 2.0], "pos2": [2.0, 3.0]}
        self.assertEqual(result, expected)

    def test_invalid_landmark_type(self):
        with self.assertRaises(ValueError):
            self.diff_estimator.compute(
                self.prev_landmarks, self.landmarks, "foot", "2D"
            )
            self.angle_estimator.compute(self.landmarks, "foot", "2D")
            self.diff_estimator.compute(self.landmarks, "foot", "2D")
            self.landmark_estimator.compute(self.landmarks, "foot", "2D")

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            self.diff_estimator.compute(
                self.prev_landmarks, self.landmarks, "hand", "4D"
            )
            self.angle_estimator.compute(self.landmarks, "hand", "4D")
            self.diff_estimator.compute(self.landmarks, "hand", "4D")
            self.landmark_estimator.compute(self.landmarks, "hand", "4D")

    def test_none_landmarks(self):
        result = self.landmark_estimator.compute(None, "pose", "2D")
        expected = [0.0, 0.0, 0.0, 0.0]
        assert_array_almost_equal(result, expected)
        result = self.landmark_estimator.compute(None, "hand", "3D")
        expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert_array_almost_equal(result, expected)

        result = self.angle_estimator.compute(None, "pose", "2D", "func")
        expected = [0.0, 0.0]
        assert_array_almost_equal(result, expected)
        result = self.angle_estimator.compute(None, "hand", "3D", "rad")
        expected = [0.0]
        assert_array_almost_equal(result, expected)

        result = self.dist_estimator.compute(None, "pose", "2D")
        expected = [0.0]
        assert_array_almost_equal(result, expected)
        result = self.dist_estimator.compute(None, "hand", "3D")
        expected = [0.0]
        assert_array_almost_equal(result, expected)

        result = self.diff_estimator.compute(None, self.next_landmarks, "pose", "2D")
        expected = [0.0, 0.0]
        assert_array_almost_equal(result, expected)
        result = self.diff_estimator.compute(None, self.next_landmarks, "hand", "3D")
        expected = [0.0, 0.0, 0.0]
        assert_array_almost_equal(result, expected)
        result = self.diff_estimator.compute(self.prev_landmarks, None, "pose", "2D")
        expected = [0.0, 0.0]
        assert_array_almost_equal(result, expected)
        result = self.diff_estimator.compute(None, None, "hand", "3D")
        expected = [0.0, 0.0, 0.0]
        assert_array_almost_equal(result, expected)
