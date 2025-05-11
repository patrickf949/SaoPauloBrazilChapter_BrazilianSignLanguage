from typing import Union, Dict, List, Tuple, Iterable
from models.landmark.utils.utils import (
    check_mode,
    check_difference_type,
    check_landmark_type,
)
import numpy as np
from models.landmark.features.base_estimator import BaseEstimator

from omegaconf import DictConfig


def difference(
    p1, p2, mode: str = "3D", diff_type: str = "diff"
) -> Union[Tuple[float, float], Tuple[float, float, float]]:
    """
    Computes the difference vector (movement) between two points.

    Parameters:
    ----------
    p1, p2 : objects
        Landmarks with attributes x, y, and optionally z.
    mode : str
        '2D' or '3D' — whether to include the z-dimension.
    diff_type : str
        - 'diff'            : raw delta vector (dx, dy, dz)
        - 'normalized_diff' : difference rescaled to [-1, 1] based on max possible shift (2.0)

    Returns:
    -------
    Tuple of float(s) representing the movement vector between p1 and p2.
    """
    check_mode(mode)
    check_difference_type(diff_type)

    if mode == "3D":
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        if diff_type == "normalized_diff":
            return dx / 2.0, dy / 2.0, dz / 2.0  # range [-1, 1]
        else:
            return dx, dy, dz
    else:
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        if diff_type == "normalized_diff":
            return dx / 2.0, dy / 2.0
        else:
            return dx, dy


class DifferencesEstimator(BaseEstimator):
    """
    Estimates frame-to-frame movement vectors (differences) for pose and hand landmarks.
    """

    def __init__(
        self,
        hand_differences: Union[str, Dict, DictConfig],
        pose_differences: Union[str, Dict, DictConfig],
    ):
        """
        Parameters:
        ----------
        hand_differences : str or dict
            Path to YAML file or dictionary with hand landmark indices.
        pose_differences : str or dict
            Path to YAML file or dictionary with pose landmark indices.
        """
        super().__init__(hand_differences, pose_differences, config_type="differences")

    def __compute_differences(
        self,
        landmark_indices: List[int],
        prev_landmarks: Iterable,
        next_landmarks: Iterable,
        mode: str,
        diff_type: str,
    ) -> List[Tuple[float]]:
        if prev_landmarks is None or next_landmarks is None:
            if mode == "3D":
                return np.zeros(shape=3 * len(landmark_indices))
            else:
                return np.zeros(shape=2 * len(landmark_indices))
        return [
            difference(next_landmarks[idx], prev_landmarks[idx], mode, diff_type)
            for idx in landmark_indices
        ]

    def __convert_to_numpy(self, features: List[Tuple[float]]) -> np.ndarray:
        return np.array(features).flatten()

    def compute(
        self,
        prev_landmarks: Iterable,
        next_landmarks: Iterable,
        landmark_type: str,
        mode: str,
        computation_type: str = "normalized_diff",
    ) -> np.ndarray:
        """
        Compute raw or normalized difference vectors for specified landmark type.

        Parameters:
        ----------
        prev_landmarks : Iterable
            Landmarks from the previous frame.
        next_landmarks : Iterable
            Landmarks from the next frame.
        mode : str
        '2D' or '3D' — defines whether to use 2 or 3 dimensions for distance.
        landmark_type : str
            Either 'pose' or 'hand'.
        computation_type : str
            'diff' or 'normalized_diff'

        Returns:
        -------
        List of movement vectors between frames.
        """
        check_landmark_type(landmark_type)
        check_difference_type(computation_type)
        check_mode(mode)

        difference_indices = (
            self.pose_values if landmark_type == "pose" else self.hand_values
        )
        return self.__convert_to_numpy(
            self.__compute_differences(
                difference_indices,
                prev_landmarks,
                next_landmarks,
                mode,
                computation_type,
            )
        )

    def compute_annotated(
        self,
        prev_landmarks: Iterable,
        next_landmarks: Iterable,
        landmark_type: str,
        mode: str,
        computation_type: str = "normalized_diff",
    ) -> Dict[str, Tuple[float]]:
        """
        Compute named difference vectors for visualization or debugging.

        Parameters:
        ----------
        prev_landmarks : Iterable
            Previous frame's landmarks.
        next_landmarks : Iterable
            Next frame's landmarks.
        mode : str
        '2D' or '3D' — defines whether to use 2 or 3 dimensions for distance.
        landmark_type : str
            Either 'pose' or 'hand'.
        computation_type : str
            'diff' or 'normalized_diff'

        Returns:
        -------
        Dict[str, Tuple] : Mapping from name to movement vector.
        """
        check_landmark_type(landmark_type)
        check_difference_type(computation_type)
        check_mode(mode)

        difference_indices = (
            self.pose_values if landmark_type == "pose" else self.hand_values
        )
        difference_names = (
            self.pose_names if landmark_type == "pose" else self.hand_names
        )
        diffs = self.__compute_differences(
            difference_indices, prev_landmarks, next_landmarks, mode, computation_type
        )
        return dict(zip(difference_names, diffs))
