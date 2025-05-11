import numpy as np
from typing import Union, Dict, List, Tuple, Iterable
from models.landmark.utils.utils import (
    check_mode,
    check_landmark_type,
    check_distance_type,
)
from models.landmark.features.base_estimator import BaseEstimator
from omegaconf import DictConfig


def distance(p1, p2, mode: str = "3D", distance_type: str = "shifted_dist") -> float:
    """
    Computes distance between two points with support for normalized and shifted variants.

    Parameters:
    ----------
    p1, p2 : objects
        Points with attributes x, y, and optionally z (for 3D).
    distance_type : str
        One of:
        - 'dist'            : Raw Euclidean distance
        - 'normalized_dist' : Distance scaled to [0, 1] based on max possible distance (√12)
        - 'shifted_dist'    : Normalized then shifted to [-1, 1] range

    Returns:
    -------
    float : Distance in the selected representation
    """
    check_mode(mode)
    check_distance_type(distance_type)

    if mode == "3D":
        dist = np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
        max_dist = np.sqrt(12)  # max distance in 3D normalized space [-1,1]
    else:
        dist = np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
        max_dist = np.sqrt(8)  # max distance in 2D normalized space [-1,1]

    if distance_type == "dist":
        return dist
    elif distance_type == "normalized_dist":
        return dist / max_dist
    elif distance_type == "shifted_dist":
        return (dist / max_dist) * 2 - 1  # scale to [-1, 1]


class DistancesEstimator(BaseEstimator):
    """
    Estimates distances between pairs of landmarks, with support for normalized and shifted distance types.
    """

    def __init__(
        self,
        hand_distances: Union[str, Dict, DictConfig],
        pose_distances: Union[str, Dict, DictConfig],
    ):
        """
        Parameters:
        ----------
        hand_distances : str or dict
            Path to YAML file or dictionary with hand landmark index pairs.
        pose_distances : str or dict
            Path to YAML file or dictionary with pose landmark index pairs.

        """
        super().__init__(hand_distances, pose_distances, config_type="distances")

    def __compute_distances(
        self,
        landmark_pairs: List[Tuple[int]],
        landmarks: Iterable,
        mode: str,
        distance_type: str,
    ) -> List[float]:
        if landmarks is None:
            return np.zeros(shape=len(landmark_pairs))
        return [
            distance(landmarks[start], landmarks[end], mode, distance_type)
            for start, end in landmark_pairs
        ]

    def compute(
        self,
        landmarks: Iterable,
        landmark_type: str,
        mode: str,
        computation_type: str = "shifted_dist",
    ) -> np.ndarray:
        """
        Compute distances between defined landmark pairs.

        Parameters:
        ----------
        landmarks : Iterable
            List of landmark objects.
        mode : str
            '2D' or '3D' — determines how distances are computed.
        landmark_type : str
            Either 'pose' or 'hand'.
        computation_type : str
            Type of distance: 'dist', 'normalized_dist', or 'shifted_dist'.

        Returns:
        -------
        List[float] : Computed distance values.
        """
        check_landmark_type(landmark_type)
        check_distance_type(computation_type)
        check_mode(mode)

        distance_pairs = (
            self.pose_values if landmark_type == "pose" else self.hand_values
        )
        return np.array(
            self.__compute_distances(distance_pairs, landmarks, mode, computation_type)
        )

    def compute_annotated(
        self,
        landmarks: Iterable,
        landmark_type: str,
        mode: str,
        computation_type: str = "shifted_dist",
    ) -> Dict[str, float]:
        """
        Compute named distances for use in analysis or visualization.

        Parameters:
        ----------
        landmarks : Iterable
            List of landmark objects.
        mode : str
            '2D' or '3D' — determines how distances are computed.
        landmark_type : str
            Either 'pose' or 'hand'.
        computation_type : str
            Type of distance to compute.

        Returns:
        -------
        Dict[str, float] : Mapping of distance name to computed value.
        """
        check_landmark_type(landmark_type)
        check_distance_type(computation_type)
        check_mode(mode)

        distance_pairs = (
            self.pose_values if landmark_type == "pose" else self.hand_values
        )
        distance_names = self.pose_names if landmark_type == "pose" else self.hand_names
        distances = self.__compute_distances(
            distance_pairs, landmarks, mode, computation_type
        )
        return dict(zip(distance_names, distances))
