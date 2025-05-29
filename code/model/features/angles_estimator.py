import numpy as np
from typing import Union, Dict, List, Tuple, Iterable
from model.utils.utils import (
    check_mode,
    check_landmark_type,
    check_angle_type,
)
from model.features.base_estimator import BaseEstimator
from omegaconf import DictConfig


def angle(
    a, b, c, mode: str = "3D", angle_type: str = "func"
) -> Union[float, Tuple[float]]:
    """
    Computes the angle between vectors AB and BC, with various output formats.

    Parameters:
    ----------
    a, b, c : objects
        Points with attributes x, y, and optionally z (for 3D).
        The angle is calculated at point `b` between vectors AB and CB.

    mode : str, optional (default='3D')
        Coordinate mode: '2D' uses only x and y, '3D' uses x, y, z.

    angle_type : str, optional (default='func')
        Output format of the angle. Supported options:
        - 'rad'                    : Returns the angle in radians [0, π].
        - 'grad'                   : Returns the angle in degrees [0, 180].
        - 'normalized_rad'         : Returns the angle normalized to [0, 1] as angle_rad / π.
        - 'shifted_rad'           : Returns the angle normalized to [-1, 1] as (angle_rad / π) * 2 - 1.
        - 'func'                  : Returns a tuple of (sin(angle_rad), cos(angle_rad)), both ∈ [-1, 1].
        - 'clockwise_rad'         : Returns the clockwise angle in radians [0, 2π].
        - 'clockwise_grad'        : Returns the clockwise angle in degrees [0, 360].
        - 'clockwise_normalized_rad': Returns the clockwise angle normalized to [0, 1] as angle_rad / (2π).
        - 'clockwise_shifted_rad' : Returns the clockwise angle normalized to [-1, 1] as (angle_rad / π) - 1.
        - 'clockwise_func'        : Returns a tuple of (sin(angle_rad), cos(angle_rad)) for clockwise angle.

    Returns:
    -------
    float or tuple of float
        Depending on `angle_type`, returns a float (angle in selected format) or a tuple (sin, cos).

    Notes:
    -----
    - If the magnitude of either vector AB or CB is zero, returns 0 (or (0, 1) for 'func') to avoid division by zero
    - For clockwise variants, the angle is measured clockwise from AB to CB

    Examples:
    --------
    angle(a, b, c, mode='2D', angle_type='normalized_rad')  # → 0.5 for 90 degrees
    angle(a, b, c, angle_type='func')                       # → (0.707, 0.707) for 45 degrees
    angle(a, b, c, angle_type='clockwise_rad')              # → 1.57 for 90 degrees clockwise
    """

    check_mode(mode)
    check_angle_type(angle_type)

    if mode == "3D":
        ab = [a.x - b.x, a.y - b.y, a.z - b.z]
        cb = [c.x - b.x, c.y - b.y, c.z - b.z]
    else:
        ab = [a.x - b.x, a.y - b.y]
        cb = [c.x - b.x, c.y - b.y]

    dot_product = sum(ai * ci for ai, ci in zip(ab, cb))
    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)

    if norm_ab * norm_cb == 0:
        return 0.0 if "func" not in angle_type else (0.0, 1.0)  # avoid division by zero

    cos_theta = dot_product / (norm_ab * norm_cb)
    cos_theta = max(min(cos_theta, 1), -1)  # clamp for stability
    angle_rad = np.arccos(cos_theta)

    # Calculate cross product to determine sign for clockwise angles
    is_clockwise = angle_type.startswith("clockwise_")
    if is_clockwise:
        if mode == "3D":
            cross_product = np.cross(ab, cb)
            sign = np.sign(cross_product[2])
        else:
            cross_product = ab[0] * cb[1] - ab[1] * cb[0]
            sign = np.sign(cross_product)
        if sign < 0:
            angle_rad = 2 * np.pi - angle_rad

    # Handle func types
    if "func" in angle_type:
        return np.sin(angle_rad), np.cos(angle_rad)

    # Handle grad types
    if "grad" in angle_type:
        return np.degrees(angle_rad)

    # Handle normalized types
    if "normalized_rad" in angle_type:
        return angle_rad / (2 * np.pi if is_clockwise else np.pi)

    # Handle shifted types
    if "shifted_rad" in angle_type:
        return (angle_rad / np.pi) - 1 if is_clockwise else (angle_rad / np.pi) * 2 - 1

    # Default to radians
    return angle_rad


class AnglesEstimator(BaseEstimator):
    """
    Estimates angles between triplets of landmarks.
    """

    def __init__(
        self,
        hand_angles: Union[str, Dict, DictConfig],
        pose_angles: Union[str, Dict, DictConfig],
    ):
        super().__init__(hand_angles, pose_angles, config_type="angles")

    def __compute_angles(
        self,
        landmark_triplets: List[Tuple[int]],
        landmarks: Iterable,
        mode: str,
        angle_type: str,
    ) -> List[float]:
        if landmarks is None:
            print("Angles Estimator: None landmarks")
            if angle_type == "func" or angle_type == "clockwise_func":  # sin, cos
                return np.zeros(shape=2 * len(landmark_triplets))
            else:
                return np.zeros(shape=len(landmark_triplets))
        return [
            angle(
                landmarks[start],
                landmarks[middle],
                landmarks[end],
                mode,
                angle_type,
            )
            for start, middle, end in landmark_triplets
        ]

    def __convert_to_numpy(self, features: List[float]) -> np.ndarray:
        return np.array(features).flatten()

    def compute(
        self, landmarks: Iterable, landmark_type: str, mode: str, computation_type: str
    ) -> np.ndarray:
        """
        Compute angles (as features) for either 'pose' or 'hand' landmarks.

        Parameters:
        ----------
        landmarks : Iterable
            List of landmark objects.
        landmark_type : str
            Either 'pose' or 'hand'.
        mode : str
        '2D' or '3D' — defines whether to use 2 or 3 dimensions for distance.
        computation_type : str
            Angle representation to use.
        Returns:
        -------
        List[float] : List of computed angles.
        """
        check_landmark_type(landmark_type)
        check_angle_type(computation_type)
        check_mode(mode)

        angle_triplets = (
            self.pose_values if landmark_type == "pose" else self.hand_values
        )
        return self.__convert_to_numpy(
            self.__compute_angles(angle_triplets, landmarks, mode, computation_type)
        )

    def compute_annotated(
        self, landmarks: Iterable, mode: str, landmark_type: str, computation_type: str
    ) -> Dict[str, float]:
        """
        Compute angles and return a dictionary with named features.

        Parameters:
        ----------
        landmarks : Iterable
            List of landmark objects.
        landmark_type : str
            Either 'pose' or 'hand'.
        mode : str
        '2D' or '3D' — defines whether to use 2 or 3 dimensions for distance.
        computation_type : str
            Angle representation to use.

        Returns:
        -------
        Dict[str, float] : Mapping from feature name to computed angle value. For visualization/debug.
        """
        check_landmark_type(landmark_type)
        check_angle_type(computation_type)
        check_mode(mode)

        angle_triplets = (
            self.pose_values if landmark_type == "pose" else self.hand_values
        )
        angle_names = self.pose_names if landmark_type == "pose" else self.hand_names
        angles = self.__compute_angles(
            angle_triplets, landmarks, mode, computation_type
        )
        return dict(zip(angle_names, angles))
