from typing import Union, Dict, List, Iterable
from abc import ABC, abstractmethod
from models.landmark.utils.utils import load_config, check_landmark_type, check_mode, minmax_scale_series, minmax_scale_single
from omegaconf import DictConfig
import numpy as np


class BaseEstimator(ABC):
    """
    Base class for estimators working with pose and hand landmarks.
    Provides shared logic for config loading, caching.
    """

    def __init__(
        self,
        hand_config: Union[str, Dict, DictConfig],
        pose_config: Union[str, Dict, DictConfig],
        config_type: str,
    ):
        self.hand_config = load_config(hand_config, f"hand_{config_type}")
        self.pose_config = load_config(pose_config, f"pose_{config_type}")

        self.hand_names = list(self.hand_config.keys())
        self.hand_values = list(self.hand_config.values())

        self.pose_names = list(self.pose_config.keys())
        self.pose_values = list(self.pose_config.values())

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_annotated(self, *args, **kwargs):
        pass


class LandmarkEstimator(BaseEstimator):
    """
    Extracts selected pose or hand landmark coordinates as feature vectors.
    """

    def __init__(
        self,
        hand_landmarks: Union[str, Dict, DictConfig],
        pose_landmarks: Union[str, Dict, DictConfig],
    ):
        """
        Parameters:
        ----------
        hand_landmarks : str or dict
            Path to YAML file or dictionary with selected hand landmark indices.
        pose_landmarks : str or dict
            Path to YAML file or dictionary with selected pose landmark indices.
        """
        super().__init__(hand_landmarks, pose_landmarks, config_type="landmarks")

    def __unpack_landmark(self, landmark, mode: str) -> List[float]:
        if landmark is None:
            return [0.0, 0.0, 0.0] if mode == "3D" else [0.0, 0.0]
        return (
            [float(landmark.x), float(landmark.y), float(landmark.z)]
            if mode == "3D"
            else [float(landmark.x), float(landmark.y)]
        )

    def __compute(
        self, landmarks_positions: List[int], landmarks: Iterable, mode: str, computation_type: str, scaling_info: Dict
    ) -> List[List[float]]:
        if landmarks is None:
            print("Landmark PositionEstimator: None landmarks") 
            if mode == "2D":
                return np.zeros(shape=2 * len(landmarks_positions), dtype=np.float32)
            else:
                return np.zeros(shape=3 * len(landmarks_positions), dtype=np.float32)
        
        # Initialize result list with zeros
        result = []
        for position in landmarks_positions:
            if position < len(landmarks) and landmarks[position] is not None:
                if mode == "2D":
                    x, y = self.__unpack_landmark(landmarks[position], mode)
                else:
                    x, y, z = self.__unpack_landmark(landmarks[position], mode)
                if computation_type == "scaled":
                    x = minmax_scale_single(x, scaling_info["max_for_scaling_x"], scaling_info["min_for_scaling_x"], scaling_info["scale_range"])
                    y = minmax_scale_single(y, scaling_info["max_for_scaling_y"], scaling_info["min_for_scaling_y"], scaling_info["scale_range"])
                if mode == "2D":
                    result.append([x, y])
                else:
                    result.append([x, y, z])
            else:
                print(f"Landmark PositionEstimator: Landmark {position} is missing or invalid")
                # If landmark is missing, add zeros
                result.append([0.0, 0.0, 0.0] if mode == "3D" else [0.0, 0.0])
        return result

    def __convert_to_numpy(self, features: List) -> np.ndarray:
        return np.array(features, dtype=np.float32).flatten()

    def compute(self, landmarks: Iterable, landmark_type: str, mode: str, computation_type: str, scaling_info: Dict) -> np.ndarray:
        """
        Extracts selected landmark coordinates as a flat feature array.

        Parameters:
        ----------
        landmarks : Iterable
            List of landmark objects.
        landmark_type : str
            Either 'pose' or 'hand'.
        mode : str
            '2D' or '3D'
        computation_type : str
            'raw' or 'scaled'
        scaling_info : Dict
            Dictionary containing scaling information.

        """
        check_landmark_type(landmark_type)
        check_mode(mode)

        landmark_positions = (
            self.pose_values if landmark_type == "pose" else self.hand_values
        )
        # print(landmark_type, mode, computation_type, len(landmark_positions))
        return self.__convert_to_numpy(
            self.__compute(landmark_positions, landmarks, mode, computation_type, scaling_info)
        )

    def compute_annotated(
        self, landmarks: Iterable, landmark_type: str, mode: str
    ) -> Dict[str, List[float]]:
        """
        Extracts selected landmark coordinates as a named dictionary.

        Parameters:
        ----------
        landmarks : Iterable
            List of landmark objects.
        landmark_type : str
            Either 'pose' or 'hand'.
        mode : str
            '2D' or '3D'

        """
        check_landmark_type(landmark_type)
        check_mode(mode)

        landmark_positions = (
            self.pose_values if landmark_type == "pose" else self.hand_values
        )
        landmark_names = self.pose_names if landmark_type == "pose" else self.hand_names

        selected_landmarks = self.__compute(landmark_positions, landmarks, mode)

        return dict(zip(landmark_names, selected_landmarks))
