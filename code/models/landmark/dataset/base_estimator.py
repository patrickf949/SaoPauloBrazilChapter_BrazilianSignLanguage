from typing import Union, Dict, List, Iterable, Tuple
from abc import ABC, abstractmethod
from models.landmark.utils import load_config, check_landmark_type, check_mode

import numpy as np

class BaseEstimator(ABC):
    """
    Base class for estimators working with pose and hand landmarks.
    Provides shared logic for config loading, caching.
    """
    def __init__(self, hand_config: Union[str, Dict], pose_config: Union[str, Dict], config_type: str):
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
    def __init__(self, hand_landmarks: Union[str, Dict],
                 pose_landmarks: Union[str, Dict]):
        """
        Parameters:
        ----------
        hand_landmarks : str or dict
            Path to YAML file or dictionary with selected hand landmark indices.
        pose_landmarks : str or dict
            Path to YAML file or dictionary with selected pose landmark indices.
        """
        super().__init__(hand_landmarks, pose_landmarks, config_type="landmarks")
    
    def __unpack_landmark(self, 
                          landmark, 
                          mode: str) -> List[float]:
        return [landmark.x, landmark.y, landmark.z] if mode == "3D" else [landmark.x, landmark.y]

    def __compute(self, landmarks_positions: List[int], 
                  landmarks: Iterable, 
                  mode: str) -> List[List[float]]:
        return [self.__unpack_landmark(landmarks[position], mode) for position in landmarks_positions]

    def __convert_to_numpy(self, features: List) -> np.ndarray:
        return np.array(features).flatten()

    def compute(self, 
                 landmarks: Iterable, landmark_type: str, mode: str) ->np.ndarray:
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

        """
        check_landmark_type(landmark_type)
        check_mode(mode)

        landmark_positions = self.pose_values if landmark_type == "pose" else self.hand_values
        return self.__convert_to_numpy(self.__compute(landmark_positions, landmarks, mode))
    
    def compute_annotated(self, landmarks: Iterable, 
                          landmark_type: str, 
                          mode: str) -> Dict[str, List[float]]:
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

        landmark_positions = self.pose_values if landmark_type == "pose" else self.hand_values
        landmark_names = self.pose_names if landmark_type == "pose" else self.hand_names

        selected_landmarks = self.__compute(landmark_positions, landmarks, mode)

        return dict(zip(landmark_names, selected_landmarks))