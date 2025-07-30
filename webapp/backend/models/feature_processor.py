import numpy as np
from typing import List, Dict, Any, Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class FeatureProcessor:
    """
    Class to process video frames into features for model input.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FeatureProcessor with configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing feature processing parameters.
        """
        self.config = config
        self.landmark_types = ['pose']  # Adjust based on config if multiple landmark types (e.g., face, hands)
        self.num_landmarks = 33  # Number of pose landmarks (MediaPipe pose model default)
        self.features_per_landmark = 4  # x, y, z, visibility
        logger.info("Initialized FeatureProcessor")

    def process_frames(
        self,
        frames: np.ndarray,
        frame_indices: List[int],
        metadata: pd.Series,
        mode: str = 'test'
    ) -> np.ndarray:
        """
        Process selected frames into feature tensors.

        Args:
            frames (np.ndarray): Array of frame dictionaries containing landmark data.
            frame_indices (List[int]): Indices of frames to process.
            metadata (pd.Series): Video metadata (e.g., fps, frame_count).
            mode (str): Processing mode ('train' or 'test').

        Returns:
            np.ndarray: Feature tensor for the selected frames.
        """
        try:
            if not frame_indices or len(frames) == 0:
                logger.error("Empty frame indices or frames provided")
                raise ValueError("Invalid frame indices or empty frames")

            # Initialize feature array
            num_frames = len(frame_indices)
            features = np.zeros((num_frames, self.num_landmarks * self.features_per_landmark))

            for i, idx in enumerate(frame_indices):
                if idx >= len(frames):
                    logger.warning(f"Frame index {idx} out of range, skipping")
                    continue

                frame = frames[idx]
                processed_frame = self._process_single_frame(frame)
                features[i] = processed_frame

            logger.info(f"Processed {num_frames} frames into features")
            return features

        except Exception as e:
            logger.error(f"Feature processing failed: {str(e)}")
            raise

    def _process_single_frame(self, frame: Dict[str, Any]) -> np.ndarray:
        """
        Process a single frame's landmarks into a feature vector.

        Args:
            frame (Dict[str, Any]): Dictionary containing landmark data for the frame.

        Returns:
            np.ndarray: Feature vector for the frame.
        """
        feature_vector = np.zeros(self.num_landmarks * self.features_per_landmark)

        if not frame:
            logger.debug("Empty frame, returning zero vector")
            return feature_vector

        # Handle both direct and wrapped landmark structures
        landmarks = frame.get('pose_landmarks', frame)

        # Process pose landmarks (dictionary with indices as keys)
        for idx in range(self.num_landmarks):
            landmark_key = str(idx)  # Landmarks are stored with string keys
            if landmark_key in landmarks:
                landmark = landmarks[landmark_key]
                base_idx = idx * self.features_per_landmark
                feature_vector[base_idx] = landmark.get('x', 0.0)
                feature_vector[base_idx + 1] = landmark.get('y', 0.0)
                feature_vector[base_idx + 2] = landmark.get('z', 0.0)
                feature_vector[base_idx + 3] = landmark.get('visibility', 0.0)

        return feature_vector