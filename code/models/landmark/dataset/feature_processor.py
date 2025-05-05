from typing import Dict, List, Any
import torch
import numpy as np


class FeatureProcessor:
    def __init__(
        self,
        configuration: Dict[str, Any],
        estimators: Dict[str, Dict[str, Any]],
        augmentations: List[Dict[str, Any]] = None,
    ):
        """
        Initialize the feature processor.
        
        Args:
            configuration: Dictionary containing landmark types, features, and ordering
            estimators: Dictionary of feature estimators and their configurations
            augmentations: List of augmentation configurations with their probabilities
        """
        self.configuration = configuration
        self.estimators = estimators
        self.augmentations = augmentations or []

    def process_frames(self, frames: List[Any], selected_indices: List[int]) -> torch.Tensor:
        """
        Process a sequence of frames to generate feature vectors.
        
        Args:
            frames: List of frames containing landmark data
            selected_indices: List of indices indicating which frames to process
            
        Returns:
            torch.Tensor: Processed features for the sequence
        """
        all_features = []
        
        for indx, i in enumerate(selected_indices):
            frame = frames[i]
            # Extract landmarks
            frame = {
                f"{key}_landmarks": frame[f"{key}_landmarks"].landmark
                if frame[f"{key}_landmarks"] is not None
                else None
                for key in self.configuration["landmark_types"]
            }

            # Apply augmentations
            for aug in self.augmentations:
                if np.random.uniform() <= aug["p"]:
                    frame = aug["augmentation"](frame)

            # Generate features
            features = self._compute_features(frame, frames, selected_indices, indx)
            
            # Add validness features
            features["validness"] = [
                int(frame[f"{key}_landmarks"] is not None)
                for key in self.configuration["landmark_types"]
            ]
            
            # Concatenate all features into a single vector
            feature_vector = np.concatenate(list(features.values()), axis=None)
            all_features.append(torch.tensor(feature_vector, dtype=torch.float))

        return torch.stack(all_features)

    def _compute_features(
        self, 
        frame: Dict[str, Any], 
        frames: List[Any],
        selected_indices: List[int],
        current_index: int
    ) -> Dict[str, np.ndarray]:
        """
        Compute all features for a single frame.
        
        Args:
            frame: Current frame's landmark data
            frames: All frames (needed for frame-to-frame features)
            selected_indices: Selected frame indices
            current_index: Index of current frame in selected_indices
            
        Returns:
            Dict[str, np.ndarray]: Computed features
        """
        features = {}
        
        # Get previous frame for frame-to-frame features
        prev_frame = {
            f"{key}_landmarks": None for key in self.configuration["landmark_types"]
        }
        if current_index > 0:
            prev_frame_idx = selected_indices[current_index - 1]
            prev_frame = frames[prev_frame_idx]
            prev_frame = {
                f"{key}_landmarks": prev_frame[f"{key}_landmarks"].landmark
                if prev_frame[f"{key}_landmarks"] is not None
                else None
                for key in self.configuration["landmark_types"]
            }

        # Compute features based on configuration
        for first_key in self.configuration[self.configuration["ordering"][0]]:
            for second_key in self.configuration[self.configuration["ordering"][1]]:
                feature_type, landmark_type = self._check_landmark_config(
                    first_key, second_key
                )
                
                if feature_type == "differences":
                    features[f"{feature_type}/{landmark_type}"] = self.estimators[
                        feature_type
                    ]["estimator"].compute(
                        prev_frame[f"{landmark_type}_landmarks"],
                        frame[f"{landmark_type}_landmarks"],
                        landmark_type=landmark_type.split("_")[-1],
                        mode=self.estimators[feature_type]["mode"],
                        computation_type=self.estimators[feature_type][
                            "computation_type"
                        ],
                    )
                else:
                    features[f"{feature_type}/{landmark_type}"] = self.estimators[
                        feature_type
                    ]["estimator"].compute(
                        frame[f"{landmark_type}_landmarks"],
                        landmark_type=landmark_type.split("_")[-1],
                        mode=self.estimators[feature_type]["mode"],
                        computation_type=self.estimators[feature_type][
                            "computation_type"
                        ],
                    )
        
        return features

    def _check_landmark_config(self, first_key: str, second_key: str) -> tuple[str, str]:
        """Check and return the correct order of feature type and landmark type."""
        if second_key in self.configuration["landmark_types"]:
            return first_key, second_key
        if first_key in self.configuration["landmark_types"]:
            return second_key, first_key

        raise Exception(
            "Error with landmark_features in dataset config, it is not specified correctly"
        ) 