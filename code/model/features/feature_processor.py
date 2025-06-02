from typing import Dict, List, Union, Any
import torch
import numpy as np
from omegaconf import DictConfig
from model.utils.utils import load_config, load_obj, minmax_scale_single, minmax_scale_series
import json
import os

class FeatureProcessor:
    """Processes landmark data into feature vectors.
    
    Note: Data augmentations are only applied during training.
    Validation and test data should be processed without augmentations
    to ensure consistent evaluation.
    """
    def __init__(
        self,
        dataset_split: str,
        dataset_config: Union[str, Dict, DictConfig],
        features_config: Union[str, Dict, DictConfig],
        augmentation_config: Union[str, Dict, DictConfig],
        landmarks_dir: str,
    ):
        """
        Initialize the feature processor.
        
        Args:
            dataset_split: Split of the dataset (train/val/test)
            dataset_config: Dictionary containing landmark types, features, and ordering
            features_config: Dictionary of feature estimators and their configurations
            augmentation_config: List of augmentation configurations with their probabilities
            landmarks_dir: Directory containing landmark data and metadata
        """
        # Extract metadata configuration from features config
        metadata_config = features_config.get("metadata", {})
        self.scale_range = metadata_config.get("scale_range", [-1, 1])
        self.include_positions = "positions" in features_config.keys()
        self.positions_computation_type = features_config.get("positions", {}).get("computation_type", None)
        self.positions_scaling_info = features_config.get("positions", {}).get("scaling_info", None)
        self.positions_mode = features_config.get("positions", {}).get("mode", None)
        self.metadata_row_features = metadata_config.get("metadata_row_features", [])
        self.metadata_json_features = metadata_config.get("metadata_json_features", [])

        configuration = {
            "landmark_types": dataset_config["landmark_types"],
            "features": [name for name in features_config.keys() if name != "metadata" and name != "positions"],  # Exclude metadata and positions from estimator features
            "ordering": dataset_config["ordering"],
        }

        estimators = {
            name: {
                "estimator": load_obj(estimator_params["class_name"])(
                    estimator_params["hand"],
                    estimator_params["pose"],
                ),
                "mode": estimator_params["mode"],
                "computation_type": estimator_params["computation_type"],
            }
            for name, estimator_params in features_config.items()
            if name != "metadata" and name != "positions"  # Skip metadata and positions as they're not estimators
        }
        
        # augmentation_config[dataset_split] should be None for val and test splits
        # But the option exists if you want to add some augmentations to val or test splits
        augmentations = (
            [
                {
                    "augmentation": load_obj(augmentation["class_name"])(
                        **augmentation["params"]
                    ),
                    "p": augmentation["p"],
                }
                for _, augmentation in augmentation_config[dataset_split].items()
            ]
            if augmentation_config[dataset_split] is not None
            else []
        )

        self.configuration = configuration
        self.estimators = estimators
        self.augmentations = augmentations or []
        self.landmarks_dir = landmarks_dir

    def process_frames(self, frames: List[Any], selected_indices: List[int], metadata_row: Dict[str, Any], dataset_split: str) -> torch.Tensor:
        """
        Process a sequence of frames to generate feature vectors.
        
        Args:
            frames: List of frames containing landmark data
            selected_indices: List of indices indicating which frames to process
            metadata_row: Metadata row containing video information
        Returns:
            torch.Tensor: Processed features for the sequence
        """
        # Select data augmentations once for the entire sequence
        selected_augmentations = []
        if dataset_split == "train":
            for aug in self.augmentations:
                if np.random.uniform() <= aug["p"]:
                    selected_augmentations.append(aug["augmentation"])
        
        # Apply augmentations to all frames
        augmented_frames = []
        for i, frame in enumerate(frames):
            # Extract landmarks
            processed_frame = {
                f"{key}_landmarks": frame[f"{key}_landmarks"].landmark
                if frame[f"{key}_landmarks"] is not None
                else None
                for key in self.configuration["landmark_types"]
            }
            
            # Apply augmentations
            for augmentation in selected_augmentations:
                processed_frame = augmentation(processed_frame)
            
            # # Apply augmentations
            # if i in selected_indices:
            #     for augmentation in selected_augmentations:
            #         processed_frame = augmentation(processed_frame)
            augmented_frames.append(processed_frame)

        # Initialise list for appending feature vectors
        all_features = []
        position_features = {f"{key}_landmarks": [] for key in self.configuration["landmark_types"]}
        
        for i, frame_idx in enumerate(selected_indices):
            # Use augmented frame
            frame = augmented_frames[frame_idx]
            
            # Generate features
            features = self._compute_estimator_features(frame, augmented_frames, selected_indices, i)
            
            # Concatenate all features into a single vector
            feature_vector = np.concatenate(list(features.values()), axis=None)
            all_features.append(torch.tensor(feature_vector, dtype=torch.float))

            # Get positions of augmented frames from selected indices
            for key in self.configuration["landmark_types"]:
                xyz = [[lm.x, lm.y, lm.z] for lm in frame[f"{key}_landmarks"]]
                xyz = np.array(xyz)
                position_features[f"{key}_landmarks"].append(xyz)
        if self.include_positions:
            # Now we have a dict of lists of the landmarks for each frame, for each landmark type
            # We need to stack them into a single array, and scale them if needed
            for key in position_features.keys():
                # Stack the landmarks for each frame into a single array
                position_features[key] = np.array(position_features[key])
                # Scale the landmarks if needed
                if self.positions_computation_type == "scaled":
                    position_features[key][:,:,0] = minmax_scale_series(
                        position_features[key][:,:,0],
                        self.positions_scaling_info[key]['input_max_x'],
                        self.positions_scaling_info[key]['input_min_x'],
                        self.scale_range
                    )
                    position_features[key][:,:,1] = minmax_scale_series(
                        position_features[key][:,:,1],
                        self.positions_scaling_info[key]['input_max_y'],
                        self.positions_scaling_info[key]['input_min_y'],
                        self.scale_range)
                
                # Convert position features to tensors
                x_positions = torch.tensor(position_features[key][:,:,0], dtype=torch.float)  # shape: [20, 21]
                y_positions = torch.tensor(position_features[key][:,:,1], dtype=torch.float)  # shape: [20, 21]
                
                # Reshape position features to match frame features
                x_positions = x_positions.reshape(x_positions.shape[0], -1)  # shape: [20, 21]
                y_positions = y_positions.reshape(y_positions.shape[0], -1)  # shape: [20, 21]
                
                # Add position features to each frame's feature vector
                for i in range(len(all_features)):
                    all_features[i] = torch.cat([all_features[i], x_positions[i], y_positions[i]])
                
                if self.positions_mode == '3D':
                    z_positions = torch.tensor(position_features[key][:,:,2], dtype=torch.float)  # shape: [20, 21]
                    z_positions = z_positions.reshape(z_positions.shape[0], -1)  # shape: [20, 21]
                    # Add z positions to each frame's feature vector
                    for i in range(len(all_features)):
                        all_features[i] = torch.cat([all_features[i], z_positions[i]])
        # Stack all frame features
        frame_features = torch.stack(all_features)  # shape: (sequence_length, n_landmark_features)
        
        # Get metadata features
        metadata_row_features = self._get_metadata_row_features(metadata_row, selected_indices)  # shape: (sequence_length, n_metadata_row_features)
        
        # Load metadata JSON file
        metadata_json_path = os.path.join(self.landmarks_dir, "individual_metadata", metadata_row["filename"].replace(".npy", ".json"))
        if not os.path.exists(metadata_json_path):
            raise FileNotFoundError(f"Metadata JSON file not found: {metadata_json_path}")
            
        try:
            with open(metadata_json_path, "r") as f:
                metadata_json = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file {metadata_json_path}: {str(e)}")
            
        metadata_json_features = self._get_metadata_json_features(metadata_json, selected_indices)  # shape: (sequence_length, n_metadata_json_features)
        
        # Concatenate all feature types along the feature dimension
        all_features = torch.cat([
            frame_features,
            metadata_row_features,
            metadata_json_features
        ], dim=1)  # shape: (sequence_length, n_landmark_features + n_metadata_row_features + n_metadata_json_features)
        return all_features

    def _compute_estimator_features(
        self, 
        frame: Dict[str, Any], 
        frames: List[Any], 
        selected_indices: List[int],
        current_index: int
    ) -> Dict[str, np.ndarray]:
        """
        Compute all estimator features for a single frame.
        
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
                f"{key}_landmarks": prev_frame[f"{key}_landmarks"]
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
    
    def _get_metadata_row_features(self, metadata_row: Dict[str, Any], selected_indices: List[int]) -> torch.Tensor:
        """Get metadata features from metadata row.
        
        Args:
            metadata_row: Dictionary containing metadata information
            selected_indices: List of indices to extract features for
            
        Returns:
            torch.Tensor: Tensor of shape (sequence_length, n_features) containing metadata features
        """        
        # Validate all required columns exist (except relative_frame_position)
        missing_columns = [feature["column"] for feature in self.metadata_row_features if feature["column"] not in metadata_row and feature["column"] != "relative_frame_position"]
        if missing_columns:
            raise KeyError(f"Missing required metadata columns: {missing_columns}")
        
        # Process each feature with its scaling parameters
        sequence_length = len(selected_indices)
        n_features = len(self.metadata_row_features)
        
        # Initialize tensor with correct shape (sequence_length, n_features)
        metadata_feature_tensor = torch.zeros((sequence_length, n_features), dtype=torch.float)
        
        # Fill in each feature column
        for feature_idx, feature in enumerate(self.metadata_row_features):
            column = feature["column"]
            if column == "relative_frame_position":
                value = metadata_row['processed_frame_count']
                # Fill the entire column with relative positions
                metadata_feature_tensor[:, feature_idx] = torch.tensor([idx/value for idx in selected_indices])
            else:
                value = metadata_row[column]
                value = minmax_scale_single(
                    value=value,
                    input_max=feature["max_for_scaling"],
                    output_range=self.scale_range
                )
                # Fill the entire column with the same scaled value
                metadata_feature_tensor[:, feature_idx] = value
        
        return metadata_feature_tensor

    def _get_nested_value(self, data: Dict[str, Any], path: List[str]) -> Any:
        """Safely get a nested value from a dictionary using a path of keys.
        
        Args:
            data: Dictionary to extract value from
            path: List of keys representing the path to the value
            
        Returns:
            The value at the specified path
            
        Raises:
            KeyError: If any key in the path doesn't exist, with a message indicating the full path
        """
        try:
            current = data
            for key in path:
                if key not in current:
                    raise KeyError(f"Missing key '{key}' in path {path}")
                current = current[key]
            return current
        except KeyError as e:
            raise KeyError(f"Failed to extract value at path {path}: {str(e)}")

    def _get_metadata_json_features(self, metadata_json: Dict[str, Any], selected_indices: List[int]) -> torch.Tensor:
        """Get metadata features from metadata json file.
        
        Args:
            metadata_json: Dictionary containing metadata information
            selected_indices: List of indices to extract features for
            
        Returns:
            torch.Tensor: Tensor of shape (sequence_length, n_features) containing metadata features
        """
        sequence_length = len(selected_indices)
        n_features = len(self.metadata_json_features)
        
        # Initialize tensor with correct shape (sequence_length, n_features)
        metadata_json_tensor = torch.zeros((sequence_length, n_features), dtype=torch.float)
        
        # Fill in each feature column
        for feature_idx, feature in enumerate(self.metadata_json_features):
            full_feature_series = self._get_nested_value(metadata_json, feature["path"])
            
            # Validate indices
            if max(selected_indices) >= len(full_feature_series):
                raise IndexError(f"Selected indices exceed feature series length ({len(full_feature_series)})")
            
            # Get the features for the selected indices
            selected_series = [full_feature_series[i] for i in selected_indices]
            
            # Scale the entire series at once
            scaled_series = minmax_scale_series(
                values=np.array(selected_series),
                input_max=feature["max_for_scaling"],
                output_range=self.scale_range
            )
            
            # Fill the entire column with scaled values
            metadata_json_tensor[:, feature_idx] = torch.from_numpy(scaled_series)

        return metadata_json_tensor