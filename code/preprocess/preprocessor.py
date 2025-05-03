import os
import json
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import shutil
import copy
from video_analyzer import validate_landmarks_format, hide_lower_body_landmarks, analyze_none_landmarks

class Preprocessor:
    """
    Preprocessor class for applying consistent transformations to videos and landmarks.
    
    This class handles:
    - Trimming videos/landmarks by start and end frames
    - Horizontal and vertical alignment
    - Scaling content
    - Temporal padding to ensure consistent duration
    - Crop and resize
    - Saving results with appropriate metadata
    """
    
    def __init__(
        self,
        metadata_row: Dict,
        params_dict: Dict,
        path_to_root: str,
        motion_version: str = "versionB",
        pose_version: str = "versionB",
        preprocess_version: str = "v3",
        verbose: bool = True,
        save_intermediate: bool = False
    ):
        """
        Initialize the Preprocessor with metadata and parameters.
        
        Args:
            metadata_row: Dictionary containing video metadata (filename, fps, width, height, etc.)
            params_dict: Dictionary of preprocessing parameters
            path_to_root: Path to the root directory of the project
            preprocess_version: Version identifier for this preprocessing
            verbose: Whether to print details about processing steps
            save_intermediate: Whether to save intermediate results after each preprocessing step
        """
        # Store inputs
        self.metadata = metadata_row
        self.params = params_dict
        self.path_to_root = path_to_root
        self.preprocess_version = preprocess_version
        self.verbose = verbose
        self.save_intermediate = save_intermediate
        
        # Extract key metadata
        self.filename = metadata_row["filename"]
        self.fps = metadata_row["fps"]
        self.width = metadata_row["width"]
        self.height = metadata_row["height"]
        self.frame_count = metadata_row["frame_count"]
        self.duration_sec = metadata_row["duration_sec"]
        
        # Set up directory paths
        self.interim_dir = os.path.join(path_to_root, "data", "interim")
        self.preprocessed_dir = os.path.join(path_to_root, "data", "preprocessed")
        
        # Raw video path
        self.raw_videos_dir = os.path.join(self.interim_dir, "RawCleanVideos")
        
        # Interim paths
        self.interim_debug_dir = os.path.join(self.interim_dir, "Debug")
        self.interim_debug_videos_dir = os.path.join(self.interim_debug_dir, "videos")
        self.interim_debug_landmarks_dir = os.path.join(self.interim_debug_dir, "landmarks")
        self.interim_motion_dir = os.path.join(self.interim_dir, "RawMotionMeasurements", motion_version)
        self.interim_landmarks_dir = os.path.join(self.interim_dir, "RawPoseLandmarks", pose_version)
        self.interim_videos_dir = os.path.join(self.interim_dir, "Videos")
        
        # Preprocessed paths
        self.preprocessed_videos_dir = os.path.join(self.preprocessed_dir, "videos", preprocess_version)
        self.preprocessed_landmarks_dir = os.path.join(self.preprocessed_dir, "landmarks", preprocess_version)
        
        # Individual metadata paths
        self.video_metadata_dir = os.path.join(self.preprocessed_dir, "videos", preprocess_version, "individual_metadata")
        self.landmarks_metadata_dir = os.path.join(self.preprocessed_dir, "landmarks", preprocess_version, "individual_metadata")
        
        # Create output directories if they don't exist
        for directory in [
            self.interim_debug_videos_dir,
            self.interim_debug_landmarks_dir,
            self.interim_motion_dir,
            self.interim_landmarks_dir,
            self.interim_videos_dir,
            self.preprocessed_videos_dir,
            self.preprocessed_landmarks_dir,
            self.video_metadata_dir,
            self.landmarks_metadata_dir
        ]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                if self.verbose:
                    print(f"Created directory: {directory}")
                
        # Validate required parameters
        self._validate_params()
        
        if verbose:
            print(f"Initialized Preprocessor for {self.filename}")
            print(f"Naming this preprocessing version: {preprocess_version}")
    
    def _validate_params(self) -> None:
        """
        Validate that all required parameters are present and have valid values.
        
        Raises:
            ValueError: If any required parameter is missing or invalid
        """
        required_params = [
            "start_frame", 
            "end_frame", 
            "face_width_aim",                           # used for horizontal scaling
            "shoulders_width_aim",                      # used for horizontal scaling
            "face_midpoint_to_shoulders_height_aim",    # used for vertical scaling
            "shoulders_y_aim",                           # used for vertical alignment
            "use_statistic",                            # used for reference point calculation
            "use_stationary_frames",                     # used for reference point calculation
            "skip_stationary_frames"                     # used for landmark extremes calculation
        ]
        
        missing_params = [param for param in required_params if param not in self.params]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
        
        # Validate parameter ranges
        if self.params["start_frame"] < 0 or self.params["start_frame"] >= self.frame_count:
            raise ValueError(f"start_frame must be between 0 and {self.frame_count-1}, got {self.params['start_frame']}")
            
        if self.params["end_frame"] <= self.params["start_frame"] or self.params["end_frame"] >= self.frame_count:
            raise ValueError(f"end_frame must be between {self.params['start_frame']+1} and {self.frame_count-1}, got {self.params['end_frame']}")
            
        # Validate scaling and alignment parameters
        if not (0 < self.params["face_width_aim"] <= 1):
            raise ValueError(f"face_width_aim must be between 0 and 1, got {self.params['face_width_aim']}")
            
        if not (0 < self.params["shoulders_width_aim"] <= 1):
            raise ValueError(f"shoulders_width_aim must be between 0 and 1, got {self.params['shoulders_width_aim']}")
            
        if not (0 < self.params["face_midpoint_to_shoulders_height_aim"] <= 1):
            raise ValueError(f"face_midpoint_to_shoulders_height_aim must be between 0 and 1, got {self.params['face_midpoint_to_shoulders_height_aim']}")
            
        if not (0 <= self.params["shoulders_y_aim"] <= 1):
            raise ValueError(f"shoulders_y_aim must be between 0 and 1, got {self.params['shoulders_y_aim']}")
            
        # Set default values for optional parameters if not provided
        if "use_statistic" not in self.params:
            self.params["use_statistic"] = "median"
        if "use_stationary_frames" not in self.params:
            self.params["use_stationary_frames"] = False
            
        # Validate reference point calculation parameters if provided
        valid_statistics = ["mean", "median", "max", "min"]
        if self.params["use_statistic"] not in valid_statistics:
            raise ValueError(f"use_statistic must be one of {valid_statistics}, got {self.params['use_statistic']}")
            
        if not isinstance(self.params["use_stationary_frames"], bool):
            raise ValueError(f"use_stationary_frames must be a boolean, got {self.params['use_stationary_frames']}")
    
    # ==========================================
    # Landmark Processing Methods
    # ==========================================
    
    def _interpolate_landmarks(self, landmarks: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Interpolate landmarks when there are None frames between valid frames.
        Handles both single-frame gaps and sequences of None frames.
        Applies to all landmark groups: pose, face, and hands.
        
        For single None frames:
        - Uses simple midpoint interpolation between valid frames before and after
        
        For sequences of None frames:
        - Uses linear interpolation to create equidistant values between valid frames
        
        For None frames at start/end:
        - Repeats the nearest valid frame
        
        Args:
            landmarks: Array of landmarks to process
            
        Returns:
            Tuple containing:
            - Array of landmarks with interpolated values where possible
            - Dictionary containing interpolation metadata
        """
        interpolation_info = {
            'pose_landmarks': {
                'total_interpolated_frames': 0,
                'interpolated_frame_indices': [],  # List of all interpolated frame indices
                'interpolated_sequences_start_end_length': []  # List of (start_idx, end_idx, sequence_length) tuples
            },
            'face_landmarks': {
                'total_interpolated_frames': 0,
                'interpolated_frame_indices': [],
                'interpolated_sequences_start_end_length': []
            },
            'left_hand_landmarks': {
                'total_interpolated_frames': 0,
                'interpolated_frame_indices': [],
                'interpolated_sequences_start_end_length': []
            },
            'right_hand_landmarks': {
                'total_interpolated_frames': 0,
                'interpolated_frame_indices': [],
                'interpolated_sequences_start_end_length': []
            }
        }
        
        # Get None analysis once before processing any landmarks
        none_analysis = analyze_none_landmarks(landmarks)
        
        if self.verbose:
            print("None analysis before interpolation:")
            for landmark_type in none_analysis:
                print(f"{landmark_type} None frames: {none_analysis[landmark_type]['none_details']}")
        
        # Process each landmark type
        for landmark_type in ['pose_landmarks', 'face_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']:
            none_frames = sorted(none_analysis[landmark_type]['none_details'])
            
            if self.verbose:
                print(f"\nProcessing {landmark_type}")
                print(f"Found None frames: {none_frames}")
            
            if not none_frames:
                continue
                
            # Find sequences of consecutive None frames
            none_sequences = []
            seq_start = none_frames[0]
            prev_frame = none_frames[0]
            
            for frame in none_frames[1:] + [none_frames[-1] + 2]:  # Add sentinel value
                if frame != prev_frame + 1:
                    # End of sequence
                    none_sequences.append((seq_start, prev_frame))
                    seq_start = frame
                prev_frame = frame
            
            if self.verbose:
                print(f"Found None sequences: {none_sequences}")
            
            # Process each sequence
            for seq_start, seq_end in none_sequences:
                seq_length = seq_end - seq_start + 1
                
                # Find valid frames before and after sequence
                before_frame_idx = seq_start - 1
                after_frame_idx = seq_end + 1
                
                # Handle sequences at start of video
                if before_frame_idx < 0:
                    if after_frame_idx >= len(landmarks) or landmarks[after_frame_idx] is None:
                        continue  # Can't interpolate if no valid frames available
                    
                    # Repeat the first valid frame
                    valid_frame = self._deep_copy_landmarks_frame(landmarks[after_frame_idx])
                    if valid_frame is None or landmark_type not in valid_frame or valid_frame[landmark_type] is None:
                        continue
                        
                    for frame_idx in range(seq_start, seq_end + 1):
                        if frame_idx >= len(landmarks):
                            break
                        new_frame = self._deep_copy_landmarks_frame(landmarks[frame_idx])
                        if new_frame is None:
                            new_frame = {}
                        new_frame[landmark_type] = copy.deepcopy(valid_frame[landmark_type])
                        landmarks[frame_idx] = new_frame
                        
                        # Track interpolation
                        interpolation_info[landmark_type]['total_interpolated_frames'] += 1
                        interpolation_info[landmark_type]['interpolated_frame_indices'].append(frame_idx)
                    
                    interpolation_info[landmark_type]['interpolated_sequences_start_end_length'].append((seq_start, seq_end, seq_length))
                    continue
                
                # Handle sequences at end of video
                if after_frame_idx >= len(landmarks):
                    if before_frame_idx < 0 or landmarks[before_frame_idx] is None:
                        continue  # Can't interpolate if no valid frames available
                    
                    # Repeat the last valid frame
                    valid_frame = self._deep_copy_landmarks_frame(landmarks[before_frame_idx])
                    if valid_frame is None or landmark_type not in valid_frame or valid_frame[landmark_type] is None:
                        continue
                        
                    for frame_idx in range(seq_start, seq_end + 1):
                        if frame_idx >= len(landmarks):
                            break
                        new_frame = self._deep_copy_landmarks_frame(landmarks[frame_idx])
                        if new_frame is None:
                            new_frame = {}
                        new_frame[landmark_type] = copy.deepcopy(valid_frame[landmark_type])
                        landmarks[frame_idx] = new_frame
                        
                        # Track interpolation
                        interpolation_info[landmark_type]['total_interpolated_frames'] += 1
                        interpolation_info[landmark_type]['interpolated_frame_indices'].append(frame_idx)
                    
                    interpolation_info[landmark_type]['interpolated_sequences_start_end_length'].append((seq_start, seq_end, seq_length))
                    continue
                
                # Get valid frames before and after sequence
                before_frame = landmarks[before_frame_idx]
                after_frame = landmarks[after_frame_idx]
                
                # Skip if either frame is invalid
                if (before_frame is None or after_frame is None or
                    landmark_type not in before_frame or landmark_type not in after_frame or
                    before_frame[landmark_type] is None or after_frame[landmark_type] is None):
                    continue
                
                # Get landmarks from before and after frames
                before_landmarks = before_frame[landmark_type].landmark
                after_landmarks = after_frame[landmark_type].landmark
                
                # Debug output for first landmark point before interpolation
                if self.verbose:
                    print(f"\nDebug - First landmark point before interpolation for sequence {seq_start}-{seq_end}:")
                    print(f"Before frame: x={before_landmarks[0].x:.4f}, y={before_landmarks[0].y:.4f}, z={before_landmarks[0].z:.4f}")
                    print(f"After frame:  x={after_landmarks[0].x:.4f}, y={after_landmarks[0].y:.4f}, z={after_landmarks[0].z:.4f}")
                
                # Create interpolated frames
                for frame_idx in range(seq_start, seq_end + 1):
                    # Create new frame by deep copying the before frame
                    new_frame = self._deep_copy_landmarks_frame(before_frame)
                    if landmark_type not in new_frame or new_frame[landmark_type] is None:
                        new_frame[landmark_type] = copy.deepcopy(before_frame[landmark_type])
                    
                    # Calculate interpolation factor (0 to 1)
                    t = (frame_idx - before_frame_idx) / (after_frame_idx - before_frame_idx)
                    
                    # Interpolate each landmark
                    for j in range(len(before_landmarks)):
                        # Linear interpolation for x, y, z coordinates
                        x = before_landmarks[j].x + t * (after_landmarks[j].x - before_landmarks[j].x)
                        y = before_landmarks[j].y + t * (after_landmarks[j].y - before_landmarks[j].y)
                        z = before_landmarks[j].z + t * (after_landmarks[j].z - before_landmarks[j].z)
                        
                        new_frame[landmark_type].landmark[j].x = x
                        new_frame[landmark_type].landmark[j].y = y
                        new_frame[landmark_type].landmark[j].z = z
                        
                        # Debug output for first landmark point after interpolation
                        if j == 0 and self.verbose:
                            print(f"Interpolated frame {frame_idx}: x={x:.4f}, y={y:.4f}, z={z:.4f}")
                    
                    # Store the interpolated frame
                    landmarks[frame_idx] = new_frame
                    
                    # Track this successful interpolation
                    interpolation_info[landmark_type]['total_interpolated_frames'] += 1
                    interpolation_info[landmark_type]['interpolated_frame_indices'].append(frame_idx)
                
                interpolation_info[landmark_type]['interpolated_sequences_start_end_length'].append((seq_start, seq_end, seq_length))
        
        if self.verbose:
            # Analyze after interpolation
            after_analysis = analyze_none_landmarks(landmarks)
            print("\nNone analysis after interpolation:")
            for landmark_type in after_analysis:
                print(f"{landmark_type} None frames: {after_analysis[landmark_type]['none_details']}")
                print(f"Reduced {landmark_type} None frames by: {len(none_analysis[landmark_type]['none_details']) - len(after_analysis[landmark_type]['none_details'])}")
            
            # Print interpolation summary
            for landmark_type in interpolation_info:
                print(f"\n{landmark_type} interpolation summary:")
                print(f"Total frames interpolated: {interpolation_info[landmark_type]['total_interpolated_frames']}")
                print(f"Sequences interpolated: {interpolation_info[landmark_type]['interpolated_sequences_start_end_length']}")
            
        return landmarks, interpolation_info

    def preprocess_landmarks(self) -> str:
        """
        Apply preprocessing steps to landmarks and save result.
                
        Returns:
            Path to preprocessed landmarks
        """
        use_stationary_frames = self.params["use_stationary_frames"]
        use_statistic = self.params["use_statistic"]
        skip_stationary_frames = self.params["skip_stationary_frames"]
        
        # Get landmarks path
        landmarks_path = os.path.join(
            self.interim_landmarks_dir, 
            f"{self.filename.split('.')[0]}.npy"
        )
        
        if not os.path.exists(landmarks_path):
            raise FileNotFoundError(f"Landmarks file not found at {landmarks_path}")
        
        if self.verbose:
            print(f"Preprocessing landmarks: {landmarks_path}")
            
        # If save_intermediate is True, save original to debug
        if self.save_intermediate:
            debug_path = os.path.join(self.interim_debug_landmarks_dir, f"{self.filename.split('.')[0]}_original.npy")
            shutil.copy2(landmarks_path, debug_path)
            if self.verbose:
                print(f">> Saved original landmarks to {debug_path}")
            
        # Load landmarks
        landmarks = np.load(landmarks_path, allow_pickle=True)
        # landmarks = hide_lower_body_landmarks(landmarks)
        # validation_result = validate_landmarks_format(landmarks)
        # if not validation_result[0]:
        #     raise ValueError(f"Invalid landmarks format in raw data: {validation_result[1]}")

        if self.verbose:
            print(f"Loaded landmarks with shape: {landmarks.shape}")

        # 1. Get Reference Points on raw landmarks
        self.raw_reference_points = self._get_reference_points_for_series(landmarks, use_statistic, use_stationary_frames)
        self.raw_landmark_extremes = self._get_landmark_extremes_for_series(landmarks, skip_stationary_frames)
        self.raw_none_analysis = analyze_none_landmarks(landmarks)

        # 2. Get scale factors and offsets
        self.raw_scale_factors_and_offsets = self._get_scale_factors_and_offsets(self.raw_reference_points)
        
        # 3. Scale landmarks
        scaled_landmarks = self._scale_landmarks(landmarks, self.raw_scale_factors_and_offsets)
        # del landmarks
        if self.save_intermediate:
            self._save_intermediate_landmarks(scaled_landmarks, "scaled")
        
        # 4. Get Reference Points on scaled landmarks
        self.scaled_reference_points = self._get_reference_points_for_series(scaled_landmarks, use_statistic, use_stationary_frames)
        self.scaled_landmark_extremes = self._get_landmark_extremes_for_series(scaled_landmarks, skip_stationary_frames)

        # 5. Get scale factors and offsets on scaled landmarks
        self.scaled_scale_factors_and_offsets = self._get_scale_factors_and_offsets(self.scaled_reference_points)

        # 6. Align landmarks
        aligned_landmarks = self._align_landmarks(scaled_landmarks, self.scaled_scale_factors_and_offsets)
        del scaled_landmarks
        if self.save_intermediate:
            self._save_intermediate_landmarks(aligned_landmarks, "aligned")
        self.aligned_reference_points = self._get_reference_points_for_series(aligned_landmarks, use_statistic, use_stationary_frames)
        self.aligned_landmark_extremes = self._get_landmark_extremes_for_series(aligned_landmarks, skip_stationary_frames)
        self.aligned_scale_factors_and_offsets = self._get_scale_factors_and_offsets(self.aligned_reference_points)

        # 7. Trim landmarks
        trimmed_landmarks = self._trim_landmarks(aligned_landmarks)
        if self.save_intermediate:
            self._save_intermediate_landmarks(trimmed_landmarks, "trimmed")
        self.trimmed_reference_points = self._get_reference_points_for_series(trimmed_landmarks, use_statistic, use_stationary_frames=False)
        self.trimmed_scale_factors_and_offsets = self._get_scale_factors_and_offsets(self.trimmed_reference_points)
        self.trimmed_landmark_extremes = self._get_landmark_extremes_for_series(trimmed_landmarks, skip_stationary_frames)
        self.trimmed_none_analysis = analyze_none_landmarks(trimmed_landmarks)

        # 8. Interpolate landmarks
        interpolated_landmarks, self.interpolation_info = self._interpolate_landmarks(trimmed_landmarks)
        if self.save_intermediate:
            self._save_intermediate_landmarks(interpolated_landmarks, "interpolated")
        self.interpolated_none_analysis = analyze_none_landmarks(interpolated_landmarks)
        
        # 9. Save preprocessed landmarks & metadata
        output_path = self._save_preprocessed_landmarks(interpolated_landmarks)
        
        # Clear processed landmarks from memory
        del interpolated_landmarks
        del trimmed_landmarks
        del aligned_landmarks
        
        return output_path
    
    def _trim_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Trim landmarks to only include the range between start_frame and end_frame.
        
        Args:
            landmarks: Array of landmarks to trim
            
        Returns:
            Array of trimmed landmarks
        """
        start_frame = self.params["start_frame"]
        end_frame = self.params["end_frame"]
        
        if start_frame >= len(landmarks) or end_frame >= len(landmarks):
            raise ValueError(f"Invalid frame range: {start_frame} to {end_frame}, landmarks have {len(landmarks)} frames")
            
        # Make a slice of the landmarks array for the specified frame range
        trimmed = landmarks[start_frame:end_frame+1]
        
        if self.verbose:
            print(f"Trimmed landmarks from {len(landmarks)} frames to {len(trimmed)} frames")
            print(f"Frame range: {start_frame} to {end_frame}")
            
        return trimmed
    
    def _deep_copy_landmarks_frame(self, frame_landmarks):
        """
        Create a deep copy of a landmarks frame while preserving MediaPipe object structure.
        
        Args:
            frame_landmarks: Dictionary containing landmark data for a single frame
            
        Returns:
            Deep copy of the landmarks frame with preserved structure
        """
        if frame_landmarks is None:
            return None
            
        # Create a copy of the frame's landmarks
        new_frame = {}
        for key in frame_landmarks.keys():
            if key not in ['pose_landmarks', 'face_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']:
                # Copy non-landmark keys directly
                new_frame[key] = frame_landmarks[key]
                continue
            
            # Keep None entries as is
            if frame_landmarks[key] is None:
                new_frame[key] = None
                continue
            
            # Get the landmarks for this type and create a deep copy
            landmarks_obj = frame_landmarks[key]
            new_frame[key] = copy.deepcopy(landmarks_obj)
        
        return new_frame
    
    def _get_landmark_extremes_for_frame(self, frame_landmarks: Dict, skip_lower_body: bool = True) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Track extreme coordinates (min/max x, min y) for each landmark type.
        
        Args:
            frame_landmarks: Dictionary containing landmark data for a single frame
            skip_lower_body: If True, ignore lower body landmarks (indices 23-32) in pose_landmarks
            
        Returns:
            Dictionary containing extreme coordinates for each landmark type and overall
        """
        extremes = {
            'overall': {'left_x': None, 'right_x': None, 'top_y': None, 'bottom_y': None},
            'pose': {'left_x': None, 'right_x': None, 'top_y': None, 'bottom_y': None},
            'face': {'left_x': None, 'right_x': None, 'top_y': None, 'bottom_y': None},
            'left_hand': {'left_x': None, 'right_x': None, 'top_y': None, 'bottom_y': None},
            'right_hand': {'left_x': None, 'right_x': None, 'top_y': None, 'bottom_y': None}
        }
        
        # Helper function to update extremes for a landmark group
        def update_category_extremes(category: str, landmarks) -> None:
            if landmarks is not None:
                # For pose landmarks, optionally skip lower body
                if category == 'pose' and skip_lower_body:
                    x_coords = [lm.x for i, lm in enumerate(landmarks.landmark) if i < 23]
                    y_coords = [lm.y for i, lm in enumerate(landmarks.landmark) if i < 23]
                else:
                    x_coords = [lm.x for lm in landmarks.landmark]
                    y_coords = [lm.y for lm in landmarks.landmark]
                    
                if x_coords and y_coords:  # Check if lists are not empty
                    extremes[category]['left_x'] = min(x_coords)
                    extremes[category]['right_x'] = max(x_coords)
                    extremes[category]['top_y'] = min(y_coords)
                    extremes[category]['bottom_y'] = max(y_coords)
                
        # Get extremes for each landmark type
        landmark_types = {
            'pose': 'pose_landmarks',
            'face': 'face_landmarks',
            'left_hand': 'left_hand_landmarks',
            'right_hand': 'right_hand_landmarks'
        }
        
        for category, key in landmark_types.items():
            update_category_extremes(category, frame_landmarks.get(key))
        
        # Calculate overall extremes
        all_x, all_y = [], []
        for category, key in landmark_types.items():
            if frame_landmarks.get(key) is not None:
                # For pose landmarks, optionally skip lower body
                if category == 'pose' and skip_lower_body:
                    all_x.extend(lm.x for i, lm in enumerate(frame_landmarks[key].landmark) if i < 23)
                    all_y.extend(lm.y for i, lm in enumerate(frame_landmarks[key].landmark) if i < 23)
                else:
                    all_x.extend(lm.x for lm in frame_landmarks[key].landmark)
                    all_y.extend(lm.y for lm in frame_landmarks[key].landmark)
        
        if all_x and all_y:
            extremes['overall']['left_x'] = min(all_x)
            extremes['overall']['right_x'] = max(all_x)
            extremes['overall']['top_y'] = min(all_y)
            extremes['overall']['bottom_y'] = max(all_y)
        return extremes

    def _get_landmark_extremes_for_series(self, landmarks: np.ndarray, skip_stationary_frames: bool = False, skip_lower_body: bool = True) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Track extreme coordinates (min/max x, min y) for each landmark type across all frames.
        
        Args:
            landmarks: Array of landmarks to analyze
            skip_stationary_frames: If True, skip frames between start_frame and end_frame
            
        Returns:
            Dictionary containing extreme coordinates for each landmark type and overall.
            If no valid landmarks were found for a particular measurement, its value will be None.
        """
        extremes_list = {
            'overall': {
                'left_x': [], 'right_x': [], 'top_y': [], 'bottom_y': []
            },
            'pose': {
                'left_x': [], 'right_x': [], 'top_y': [], 'bottom_y': []
            },
            'face': {
                'left_x': [], 'right_x': [], 'top_y': [], 'bottom_y': []
            },
            'left_hand': {
                'left_x': [], 'right_x': [], 'top_y': [], 'bottom_y': []
            },
            'right_hand': {
                'left_x': [], 'right_x': [], 'top_y': [], 'bottom_y': []
            }
        }
        for frame_idx in range(len(landmarks)):
            if skip_stationary_frames and self.params["start_frame"] <= frame_idx <= self.params["end_frame"]:
                continue
            frame_landmarks = landmarks[frame_idx]
            frame_extremes = self._get_landmark_extremes_for_frame(frame_landmarks, skip_lower_body)
            for category in extremes_list.keys():
                for key in extremes_list[category].keys():
                    extremes_list[category][key].append(frame_extremes[category][key])
        
        extremes = {
            'overall': {'left_x': None, 'right_x': None, 'top_y': None, 'bottom_y': None},
            'pose': {'left_x': None, 'right_x': None, 'top_y': None, 'bottom_y': None},
            'face': {'left_x': None, 'right_x': None, 'top_y': None, 'bottom_y': None},
            'left_hand': {'left_x': None, 'right_x': None, 'top_y': None, 'bottom_y': None},
            'right_hand': {'left_x': None, 'right_x': None, 'top_y': None, 'bottom_y': None},
        }
        
        for category in extremes_list.keys():
            for key in extremes_list[category].keys():
                # Filter out None values
                valid_values = [x for x in extremes_list[category][key] if x is not None]
                if valid_values:  # Only compute min/max if we have valid values
                    if key in ['left_x', 'top_y']:
                        extremes[category][key] = min(valid_values)
                    elif key in ['right_x', 'bottom_y']:
                        extremes[category][key] = max(valid_values)
                # If no valid values, extremes[category][key] remains None
        
        return extremes
    
    def _get_reference_points_for_frame(self, frame_landmarks: Dict) -> Dict[str, Optional[float]]:
        """
        Calculate reference points for a single frame of landmarks.
        
        Args:
            frame_landmarks: Dictionary containing landmark data for a single frame
            
        Returns:
            Dictionary containing reference points:
            {
                'face_midpoint_x': Optional[float],
                'face_midpoint_y': Optional[float],
                'face_width': Optional[float],
                'face_height': Optional[float],
                'shoulders_midpoint_x': Optional[float],
                'shoulders_midpoint_y': Optional[float],
                'shoulders_width': Optional[float],
                'face_midpoint_to_shoulders_height': Optional[float],
            }
        """
        reference_points = {}

        # face
        if frame_landmarks['face_landmarks']:
            face_landmarks = frame_landmarks['face_landmarks'].landmark
            
            # Get relevant landmarks
            left_face = face_landmarks[234]  # Left face edge
            right_face = face_landmarks[454]  # Right face edge
            top_head = face_landmarks[10]  # Top of head
            chin = face_landmarks[152]  # Chin

            # Face midpoint
            reference_points['face_midpoint_x'] = (left_face.x + right_face.x) / 2
            reference_points['face_midpoint_y'] = (left_face.y + right_face.y) / 2

            # Face width (distance between left and right face edges)
            reference_points['face_width'] = np.sqrt(
                (left_face.x - right_face.x)**2 + 
                (left_face.y - right_face.y)**2
            )
            
            # Face height (distance between top of head and chin)
            reference_points['face_height'] = np.sqrt(
                (top_head.x - chin.x)**2 + 
                (top_head.y - chin.y)**2
            )
        else:
            reference_points['face_width'] = None
            reference_points['face_height'] = None
            reference_points['face_midpoint_x'] = None
            reference_points['face_midpoint_y'] = None

        # shoulders
        if frame_landmarks['pose_landmarks']:
            pose_landmarks = frame_landmarks['pose_landmarks'].landmark
            
            # Get relevant landmarks
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]

            # Shoulder midpoint
            reference_points['shoulders_midpoint_x'] = (left_shoulder.x + right_shoulder.x) / 2
            reference_points['shoulders_midpoint_y'] = (left_shoulder.y + right_shoulder.y) / 2

            # Shoulder width
            reference_points['shoulders_width'] = np.sqrt(
                (left_shoulder.x - right_shoulder.x)**2 + 
                (left_shoulder.y - right_shoulder.y)**2
            )
        else:
            reference_points['shoulders_midpoint_x'] = None
            reference_points['shoulders_midpoint_y'] = None
            reference_points['shoulders_width'] = None

        # face midpoint to shoulders height
        if all(reference_points[key] is not None for key in ['face_midpoint_x', 'face_midpoint_y', 'shoulders_midpoint_x', 'shoulders_midpoint_y']):
            # actual distance
            reference_points['face_midpoint_to_shoulders_height'] = np.sqrt(
                (reference_points['face_midpoint_x'] - reference_points['shoulders_midpoint_x'])**2 + 
                (reference_points['face_midpoint_y'] - reference_points['shoulders_midpoint_y'])**2
            )
        else:
            reference_points['face_midpoint_to_shoulders_height'] = None

        return reference_points
    
    def _get_reference_points_for_series(self, landmarks: np.ndarray, use_statistic: str = "median", use_stationary_frames: bool = False) -> Tuple[Dict[str, float], int]:
        """
        Calculate the reference points for a series of frames.
        
        Args:
            landmarks: Array of landmarks to measure
            use_statistic: Which statistic to use for reference points ("mean", "median", "max", "min")
            use_stationary_frames: If True, only use frames from before start_frame and after end_frame
            
        Returns:
            Tuple containing:
            1. Dictionary of reference points with the specified statistic applied
            2. Number of frames that had None landmarks
        """
        all_reference_points = {}
        none_frames = 0
        total_frames_used = 0

        for frame_idx in range(len(landmarks)):
            # Skip frames in the motion range if use_stationary_frames is True
            if use_stationary_frames:
                if self.params["start_frame"] <= frame_idx <= self.params["end_frame"]:
                    continue
            
            frame_landmarks = landmarks[frame_idx]
            reference_points = self._get_reference_points_for_frame(frame_landmarks)
            
            # Check if any measurements were None
            if any(v is None for v in reference_points.values()):
                none_frames += 1
            else:
                total_frames_used += 1
                
            for key in reference_points.keys():
                if key not in all_reference_points:
                    all_reference_points[key] = []
                if reference_points[key] is not None:  # Only append non-None values
                    all_reference_points[key].append(reference_points[key])

        # Calculate statistics for each reference point
        series_reference_points = {}
        series_reference_points['use_statistic'] = use_statistic
        series_reference_points['use_stationary_frames'] = use_stationary_frames
        series_reference_points['total_frames_used'] = total_frames_used
        series_reference_points['frames_with_none_landmarks'] = none_frames
        for key, values in all_reference_points.items():
            if not values:  # If no valid measurements were collected
                series_reference_points[key] = None
                continue
                
            values = np.array(values)
            if use_statistic == "mean":
                series_reference_points[key] = np.mean(values)
            elif use_statistic == "median":
                series_reference_points[key] = np.median(values)
            elif use_statistic == "max":
                series_reference_points[key] = np.max(values)
            elif use_statistic == "min":
                series_reference_points[key] = np.min(values)
            else:
                raise ValueError(f"Invalid statistic: {use_statistic}")

        if self.verbose and use_stationary_frames:
            print(f"Used {total_frames_used} stationary frames for reference points calculation. {none_frames} frames had None landmarks.")


        return series_reference_points
    
    def _get_scale_factors_and_offsets(self, reference_points: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate the scale factors and offsets for landmarks based on current measurements and target values.
        
        Args:
            reference_points: Dictionary containing reference points
            
        Returns:
            Dictionary containing scale factors:
            {
                'x_scale': float,  # Horizontal scale factor
                'y_scale': float   # Vertical scale factor
            }
        """
        # Get target values from params
        x_aim = 0.5
        shoulders_y_aim = self.params["shoulders_y_aim"]
        face_width_aim = self.params["face_width_aim"]
        shoulders_width_aim = self.params["shoulders_width_aim"]
        face_midpoint_to_shoulders_height_aim = self.params["face_midpoint_to_shoulders_height_aim"]
               
        # Validate measurements
        if reference_points['face_midpoint_x'] is None or reference_points['shoulders_midpoint_x'] is None:
            raise ValueError("Could not calculate horizontal alignment from landmarks")
        if reference_points['shoulders_midpoint_y'] is None:
            raise ValueError("Could not calculate vertical alignment from landmarks")
        if reference_points['face_width'] is None:
            raise ValueError("Could not calculate face width from landmarks")
        if reference_points['face_midpoint_to_shoulders_height'] is None:
            raise ValueError("Could not calculate face-to-shoulders height from landmarks")
        
        # Calculate horizontal offset based on face & shoulders
        reference_x = (reference_points['shoulders_midpoint_x'] + reference_points['face_midpoint_x']) / 2
        horizontal_offset = x_aim - reference_x
        # Calculate vertical offset based on shoulders
        vertical_offset = shoulders_y_aim - reference_points['shoulders_midpoint_y']
        # Calculate scale factors
        face_x_scale = face_width_aim / reference_points['face_width']
        shoulders_x_scale = shoulders_width_aim / reference_points['shoulders_width']
        x_scale = max(face_x_scale, shoulders_x_scale)
        y_scale = face_midpoint_to_shoulders_height_aim / reference_points['face_midpoint_to_shoulders_height']
        
        scale_factors_and_offsets = {
            'horizontal_offset': round(horizontal_offset, 3),
            'vertical_offset': round(vertical_offset, 3),
            'x_scale': x_scale,
            'y_scale': y_scale,
            'face_x_scale': face_x_scale,
            'shoulders_x_scale': shoulders_x_scale,
        }
        return scale_factors_and_offsets
    
    def _scale_landmarks(self, landmarks: np.ndarray, scale_factors: Dict[str, float]) -> np.ndarray:
        """
        Scale landmarks based on current measurements and target values.
        
        Args:
            landmarks: Array of landmarks to scale
            use_statistic: Which statistic to use for reference points
            
        Returns:
            Scaled landmarks
        """
        
        # Apply scaling
        scaled_landmarks = []
        
        for frame_idx in range(len(landmarks)):
            frame_landmarks = landmarks[frame_idx]
            if frame_landmarks is None:
                scaled_landmarks.append(None)
                continue
            
            # Make a deep copy of the frame landmarks
            new_frame = self._deep_copy_landmarks_frame(frame_landmarks)
            
            # Apply scaling
            for key in ['pose_landmarks', 'face_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']:
                if key in new_frame and new_frame[key] is not None:
                    for i, lm in enumerate(new_frame[key].landmark):
                        new_frame[key].landmark[i].x = lm.x * scale_factors['x_scale']
                        new_frame[key].landmark[i].y = lm.y * scale_factors['y_scale']
            
            scaled_landmarks.append(new_frame)
        
        # Convert to numpy array
        scaled_landmarks = np.array(scaled_landmarks, dtype=object)
        
        if self.verbose:
            print(f"Scaled landmarks with factors: x={scale_factors['x_scale']}, y={scale_factors['y_scale']}")

        return scaled_landmarks
    

    def _align_landmarks(self, landmarks: np.ndarray, offsets: Dict[str, float]) -> Tuple[np.ndarray, float, float]:
        """
        Align landmarks horizontally and vertically based on calculated offsets.
        
        Args:
            landmarks: Array of landmarks to align
            use_statistic: Which statistic to use for reference points
            
        Returns:
            Tuple of (aligned landmarks, horizontal offset, vertical offset)
        """
        # Get alignment offsets
        horizontal_offset = offsets['horizontal_offset']
        vertical_offset = offsets['vertical_offset']
        
        # Apply shifts
        aligned_landmarks = []
        
        # Process each frame's landmarks
        for frame_idx in range(len(landmarks)):
            frame_landmarks = landmarks[frame_idx]
            
            # Make a deep copy of the frame landmarks
            new_frame = self._deep_copy_landmarks_frame(frame_landmarks)
            if new_frame is None:
                aligned_landmarks.append(None)
                continue
            
            # Update the coordinates with offsets
            for key in ['pose_landmarks', 'face_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']:
                if key in new_frame and new_frame[key] is not None:
                    for i, lm in enumerate(new_frame[key].landmark):
                        # Add offsets directly
                        new_frame[key].landmark[i].x = lm.x + horizontal_offset
                        new_frame[key].landmark[i].y = lm.y + vertical_offset
            
            aligned_landmarks.append(new_frame)
        
        # Convert to numpy array
        aligned_landmarks = np.array(aligned_landmarks, dtype=object)
        
        if self.verbose:
            print(f"Aligned landmarks with offsets: horizontal={horizontal_offset}, vertical={vertical_offset}")
            
        return aligned_landmarks

    def _save_preprocessed_landmarks(self, landmarks: np.ndarray) -> str:
        """
        Save the fully preprocessed landmarks to the preprocessed directory.
        
        Args:
            landmarks: Array of landmarks to save
            
        Returns:
            Path to saved landmarks
        """
        output_filename = f"{self.filename.split('.')[0]}.npy"
        output_path = os.path.join(self.preprocessed_landmarks_dir, output_filename)
        
        # Save landmarks
        np.save(output_path, landmarks)
        
        # Save metadata
        self._save_landmarks_metadata(output_path)
        
        if self.verbose:
            print(f">> Saved preprocessed landmarks to {output_path}")
            
        # Update the landmarks metadata CSV
        self._update_landmarks_metadata_csv()
        
        # If save_intermediate is True, also save to debug directory
        if self.save_intermediate:
            debug_path = os.path.join(self.interim_debug_landmarks_dir, f"{self.filename.split('.')[0]}_final.npy")
            np.save(debug_path, landmarks)
            if self.verbose:
                print(f">> Saved preprocessed landmarks to {debug_path}")
            
        return output_path
    
    def _save_landmarks_metadata(self, landmarks_path: str) -> None:
        """
        Save metadata for preprocessed landmarks.
        
        Args:
            landmarks_path: Path to the landmarks file
        """
        def _convert_for_json(obj):
            """Recursively convert data types to JSON serializable types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return _convert_for_json(obj.tolist())
            elif isinstance(obj, dict):
                return {key: _convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list) or isinstance(obj, tuple):
                return [_convert_for_json(item) for item in obj]
            elif hasattr(obj, 'to_dict'):
                return _convert_for_json(obj.to_dict())
            elif hasattr(obj, 'item'):  # For numpy scalars
                return obj.item()
            else:
                return obj
                
        # Create metadata
        original_metadata = self.metadata
        preprocessing_params = self.params
        
        # Collect all reference points and scale factors information
        detailed_start_end_info = {
            "start_time": preprocessing_params['start_frame'] / original_metadata['fps'],
            "end_time": preprocessing_params['end_frame'] / original_metadata['fps'],
            "start_percent": preprocessing_params['start_frame'] / original_metadata['frame_count'],
            "end_percent": preprocessing_params['end_frame'] / original_metadata['frame_count'],
            "new_frame_count": preprocessing_params['end_frame'] - preprocessing_params['start_frame'],
            "new_duration_sec": (preprocessing_params['end_frame'] - preprocessing_params['start_frame']) / original_metadata['fps'],
            "new_duration_percent": (preprocessing_params['end_frame'] - preprocessing_params['start_frame']) / original_metadata['frame_count']
        }
        
        interpolation_binary_arrays = {}
        interpolation_sequence_length_arrays = {}
        interpolation_binary_arrays_no_trailing_values = {}
        interpolation_sequence_length_arrays_no_trailing_values = {}

        for landmark_type in ['left_hand_landmarks', 'right_hand_landmarks']:
            trimmed_length = preprocessing_params['end_frame'] - preprocessing_params['start_frame'] + 1

            interpolated_frames = self.interpolation_info[landmark_type]['interpolated_frame_indices']
            interpolation_binary_arrays[landmark_type] = np.zeros(trimmed_length, dtype=int)
            for frame_idx in interpolated_frames:
                interpolation_binary_arrays[landmark_type][frame_idx] = 1

            start_end_length = self.interpolation_info[landmark_type]['interpolated_sequences_start_end_length']
            interpolation_sequence_length_arrays[landmark_type] = np.zeros(trimmed_length, dtype=int)
            for start, end, length in start_end_length:
                interpolation_sequence_length_arrays[landmark_type][start : end+1] = length
                
            # Initialize no_trailing_values arrays with copies of the original arrays
            interpolation_binary_arrays_no_trailing_values[landmark_type] = interpolation_binary_arrays[landmark_type].copy()
            interpolation_sequence_length_arrays_no_trailing_values[landmark_type] = interpolation_sequence_length_arrays[landmark_type].copy()

        # Now process the no_trailing_values arrays
        for landmark_type in ['left_hand_landmarks', 'right_hand_landmarks']:
            # Reset sequences at start
            for i in range(trimmed_length):
                if interpolation_binary_arrays_no_trailing_values[landmark_type][i] == 0:
                    break
                interpolation_binary_arrays_no_trailing_values[landmark_type][i] = 0
                interpolation_sequence_length_arrays_no_trailing_values[landmark_type][i] = 0
            
            # Reset sequences at end
            for i in range(trimmed_length - 1, -1, -1):
                if interpolation_binary_arrays_no_trailing_values[landmark_type][i] == 0:
                    break
                interpolation_binary_arrays_no_trailing_values[landmark_type][i] = 0
                interpolation_sequence_length_arrays_no_trailing_values[landmark_type][i] = 0

        landmark_arrays = {}
        for landmark_type in ['left_hand_landmarks', 'right_hand_landmarks']:
            landmark_arrays[landmark_type] = {
                'interpolation_binary_array': interpolation_binary_arrays[landmark_type],
                'interpolation_sequence_length_array': interpolation_sequence_length_arrays[landmark_type],
                'interpolation_binary_array_no_trailing_values': interpolation_binary_arrays_no_trailing_values[landmark_type],
                'interpolation_sequence_length_array_no_trailing_values': interpolation_sequence_length_arrays_no_trailing_values[landmark_type]
            }

        metadata = {
            "metadata": {
                **original_metadata,
                "preprocessing_version": self.preprocess_version,
                **preprocessing_params,
                **detailed_start_end_info,
                "use_stationary_frames": preprocessing_params['use_stationary_frames'],
                "skip_stationary_frames": preprocessing_params['skip_stationary_frames']
            },
            # Add processing stages summary
            "raw": {
                "reference_points": self.raw_reference_points,
                "scale_factors_and_offsets": self.raw_scale_factors_and_offsets,
                "landmark_extremes": {"overall": self.raw_landmark_extremes['overall']},
                "none_analysis": self.raw_none_analysis
            },
            "scaled": {
                "reference_points": self.scaled_reference_points,
                "scale_factors_and_offsets": self.scaled_scale_factors_and_offsets,
                "landmark_extremes": {"overall": self.scaled_landmark_extremes['overall']},
            },
            "aligned": {
                "reference_points": self.aligned_reference_points,
                "scale_factors_and_offsets": self.aligned_scale_factors_and_offsets,
                "landmark_extremes": {"overall": self.aligned_landmark_extremes['overall']},
            },
            "trimmed": {
                "reference_points": self.trimmed_reference_points,
                "scale_factors_and_offsets": self.trimmed_scale_factors_and_offsets,
                "landmark_extremes": self.trimmed_landmark_extremes,
                "none_analysis": self.trimmed_none_analysis
            },
            "interpolated": {
                "none_analysis": {"overall": self.interpolated_none_analysis['overall']},
                "interpolation_info": self.interpolation_info,
            },
            "landmark_none_mask_arrays": landmark_arrays
        }
        
        # Convert to JSON serializable types
        metadata = _convert_for_json(metadata)
        
        # Save to individual metadata file
        metadata_filename = f"{self.filename.split('.')[0]}.json"
        metadata_path = os.path.join(self.landmarks_metadata_dir, metadata_filename)
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        if self.verbose:
            print(f">> Saved landmarks metadata to {metadata_path}")
            print("   Included information for all processing stages: raw, scaled, aligned, and final")
    
    def _update_landmarks_metadata_csv(self) -> None:
        """
        Update the landmarks metadata CSV file.
        Handles schema changes by ensuring new columns are added to existing data.
        """
        csv_path = os.path.join(self.preprocessed_dir, f"landmarks_metadata_{self.preprocess_version}.csv")
        
        # Calculate processed values
        processed_frame_count = self.params["end_frame"] - self.params["start_frame"] + 1
        processed_duration_sec = processed_frame_count / self.fps
        
        # Create a new row for this video's landmarks
        row = {
            "filename": self.filename,
            "label": self.metadata.get("label", ""),
            "data_source": self.metadata.get("data_source", ""),
            "original_fps": self.fps,
            "original_frame_count": self.frame_count,
            "original_duration_sec": self.duration_sec,
            "start_frame": self.params["start_frame"],
            "end_frame": self.params["end_frame"],
            "processed_frame_count": processed_frame_count,
            "processed_duration_sec": processed_duration_sec,
            "preprocess_version": self.preprocess_version,
            # Add key landmark statistics
            "face_width": self.raw_reference_points.get('face_width'),
            "shoulders_width": self.raw_reference_points.get('shoulders_width'),
            "face_midpoint_to_shoulders_height": self.raw_reference_points.get('face_midpoint_to_shoulders_height'),
            # Add interpolation statistics
            "left_hand_interpolated_none_frames": self.interpolation_info['left_hand_landmarks']['total_interpolated_frames'],
            "right_hand_interpolated_none_frames": self.interpolation_info['right_hand_landmarks']['total_interpolated_frames'],
        }
        
        # Convert row to DataFrame
        new_df = pd.DataFrame([row])
        
        # Track if we're creating a new file
        is_new_file = not os.path.exists(csv_path)
        
        # If file exists, load it and check for schema changes
        old_cols = set()
        if not is_new_file:
            df = pd.read_csv(csv_path)
            old_cols = set(df.columns)
            
            # Check if this file is already in the CSV
            if self.filename in df["filename"].values:
                # Get the index of the existing row
                idx = df.index[df["filename"] == self.filename].tolist()[0]
                
                # Update existing columns and add new ones
                for col in new_df.columns:
                    df.loc[idx, col] = new_df.iloc[0][col]
            else:
                # Append the new row, pandas will automatically align columns
                df = pd.concat([df, new_df], ignore_index=True)
        else:
            # If file doesn't exist, use the new DataFrame
            df = new_df
        
        # Check for column changes before saving
        if not is_new_file:
            new_cols = set(new_df.columns)
            added_cols = new_cols - old_cols
            if added_cols and self.verbose:
                print(f">> Adding new columns to landmarks metadata: {added_cols}")
        
        # Save the CSV
        df.to_csv(csv_path, index=False)
        
        if self.verbose:
            if is_new_file:
                print(f">> Created new landmarks metadata CSV at {csv_path}")
            else:
                print(f">> Updated landmarks metadata CSV at {csv_path}")

    def _debug_landmarks_format(self, landmarks1, landmarks2, frame_idx=0, label1="Original", label2="Modified"):
        """
        Debug function to compare landmark formats and check for data loss between two landmark arrays.
        
        Args:
            landmarks1: First landmarks array
            landmarks2: Second landmarks array
            frame_idx: Frame index to compare (default: 0)
            label1: Label for first landmarks (default: "Original")
            label2: Label for second landmarks (default: "Modified")
        """
        if frame_idx >= len(landmarks1) or frame_idx >= len(landmarks2):
            print(f"Invalid frame index: {frame_idx}, arrays have lengths {len(landmarks1)} and {len(landmarks2)}")
            return
        
        frame1 = landmarks1[frame_idx]
        frame2 = landmarks2[frame_idx]
        
        print(f"Comparing {label1} vs {label2} landmarks for frame {frame_idx}")
        print(f"{label1} frame type: {type(frame1)}")
        print(f"{label2} frame type: {type(frame2)}")
        
        # Check keys in dictionary
        if isinstance(frame1, dict) and isinstance(frame2, dict):
            print(f"\nKeys in {label1}: {sorted(frame1.keys())}")
            print(f"Keys in {label2}: {sorted(frame2.keys())}")
            
            # Compare landmark types
            for landmark_type in ['pose_landmarks', 'face_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']:
                if landmark_type in frame1 and landmark_type in frame2:
                    lm1 = frame1[landmark_type]
                    lm2 = frame2[landmark_type]
                    
                    print(f"\n{landmark_type}:")
                    print(f"  {label1} type: {type(lm1)}")
                    print(f"  {label2} type: {type(lm2)}")
                    
                    if lm1 is None or lm2 is None:
                        print(f"  One of the {landmark_type} is None")
                        continue
                    
                    # Check attributes
                    lm1_attrs = dir(lm1)
                    lm2_attrs = dir(lm2)
                    
                    print(f"  {label1} attributes count: {len(lm1_attrs)}")
                    print(f"  {label2} attributes count: {len(lm2_attrs)}")
                    
                    # Check for missing attributes
                    missing_attrs = [attr for attr in lm1_attrs if attr not in lm2_attrs]
                    if missing_attrs:
                        print(f"  Attributes in {label1} but not in {label2}: {missing_attrs}")
                    
                    # Compare landmark count
                    if hasattr(lm1, 'landmark') and hasattr(lm2, 'landmark'):
                        print(f"  {label1} landmark count: {len(lm1.landmark)}")
                        print(f"  {label2} landmark count: {len(lm2.landmark)}")
                        
                        # Check first landmark
                        if len(lm1.landmark) > 0 and len(lm2.landmark) > 0:
                            l1 = lm1.landmark[0]
                            l2 = lm2.landmark[0]
                            
                            print(f"  First landmark in {label1}: (x={l1.x:.4f}, y={l1.y:.4f}, z={l1.z:.4f})")
                            print(f"  First landmark in {label2}: (x={l2.x:.4f}, y={l2.y:.4f}, z={l2.z:.4f})")
                            
                            # Check attributes of individual landmarks
                            l1_attrs = dir(l1)
                            l2_attrs = dir(l2)
                            missing_landmark_attrs = [attr for attr in l1_attrs if attr not in l2_attrs]
                            if missing_landmark_attrs:
                                print(f"  Landmark attributes in {label1} but not in {label2}: {missing_landmark_attrs}")

    def _save_intermediate_landmarks(self, landmarks: np.ndarray, step_name: str) -> str:
        """
        Save intermediate landmarks to the debug directory.
        
        Args:
            landmarks: Array of landmarks to save
            step_name: Name of the preprocessing step
            
        Returns:
            Path to saved landmarks
        """
        output_filename = f"{self.filename.split('.')[0]}_{step_name}.npy"
        output_path = os.path.join(self.interim_debug_landmarks_dir, output_filename)
        
        # Save landmarks
        np.save(output_path, landmarks)
        
        if self.verbose:
            print(f">> Saved intermediate {step_name} landmarks to {output_path}\n")
            
        return output_path

    # ==========================================
    # Video Processing Methods
    # ==========================================
    
    def preprocess_video(self) -> str:
        """
        Apply preprocessing steps to video and save result.
        
        This method requires preprocess_landmarks to be run first to calculate scale factors and offsets.
        
        Returns:
            Path to preprocessed video
            
        Raises:
            RuntimeError: If preprocess_landmarks hasn't been run first
        """
        # Check if preprocess_landmarks has been run
        if not hasattr(self, 'raw_scale_factors_and_offsets'):
            raise RuntimeError("preprocess_landmarks must be run before preprocess_video to calculate scale factors and offsets")
            
        video_path = os.path.join(self.raw_videos_dir, self.filename)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")
        
        if self.verbose:
            print(f"Preprocessing video: {video_path}")
            
        # If save_intermediate is True, save original to debug
        if self.save_intermediate:
            debug_path = os.path.join(self.interim_debug_videos_dir, f"{self.filename.split('.')[0]}_original.mp4")
            shutil.copy2(video_path, debug_path)
            if self.verbose:
                print(f">> Saved original video to {debug_path}")
            
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if self.verbose:
            print(f"Loaded {len(frames)} frames from video")
            
        # Step 1: Trim frames
        trimmed_frames = self._trim_frames(frames)
        if self.save_intermediate:
            self._save_interim_video(trimmed_frames, "trimmed")
        del frames  # Clear original frames from memory
        frames = None
            
        # Step 2: Scale content using raw scale factors
        scaled_frames = self._scale_frames(trimmed_frames)
        if self.save_intermediate:
            self._save_interim_video(scaled_frames, "scaled")
        del trimmed_frames  # Clear trimmed frames from memory
        trimmed_frames = None
            
        # Step 3: Align frames using scaled offsets
        aligned_frames = self._align_frames(scaled_frames)
        if self.save_intermediate:
            self._save_interim_video(aligned_frames, "aligned")
        del scaled_frames  # Clear scaled frames from memory
        scaled_frames = None
            
        # Step 4: Crop/resize if needed
        processed_frames = self._crop_resize_frames(aligned_frames)
        if self.save_intermediate:
            self._save_interim_video(processed_frames, "cropped")
        del aligned_frames  # Clear aligned frames from memory
        aligned_frames = None
        
        # Step 5: Save final preprocessed video
        output_path = self._save_preprocessed_video(processed_frames)
        
        # Clear processed frames from memory
        del processed_frames
        processed_frames = None
        
        return output_path
    
    def _trim_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Trim frames to only include the range between start_frame and end_frame.
        
        Args:
            frames: List of frames to trim
            
        Returns:
            List of trimmed frames
        """
        start_frame = self.params["start_frame"]
        end_frame = self.params["end_frame"]
        
        if start_frame >= len(frames) or end_frame >= len(frames):
            raise ValueError(f"Invalid frame range: {start_frame} to {end_frame}, video has {len(frames)} frames")
            
        trimmed = frames[start_frame:end_frame+1]
        
        if self.verbose:
            print(f"Trimmed video from {len(frames)} frames to {len(trimmed)} frames")
            print(f"Frame range: {start_frame} to {end_frame}")
            
        return trimmed
    
    def _align_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Align frames according to x_offset and y_offset from landmark processing.
        
        Args:
            frames: List of frames to align
            
        Returns:
            List of aligned frames
        """
        x_offset = self.raw_scale_factors_and_offsets['horizontal_offset']
        y_offset = self.raw_scale_factors_and_offsets['vertical_offset']
        
        height, width = frames[0].shape[:2]
        
        aligned_frames = []
        for frame in frames:
            # Create transformation matrix for translation
            M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
            
            # Apply translation
            aligned = cv2.warpAffine(frame, M, (width, height))
            aligned_frames.append(aligned)
            
        if self.verbose:
            print(f"Aligned frames with offsets: x={x_offset}, y={y_offset}")
            
        return aligned_frames
    
    def _scale_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Scale frames according to x_scale and y_scale from landmark processing.
        
        Args:
            frames: List of frames to scale
            
        Returns:
            List of scaled frames
        """
        x_scale = self.raw_scale_factors_and_offsets['x_scale']
        y_scale = self.raw_scale_factors_and_offsets['y_scale']
        
        height, width = frames[0].shape[:2]
        
        # Calculate new dimensions after scaling
        new_width = int(width * x_scale)
        new_height = int(height * y_scale)
        
        scaled_frames = []
        for frame in frames:
            # Create a resized frame
            resized = cv2.resize(frame, (new_width, new_height))
            
            if new_width <= width and new_height <= height:
                # Smaller or equal to original: create a canvas filled with edge colors
                
                # Create canvas filled with colors from the edges of the original frame
                # For the top and bottom, use the respective rows
                # For the left and right sides, use the respective columns
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Fill with colors from the top edge (first row) of the original frame
                top_color = frame[0:1, :]
                for i in range(height):
                    canvas[i, :] = top_color
                
                # Calculate placement
                # For x: center the resized content
                x_offset = max(0, (width - new_width) // 2)
                # For y: place at the bottom with a small offset
                y_offset = max(0, (height - new_height))
                
                # Place the resized frame on the canvas
                canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            else:
                # Larger than original: crop it to fit the canvas
                crop_x = max(0, (new_width - width) // 2)
                crop_y = max(0, (new_height - height))
                canvas = resized[crop_y:crop_y+height, crop_x:crop_x+width]
                
            scaled_frames.append(canvas)
            
        if self.verbose:
            print(f"Scaled frames with factors: x={x_scale}, y={y_scale}")
            print(f"Using edge colors to fill any empty space from scaling")
            
        return scaled_frames
    
    def _crop_resize_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply any final cropping or resizing to frames.
        
        Args:
            frames: List of frames to process
            
        Returns:
            List of processed frames
        """
        # This is a placeholder for any final resizing or cropping
        # For now, we'll just return the frames as-is
        return frames
    
    def _save_preprocessed_video(self, frames: List[np.ndarray]) -> str:
        """
        Save the fully preprocessed frames to the preprocessed directory.
        
        Args:
            frames: List of frames to save
            
        Returns:
            Path to saved video
        """
        # Ensure output directories exist
        os.makedirs(self.preprocessed_videos_dir, exist_ok=True)
        os.makedirs(self.video_metadata_dir, exist_ok=True)
        
        output_filename = self.filename
        output_path = os.path.join(self.preprocessed_videos_dir, output_filename)
        
        # Try different codecs in order of preference
        # On Linux, we try XVID first as it's commonly available
        if os.name == 'posix':  # Linux/Unix
            codecs = ['XVID', 'mp4v', 'avc1']
        else:  # Windows/Other
            codecs = ['avc1', 'mp4v', 'XVID']
            
        success = False
        used_codec = None
        
        for codec in codecs:
            try:
                if self.verbose:
                    print(f"Attempting to use {codec} codec...")
                    
                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*codec)
                height, width = frames[0].shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
                
                if out.isOpened():
                    # Write frames
                    for frame in frames:
                        out.write(frame)
                    out.release()
                    success = True
                    used_codec = codec
                    if self.verbose:
                        print(f"Successfully saved video using {codec} codec")
                    break
                else:
                    out.release()
                    if self.verbose:
                        print(f"Could not initialize {codec} codec")
            except Exception as e:
                if self.verbose:
                    print(f"Failed to use {codec} codec: {str(e)}")
                continue
        
        if not success:
            raise RuntimeError(
                "Failed to save video with any available codec. "
                "On Linux, you might need to install additional codecs:\n"
                "sudo apt-get update && sudo apt-get install -y ubuntu-restricted-extras"
            )
        
        # Save metadata
        self._save_video_metadata(output_path, used_codec)
        
        if self.verbose:
            print(f">> Saved preprocessed video to {output_path} using {used_codec} codec")
            
        # Update the video metadata CSV
        self._update_video_metadata_csv()
        
        # If save_intermediate is True, also save to debug directory
        if self.save_intermediate:
            os.makedirs(self.interim_debug_videos_dir, exist_ok=True)
            debug_path = os.path.join(self.interim_debug_videos_dir, f"{self.filename.split('.')[0]}_final.mp4")
            
            # Try to save debug video with same codec that worked for main video
            out = cv2.VideoWriter(debug_path, fourcc, self.fps, (width, height))
            if out.isOpened():
                for frame in frames:
                    out.write(frame)
                out.release()
                if self.verbose:
                    print(f">> Saved debug video to {debug_path} using {used_codec} codec")
            else:
                if self.verbose:
                    print(f"Warning: Could not save debug video to {debug_path}")
            
        return output_path
        
    def _save_video_metadata(self, video_path: str, used_codec: str = None) -> None:
        """
        Save metadata for a preprocessed video.
        
        Args:
            video_path: Path to the video file
            used_codec: The codec used to save the video
        """
        def _convert_for_json(obj):
            """Recursively convert data types to JSON serializable types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return _convert_for_json(obj.tolist())
            elif isinstance(obj, dict):
                return {key: _convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list) or isinstance(obj, tuple):
                return [_convert_for_json(item) for item in obj]
            elif hasattr(obj, 'to_dict'):
                return _convert_for_json(obj.to_dict())
            elif hasattr(obj, 'item'):  # For numpy scalars
                return obj.item()
            else:
                return obj
        
        # Create metadata
        original_metadata = self.metadata
        preprocessing_params = self.params
        
        # Calculate processed frame count and duration
        processed_frame_count = self.params["end_frame"] - self.params["start_frame"] + 1
        processed_duration_sec = processed_frame_count / self.fps
        
        # Detailed start/end information
        detailed_start_end_info = {
            "start_time": preprocessing_params['start_frame'] / original_metadata['fps'],
            "end_time": preprocessing_params['end_frame'] / original_metadata['fps'],
            "start_percent": preprocessing_params['start_frame'] / original_metadata['frame_count'],
            "end_percent": preprocessing_params['end_frame'] / original_metadata['frame_count'],
            "new_frame_count": processed_frame_count,
            "new_duration_sec": processed_duration_sec,
            "new_duration_percent": processed_frame_count / original_metadata['frame_count']
        }
        
        metadata = {
                **original_metadata,
                "preprocessing_version": self.preprocess_version,
                **preprocessing_params,
                **detailed_start_end_info,
                "use_stationary_frames": preprocessing_params.get('use_stationary_frames', False),
                "skip_stationary_frames": preprocessing_params.get('skip_stationary_frames', False),
                "codec_used": used_codec,

                "reference_points": self.raw_reference_points,
                "scale_factors_and_offsets": self.raw_scale_factors_and_offsets,
                "landmark_extremes": {"overall": self.raw_landmark_extremes['overall']},
            },
                
        # Convert to JSON serializable types
        metadata = _convert_for_json(metadata)
        
        # Save to individual metadata file
        metadata_filename = f"{self.filename.split('.')[0]}.json"
        metadata_path = os.path.join(self.video_metadata_dir, metadata_filename)
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        if self.verbose:
            print(f">> Saved video metadata to {metadata_path}")
    
    def _update_video_metadata_csv(self) -> None:
        """
        Update the video metadata CSV file.
        Handles schema changes by ensuring new columns are added to existing data.
        """
        csv_path = os.path.join(self.preprocessed_dir, f"video_metadata_{self.preprocess_version}.csv")
        
        # Calculate processed values
        processed_frame_count = self.params["end_frame"] - self.params["start_frame"] + 1
        processed_duration_sec = processed_frame_count / self.fps
        
        # Create a new row for this video
        row = {
            "filename": self.filename,
            "label": self.metadata.get("label", ""),
            "data_source": self.metadata.get("data_source", ""),
            "original_fps": self.fps,
            "original_width": self.width,
            "original_height": self.height,
            "original_frame_count": self.frame_count,
            "original_duration_sec": self.duration_sec,
            "start_frame": self.params["start_frame"],
            "end_frame": self.params["end_frame"],
            "processed_frame_count": processed_frame_count,
            "processed_duration_sec": processed_duration_sec,
            "preprocess_version": self.preprocess_version,
            # Add scaling and alignment info
            "horizontal_offset": self.raw_scale_factors_and_offsets['horizontal_offset'],
            "vertical_offset": self.raw_scale_factors_and_offsets['vertical_offset'],
            "x_scale": self.raw_scale_factors_and_offsets['x_scale'],
            "y_scale": self.raw_scale_factors_and_offsets['y_scale'],
            # Reference the landmarks metadata
            "landmarks_metadata_file": f"{self.filename.split('.')[0]}.json",
            "landmarks_preprocess_version": self.preprocess_version
        }
        
        # Convert row to DataFrame
        new_df = pd.DataFrame([row])
        
        # Track if we're creating a new file
        is_new_file = not os.path.exists(csv_path)
        
        # If file exists, load it and check for schema changes
        old_cols = set()
        if not is_new_file:
            df = pd.read_csv(csv_path)
            old_cols = set(df.columns)
            
            # Check if this file is already in the CSV
            if self.filename in df["filename"].values:
                # Get the index of the existing row
                idx = df.index[df["filename"] == self.filename].tolist()[0]
                
                # Update existing columns and add new ones
                for col in new_df.columns:
                    df.loc[idx, col] = new_df.iloc[0][col]
            else:
                # Append the new row, pandas will automatically align columns
                df = pd.concat([df, new_df], ignore_index=True)
        else:
            # If file doesn't exist, use the new DataFrame
            df = new_df
        
        # Check for column changes before saving
        if not is_new_file:
            new_cols = set(new_df.columns)
            added_cols = new_cols - old_cols
            if added_cols and self.verbose:
                print(f">> Adding new columns to video metadata: {added_cols}")
        
        # Save the CSV
        df.to_csv(csv_path, index=False)
        
        if self.verbose:
            if is_new_file:
                print(f">> Created new video metadata CSV at {csv_path}")
            else:
                print(f">> Updated video metadata CSV at {csv_path}")

    def _save_interim_video(self, frames: List[np.ndarray], step_name: str) -> str:
        """
        Save an intermediate video for debugging.
        
        Args:
            frames: List of frames to save
            step_name: Name of the preprocessing step
            
        Returns:
            Path to saved video
        """
        output_filename = f"{self.filename.split('.')[0]}_{step_name}.mp4"
        output_path = os.path.join(self.interim_debug_videos_dir, output_filename)
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames[0].shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        # Write frames
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        if self.verbose:
            print(f">> Saved intermediate {step_name} video to {output_path}\n")
            
        return output_path

    # ==========================================
    # Common Methods
    # ==========================================
    
    def _update_metadata_csv(self) -> None:
        """
        This method is deprecated. Use _update_landmarks_metadata_csv() or _update_video_metadata_csv() instead.
        """
        pass
