import os
import json
import time
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any, Union
import datetime
import copy

from preprocess import motion_detection as md

try:
    from preprocess import mediapipe_holistic as mph
except ImportError:
    print("MediaPipeHolistic not available. Pose estimation functions will not work.")


def get_processing_params(analysis_info):
    """
    Utility function for getting processing parameters from analysis info.
    Will be applied to the analysis info dict saved at the end of VideoAnalyzer.

    """
    reference_values = {
        'face_horizontal_offset': 0.5,
        'face_vertical_offset': 0.275,
        'face_width': 0.11,
        'face_height': 0.23,
        'shoulders_horizontal_offset': 0.5,
        'shoulders_vertical_offset': 0.52,
        'shoulders_width': 0.25,
        'target_duration': 6
        }

    start_frame = analysis_info["motion_analysis"]["start_frame"]
    end_frame = analysis_info["motion_analysis"]["end_frame"]

    # horizontal offset
    shoulders_median = analysis_info["pose_analysis"]["horizontal_offsets"]["shoulders"]["median"]
    face_median = analysis_info["pose_analysis"]["horizontal_offsets"]["face"]["median"]

    shoulders_reference = 0.5 # (because we want the signer to be horizontally centered)
    face_reference = 0.5 # (because we want the signer to be horizontally centered)

    shoulders_offset = shoulders_reference - shoulders_median
    face_offset = face_reference - face_median

    shoulders_weight = 0.7 # (arbitrarily chosen, because I assume shoulders move less than the face)
    face_weight = 0.3 # (arbitrarily chosen, because I assume shoulders move less than the face)
    horizontal_offset = shoulders_weight * shoulders_offset + face_weight * face_offset

    # Measurements from the video
    ## Horizontal
    shoulder_width = analysis_info["pose_analysis"]["landmark_measurements"]["shoulder_width"]["mean"]
    face_width = analysis_info["pose_analysis"]["landmark_measurements"]["face_width"]["mean"]
    ## Vertical
    face_height = analysis_info["pose_analysis"]["landmark_measurements"]["face_height"]["mean"]
    shoulders_to_bottom = 1 - analysis_info["pose_analysis"]["vertical_offsets"]["shoulders"]["median"]

    # Reference values to scale to
    ## Horizontal
    reference_shoulder_width = reference_values["shoulders_width"]
    reference_face_width = reference_values["face_width"]
    ## Vertical
    reference_face_height = reference_values["face_height"]
    reference_shoulders_to_bottom = 1 - reference_values["shoulders_vertical_offset"]

    # Scale Factors
    ## Horizontal
    shoulder_width_weight = 0.7
    face_width_weight = 0.3
    x_scale_factor = shoulder_width_weight * reference_shoulder_width / shoulder_width + face_width_weight * reference_face_width / face_width
    ## Vertical
    face_height_weight = 0.7
    shoulders_to_bottom_weight = 0.3
    y_scale_factor = face_height_weight * reference_face_height / face_height + shoulders_to_bottom_weight * reference_shoulders_to_bottom / shoulders_to_bottom


    # Measured
    shoulders_median = analysis_info["pose_analysis"]["vertical_offsets"]["shoulders"]["median"]
    face_median = analysis_info["pose_analysis"]["vertical_offsets"]["face"]["median"]
    # Reference
    reference_shoulders = reference_values["shoulders_vertical_offset"]
    reference_face = reference_values["face_vertical_offset"]

    shoulders_offset = reference_shoulders - shoulders_median
    face_offset = reference_face - face_median

    # Weighted Average
    shoulders_weight = 0.6
    face_weight = 0.4
    vertical_offset = shoulders_weight * shoulders_offset + face_weight * face_offset


    params_dict = {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "horizontal_offset": horizontal_offset,
        "vertical_offset": vertical_offset,
        "x_scale_factor": x_scale_factor,
        "y_scale_factor": y_scale_factor,
        "target_duration": reference_values["target_duration"],
    }

    return params_dict

def validate_landmarks_format(landmarks_data: np.ndarray) -> Tuple[bool, List[str]]:
    """
    Validate the basic format of landmarks data.
    
    Args:
        landmarks_data: numpy array of landmark frames, where each frame contains
                       face_landmarks, pose_landmarks, and hand landmarks
        
    Returns:
        Tuple of (is_valid, error_messages)
        
    Note:
        - Coordinate validation is skipped for pose landmarks since they can validly be outside [0,1] range.
        - Face and hand landmarks must be in [0,1] range.
        - Lower body pose landmarks (indices 23-32) should have visibility=0.1.
    """
    landmarks_and_lengths = {
        'face_landmarks': 478,
        'pose_landmarks': 33,  # full body (0-32)
        'left_hand_landmarks': 21,
        'right_hand_landmarks': 21
    }
    errors = []
    
    for i, frame_landmarks in enumerate(landmarks_data):
        if frame_landmarks is None:
            continue
            
        # Check each landmark type
        for key, expected_length in landmarks_and_lengths.items():
            if key not in frame_landmarks:
                errors.append(f"Missing {key} in landmark data frame {i}")
                continue
                
            landmarks = frame_landmarks[key]
            if landmarks is not None:
                # Check length
                actual_length = len(landmarks.landmark) if hasattr(landmarks, 'landmark') else 0
                if actual_length != expected_length:
                    errors.append(f"Invalid length for {key} in frame {i}: expected {expected_length}, got {actual_length}")
                
                # Check coordinate values for face and hand landmarks
                if hasattr(landmarks, 'landmark') and key != 'pose_landmarks':
                    for j, landmark in enumerate(landmarks.landmark):
                        if landmark is not None:
                            if not (0 <= landmark.x <= 1 and 0 <= landmark.y <= 1):
                                errors.append(f"Invalid coordinates in frame {i}, {key} landmark {j}: x={landmark.x}, y={landmark.y}")
                
                # Special check for pose landmarks
                elif key == 'pose_landmarks' and hasattr(landmarks, 'landmark'):
                    for j, landmark in enumerate(landmarks.landmark):
                        if landmark is not None:
                            # For lower body landmarks (23-32), visibility should be 0.1
                            if 23 <= j <= 32:
                                if abs(landmark.visibility - 0.1) > 1e-6:  # Using small epsilon for float comparison
                                    errors.append(f"Lower body pose landmark {j} in frame {i} should have visibility=0.1, got visibility={landmark.visibility}")
                               
    return len(errors) == 0, errors

def hide_lower_body_landmarks(landmarks_data: np.ndarray) -> np.ndarray:
    """
    Process landmarks data while preserving MediaPipe format.
    Note: According to MediaPipe format, if landmarks are present, all landmarks must be included.
    For lower body landmarks (indices 23-32), we set their visibility to 0.1 to effectively hide them
    while maintaining the required format.
    
    Args:
        landmarks_data: np.ndarray containing all landmarks data
        
    Returns:
        np.ndarray with processed landmarks data
    """
    reduced_landmarks_data = []
    
    # Process each frame's landmarks
    for frame_landmarks in landmarks_data:
        if frame_landmarks is None:
            reduced_landmarks_data.append(None)
            continue
            
        # Create a new frame with only the desired landmark types
        new_frame = {}
        for key in ['face_landmarks', 'left_hand_landmarks', 'right_hand_landmarks', 'pose_landmarks']:
            # Keep None entries as is
            if frame_landmarks[key] is None:
                new_frame[key] = None
                continue
                
            # Get the landmarks for this type and create a deep copy
            landmarks_obj = frame_landmarks[key]
            new_frame[key] = copy.deepcopy(landmarks_obj)
            
            # For pose landmarks, set lower body landmarks' visibility to 0.1
            if key == 'pose_landmarks' and new_frame[key] is not None:
                for i in range(23, 33):  # Lower body landmark indices
                    new_frame[key].landmark[i].visibility = 0.1
        
        reduced_landmarks_data.append(new_frame)

    # Convert to numpy array
    reduced_landmarks_data = np.array(reduced_landmarks_data, dtype=object)
    return reduced_landmarks_data

def _init_landmark_stats() -> Dict:
    """Initialize the statistics dictionary for a single landmark type."""
    return {
        'total_frames': 0,
        'none_frames': 0,
        'frame_percentage': 0.0,
        'continuous': True,
        'first_valid': None,
        'last_valid': None,
        'valid_range_total_frames': 0,
        'valid_range_none_frames': 0,
        'valid_range_frame_percentage': 0.0,
        'valid_range_none_details': [],  # List of frame indices that are None in valid range
        'none_details': []  # List of all frame indices that are None
    }

def analyze_none_landmarks(landmarks_data: np.ndarray) -> Dict:
    """
    Analyze None values in landmarks data according to MediaPipe format.
    
    Args:
        landmarks_data: numpy array of landmark frames
        
    Returns:
        Dictionary with statistics about None values, including:
        - Overall statistics about frames with None values
        - For each landmark type (face, pose, left_hand, right_hand):
          * Total frames where the group is None
          * Percentage of frames where the group is None
          * Continuity analysis
          * Frame indices where the group is None (both within valid range and overall)
    """
    # Initialize stats dictionary
    stats = {
        'overall': {
            'total_frames': len(landmarks_data),
            'total_landmark_frames': len(landmarks_data) * 4,  # 4 landmark groups per frame
            'none_landmark_frames': 0,  # Total None landmarks across all groups
            'landmark_frame_percentage': 0.0,  # Percentage of None landmarks
            'none_details': []  # List of frame indices where any landmark group is None
        }
    }
    
    # Expected landmark groups
    landmark_groups = {
        'face_landmarks': 468,
        'pose_landmarks': 23,  # upper body only (0-22)
        'left_hand_landmarks': 21,
        'right_hand_landmarks': 21
    }
    
    # Initialize stats for each landmark group
    for key in landmark_groups:
        stats[key] = _init_landmark_stats()
        stats[key]['total_frames'] = len(landmarks_data)
    
    # Initialize tracking arrays for presence
    presence_by_type = {key: np.zeros(len(landmarks_data), dtype=bool) for key in landmark_groups}
    
    # Process each frame
    for frame_idx, frame_landmarks in enumerate(landmarks_data):
        frame_has_none = False
        
        if frame_landmarks is None:
            # If entire frame is None, all landmark groups are None
            stats['overall']['none_landmark_frames'] += 4  # All 4 groups are None
            frame_has_none = True
            for key in landmark_groups:
                stats[key]['none_frames'] += 1
                stats[key]['none_details'].append(frame_idx)
            continue
            
        # Process each landmark group
        for key in landmark_groups:
            if key not in frame_landmarks:
                stats[key]['none_frames'] += 1
                stats['overall']['none_landmark_frames'] += 1
                stats[key]['none_details'].append(frame_idx)
                frame_has_none = True
                continue
                
            group = frame_landmarks[key]
            if group is None:
                stats[key]['none_frames'] += 1
                stats['overall']['none_landmark_frames'] += 1
                stats[key]['none_details'].append(frame_idx)
                frame_has_none = True
            else:
                presence_by_type[key][frame_idx] = True
        
        if frame_has_none:
            stats['overall']['none_details'].append(frame_idx)
    
    # Calculate valid range statistics for each group
    for key in landmark_groups:
        presence = presence_by_type[key]
        if np.any(presence):
            valid_frames = np.where(presence)[0]
            first_valid = valid_frames[0]
            last_valid = valid_frames[-1]
            
            valid_range_length = last_valid - first_valid + 1
            
            # Update stats
            stats[key].update({
                'continuous': np.all(presence[first_valid:last_valid + 1]),
                'first_valid': int(first_valid),
                'last_valid': int(last_valid),
                'valid_range_total_frames': valid_range_length,
                'valid_range_none_frames': valid_range_length - np.sum(presence[first_valid:last_valid + 1]),
                'valid_range_frame_percentage': (1 - np.mean(presence[first_valid:last_valid + 1])) * 100,
                'valid_range_none_details': [
                    int(idx) for idx in range(first_valid, last_valid + 1) 
                    if not presence[idx]
                ]
            })
        
        # Calculate overall percentage
        stats[key]['frame_percentage'] = (stats[key]['none_frames'] / stats[key]['total_frames']) * 100
        
        # Sort none_details for consistency
        stats[key]['none_details'].sort()
    
    # Calculate overall landmark frame percentage
    stats['overall']['landmark_frame_percentage'] = (
        stats['overall']['none_landmark_frames'] / stats['overall']['total_landmark_frames']
    ) * 100
    
    # Sort overall none_details for consistency
    stats['overall']['none_details'].sort()
    
    return stats

class VideoAnalyzer:
    """
    VideoAnalyzer class for processing and analyzing videos with motion detection and pose estimation.
    
    This class handles:
    - Motion detection and analysis
    - Pose estimation using MediaPipe Holistic
    - Result caching and loading
    - Parameter management
    """
    
    def __init__(
        self,
        metadata_row: Dict,
        timestamp: str,
        path_to_root: str,
        params_dict: Dict = None,
        verbose: bool = False,
        motion_detection_version: str = "versionA",
        pose_detection_version: str = "versionA",
        ):
        """
        Initialize the VideoAnalyzer with metadata and parameters.
        
        Args:
            metadata_row: Dictionary containing video metadata (filename, fps, width, height, etc.)
            timestamp: Timestamp for this analysis run
            path_to_root: Path to the root directory of the project
            params_dict: Dictionary of parameters for analysis (thresholds, etc.)
            verbose: Whether to print details about parameter sources
            motion_detection_version: Version of motion detection to use (default: "versionA")
            pose_detection_version: Version of pose detection to use (default: "versionA")
        """
        # Set required parameters as attributes
        self.timestamp = timestamp
        self.path_to_root = path_to_root
        self.motion_detection_version = motion_detection_version
        self.pose_detection_version = pose_detection_version
        self.verbose = verbose
        
        # Set up base directories
        self.interim_dir = os.path.join(self.path_to_root, "data", "interim")
        self.motion_detection_dir = os.path.join(self.interim_dir, "RawMotionMeasurements")
        self.pose_estimation_dir = os.path.join(self.interim_dir, "RawPoseLandmarks")
        self.results_dir = os.path.join(
            self.interim_dir, 
            "Analysis", 
            f"{self.timestamp}_motion{self.motion_detection_version}_pose{self.pose_detection_version}"
        )
        
        # Create directories if they don't exist
        for directory in [
            os.path.join(self.interim_dir),
            os.path.join(self.motion_detection_dir),
            os.path.join(self.motion_detection_dir, self.motion_detection_version),
            os.path.join(self.pose_estimation_dir),
            os.path.join(self.pose_estimation_dir, self.pose_detection_version), 
            os.path.join(self.results_dir),
        ]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                if self.verbose:
                    print(f"Created directory: {directory}")

        # Set attributes from metadata
        self.attributes_from_metadata(metadata_row)
        
        # Set parameters from params_dict or use defaults
        self.params = {}
        self.check_params(params_dict or {}, metadata_row, verbose=verbose)
        
        # Initialize results containers
        self.motion_data = {}
        self.pose_data = {}
        self.start_frame = None
        self.end_frame = None
        
        if self.verbose:
            print(f"Initialized VideoAnalyzer for {self.filename}")
            print(f"Video properties: {self.fps} fps, {self.width}x{self.height}, {self.duration_sec:.2f} seconds")
    
    def attributes_from_metadata(self, metadata: Dict) -> None:
        """
        Set object attributes from metadata dictionary.
        
        Args:
            metadata: Dictionary containing video metadata
        """
        for key, value in metadata.items():
            setattr(self, key, value)
        
        # Ensure required attributes exist
        required_attrs = ["filename", "data_source", "fps", "width", "height", "duration_sec", "frame_count"]
        missing_attrs = [attr for attr in required_attrs if not hasattr(self, attr)]
        
        if missing_attrs:
            raise ValueError(f"Missing required metadata attributes: {', '.join(missing_attrs)}")
    
    def check_params(self, params: Dict, metadata: Dict, verbose: bool = True) -> None:
        """
        Check and set parameters from params dict, metadata, or defaults.
        
        Args:
            params: Dictionary of parameters for analysis
            metadata: Dictionary containing video metadata
            verbose: Whether to print details about parameter sources
        """
        # Define default parameters
        default_params = {
            # Motion detection parameters
            "motion_detection_method": ["landmarks"],
            "motion_avg_weights": {"basic": 0.3, "bg_sub": 0.7},
            "moving_avg_window_duration": 0.334,  # in seconds
            
            # Motion analysis parameters
            "motion_analysis_method": "simple",  # 'simple' or 'peaks'
            "motion_start_threshold": 0.2,
            "motion_end_threshold": 0.2,
            
            # # Peak detection parameters (used if motion_analysis_method='peaks')
            # "min_motion_peak_height": 0.35,
            # "min_motion_peak_width_seconds": 0.2,
            # "slope_window_size": 5,
            # "min_slope_peak_height": 0.1,
            # "min_slope_peak_width_seconds": 0.2,
            # "start_buffer_seconds": 0.15,
            # "end_buffer_seconds": 0.15,
            
            # Pose estimation parameters
            "pose_static_image_mode": False,
            "pose_model_complexity": 1,
            "pose_smooth_landmarks": True,
            "pose_min_detection_confidence": 0.5,
            "pose_min_tracking_confidence": 0.5,
            "pose_face_ref": "mean_point",  # Reference point for face offset calculation
            
            # Processing parameters
            "reuse_results": True
        }
        
        # Track parameter sources for verbose output
        from_params = {}
        from_metadata = {}
        from_defaults = {}

        # Set parameters, with priority: params > metadata > defaults
        for key, default_value in default_params.items():
            # Try to get from params dict
            if key in params:
                self.params[key] = params[key]
                from_params[key] = params[key]
            # Try to get from metadata
            elif key in metadata:
                self.params[key] = metadata[key]
                from_metadata[key] = metadata[key]
            # Use default
            else:
                self.params[key] = default_value
                from_defaults[key] = default_value
        
        # Save params for reference
        params_path = os.path.join(
            self.results_dir,
            f"{self.filename.split('.')[0]}_params.json"
            )
        with open(params_path, "w") as f:
            json.dump(self.params, f, indent=4)
        
        # Print parameter sources if verbose is True
        if verbose:
            if from_metadata:
                print("Using parameters from metadata:")
                for key, value in from_metadata.items():
                    print(f"\t{key} = {value}")
            
            if from_params:
                print("Using parameters from params dict:")
                for key, value in from_params.items():
                    print(f"\t{key} = {value}")
            
            if from_defaults:
                print("Using parameters from default values:")
                for key, value in from_defaults.items():
                    print(f"\t{key} = {value}")
    
    def get_raw_video_path(self) -> str:
        """Get the path to the raw video file based on metadata."""
        path_to_root = self.path_to_root
        filename = self.filename
        
        # Look in combined folder
        video_path = os.path.join(path_to_root, "data", "interim", "RawCleanVideos", filename)
        if os.path.exists(video_path):
            return video_path
        
        raise FileNotFoundError(f"Video file not found at {video_path}")
    
    def load_previous_results(self, result_type: str, verbose: bool = True) -> Dict:
        """
        Load previously saved results if they exist.
        
        Args:
            result_type: Type of results to load ('motion' or 'pose')
            verbose: Whether to print status messages
            
        Returns:
            Dictionary containing loaded results or empty dict if none found
        """
        motion_path = os.path.join(self.motion_detection_dir, self.motion_detection_version)
        pose_path = os.path.join(self.pose_estimation_dir, self.pose_detection_version)
        
        # Load results
        results = {}
        
        if result_type == "motion":
            # Load motion data
            motion_basic_path = os.path.join(motion_path, f"{self.filename.split('.')[0]}_basic.npy")
            motion_bg_sub_path = os.path.join(motion_path, f"{self.filename.split('.')[0]}_bg_sub.npy")
            motion_landmark_path = os.path.join(motion_path, f"{self.filename.split('.')[0]}_landmark.npy")
            
            if os.path.exists(motion_basic_path):
                results["basic_motion_raw"] = np.load(motion_basic_path)
            elif verbose:
                print(f"Loaded motion basic data from {motion_basic_path}")
            if os.path.exists(motion_bg_sub_path):
                results["bg_sub_motion_raw"] = np.load(motion_bg_sub_path)
            elif verbose:
                print(f"Loaded motion bg sub data from {motion_bg_sub_path}")
            if os.path.exists(motion_landmark_path):
                results["landmarks_motion_raw"] = np.load(motion_landmark_path)
            elif verbose:
                print(f"Loaded motion landmark data from {motion_landmark_path}")
            
        elif result_type == "pose":
            # Load pose data
            params_path = os.path.join(pose_path, "mediapipe_params.json")
            landmarks_path = os.path.join(pose_path, f"{self.filename.split('.')[0]}.npy")
            
            if os.path.exists(params_path) and os.path.exists(landmarks_path):
                with open(params_path, "r") as f:
                    results["params"] = json.load(f)
                
                results["landmarks_raw"] = np.load(landmarks_path, allow_pickle=True)
                if verbose:
                    print(f"Loaded pose estimation results from {self.pose_detection_version}")
            else:
                if verbose:
                    print(f"No pose estimation results found for {self.filename} in {self.pose_detection_version}")
        
        return results
    
    def pose_detect(self, load: Optional[bool] = None) -> Dict:
        """
        Perform pose estimation on the video using MediaPipe Holistic.
        
        Args:
            load: Whether to load previous results if available
            
        Returns:
            Dictionary containing pose estimation results
        """
        verbose = self.params.get("verbose", True)
        # Use provided parameter or fall back to class parameter
        load = load if load is not None else self.params.get("reuse_results")
        
        # Try to load previous results if requested
        if load:
            previous_results = self.load_previous_results("pose", verbose=verbose)
            if previous_results:
                self.pose_data = previous_results
                return self.pose_data
        
        # If no previous results or not loading, run pose estimation
        video_path = self.get_raw_video_path()
        if verbose:
            print(f"Running pose estimation on {video_path}")
        
        # Create MediaPipe instance with parameters
        mp_holistic = mph.MediaPipeHolistic(
            static_image_mode=self.params.get("pose_static_image_mode"),
            model_complexity=self.params.get("pose_model_complexity"),
            smooth_landmarks=self.params.get("pose_smooth_landmarks"),
            min_detection_confidence=self.params.get("pose_min_detection_confidence"),
            min_tracking_confidence=self.params.get("pose_min_tracking_confidence")
        )
        
        # Process video
        landmarks_list = mp_holistic.process_video(video_path)
        
        # Store results
        self.pose_data = {
            "landmarks_raw": landmarks_list,
            "params": {
                "pose_static_image_mode": self.params.get("pose_static_image_mode"),
                "pose_model_complexity": self.params.get("pose_model_complexity"),
                "pose_smooth_landmarks": self.params.get("pose_smooth_landmarks"),
                "pose_min_detection_confidence": self.params.get("pose_min_detection_confidence"),
                "pose_min_tracking_confidence": self.params.get("pose_min_tracking_confidence")
            },
        }
        
        # Save results
        self._save_pose_results(verbose)
        
        return self.pose_data
    
    def _save_pose_results(self, verbose: bool = True) -> None:
        """Save pose estimation results to disk."""
        # Create directory for results
        result_dir = os.path.join(
            self.pose_estimation_dir, 
            self.pose_detection_version
        )
        os.makedirs(result_dir, exist_ok=True)
        
        # Save landmarks data
        np.save(os.path.join(result_dir, f"{self.filename.split('.')[0]}.npy"), self.pose_data["landmarks_raw"])
        
        # Save MediaPipe parameters
        with open(os.path.join(result_dir, "mediapipe_params.json"), "w") as f:
            json.dump(self.pose_data["params"], f, indent=4)

        if verbose:
            print(f"Saved pose estimation results to {result_dir}")

    def pose_analyze(self) -> Dict:
        """
        Analyze pose data to get various measurements and statistics.
        
        Returns:
            Dictionary containing analysis results
        """
        if not hasattr(self, "pose_data") or not self.pose_data:
            raise ValueError("No pose data available. Run pose_detect first.")
        
        if self.verbose:
            print("Analyzing pose data...")
        
        mp_holistic = mph.MediaPipeHolistic()
        # Get horizontal offsets
        horizontal_offsets = mp_holistic.get_video_horizontal_offsets(
            self.pose_data["landmarks_raw"],
            face_ref=self.params.get("pose_face_ref", "mean_point")
        )
        
        # Get vertical offsets
        vertical_offsets = mp_holistic.get_video_vertical_offsets(
            self.pose_data["landmarks_raw"],
            face_ref=self.params.get("pose_face_ref", "mean_point")
        )
        
        # Get landmark measurements
        landmark_measurements = mp_holistic.get_video_landmark_measurements(
            self.pose_data["landmarks_raw"]
        )
        
        # Store analysis results
        self.pose_data["analysis"] = {
            "horizontal_offsets": horizontal_offsets,
            "vertical_offsets": vertical_offsets,
            "landmark_measurements": landmark_measurements
        }
        
        if self.verbose:
            print("Pose analysis complete")
            print(f"Horizontal offsets: {horizontal_offsets}")
            print(f"Vertical offsets: {vertical_offsets}")
            print(f"Landmark measurements: {landmark_measurements}")
        
        return self.pose_data["analysis"]

    def motion_detect(self, window_duration: Optional[float] = None, load: Optional[bool] = None) -> Dict:
        """
        Perform motion detection on the video using landmarks method.
        
        Args:
            window_duration: Duration in seconds for moving average window
            load: Whether to load previous results if available
            
        Returns:
            Dictionary containing motion detection results
        """
        # Check if pose detection has been run
        if not hasattr(self, "pose_data") or not self.pose_data:
            raise ValueError("No pose data available. You must run pose_detect before motion_detect.")
        
        # Use provided parameters or fall back to class parameters
        window_duration = window_duration or self.params.get("moving_avg_window_duration")
        load = load if load is not None else self.params.get("reuse_results")
        
        # Try to load previous results if requested
        if load:
            previous_results = self.load_previous_results("motion", verbose=self.verbose)
            if previous_results:
                self.motion_data = previous_results
                return self.motion_data
        
        # If no previous results or not loading, run motion detection
        if self.verbose:
            print(f"Running motion detection using landmarks method")
        
        # Get landmarks path from pose data
        landmarks_path = os.path.join(
            self.pose_estimation_dir,
            self.pose_detection_version,
            f"{self.filename.split('.')[0]}.npy"
        )
        
        # Run motion detection using landmarks
        motion_raw = md.measure_landmark_change(
            landmarks_path,
            use_pose=True,
            use_face=False,
            use_hands=True,
            combination_method='mean'
        )
        
        # Store raw results with consistent key name
        self.motion_data = {
            "landmarks_motion_raw": np.array(motion_raw)
        }
                
        # Save results
        self._save_raw_motion_results()
        
        return self.motion_data
    
    def _save_raw_motion_results(self, verbose: bool = True) -> None:
        """Save only the raw motion detection results to disk."""
        # Create directory for results
        result_dir = os.path.join(
            self.motion_detection_dir,
            self.motion_detection_version
        )
        os.makedirs(result_dir, exist_ok=True)
        
        # Save raw motion data
        np.save(os.path.join(result_dir, f"{self.filename.split('.')[0]}_landmarks.npy"), self.motion_data["landmarks_motion_raw"])
        
        if verbose:
            print(f"Saved motion detection results to {result_dir}")
    
    def motion_analyze(self, window_duration: Optional[float] = None, load: Optional[bool] = None) -> Dict:
        """
        Analyze motion data to find motion boundaries using either simple threshold or peaks method.
        
        Args:
            window_duration: Duration in seconds for moving average window
            load: Whether to load previous results if available
            
        Returns:
            Dictionary containing motion analysis results
        """
        # Check if motion detection has been run
        if not hasattr(self, "motion_data") or not self.motion_data:
            raise ValueError("No motion data available. You must run motion_detect before motion_analyze.")
        
        # Try to load previous results if requested
        if load:
            previous_results = self.load_previous_results("motion_analysis", verbose=self.verbose)
            if previous_results:
                self.motion_analysis = previous_results
                return self.motion_analysis
        
        # Process raw motion data
        window_duration = window_duration or self.params.get("moving_avg_window_duration")
        # Apply moving average
        smoothed_motion = md.moving_average(self.motion_data["landmarks_motion_raw"], self.fps, window_duration, verbose=self.verbose)
        # Normalize data
        motion_data = md.normalize_list_of_data(smoothed_motion)
        
        # Get parameters
        method = self.params.get("motion_analysis_method", "simple")  # Default to simple method
        
        if self.verbose:
            print(f"Analyzing motion using {method} method")
        
        if method == "peaks":
            # Use peaks method with parameters from self.params
            boundaries = md.find_motion_boundaries_peaks(
                motion_data,
                fps=self.fps,
                min_motion_peak_height=self.params.get("min_motion_peak_height", 0.35),
                min_motion_peak_width_seconds=self.params.get("min_motion_peak_width_seconds", 0.2),
                slope_window_size=self.params.get("slope_window_size", 5),
                min_slope_peak_height=self.params.get("min_slope_peak_height", 0.1),
                min_slope_peak_width_seconds=self.params.get("min_slope_peak_width_seconds", 0.2),
                start_buffer_seconds=self.params.get("start_buffer_seconds", 0.15),
                end_buffer_seconds=self.params.get("end_buffer_seconds", 0.15),
                fallback_simple_start_threshold=self.params.get("motion_start_threshold", 0.3),
                fallback_simple_end_threshold=self.params.get("motion_end_threshold", 0.3),
                verbose=self.verbose
            )
            start_frame, end_frame = boundaries[0], boundaries[1]
        else:  # simple method
            start_frame, end_frame = md.find_motion_boundaries_simple(
                motion_data,
                start_threshold=self.params.get("motion_start_threshold", 0.2),
                end_threshold=self.params.get("motion_end_threshold", 0.2)
            )

        # Hard coding for special cases
        if self.filename == "vagina_ne_1.mp4":
            start_frame = 3
            end_frame = 28
        
        if start_frame is None or end_frame is None:
            raise ValueError("Failed to detect motion boundaries")
        
        # Store results
        self.motion_analysis = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_frame / self.fps,
            "end_time": end_frame / self.fps,
            "duration_frames": end_frame - start_frame,
            "duration_sec": (end_frame - start_frame) / self.fps
        }
        
        # Save results
        self._save_motion_analysis()
        
        return self.motion_analysis
    
    def _save_motion_analysis(self, verbose: bool = True) -> None:
        """Save motion analysis results to disk."""
        # Create directory for results
        result_dir = os.path.join(
            self.motion_detection_dir,
            self.motion_detection_version
        )
        os.makedirs(result_dir, exist_ok=True)
        
        # Save motion analysis data
        np.save(os.path.join(result_dir, f"{self.filename.split('.')[0]}_motion_analysis.npy"), self.motion_analysis)
        
        if verbose:
            print(f"Saved motion analysis results to {result_dir}")

    def save_analysis_info(self) -> Dict:
        """
        Save analysis information to a JSON file.
        
        Returns:
            Dictionary of analysis info that was saved
        """
        def convert_numpy(obj):
            """Convert numpy types to Python native types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        # Check for required data
        if not hasattr(self, "motion_data") or not self.motion_data:
            raise ValueError("Motion data not available. Run motion_detect first.")
        if not hasattr(self, "motion_analysis"):
            raise ValueError("Motion analysis not available. Run motion_analyze first.")
            
        if not hasattr(self, "pose_data") or not self.pose_data:
            raise ValueError("Pose data not available. Run pose_detect first.")
        if "analysis" not in self.pose_data:
            raise ValueError("Pose analysis not available. Run pose_analyze first.")
        
        # Convert metadata values
        metadata = {
            "filename": self.filename,
            "fps": int(self.fps),
            "width": int(self.width),
            "height": int(self.height),
            "duration_sec": float(self.duration_sec),
            "frame_count": int(self.frame_count),
            "data_source": getattr(self, "data_source", None),
            "label": getattr(self, "label", None),
            "motion_detection_version": self.motion_detection_version,
            "pose_detection_version": self.pose_detection_version,
            "timestamp": self.timestamp
        }
        
        # Convert motion analysis values
        motion_analysis = {
            "start_frame": int(self.motion_analysis["start_frame"]),
            "end_frame": int(self.motion_analysis["end_frame"]),
            "start_time": float(self.motion_analysis["start_time"]),
            "end_time": float(self.motion_analysis["end_time"]),
            "duration_frames": int(self.motion_analysis["duration_frames"]),
            "duration_sec": float(self.motion_analysis["duration_sec"])
        }
        
        # Collect all analysis info
        analysis_info = {
            "metadata": metadata,
            "params": self.params,
            "motion_analysis": motion_analysis,
        }
        
        # Add pose analysis results
        pose_analysis = self.pose_data["analysis"]
        analysis_info["pose_analysis"] = {
            "frames_with_landmarks": len(self.pose_data["landmarks_raw"]),
            "horizontal_offsets": convert_numpy(pose_analysis["horizontal_offsets"]),
            "vertical_offsets": convert_numpy(pose_analysis["vertical_offsets"]),
            "landmark_measurements": convert_numpy(pose_analysis["landmark_measurements"]),
        }
        
        # Save to file
        output_path = os.path.join(self.results_dir, f"{self.filename.split('.')[0]}_analysis_info.json")
        with open(output_path, "w") as f:
            json.dump(analysis_info, f, indent=4)
        
        if self.verbose:
            print(f"Saved analysis info to {output_path}")
        return analysis_info