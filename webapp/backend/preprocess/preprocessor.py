import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Class to preprocess video landmarks for sign language prediction.
    """
    def __init__(
        self,
        metadata: pd.Series,
        preprocess_params: Dict[str, Any],
        base_dir: str,
        motion_version: str,
        pose_version: str,
        preprocess_version: str,
        verbose: bool = False,
        save_intermediate: bool = False
    ):
        """
        Initialize Preprocessor with metadata and configuration.

        Args:
            metadata (pd.Series): Video metadata including filename, fps, etc.
            preprocess_params (Dict[str, Any]): Parameters for preprocessing (e.g., normalization targets).
            base_dir (str): Base directory for input and output files.
            motion_version (str): Version for motion detection.
            pose_version (str): Version for pose detection.
            preprocess_version (str): Version for preprocessing.
            verbose (bool): Whether to print detailed logs.
            save_intermediate (bool): Whether to save intermediate results.
        """
        self.metadata = metadata
        self.preprocess_params = preprocess_params
        self.base_dir = Path(base_dir)
        self.motion_version = motion_version
        self.pose_version = pose_version
        self.preprocess_version = preprocess_version
        self.verbose = verbose
        self.save_intermediate = save_intermediate

        # Paths
        self.input_pose_path = self.base_dir / "data" / "interim" / "analysis" / "00" / f"{self.metadata['filename'].replace('.mp4', '')}_pose_{self.pose_version}.npy"
        self.output_dir = self.base_dir / "data" / "preprocessed" / "landmarks" / self.preprocess_version
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / f"{self.metadata['filename'].replace('.mp4', '')}.npy"
        self.metadata_csv_path = self.base_dir / "data" / "preprocessed" / f"landmarks_metadata_{self.preprocess_version}.csv"

        if self.verbose:
            logger.info(f"Initialized Preprocessor for {self.metadata['filename']}")

    def preprocess_landmarks(self) -> Path:
        """
        Preprocess pose landmarks (e.g., normalize, crop frames) and save results.

        Returns:
            Path: Path to the preprocessed landmarks file.
        """
        try:
            # Load pose landmarks
            if not self.input_pose_path.exists():
                logger.error(f"Pose landmarks file not found: {self.input_pose_path}")
                raise FileNotFoundError(f"Pose landmarks file not found: {self.input_pose_path}")
            landmarks = np.load(self.input_pose_path, allow_pickle=True)

            if len(landmarks) == 0:
                logger.error(f"No landmarks found in {self.input_pose_path}")
                raise ValueError("No landmarks available")

            # Crop frames based on motion analysis
            start_frame = self.preprocess_params.get("start_frame", 0)
            end_frame = self.preprocess_params.get("end_frame", len(landmarks) - 1)
            if end_frame < start_frame:
                logger.error(f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}")
                raise ValueError("Invalid frame range")
            landmarks = landmarks[start_frame:end_frame + 1]

            # Normalize landmarks
            normalized_landmarks = self._normalize_landmarks(landmarks)

            # Save preprocessed landmarks
            np.save(self.output_path, normalized_landmarks)
            logger.info(f"Preprocessed landmarks saved to {self.output_path}")

            # Update metadata CSV
            self._update_metadata_csv()

            return self.output_path

        except Exception as e:
            logger.error(f"Preprocessing failed for {self.metadata['filename']}: {str(e)}")
            raise

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        normalized = []
        for frame_landmarks in landmarks:
            if not frame_landmarks:
                normalized.append({'pose_landmarks': {}})
                continue

            # Extract key points (e.g., face, shoulders)
            face_points = []
            shoulder_points = []
            for idx, lm in frame_landmarks.items():
                if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    face_points.append([lm['x'], lm['y']])
                elif idx in [11, 12]:
                    shoulder_points.append([lm['x'], lm['y']])

            if not face_points or not shoulder_points:
                normalized.append({'pose_landmarks': {}})
                continue

            face_points = np.array(face_points)
            shoulder_points = np.array(shoulder_points)

            # Compute normalization factors
            face_width = np.ptp(face_points[:, 0]) if len(face_points) > 1 else 1.0
            shoulders_width = np.ptp(shoulder_points[:, 0]) if len(shoulder_points) > 1 else 1.0
            face_midpoint_y = np.mean(face_points[:, 1]) if len(face_points) > 0 else 0.5
            shoulders_y = np.mean(shoulder_points[:, 1]) if len(shoulder_points) > 0 else 0.5

            # Apply normalization
            scale_x = self.preprocess_params['shoulders_width_aim'] / shoulders_width if shoulders_width > 0 else 1.0
            scale_y = scale_x
            trans_x = self.preprocess_params.get('shoulders_x_aim', 0.5) - np.mean(shoulder_points[:, 0]) * scale_x
            trans_y = self.preprocess_params['shoulders_y_aim'] - shoulders_y * scale_y

            norm_landmarks = {}
            for idx, lm in frame_landmarks.items():
                norm_landmarks[idx] = {
                    'x': lm['x'] * scale_x + trans_x,
                    'y': lm['y'] * scale_y + trans_y,
                    'z': lm.get('z', 0.0),
                    'visibility': lm.get('visibility', 0.0)
                }

            normalized.append({'pose_landmarks': norm_landmarks})

        return np.array(normalized)
    def _update_metadata_csv(self):
        """
        Update the landmarks metadata CSV file with preprocessing details.
        """
        try:
            metadata_entry = {
                'filename': self.metadata['filename'],
                'fps': self.metadata.get('fps', 0.0),
                'frame_count': self.metadata.get('frame_count', 0),
                'preprocess_version': self.preprocess_version,
                'start_frame': self.preprocess_params.get("start_frame", 0),
                'end_frame': self.preprocess_params.get('end_frame', 0),
                'output_path': str(self.output_path)
            }

            if self.metadata_csv_path.exists():
                metadata = pd.read_csv(self.metadata_csv_path)
                metadata = metadata[metadata['filename'] != self.metadata['filename']]  # Remove old entry if exists
                metadata = pd.concat([metadata, pd.DataFrame([metadata_entry])], ignore_index=True)
            else:
                metadata = pd.DataFrame([metadata_entry])

            metadata.to_csv(self.metadata_csv_path, index=False)
            logger.info(f"Updated metadata CSV: {self.metadata_csv_path}")

        except Exception as e:
            logger.error(f"Failed to update metadata CSV for {self.metadata['filename']}: {str(e)}")
            raise