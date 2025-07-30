import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import mediapipe as mp
from typing import Dict, Any, Optional
import logging
import json



logger = logging.getLogger(__name__)

class VideoAnalyzer:
    """
    Class to analyze videos for pose detection and motion analysis.
    """
    def __init__(
        self,
        metadata: pd.Series,
        timestamp: str,
        base_dir: str,
        verbose: bool = False,
        motion_detection_version: str = "v0",
        pose_detection_version: str = "v0"
    ):
        """
        Initialize VideoAnalyzer with metadata and configuration.

        Args:
            metadata (pd.Series): Video metadata including filename, fps, etc.
            timestamp (str): Timestamp for output directories.
            base_dir (str): Base directory for saving analysis results.
            verbose (bool): Whether to print detailed logs.
            motion_detection_version (str): Version for motion detection.
            pose_detection_version (str): Version for pose detection.
        """
        self.metadata = metadata
        self.timestamp = timestamp
        self.base_dir = Path(base_dir)
        self.verbose = verbose
        self.motion_version = motion_detection_version
        self.pose_version = pose_detection_version

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Paths for saving results
        self.analysis_dir = self.base_dir / "data" / "interim" / "analysis" / f"{self.timestamp}"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.pose_results_path = self.analysis_dir / f"{self.metadata['filename'].replace('.mp4', '')}_pose_{self.pose_version}.npy"
        self.motion_results_path = self.analysis_dir / f"{self.metadata['filename'].replace('.mp4', '')}_motion_{self.motion_version}.npy"

        if self.verbose:
            logger.info(f"Initialized VideoAnalyzer for {self.metadata['filename']}")

    def pose_detect(self) -> None:
        """
        Perform pose detection on the video and save landmarks.
        """
        try:
            video_path = self.base_dir / "data" / "interim" / "RawCleanVideos" / self.metadata['filename']
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                raise ValueError(f"Cannot open video: {video_path}")

            landmarks = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)

                if results.pose_landmarks:
                    landmarks.append(self._landmarks_to_dict(results.pose_landmarks))
                else:
                    landmarks.append({})

            cap.release()
            np.save(self.pose_results_path, landmarks)
            logger.info(f"Pose detection completed, saved to {self.pose_results_path}")

        except Exception as e:
            logger.error(f"Pose detection failed for {self.metadata['filename']}: {str(e)}")
            raise

    def pose_analyze(self) -> None:
        """
        Analyze pose landmarks to compute additional metrics (e.g., keypoint statistics).
        """
        try:
            landmarks = np.load(self.pose_results_path, allow_pickle=True)
            if len(landmarks) == 0:
                logger.error(f"No pose landmarks found in {self.pose_results_path}")
                raise ValueError("No pose landmarks available")

            # Example analysis: Compute average keypoint positions
            valid_landmarks = [lm for lm in landmarks if lm]
            if not valid_landmarks:
                logger.warning(f"No valid landmarks for {self.metadata['filename']}")
                return

            # Placeholder for pose analysis logic (e.g., compute distances, angles)
            logger.info(f"Pose analysis completed for {self.metadata['filename']}")

        except Exception as e:
            logger.error(f"Pose analysis failed for {self.metadata['filename']}: {str(e)}")
            raise

    def motion_detect(self) -> None:
        """
        Perform motion detection on the video and save results.
        """
        try:
            video_path = self.base_dir / "data" / "interim" / "RawCleanVideos" / self.metadata['filename']
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                raise ValueError(f"Cannot open video: {video_path}")

            prev_frame = None
            motion_scores = []
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_frame is not None:
                    diff = cv2.absdiff(gray, prev_frame)
                    motion_score = np.mean(diff)
                    motion_scores.append(motion_score)
                prev_frame = gray
                frame_idx += 1

            cap.release()
            np.save(self.motion_results_path, motion_scores)
            logger.info(f"Motion detection completed, saved to {self.motion_results_path}")

        except Exception as e:
            logger.error(f"Motion detection failed for {self.metadata['filename']}: {str(e)}")
            raise

    def motion_analyze(self) -> Dict[str, Any]:
        """
        Analyze motion results to determine start and end frames of significant motion.

        Returns:
            Dict[str, Any]: Motion analysis results (start_frame, end_frame, etc.).
        """
        try:
            motion_scores = np.load(self.motion_results_path, allow_pickle=True)
            if len(motion_scores) == 0:
                logger.error(f"No motion scores found in {self.motion_results_path}")
                raise ValueError("No motion scores available")

            # Simple threshold-based detection
            threshold = np.mean(motion_scores) + np.std(motion_scores)
            significant_motion = motion_scores > threshold
            start_frame = np.argmax(significant_motion) if np.any(significant_motion) else 0
            end_frame = len(motion_scores) - 1 - np.argmax(significant_motion[::-1]) if np.any(significant_motion) else len(motion_scores) - 1

            results = {
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "motion_scores": motion_scores.tolist()
            }
            logger.info(f"Motion analysis completed for {self.metadata['filename']}: {results}")
            return results

        except Exception as e:
            logger.error(f"Motion analysis failed for {self.metadata['filename']}: {str(e)}")
            raise

    def save_analysis_info(self) -> Dict[str, Any]:
        """
        Save analysis results to a JSON file and return them.

        Returns:
            Dict[str, Any]: Combined pose and motion analysis results.
        """
        try:
            analysis_info = {
                "pose_analysis": {
                    "pose_results_path": str(self.pose_results_path),
                    "pose_version": self.pose_version
                },
                "motion_analysis": self.motion_analyze(),
                "motion_results_path": str(self.motion_results_path),
                "motion_version": self.motion_version
            }
            output_path = self.analysis_dir / f"{self.metadata['filename'].replace('.mp4', '')}_analysis_info.json"
            with open(output_path, "w") as f:
                json.dump(analysis_info, f, indent=2)
            logger.info(f"Analysis info saved to {output_path}")
            return analysis_info

        except Exception as e:
            logger.error(f"Failed to save analysis info for {self.metadata['filename']}: {str(e)}")
            raise

    def _landmarks_to_dict(self, landmarks) -> Dict[str, Any]:
        """
        Convert MediaPipe landmarks to a dictionary.

        Args:
            landmarks: MediaPipe pose landmarks object.

        Returns:
            Dict[str, Any]: Dictionary of landmark coordinates.
        """
        return {
            idx: {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            } for idx, lm in enumerate(landmarks.landmark)
        }



