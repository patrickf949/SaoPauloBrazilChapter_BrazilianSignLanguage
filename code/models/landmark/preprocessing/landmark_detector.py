import cv2
import numpy as np
import mediapipe as mp
import os
from pathlib import Path
from tqdm import tqdm
import random
from typing import Tuple, List, Optional

class LandmarkDetector:
    def __init__(self, num_frames: int = 113):
        # Initialize Mediapipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.num_frames = num_frames  # Number of frames to standardize to

    def process_dataset(self,
                       train_path: str,
                       test_path: str,
                       output_base_folder: str,
                       augment_training: bool = True,
                       num_augmentations: int = 4) -> None:
        """
        Main pipeline to process both training and testing datasets
        """
        # Create output directories
        output_base = Path(output_base_folder)
        train_output = output_base / 'train'
        test_output = output_base / 'test'

        # Process training data with augmentation
        print("Processing training data...")
        self._process_dataset_split(
            train_path,
            train_output,
            is_training=True,
            augment=augment_training,
            num_augmentations=num_augmentations
        )

        # Process test data without augmentation
        print("\nProcessing test data...")
        self._process_dataset_split(
            test_path,
            test_output,
            is_training=False
        )

    def _process_dataset_split(self,
                             input_path: str,
                             output_path: Path,
                             is_training: bool = False,
                             augment: bool = False,
                             num_augmentations: int = 4) -> None:
        """
        Process a single dataset split (train or test)
        """
        input_path = Path(input_path)
        output_path.mkdir(parents=True, exist_ok=True)

        for video_file in tqdm(input_path.glob('*.mp4'), desc="Processing words"):
            # Process original video
            processed_frames = self._process_video(str(video_file))
            if processed_frames is not None:
                keypoints = self._extract_keypoints_sequence(processed_frames)

                # Reshape keypoints to (113, 225) format
                reshaped_keypoints = self._reshape_keypoints(keypoints)

                # Save original keypoints
                output_file =  output_path / f"{video_file.stem}.npy"
                print(output_file)
                np.save(output_file, reshaped_keypoints)

                # Generate augmentations for training data
                if is_training and augment:
                    for i in range(num_augmentations):
                        aug_keypoints = self._augment_landmarks(keypoints)
                        aug_reshaped = self._reshape_keypoints(aug_keypoints)
                        aug_output_file = output_path / f"{video_file.stem}_aug_{i+1}.npy"
                        np.save(aug_output_file, aug_reshaped)

    def _reshape_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Reshape keypoints from (num_frames, 75, 3) to (113, 225) format
        75 landmarks = 33 pose + 21 left hand + 21 right hand
        113 = number of selected frames
        225 = 75 landmarks * 3 coordinates
        """
        # First ensure we have exactly num_frames frames
        if len(keypoints) != self.num_frames:
            indices = np.linspace(0, len(keypoints)-1, self.num_frames, dtype=int)
            keypoints = keypoints[indices]

        # Reshape to 113 frames
        keypoints_113 = np.zeros((113, 225))  # Initialize with zeros

        # Fill the available frames (up to min(113, num_frames))
        num_frames_to_use = min(113, self.num_frames)

        # Reshape each frame's keypoints to a 1D array of 225 values
        for i in range(num_frames_to_use):
            flat_keypoints = keypoints[i].reshape(-1)  # Flatten 75x3 to 225
            keypoints_113[i] = flat_keypoints

        return keypoints_113

    def _process_video(self, video_path: str) -> Optional[List[np.ndarray]]:
        """
        Process a single video: trim and standardize frames
        """
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            if not frames:
                return None

            # Calculate motion scores for trimming
            motion_scores = []
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i-1], frames[i])
                motion_scores.append(np.mean(diff))

            # Trim based on motion
            threshold = np.mean(motion_scores) * 0.3
            start_idx = next((i for i, score in enumerate(motion_scores)
                            if score > threshold), 0)
            end_idx = len(frames) - next((i for i, score in enumerate(reversed(motion_scores))
                                        if score > threshold), 0)

            # Apply trimming
            frames = frames[max(0, start_idx-5):min(len(frames), end_idx+5)]

            # Standardize length
            if frames:
                indices = np.linspace(0, len(frames)-1, self.num_frames, dtype=int)
                frames = [frames[i] for i in indices]
                return frames

            return None

        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return None

    def _extract_keypoints_sequence(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract keypoints from a sequence of frames
        """
        keypoints_sequence = []

        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(frame_rgb)

            # Initialize keypoints arrays
            pose_keypoints = np.zeros((33, 3))
            left_hand_keypoints = np.zeros((21, 3))
            right_hand_keypoints = np.zeros((21, 3))

            # Extract pose landmarks if detected
            if results.pose_landmarks:
                pose_keypoints = np.array([[lm.x, lm.y, lm.z]
                                         for lm in results.pose_landmarks.landmark])

            # Extract hand landmarks if detected
            if results.left_hand_landmarks:
                left_hand_keypoints = np.array([[lm.x, lm.y, lm.z]
                                              for lm in results.left_hand_landmarks.landmark])
            if results.right_hand_landmarks:
                right_hand_keypoints = np.array([[lm.x, lm.y, lm.z]
                                               for lm in results.right_hand_landmarks.landmark])

            # Combine all keypoints
            frame_keypoints = np.concatenate([
                pose_keypoints,
                left_hand_keypoints,
                right_hand_keypoints
            ])
            keypoints_sequence.append(frame_keypoints)

        return np.array(keypoints_sequence)

    def _augment_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to landmarks sequence
        """
        angle = random.uniform(-10, 10) if random.random() < 0.7 else 0
        noise_std = 0.01

        augmented_frames = []
        for frame in landmarks:
            augmented = frame.copy()

            # Apply rotation
            if angle != 0:
                augmented[:, :2] = self._rotate_landmarks(augmented, angle_deg=angle)

            # Apply noise
            if random.random() < 0.5:
                augmented = self._add_noise(augmented, noise_std=noise_std)

            augmented_frames.append(augmented)

        return np.array(augmented_frames)

    def _rotate_landmarks(self, landmarks: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        Rotate landmarks in X-Y plane
        """
        angle_rad = np.radians(angle_deg)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        return landmarks[:, :2] @ rotation_matrix.T

    def _add_noise(self, landmarks: np.ndarray, noise_std: float) -> np.ndarray:
        """
        Add Gaussian noise to landmarks
        """
        noise = np.random.normal(0, noise_std, landmarks.shape)
        return landmarks + noise