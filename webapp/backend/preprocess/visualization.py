import cv2
from typing import List, Dict
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def get_frame(video_path: str, frame_indices: List[int]) -> List:
    """
    Extract specific frames from a video.

    Args:
        video_path: Path to the video file.
        frame_indices: List of frame indices to extract.

    Returns:
        List of BGR frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def extract_all_frames(video_path: str) -> List:
    """
    Extract all frames from a video.

    Args:
        video_path: Path to the video file.

    Returns:
        List of BGR frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def visualize_landmarks(frame, landmarks: Dict[str, list], connections=None):
    """
    Draw landmarks on a frame for debugging/visualization.

    Args:
        frame: BGR image (numpy array).
        landmarks: Dict of landmark lists (e.g., {'pose': [...], 'hand': [...]})
        connections: Optional connections for drawing skeletons.
    """
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils

    for lm_type, lm_list in landmarks.items():
        if lm_list:
            mp_drawing.draw_landmarks(frame, lm_list, connections[lm_type] if connections else None)

    return frame


def draw_landmarks_on_video_with_frame(
    results_list: np.ndarray,
    output_path: str,
    fps: float
) -> List[np.ndarray]:
    """
    Draw pose landmarks on video frames and save the resulting video.

    Args:
        results_list (np.ndarray): Array of dictionaries containing landmark data for each frame.
        output_path (str): Path to save the output video file.
        fps (float): Frames per second for the output video.

    Returns:
        List[np.ndarray]: List of annotated frames.
    """
    try:
        # Validate inputs
        if len(results_list) == 0:
            logger.error("Empty results list provided")
            raise ValueError("Results list is empty")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize video writer
        frame_shape = (480, 640, 3)  # Default resolution (height, width, channels)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_shape[1], frame_shape[0]))

        if not out.isOpened():
            logger.error(f"Failed to initialize video writer for {output_path}")
            raise RuntimeError(f"Cannot open video writer for {output_path}")

        annotated_frames = []

        # Process each frame
        for frame_idx, frame_data in enumerate(results_list):
            # Create a blank frame
            frame = np.zeros(frame_shape, dtype=np.uint8)

            if not frame_data:
                logger.debug(f"No landmarks for frame {frame_idx}")
                annotated_frames.append(frame)
                out.write(frame)
                continue

            # Access landmarks (handle both direct and wrapped structures)
            landmarks = frame_data.get('pose_landmarks', frame_data)

            if not landmarks:
                logger.debug(f"No landmarks for frame {frame_idx}")
                annotated_frames.append(frame)
                out.write(frame)
                continue

            # Draw landmarks and connections
            for idx, lm in landmarks.items():
                # Ensure lm has 'x' and 'y' keys
                if 'x' not in lm or 'y' not in lm:
                    logger.debug(f"Missing 'x' or 'y' in landmark {idx} for frame {frame_idx}")
                    continue
                x = int(lm['x'] * frame_shape[1])
                y = int(lm['y'] * frame_shape[0])
                # Draw landmark point
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Draw connections (example: connect shoulders, elbows, etc.)
            connections = [
                (11, 12),  # Shoulders
                (11, 13),  # Left shoulder to elbow
                (12, 14),  # Right shoulder to elbow
                (13, 15),  # Left elbow to wrist
                (14, 16),  # Right elbow to wrist
            ]
            for start_idx, end_idx in connections:
                start_key, end_key = str(start_idx), str(end_idx)
                if start_key in landmarks and end_key in landmarks:
                    start = landmarks[start_key]
                    end = landmarks[end_key]
                    if 'x' not in start or 'y' not in start or 'x' not in end or 'y' not in end:
                        logger.debug(f"Missing coordinates for connection {start_idx}-{end_idx} in frame {frame_idx}")
                        continue
                    start_pt = (int(start['x'] * frame_shape[1]), int(start['y'] * frame_shape[0]))
                    end_pt = (int(end['x'] * frame_shape[1]), int(end['y'] * frame_shape[0]))
                    cv2.line(frame, start_pt, end_pt, (255, 255, 0), 2)

            # Add frame number
            cv2.putText(
                frame,
                f"Frame: {frame_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            annotated_frames.append(frame)
            out.write(frame)

        # Release video writer
        out.release()
        logger.info(f"Skeleton video saved to {output_path}")

        return annotated_frames

    except Exception as e:
        logger.error(f"Failed to draw landmarks on video: {str(e)}")
        raise e
                     