import cv2
from typing import List, Dict
import mediapipe as mp

# Initialize MediaPipe solutions
mp_holistic = mp.solutions.holistic

def extract_landmarks_from_video(video_path: str) -> List[Dict[str, any]]:
    """
    Extract pose and hand landmarks from every frame of the video.

    Args:
        video_path: Path to the video file.

    Returns:
        A list of dictionaries containing landmarks for each frame.
        Example:
        [
            {
                "pose_landmarks": results.pose_landmarks,
                "left_hand_landmarks": results.left_hand_landmarks,
                "right_hand_landmarks": results.right_hand_landmarks
            },
            ...
        ]
    """
    cap = cv2.VideoCapture(video_path)
    holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1)

    frame_landmarks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Holistic
        results = holistic.process(rgb_frame)

        frame_data = {
            "pose_landmarks": results.pose_landmarks,
            "left_hand_landmarks": results.left_hand_landmarks,
            "right_hand_landmarks": results.right_hand_landmarks
        }

        frame_landmarks.append(frame_data)

    cap.release()
    holistic.close()
    return frame_landmarks
