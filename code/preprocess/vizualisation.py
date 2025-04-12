import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
import mediapipe as mp
from typing import Dict, List, Optional, Tuple

def get_frame(frame_index, video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # Set frame index
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Could not read frame {frame_index} from video")
        frame = None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def landmarks_to_key_landmarks(mediapipe_landmarks_set):
    face_horizontal = mediapipe_landmarks_set['face_landmarks'].landmark[5].x # nose tip
    face_vertical = mediapipe_landmarks_set['face_landmarks'].landmark[5].y # nose tip
    left_face = mediapipe_landmarks_set['face_landmarks'].landmark[234]  # Left face edge
    right_face = mediapipe_landmarks_set['face_landmarks'].landmark[454]  # Right face edge
    face_width = np.sqrt(
        (left_face.x - right_face.x)**2 + 
        (left_face.y - right_face.y)**2
    )
    top_head = mediapipe_landmarks_set['face_landmarks'].landmark[10]  # Top of head
    chin = mediapipe_landmarks_set['face_landmarks'].landmark[152]  # Chin
    face_height = np.sqrt(
        (top_head.x - chin.x)**2 + 
        (top_head.y - chin.y)**2
    )
    left_shoulder = mediapipe_landmarks_set['pose_landmarks'].landmark[11]
    right_shoulder = mediapipe_landmarks_set['pose_landmarks'].landmark[12]
    shoulders_horizontal = (left_shoulder.x + right_shoulder.x) / 2
    shoulders_vertical = (left_shoulder.y + right_shoulder.y) / 2
    shoulders_width = np.sqrt(
        (left_shoulder.x - right_shoulder.x)**2 + 
        (left_shoulder.y - right_shoulder.y)**2
    )
    
    key_points = {
        'face_horizontal_offset': face_horizontal,
        'face_vertical_offset': face_vertical,
        'face_width': face_width,
        'face_height': face_height,
        'shoulders_horizontal_offset': shoulders_horizontal,
        'shoulders_vertical_offset': shoulders_vertical,
        'shoulders_width': shoulders_width,
    }
    return key_points

def plot_key_landmarks(
    landmark_points: dict, 
    frame = None, 
    line_color = (0,255,0),
    line_width = 2,
    axis_lines = True,
    triangle = True,
    ):
    """
    Plotting the key landmarks used as references to get the Horizontal and Vertical offsets, and the Scaling Factors.
    """
    if frame is None:
        frame = np.zeros([1000,1000,3])
    frame_height, frame_width = frame.shape[0], frame.shape[1]

    # get face points    
    face_center_x = landmark_points['face_horizontal_offset']*frame_width
    face_center_y = landmark_points['face_vertical_offset']*frame_height
    face_width = landmark_points['face_width']*frame_width
    face_height = landmark_points['face_height']*frame_height

    # draw an oval matching the points
    # cv2.ellipse(image, center, axes, angle, startAngle, endAngle, color, thickness)
    cv2.ellipse(
        frame, 
        (int(face_center_x), int(face_center_y)), 
        (int(face_width/2), int(face_height/2)), 
        0, 
        0, 
        360, 
        line_color, 
        line_width
    )
    # draw line on the ellipse axes
    if axis_lines:
        cv2.line(
            frame, 
            (int(face_center_x - face_width/2), int(face_center_y)), 
            (int(face_center_x + face_width/2), int(face_center_y)), 
            line_color, 
            line_width
        )
        cv2.line(
            frame, 
            (int(face_center_x), int(face_center_y - face_height/2)), 
            (int(face_center_x), int(face_center_y + face_height/2)), 
            line_color, 
            line_width
        )
    
    # get shoulder points
    shoulders_center_x = landmark_points['shoulders_horizontal_offset']*frame_width
    shoulders_center_y = landmark_points['shoulders_vertical_offset']*frame_height
    shoulders_width = landmark_points['shoulders_width']*frame_width

    if triangle:
        shoulder_left = (int(shoulders_center_x - shoulders_width/2), int(shoulders_center_y))
        shoulder_right = (int(shoulders_center_x + shoulders_width/2), int(shoulders_center_y))
        torso_bottom = (int(shoulders_center_x), frame.shape[0])

        cv2.line(
            frame,
            shoulder_left,
            shoulder_right,
            line_color, 
            line_width
        )
        cv2.line(
            frame,
            shoulder_left,
            torso_bottom,
            line_color, 
            line_width
        )
        cv2.line(
            frame,
            shoulder_right,
            torso_bottom,
            line_color, 
            line_width
        )
        
    if not triangle:
        # draw a rectangle from the shoulder line to the bottom
        shoulder_left = (int(shoulders_center_x - shoulders_width/2), int(shoulders_center_y))
        shoulder_right = (int(shoulders_center_x + shoulders_width/2), frame.shape[0]) 
        cv2.rectangle(
            frame,
            shoulder_left,
            shoulder_right,
            line_color, 
            line_width
        )
    if axis_lines:
        # draw line on the shoulder line
        cv2.line(
            frame, 
            (int(shoulders_center_x), int(shoulders_center_y)), 
            (int(shoulders_center_x), frame.shape[0]), 
            line_color, 
            line_width
        )
    return frame

### MediaPipe Pose Landmarks Visualisation


class MediaPipeHolistic:
    def __init__(self,
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 refine_face_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Holistic detector for combined face, pose, and hand detection.
        
        Args:
            static_image_mode: Whether to treat input images as a video stream (False) or independent images (True).
                             For video processing, set to False to enable tracking instead of detection per image.
            model_complexity: Complexity of the pose landmark model: 0, 1 or 2. 
                            Landmark accuracy as well as inference latency generally go up with the model complexity.
                            0: Fastest but least accurate
                            1: Good balance of speed and accuracy
                            2: Slowest but most accurate
            smooth_landmarks: Whether to filter landmarks across different input images to reduce jitter.
                            Should be disabled for static images.
            enable_segmentation: Whether to generate segmentation mask.
            smooth_segmentation: Whether to filter segmentation across different input images to reduce jitter.
                               Should be disabled for static images.
            refine_face_landmarks: Whether to further refine the landmark coordinates around the eyes and lips, 
                                 and output additional landmarks around the irises.
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for person detection to be considered successful.
            min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the pose landmarks to be considered tracked 
                                  successfully. Only used when static_image_mode is False.
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            refine_face_landmarks=refine_face_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )    

    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> Dict:
        """
        Process a single frame and detect holistic landmarks.
        
        Args:
            frame: Input frame in BGR format
            timestamp_ms: Timestamp of the frame in milliseconds
        
        Returns:
            Dictionary containing detection results with the following keys:
            - face_landmarks: 468 face landmarks if detected
            - pose_landmarks: 33 pose landmarks if detected
            - left_hand_landmarks: 21 left hand landmarks if detected
            - right_hand_landmarks: 21 right hand landmarks if detected
            - timestamp_ms: Frame timestamp
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with holistic model
        results = self.holistic.process(frame_rgb)  # Pass numpy array directly
        
        # Package results
        return {
            'face_landmarks': results.face_landmarks,  # 468 landmarks
            'pose_landmarks': results.pose_landmarks,  # 33 landmarks
            'left_hand_landmarks': results.left_hand_landmarks,  # 21 landmarks
            'right_hand_landmarks': results.right_hand_landmarks,  # 21 landmarks
            'pose_world_landmarks': results.pose_world_landmarks,  # 3D pose landmarks
            'timestamp_ms': timestamp_ms
        }

    def process_video(self, video_path: str, output_path: Optional[str] = None) -> List[Dict]:
        """
        Process a video file and detect holistic landmarks in each frame.
        No post-processing (filtering, centering, scaling) is applied.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save the processed video with visualizations
        
        Returns:
            List of dictionaries containing detection results for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_time_ms = int(1000 / fps)
        all_results = []
        current_time_ms = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                results = self.process_frame(frame, current_time_ms)
                all_results.append(results)
                
                # Visualize if output path is provided
                if output_path:
                    annotated_frame = self.draw_landmarks_on_frame(frame, results)
                    out.write(annotated_frame)
                
                current_time_ms += frame_time_ms
        
        finally:
            cap.release()
            if output_path:
                out.release()
        
        return all_results
    
    def __del__(self):
        """Clean up resources"""
        self.holistic.close()

    def draw_landmarks_on_frame(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw landmarks and connections on the image.
        
        Args:
            image: Input image in BGR format
            results: Dictionary containing detection results
        
        Returns:
            Image with landmarks drawn
        """
        # Create a copy of the image to draw on
        annotated_image = image.copy()
        
        # Draw face landmarks
        if results['face_landmarks']:
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=results['face_landmarks'],
                connections=self.mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            # Draw face contours
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=results['face_landmarks'],
                connections=self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        
        # Draw pose landmarks
        if results['pose_landmarks']:
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=results['pose_landmarks'],
                connections=self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw left hand landmarks
        if results['left_hand_landmarks']:
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=results['left_hand_landmarks'],
                connections=self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Draw right hand landmarks
        if results['right_hand_landmarks']:
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=results['right_hand_landmarks'],
                connections=self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return annotated_image

    def draw_landmarks_on_video(self, 
                              video_path: str,
                              results_list: List[Dict],
                              output_path: str) -> None:
        """
        Process a video and draw landmarks on each frame.
        
        Args:
            video_path: Path to input video file
            results_list: List of dictionaries containing detection results for each frame
            output_path: Path to save the processed video with landmarks drawn
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx >= len(results_list):
                    break
                
                # Get results for this frame
                results = results_list[frame_idx]
                
                # Draw landmarks on frame
                annotated_frame = self.draw_landmarks_on_frame(frame, results)
                
                # Write to output video
                out.write(annotated_frame)
                
                frame_idx += 1
        
        finally:
            cap.release()
            out.release()
        print(f"Landmarks drawn video saved to {output_path}")

    def draw_landmarks_3d(self, results: Dict, figsize: Tuple[int, int] = (10, 10),
                         elev: float = 0, azim: float = 0,
                         xlim: Optional[Tuple[float, float]] = None,
                         ylim: Optional[Tuple[float, float]] = None,
                         zlim: Optional[Tuple[float, float]] = None) -> plt.Figure:
        """
        Draw landmarks in 3D using matplotlib.
        
        Args:
            results: Dictionary containing detection results
            figsize: Figure size in inches (width, height)
            elev: Elevation angle in degrees (vertical rotation)
                 0 = front view, positive = looking down, negative = looking up
            azim: Azimuth angle in degrees (horizontal rotation)
                 0 = front view, 90 = right side, -90 = left side, 180 = back
            xlim: Optional tuple of (min, max) for x-axis limits
            ylim: Optional tuple of (min, max) for y-axis limits
            zlim: Optional tuple of (min, max) for z-axis limits
        
        Returns:
            Matplotlib figure with 3D visualization
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Draw pose landmarks if available
        if results['pose_world_landmarks']:
            landmarks = results['pose_world_landmarks'].landmark
            
            # Transform coordinates to make front view the default
            # MediaPipe: x = right/left, y = up/down, z = forward/backward
            # Desired: x = right/left, y = forward/backward, z = up/down
            x = [lm.x for lm in landmarks]  # x stays the same (right/left)
            y = [-lm.z for lm in landmarks]  # negative z becomes y (forward/backward)
            z = [-lm.y for lm in landmarks]  # negative y becomes z (up/down)
            
            # Draw hands first if available, using pose landmarks as reference points
            hand_landmarks = {
                'left_hand_landmarks': (
                    '#FF3366', 'Left Hand',  # Bright pink
                    landmarks[15],  # Left wrist landmark
                    landmarks[13]   # Left elbow landmark for orientation
                ),
                'right_hand_landmarks': (
                    '#33FF66', 'Right Hand',  # Bright green
                    landmarks[16],  # Right wrist landmark
                    landmarks[14]   # Right elbow landmark for orientation
                )
            }
            
            for hand_key, (color, label, wrist_ref, elbow_ref) in hand_landmarks.items():
                if results[hand_key]:
                    hand_landmarks = results[hand_key].landmark
                    
                    # Transform hand coordinates
                    hand_x = []
                    hand_y = []
                    hand_z = []
                    scale = 1  # Increased scale factor for hand size
                    
                    # Transform wrist reference point
                    wrist_pos = np.array([wrist_ref.x, -wrist_ref.z, -wrist_ref.y])
                    
                    for lm in hand_landmarks:
                        # Center hand at wrist and scale
                        rel_x = (lm.x - hand_landmarks[0].x) * scale
                        rel_y = -(lm.z - hand_landmarks[0].z) * scale  # Transform z to y
                        rel_z = -(lm.y - hand_landmarks[0].y) * scale  # Transform y to z
                        
                        # Add to wrist position
                        hand_x.append(wrist_pos[0] + rel_x)
                        hand_y.append(wrist_pos[1] + rel_y)
                        hand_z.append(wrist_pos[2] + rel_z)
                    
                    # Plot hand points with larger markers
                    ax.scatter(hand_x, hand_y, hand_z, 
                             c=color, marker='o', s=100,  
                             label=label, zorder=3)  # Higher zorder to stay on top
                    
                    # Draw hand connections with thicker lines
                    for connection in self.mp_holistic.HAND_CONNECTIONS:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        ax.plot([hand_x[start_idx], hand_x[end_idx]],
                               [hand_y[start_idx], hand_y[end_idx]],
                               [hand_z[start_idx], hand_z[end_idx]], 
                               color, linewidth=2, zorder=2)  # Thicker lines
            
            # Then draw pose skeleton
            # Plot pose points with smaller markers
            ax.scatter(x, y, z, c='#3366FF', marker='o', s=50,  # Bright blue, smaller circles
                      label='Pose', zorder=1)
            
            # Draw pose connections
            for connection in self.mp_holistic.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                ax.plot([x[start_idx], x[end_idx]],
                       [y[start_idx], y[end_idx]],
                       [z[start_idx], z[end_idx]], 
                       color='#CCCCCC', linewidth=1, zorder=0)  # Light gray, thinner lines
        
        # Set axis labels with correct orientation description
        ax.set_xlabel('X (Right/Left)')
        ax.set_ylabel('Y (Forward/Backward)')
        ax.set_zlabel('Z (Up/Down)')
        
        # Set axis limits if provided
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if zlim:
            ax.set_zlim(zlim)
        
        # Add legend
        ax.legend()
        
        # Equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        plt.tight_layout()
        return fig

def plot_video_frames_with_fps(video_path: str, duration_seconds: int = 1, frame_height: int = 50, spacing: int = 10):
    """
    Plot a video timeline with vertical lines for each frame and small images of frames below the plot.

    Args:
        video_path: Path to the video file.
        duration_seconds: Number of seconds to display on the x-axis.
        frame_height: Height of each frame image in the plot (width is scaled proportionally).
        spacing: Spacing between rows of frame images.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    central_frame = total_frames // 2
    duration_frames = int(fps * duration_seconds)
    half_duration_frames = duration_frames // 2
    start_frame = max(0, central_frame - half_duration_frames)
    end_frame = min(total_frames, central_frame + half_duration_frames)

    # Extract frames and timestamps
    frames = []
    timestamps = []
    for i in range(total_frames):
        if i < start_frame:
            continue
        if i > end_frame:
            break
        # Set the frame position and read the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        timestamps.append((i - start_frame) / fps)
    if len(frames) != duration_frames:
        print(f"Warning: Extracted {len(frames)} frames, but expected {duration_frames} frames.")
    cap.release()

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_xlim(0, duration_seconds)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frames")
    ax.set_title("Video Frames and FPS Visualization")

    # Plot vertical lines for each frame
    for t in timestamps:
        ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
    # get the plot as an image
    fig.canvas.draw()
    plot_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)

    # TODO: Fix this code
    # # Add frame images below the plot
    # frame_width = int(frame_height * frames[0].shape[1] / frames[0].shape[0])
    # max_row_width = fig.get_size_inches()[0] * fig.dpi  # Maximum width in pixels
    # current_x = 0
    # current_y = fig.get_size_inches()[1] * fig.dpi - frame_height - spacing

    # for i, (frame, t) in enumerate(zip(frames, timestamps)):
    #     if current_x + frame_width > max_row_width:
    #         current_x = 0
    #         current_y -= frame_height + spacing

    #     # Add the frame image
    #     fig.figimage(frame, xo=current_x, yo=current_y, origin='upper')
    #     current_x += frame_width + spacing

    return plot_image