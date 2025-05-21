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
import copy
import matplotlib as mpl

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

# Initialize MediaPipe drawing utilities at module level
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def draw_landmarks_on_frame(image: np.ndarray, results: Dict) -> np.ndarray:
    """
    Draw landmarks and connections on the image, adding dark grey padding if landmarks exceed [0,1] range.
    Padding is capped at 3 times the original image size in any direction.
    MediaPipe's visibility attribute will naturally handle hiding of landmarks.
    A light border is drawn around the original frame area.
    
    Args:
        image: Input image in BGR format
        results: Dictionary containing detection results
    
    Returns:
        Image with landmarks drawn, potentially with padding if landmarks exceed normal bounds
    """
    # Find the maximum extent of landmarks in each direction
    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
    
    def update_bounds(landmarks):
        nonlocal min_x, max_x, min_y, max_y
        if landmarks:
            for landmark in landmarks.landmark:
                # Skip landmarks with very low visibility
                if hasattr(landmark, 'visibility') and landmark.visibility < 0.75:
                    continue
                min_x = max(-2, min(min_x, landmark.x))  # Cap at -2 (allows for 3x padding on left)
                max_x = min(3, max(max_x, landmark.x))   # Cap at 3
                min_y = max(-2, min(min_y, landmark.y))  # Cap at -2 (allows for 3x padding on top)
                max_y = min(3, max(max_y, landmark.y))   # Cap at 3
    
    # Check all landmark types
    for key in ['face_landmarks', 'pose_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']:
        if key in results and results[key]:
            update_bounds(results[key])
    
    # If no valid landmarks were found, reset bounds to default
    if min_x == float('inf'):
        min_x, max_x, min_y, max_y = 0, 1, 0, 1
    
    # Calculate required padding
    pad_left = max(0, int(-min_x * image.shape[1]))
    pad_right = max(0, int((max_x - 1) * image.shape[1]))
    pad_top = max(0, int(-min_y * image.shape[0]))
    pad_bottom = max(0, int((max_y - 1) * image.shape[0]))
    
    # Add padding if needed
    if pad_left or pad_right or pad_top or pad_bottom:
        # Create dark grey padding (RGB: 64,64,64)
        padded_image = np.full((
            image.shape[0] + pad_top + pad_bottom,
            image.shape[1] + pad_left + pad_right,
            3
        ), 64, dtype=np.uint8)
        
        # Copy original image into padded area
        padded_image[pad_top:pad_top+image.shape[0], 
                    pad_left:pad_left+image.shape[1]] = image
        
        # Draw a light grey border (RGB: 192,192,192) around the original frame
        border_color = (192, 192, 192)
        border_thickness = 2
        cv2.rectangle(
            padded_image,
            (pad_left, pad_top),
            (pad_left + image.shape[1], pad_top + image.shape[0]),
            border_color,
            border_thickness
        )
        
        # Adjust landmark coordinates to account for padding
        scale_x = padded_image.shape[1] / image.shape[1]
        scale_y = padded_image.shape[0] / image.shape[0]
        offset_x = pad_left / padded_image.shape[1]
        offset_y = pad_top / padded_image.shape[0]
        
        def adjust_landmarks(landmarks):
            if landmarks:
                adjusted = copy.deepcopy(landmarks)
                for landmark in adjusted.landmark:
                    # Scale and shift the coordinates to the new image size
                    landmark.x = (landmark.x / scale_x) + offset_x
                    landmark.y = (landmark.y / scale_y) + offset_y
                return adjusted
            return None
        
        # Create adjusted results
        adjusted_results = {}
        for key in results:
            if key in ['face_landmarks', 'pose_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']:
                adjusted_results[key] = adjust_landmarks(results[key])
            else:
                adjusted_results[key] = results[key]
        
        # Draw landmarks on padded image
        annotated_image = padded_image.copy()
        
        # Draw face landmarks
        if adjusted_results['face_landmarks']:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=adjusted_results['face_landmarks'],
                connections=mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=adjusted_results['face_landmarks'],
                connections=mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        
        # Draw pose landmarks
        if adjusted_results['pose_landmarks']:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=adjusted_results['pose_landmarks'],
                connections=mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw left hand landmarks
        if adjusted_results['left_hand_landmarks']:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=adjusted_results['left_hand_landmarks'],
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Draw right hand landmarks
        if adjusted_results['right_hand_landmarks']:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=adjusted_results['right_hand_landmarks'],
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return annotated_image
    
    # If no padding needed, proceed with original image
    annotated_image = image.copy()
    
    # Draw face landmarks
    if results['face_landmarks']:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results['face_landmarks'],
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results['face_landmarks'],
            connections=mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )
    
    # Draw pose landmarks
    if results['pose_landmarks']:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results['pose_landmarks'],
            connections=mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    
    # Draw left hand landmarks
    if results['left_hand_landmarks']:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results['left_hand_landmarks'],
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
        )
    
    # Draw right hand landmarks
    if results['right_hand_landmarks']:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results['right_hand_landmarks'],
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
        )
    
    return annotated_image

def draw_landmarks_on_frame_no_padding(image: np.ndarray, results: Dict) -> np.ndarray:
    """
    Draw landmarks and connections on the image without padding.
    Assumes all landmarks exist in the results dictionary.
    """
    annotated_image = image.copy()
    
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results['face_landmarks'],
        connections=mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results['face_landmarks'],
        connections=mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
    )
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results['pose_landmarks'],
        connections=mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results['left_hand_landmarks'],
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
    )
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results['right_hand_landmarks'],
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
    )
    
    return annotated_image

def draw_landmarks_on_video(video_path: str,
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
            annotated_frame = draw_landmarks_on_frame(frame, results)
            
            # Write to output video
            out.write(annotated_frame)
            
            frame_idx += 1
    
    finally:
        cap.release()
        out.release()
    print(f"Landmarks drawn video saved to {output_path}")

def draw_landmarks_on_video_with_frame(results_list: List[Dict],
                                     output_path: str,
                                     base_frame: Optional[np.ndarray] = None,
                                     fps: float = 30.0) -> List[np.ndarray]:
    """
    Create a video by drawing landmarks on frames, either using a custom base frame or black frames.
    
    Args:
        results_list: List of MediaPipe results dictionaries containing landmarks
        output_path: Path where the output video will be saved
        base_frame: Optional base frame to use (if None, black frames will be used)
        fps: Frames per second for the output video (default: 30.0)
        
    Returns:
        List of annotated frames as numpy arrays
    """    
    # If base_frame is not provided, create a black frame
    if base_frame is None:
        base_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Generate all frames
    annotated_frames = []
    for results in results_list:
        frame = base_frame.copy()
        annotated_frames.append(draw_landmarks_on_frame_no_padding(frame, results))
    
    # Ensure output path ends with .mp4
    if not output_path.lower().endswith('.mp4'):
        output_path = output_path + '.mp4'
    
    # Create video with MPEG-4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, base_frame.shape[1::-1])
    
    try:
        for frame in annotated_frames:
            out.write(frame)
    finally:
        out.release()
    
    return annotated_frames

def draw_landmarks_3d(results: Dict, figsize: Tuple[int, int] = (10, 10),
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
                for connection in mp_holistic.HAND_CONNECTIONS:
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
        for connection in mp_holistic.POSE_CONNECTIONS:
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

def generate_paired_sample_data(series_count=50, min_length=50, max_length=500, 
                              correlation=0.7, seed=42):
    """
    Generate sample data for paired series (left and right channels).
    
    Parameters:
    - series_count: Number of series pairs to generate
    - min_length: Minimum length of each series
    - max_length: Maximum length of each series
    - correlation: How correlated the errors should be between left and right channels (0-1)
    - seed: Random seed for reproducibility
    
    Returns:
    - Dictionary with series data
    """
    import numpy as np
    
    np.random.seed(seed)
    series_dict = {}

    for i in range(series_count):
        # Generate random length between min and max - same for both channels
        length = np.random.randint(min_length, max_length)
        
        # Generate left channel
        left_series = np.zeros(length)
        
        # Add isolated errors (1% chance)
        error_mask_left = np.random.random(length) < 0.01
        left_series[error_mask_left] = 1
        
        # Add some error sequences in random locations for left channel
        if np.random.random() < 0.3:  # 30% of series have error sequences
            seq_start = np.random.randint(0, length - 10)
            seq_length = np.random.randint(3, 10)
            left_series[seq_start:seq_start+seq_length] = 1
        
        # Add some periodic errors in some series for left channel
        if i % 7 == 0:  # Every 7th series has periodic errors
            period = np.random.randint(20, 50)
            left_series[::period] = 1
        
        # Generate right channel with correlation to left
        right_series = np.zeros(length)
        
        # Copy some errors from left channel based on correlation
        for j in range(length):
            if left_series[j] == 1 and np.random.random() < correlation:
                right_series[j] = 1
        
        # Add some unique errors to right channel
        unique_error_mask = np.logical_and(left_series == 0, np.random.random(length) < 0.01)
        right_series[unique_error_mask] = 1
        
        # Add some unique streaks to right channel
        if np.random.random() < 0.2:  # 20% chance of unique streak in right
            seq_start = np.random.randint(0, length - 8)
            seq_length = np.random.randint(3, 8)
            right_series[seq_start:seq_start+seq_length] = 1
        
        # Store both channels in the dictionary
        series_dict[f"Series_{i}_Left"] = left_series
        series_dict[f"Series_{i}_Right"] = right_series
    
    return series_dict

def series_none_frame_visualization(series_dict, max_display_pairs=None, figsize=(18, 12), line_width=8,
                                     background_alpha=0.1, streak_length_cap=10, bar_spacing=0.0, cbar_spacing=0.25,
                                     left_cmap=plt.cm.Blues, right_cmap=plt.cm.Reds, sort_length=False):
    """
    Visualize paired series (left/right channels) with streaks colored by length (capped at 10).
    Left and right channels use truncated colormaps (Blues, Reds) for better visibility of short streaks.
    Color intensity increases with streak length. Separate colorbars are shown for each channel.
    The space between the main plot and error bar, and between the error bar and each colorbar, is adjustable.
    Colormaps for left and right channels can be specified.
    
    Args:
        series_dict: Dictionary containing the series data
        max_display_pairs: Maximum number of pairs to display
        figsize: Figure size
        line_width: Width of the lines
        background_alpha: Alpha value for background shading
        streak_length_cap: Maximum streak length for color mapping
        bar_spacing: Spacing between main plot and error bar
        cbar_spacing: Spacing for colorbars
        left_cmap: Colormap for left channel
        right_cmap: Colormap for right channel
        sort_length: If True, sort pairs by their series length
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
            cmap(np.linspace(minval, maxval, n))
        )
        return new_cmap
    
    # Get all series keys and separate into left and right channels
    all_keys = list(series_dict.keys())
    
    # Group keys into pairs based on series number
    series_pairs = {}
    for key in all_keys:
        if "_Left" in key:
            series_num = key.split("_Left")[0]
            channel = "Left"
        elif "_Right" in key:
            series_num = key.split("_Right")[0]
            channel = "Right"
        else:
            continue
        if series_num not in series_pairs:
            series_pairs[series_num] = {}
        series_pairs[series_num][channel] = key

    # Calculate series length for each pair if sorting is requested
    if sort_length:
        pair_lengths = {}
        for pair_key, channels in series_pairs.items():
            # Get the length of the series (both left and right have same length)
            series_length = len(series_dict[channels["Left"]])
            pair_lengths[pair_key] = series_length
        
        # Sort pairs by series length
        sorted_pair_keys = sorted(series_pairs.keys(), key=lambda x: pair_lengths[x], reverse=True)
    else:
        sorted_pair_keys = list(series_pairs.keys())
    
    # Limit the number of pairs to display if specified
    if max_display_pairs is not None and max_display_pairs < len(sorted_pair_keys):
        sorted_pair_keys = sorted_pair_keys[:max_display_pairs]
    
    # Flatten the pairs into a display order (left then right for each pair)
    display_keys = []
    for pair_key in sorted_pair_keys:
        if "Left" in series_pairs[pair_key]:
            display_keys.append(series_pairs[pair_key]["Left"])
        if "Right" in series_pairs[pair_key]:
            display_keys.append(series_pairs[pair_key]["Right"])
    
    # Create the figure with a grid layout for the main plot, bar spacer, error bar, cbar spacers, and colorbars
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 7, width_ratios=[4, bar_spacing, 1, cbar_spacing, 0.1, cbar_spacing, 0.1],
                          left=0.1, right=0.98, bottom=0.1, top=0.9,
                          wspace=0.0)
    main_ax = fig.add_subplot(gs[0, 0])
    # gs[0, 1] is the spacer between main_ax and right_ax
    right_ax = fig.add_subplot(gs[0, 2], sharey=main_ax)
    # gs[0, 3] and gs[0, 5] are spacers for colorbars
    cbar_ax_left = fig.add_subplot(gs[0, 4])
    cbar_ax_right = fig.add_subplot(gs[0, 6])
    
    # Prepare truncated colormaps
    left_base_color = left_cmap(0.2)
    right_base_color = right_cmap(0.1)
    left_cmap_trunc = truncate_colormap(left_cmap, 0.5, 1.0)
    right_cmap_trunc = truncate_colormap(right_cmap, 0.5, 1.0)
    norm = mpl.colors.Normalize(vmin=1, vmax=streak_length_cap)
    
    # Track maximum time index for x-axis limits
    max_time = 0
    error_counts = np.zeros(len(display_keys))
    
    # For colorbar legend
    streak_lengths = np.arange(1, streak_length_cap+1)
    
    for i, series_name in enumerate(display_keys):
        series = series_dict[series_name]
        max_time = max(max_time, len(series))
        error_positions = np.where(series == 1)[0]
        pair_idx = i // 2
        if pair_idx % 2 == 0:
            if i % 2 == 0:
                main_ax.axhspan(i-0.4, i+1.4, color='gray', alpha=background_alpha)
        pair_shift = 0.125
        if "_Right" in series_name:
            pair_shift *= -1
        error_counts[i] = len(error_positions)/len(series) * 100
        main_ax.plot(range(len(series)), [i+pair_shift]*len(series), color=left_base_color if "_Left" in series_name else right_base_color, linewidth=line_width, solid_capstyle='butt',)
        if len(error_positions) == 0:
            continue
        # plot line for full series with base_color
        # Group consecutive positions into streaks
        streaks = []
        current_streak = [error_positions[0]]
        for j in range(1, len(error_positions)):
            if error_positions[j] == error_positions[j-1] + 1:
                current_streak.append(error_positions[j])
            else:
                streaks.append(current_streak)
                current_streak = [error_positions[j]]
        streaks.append(current_streak)
        is_left_channel = "_Left" in series_name
        for streak in streaks:
            streak_len = min(len(streak), streak_length_cap)
            color = left_cmap_trunc(norm(streak_len)) if is_left_channel else right_cmap_trunc(norm(streak_len))
            main_ax.plot(streak, [i+pair_shift]*len(streak), color=color, linewidth=line_width, solid_capstyle='butt',)
    # Y labels
    series_ids = []
    for key in display_keys:
        if "_Left" in key:
            series_ids.append(f"{key.split('_Left')[0]}")
    main_ax.set_yticks(np.arange(0.5, len(display_keys)+0.5, 2))
    main_ax.set_yticklabels(series_ids)
    main_ax.set_xlabel('Frame Number')
    main_ax.set_ylabel('Filename')
    main_ax.set_title('Visualization of Frames with None Landmarks')
    main_ax.set_xlim(-1, max_time + 0)
    main_ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    # main_ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    # Right margin: error counts
    grey_color = (0.65,0.65,0.65)
    white_color = (1,1,1)
    bar_colors = [grey_color, grey_color, white_color, white_color] * (len(display_keys) //4) + [white_color, white_color] * int(len(display_keys)%4/2)
    barh_positions = []
    for barh_position in range(len(display_keys)):
        if barh_position % 2 == 1:
            barh_positions.append(barh_position+pair_shift)
        else:
            barh_positions.append(barh_position-pair_shift)
    barh = right_ax.barh(barh_positions, error_counts, color=bar_colors, edgecolor='black', alpha=0.5, height=.5)
    # right_ax.set_yticks([])
    right_ax.set_xlabel('% of Frame with None Landmarks')
    # right_ax.spines['top'].set_visible(False)
    right_ax.spines['right'].set_visible(False)
    right_ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    # right_ax.set_xlim(0, 100)
    plt.setp(right_ax.get_yticklabels(), visible=False)
    # Separate colorbars for left and right
    left_cbar = mpl.colorbar.ColorbarBase(cbar_ax_left, cmap=left_cmap_trunc, norm=norm, orientation='vertical', label='None Frame Streak Length (Left Hand Landmarks)')
    right_cbar = mpl.colorbar.ColorbarBase(cbar_ax_right, cmap=right_cmap_trunc, norm=norm, orientation='vertical', label='None Frame Streak Length (Right Hand Landmarks)')
    cbar_ax_left.yaxis.set_ticks_position('left')
    cbar_ax_right.yaxis.set_ticks_position('right')
    plt.tight_layout()
    plt.close('all')
    return fig