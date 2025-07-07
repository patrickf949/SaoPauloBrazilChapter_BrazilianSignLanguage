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
from PIL import Image
from matplotlib import colors

"""
This file contains functions for creating various visualizations for the report.
"""

color_dict = {
    'ne': 'mediumorchid',       # i
    'sb': 'dodgerblue',     # s
    'uf': 'mediumseagreen',     # u
    'vl': 'darkorange',      # v
}
color_list = ['mediumorchid', 'dodgerblue', 'mediumseagreen', 'darkorange', 'darkorange', 'darkorange']
color_list_rgb = [colors.to_rgb(color) for color in color_list]
color_list_rgb_int = [(int(color[0]*255), int(color[1]*255), int(color[2]*255)) for color in color_list_rgb]
data_source_list = ['INES', 'SignBank', 'UFV', 'V-Librasil', 'V-Librasil', 'V-Librasil']

def all_signs_for_word(frames, word):
    """
    Concatenate all frames for a word, adding a border and a text label.
    """
    font = cv2.FONT_HERSHEY_TRIPLEX

    # find the largest frame
    max_area_index = np.argmax([frame.shape[0] * frame.shape[1] for frame in frames])
    max_area_frame = frames[max_area_index]
    target_height = max_area_frame.shape[0]
    target_width = max_area_frame.shape[1]
    target_aspect_ratio = target_width / target_height
    # pad frames to the target aspect ratio
    final_frames = []
    for i, frame in enumerate(frames):
        aspect_ratio = frame.shape[1] / frame.shape[0]

        if aspect_ratio < target_aspect_ratio:
            padded_width = int(frame.shape[0] * target_aspect_ratio)
            padding_width = (padded_width - frame.shape[1]) // 2
            padded_frame = cv2.copyMakeBorder(frame, 0, 0, padding_width, padding_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            padded_frame = frame.copy()

        resized_frame = cv2.resize(padded_frame, (target_width, target_height))
        color = color_list_rgb_int[i]
        bordered_frame = cv2.copyMakeBorder(resized_frame, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=color)
        
        annotated_frame = cv2.copyMakeBorder(bordered_frame, 0, 350, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        data_source = data_source_list[i]
        font_size = 6
        font_thickness = 5
        textsize = cv2.getTextSize(data_source, font, font_size, font_thickness)[0]
        text_x = (target_width - textsize[0]) // 2
        annotated_frame = cv2.putText(annotated_frame, data_source, (text_x, 1420), font, font_size, (0, 0, 0), font_thickness)

        final_frames.append(annotated_frame)

        if i < len(frames) - 1:
            gap = np.full((target_height + 450, 250, 3), 255, dtype=np.uint8)
            final_frames.append(gap)

    # concatenate the frames
    new_frame = np.concatenate(final_frames, axis=1)
    new_frame = cv2.copyMakeBorder(new_frame, 600, 0, 300, 300, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    font_size = 10
    font_thickness = 15
    textsize = cv2.getTextSize(word, font, font_size, font_thickness)[0]
    text_x = (new_frame.shape[1] - textsize[0]) // 2
    new_frame = cv2.putText(new_frame, word, (text_x, 400), font, font_size, (0, 0, 0), font_thickness)

    return new_frame


def video_to_gif(video_path: str, 
                output_path: str,
                fps: int = 10,
                max_width: Optional[int] = None,
                max_height: Optional[int] = None,
                start_time: float = None,
                end_time: float = None,
                quality: int = 85,
                optimize: bool = True,
                loop: int = 0,
                final_frame_duration: int = 1000) -> str:
    """
    Convert a video file to GIF format using OpenCV and PIL.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save the output GIF file
        fps: Frames per second for the output GIF (default: 10)
        max_width: Maximum width of the GIF in pixels (if None, maintains original width)
        max_height: Maximum height of the GIF in pixels (if None, maintains original height)
        start_time: Start time in seconds (if None, starts from beginning)
        end_time: End time in seconds (if None, goes to end)
        quality: GIF quality from 1-100 (default: 85)
        optimize: Whether to optimize the GIF (default: True)
        loop: Number of loops (0 = infinite, 1 = no loop, etc.)
        final_frame_duration: How long to display the final frame in milliseconds (default: 1000)
        
    Returns:
        Path to the created GIF file
        
    Raises:
        ValueError: If video file cannot be opened
        RuntimeError: If no frames are extracted
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frame range
    start_frame = 0
    end_frame = total_frames
    
    if start_time is not None:
        start_frame = int(start_time * video_fps)
    if end_time is not None:
        end_frame = int(end_time * video_fps)
    
    # Ensure valid frame range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))
    
    # Calculate frame step to achieve desired fps
    frame_step = max(1, int(video_fps / fps))
    
    # Calculate output dimensions
    output_width = width
    output_height = height
    
    # Scale if max dimensions are provided
    if max_width is not None or max_height is not None:
        aspect_ratio = width / height
        if max_width is not None and max_height is not None:
            # Scale to fit within both constraints
            scale_w = max_width / width
            scale_h = max_height / height
            scale = min(scale_w, scale_h)
            output_width = int(width * scale)
            output_height = int(height * scale)
        elif max_width is not None:
            # Scale by width
            output_width = max_width
            output_height = int(max_width / aspect_ratio)
        else:  # max_height is not None
            # Scale by height
            output_height = max_height
            output_width = int(max_height * aspect_ratio)
    
    # Extract frames
    frames = []
    durations = []  # List to store duration for each frame
    frame_indices = range(start_frame, end_frame, frame_step)
    base_duration = int(1000 / fps)  # Duration in milliseconds
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame if dimensions changed
        if output_width != width or output_height != height:
            frame_resized = cv2.resize(frame_rgb, (output_width, output_height), 
                                     interpolation=cv2.INTER_LANCZOS4)
        else:
            frame_resized = frame_rgb
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_resized)
        frames.append(pil_image)
        durations.append(base_duration)
    
    cap.release()
    
    if not frames:
        raise RuntimeError("No frames were extracted from the video")
    
    # Add final frame duration
    durations[-1] = final_frame_duration
    
    # If output_path has a directory component, ensure it exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save with optimization
    if optimize:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=loop,
            optimize=True,
            quality=quality
        )
    else:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=loop,
            quality=quality
        )
    
    print(f"GIF created successfully: {output_path}")
    print(f"  - Frames: {len(frames)}")
    print(f"  - Duration: {(sum(durations) / 1000):.2f} seconds")
    print(f"  - Size: {output_width}x{output_height}")
    print(f"  - File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    return output_path

def apply_to_videos(combine_function, video_paths: List[str], save_path: str, 
                   fps: float = None, codec: str = 'avc1', 
                   start_time: float = None, end_time: float = None,
                   max_dimension: int = 1920,
                   verbose: bool = True) -> str:
    """
    Apply a combining function to multiple videos frame by frame and save the result.
    For videos shorter than the longest video, their final frame will be frozen until the end.
    
    Args:
        combine_function: Function that takes a list of frames (one from each video) 
                         and returns a single combined frame
        video_paths: List of paths to input video files
        save_path: Path to save the output video
        fps: Output video FPS (if None, uses the FPS of the first video)
        codec: Video codec to use (default: 'avc1')
        start_time: Start time in seconds (if None, starts from beginning)
        end_time: End time in seconds (if None, goes to end of longest video)
        max_dimension: Maximum dimension (width or height) for the output video
        verbose: Whether to print progress information
        
    Returns:
        Path to the saved video
        
    Raises:
        ValueError: If video files cannot be opened or have incompatible properties
        RuntimeError: If no frames are processed
    """
    if not video_paths:
        raise ValueError("At least one video path must be provided")
    
    # Open all video captures
    caps = []
    video_properties = []
    
    for i, video_path in enumerate(video_paths):
        # Check if it's a PNG image
        if video_path.lower().endswith('.png'):
            # Load PNG image
            image = cv2.imread(video_path)
            if image is None:
                raise ValueError(f"Could not load PNG image: {video_path}")
            
            # Get image properties
            height, width = image.shape[:2]
            
            # For images, we'll use the output FPS and treat them as static frames
            # We'll calculate duration based on the longest video later
            video_properties.append({
                'fps': None,  # Will be set to output FPS
                'total_frames': None,  # Will be calculated based on duration
                'width': width,
                'height': height,
                'duration': None,  # Will be set to match longest video
                'is_image': True,
                'image_data': image
            })
            
            caps.append(None)  # No video capture for images
            
            if verbose:
                print(f"Image {i+1}: {video_path}")
                print(f"  - Type: PNG Image")
                print(f"  - Size: {width}x{height}")
                print(f"  - Duration: Will match longest video")
        else:
            # Handle video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if video_fps <= 0:
                video_fps = 30.0  # Default FPS if not available
            
            video_properties.append({
                'fps': video_fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'duration': total_frames / video_fps,
                'is_image': False
            })
            
            caps.append(cap)
            
            if verbose:
                print(f"Video {i+1}: {video_path}")
                print(f"  - FPS: {video_fps:.2f}")
                print(f"  - Frames: {total_frames}")
                print(f"  - Size: {width}x{height}")
                print(f"  - Duration: {total_frames/video_fps:.2f}s")
    
    # Determine output FPS
    if fps is None:
        # Use the highest FPS among all videos to preserve maximum temporal resolution
        # Filter out images (which have fps=None)
        video_fps_values = [props['fps'] for props in video_properties if not props.get('is_image', False)]
        if video_fps_values:
            fps = max(video_fps_values)
        else:
            fps = 30.0  # Default if only images are provided
        if verbose:
            print(f"Using highest FPS ({fps:.2f}) from input videos as output FPS")
    
    # Update image properties to use output FPS
    for props in video_properties:
        if props.get('is_image', False):
            props['fps'] = fps
            # Duration will be set based on longest video
    
    # Determine frame range
    if start_time is not None:
        start_frame = int(start_time * fps)
    else:
        start_frame = 0
    
    # Find the longest video duration
    # Filter out images for duration calculation
    video_durations = [props['duration'] for props in video_properties if not props.get('is_image', False)]
    if video_durations:
        max_duration = max(video_durations)
    else:
        max_duration = 5.0  # Default duration if only images are provided
    
    if end_time is not None:
        end_time = min(end_time, max_duration)
    else:
        end_time = max_duration
    
    # Update image durations to match the output duration
    for props in video_properties:
        if props.get('is_image', False):
            props['duration'] = end_time
            props['total_frames'] = int(end_time * fps)
    
    end_frame = int(end_time * fps)
    
    if verbose:
        print(f"\nProcessing frames {start_frame} to {end_frame} at {fps} FPS")
        print(f"Output duration: {(end_frame - start_frame) / fps:.2f}s")
    
    # Convert save_path to absolute path and ensure output directory exists
    save_path = os.path.abspath(save_path)
    save_dir = os.path.dirname(save_path)
    if save_dir:  # Only create directory if there is a path component
        os.makedirs(save_dir, exist_ok=True)
    
    # Process first frame to get output dimensions
    frames = []
    for i, cap in enumerate(caps):
        if video_properties[i].get('is_image', False):
            frame = video_properties[i]['image_data'].copy()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((video_properties[i]['height'], 
                                video_properties[i]['width'], 3), dtype=np.uint8)
        frames.append(frame)
    
    # Get combined frame dimensions
    combined_frame = combine_function(frames)
    height, width = combined_frame.shape[:2]
    
    # Scale down if necessary
    scale = 1.0
    if width > max_dimension or height > max_dimension:
        scale = min(max_dimension / width, max_dimension / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        if verbose:
            print(f"Scaling output from {width}x{height} to {new_width}x{new_height}")
        width, height = new_width, new_height
        combined_frame = cv2.resize(combined_frame, (width, height))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    if not out.isOpened():
        # Try alternative codecs
        alternative_codecs = ['mp4v', 'XVID', 'MJPG']
        for alt_codec in alternative_codecs:
            if verbose:
                print(f"Trying alternative codec: {alt_codec}")
            fourcc = cv2.VideoWriter_fourcc(*alt_codec)
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            if out.isOpened():
                break
        if not out.isOpened():
            raise RuntimeError(f"Could not initialize video writer for {save_path}")
    
    if verbose:
        print(f"Output video size: {width}x{height}")
    
    # Write first frame
    out.write(combined_frame)
    frame_count = 1
    
    # Keep track of last valid frames for each video
    last_valid_frames = frames.copy()
    
    # Process remaining frames
    for frame_idx in range(start_frame + 1, end_frame):
        # Read frames from all videos and images
        frames = []
        for i, cap in enumerate(caps):
            if video_properties[i].get('is_image', False):
                # For images, always return the same image data
                frame = video_properties[i]['image_data'].copy()
            else:
                # Handle video frames
                # Calculate the frame index for this video based on its FPS
                video_frame_idx = int(frame_idx * video_properties[i]['fps'] / fps)
                
                # If we've reached the end of this video, use its last valid frame
                if video_frame_idx >= video_properties[i]['total_frames']:
                    frame = last_valid_frames[i].copy()
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        last_valid_frames[i] = frame.copy()  # Update last valid frame
                    else:
                        # If we can't read the frame, use the last valid frame
                        frame = last_valid_frames[i].copy()
            
            frames.append(frame)
        
        # Apply the combining function and scale if necessary
        combined_frame = combine_function(frames)
        if scale != 1.0:
            combined_frame = cv2.resize(combined_frame, (width, height))
        
        # Write the combined frame
        out.write(combined_frame)
        frame_count += 1
        
        # Print progress
        if verbose and frame_count % 30 == 0:  # Every 30 frames
            progress = (frame_idx - start_frame + 1) / (end_frame - start_frame) * 100
            print(f"Progress: {progress:.1f}% ({frame_count} frames processed)")
    
    # Clean up
    for cap in caps:
        if cap is not None:  # Only release video captures, not images
            cap.release()
    if out is not None:
        out.release()
    
    if verbose:
        print(f"\nVideo saved successfully: {save_path}")
        print(f"Total frames processed: {frame_count}")
        print(f"Final duration: {frame_count / fps:.2f}s")
        print(f"File size: {os.path.getsize(save_path) / (1024*1024):.1f} MB")
    
    return save_path

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