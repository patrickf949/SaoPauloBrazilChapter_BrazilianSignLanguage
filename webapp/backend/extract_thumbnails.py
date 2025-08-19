import cv2
import os
from pathlib import Path

'''
This script extracts the middle frame of each video in the sample_videos directory and saves it as a JPEG file in the thumbnails directory.

To run this script, you need to have OpenCV installed. You can install it using the following command:

pip install opencv-python

'''

# Configure paths
videos_dir = Path('sample_videos')  # relative to backend folder
output_dir = Path('../frontend/public/thumbnails')  # relative to backend folder

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Process each video
for video_path in videos_dir.glob('*.mp4'):
    print(f"Processing {video_path.name}...")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"Warning: Could not read frames from {video_path.name}")
        continue
    
    # Set position to middle frame
    middle_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    
    # Read the middle frame
    ret, frame = cap.read()
    
    if ret:
        # Save frame as JPEG
        output_path = output_dir / f"{video_path.stem}.jpg"
        cv2.imwrite(str(output_path), frame)
        print(f"Saved thumbnail to {output_path}")
    else:
        print(f"Error: Could not read middle frame from {video_path.name}")
    
    # Release video
    cap.release()

print("\nDone! All thumbnails have been extracted.")
