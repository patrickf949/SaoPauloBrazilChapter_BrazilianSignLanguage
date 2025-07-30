import cv2
from pathlib import Path
from fastapi import UploadFile
from typing import Dict, Optional, Union
import logging
import tempfile
import shutil

logger = logging.getLogger(__name__)

async def get_video_metadata(video_input: Union[str, UploadFile]) -> Optional[Dict[str, any]]:
    """
    Extract metadata from a video file, either from a file path or an uploaded file.

    Args:
        video_input (Union[str, UploadFile]): Either a path to the video file or a FastAPI UploadFile object.

    Returns:
        Dict[str, any]: Dictionary containing video metadata (filename, fps, frame_count, duration, width, height).
        None: If metadata extraction fails.
    """
    temp_file = None
    try:
        # Handle input type
        if isinstance(video_input, str):
            video_path = Path(video_input)
            filename = video_path.name
        elif isinstance(video_input, UploadFile):
            # Validate file extension
            if not video_input.filename.lower().endswith('.mp4'):
                logger.error(f"Invalid file extension for {video_input.filename}: Only .mp4 supported")
                return None
            filename = video_input.filename
            # Create a temporary file to store the uploaded video
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            shutil.copyfileobj(video_input.file, temp_file)
            temp_file.close()
            video_path = Path(temp_file.name)
        else:
            logger.error(f"Invalid input type: {type(video_input)}")
            return None

        # Validate file existence
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None

        # Open video with OpenCV
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None

        # Extract metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        metadata = {
            "filename": filename,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "width": width,
            "height": height
        }

        # Release video capture
        cap.release()
        logger.info(f"Extracted metadata for {filename}: {metadata}")

        return metadata

    except Exception as e:
        logger.error(f"Error extracting metadata for {filename if 'filename' in locals() else 'unknown'}: {str(e)}")
        return None
    finally:
        # Clean up temporary file if created
        if temp_file is not None:
            try:
                Path(temp_file.name).unlink()
                logger.info(f"Cleaned up temporary file: {temp_file.name}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_file.name}: {str(e)}")