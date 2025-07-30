import shutil
from pathlib import Path
from fastapi import UploadFile
import logging

logger = logging.getLogger(__name__)

async def save_uploaded_file(file: UploadFile, interim_dir: Path) -> Path:
    """
    Save an uploaded file to the interim directory.
    """
    video_path = interim_dir / file.filename
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return video_path

def cleanup_files(file_paths: list[Path]) -> None:
    """
    Clean up interim files to free disk space.
    """
    for file_path in file_paths:
        try:
            file_path.unlink()
            logger.info(f"Deleted interim file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete {file_path}: {e}")