from moviepy import VideoFileClip
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VideoConversionService:
    @staticmethod
    def convert_to_mp4(input_path: str) -> Path:
        logger.info(f"Converting {input_path} to MP4")
        try:
            clip = VideoFileClip(str(input_path))
            clip.write_videofile(
                str(input_path),
                codec="libx264",
                audio_codec="aac",
                logger=None
            )
            clip.close()
            return input_path
        except Exception as e:
            logger.error(f"Failed to convert video: {e}")
            raise
