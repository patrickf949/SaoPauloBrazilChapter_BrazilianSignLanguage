import cloudinary.uploader
from pathlib import Path
import logging
import numpy as np

from fastapi import HTTPException

from services.converter import VideoConversionService

logger = logging.getLogger(__name__)

class ResultsSaver:
    @staticmethod
    def save_txt(
        prediction: int,
        probs: np.ndarray,
        label_encoding: dict,
        filename: str,
        output_dir: Path
    ) -> dict:
        probs_dict = {label_encoding[str(i)]: float(p) for i, p in enumerate(probs)}
        probs_str = ", ".join(f"{label}: {prob:.3f}" for label, prob in probs_dict.items())

        # Save locally (optional, maybe for debugging)
        output_txt_path = output_dir / filename.replace(".mp4", ".txt")
        with open(output_txt_path, "w") as f:
            f.write(f"prediction: \n\tclass {prediction}: {label_encoding[str(prediction)]}\n")
            f.write(f"probabilities: {probs_str}\n")
        logger.info(f"Results saved to {output_txt_path}")

        return {
            "txt_path": str(output_txt_path),
            "probs_dict": probs_dict
        }

    @staticmethod
    def upload_video_to_cloudinary(video_file: str, public_id: str = None) -> str:
        try:
            VideoConversionService.convert_to_mp4(video_file)
            video_path = Path(video_file)
            response = cloudinary.uploader.upload_large(
                str(video_file),
                resource_type="video",
                public_id=public_id or video_path.stem,
                chunk_size=6000000  # Optional: for large uploads
            )
            url = response.get("secure_url")
            logger.info(f"Uploaded video to Cloudinary: {url}")
            
            return url
        except Exception as e:
            logger.error(f"Cloudinary upload failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to upload video to Cloudinary")
