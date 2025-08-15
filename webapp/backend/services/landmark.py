from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
import logging
from services.model_references import draw_landmarks_on_video_with_frame


logger = logging.getLogger(__name__)

class LandmarkDrawerService:
    @staticmethod
    def draw(landmarks_path: Path, metadata_row: pd.Series, config: Config, filename: str) -> Path:
        landmarks = np.load(landmarks_path, allow_pickle=True)
        output_video_path = config.OUTPUT_DIR / filename

        draw_landmarks_on_video_with_frame(
            results_list=landmarks,
            output_path=str(output_video_path),
            fps=metadata_row['fps']
        )
        logger.info(f"Skeleton video saved to {output_video_path}")
        return output_video_path
