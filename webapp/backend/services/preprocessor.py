from pathlib import Path
import pandas as pd
import logging

from config import Config
from services.model_references import VideoAnalyzer,Preprocessor



logger = logging.getLogger(__name__)
class PreprocessorService:
    @staticmethod
    def preprocess(metadata_row: pd.Series, analyzer: VideoAnalyzer, config: Config) -> Path:
        analysis_info = analyzer.save_analysis_info()
        preprocess_params = config.PREPROCESSING_PARAMS.copy()
        preprocess_params.update({
            "start_frame": analysis_info['motion_analysis']['start_frame'],
            "end_frame": analysis_info['motion_analysis']['end_frame'],
        })

        preprocessor = Preprocessor(
            metadata_row,
            preprocess_params,
            str(config.BASE_DIR),
            config.MOTION_VERSION,
            config.POSE_VERSION,
            config.PREPROCESS_VERSION,
            verbose=False,
            save_intermediate=False,
        )
        landmarks_path = preprocessor.preprocess_landmarks()
        logger.info(f"Preprocessed landmarks saved to {landmarks_path}")
        return landmarks_path
