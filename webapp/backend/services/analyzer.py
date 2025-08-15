import logging
import pandas as pd
from config import Config
from services.model_references import VideoAnalyzer


logger = logging.getLogger(__name__)
class VideoAnalyzerService:
    @staticmethod
    def analyze(metadata_row: pd.Series, config: Config) -> VideoAnalyzer:
        analyzer = VideoAnalyzer(
            metadata_row,
            config.TIMESTAMP,
            str(config.BASE_DIR),
            verbose=False,
            motion_detection_version=config.MOTION_VERSION,
            pose_detection_version=config.POSE_VERSION
        )
        analyzer.pose_detect()
        analyzer.pose_analyze()
        analyzer.motion_detect()
        analyzer.motion_analyze()
        analyzer.save_analysis_info()
        logger.info("Video analysis completed")
        return analyzer
