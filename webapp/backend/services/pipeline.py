import numpy as np
import pandas as pd
import logging
from services.model_references import FeatureProcessor

logger = logging.getLogger(__name__)
class FeaturePipeline:
    @staticmethod
    def process(
        frames: np.ndarray,
        sample_indices: list,
        metadata_row: pd.Series,
        feature_processor: FeatureProcessor
    ) -> np.ndarray:
        sample_feature_tensors = []
        for sample_idx_list in sample_indices:
            sample_features = feature_processor.process_frames(
                frames, sample_idx_list, metadata_row, 'test'
            )
            sample_feature_tensors.append(sample_features)

        logger.info("Feature processing completed")
        return sample_feature_tensors[0]  # Only first is used





