import logging
from pathlib import Path
import numpy as np
import json
from services.model_references import InferenceEngine


logger = logging.getLogger(__name__)
class PredictionService:
    @staticmethod
    def predict_and_format(
        input_sample: np.ndarray,
        inference_engine: InferenceEngine,
        label_encoding_path: Path
    ) -> tuple[int, np.ndarray, dict]:
        prediction, probs = inference_engine.predict(
            inputs=input_sample,
            return_full_probs=True
        )
        logger.info(f"Prediction: {prediction}, Probabilities: {probs.tolist()}")

        with open(label_encoding_path, "r") as f:
            label_encoding = json.load(f)

        return int(prediction), probs, label_encoding
