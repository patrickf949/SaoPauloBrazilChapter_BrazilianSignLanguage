from pydantic import BaseModel
from typing import Dict

class Prediction(BaseModel):
    class_id: int
    label: str

class PredictionResponse(BaseModel):
    prediction: Prediction
    probabilities: Dict[str, float]
    output_files: Dict[str, str]