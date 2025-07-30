import torch
import numpy as np
from typing import Any, Optional, Tuple, Union

class InferenceEngine:
    def __init__(self, model, device: str, ensemble_strategy: Optional[str] = None):
        self.model = model
        self.device = device
        self.ensemble_strategy = ensemble_strategy

    def predict(self, inputs: Union[torch.Tensor, np.ndarray], return_full_probs: bool = False) -> Union[int, Tuple[int, np.ndarray]]:
        """
        Make a prediction using the model.

        Args:
            inputs: Input features as a PyTorch tensor or NumPy array.
            return_full_probs: If True, return prediction and probabilities.

        Returns:
            Prediction index or tuple of (prediction, probabilities).
        """
        # Convert NumPy array to PyTorch tensor if needed
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        
        inputs = inputs.to(self.device)
        self.model.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            probs = torch.softmax(outputs, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
        
        if return_full_probs:
            return prediction, probs.cpu().numpy()
        return prediction