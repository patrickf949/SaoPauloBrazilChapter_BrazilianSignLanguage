import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import torch

class EvaluationMetrics:
    """
    A class to compute various evaluation metrics from model predictions and ground truth labels.
    Works with outputs from InferenceEngine.predict() method.
    """
    def __init__(
        self,
        predictions: Union[np.ndarray, Dict[int, Union[float, np.ndarray]]],
        labels: Union[np.ndarray, Dict[int, Union[float, np.ndarray]]],
        probabilities: Optional[Union[np.ndarray, Dict[int, np.ndarray]]] = None,
        num_classes: Optional[int] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize with predictions and labels from InferenceEngine.
        
        Args:
            predictions: Model predictions (class indices)
            labels: Ground truth labels
            probabilities: Optional probability distributions for each prediction
            num_classes: Number of classes in the dataset
            class_names: Optional list of class names for detailed reporting
        """
        # Convert to numpy for easier computation
        self.predictions = self._to_numpy(predictions)
        self.labels = self._to_numpy(labels)
        self.probabilities = self._to_numpy(probabilities) if probabilities is not None else None
        
        # Set number of classes
        if num_classes is None:
            self.num_classes = max(self.predictions.max(), self.labels.max()) + 1
        else:
            self.num_classes = num_classes
            
        self.class_names = class_names or [f"Class {i}" for i in range(self.num_classes)]
        
        # Compute basic metrics
        self._compute_basic_metrics()
        
    def _to_numpy(self, data: Union[np.ndarray, Dict[int, Union[float, np.ndarray]]]) -> np.ndarray:
        """Convert dict of values to numpy array if needed."""
        if isinstance(data, dict):
            # For ensemble predictions, convert dict to array
            return np.array([v for v in data.values()])
        return np.array(data)
        
    def _compute_basic_metrics(self):
        """Compute basic metrics that are always available."""
        self.accuracy = (self.predictions == self.labels).mean()
        self.confusion_mat = confusion_matrix(self.labels, self.predictions)
        
        # Per-class metrics
        self.class_metrics = {}
        for i in range(self.num_classes):
            tp = ((self.predictions == i) & (self.labels == i)).sum()
            fp = ((self.predictions == i) & (self.labels != i)).sum()
            fn = ((self.predictions != i) & (self.labels == i)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            self.class_metrics[i] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': (self.labels == i).sum()
            }
            
    def get_accuracy(self) -> float:
        """Get overall accuracy."""
        return self.accuracy
        
    def get_topk_accuracy(self, k: int = 5) -> float:
        """
        Get top-k accuracy if probabilities are available.
        
        Args:
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy
        """
        if self.probabilities is None:
            raise ValueError("Probabilities not available for top-k accuracy")
            
        # Get top k predictions
        topk_preds = np.argsort(self.probabilities, axis=1)[:, -k:]
        # Check if true label is in top k
        correct = np.array([label in preds for label, preds in zip(self.labels, topk_preds)])
        return correct.mean()
        
    def get_class_metrics(self) -> Dict[int, Dict[str, float]]:
        """Get per-class metrics (precision, recall, F1)."""
        return self.class_metrics
        
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return self.confusion_mat
        
    def plot_confusion_matrix(
        self,
        normalize: bool = False,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'Blues',
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            normalize: Whether to normalize the confusion matrix
            figsize: Figure size
            cmap: Colormap
            save_path: Optional path to save the plot
        """
        cm = self.confusion_mat
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap=cmap,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            linewidths=0.25,  # Add lines between cells
            linecolor='grey'  # Make the lines black for better visibility
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        return classification_report(
            self.labels,
            self.predictions,
            target_names=self.class_names,
            digits=4
        )
        
    def get_loss(self, loss_fn: torch.nn.Module) -> float:
        """
        Compute loss value if probabilities are available.
        
        Args:
            loss_fn: Loss function to use
            
        Returns:
            Average loss value
        """
        if self.probabilities is None:
            raise ValueError("Probabilities not available for loss computation")
            
        # Convert to tensors for loss computation
        probs = torch.tensor(self.probabilities)
        labels = torch.tensor(self.labels)
        
        # Compute loss
        loss = loss_fn(probs, labels)
        return loss.item()
        
    def get_error_analysis(self) -> Dict[str, List[int]]:
        """
        Get indices of misclassified samples for error analysis.
        
        Returns:
            Dictionary mapping error types to sample indices
        """
        errors = defaultdict(list)
        
        for i, (pred, label) in enumerate(zip(self.predictions, self.labels)):
            if pred != label:
                errors[f"{self.class_names[label]}->{self.class_names[pred]}"].append(i)
                
        return dict(errors)
