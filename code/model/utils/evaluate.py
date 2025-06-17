import numpy as np
from typing import List, Optional, Tuple, Dict
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
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
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
        self.predictions = np.array(predictions)
        self.labels = np.array(labels)
        self.probabilities = np.array(probabilities) if probabilities is not None else None
        
        # Set number of classes
        if num_classes is None:
            self.num_classes = max(self.predictions.max(), self.labels.max()) + 1
        else:
            self.num_classes = num_classes
            
        self.class_names = class_names or [f"Class {i}" for i in range(self.num_classes)]
        
        # Compute basic metrics
        self._compute_basic_metrics()
        
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

    def plot_probability_heatmap(
        self,
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = 'YlOrRd',
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot a heatmap showing the average probability distribution for each true class.
        This helps visualize how confident the model is in its predictions, even when they're incorrect.
        
        Args:
            figsize: Figure size
            cmap: Colormap for the heatmap
            save_path: Optional path to save the plot
        """
        if self.probabilities is None:
            raise ValueError("Probabilities not available for probability heatmap")
            
        # Calculate average probabilities for each true class
        avg_probs = np.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            mask = self.labels == i
            if mask.any():
                avg_probs[i] = self.probabilities[mask].mean(axis=0)
                
        # Format the numbers: 0 and 1 with no decimals, others with 2 decimals
        formatted_probs = [[f"{x:.0f}" if np.round(x, 2) == 0.0 or np.round(x, 2) == 1.0 else f"{x:.2f}" for x in row] for row in avg_probs]
                
        plt.figure(figsize=figsize)
        sns.heatmap(
            avg_probs,
            annot=formatted_probs,
            fmt='',
            cmap=cmap,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            linewidths=0.25,
            linecolor='grey',
            annot_kws={'size': int(200/len(self.class_names))}  # Make the annotation text smaller
        )
        plt.title('Average Probability Distribution by True Class')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class Probability')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
