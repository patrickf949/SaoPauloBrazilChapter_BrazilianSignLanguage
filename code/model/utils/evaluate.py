import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import pandas as pd
from inference import InferenceEngine


def evaluate(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cuda",
    top_k: Optional[int] = None,
    criterion=None,
    return_logits: bool = False,
) -> Union[
    float,  # Just accuracy
    Tuple[float, float],  # acc, loss
    Tuple[float, float, float],  # acc, topk, loss
    Tuple[torch.Tensor, torch.Tensor]  # all_logits, all_labels
]:
    """
    Flexible evaluation function supporting basic metrics and raw logits.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        top_k: Optional number of top predictions to consider
        criterion: Optional loss criterion
        return_logits: Whether to return raw logits and labels
        
    Returns:
        Different return types based on parameters:
        - Basic accuracy if no other parameters
        - (accuracy, loss) if criterion provided
        - (accuracy, topk_accuracy, loss) if top_k provided
        - (all_logits, all_labels) if return_logits=True
    """
    engine = InferenceEngine(model=model, device=device)
    
    all_logits = []
    all_labels = []
    correct_topk = 0
    total = 0
    test_loss = 0.0

    for batch in test_loader:
        features, labels = batch[:2]  # Get first two elements (features, labels)
        y = labels.squeeze(0)
        all_labels.append(y)
        
        # Get predictions
        logits = engine.predict(features, return_logits=True)
        all_logits.append(logits)
        
        # Calculate loss if criterion provided
        if criterion is not None:
            loss = criterion(logits, y.to(device))
            test_loss += loss.item()
            
        # Calculate top-k accuracy if requested
        if top_k is not None:
            topk_preds = torch.topk(logits, k=top_k, dim=1).indices
            correct_topk += sum([y.item() in topk_preds[i] for i in range(topk_preds.size(0))])
            total += logits.size(0)

    # Stack all results
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    if return_logits:
        return all_logits, all_labels
        
    # Calculate metrics
    predictions = torch.argmax(all_logits, dim=1)
    accuracy = accuracy_score(all_labels.cpu(), predictions.cpu())
    
    if criterion is not None:
        avg_loss = test_loss / len(test_loader)
        if top_k is not None:
            acc_topk = correct_topk / total if total > 0 else 0
            return accuracy, acc_topk, avg_loss
        return accuracy, avg_loss
        
    if top_k is not None:
        acc_topk = correct_topk / total if total > 0 else 0
        return accuracy, acc_topk
        
    return accuracy


def evaluate_detailed(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: str = "cuda",
    class_names: Optional[Dict[int, str]] = None,
    save_confusion_matrix: Optional[str] = None,
    ensemble_strategy: Optional[str] = None,
    return_confidence: bool = False,
    criterion = None,
    num_workers: int = 0,
    batch_size: int = 32,
) -> Union[
    Tuple[Dict[str, float], pd.DataFrame],
    Tuple[Dict[str, float], pd.DataFrame, Dict[int, float]]
]:
    """
    Detailed evaluation with optional ensemble prediction and confidence scores.
    
    Args:
        model: The model to evaluate
        dataset: Dataset to evaluate on
        device: Device to run evaluation on
        class_names: Optional dictionary mapping class indices to class names
        save_confusion_matrix: Optional path to save confusion matrix plot
        ensemble_strategy: Optional ensemble strategy ("majority", "logits_average", "confidence_weighted")
        return_confidence: Whether to return confidence scores
        criterion: Optional loss criterion
        num_workers: Number of workers for data loading
        batch_size: Batch size for evaluation (ignored if ensemble_strategy is provided)
        
    Returns:
        Tuple containing:
            - Dictionary of overall metrics
            - DataFrame containing the confusion matrix
            - Optional dictionary mapping sample/series IDs to confidence scores
    """
    # If using ensemble, delegate to evaluate_with_ensemble
    if ensemble_strategy is not None:
        metrics, cm_df, _, confidences = evaluate_with_ensemble(
            model=model,
            dataset=dataset,
            device=device,
            class_names=class_names,
            save_confusion_matrix=save_confusion_matrix,
            ensemble_strategy=ensemble_strategy,
            criterion=criterion,
            num_workers=num_workers
        )
        if return_confidence:
            return metrics, cm_df, confidences
        return metrics, cm_df
    
    # Create dataloader for regular evaluation
    eval_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Initialize inference engine
    engine = InferenceEngine(
        model=model,
        device=device
    )
    
    # Get predictions and optionally confidences
    if return_confidence:
        predictions = engine.predict(
            eval_loader,
            return_confidence=True
        )
        # Separate predictions and confidences
        y_pred = []
        y_true = []
        confidences = {}
        
        for batch_idx, (pred_conf, labels, _) in enumerate(zip(predictions.values(), eval_loader)):
            pred, conf = pred_conf
            batch_labels = labels.view(-1).cpu().numpy()
            y_true.extend(batch_labels)
            y_pred.extend(pred.cpu().numpy())
            confidences[batch_idx] = conf
    else:
        predictions = engine.predict(eval_loader)
        y_pred = []
        y_true = []
        
        for batch_idx, (pred, labels, _) in enumerate(zip(predictions.values(), eval_loader)):
            batch_labels = labels.view(-1).cpu().numpy()
            y_true.extend(batch_labels)
            y_pred.extend(pred.cpu().numpy())
    
    # Calculate loss if criterion provided
    test_loss = 0.0
    if criterion is not None:
        with torch.no_grad():
            for features, labels, _ in eval_loader:
                logits = engine.predict(features, return_logits=True)
                loss = criterion(logits, labels.to(device))
                test_loss += loss.item()
        test_loss /= len(eval_loader)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
    }
    if criterion is not None:
        metrics['loss'] = test_loss
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Convert class indices to names if provided
    if class_names is not None:
        index = [class_names.get(i, str(i)) for i in range(len(cm))]
    else:
        index = list(range(len(cm)))
    
    # Create confusion matrix DataFrame
    cm_df = pd.DataFrame(
        cm, 
        index=index,
        columns=index
    )
    
    # Plot confusion matrix if save path provided
    if save_confusion_matrix:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_df, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            square=True
        )
        plt.title('Confusion Matrix' + (f' (Ensemble: {ensemble_strategy})' if ensemble_strategy else ''))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_confusion_matrix)
        plt.close()
    
    if return_confidence:
        return metrics, cm_df, confidences
    return metrics, cm_df


def print_evaluation_results(
    metrics: Dict[str, float],
    confusion_matrix_df: pd.DataFrame,
    confidences: Optional[Dict[int, float]] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> None:
    """
    Print evaluation results in a formatted way.
    
    Args:
        metrics: Dictionary of overall metrics
        confusion_matrix_df: DataFrame containing confusion matrix
        confidences: Optional dictionary mapping indices to confidence scores
        class_names: Optional dictionary mapping class indices to class names
    """
    print("Overall Metrics:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric.capitalize()}: {value:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix_df)
    
    if confidences:
        print("\nConfidence Statistics:")
        conf_values = list(confidences.values())
        print(f"Mean confidence: {np.mean(conf_values):.4f}")
        print(f"Std confidence: {np.std(conf_values):.4f}")
        print(f"Min confidence: {np.min(conf_values):.4f}")
        print(f"Max confidence: {np.max(conf_values):.4f}")
        
        # Print samples with lowest confidence
        print("\nLowest Confidence Predictions:")
        n_lowest = min(5, len(confidences))
        lowest_conf = sorted(confidences.items(), key=lambda x: x[1])[:n_lowest]
        for idx, conf in lowest_conf:
            sample_name = f"Sample {idx}"
            print(f"{sample_name}: {conf:.4f}")

def evaluate_with_ensemble(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: str = "cuda",
    class_names: Optional[Dict[int, str]] = None,
    save_confusion_matrix: Optional[str] = None,
    ensemble_strategy: str = "majority",
    criterion = None,
    num_workers: int = 0,
) -> Tuple[Dict[str, float], pd.DataFrame, str, Dict[int, float]]:
    """
    Evaluate model using ensemble predictions from multiple samples per series.
    
    Args:
        model: The trained model
        dataset: Dataset to evaluate on
        device: Device to run evaluation on
        class_names: Optional dictionary mapping class indices to class names
        save_confusion_matrix: Optional path to save confusion matrix plot
        ensemble_strategy: Strategy for combining multiple predictions ("majority", "logits_average", "confidence_weighted")
        criterion: Optional loss criterion
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple containing:
            - Dictionary of overall metrics
            - DataFrame containing the confusion matrix
            - String containing the classification report
            - Dictionary mapping series IDs to confidence scores
    """
    # Create dataloaders - batch_size=1 for ensemble prediction, larger batch for loss calculation
    ensemble_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )
    
    # For loss calculation, we can use larger batches
    eval_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,  # Larger batch size for efficiency
        shuffle=False,
        num_workers=num_workers
    ) if criterion is not None else None
    
    # Initialize ensemble predictor
    predictor = InferenceEngine(
        model=model,
        device=device,
        ensemble_strategy=ensemble_strategy
    )
    
    # Get predictions with confidence scores
    predictions = predictor.predict(ensemble_loader, return_confidence=True)
    
    # Separate predictions and confidences
    y_true = []
    y_pred = []
    confidences = {}
    
    # Collect video IDs and predictions
    for batch_idx, (_, labels, _) in enumerate(ensemble_loader):
        # With batch_size=1, batch_idx equals sample_idx
        video_id = ensemble_loader.dataset.get_video_idx(batch_idx)
        
        pred_conf = predictions[batch_idx]
        pred, conf = pred_conf
        
        y_true.append(labels.item())
        y_pred.append(pred.item())
        confidences[video_id] = conf
    
    # Calculate loss if criterion provided
    test_loss = 0.0
    if criterion is not None:
        with torch.no_grad():
            for features, labels, _ in eval_loader:
                logits = predictor.predict(features, return_logits=True)
                loss = criterion(logits, labels.to(device))
                test_loss += loss.item()
        test_loss /= len(eval_loader)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
    }
    if criterion is not None:
        metrics['loss'] = test_loss
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Convert class indices to names if provided
    if class_names is not None:
        index = [class_names.get(i, str(i)) for i in range(len(cm))]
    else:
        index = list(range(len(cm)))
    
    # Create confusion matrix DataFrame
    cm_df = pd.DataFrame(
        cm, 
        index=index,
        columns=index
    )
    
    # Generate classification report
    if class_names is not None:
        target_names = [class_names.get(i, str(i)) for i in range(len(cm))]
    else:
        target_names = None
    
    report = classification_report(
        y_true, 
        y_pred,
        target_names=target_names,
        zero_division=0
    )
    
    # Plot confusion matrix if save path provided
    if save_confusion_matrix:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_df, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            square=True
        )
        plt.title(f'Confusion Matrix (Ensemble: {ensemble_strategy})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_confusion_matrix)
        plt.close()
    
    return metrics, cm_df, report, confidences

def print_ensemble_evaluation_results(
    metrics: Dict[str, float],
    confusion_matrix_df: pd.DataFrame,
    classification_report_str: str,
    confidences: Dict[int, float],
    class_names: Optional[Dict[int, str]] = None,
) -> None:
    """
    Print ensemble evaluation results in a formatted way.
    
    Args:
        metrics: Dictionary of overall metrics
        confusion_matrix_df: DataFrame containing confusion matrix
        classification_report_str: Classification report string
        confidences: Dictionary mapping video IDs to confidence scores
        class_names: Optional dictionary mapping class indices to class names
    """
    print("\n=== Ensemble Evaluation Results ===")
    
    print("\nOverall Metrics:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric.capitalize()}: {value:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix_df)
    
    print("\nDetailed Classification Report:")
    print(classification_report_str)
    
    print("\nConfidence Statistics:")
    conf_values = list(confidences.values())
    print(f"Mean confidence: {np.mean(conf_values):.4f}")
    print(f"Std confidence: {np.std(conf_values):.4f}")
    print(f"Min confidence: {np.min(conf_values):.4f}")
    print(f"Max confidence: {np.max(conf_values):.4f}")
    
    # Print videos with lowest confidence
    print("\nLowest Confidence Predictions:")
    n_lowest = min(5, len(confidences))
    lowest_conf = sorted(confidences.items(), key=lambda x: x[1])[:n_lowest]
    for vid_id, conf in lowest_conf:
        video_name = f"Video {vid_id}"
        print(f"{video_name}: {conf:.4f}")
