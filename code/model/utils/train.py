import torch
from torch import nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Subset, DataLoader
from model.dataset.landmark_dataset import LandmarkDataset
from model.dataset.dataloader_functions import collate_func_pad
from typing import Dict, Tuple, List
import numpy as np
import copy

def training_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: str,
    optimizer: torch.optim.Optimizer,
    criterion,
) -> float:
    """Perform a single training step.
    
    Args:
        model: The neural network model
        batch: Tuple of (features, labels, attention_mask)
        device: Device to run the model on
        optimizer: Optimizer for updating model parameters
        criterion: Loss function
        
    Returns:
        float: The loss value for this step
    """
    features, labels, attention_mask = batch
    y = labels.squeeze().to(device)
    features = features.to(device)
    attention_mask = attention_mask.to(device)

    optimizer.zero_grad()
    logits = model(features, attention_mask=attention_mask)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    return loss.item()


def validation_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: str,
    criterion,
) -> float:
    """Perform a single validation step.
    
    Args:
        model: The neural network model
        batch: Tuple of (features, labels, attention_mask)
        device: Device to run the model on
        criterion: Loss function
        
    Returns:
        float: The loss value for this step
    """
    features, labels, attention_mask = batch
    y = labels.squeeze().to(device)
    features = features.to(device)
    attention_mask = attention_mask.to(device)

    logits = model(features, attention_mask=attention_mask)
    loss = criterion(logits, y)

    return loss.item()


def train_epoch_fold(
    epoch: int,
    k_folds: int,
    model: nn.Module,
    datasets: Dict[str, LandmarkDataset],
    batch_size: int,
    device: str,
    optimizer: torch.optim.Optimizer,
    criterion,
) -> Tuple[float, float, List[Tuple[float, float]], List[dict]]:
    """Train the model for one epoch using k-fold cross-validation (StratifiedGroupKFold).
    
    Args:
        epoch: Current epoch number
        k_folds: Number of folds for cross-validation
        model: The neural network model
        datasets: Dictionary containing train dataset
        batch_size: Batch size for training
        device: Device to run the model on
        optimizer: Optimizer for updating model parameters
        criterion: Loss function
        
    Returns:
        Tuple containing:
            - Average training loss across all folds
            - Average validation loss across all folds
            - List of (train_loss, val_loss) tuples for each fold
            - List of statistics for each fold
    """
    train_dataset = datasets["train_dataset"]
    # val_dataset is a copy of train_dataset, but with augmentation disabled
    val_dataset = copy.deepcopy(train_dataset)
    val_dataset.dataset_split = "val"
    
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty")
    if len(train_dataset) != len(val_dataset):
        raise ValueError("Training and validation datasets don't have the same length")
    
    # Create groups array where each sample from the same video gets the same group number
    groups = []
    labels = []
    for video_idx in range(len(train_dataset.metadata)):
        video_label = train_dataset.metadata.iloc[video_idx]["label_encoded"]
        groups.extend([video_idx] * train_dataset.samples_per_video[video_idx])
        labels.extend([video_label] * train_dataset.samples_per_video[video_idx])
    
    groups = np.array(groups)
    stratified_group_kfold = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42 + epoch)
    fold_indices = list(stratified_group_kfold.split(range(len(train_dataset)), labels, groups=groups))

    total_train_loss = 0
    total_val_loss = 0
    fold_metrics = []  # Store per-fold metrics
    fold_stats = []    # Store fold statistics

    print(f"{k_folds}-fold Cross-Validation Results (using StratifiedGroupKFold):")
    # Iterate through all folds for this epoch
    for fold, (train_ids, val_ids) in enumerate(fold_indices):
        print(f"- Fold {fold + 1} -")

        # Calculate fold statistics
        train_groups = np.unique(groups[train_ids])
        val_groups = np.unique(groups[val_ids])
        fold_k_stats = {
            'train_samples': len(train_ids),
            'val_samples': len(val_ids),
            'train_groups': len(train_groups),
            'val_groups': len(val_groups),
            'train_samples_per_group': len(train_ids) / len(train_groups),
            'val_samples_per_group': len(val_ids) / len(val_groups)
        }
        fold_stats.append(fold_k_stats)

        train_loader = DataLoader(
            Subset(train_dataset, train_ids),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_func_pad,
        )
        val_loader = DataLoader(
            Subset(val_dataset, val_ids),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_func_pad,
        )

        # ----- training step -----
        model.train()
        fold_train_loss = 0
        for idx, batch in enumerate(train_loader):
            fold_train_loss += training_step(model, batch, device, optimizer, criterion)

        avg_fold_train_loss = fold_train_loss / len(train_loader)
        total_train_loss += avg_fold_train_loss

        # ----- validation step -----
        model.eval()
        fold_val_loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                fold_val_loss += validation_step(model, batch, device, criterion)

        avg_fold_val_loss = fold_val_loss / len(val_loader)
        total_val_loss += avg_fold_val_loss

        # Store per-fold metrics
        fold_metrics.append((avg_fold_train_loss, avg_fold_val_loss))

        print(
            f"\tTrain Loss: {avg_fold_train_loss:.4f} | "
            f"{fold_k_stats['train_samples']} samples from {fold_k_stats['train_groups']} groups "
            f"({fold_k_stats['train_samples_per_group']:.1f} per)"
        )
        print(
            f"\tVal Loss: {avg_fold_val_loss:.4f}   | "
            f"{fold_k_stats['val_samples']} samples from {fold_k_stats['val_groups']} groups "
            f"({fold_k_stats['val_samples_per_group']:.1f} per)"
        )

    # Return average losses across all folds and per-fold metrics
    avg_train_loss = total_train_loss / k_folds
    avg_val_loss = total_val_loss / k_folds

    return avg_train_loss, avg_val_loss, fold_metrics, fold_stats


def train_epoch(
    model: nn.Module,
    device: str,
    datasets: Dict[str, LandmarkDataset],
    optimizer: torch.optim.Optimizer,
    criterion,
    batch_size: int,
) -> Tuple[float, float]:
    """Train the model for one epoch using standard train/val split.
    
    Args:
        model: The neural network model
        device: Device to run the model on
        datasets: Dictionary containing train and validation datasets
        optimizer: Optimizer for updating model parameters
        criterion: Loss function
        batch_size: Batch size for training
        
    Returns:
        Tuple containing:
            - Average training loss
            - Average validation loss
    """
    if len(datasets["train_dataset"]) == 0:
        raise ValueError("Train dataset is empty")
    if len(datasets["val_dataset"]) == 0:
        raise ValueError("Validation dataset is empty")

    # Create DataLoaders from datasets
    train_loader = DataLoader(
        datasets["train_dataset"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_func_pad,
    )
    val_loader = DataLoader(
        datasets["val_dataset"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_func_pad,
    )

    # ----- training step -----
    model.train()
    total_loss = 0
    for idx, batch in enumerate(train_loader):
        total_loss += training_step(model, batch, device, optimizer, criterion)

    avg_train_loss = total_loss / len(train_loader)

    # ----- validation step -----
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            val_loss += validation_step(model, batch, device, criterion)

    avg_val_loss = val_loss / len(val_loader)

    return avg_train_loss, avg_val_loss
