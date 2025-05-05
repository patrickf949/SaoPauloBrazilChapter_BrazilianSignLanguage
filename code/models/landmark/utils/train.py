import torch
from torch import nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Subset, DataLoader
from models.landmark.dataset.landmark_dataset import LandmarkDataset
from models.landmark.dataset.dataloader_functions import collate_func_pad
from typing import Dict
import numpy as np


def train_epoch_fold(
    epoch: int,
    k_folds: int,
    model: nn.Module,
    datasets: Dict[str, LandmarkDataset],
    batch_size: int,
    device: str,
    optimizer: torch.optim.Optimizer,
    criterion,
):
    dataset = datasets["train_dataset"]
    
    # Create groups array where each sample from the same video gets the same group number
    groups = []
    labels = []
    for video_idx in range(len(dataset.data)):
        video_label = dataset.data.iloc[video_idx]["label_encoded"]
        groups.extend([video_idx] * dataset.samples_per_video[video_idx])
        labels.extend([video_label] * dataset.samples_per_video[video_idx])
    
    stratified_group_kfold = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42 + epoch)
    fold_indices = list(stratified_group_kfold.split(range(len(dataset)), labels, groups=groups))

    total_train_loss = 0
    total_val_loss = 0

    print(f"{k_folds}-fold Cross-Validation Results (using StratifiedGroupKFold):")
    # Iterate through all folds for this epoch
    for fold, (train_ids, val_ids) in enumerate(fold_indices):
        train_loader = DataLoader(
            Subset(dataset, train_ids),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_func_pad,
        )
        val_loader = DataLoader(
            Subset(dataset, val_ids),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_func_pad,
        )

        # ----- training step -----
        model.train()
        fold_train_loss = 0

        for features, labels, attention_mask in train_loader:
            y = labels.squeeze().to(device)
            features = features.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            logits = model(features, attention_mask=attention_mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            fold_train_loss += loss.item()

        avg_fold_train_loss = fold_train_loss / len(train_loader)
        total_train_loss += avg_fold_train_loss

        # ----- validation step -----
        model.eval()
        fold_val_loss = 0
        with torch.no_grad():
            for features, labels, attention_mask in val_loader:
                y = labels.squeeze().to(device)
                features = features.to(device)
                attention_mask = attention_mask.to(device)
                logits = model(features, attention_mask=attention_mask)
                loss = criterion(logits, y)
                fold_val_loss += loss.item()

        avg_fold_val_loss = fold_val_loss / len(val_loader)
        total_val_loss += avg_fold_val_loss

        print(f"\tFold {fold + 1} - Train Loss: {avg_fold_train_loss:.4f} | Val Loss: {avg_fold_val_loss:.4f}")

    # Return average losses across all folds
    avg_train_loss = total_train_loss / k_folds
    avg_val_loss = total_val_loss / k_folds

    return avg_train_loss, avg_val_loss


def train_epoch(
    model: nn.Module,
    device: str,
    datasets: Dict[str, LandmarkDataset],
    optimizer: torch.optim.Optimizer,
    criterion,
):
    train_loader = datasets["train_dataset"]
    val_loader = datasets["val_dataset"]

    model.train()
    total_loss = 0

    for idx, batch in enumerate(train_loader):
        features, labels = batch
        y = labels.squeeze().to(device)
        features = features.to(device)

        optimizer.zero_grad()
        logits = model(features)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # ----- validation step -----
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            features, labels = batch
            y = labels.squeeze(0).to(device)
            features = features.to(device)

            logits = model(features)
            loss = criterion(logits, y)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    return avg_train_loss, avg_val_loss
