import torch
from torch import nn
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from models.landmark.models.transformers import TransformerClassifier
from models.landmark.dataset.landmark_dataset import LandmarkDataset
from models.landmark.dataset.dataloader_functions import collate_fn_pad
from typing import Dict



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
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_indices = list(kfold.split(dataset))

    fold = epoch % k_folds
    print(f"\n--- Fold {fold + 1}/{k_folds} ---")

    train_ids, val_ids = fold_indices[fold]
    train_loader = DataLoader(
        Subset(dataset, train_ids),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_pad,
    )
    val_loader = DataLoader(
        Subset(dataset, val_ids),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_pad,
    )

    # ----- training step -----
    model.train()
    total_loss = 0

    for features, labels, attention_mask in train_loader:
        y = labels.squeeze().to(device)
        features = features.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()
        logits = model(features, attention_mask=attention_mask)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # ----- validation step -----
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for features, labels, attention_mask in val_loader:
            y = labels.squeeze().to(device)
            features = features.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(features, attention_mask=attention_mask)
            loss = criterion(logits, y)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

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



if __name__ == "__main__":
    train_dataset = LandmarkDataset(
        "/home/ana/Projects/Omdena/HealthSignLangBrazil/SaoPauloBrazilChapter_BrazilianSignLanguage/code/models/landmark/dataset/configs/dataset.yaml",
        "train",
    )
    test_dataset = LandmarkDataset(
        "/home/ana/Projects/Omdena/HealthSignLangBrazil/SaoPauloBrazilChapter_BrazilianSignLanguage/code/models/landmark/dataset/configs/dataset.yaml",
        "test",
    )
    model = TransformerClassifier(input_size=212, num_classes=25)
    train_rotating_folds(
        model,
        train_dataset,
        test_dataset,
        "/home/ana/Projects/Omdena/HealthSignLangBrazil/SaoPauloBrazilChapter_BrazilianSignLanguage/code/models/landmark/training/configs/training.yaml",
    )
