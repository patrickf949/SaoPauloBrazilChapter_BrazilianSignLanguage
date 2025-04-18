import torch
from torch import nn
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from typing import Union, Dict
import os
import csv
from models.landmark.utils import load_config
from models.landmark.training.train import evaluate
from models.landmark.training.transformers import TransformerClassifier
from models.landmark.dataset.landmark_dataset import LandmarkDataset


def train_rotating_folds(
    model,
    dataset: LandmarkDataset,
    test_dataset: LandmarkDataset,
    config: Union[str, Dict],
    k_folds: int = 5,
):
    config = load_config(config, "training_config")
    device = config["device"]
    num_epochs = config["num_epochs"]
    batch_size = config.get("batch_size", 32)

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_indices = list(kfold.split(dataset))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))

    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    log_data = []

    for epoch in range(num_epochs):
        fold = epoch % k_folds
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} | Fold {fold + 1}/{k_folds} ---")

        train_ids, val_ids = fold_indices[fold]
        train_loader = DataLoader(
            Subset(dataset, train_ids), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            Subset(dataset, val_ids), batch_size=batch_size, shuffle=False
        )

        # ----- training step -----
        model.train()
        total_loss = 0

        for features, labels in train_loader:
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
            for features, labels in val_loader:
                y = labels.squeeze().to(device)
                features = features.to(device)

                logits = model(features)
                loss = criterion(logits, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        log_data.append(
            {
                "epoch": epoch + 1,
                "fold": fold + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }
        )

        # ----- early stopping -----
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{config['patience']}")
            if patience_counter >= config["patience"]:
                print("Early stopping triggered.")
                break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Final evaluation (optional - could be on last fold or an extra hold-out set)
    test_loader = DataLoader(test_dataset, batch_size=1)
    acc = evaluate(model, test_loader, device)
    print(f"\nBest Epoch: {best_epoch} | Final Val Accuracy: {acc:.4f}")

    # ----- optional logging to file -----
    if "log_path" in config:
        log_path = config["log_path"]
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["epoch", "fold", "train_loss", "val_loss"]
            )
            writer.writeheader()
            writer.writerows(log_data)
        print(f"Training log saved to: {log_path}")

    return acc, best_epoch, log_data


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
