from models.landmark.utils.utils import load_obj
import os
from torch import nn
from torch.utils.data import DataLoader
import csv
from models.landmark.utils.train import (
    train_epoch,
    train_epoch_fold,
)
from models.landmark.utils.evaluate import evaluate
from models.landmark.dataset.landmark_dataset import LandmarkDataset
from models.landmark.dataset.dataloader_functions import collate_fn_pad
from omegaconf import DictConfig
import hydra


def get_dataset(config: DictConfig):
    collate_fn = (
        collate_fn_pad if "intervals" in config.dataset.frame_interval_fn else None
    )
    batch_size = config.training.batch_size

    def create_dataloader(split: str, shuffle: bool):
        return DataLoader(
            LandmarkDataset(
                config.dataset, config.features, config.augmentation, split
            ),
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    if config.training.type == "cross_validation":
        datasets = {
            "train_dataset": LandmarkDataset(
                config.dataset, config.features, config.augmentation, "train"
            ),
            "test_dataset": LandmarkDataset(
                config.dataset, config.features, config.augmentation, "test"
            ),
        }
    else:
        datasets = {
            "train_dataset": create_dataloader("train", shuffle=True),
            "val_dataset": create_dataloader("val", shuffle=False),
            "test_dataset": create_dataloader("test", shuffle=False),
        }

    return datasets


@hydra.main(version_base=None, config_path="./configs", config_name="train_config")
def train(config: DictConfig):
    device = config.training.device
    num_epochs = config.training.num_epochs

    model = load_obj(config.model.class_name)(**config.model.params)
    model.to(device)

    optimizer = load_obj(config.optimizer.class_name)(
        model.parameters(), **config.optimizer.params
    )

    scheduler = load_obj(config.scheduler.class_name)(
        optimizer, **config.scheduler.params
    )

    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    log_data = []

    datasets = get_dataset(config)

    for epoch in range(num_epochs):
        if config.training.type == "cross_validation":
            avg_train_loss, avg_val_loss = train_epoch_fold(
                epoch,
                config.training.k_folds,
                model,
                datasets,
                config.training.batch_size,
                device,
                optimizer,
                criterion,
            )
        else:
            avg_train_loss, avg_val_loss = train_epoch(
                model, device, datasets, optimizer, criterion
            )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        log_data.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{config.training.patience}")
            if patience_counter >= config.training.patience:
                print("Early stopping triggered.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    # ----- evaluate -----
    # Final evaluation (optional - could be on last fold or an extra hold-out set)
    test_loader = DataLoader(datasets["test_dataset"], batch_size=1)
    top1_acc, topk_acc = evaluate(model, test_loader, device)
    print(
        f"\nBest Epoch: {best_epoch} | Final Val Accuracy (Top-1): {top1_acc:.4f} | Top-K Accuracy: {topk_acc:.4f}"
    )

    # ----- optional logging to file -----
    if "log_path" in config:
        log_path = config["log_path"]
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writeheader()
            writer.writerows(log_data)
        print(f"Training log saved to: {log_path}")

    return top1_acc, topk_acc, best_epoch, log_data


if __name__ == "__main__":
    train()
