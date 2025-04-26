from models.landmark.utils import load_config, load_obj
from typing import Union, Dict
import os
from torch import nn
from torch.utils.data import DataLoader
import csv
from models.landmark.training.train_functions import (
    evaluate,
    train_epoch,
    train_epoch_fold,
)
from models.landmark.dataset.landmark_dataset import LandmarkDataset
from models.landmark.dataset.dataloader_functions import collate_fn_pad


def get_dataset(
    config_base_dir: str, dataset_config: str, training_type: str, batch_size: int
):
    dataset_base_dir = (
        f"{os.path.dirname(os.path.dirname(config_base_dir))}/dataset/configs/"
    )
    dataset_config = load_config(f"{dataset_base_dir}/{dataset_config}.yaml")

    for estimator in dataset_config["estimators"]:
        for parameter, feature_path in dataset_config["estimators"][estimator][
            "parameters"
        ].items():
            dataset_config["estimators"][estimator]["parameters"][parameter] = (
                f"{dataset_base_dir}/{feature_path}.yaml"
            )

    dataset_config["data_dir"] = [
        f"{os.path.dirname(os.path.dirname(config_base_dir))}/{dataset_config['data_dir']}"
    ]

    dataset_config["data_path"] = [
        f"{os.path.dirname(os.path.dirname(config_base_dir))}/{dataset_config['data_path']}"
    ]
    collate_fn = (
        collate_fn_pad if "intervals" in dataset_config["frame_interval_fn"] else None
    )
    if training_type == "cross_validation":
        datasets = {"train_dataset": LandmarkDataset(dataset_config, "train")}
    else:
        if collate_fn is not None:
            datasets = {
                "train_dataset": DataLoader(
                    LandmarkDataset(dataset_config, "train"),
                    shuffle=True,
                    collate_fn=collate_fn,
                    batch_size=batch_size,
                ),
                "val_dataset": DataLoader(
                    LandmarkDataset(dataset_config, "val"),
                    shuffle=False,
                    collate_fn=collate_fn,
                    batch_size=batch_size,
                ),
            }
        else:
            datasets = {
                "train_dataset": DataLoader(
                    LandmarkDataset(dataset_config, "train"),
                    shuffle=True,
                    batch_size=batch_size,
                ),
                "val_dataset": DataLoader(
                    LandmarkDataset(dataset_config, "val"),
                    shuffle=False,
                    batch_size=batch_size,
                ),
            }

    return datasets


def train(config: Union[str, Dict]):
    config_base_dir = os.path.dirname(config)
    config = load_config(config, "training_config")
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    patience = config["patience"]
    device = config["device"]
    log_path = config["log_path"]

    model_config = load_config(f"{config_base_dir}/model/{config['model']}.yaml")
    model = load_obj(model_config["class_name"])(**model_config["parameters"])
    optimizer_config = load_config(
        f"{config_base_dir}/optimizer/{config['optimizer']}.yaml"
    )
    optimizer = load_obj(optimizer_config["class_name"])(
        model.parameters(), **optimizer_config["parameters"]
    )
    scheduler_config = load_config(
        f"{config_base_dir}/scheduler/{config['scheduler']}.yaml"
    )
    scheduler = load_obj(scheduler_config["class_name"])(
        optimizer, **scheduler_config["parameters"]
    )

    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    log_data = []

    datasets = load_dataset(config_base_dir, config["dataset"], config["training_type"])

    for epoch in range(num_epochs):
        if config["training_type"] == "cross_validation":
            train_epoch_fold(
                epoch,
            )
        else:
            train_epoch()

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
            print(f"Patience: {patience_counter}/{config['patience']}")
            if patience_counter >= config["patience"]:
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

    return acc, best_epoch, log_data
