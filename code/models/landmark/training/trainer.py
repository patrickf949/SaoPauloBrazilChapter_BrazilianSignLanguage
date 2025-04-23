from models.landmark.utils import load_config, load_obj
from typing import Union, Dict
import os 
from torch import nn
from torch.utils.data import DataLoader
import csv
from models.landmark.training.train_functions import evaluate

def train(config: Union[str, Dict]):
    config_base_dir = os.path.basename(config)
    config = load_config(config, "training_config")
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    patience = config["patience"]
    device = config["device"]
    log_path = config["log_path"]

    model_config = load_config(f"{config_base_dir}/model/{config["model"]}.yaml")
    model = load_obj(model_config["class_name"])(**model_config["parameters"])
    optimizer_config = load_config(f"{config_base_dir}/optimizer/{config["optimizer"]}.yaml")
    optimizer = load_obj(optimizer_config["class_name"])(model.parameters(), **optimizer_config["parameters"])
    scheduler_config = load_config(f"{config_base_dir}/scheduler/{config["scheduler"]}.yaml")
    scheduler = load_obj(scheduler_config["class_name"])(optimizer, **scheduler_config["parameters"])

    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    log_data = []


    for epoch in range(num_epochs):
        if config["training_type"] == "cross_validation":
            ...
        else:
            ...

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
    test_loader = DataLoader(test_dataset, batch_size=1)
    top1_acc, topk_acc = evaluate(model, test_loader, device)
    print(f"\nBest Epoch: {best_epoch} | Final Val Accuracy (Top-1): {top1_acc:.4f} | Top-K Accuracy: {topk_acc:.4f}")

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