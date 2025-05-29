from models.landmark.utils.utils import load_obj
from models.landmark.features.features_analysis import analyze_dataset_features
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
from models.landmark.dataset.dataloader_functions import collate_func_pad
from models.landmark.utils.path_utils import get_save_paths, get_data_paths, load_base_paths
from models.landmark.dataset.prepare_data import prepare_training_metadata, prepare_landmark_arrays
from models.landmark.utils.logging import TrainingLogger
import torch
from omegaconf import DictConfig
import hydra
from models.landmark.utils.utils import set_config_param
from datetime import datetime

# Ensure base output directories exist
paths = load_base_paths()
os.makedirs(paths.logs_base, exist_ok=True)
os.makedirs(os.path.join(paths.logs_base, "runs"), exist_ok=True)

def get_dataset(config: DictConfig):
    # Always generate fresh training metadata
    landmarks_dir, metadata_path = get_data_paths(config.dataset.data_version)
    print(f"Generating training metadata at {metadata_path}...")
    prepare_training_metadata(config.dataset.data_version)
    # Prepare pure numpy arrays of landmark data to be used for faster feature processing
    # print(f"Generating landmark arrays from {landmarks_dir}...")
    # prepare_landmark_arrays(landmarks_dir, config.features.positions)

    # Create datasets for all splits
    # The only difference between each is the augmentation step
    # If you use the val_dataset or test_dataset, augmentation is not applied because dataset_split is not "train"
    # Currently the logic of the training process with cross validation doesn't use the val_dataset, just the train_dataset, which could be improved
    datasets = {
        "train_dataset": LandmarkDataset(
            config.dataset, config.features, config.augmentation, "train"
        ),
        "val_dataset": LandmarkDataset(
            config.dataset, config.features, config.augmentation, "val"
        ),
        "test_dataset": LandmarkDataset(
            config.dataset, config.features, config.augmentation, "test"
        ),
    }

    return datasets

@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="train_config",
)
def train(config: DictConfig):
    device = config.training.device
    num_epochs = config.training.num_epochs

    # Get dataset first
    datasets = get_dataset(config)
    
    # Analyze dataset and get feature dimensions
    n_features = analyze_dataset_features(datasets["train_dataset"])
    
    # Update model config with actual input size
    set_config_param(config.model.params, "input_size", n_features)
    
    # Initialize model with updated config
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

    # Get save paths
    log_path, best_model_path, final_model_path = get_save_paths(config)

    # Initialize logger
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(paths.logs_base, "runs", current_time)
    k_folds = config.training.k_folds if config.training.type == "cross_validation" else None
    logger = TrainingLogger(log_dir, log_path, k_folds)

    # Remove model graph logging - causing tracing issues
    # try:
    #     if isinstance(datasets["train_dataset"], DataLoader):
    #         sample_input = next(iter(datasets["train_dataset"]))[0]
    #     else:
    #         sample_input = next(iter(datasets["train_dataset"]))[0]
    #     
    #     if not isinstance(sample_input, torch.Tensor):
    #         sample_input = torch.tensor(sample_input)
    #     sample_input = sample_input.unsqueeze(0).to(device)
    #     logger.writer.add_graph(model, sample_input)
    # except Exception as e:
    #     print(f"Could not add model graph to TensorBoard: {e}")

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        if config.training.type == "cross_validation":
            # Modified to get per-fold metrics
            avg_train_loss, avg_val_loss, fold_metrics, fold_stats = train_epoch_fold(
                epoch,
                config.training.k_folds,
                model,
                datasets,
                config.training.batch_size,
                device,
                optimizer,
                criterion,
            )
            
            # Log metrics
            logger.log_fold_training(epoch, fold_metrics, fold_stats, avg_train_loss, avg_val_loss)
        else:
            avg_train_loss, avg_val_loss = train_epoch(
                model, device, datasets, optimizer, criterion, config.training.batch_size
            )
            
            # Log metrics
            logger.log_standard_training(epoch, avg_train_loss, avg_val_loss)

        print(
            f"Epoch Average Results:\n\tTrain Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
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

        # Step the scheduler if it exists
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

    # Load the best model state before final evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)
        # Save best model checkpoint if enabled
        if best_model_path is not None:
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, best_model_path)
            print(f"Saved best model checkpoint to: {best_model_path}")

    # Save final model state if enabled
    if final_model_path is not None:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_val_loss,
        }, final_model_path)
        print(f"Saved final model checkpoint to: {final_model_path}")

    # ----- evaluate -----
    # Final evaluation (optional - could be on last fold or an extra hold-out set)
    test_loader = DataLoader(datasets["test_dataset"], batch_size=1)
    top1_acc, topk_acc, test_loss = evaluate(model, test_loader, device, criterion=criterion)
    print(
        f"\nBest Epoch: {best_epoch} | Final Val Accuracy (Top-1): {top1_acc:.4f} | Top-K Accuracy: {topk_acc:.4f} | Test Loss: {test_loss:.4f}"
    )

    # Close logger
    logger.close()

    return top1_acc, topk_acc, test_loss, best_epoch, log_data


if __name__ == "__main__":
    train()
