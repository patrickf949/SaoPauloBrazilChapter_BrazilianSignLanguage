from model.utils.utils import load_obj
from model.features.features_analysis import analyze_dataset_features
import os
from torch import nn
from torch.utils.data import DataLoader
import csv
from model.utils.train import (
    train_epoch,
    train_epoch_fold,
)
from model.utils.evaluate import evaluate
from model.dataset.landmark_dataset import LandmarkDataset
from model.dataset.dataloader_functions import collate_func_pad
from model.utils.path_utils import get_save_paths, get_data_paths, load_base_paths
from model.dataset.prepare_data import prepare_training_metadata, prepare_landmark_arrays
from model.utils.logging import TrainingLogger
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from model.utils.utils import set_config_param
from datetime import datetime

# Ensure base output directories exist
paths = load_base_paths()
os.makedirs(paths.logs_base, exist_ok=True)
os.makedirs(os.path.join(paths.logs_base, "runs"), exist_ok=True)

def save_checkpoint(state: dict, filepath: str, config: DictConfig) -> None:
    """Save training checkpoint to file.
    
    Args:
        state: Dictionary containing checkpoint data
        filepath: Path to save checkpoint to
        config: Current training configuration
    """
    # Convert config to dict and add to state
    state['config'] = OmegaConf.to_container(config, resolve=True)
    torch.save(state, filepath)
    print(f"Saved checkpoint to: {filepath}")

def load_checkpoint(filepath: str) -> dict:
    """Load training checkpoint from file.
    
    Args:
        filepath: Path to checkpoint file
        
    Returns:
        Dictionary containing checkpoint data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint found at {filepath}")
    
    checkpoint = torch.load(filepath)
    print(f"Loaded checkpoint from: {filepath}")
    return checkpoint

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
    # Check if resuming training
    resume = config.training.get('resume', False)
    checkpoint_dir = config.training.get('checkpoint_dir', None)
    
    if resume and checkpoint_dir:
        # Load checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        checkpoint = load_checkpoint(checkpoint_path)
        
        # Use the saved config
        config = OmegaConf.create(checkpoint['config'])
        
        # Keep the resume settings from current config
        config.training.resume = True
        config.training.checkpoint_dir = checkpoint_dir
        
        current_time = checkpoint['timestamp']  # Use original timestamp
        log_dir = os.path.join(paths.logs_base, "runs", current_time)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        patience_counter = checkpoint['patience_counter']
        best_epoch = checkpoint['best_epoch']
        best_model_state = checkpoint['best_model_state']
    else:
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = os.path.join(paths.logs_base, "runs", current_time)
        start_epoch = 0
        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        best_epoch = 0

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

    # Load model and optimizer states if resuming
    if resume and checkpoint_dir:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Get save paths
    log_path, best_model_path, final_model_path = get_save_paths(config)

    # Initialize logger
    k_folds = config.training.k_folds if config.training.type == "cross_validation" else None
    logger = TrainingLogger(log_dir, log_path, k_folds)

    checkpoint_path = os.path.join(log_dir, "latest_checkpoint.pt")
    
    log_data = []

    for epoch in range(start_epoch, num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        if config.training.type == "cross_validation":
            # Modified to get per-fold metrics
            avg_train_loss, avg_val_loss, fold_metrics, fold_stats = train_epoch_fold(
                epoch,
                config.training.k_folds,
                model,
                datasets,
                config.training.train_batch_size,
                config.training.val_batch_size,
                device,
                optimizer,
                criterion,
            )
            
            # Log metrics
            logger.log_fold_training(epoch, fold_metrics, fold_stats, avg_train_loss, avg_val_loss)
        else:
            avg_train_loss, avg_val_loss = train_epoch(
                model, device, datasets, optimizer, criterion, config.training.train_batch_size, config.training.val_batch_size
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
            
            # Save best model checkpoint when we find a new best
            if best_model_path is not None:
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': best_loss,
                }, best_model_path)
                print(f"Saved new best model checkpoint to: {best_model_path}")
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

        # Save checkpoint at the end of each epoch
        checkpoint = {
            'epoch': epoch + 1,  # Save next epoch to resume from
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_val_loss,
            'best_loss': best_loss,
            'patience_counter': patience_counter,
            'best_epoch': best_epoch,
            'best_model_state': best_model_state,
            'timestamp': current_time
        }
        save_checkpoint(checkpoint, checkpoint_path, config)

    # Load the best model state before final evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)

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
