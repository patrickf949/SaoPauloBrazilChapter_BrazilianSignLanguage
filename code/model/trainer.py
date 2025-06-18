import csv
import json
import os
import time
from datetime import datetime

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from model.dataset.dataloader_functions import collate_func_pad
from model.dataset.landmark_dataset import LandmarkDataset
from model.dataset.prepare_data import prepare_training_metadata, prepare_landmark_arrays
from model.features.features_analysis import analyze_dataset_features

from model.utils.logging import TrainingLogger
from model.utils.path_utils import get_save_paths, get_data_paths, load_base_paths
from model.utils.train import train_epoch, train_epoch_fold
from model.utils.utils import load_obj, set_config_param
from model.utils.inference import InferenceEngine
from model.utils.evaluate import EvaluationMetrics

# Ensure base output directories exist
paths = load_base_paths()
os.makedirs(paths.logs_base, exist_ok=True)
os.makedirs(os.path.join(paths.logs_base, "runs"), exist_ok=True)

def save_checkpoint(state: dict, filepath: str) -> None:
    """Save training checkpoint to file.
    
    Args:
        state: Dictionary containing checkpoint data
        filepath: Path to save checkpoint to
    """
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
    # ----- Setup paths and resume state -----
    resume = config.training.get('resume', False)
    run_dir = config.training.get('run_dir', None)
    
    if resume and run_dir:
        # Load checkpoint and config from previous run
        checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
        checkpoint = load_checkpoint(checkpoint_path)
        
        # Load saved config but preserve resume settings
        config_path = os.path.join(run_dir, "config.yaml")
        saved_config = OmegaConf.load(config_path)
        saved_config.training.resume = resume
        saved_config.training.run_dir = run_dir
        config = saved_config
        
        # Restore training state
        current_time = checkpoint['timestamp']
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        patience_counter = checkpoint['patience_counter']
        best_epoch = checkpoint['best_epoch']
        best_model_state = checkpoint['best_model_state']

        print(f"Resuming training using files in {run_dir}")
        print(f"\t Starting from epoch {start_epoch}, with patience counter at {patience_counter}")
    else:
        # Initialize new training state
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        start_epoch = 0
        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        best_epoch = 0

    # Get paths for saving artifacts
    log_path, checkpoint_path, best_model_path, config_path, dataset_path, best_dataset_path = get_save_paths(config)

    # ----- Data preparation -----
    # Get dataset
    # datasets = get_dataset(config)
    _, label_mapping_path = prepare_training_metadata(config.dataset.data_version, return_paths=True)
    verify_dataset = LandmarkDataset(
        config.dataset, config.features, config.augmentation, None, seed=42
    )

    # For new training runs, analyze features and update config with actual input size
    if not resume:
        n_features = analyze_dataset_features(verify_dataset)
        set_config_param(config.model.params, "input_size", n_features)
        OmegaConf.save(config, config_path)

    # ----- Model initialization -----
    device = config.training.device
    num_epochs = config.training.num_epochs
    
    # Initialize model
    model = load_obj(config.model.class_name)(**config.model.params)
    model.to(device)

    # Initialize training components
    optimizer = load_obj(config.optimizer.class_name)(
        model.parameters(), **config.optimizer.params
    )
    scheduler = load_obj(config.scheduler.class_name)(
        optimizer, **config.scheduler.params
    )
    criterion = nn.CrossEntropyLoss()

    # Restore model state if resuming
    if resume and run_dir:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # ----- Logging setup -----
    k_folds = config.training.k_folds if config.training.type == "cross_validation" else None
    logger = TrainingLogger(os.path.dirname(log_path), log_path, k_folds, resume)

    # ----- Training loop -----
    for epoch in range(start_epoch, num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        if config.training.type == "cross_validation":
            # Sample a new dataset for this epoch
            train_dataset = LandmarkDataset(
                config.dataset, config.features, config.augmentation, "train", seed=epoch
            )
            # Modified to get per-fold metrics
            avg_train_loss, avg_val_loss, fold_metrics, fold_stats = train_epoch_fold(
                epoch,
                config.training.k_folds,
                model,
                train_dataset,
                config.training.train_batch_size,
                config.training.val_batch_size,
                device,
                optimizer,
                criterion,
            )
            
            # Log metrics
            logger.log_fold_training(epoch, fold_metrics, fold_stats, avg_train_loss, avg_val_loss, current_lr)
        else:
            # Sample a new dataset for this epoch
            train_dataset = LandmarkDataset(
                config.dataset, config.features, config.augmentation, "train", seed=epoch
            )
            val_dataset = LandmarkDataset(
                config.dataset, config.features, config.augmentation, "val", seed=epoch
            )
            datasets = {
                "train_dataset": train_dataset,
                "val_dataset": val_dataset,
            }
            avg_train_loss, avg_val_loss = train_epoch(
                model, device, datasets, optimizer, criterion, config.training.train_batch_size, config.training.val_batch_size
            )
            
            # Log metrics
            logger.log_standard_training(epoch, avg_train_loss, avg_val_loss, current_lr)
 
        print(
            f"Epoch Average Results:\n\tTrain Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model checkpoint when we find a new best
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, best_model_path)
            print(f"Saved new best model checkpoint to: {best_model_path}")
            # save best dataset
            train_dataset.save(best_dataset_path, epoch)
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
        save_checkpoint(checkpoint, checkpoint_path)
        # save dataset
        start_time = time.time()
        train_dataset.save(dataset_path, epoch)
        end_time = time.time()
        print(f"Time taken to save dataset: {end_time - start_time} seconds")

    # Load the best model state before final evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)

    # ----- evaluate -----
    # Final evaluation using inference engine with detailed metrics
    # Use a reasonable batch size for testing:
    # - Large enough for efficient processing
    # - Small enough to avoid memory issues
    # - If using ensemble prediction, samples from the same video will be automatically grouped
    test_dataset = LandmarkDataset(
        config.dataset, config.features, config.augmentation, "test", seed=42
    )
    with open(label_mapping_path, "r") as f:
        label_encoding = json.load(f)
    class_names = [label_encoding[str(i)] for i in range(len(label_encoding))]

    sample_inference = InferenceEngine(
        model=model,
        device='cpu',
        ensemble_strategy=None
    )
    sample_test_preds, sample_test_labels, sample_test_probs = sample_inference.predict(test_dataset, return_labels=True, return_full_probs=True)

    sample_test_eval = EvaluationMetrics(
        sample_test_preds,
        sample_test_labels,
        sample_test_probs,
        num_classes=len(class_names),
        class_names=class_names
    )
    
    metrics = {
        'sample_acc': sample_test_eval.accuracy,
        'sample_loss': sample_test_eval.get_loss(nn.CrossEntropyLoss()),
        'sample_topk_acc_2': sample_test_eval.get_topk_accuracy(2),
        'sample_topk_acc_3': sample_test_eval.get_topk_accuracy(3),
        'sample_topk_acc_4': sample_test_eval.get_topk_accuracy(4),
        'sample_topk_acc_5': sample_test_eval.get_topk_accuracy(5),
    }
    # Print results
    print("\n=== Evaluation Results (best model) ===")
    print(metrics)
    sample_test_eval.plot_confusion_matrix(save_path=os.path.join(os.path.dirname(log_path), "confusion_matrix.png"))
    # Save detailed results
    results_path = os.path.join(os.path.dirname(log_path), "evaluation_results.txt")
    with open(results_path, "w") as f:
        f.write("=== Evaluation Results ===\n\n")
        f.write("Overall Metrics:\n")
        for metric, value in metrics.items():
            if value is not None:
                f.write(f"{metric.capitalize()}: {value:.4f}\n")
        
        # if confidences:
        #     f.write("\n\nConfidence Statistics:\n")
        #     conf_values = list(confidences.values())
        #     f.write(f"Mean confidence: {np.mean(conf_values):.4f}\n")
        #     f.write(f"Std confidence: {np.std(conf_values):.4f}\n")
        #     f.write(f"Min confidence: {np.min(conf_values):.4f}\n")
        #     f.write(f"Max confidence: {np.max(conf_values):.4f}\n")
            
        #     # Write samples with lowest confidence
        #     f.write("\nLowest Confidence Predictions:\n")
        #     n_lowest = min(5, len(confidences))
        #     lowest_conf = sorted(confidences.items(), key=lambda x: x[1])[:n_lowest]
        #     for idx, conf in lowest_conf:
        #         sample_name = f"Sample {idx}"
        #         f.write(f"{sample_name}: {conf:.4f}\n")

    # Close logger
    logger.close()

    return metrics["sample_acc"], metrics.get("sample_loss", 0.0), best_epoch


if __name__ == "__main__":
    train()
