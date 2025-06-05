import os
import csv
from typing import Dict, List, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir: str, log_path: Optional[str] = None, k_folds: Optional[int] = None, resume: bool = False):
        """Initialize the training logger.
        
        Args:
            log_dir: Directory for TensorBoard logs, should be under logs_base/runs/timestamp
            log_path: Path for CSV logs (optional), should be under logs_base/experiment_name.csv
            k_folds: Number of folds for cross-validation (optional)
            resume: Whether training is being resumed from checkpoint
        """
        # Initialize TensorBoard writer
        tensorboard_dir = os.path.join(log_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_dir)
        print(f"TensorBoard logs will be saved to: {tensorboard_dir}")
        
        # Write a test scalar to verify TensorBoard is working
        self.writer.add_scalar('test/initialization', 0.0, 0)
        self.writer.flush()
        
        # Initialize CSV writer if path provided
        self.csv_writer = None
        self.log_file = None
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            
            # Define CSV headers based on training type
            if k_folds:
                fieldnames = ["timestamp", "epoch", "avg_train_loss", "avg_val_loss", "learning_rate"]
                # Add per-fold metrics
                for fold in range(k_folds):
                    fieldnames.extend([
                        f"fold{fold}_train_loss",
                        f"fold{fold}_val_loss",
                        f"fold{fold}_train_samples",
                        f"fold{fold}_val_samples",
                        f"fold{fold}_train_groups",
                        f"fold{fold}_val_groups",
                        f"fold{fold}_train_samples_per_group",
                        f"fold{fold}_val_samples_per_group"
                    ])
                # Add average fold statistics
                fieldnames.extend([
                    "avg_train_samples",
                    "avg_val_samples",
                    "avg_train_groups",
                    "avg_val_groups",
                    "avg_train_samples_per_group",
                    "avg_val_samples_per_group"
                ])
            else:
                fieldnames = ["timestamp", "epoch", "train_loss", "val_loss", "learning_rate"]
            
            # Open file in append mode if resuming, write mode if new training
            mode = 'a' if resume else 'w'
            self.log_file = open(log_path, mode, newline="")
            self.csv_writer = csv.DictWriter(self.log_file, fieldnames=fieldnames)
            
            # Only write header if this is a new file
            if not resume:
                self.csv_writer.writeheader()
            
            print(f"CSV logs will be {'appended to' if resume else 'saved to'}: {log_path}")

    def log_fold_training(
        self,
        epoch: int,
        fold_metrics: List[Tuple[float, float]],
        fold_stats: List[Dict],
        avg_train_loss: float,
        avg_val_loss: float,
        learning_rate: float = None
    ) -> None:
        """Log metrics for cross-validation training."""
        # Calculate average statistics across folds
        avg_train_samples = sum(s['train_samples'] for s in fold_stats) / len(fold_stats)
        avg_val_samples = sum(s['val_samples'] for s in fold_stats) / len(fold_stats)
        avg_train_groups = sum(s['train_groups'] for s in fold_stats) / len(fold_stats)
        avg_val_groups = sum(s['val_groups'] for s in fold_stats) / len(fold_stats)
        avg_train_samples_per_group = sum(s['train_samples_per_group'] for s in fold_stats) / len(fold_stats)
        avg_val_samples_per_group = sum(s['val_samples_per_group'] for s in fold_stats) / len(fold_stats)

        # Log to TensorBoard
        # Log average losses
        self.writer.add_scalars('epoch_loss', {
            'train': avg_train_loss,
            'val': avg_val_loss
        }, epoch)
        
        # Log learning rate if provided
        if learning_rate is not None:
            self.writer.add_scalar('learning_rate', learning_rate, epoch)
        
        # Log all fold train losses on same plot
        fold_train_dict = {f'fold_{idx}': loss for idx, (loss, _) in enumerate(fold_metrics)}
        self.writer.add_scalars('fold_loss/train', fold_train_dict, epoch)
        
        # Log all fold validation losses on same plot
        fold_val_dict = {f'fold_{idx}': loss for idx, (_, loss) in enumerate(fold_metrics)}
        self.writer.add_scalars('fold_loss/val', fold_val_dict, epoch)

        # Force write to disk
        self.writer.flush()

        # Log to CSV if enabled - write once per epoch with all fold information
        if self.csv_writer:
            log_row = {
                "timestamp": datetime.now().strftime("%m-%d %H:%M"),
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
                "learning_rate": learning_rate,
                "avg_train_samples": avg_train_samples,
                "avg_val_samples": avg_val_samples,
                "avg_train_groups": avg_train_groups,
                "avg_val_groups": avg_val_groups,
                "avg_train_samples_per_group": avg_train_samples_per_group,
                "avg_val_samples_per_group": avg_val_samples_per_group
            }
            
            # Add per-fold data
            for fold_idx, ((fold_train_loss, fold_val_loss), fold_stat) in enumerate(zip(fold_metrics, fold_stats)):
                log_row.update({
                    f"fold{fold_idx}_train_loss": fold_train_loss,
                    f"fold{fold_idx}_val_loss": fold_val_loss,
                    f"fold{fold_idx}_train_samples": fold_stat['train_samples'],
                    f"fold{fold_idx}_val_samples": fold_stat['val_samples'],
                    f"fold{fold_idx}_train_groups": fold_stat['train_groups'],
                    f"fold{fold_idx}_val_groups": fold_stat['val_groups'],
                    f"fold{fold_idx}_train_samples_per_group": fold_stat['train_samples_per_group'],
                    f"fold{fold_idx}_val_samples_per_group": fold_stat['val_samples_per_group']
                })
            self.csv_writer.writerow(log_row)
            # Ensure the CSV is written to disk after each epoch
            self.log_file.flush()

    def log_standard_training(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float = None
    ) -> None:
        """Log metrics for standard training."""
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Log learning rate if provided
        if learning_rate is not None:
            self.writer.add_scalar('learning_rate', learning_rate, epoch)
        
        # Force write to disk
        self.writer.flush()
        
        # Log to CSV if enabled
        if self.csv_writer:
            self.csv_writer.writerow({
                "timestamp": datetime.now().strftime("%m-%d %H:%M"),
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate
            })
            # Ensure the CSV is written to disk after each epoch
            self.log_file.flush()

    def close(self) -> None:
        """Close all logging resources."""
        self.writer.close()
        if self.log_file:
            self.log_file.close() 