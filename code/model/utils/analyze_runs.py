import os
import pandas as pd
from omegaconf import OmegaConf
import json
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from model.dataset.landmark_dataset import LandmarkDataset
from model.utils.inference_then_evaluate import evaluate_detailed
from model.utils.utils import load_obj

def get_results_from_training_log(training_log):
    total_epochs = len(training_log)
    max_epoch = training_log['epoch'].max()
    if max_epoch != total_epochs:
        print(f"Warning: Training log has {total_epochs} epochs, but the last epoch is {max_epoch}")

    final_epoch = training_log['epoch'].iloc[-1]
    final_train_loss = training_log['avg_train_loss'].iloc[-1]
    final_val_loss = training_log['avg_val_loss'].iloc[-1]

    # Get best model
    best_model_idx = training_log['avg_val_loss'].idxmin()
    best_epoch = training_log.loc[best_model_idx, 'epoch']
    best_train_loss = training_log.loc[best_model_idx, 'avg_train_loss']
    best_val_loss = training_log.loc[best_model_idx, 'avg_val_loss']

    results = {
        'total_epochs': total_epochs,
        'best_epoch': best_epoch,
        'best_train_loss': best_train_loss,
        'best_val_loss': best_val_loss,
        'final_epoch': final_epoch,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
    }

    return results

def get_results_from_config(config):
    model = config['model']['class_name'].split('.')[-1]
    position = 'position' in config['features'].keys()
    angles = 'angles' in config['features'].keys()
    differences = 'differences' in config['features'].keys()
    distances = 'distances' in config['features'].keys()
    metadata = 'metadata' in config['features'].keys()
    p_rotate = config['augmentation']['train']['rotate']['p']
    p_noise = config['augmentation']['train']['noise']['p']

    results = {
        'model': model,
        'position': position,
        'angles': angles,
        'differences': differences,
        'distances': distances,
        'metadata': metadata,
        'p_rotate': p_rotate,
        'p_noise': p_noise,
    }
    return results

def make_results_row(run_dir, run_name):
    results = {
        'run_name': run_name,
    }
    training_log = pd.read_csv(os.path.join(run_dir, run_name, "training_log.csv"))
    config = OmegaConf.load(os.path.join(run_dir, run_name, "config.yaml"))
    results.update(get_results_from_training_log(training_log))
    results.update(get_results_from_config(config))
    return results

def load_checkpoint(filepath: str, device: str = "cuda") -> dict:
    """Load training checkpoint from file.
    
    Args:
        filepath: Path to checkpoint file
        
    Returns:
        Dictionary containing checkpoint data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint found at {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    print(f"Loaded checkpoint from: {filepath}")
    return checkpoint

def load_model(model_path: str, config_path: str, device: str = "cuda"):
    config = OmegaConf.load(config_path)
    checkpoint = load_checkpoint(model_path, device)
    model = load_obj(config.model.class_name)(**config.model.params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model

def test_model(model_path: str, config_path: str, device: str = "cuda"):
    """Test model on data.
    
    Args:
        model_path: Path to model file
        data_path: Path to data file
        device: Device to use for testing
    """
    model = load_model(model_path, config_path, device)
    config = OmegaConf.load(config_path)
    test_dataset = LandmarkDataset(config.dataset, config.features, config.augmentation, "test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    json_path = os.path.join(config.dataset.paths.metadata_base, "label_encoding.json")
    with open(json_path, "r", encoding="utf-8") as f:
        label_encoding = json.load(f)

    criterion = nn.CrossEntropyLoss()
    # eval by sample
    sample_metrics, cm_df = evaluate_detailed(
        model, 
        test_loader, 
        device=device, 
        class_names=label_encoding,
        ensemble_strategy=None,
        return_confidence=True, 
        criterion=criterion
    )
    return sample_metrics, cm_df