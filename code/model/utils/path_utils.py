import os
from datetime import datetime
from typing import Optional, Tuple
from omegaconf import DictConfig, OmegaConf

def load_base_paths() -> DictConfig:
    """
    Load base paths from dataset config.
    
    Returns:
        DictConfig containing base paths with the following structure:
        - preprocessed_base: "data/preprocessed"
        - metadata_base: "modelling/metadata"
        - logs_base: "modelling/runs"
        - landmark_arrays_base: "modelling/landmark_arrays"
    """
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "dataset", "dataset.yaml")
    config = OmegaConf.load(config_path)
    return config.paths

def get_data_paths(data_version: str) -> tuple[str, str]:
    """
    Get standardized paths for data directory and metadata file.
    
    Args:
        data_version: Version string (e.g. 'v2')
        
    Returns:
        Tuple of (landmarks_dir, metadata_path)
        - landmarks_dir: preprocessed_base/landmarks/data_version
        - metadata_path: metadata_base/landmarks_metadata_data_version_training.csv
    """
    paths = load_base_paths()
    
    # Construct full paths
    landmarks_dir = os.path.join(paths.preprocessed_base, "landmarks", data_version)
    metadata_path = os.path.join(paths.metadata_base, f"landmarks_metadata_{data_version}_training.csv")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    return landmarks_dir, metadata_path

def generate_run_name(config: DictConfig) -> str:
    """
    Generate a descriptive run name including timestamp and relevant config info.
    
    Args:
        config: The experiment configuration
    
    Returns:
        A string in format: YYYYMMDD_HHMMSS_[model]_[experiment_name]
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get model name without full path
    model_name = config.model.class_name.split('.')[-1].replace('Classifier', '')
    
    # Build name components
    name_parts = [timestamp, model_name]
    
    # Add experiment name if provided
    if config.general.experiment_name:
        name_parts.append(config.general.experiment_name)
    if config.general.author:
        name_parts.append(config.general.author)
        
    return '_'.join(name_parts)

def get_save_paths(config: DictConfig) -> tuple[str, str, str, str]:
    """
    Generate paths for saving experiment artifacts.
    
    Args:
        config: The experiment configuration
        
    Returns:
        Tuple of (log_path, checkpoint_path, best_model_path, config_path)
    """
    paths = load_base_paths()
    
    # If resuming, use the existing run directory
    if config.training.get('resume', False) and config.training.get('run_dir'):
        run_dir = config.training.run_dir
    else:
        # Generate new run directory with descriptive name
        run_name = generate_run_name(config)
        run_dir = os.path.join(paths.logs_base, run_name)
    
    # Create run directory
    os.makedirs(run_dir, exist_ok=True)
    
    # Define paths for artifacts
    log_path = os.path.join(run_dir, "training_log.csv")
    checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
    best_model_path = os.path.join(run_dir, "best_model.pt")
    config_path = os.path.join(run_dir, "config.yaml")
    
    return log_path, checkpoint_path, best_model_path, config_path