import os
from datetime import datetime
from typing import Optional, Tuple
from omegaconf import DictConfig, OmegaConf

def load_base_paths() -> DictConfig:
    """
    Load base paths from dataset config.
    
    Returns:
        DictConfig containing base paths
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
    """
    paths = load_base_paths()
    
    # Construct full paths
    landmarks_dir = os.path.join(paths.preprocessed_base, "landmarks", data_version)
    metadata_path = os.path.join(paths.metadata_base, f"landmarks_metadata_{data_version}_training.csv")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    return landmarks_dir, metadata_path

def generate_experiment_filename(config: DictConfig, prefix: str = "", suffix: str = "") -> str:
    """
    Generate a standardized filename for experiment artifacts.
    This function can be modified to change how filenames are structured.
    
    Args:
        config: The experiment configuration
        prefix: Optional prefix to add before the timestamp
        suffix: Optional suffix to add after the metadata
    
    Returns:
        A filename string in the format: [prefix]_timestamp_[metadata]_[suffix]
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add metadata components - modify this section to change what goes into the filename
    metadata_parts = []
    if config.general.experiment_name:
        metadata_parts.append(config.general.experiment_name)
    if config.model.class_name.split('.')[-1]:  # Get the model class name without the full path
        metadata_parts.append(config.model.class_name.split('.')[-1])
    
    # Combine all parts
    filename_parts = [timestamp]
    if metadata_parts:
        filename_parts.append('_'.join(metadata_parts))
    if suffix:
        filename_parts.append(suffix)
    
    # Combine with prefix if provided
    final_name = '_'.join(part for part in filename_parts if part)
    if prefix:
        final_name = f"{prefix}_{final_name}"
    
    return final_name

def get_save_paths(config: DictConfig) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Generate paths for saving experiment artifacts.
    
    Args:
        config: The experiment configuration
        
    Returns:
        Tuple of (log_path, best_model_path, final_model_path)
        Returns None for each path if saving is disabled
    """
    # If saving is disabled, return None for all paths
    if not config.general.enable_saving:
        return None, None, None
        
    paths = load_base_paths()
    
    # Generate base filename
    base_filename = generate_experiment_filename(config)
    
    # Create full paths
    log_path = os.path.join(paths.logs_base, f"{base_filename}.csv")
    best_model_path = os.path.join(paths.models_base, f"{base_filename}_best.pt")
    final_model_path = os.path.join(paths.models_base, f"{base_filename}_final.pt")
    
    # Ensure directories exist
    os.makedirs(paths.logs_base, exist_ok=True)
    os.makedirs(paths.models_base, exist_ok=True)
    
    return log_path, best_model_path, final_model_path 