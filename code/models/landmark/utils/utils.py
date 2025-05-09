import yaml
from typing import Union, Dict, Any, List
import numpy as np
import torch
import random
import importlib
from omegaconf import DictConfig, OmegaConf


def check_mode(mode: str):
    if mode not in ["2D", "3D"]:
        raise ValueError("Parameter 'mode' must be equal '2D' or '3D'")


def check_landmark_type(landmark_type: str):
    if landmark_type not in ["hand", "pose"]:
        raise ValueError("Parameter 'landmark_type' must be equal 'hand' or 'pose'")


def check_angle_type(angle_type: str):
    if angle_type not in ["rad", "grad", "func", "normalized_rad", "shifted_rad"]:
        raise ValueError(
            "Parameter 'angle_type' should be one of ['rad', 'grad', 'func', 'normalized_rad', 'shifted_rad']"
        )


def check_distance_type(distance_type: str):
    if distance_type not in ["dist", "normalized_dist", "shifted_dist"]:
        raise ValueError(
            "Parameter 'distance_type' should be one of ['dist', 'normalized_dist', 'shifted_dist']"
        )


def check_difference_type(diff_type: str):
    if diff_type not in ["diff", "normalized_diff"]:
        raise ValueError("Parameter 'diff_type' must be 'diff' or 'normalized_diff'")


def load_config(config: Union[str, Dict, DictConfig], config_name: str) -> Dict:
    if isinstance(config, str):
        with open(config, "r") as f:
            return yaml.safe_load(f)
    elif isinstance(config, dict) or isinstance(config, DictConfig):
        return config
    else:
        raise ValueError(
            f"Parameter {config_name} must be either a file path or dictionary."
        )


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)

def set_config_param(config: Union[DictConfig, dict], key_path: str, value: any) -> None:
    if isinstance(config, dict):
        config = OmegaConf.create(config)

    # Remember current struct state
    original_struct = OmegaConf.is_struct(config)
    OmegaConf.set_struct(config, False)

    # Set the value
    OmegaConf.update(config, key_path, value, merge=False)

    # Restore original struct state
    OmegaConf.set_struct(config, original_struct)

def minmax_scale_single(
    value: Union[float, int],
    input_max: float,
    input_min: float = 0,
    output_range: List[int] = [0, 1],
) -> float:
    """
    Scale a single value to a specified range using min-max scaling.
    This is useful when you already know the min/max of your data range
    and want to scale individual values.
    
    Args:
        value: Single value to scale
        data_min: Minimum value of the input data range
        data_max: Maximum value of the input data range
        scale_range: List of [min, max] values for output range, default [0,1]
        clip: Whether to clip values outside the range
        
    Returns:
        Scaled value as float
    """
    if len(output_range) != 2:
        raise ValueError("output_range must be a list of two integers [min, max]")
        
    min_val, max_val = output_range
    
    # Scale to desired range
    scaled = (value - input_min) / (input_max - input_min) * (max_val - min_val) + min_val
    
    scaled = max(min_val, min(max_val, scaled))
    
    return float(scaled)

def minmax_scale_series(
    values: Union[np.ndarray, torch.Tensor],
    input_max: float,
    input_min: float = 0,
    output_range: List[int] = [0, 1],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Scale values to a specified range using min-max scaling.
    
    Args:
        values: Input array or tensor to scale
        output_range: List of [min, max] values for output range, default [0,1]
        
    Returns:
        Scaled values in the same format as input (numpy array or torch tensor)
    """
    if len(output_range) != 2:
        raise ValueError("output_range must be a list of two integers [min, max]")
        
    # Convert to numpy for processing
    if isinstance(values, torch.Tensor):
        values_np = values.detach().cpu().numpy()
        is_tensor = True
    else:
        values_np = values
        is_tensor = False
        
    # Apply scaling to each value
    scaled = np.array([
        minmax_scale_single(
            float(x), input_max, input_min,
            output_range
        )
        for x in values_np.flatten()
    ]).reshape(values_np.shape)
    
    # Convert back to tensor if input was tensor
    if is_tensor:
        scaled = torch.from_numpy(scaled).to(values.device)
    
    return scaled