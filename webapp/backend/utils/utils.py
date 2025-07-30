import yaml
from typing import Union, Dict, Any, List
import numpy as np
import torch
import random
import importlib
from omegaconf import DictConfig, OmegaConf

def check_mode(mode: str):
    if mode not in ["2D", "3D"]:
        raise ValueError("Parameter 'mode' must be '2D' or '3D'")

def check_landmark_type(landmark_type: str):
    if landmark_type not in ["hand", "pose"]:
        raise ValueError("Parameter 'landmark_type' must be 'hand' or 'pose'")

def check_angle_type(angle_type: str):
    valid_types = [
        "rad", "grad",
        "normalized_rad", "shifted_rad",
        "func",
        "clockwise_rad", "clockwise_grad",
        "clockwise_normalized_rad", "clockwise_shifted_rad",
        "clockwise_func"
    ]
    if angle_type not in valid_types:
        raise ValueError(f"Parameter 'angle_type' must be one of {valid_types}")

def check_distance_type(distance_type: str):
    if distance_type not in ["dist", "normalized_dist", "shifted_dist"]:
        raise ValueError("Parameter 'distance_type' must be 'dist', 'normalized_dist', or 'shifted_dist'")

def check_difference_type(diff_type: str):
    if diff_type not in ["diff", "normalized_diff"]:
        raise ValueError("Parameter 'diff_type' must be 'diff' or 'normalized_diff'")

def load_config(config: Union[str, Dict, DictConfig], config_name: str) -> Dict:
    """Load YAML or dict config."""
    if isinstance(config, str):
        with open(config, "r") as f:
            return yaml.safe_load(f)
    elif isinstance(config, dict) or isinstance(config, DictConfig):
        return config
    else:
        raise ValueError(f"{config_name} must be a file path or dictionary.")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Dynamically import an object from a module path.
    Example: load_obj("my_module.MyClass")
    """
    obj_path_list = obj_path.rsplit(".", 1)
    module_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(module_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{module_path}`.")
    return getattr(module_obj, obj_name)

def set_config_param(config: Union[DictConfig, dict], key_path: str, value: Any) -> None:
    if isinstance(config, dict):
        config = OmegaConf.create(config)

    original_struct = OmegaConf.is_struct(config)
    OmegaConf.set_struct(config, False)
    OmegaConf.update(config, key_path, value, merge=False)
    OmegaConf.set_struct(config, original_struct)

def minmax_scale_single(
    value: Union[float, int],
    input_max: float,
    input_min: float = 0,
    output_range: List[int] = [0, 1],
) -> float:
    """Scale a single value to a range using min-max scaling."""
    if len(output_range) != 2:
        raise ValueError("output_range must be [min, max]")
    min_val, max_val = output_range
    scaled = (value - input_min) / (input_max - input_min) * (max_val - min_val) + min_val
    scaled = max(min_val, min(max_val, scaled))
    return float(scaled)

def minmax_scale_series(
    values: Union[np.ndarray, torch.Tensor],
    input_max: float,
    input_min: float = 0,
    output_range: List[int] = [0, 1],
) -> Union[np.ndarray, torch.Tensor]:
    """Scale an array or tensor using min-max scaling."""
    if len(output_range) != 2:
        raise ValueError("output_range must be [min, max]")
    min_val, max_val = output_range

    if isinstance(values, torch.Tensor):
        values_np = values.detach().cpu().numpy()
        is_tensor = True
    else:
        values_np = np.asarray(values, dtype=np.float64)
        is_tensor = False

    scaled = (values_np - input_min) / (input_max - input_min) * (max_val - min_val) + min_val
    scaled = np.clip(scaled, min_val, max_val)

    if is_tensor:
        scaled = torch.from_numpy(scaled).to(values.device)

    return scaled
