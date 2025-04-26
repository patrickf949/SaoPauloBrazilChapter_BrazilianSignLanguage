import yaml
from typing import Union, Dict, Any
import numpy as np
import torch
import random
import importlib


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


def load_config(config: Union[str, Dict], config_name: str) -> Dict:
    if isinstance(config, str):
        with open(config, "r") as f:
            return yaml.safe_load(f)
    elif isinstance(config, dict):
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
