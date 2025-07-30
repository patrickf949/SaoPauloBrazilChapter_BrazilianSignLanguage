import numpy as np
from typing import List, Dict, Any, Callable

def uniform_sampling(num_frames: int, params: Dict[str, Any]) -> List[int]:
    """
    Sample frames uniformly from the video.

    Args:
        num_frames (int): Total number of frames in the video.
        params (Dict[str, Any]): Sampling parameters (e.g., num_samples).

    Returns:
        List[int]: List of frame indices.
    """
    num_samples = params.get("num_samples", 30)
    if num_frames <= num_samples:
        return list(range(num_frames))
    step = num_frames / num_samples
    indices = [int(i * step) for i in range(num_samples)]
    return indices

def multiple_uniform_sampling(num_frames: int, params: Dict[str, Any]) -> List[List[int]]:
    """
    Sample frames uniformly from multiple segments of the video.

    Args:
        num_frames (int): Total number of frames in the video.
        params (Dict[str, Any]): Parameters including num_segments and num_samples_per_segment.

    Returns:
        List[List[int]]: List of lists, each containing frame indices for one segment.
    """
    num_segments = params.get("num_segments", 3)
    num_samples_per_segment = params.get("num_samples_per_segment", 10)
    if num_frames < num_segments:
        return [list(range(num_frames))]

    segment_length = num_frames // num_segments
    all_indices = []
    for i in range(num_segments):
        start = i * segment_length
        end = (i + 1) * segment_length if i < num_segments - 1 else num_frames
        segment_frames = end - start
        if segment_frames <= num_samples_per_segment:
            indices = list(range(start, end))
        else:
            step = segment_frames / num_samples_per_segment
            indices = [start + int(j * step) for j in range(num_samples_per_segment)]
        all_indices.append(indices)
    return all_indices

def get_sampling_function(method: str) -> Callable:
    """
    Return the appropriate frame sampling function based on the method name.

    Args:
        method (str): Name of the sampling method.

    Returns:
        Callable: Function to perform frame sampling.

    Raises:
        ValueError: If the method is not recognized.
    """
    sampling_methods = {
        "uniform": uniform_sampling,
        "multiple_uniform": multiple_uniform_sampling
    }
    if method not in sampling_methods:
        raise ValueError(f"Unknown sampling method: {method}")
    return sampling_methods[method]