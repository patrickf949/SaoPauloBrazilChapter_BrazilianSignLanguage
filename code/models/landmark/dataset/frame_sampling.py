from typing import List, Dict, Any, Callable
import numpy as np


def uniform_sampling(num_frames: int, params: Dict[str, Any]) -> List[List[int]]:
    """Sample frames at uniform intervals.
    
    Required params:
        frames_per_sample (int): Number of frames to select
        num_samples (int, optional): Number of samples to generate, defaults to 1
        min_spacing (int, optional): Minimum frames between selected frames
    """
    required = {'frames_per_sample'}
    if not all(p in params for p in required):
        raise ValueError(f"Missing required parameters: {required - set(params.keys())}")
        
    frames_per_sample = params['frames_per_sample']
    num_samples = params.get('num_samples', 1)
    min_spacing = params.get('min_spacing')
    
    samples = []
    spacing = (num_frames - 1) / (frames_per_sample - 1)
    
    for i in range(num_samples):
        if min_spacing and spacing < min_spacing:
            break
            
        offset = (i * spacing / num_samples) if i > 0 else 0
        indices = list(np.linspace(offset, num_frames - 1, frames_per_sample, dtype=int))
        if indices[-1] < num_frames:
            samples.append(indices)
            
    return samples if samples else [list(np.linspace(0, num_frames - 1, frames_per_sample, dtype=int))]


def random_sampling(num_frames: int, params: Dict[str, Any]) -> List[List[int]]:
    """Sample random frames with optional minimum spacing.
    
    Required params:
        frames_per_sample (int): Number of frames per sample
        min_spacing (int, optional): Minimum frames between selected frames
        num_samples (int, optional): Number of samples to generate, defaults to 1
    """
    required = {'frames_per_sample'}
    if not all(p in params for p in required):
        raise ValueError(f"Missing required parameters: {required - set(params.keys())}")
        
    frames_per_sample = params['frames_per_sample']
    min_spacing = params.get('min_spacing')
    num_samples = params.get('num_samples', 1)
    
    def get_spaced_random_sample():
        if min_spacing is None:
            # Simple random sampling if no spacing constraint
            return sorted(np.random.choice(num_frames, frames_per_sample, replace=False))
            
        # With spacing constraint
        indices = []
        attempts = 0
        max_attempts = 100
        
        while len(indices) < frames_per_sample and attempts < max_attempts:
            if not indices:
                # First frame can be anywhere
                idx = np.random.randint(0, num_frames - (frames_per_sample - 1) * min_spacing)
                indices.append(idx)
            else:
                # Next frame must respect spacing from previous
                last_idx = indices[-1]
                min_idx = last_idx + min_spacing
                max_idx = num_frames - (frames_per_sample - len(indices) - 1) * min_spacing
                
                if min_idx >= max_idx:
                    # Can't satisfy spacing constraint
                    indices = []
                    attempts += 1
                    continue
                    
                idx = np.random.randint(min_idx, max_idx)
                indices.append(idx)
                
        return sorted(indices) if len(indices) == frames_per_sample else None

    samples = []
    for _ in range(num_samples):
        sample = get_spaced_random_sample()
        if sample is not None:
            samples.append(sample)
            
    # Fallback to uniform sampling if we couldn't generate valid samples
    return samples if samples else uniform_sampling(num_frames, {'frames_per_sample': frames_per_sample})


def exhaustive_sampling(num_frames: int, params: Dict[str, Any]) -> List[List[int]]:
    """Sample all possible frame sequences without replacement.
    
    Required params:
        frames_per_sample (int): Number of frames per sequence
        min_spacing (int, optional): Minimum frames between selected frames
    """
    required = {'frames_per_sample'}
    if not all(p in params for p in required):
        raise ValueError(f"Missing required parameters: {required - set(params.keys())}")
        
    frames_per_sample = params['frames_per_sample']
    min_spacing = params.get('min_spacing', 1)  # Default to 1 to avoid duplicates
    
    samples = []
    used_frames = set()
    
    while len(used_frames) < num_frames:
        # Find available frames
        available = [i for i in range(num_frames) if i not in used_frames]
        if len(available) < frames_per_sample:
            break
            
        # Try to find a valid sequence
        sequence = []
        for _ in range(frames_per_sample):
            valid_frames = [
                f for f in available 
                if not sequence or f - sequence[-1] >= min_spacing
            ]
            if not valid_frames:
                break
            frame = np.random.choice(valid_frames)
            sequence.append(frame)
            available.remove(frame)
            
        if len(sequence) == frames_per_sample:
            samples.append(sorted(sequence))
            used_frames.update(sequence)
            
    return samples if samples else uniform_sampling(num_frames, {'frames_per_sample': frames_per_sample})


SAMPLING_METHODS = {
    "uniform": uniform_sampling,
    "random": random_sampling,
    "exhaustive": exhaustive_sampling,
}


def get_sampling_function(method: str) -> Callable:
    """Get a frame sampling function.
    
    Args:
        method (str): Name of the sampling method to use
        
    Returns:
        callable: Function that takes (num_frames: int, params: Dict[str, Any]) 
                 and returns List[List[int]] (list of samples)
    """
    if method not in SAMPLING_METHODS:
        raise ValueError(f"Unknown sampling method: {method}. Available methods: {list(SAMPLING_METHODS.keys())}")
        
    return SAMPLING_METHODS[method] 