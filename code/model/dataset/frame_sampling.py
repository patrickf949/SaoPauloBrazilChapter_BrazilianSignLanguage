from typing import List, Dict, Any, Callable
import numpy as np


def uniform_sampling(num_frames: int, params: Dict[str, Any]) -> List[List[int]]:
    """Sample frames at uniform intervals.
    
    Required params:
        frames_per_sample (int): Number of frames to select
        num_samples (int, optional): Number of samples to generate, defaults to 1
        min_spacing (int, optional): Minimum frames between selected frames, defaults to 0
    """
    required = {'frames_per_sample'}
    if not all(p in params for p in required):
        raise ValueError(f"Missing required parameters: {required - set(params.keys())}")
        
    frames_per_sample = params['frames_per_sample']
    num_samples = params.get('num_samples', 1)
    min_spacing = params.get('min_spacing', 0)
    
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


def multiple_sampling_without_replacement_uniform(num_frames: int, params: Dict[str, Any]) -> List[List[int]]:
    """Sample multiple frames without replacement.
    
    Required params:
        frames_per_sample (int): Number of frames per sample
        num_samples_limit (int, optional): Number of samples to generate, defaults to no limit (all possible samples)
        replacement_rate (float, optional): The rate of replacement, defaults to 0.0. For example, if 0.1, 
            then 10% of sampled frames will be put back into the available pool.
        include_remaining (bool, optional): Whether to include the remaining frames in the last sample, defaults to True.
            If True and there are insufficient remaining frames for a complete sample, will combine remaining frames
            with additional sampled frames to create one final complete sample.
        seed (int, optional): Random seed for reproducible sampling, defaults to None.

    Returns:
        List[List[int]]: A list of lists of frame indices
    """
    # Initialize & check required parameters
    required = {'frames_per_sample'}
    if not all(p in params for p in required):
        raise ValueError(f"Missing required parameters: {required - set(params.keys())}")
        
    frames_per_sample = params['frames_per_sample']
    num_samples_limit = params.get('num_samples_limit', None)
    replacement_rate = params.get('replacement_rate', 0.0)
    include_remaining = params.get('include_remaining', True)
    seed = params.get('seed', None)

    if seed is not None:
        np.random.seed(seed)

    if replacement_rate < 0.0 or replacement_rate > 1.0:
        raise ValueError("Replacement rate must be between 0.0 and 1.0")
        
    if num_samples_limit is None:
        num_samples_limit = np.inf

    # Initialize available frames
    available_frames = list(range(num_frames))
    samples = []
    
    # Main sampling loop
    while len(samples) < num_samples_limit and len(available_frames) >= frames_per_sample:
        # Sample frames without replacement
        sample = list(np.random.choice(available_frames, size=frames_per_sample, replace=False))
        samples.append(sorted(sample))
        
        # Calculate how many frames to replace
        num_frames_to_replace = int(frames_per_sample * replacement_rate)
        
        if num_frames_to_replace > 0:
            # Randomly select frames to put back
            frames_to_replace = list(np.random.choice(sample, size=num_frames_to_replace, replace=False))
            frames_to_remove = [f for f in sample if f not in frames_to_replace]
            
            # Update available frames
            available_frames = [f for f in available_frames if f not in frames_to_remove]
            available_frames.extend(frames_to_replace)
        else:
            # Remove all sampled frames from available pool
            available_frames = [f for f in available_frames if f not in sample]
    
    # Handle remaining frames if enabled and we have some frames left
    if include_remaining and len(available_frames) > 0 and len(available_frames) < frames_per_sample:
        # Use all remaining frames
        final_sample = available_frames.copy()
        
        # Calculate how many additional frames we need
        frames_needed = frames_per_sample - len(final_sample)
        
        # Get pool of frames we can sample from (all frames except those in final_sample)
        available_for_final = [f for f in range(num_frames) if f not in final_sample]
        
        # Sample additional frames needed
        if frames_needed > 0 and available_for_final:
            additional_frames = list(np.random.choice(available_for_final, size=frames_needed, replace=False))
            final_sample.extend(additional_frames)
            
        if len(final_sample) == frames_per_sample:
            samples.append(sorted(final_sample))
            
    return samples


SAMPLING_METHODS = {
    "uniform": uniform_sampling,
    "random": random_sampling,
    "exhaustive": exhaustive_sampling,
    "multiple_uniform": multiple_sampling_without_replacement_uniform
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