"""Utilities for analyzing datasets and their features."""
import time

verbose = False

def analyze_dataset_features(dataset):
    """Perform comprehensive analysis of the dataset features.
    
    Args:
        dataset: LandmarkDataset instance
        
    Returns:
        int: Number of features in the dataset
    """
    start_time = time.time()
    
    # Initialize analysis variables
    feature_dims = set()
    feature_ranges = {}
    timing_stats = {
            'index_calculation': 0.0,
            'frame_loading': 0.0,
            'sample_selection': 0.0,
            'metadata_lookup': 0.0,
            'feature_processing': 0.0,
            'label_creation': 0.0,
            'analysis_time': 0.0,
        }
    
    # Single pass through all samples in the dataset for all analysis
    for sample_idx, (features, _, timing) in enumerate(dataset):
        analysis_start_time = time.time()
        # Check dimensions
        feature_dims.add(features.shape[1])
        if len(feature_dims) > 1:
            raise ValueError(f"Inconsistent feature dimensions found: {feature_dims}, sample index: {sample_idx}")
        
        # Initialize range tracking with correct number of features using first sample
        if not feature_ranges:
            feature_ranges = {feature_idx: {'min': float('inf'), 'max': float('-inf')} 
                            for feature_idx in range(features.shape[1])}
        
        # Analyze feature ranges for each feature (of each sample)
        for feature_idx in range(features.shape[1]):
            feature_min = features[:, feature_idx].min().item()
            feature_max = features[:, feature_idx].max().item()
            
            # Update global min/max for this feature
            feature_ranges[feature_idx]['min'] = min(feature_ranges[feature_idx]['min'], feature_min)
            feature_ranges[feature_idx]['max'] = max(feature_ranges[feature_idx]['max'], feature_max)

        # for key, value in timing.items(): 
        #     timing_stats[key] += value
        timing_stats['analysis_time'] += time.time() - analysis_start_time
    
    analysis_time = time.time() - start_time
    n_features = feature_dims.pop()

    # Identify constant features (always 0 or always 1)
    const_0_idx = [i for i, range_info in feature_ranges.items() 
                   if range_info['min'] == 0 and range_info['max'] == 0]
    const_1_idx = [i for i, range_info in feature_ranges.items() 
                   if range_info['min'] == 1 and range_info['max'] == 1]
    
    # Identify variable features in different ranges, excluding constant features
    var_0_1_idx = [i for i, range_info in feature_ranges.items() 
                   if range_info['min'] >= 0 and range_info['max'] <= 1 
                   and i not in const_0_idx and i not in const_1_idx]
    
    var_neg1_1_idx = [i for i, range_info in feature_ranges.items() 
                      if range_info['min'] >= -1 and range_info['max'] <= 1
                      and i not in const_0_idx and i not in const_1_idx]
    
    # Features that are in [-1,1] but NOT in [0,1]
    var_neg1_1_only_idx = [i for i in var_neg1_1_idx 
                          if i not in var_0_1_idx]
    
    # Features that are not in any of the standard ranges
    other_idx = [i for i, _range_info_ in feature_ranges.items() 
                 if i not in var_neg1_1_idx 
                 and i not in const_0_idx and i not in const_1_idx]

    n_const_0 = len(const_0_idx)
    n_const_1 = len(const_1_idx)
    n_var_0_1 = len(var_0_1_idx)
    n_var_neg1_1 = len(var_neg1_1_idx)
    n_var_neg1_1_only = len(var_neg1_1_only_idx)
    n_other = len(other_idx)

    # Verify total number of features
    if not (n_const_0 + n_const_1 + n_var_0_1 + n_var_neg1_1_only + n_other == n_features):
        raise ValueError(f"Inconsistent number of features: {n_const_0} + {n_const_1} + {n_var_0_1} + {n_var_neg1_1_only} + {n_other} != {n_features}")
    
    # Verify that var_neg1_1 is the union of var_0_1 and var_neg1_1_only
    if not (n_var_0_1 + n_var_neg1_1_only == n_var_neg1_1):
        raise ValueError(f"Inconsistent number of features in [-1,1] range: {n_var_0_1} + {n_var_neg1_1_only} != {n_var_neg1_1}")
    
    # Verify that constant features are not being double counted
    if not (n_const_0 + n_const_1 + n_var_0_1 + n_var_neg1_1_only + n_other == n_var_neg1_1 + n_const_0 + n_const_1 + n_other):
        raise ValueError(f"Double counting detected in feature categories")
    
    print("\nDataset Features Analysis:")
    print(f"- feature dimensions: {n_features} (consistent across all samples)")
    print(f"- pass through dataset completed in {analysis_time:.2f} seconds ({sample_idx} samples, at {analysis_time/sample_idx:.2f} seconds/sample)")
    print(f"- timing breakdown (with % of total time):")
    timing_stats['total'] = sum(timing_stats.values())
    for key, value in timing_stats.items():
        print(f"\t- {key}: {value:.2f}s ({value/analysis_time*100:.1f}%)")
    
    # Print results
    print("\nFeature Range Analysis:")
    print(f"total features: {n_features}")
    print(f"- features with constant value 0: {n_const_0}")
    print(f"- features with constant value 1: {n_const_1}")
    print(f"- features in range [0,1]: {n_var_0_1}")
    print(f"- features in range [-1,1]: {n_var_neg1_1_only}")
    print(f"- features in other ranges: {n_other}")
    
    # Print details of features not in standard ranges
    print("\nFeature Range Details:")
    if n_other > 0:
        print(f"- {n_other} features in other ranges:")
        for i in other_idx:
            print(f"- feature {i}: [{feature_ranges[i]['min']:.3f}, {feature_ranges[i]['max']:.3f}]")
    else:
        print("- none (all features in standard ranges)")

    if verbose:
        print(f"- features in range [0,1]:")
        for i in var_0_1_idx:
            print(f"\t- feature {i}: [{feature_ranges[i]['min']:.3f}, {feature_ranges[i]['max']:.3f}]")
        print(f"- features in range [-1,1]:")
        for i in var_neg1_1_only_idx:
            print(f"\t- feature {i}: [{feature_ranges[i]['min']:.3f}, {feature_ranges[i]['max']:.3f}]")
        print(f"- features in other ranges:")
        for i in other_idx:
            print(f"\t- feature {i}: [{feature_ranges[i]['min']:.3f}, {feature_ranges[i]['max']:.3f}]")
    return n_features

def indices_info_string_basic(indices):
    """Convert a list of indices into a basic readable range string.
    
    Args:
        indices: List of integers
        
    Returns:
        str: Formatted string like "0-2, 5-7, 12, 15"
    """
    if not indices:
        return "none"
        
    # Sort indices to ensure proper range detection
    indices = sorted(indices)
    ranges = []
    start = indices[0]
    prev = start
    
    for curr in indices[1:]:
        if curr != prev + 1:
            # End of a range
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
            start = curr
        prev = curr
    
    # Handle the last range
    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{prev}")
        
    return ", ".join(ranges)

def indices_info_string_patterns(indices):
    """Convert a list of indices into a pattern-aware readable range string.
    
    Args:
        indices: List of integers
        
    Returns:
        str: Formatted string like "0-22 (even numbers), 24-84 (even numbers)"
    """
    if not indices:
        return "none"
        
    # Sort indices to ensure proper range detection
    indices = sorted(indices)
    ranges = []
    start = indices[0]
    prev = start
    
    def is_even_sequence(nums):
        return all(n % 2 == 0 for n in nums)
        
    def is_odd_sequence(nums):
        return all(n % 2 == 1 for n in nums)
        
    def is_consecutive(nums):
        return all(nums[i] + 1 == nums[i+1] for i in range(len(nums)-1))
        
    def get_pattern_description(nums):
        if is_even_sequence(nums):
            return "even numbers"
        elif is_odd_sequence(nums):
            return "odd numbers"
        elif is_consecutive(nums):
            return "consecutive"
        return None
    
    current_range = [start]
    
    for curr in indices[1:]:
        # Check if current number continues the pattern
        if is_even_sequence(current_range) and curr == prev + 2:
            current_range.append(curr)
        elif is_odd_sequence(current_range) and curr == prev + 2:
            current_range.append(curr)
        elif is_consecutive(current_range) and curr == prev + 1:
            current_range.append(curr)
        else:
            # End of a range - check what pattern it had
            if len(current_range) > 1:
                pattern = get_pattern_description(current_range)
                if pattern:
                    ranges.append(f"{start}-{prev} ({pattern})")
                else:
                    ranges.append(f"{start}-{prev}")
            else:
                ranges.append(str(start))
            
            # Start new range
            start = curr
            current_range = [curr]
        prev = curr
    
    # Handle the last range
    if len(current_range) > 1:
        pattern = get_pattern_description(current_range)
        if pattern:
            ranges.append(f"{start}-{prev} ({pattern})")
        else:
            ranges.append(f"{start}-{prev}")
    else:
        ranges.append(str(start))
        
    return ", ".join(ranges)