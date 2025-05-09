"""Utilities for analyzing datasets and their features."""
import time

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
    
    # Single pass through dataset for all analysis
    for idx, (features, _) in enumerate(dataset):
        # Check dimensions
        feature_dims.add(features.shape[1])
        if len(feature_dims) > 1:
            raise ValueError(f"Inconsistent feature dimensions found: {feature_dims}, sample index: {idx}")
            
        # Initialize range tracking on first sample
        if not feature_ranges:
            feature_ranges = {i: {'min': float('inf'), 'max': float('-inf')} 
                            for i in range(features.shape[1])}
        
        # Analyze feature ranges
        for i in range(features.shape[1]):
            feature_min = features[:, i].min().item()
            feature_max = features[:, i].max().item()
            
            # Update global min/max for this feature
            feature_ranges[i]['min'] = min(feature_ranges[i]['min'], feature_min)
            feature_ranges[i]['max'] = max(feature_ranges[i]['max'], feature_max)
    
    analysis_time = time.time() - start_time
    
    n_features = feature_dims.pop()
    
    # Count features in each range
    in_range_0_1 = sum(1 for range_info in feature_ranges.values() 
                      if range_info['min'] >= 0 and range_info['max'] <= 1)
    in_range_neg1_1 = sum(1 for range_info in feature_ranges.values() 
                         if range_info['min'] >= -1 and range_info['max'] <= 1)
    # Features in [-1,1] but not in [0,1]
    in_range_neg1_1_only = in_range_neg1_1 - in_range_0_1
    # Features not in either range
    other_ranges = n_features - in_range_neg1_1
    
    print("\nDataset Features Analysis:")
    print(f"- feature dimensions: {n_features} (consistent across all samples)")
    print(f"- pass through dataset completed in {analysis_time:.2f} seconds ({idx} samples, at {analysis_time/idx:.2f} seconds/sample)")
    
    # Print results
    print("\nFeature Range Analysis:")
    print(f"- total features: {n_features}")
    print(f"- features in range [0,1]: {in_range_0_1}")
    print(f"- features in range [-1,1]: {in_range_neg1_1_only}")
    print(f"- features in other ranges: {other_ranges}")
    
    # Print details of features not in standard ranges
    print("\nFeatures outside standard ranges:")
    found_outside = False
    for i, range_info in feature_ranges.items():
        if not (range_info['min'] >= -1 and range_info['max'] <= 1):
            found_outside = True
            print(f"- feature {i}: [{range_info['min']:.3f}, {range_info['max']:.3f}]")
    if not found_outside:
        print("- none (all features in standard ranges)")
    
    return n_features 