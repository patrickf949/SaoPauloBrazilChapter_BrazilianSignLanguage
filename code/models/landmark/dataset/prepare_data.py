import pandas as pd
import numpy as np
import os
from omegaconf import OmegaConf, DictConfig
from models.landmark.utils.utils import minmax_scale_series

def load_base_paths():
    """
    Load base paths from dataset config.
    
    Returns:
        DictConfig containing base paths
    """
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "dataset", "dataset.yaml")
    config = OmegaConf.load(config_path)
    return config.paths

def validate_filename_column(df: pd.DataFrame) -> pd.DataFrame:
    # Check if 'filename' column exists
    if 'filename' not in df.columns:
        raise ValueError("'filename' column not found in the DataFrame")
    
    # Make sure the filenames end with .npy
    df['filename'] = df['filename'].apply(lambda x: x.split('.')[0] + '.npy' if not x.endswith('.npy') else x)
    
    # Check if 'filename' column is unique
    if df['filename'].duplicated().any():
        raise ValueError("'filename' column contains duplicate values")
    
    # Make sure the filenames end with .npy
    if not all(filename.endswith('.npy') for filename in df['filename']):
        raise ValueError("All filenames must end with '.npy'")
    
    return df

def encode_label(df: pd.DataFrame) -> pd.DataFrame:
    label_mapping = {label: idx for idx, label in enumerate(set(df["label"]))}
    df["label_encoded"] = df["label"].map(label_mapping)
    return df

def train_test_split(df: pd.DataFrame) -> pd.DataFrame:
    # should already be sorted, but just in case
    df = df.sort_values(["label", "data_source"]).reset_index(drop=True)

    # 6 letters, 1 will be assigned to each of the 6 videos for each label
    letters = ["A", "B", "C", "D", "E", "F"]

    # Initialize an empty list to store the split groups
    dataset_split_groups = []

    # Iterate over each group of labels
    # (not changing anything in the df here, just looping to match the length)
    for label, _ in df.groupby("label"):
        # Shift the letters along by moving the first letter to the end ABCDEF -> BCDAEF
        letters = letters[1:] + letters[:1]
        # Append the split group letters to the list
        dataset_split_groups.extend(letters)

    # Add the split groups to the DataFrame
    df["dataset_split_group"] = dataset_split_groups

    # This dict will use 3/6 for train, 2/6 for val, 1/6 for test
    # If want to change the split, or which videos are used for each split, we can change this dict
    dataset_split_dict = {
        "A": "train",
        "B": "train",
        "C": "train",
        "D": "train",
        "E": "train",
        "F": "test",
    }

    # Map the split groups to the dataset splits
    df["dataset_split"] = df["dataset_split_group"].map(dataset_split_dict)
    return df.drop(columns=["dataset_split_group"])

def prepare_training_metadata(data_version: str) -> None:
    """
    Read the preprocessed metadata file, add training-specific columns (label encoding and dataset splits),
    and save to the training metadata location.
    
    Args:
        data_version: Version string (e.g. 'v2')
    """
    paths = load_base_paths()
    
    # Construct paths
    preprocessed_metadata_path = os.path.join(paths.preprocessed_base, f"landmarks_metadata_{data_version}.csv")
    training_metadata_path = os.path.join(paths.metadata_base, f"landmarks_metadata_{data_version}_training.csv")
    
    # Ensure metadata directory exists
    os.makedirs(os.path.dirname(training_metadata_path), exist_ok=True)
    
    # Read preprocessed metadata
    df = pd.read_csv(preprocessed_metadata_path)
    df = validate_filename_column(df)
    # Add training-specific columns
    df = encode_label(df)
    df = train_test_split(df)
    
    # Save to training location
    df.to_csv(training_metadata_path, index=False)
    print(f"Training metadata saved to: {training_metadata_path}")

def prepare_landmark_arrays(load_landmarks_dir: str, positions_config: DictConfig) -> None:

    landmark_series_fns = [fn for fn in os.listdir(load_landmarks_dir) if fn.endswith('.npy')]

    paths = load_base_paths()
    save_landmarks_dir = paths.landmark_arrays_base
    if not os.path.exists(save_landmarks_dir):
        os.makedirs(save_landmarks_dir, exist_ok=True)

    landmark_keys = [
        "pose_landmarks",
        "left_hand_landmarks",
        "right_hand_landmarks",
        # "face_landmarks",
    ]

    scaling_info = positions_config.scaling_info

    for series_fn in landmark_series_fns:
        series_fp = os.path.join(load_landmarks_dir, series_fn)
        series = np.load(series_fp, allow_pickle=True)

        for landmark_key in landmark_keys:
            series_xyz = []
            for frame in series:
                xyz = [[lm.x, lm.y, lm.z] for lm in frame[landmark_key].landmark]
                xyz = np.array(xyz)
                series_xyz.append(xyz)
            series_xyz = np.array(series_xyz)

            if positions_config.computation_type == "scaled":
                # scale x
                series_xyz[:,:,0] = minmax_scale_series(
                    series_xyz[:,:,0],
                    scaling_info[landmark_key]['input_max_x'],
                    scaling_info[landmark_key]['input_min_x'],
                    scaling_info['scale_range']
                )
                # scale y
                series_xyz[:,:,1] = minmax_scale_series(
                    series_xyz[:,:,1],
                    scaling_info[landmark_key]['input_max_y'],
                    scaling_info[landmark_key]['input_min_y'],
                    scaling_info['scale_range'])
                
            save_fp = os.path.join(save_landmarks_dir, series_fn.replace('.npy', f'_{landmark_key}.npy'))
            np.save(save_fp, series_xyz)