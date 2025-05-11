import pandas as pd
import os
from omegaconf import OmegaConf

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
