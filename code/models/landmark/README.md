# Brazilian Sign Language Recognition System

This codebase implements a machine learning pipeline for Brazilian Sign Language recognition using landmarks (pose and hand positions) extracted from video data.

## System Architecture

The system follows a modular design with these main components:

### 1. Feature Engineering
Extracts meaningful features from raw landmark data:
- **Angles**: Calculates angles between landmark triplets (e.g., finger joints)
- **Distances**: Measures distances between landmark pairs
- **Frame-to-frame differences**: Captures motion dynamics between consecutive frames

### 2. Dataset Pipeline
Handles loading, processing, and augmentation:
- `LandmarkDataset` class loads preprocessed landmark data
- Frame selection based on various interval strategies
- Augmentations like rotation and noise addition

### 3. Model Architecture
Several model options are available:
- **Transformer-based** (main model)
- **RNN-based** classifiers (LSTM, GRU)
- **KNN classifier**

### 4. Training Pipeline
Manages the training process:
- Supports cross-validation or regular train/val/test splits
- Early stopping with patience
- Logging metrics

## How to Run Training

To run training:

```bash
# Navigate to the landmark directory
cd /path/to/code/models/landmark

# Run training with default config
python trainer.py

# Run with specific config
python trainer.py --config-name=your_custom_config
```

## Configuration System

All aspects of the system are configured through YAML files in the `configs/` directory, using Hydra for configuration management:

- **Main config**: `configs/train_config.yaml`
- **Dataset config**: `configs/dataset/dataset.yaml`
- **Model configs**: `configs/model/*.yaml`
- **Feature configs**: `configs/features/*.yaml`
- **Augmentation configs**: `configs/augmentation/*.yaml`
- **Training configs**: `configs/training/*.yaml`
- **Optimizer configs**: `configs/optimizer/*.yaml`
- **Scheduler configs**: `configs/scheduler/*.yaml`

## Feature Engineering Details

The feature engineering process transforms raw landmark coordinates into meaningful features for sign language recognition:

### Feature Estimator Classes

Each type of feature is computed by a dedicated estimator class:

1. **`AnglesEstimator`** (`dataset/angles_estimator.py`)
   - Computes angles between triplets of landmarks (e.g., between finger joints)
   - Supports various angle representations: radians, degrees, normalized, or (sin, cos) pairs
   - Example: `thumb_bend: [1, 2, 3]` calculates the angle between landmarks 1, 2, and 3

2. **`DistancesEstimator`** (`dataset/distances_estimator.py`)
   - Calculates distances between pairs of landmarks
   - Supports raw, normalized, or shifted distance formats
   - Example: `left_hand_to_head: [15, 0]` measures distance between landmarks 15 and 0

3. **`DifferencesEstimator`** (`dataset/frame2frame_differences_estimator.py`)
   - Captures motion by calculating position changes between consecutive frames
   - Tracks specified landmark indices to measure movement
   - Example: `wrist: 0` tracks the movement of landmark 0 (wrist) between frames

4. **`LandmarkEstimator`** (`dataset/base_estimator.py`)
   - Extracts raw landmark coordinates as features
   - Useful as a baseline or combined with other features

### Feature Computation Pipeline

Features are computed in the `__getitem__` method of the `LandmarkDataset` class:

1. Load a sequence of landmarks from .npy files
2. For each frame, apply augmentations (if enabled)
3. For each feature type and landmark type (pose, left_hand, right_hand):
   - Call the appropriate estimator's `compute` method
   - Store the feature vector
4. Concatenate all feature vectors into a single tensor
5. Return the features along with the label

### Configuration-Driven Features

All feature definitions are stored in YAML files in `configs/features/`:
- Features can be added, modified, or disabled by editing these files
- The `train_config.yaml` file controls which feature types are included

## Data Splitting & Cross-Validation

The system provides two approaches to data management for training:

### Train/Test Splitting

Data is divided into training and test sets at the video level (not frame level):

1. **Split Mechanism** (`dataset/prepare_data_csv.py`):
   - Videos are categorized using a shifting letter-based system (A-F)
   - By default, 5/6 of videos per label go to training (A-E), 1/6 to test (F)
   - Letter assignments are intentionally shifted for each label to prevent biases
   - Example mapping:
     ```python
     dataset_split_dict = {
         "A": "train", "B": "train", "C": "train", 
         "D": "train", "E": "train", "F": "test"
     }
     ```

2. **Process Flow**:
   - Splitting happens in the `prepare_training_metadata` function
   - The resulting metadata contains a `dataset_split` column
   - `LandmarkDataset` filters data based on the requested split

### Cross-Validation

When `config.training.type` is set to "cross_validation":

1. **K-Fold Implementation** (`utils/train.py`):
   - Uses scikit-learn's `KFold` to create train/validation splits
   - The number of folds is configurable (default is 5)
   - Example configuration:
     ```yaml
     training:
       type: cross_validation
       k_folds: 5
     ```

2. **Execution Process**:
   - Only the "train" portion of the data is used for cross-validation
   - In each fold, this data is further split into training and validation sets
   - Model is trained on all folds and performance averaged
   - Final evaluation uses the separate "test" set that was never seen during training

### Benefits of This Approach

- **Robustness**: Cross-validation provides better estimates of model performance
- **Data Efficiency**: Makes best use of limited training data
- **Hold-out Testing**: Final test set ensures honest evaluation on unseen data
- **Configurability**: Easily switch between standard and cross-validation approaches

## Making Modifications

### 1. Changing Configuration Settings

#### Dataset Configuration
Edit `configs/dataset/dataset.yaml`:
```yaml
data_dir: data/processed/v2
data_path: data/preprocessed/landmarks_metadata_v3_training.csv

frame_interval_fn: random_timestamps  # Options: random_timestamps, uniform_timestamps, uniform_intervals
interval: 15  # Number of frames to sample

landmark_types: [pose, right_hand, left_hand]
ordering: [features, landmark_types]
```

#### Training Configuration
Edit `configs/training/training.yaml`:
```yaml
num_epochs: 300
batch_size: 256
patience: 50
lr: 1e-4
device: cuda
type: cross_validation  # Options: cross_validation, standard
k_folds: 5
```

#### Model Configuration
Edit model config files in `configs/model/`:
```yaml
# Example: configs/model/transformer.yaml
class_name: models.landmark.models.transformers.TransformerClassifier
params:
  input_size: 215
  num_classes: 25
  d_model: 128
  nhead: 8
  num_layers: 2
  dim_feedforward: 256
  dropout: 0.1
```

#### Augmentation Configuration
Edit `configs/augmentation/base_augs.yaml`:
```yaml
train: 
  rotate: 
    class_name: models.landmark.dataset.augmentations.RotateLandmarks
    p: 0.5
    params:
      angle_range: 10

  noise:
    class_name: models.landmark.dataset.augmentations.LandmarksNoise
    p: 0.5
    params:
      noise_std: 0.01

val:
test: 
```

### 2. Feature Engineering Changes

#### Modifying Existing Features

Edit the feature config files in `configs/features/`:

Example for angles (`configs/features/angles.yaml`):
```yaml
angles:
  class_name: models.landmark.dataset.angles_estimator.AnglesEstimator
  hand:
    thumb_base: [0, 1, 2]  # wrist → thumb_cmc → thumb_mcp
    thumb_bend: [1, 2, 3]
    # ... more angles
  pose: 
    right_shoulder_angle: [12, 14, 16]
    # ... more angles
  computation_type: func  # Options: rad, grad, normalized_rad, shifted_rad, func
  mode: 3D  # Options: 2D, 3D
```

#### Creating New Feature Types

1. Create a new estimator class in `dataset/` similar to `angles_estimator.py`
2. Inherit from `BaseEstimator` and implement the `compute` method
3. Add a new config file in `configs/features/`
4. Add the new feature to the `defaults` list in `train_config.yaml`

Example of a new estimator class:
```python
class YourNewEstimator(BaseEstimator):
    def __init__(self, hand_config, pose_config):
        super().__init__(hand_config, pose_config, config_type="your_feature_type")
        
    def compute(self, landmarks, landmark_type, mode, computation_type):
        # Implement your feature computation logic here
        # Return numpy array of features
```

### 3. Adding Data Augmentation

1. Create a new augmentation class in `dataset/augmentations.py`:
```python
class YourNewAugmentation:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
        
    def __call__(self, landmarks: Dict) -> Dict:
        # Implement your augmentation logic here
        # Landmarks dict contains keys like 'pose_landmarks', 'left_hand_landmarks', etc.
        return landmarks
```

2. Add it to your augmentation config file:
```yaml
train:
  your_aug_name:
    class_name: models.landmark.dataset.augmentations.YourNewAugmentation
    p: 0.5  # probability of applying this augmentation
    params:
      param1: value1
      param2: value2
```

### 4. Using Different Data

To use different data:

1. Make sure your data follows the expected format:
   - Landmarks stored as numpy arrays (with hand and pose landmarks)
   - CSV metadata file with columns like 'filename', 'dataset_split', 'label_encoded', 'start_frame', 'end_frame'

2. Update the dataset config to point to your data:
```yaml
data_dir: path/to/your/landmark/files
data_path: path/to/your/metadata.csv
```

### 5. Model Architecture Changes

To experiment with different model architectures:

1. Check existing models in the `models/` directory:
   - `transformers.py`: Transformer-based classifiers
   - `rnn_classifiers.py`: LSTM and GRU models
   - `knn_classifier.py`: K-nearest neighbors

2. Create a new model file with your architecture:
```python
import torch.nn as nn

class YourNewModel(nn.Module):
    def __init__(self, input_size, num_classes, **params):
        super().__init__()
        # Define your model architecture
        
    def forward(self, x):
        # Implement forward pass
        return output
```

3. Create a new config file in `configs/model/your_model.yaml`:
```yaml
class_name: models.landmark.models.your_model.YourNewModel
params:
  input_size: 215
  num_classes: 25
  # Add your model-specific parameters
```

## Directory Structure

- `trainer.py`: Main training script
- `dataset/`: Contains dataset classes and feature computation
  - `landmark_dataset.py`: Main dataset class
  - `*_estimator.py`: Feature computation classes
  - `augmentations.py`: Data augmentation implementations
- `models/`: Neural network model implementations
- `utils/`: Utility functions for training, evaluation, etc.
- `configs/`: Configuration files
- `visualization/`: Tools for visualizing data and results

## Tips for Experimentation

- **Start with small changes**: Adjust one parameter at a time
- **Monitor validation metrics**: Watch for validation loss and accuracy to avoid overfitting
- **Try different feature combinations**: Experiment with which features work best for your data
- **Data augmentation impact**: Adjust augmentation parameters to balance realism and variety
- **Frame sampling strategy**: The choice of frame selection can significantly impact performance

## Common Issues and Solutions

- **CUDA out of memory**: Reduce batch size or model complexity
- **Slow training**: Check if you're using GPU, reduce data complexity
- **Overfitting**: Increase augmentation, add regularization, reduce model complexity
- **Poor validation performance**: Try different feature combinations, adjust model architecture 