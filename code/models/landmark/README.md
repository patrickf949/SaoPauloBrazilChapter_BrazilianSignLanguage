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