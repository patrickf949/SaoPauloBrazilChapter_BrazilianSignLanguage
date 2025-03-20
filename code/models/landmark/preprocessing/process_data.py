from landmark_detector import LandmarkDetector

config = {
 "train_path": "",
 "test_path": "",
 "output_base_folder": "",
 "augment_training": True,
 "num_augmentations": 4
}
import os

if __name__ == "__main__":
    # Define paths
    print(os.getcwd())
    # Initialize and run processor
    processor = LandmarkDetector()
    processor.process_dataset(
        **config
    )