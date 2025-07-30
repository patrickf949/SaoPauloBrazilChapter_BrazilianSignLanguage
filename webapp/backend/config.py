from pathlib import Path
from omegaconf import OmegaConf

class Config:
    # Base directories
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    INTERIM_DIR = DATA_DIR / "interim" / "RawCleanVideos"
    PREPROCESSED_DIR = DATA_DIR / "preprocessed"
    OUTPUT_DIR = DATA_DIR / "output"
    MODEL_DIR = BASE_DIR / "model"

    # File paths
    CONFIG_PATH = MODEL_DIR / "config.yaml"
    CHECKPOINT_PATH = MODEL_DIR / "model_checkpoint.pt"
    LABEL_ENCODING_PATH = MODEL_DIR / "label_encoding.json"

    # Versions
    TIMESTAMP = "00"
    MOTION_VERSION = "v0"
    POSE_VERSION = "v0"
    PREPROCESS_VERSION = "v0"

    # Preprocessing parameters
    PREPROCESSING_PARAMS = {
        "face_width_aim": 0.155,
        "shoulders_width_aim": 0.35,
        "face_midpoint_to_shoulders_height_aim": 0.275,
        "shoulders_y_aim": 0.52,
        "shoulders_x_aim": 0.5,  # Added to center shoulders horizontally
        "use_statistic": "mean",
        "use_stationary_frames": True,
        "skip_stationary_frames": False,
    }


    def __init__(self):
        # Load YAML configuration
        self.config_yaml = OmegaConf.load(self.CONFIG_PATH)

        # Ensure directories exist
        for directory in [self.INTERIM_DIR, self.PREPROCESSED_DIR, self.OUTPUT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_instance(cls):
        return cls()

