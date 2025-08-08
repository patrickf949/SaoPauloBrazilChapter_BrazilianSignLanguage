from fastapi import HTTPException
import pandas as pd

from config import Config
from services.model_references import get_video_metadata

class MetadataExtractor:
    @staticmethod
    def extract(video_path: str) -> pd.Series:
        metadata = get_video_metadata(video_path)
        if not metadata:
            raise HTTPException(status_code=500, detail="Failed to extract video metadata")
        metadata['data_source'] = 'app'
        return pd.Series(metadata)


class MetadataFilterService:
    @staticmethod
    def load_and_filter(config: Config, filename: str) -> pd.Series:
        metadata_csv_path = config.PREPROCESSED_DIR / f"landmarks_metadata_{config.PREPROCESS_VERSION}.csv"
        if not metadata_csv_path.exists():
            raise HTTPException(status_code=500, detail="Metadata CSV not found")

        df = pd.read_csv(metadata_csv_path)
        filtered = df[df['filename'] == filename]

        if len(filtered) == 0:
            raise HTTPException(status_code=404, detail=f"No metadata found for video {filename}")
        if len(filtered) > 1:
            raise HTTPException(status_code=500, detail=f"Multiple metadata entries found for video {filename}")

        row = filtered.iloc[0].copy()
        row['filename'] = row['filename'].replace(".mp4", ".npy")
        return row