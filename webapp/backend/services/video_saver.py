
from pathlib import Path
from fastapi import UploadFile

from utils import save_uploaded_file


class VideoSaver:
    @staticmethod
    async def save(file: UploadFile, interim_dir: Path) -> Path:
        return await save_uploaded_file(file, interim_dir)
