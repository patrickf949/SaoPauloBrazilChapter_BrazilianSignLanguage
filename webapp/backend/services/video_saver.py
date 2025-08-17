
from pathlib import Path
from fastapi import HTTPException, UploadFile, status

from services.converter import VideoConversionService
from utils import copy_existing_video, save_uploaded_file


class RequestVideoSaver:
    @staticmethod
    async def save(file: UploadFile, interim_dir: Path) -> Path:
        return await save_uploaded_file(file, interim_dir)
        # VideoConversionService.convert_to_mp4(path)
        # return path
    

class LabelVideoSaver:
    @staticmethod
    async def save(file: Path, interim_dir: Path) -> Path:
        try:

            return await copy_existing_video(file, interim_dir)
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Choose a valid video: {error}"
            )
