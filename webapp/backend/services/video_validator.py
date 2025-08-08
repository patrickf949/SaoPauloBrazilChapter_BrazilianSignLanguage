from fastapi import HTTPException


class VideoValidator:
    @staticmethod
    def validate_extension(filename: str):
        if not filename.lower().endswith(".mp4"):
            raise HTTPException(status_code=400, detail="Only .mp4 files are supported")
