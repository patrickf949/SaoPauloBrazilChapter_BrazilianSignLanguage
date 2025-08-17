import tempfile
from fastapi import HTTPException, UploadFile, status
from moviepy import VideoFileClip


MAX_FILE_SIZE = 250 * 1024 * 1024  # 250 MB in bytes


class VideoValidator:
    @staticmethod
    def validate_extension(filename: str):
        if not filename.lower().endswith(".mp4"):
            raise HTTPException(
                status_code=400, detail="Only .mp4 files are supported")

    @staticmethod
    async def validate_file(file: UploadFile):
        VideoValidator.validate_extension(file.filename)
        await VideoValidator.validate_file_size(file)
        await VideoValidator.validate_duration(file)

    @staticmethod
    async def validate_file_size(file: UploadFile):
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File should not exceed the 250MB."
            )
        await file.seek(0)


    @staticmethod
    async def validate_duration(file: UploadFile):
        # Save to a temporary file to read with moviepy
        
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            tmp.write(await file.read())
            tmp.flush()

            clip = VideoFileClip(tmp.name)
            duration = clip.duration  # in seconds
            clip.close()

            if duration > 20:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, 
                    detail="Video too long (max 20s)"
                )
