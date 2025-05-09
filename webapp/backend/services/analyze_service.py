from pathlib import Path
from fastapi import UploadFile

VIDEO_DIR = Path("videos")
VIDEO_DIR.mkdir(exist_ok=True)

async def process_video(video: UploadFile) -> dict:
    video_path = VIDEO_DIR / video.filename
    content = await video.read()
    video_path.write_bytes(content)
    return {"message": f"Video '{video.filename}' uploaded and analyzed."}

def process_label(label: str) -> dict:
    return {"message": f"Label '{label}' received and analyzed."}

def list_all_videos() -> list:
    return [f.name for f in VIDEO_DIR.iterdir() if f.is_file()]
