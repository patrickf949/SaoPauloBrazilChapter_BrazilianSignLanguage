from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
from services.analyze_service import process_video, process_label, list_all_videos

router = APIRouter()

@router.post("/analyze")
async def analyze(
    video: Optional[UploadFile] = File(None),
    label: Optional[str] = Form(None)
):
    if video:
        return await process_video(video)

    if label:
        return process_label(label)

    raise HTTPException(status_code=400, detail="Provide either a video or a label.")


@router.get("/")
def list_videos():
    return {"videos": list_all_videos()}
