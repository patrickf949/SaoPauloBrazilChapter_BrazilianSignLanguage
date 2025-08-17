import datetime
import pathlib
import logging
from fastapi import FastAPI, Request
from api.routes import router as api_router

logger = logging.getLogger(__name__)
app = FastAPI(title="Video Sign Language Prediction API")

# Middleware for heartbeat
heartbeat_file = pathlib.Path("/tmp/last_request.txt")

@app.middleware("http")
async def track_request(request: Request, call_next):
    heartbeat_file.write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())
    return await call_next(request)

# Register routes
app.include_router(api_router)
