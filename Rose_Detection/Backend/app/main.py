"""FastAPI backend for rose disease detection using the detect-track-count pipeline."""

import base64
import io
import logging
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

from app.config import Settings, get_settings
from app.pipeline.affliction_mapper import map_results, parse_pv_token
from app.pipeline.classifier import YOLOTFLiteDetector
from app.pipeline.orchestrator import run_pipeline
from app.pipeline.visualizer import annotate_image
from app.pv_router import router as pv_router
from app.schemas import Detection, SummaryResponse

# ── Globals populated at startup ─────────────────────────────────────────────
detector: YOLOTFLiteDetector | None = None
settings: Settings | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, settings
    settings = get_settings()
    detector = YOLOTFLiteDetector(
        model_path=settings.model_path,
        labels_path=settings.labels_path,
    )
    yield


app = FastAPI(
    title="PlantVillage Rose Disease Detection",
    description=(
        "Rose disease detection pipeline with motion-compensated burst deduplication.\n\n"
        "## Detection Pipeline (our API)\n"
        "- `POST /api/v1/detect` — upload images for detect-track-count\n"
        "- `GET /api/v1/jobs/{job_id}` — retrieve job results\n"
        "- `GET /api/v1/jobs/{job_id}/frames/{index}/image` — annotated frame JPEG\n\n"
        "## PlantVillage Admin (upstream reference)\n"
        "- `GET /admin/flowers/scouts` — list scouts\n"
        "- `GET /admin/flowers/scouts/{id}` — scout detail with partitions & beds\n"
        "- `GET /admin/flowers/scouts/{id}/partitions/{id}` — beds with photos\n"
        "- `GET /admin/flowers/greenhouses` — list greenhouses\n\n"
        "**Data flow**: Greenhouse → Scout → Partition (A/B/C) → Bed → Photos + Afflictions"
    ),
    version="2.0.0",
    lifespan=lifespan,
)
app.include_router(pv_router)

# In-memory job store  (results + original images for visualization)
jobs: dict[str, dict] = {}
job_images: dict[str, list[Image.Image]] = {}


async def _fire_callback(callback_url: str, payload: dict) -> None:
    """POST pipeline results to the caller's callback URL."""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            await client.post(callback_url, json=payload)
        except httpx.HTTPError:
            pass  # best-effort; caller can poll /jobs/{job_id}


@app.post("/api/v1/detect", response_model=SummaryResponse)
async def detect(
    background_tasks: BackgroundTasks,
    images: list[UploadFile] = File(...),
    token: str = Form(default=""),
    callback_url: str = Form(default=""),
    confidence_threshold: float | None = Form(default=None),
    min_hits: int | None = Form(default=None),
    visualize: bool = Form(default=True),
    motion_compensation: bool | None = Form(default=None),
):
    """Accept sequential images and return detect-track-count results.

    Set visualize=true (default) to embed annotated images as base64 in each frame.
    """
    job_id = str(uuid.uuid4())

    # Read images in upload order (FastAPI preserves multipart ordering)
    image_pairs: list[tuple[str, Image.Image]] = []
    skipped: list[str] = []
    for img_file in images:
        image_bytes = await img_file.read()
        if not image_bytes:
            skipped.append(img_file.filename or "unknown")
            continue
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, Exception) as exc:
            logger.warning("Skipping %s: %s", img_file.filename, exc)
            skipped.append(img_file.filename or "unknown")
            continue
        image_pairs.append((img_file.filename or "unknown", image))

    if not image_pairs:
        raise HTTPException(
            status_code=400,
            detail=f"No valid images uploaded. Skipped: {skipped}",
        )

    result = run_pipeline(
        image_pairs=image_pairs,
        token=token,
        job_id=job_id,
        detector=detector,
        settings=settings,
        confidence_threshold=confidence_threshold,
        min_hits=min_hits,
        motion_compensation=motion_compensation,
    )
    result.skipped_files = skipped

    # PlantVillage affliction mapping + token parsing
    pv_context = parse_pv_token(token)
    afflictions, condition = map_results(result.unique_detections)
    result.afflictions = afflictions
    result.condition = condition
    result.scout_id = pv_context["scout_id"]
    result.partition_id = pv_context["partition_id"]
    result.bed_id = pv_context["bed_id"]

    # Embed annotated images as base64 data URIs
    if visualize:
        for frame, (_, original) in zip(result.per_frame_details, image_pairs):
            img_annotated = annotate_image(original, frame.detections)
            buf = io.BytesIO()
            img_annotated.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            frame.annotated_image = f"data:image/jpeg;base64,{b64}"

    jobs[job_id] = result.model_dump()
    job_images[job_id] = [img for _, img in image_pairs]

    if callback_url:
        background_tasks.add_task(_fire_callback, callback_url, result.model_dump())

    return result


@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str):
    """Retrieve job results by ID."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/api/v1/jobs/{job_id}/frames/{frame_index}/image")
async def get_frame_image(job_id: str, frame_index: int):
    """Return the annotated image for a specific frame of a job.

    Open in a browser:  http://localhost:8000/api/v1/jobs/{job_id}/frames/0/image
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    stored = jobs[job_id]
    frames = stored.get("per_frame_details", [])
    if frame_index < 0 or frame_index >= len(frames):
        raise HTTPException(status_code=404, detail="Frame index out of range")

    if job_id not in job_images or frame_index >= len(job_images[job_id]):
        raise HTTPException(status_code=410, detail="Original images no longer in memory")

    original = job_images[job_id][frame_index]
    detections = [Detection(**d) for d in frames[frame_index]["detections"]]
    annotated = annotate_image(original, detections)

    buf = io.BytesIO()
    annotated.save(buf, format="JPEG", quality=90)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "YOLO11s TFLite",
        "model_file": settings.model_path.name if settings else "not loaded",
        "labels": detector.labels if detector else [],
    }
