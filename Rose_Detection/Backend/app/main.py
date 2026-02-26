import io
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.pipeline.classifier import YOLOTFLiteDetector
from app.pipeline.tiler import nms_per_class
from app.schemas import DetectResponse

logger = logging.getLogger(__name__)

detector: YOLOTFLiteDetector | None = None
settings = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, settings
    settings = get_settings()
    detector = YOLOTFLiteDetector(
        model_path=settings.model_path,
        labels_path=settings.labels_path,
    )
    yield


app = FastAPI(title="Rose Disease Detection", version="3.0.0", lifespan=lifespan)


@app.post("/detect", response_model=DetectResponse)
async def detect(
    image: UploadFile = File(...),
    confidence_threshold: float | None = Form(default=None),
):
    image_bytes = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, Exception) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    conf = confidence_threshold if confidence_threshold is not None else settings.confidence_threshold
    img_arr = np.array(pil_image, dtype=np.uint8)

    dets = detector.detect(img_arr, conf)
    dets = nms_per_class(dets, settings.nms_iou_threshold)

    return DetectResponse(detections=dets)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "YOLO11s TFLite",
        "labels": detector.labels if detector else [],
    }
