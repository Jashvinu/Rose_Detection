import io
import logging
from contextlib import asynccontextmanager

import boto3
import numpy as np
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.pipeline.tiler import nms_per_class
from app.schemas import DetectFlowersRequest, DetectResponse

logger = logging.getLogger(__name__)

detector = None
settings = None


def create_detector(app_settings):
    from app.pipeline.classifier import YOLOTFLiteDetector

    return YOLOTFLiteDetector(
        model_path=app_settings.model_path,
        labels_path=app_settings.labels_path,
    )


def get_s3_client():
    return boto3.client("s3")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError("image_s3_uri must start with s3://")
    stripped = uri[len("s3://"):]
    if "/" not in stripped:
        raise ValueError("image_s3_uri must include a bucket and key")
    bucket, key = stripped.split("/", 1)
    if not bucket or not key:
        raise ValueError("image_s3_uri must include a bucket and key")
    return bucket, key


def fetch_image_bytes_from_s3(uri: str) -> bytes:
    bucket, key = parse_s3_uri(uri)
    client = get_s3_client()
    response = client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, settings
    settings = get_settings()
    detector = create_detector(settings)
    yield


app = FastAPI(title="Rose Disease Detection", version="3.0.0", lifespan=lifespan)


@app.post("/detect_flowers", response_model=DetectResponse)
async def detect_flowers(payload: DetectFlowersRequest):
    try:
        image_bytes = fetch_image_bytes_from_s3(payload.image_s3_uri)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (BotoCoreError, ClientError) as exc:
        logger.exception("Failed to fetch image from S3", extra={"image_s3_uri": payload.image_s3_uri})
        raise HTTPException(status_code=502, detail=f"Failed to fetch image from S3: {exc}")
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, Exception) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    img_arr = np.array(pil_image, dtype=np.uint8)

    dets = detector.detect(img_arr, settings.confidence_threshold)
    dets = nms_per_class(dets, settings.nms_iou_threshold)

    return DetectResponse(detections=dets)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "YOLO11s TFLite",
        "labels": detector.labels if detector else [],
    }
