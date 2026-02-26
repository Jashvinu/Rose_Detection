from __future__ import annotations
from pydantic import BaseModel


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: BBox


class DetectResponse(BaseModel):
    detections: list[Detection]


class DetectFlowersRequest(BaseModel):
    image_s3_uri: str
