"""Pydantic data models for the detect-track-count pipeline."""

from __future__ import annotations

from pydantic import BaseModel


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    bbox: BBox
    label: str
    confidence: float
    frame_index: int


class Track(BaseModel):
    track_id: int
    observations: list[Detection] = []
    confirmed: bool = False
    counted: bool = False
    final_label: str | None = None
    first_seen_frame: int | None = None


class UniqueDetection(BaseModel):
    track_id: int
    label: str
    confidence: float
    bbox: BBox
    frame_index: int
    filename: str


class Affliction(BaseModel):
    id: int
    name: str


class FrameDetail(BaseModel):
    frame_index: int
    filename: str
    detections: list[Detection]
    active_tracks: int
    annotated_image: str | None = None  # base64 JPEG data URI


class SummaryResponse(BaseModel):
    token: str
    job_id: str
    status: str
    summary: dict[str, int]
    per_frame_details: list[FrameDetail]
    skipped_files: list[str] = []
    unique_detections: list[UniqueDetection] = []
    unique_frames: list[int] = []
    # PlantVillage integration fields
    afflictions: list[Affliction] = []
    condition: str | None = None
    scout_id: int | None = None
    partition_id: int | None = None
    bed_id: int | None = None
