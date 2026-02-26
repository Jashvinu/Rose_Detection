"""Orchestrator: wires detector -> tracker -> counter into a single pipeline run."""

from __future__ import annotations

import numpy as np
from PIL import Image

from app.config import Settings
from app.pipeline.classifier import YOLOTFLiteDetector
from app.pipeline.counter import finalize_tracks
from app.pipeline.motion import MotionCompensator
from app.pipeline.tiler import detect_in_image
from app.pipeline.tracker import IoUTracker
from app.schemas import FrameDetail, SummaryResponse


def run_pipeline(
    image_pairs: list[tuple[str, Image.Image]],
    token: str,
    job_id: str,
    detector: YOLOTFLiteDetector,
    settings: Settings,
    *,
    confidence_threshold: float | None = None,
    min_hits: int | None = None,
    motion_compensation: bool | None = None,
) -> SummaryResponse:
    """Run the full detect-track-count pipeline on ordered images."""
    conf_thresh = confidence_threshold if confidence_threshold is not None else settings.confidence_threshold
    hits = min_hits if min_hits is not None else settings.min_hits
    use_motion = motion_compensation if motion_compensation is not None else settings.motion_compensation

    tracker = IoUTracker(
        match_iou_threshold=settings.match_iou_threshold,
        max_age=settings.max_age,
        min_hits=hits,
    )

    compensator: MotionCompensator | None = None
    if use_motion:
        compensator = MotionCompensator(
            max_features=settings.motion_max_features,
            min_matches=settings.motion_min_matches,
        )

    filenames = [fn for fn, _ in image_pairs]
    frame_details: list[FrameDetail] = []

    for frame_index, (filename, image) in enumerate(image_pairs):
        detections = detect_in_image(
            image=image,
            detector=detector,
            frame_index=frame_index,
            confidence_threshold=conf_thresh,
            nms_iou_threshold=settings.nms_iou_threshold,
        )

        homography: np.ndarray | None = None
        if compensator is not None:
            homography = compensator.update(np.array(image))

        active_count = tracker.update(detections, homography=homography)

        frame_details.append(
            FrameDetail(
                frame_index=frame_index,
                filename=filename,
                detections=detections,
                active_tracks=active_count,
            )
        )

    all_tracks = tracker.finalize()
    summary, _, unique_detections = finalize_tracks(
        all_tracks, settings.class_agreement_ratio, filenames=filenames
    )
    unique_frames = sorted({ud.frame_index for ud in unique_detections})

    return SummaryResponse(
        token=token,
        job_id=job_id,
        status="completed",
        summary=summary,
        per_frame_details=frame_details,
        unique_detections=unique_detections,
        unique_frames=unique_frames,
    )
