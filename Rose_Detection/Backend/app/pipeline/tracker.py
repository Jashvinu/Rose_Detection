"""IoU-based cross-frame tracker (simplified ByteTrack, no Kalman filter)."""

from __future__ import annotations

import numpy as np

from app.pipeline.tiler import compute_iou
from app.schemas import BBox, Detection, Track


class IoUTracker:
    """Match detections across frames using greedy IoU matching."""

    def __init__(
        self,
        match_iou_threshold: float = 0.3,
        max_age: int = 10,
        min_hits: int = 3,
    ):
        self.match_iou_threshold = match_iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self._next_id = 1
        self._tracks: list[_ActiveTrack] = []
        self.finished_tracks: list[Track] = []

    def update(
        self,
        detections: list[Detection],
        homography: np.ndarray | None = None,
    ) -> int:
        """Process one frame of detections. Returns number of active tracks.

        If a homography (prev→current) is provided, each track's last bbox is
        warped into the current frame's coordinate space before IoU matching.
        """
        from app.pipeline.motion import warp_bbox

        num_existing = len(self._tracks)

        # Build IoU pairs between existing tracks and new detections
        pairs: list[tuple[float, int, int]] = []  # (iou, track_idx, det_idx)
        for t_idx in range(num_existing):
            track_bbox = self._tracks[t_idx].last_bbox()
            if homography is not None:
                track_bbox = warp_bbox(track_bbox, homography)
            for d_idx, det in enumerate(detections):
                iou = compute_iou(track_bbox, det.bbox)
                if iou >= self.match_iou_threshold:
                    pairs.append((iou, t_idx, d_idx))

        # Greedy matching sorted by IoU descending
        pairs.sort(key=lambda p: p[0], reverse=True)
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        for _, t_idx, d_idx in pairs:
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue
            self._tracks[t_idx].add_observation(detections[d_idx])
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)

        # Age unmatched pre-existing tracks
        for t_idx in range(num_existing):
            if t_idx not in matched_tracks:
                self._tracks[t_idx].age += 1

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                trk = _ActiveTrack(track_id=self._next_id, min_hits=self.min_hits)
                self._next_id += 1
                trk.add_observation(det)
                self._tracks.append(trk)

        # Retire stale tracks
        surviving: list[_ActiveTrack] = []
        for trk in self._tracks:
            if trk.age > self.max_age:
                self.finished_tracks.append(trk.to_track())
            else:
                surviving.append(trk)

        self._tracks = surviving
        return len(self._tracks)

    def finalize(self) -> list[Track]:
        """Flush all remaining active tracks after the last frame."""
        for trk in self._tracks:
            self.finished_tracks.append(trk.to_track())
        self._tracks.clear()
        return self.finished_tracks


class _ActiveTrack:
    """Internal mutable track used during processing."""

    def __init__(self, track_id: int, min_hits: int):
        self.track_id = track_id
        self.min_hits = min_hits
        self.observations: list[Detection] = []
        self.age: int = 0  # frames since last match

    def add_observation(self, det: Detection) -> None:
        self.observations.append(det)
        self.age = 0

    def last_bbox(self):
        return self.observations[-1].bbox

    def to_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            observations=list(self.observations),
            confirmed=len(self.observations) >= self.min_hits,
        )
