"""Motion compensation via ORB feature matching + RANSAC homography."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.schemas import BBox

logger = logging.getLogger(__name__)


class MotionCompensator:
    """Estimate frame-to-frame homography using ORB features."""

    def __init__(self, max_features: int = 500, min_matches: int = 10):
        self.max_features = max_features
        self.min_matches = min_matches
        self.min_inlier_ratio = 0.3
        self._orb = cv2.ORB_create(nfeatures=max_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._prev_gray: np.ndarray | None = None
        self._prev_kp = None
        self._prev_des = None

    def update(self, image: np.ndarray) -> np.ndarray | None:
        """Compute homography from previous frame to current frame.

        Args:
            image: BGR or RGB uint8 image (H, W, 3).

        Returns:
            3x3 homography matrix mapping prev-frame coords to current-frame
            coords, or None on the first frame or when matching fails.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kp, des = self._orb.detectAndCompute(gray, None)

        H = None

        if self._prev_des is not None and des is not None and len(kp) >= 4:
            matches = self._bf.knnMatch(self._prev_des, des, k=2)

            # Lowe's ratio test
            good = []
            for m_pair in matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            if len(good) >= self.min_matches:
                src_pts = np.float32(
                    [self._prev_kp[m.queryIdx].pt for m in good]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kp[m.trainIdx].pt for m in good]
                ).reshape(-1, 1, 2)

                H_candidate, mask = cv2.findHomography(
                    src_pts, dst_pts, cv2.RANSAC, 5.0
                )

                if H_candidate is not None and mask is not None:
                    inlier_ratio = mask.sum() / len(mask)
                    if inlier_ratio >= self.min_inlier_ratio:
                        H = H_candidate
                    else:
                        logger.debug(
                            "Homography rejected: inlier ratio %.2f < %.2f",
                            inlier_ratio,
                            self.min_inlier_ratio,
                        )
            else:
                logger.debug(
                    "Too few good matches: %d < %d", len(good), self.min_matches
                )

        self._prev_gray = gray
        self._prev_kp = kp
        self._prev_des = des
        return H


def warp_bbox(bbox: BBox, H: np.ndarray) -> BBox:
    """Warp a bounding box through a homography by transforming its 4 corners.

    Args:
        bbox: source bounding box.
        H: 3x3 homography matrix.

    Returns:
        Axis-aligned bounding box enclosing the warped corners.
    """
    corners = np.float32([
        [bbox.x1, bbox.y1],
        [bbox.x2, bbox.y1],
        [bbox.x2, bbox.y2],
        [bbox.x1, bbox.y2],
    ]).reshape(-1, 1, 2)

    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    return BBox(
        x1=float(warped[:, 0].min()),
        y1=float(warped[:, 1].min()),
        x2=float(warped[:, 0].max()),
        y2=float(warped[:, 1].max()),
    )
