"""Count-once confirmation logic with majority-vote class labels."""

from __future__ import annotations

from collections import Counter

from app.schemas import Track, UniqueDetection


def finalize_tracks(
    tracks: list[Track],
    class_agreement_ratio: float = 0.7,
    filenames: list[str] | None = None,
) -> tuple[dict[str, int], list[Track], list[UniqueDetection]]:
    """Assign final labels and produce summary counts.

    Returns (summary_counts, annotated_tracks, unique_detections).
    Only confirmed tracks with sufficient class agreement are counted.
    """
    summary: dict[str, int] = {}
    annotated: list[Track] = []
    unique_detections: list[UniqueDetection] = []

    for track in tracks:
        if not track.confirmed:
            annotated.append(track)
            continue

        # Majority-vote class label
        label_counts = Counter(obs.label for obs in track.observations)
        top_label, top_count = label_counts.most_common(1)[0]
        agreement = top_count / len(track.observations)

        if agreement >= class_agreement_ratio:
            track.final_label = top_label
            track.counted = True
            first_obs = track.observations[0]
            track.first_seen_frame = first_obs.frame_index
            summary[top_label] = summary.get(top_label, 0) + 1

            filename = ""
            if filenames and 0 <= first_obs.frame_index < len(filenames):
                filename = filenames[first_obs.frame_index]

            unique_detections.append(
                UniqueDetection(
                    track_id=track.track_id,
                    label=top_label,
                    confidence=first_obs.confidence,
                    bbox=first_obs.bbox,
                    frame_index=first_obs.frame_index,
                    filename=filename,
                )
            )

        annotated.append(track)

    summary["total_detections"] = sum(v for k, v in summary.items() if k != "total_detections")
    return summary, annotated, unique_detections
