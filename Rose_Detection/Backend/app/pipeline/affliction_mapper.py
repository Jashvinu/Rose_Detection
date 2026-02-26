"""Map TFLite model labels to PlantVillage affliction IDs and derive condition."""

from __future__ import annotations

from app.schemas import Affliction, UniqueDetection

# PlantVillage affliction definitions mapped to YOLO11s label names.
# healthy_leaf is intentionally absent — it produces no affliction.
LABEL_TO_AFFLICTION: dict[str, Affliction] = {
    "downy_mildew_leaf": Affliction(id=1, name="Downy Mildew"),
    "powdery_mildew_leaf": Affliction(id=2, name="Powdery Mildew"),
    "two_spotted_spider_mite_damage_leaf": Affliction(id=3, name="Mite"),
    "unknown_disease_leaf": Affliction(id=5, name="Unknown Disease"),
    "chemical_residue_leaf": Affliction(id=6, name="Chemical Residue"),
}


def map_results(
    unique_detections: list[UniqueDetection],
) -> tuple[list[Affliction], str]:
    """Convert unique detections into PlantVillage afflictions and a condition.

    Returns:
        (afflictions, condition) where condition is one of
        "healthy", "new", or "existing".
    """
    if not unique_detections:
        return [], "healthy"

    seen: dict[int, Affliction] = {}
    for det in unique_detections:
        aff = LABEL_TO_AFFLICTION.get(det.label)
        if aff and aff.id not in seen:
            seen[aff.id] = aff

    afflictions = sorted(seen.values(), key=lambda a: a.id)
    condition = "new" if afflictions else "healthy"
    return afflictions, condition


def parse_pv_token(token: str) -> dict[str, int | None]:
    """Parse a PlantVillage-style token into context fields.

    Token format: "scout:{id}|partition:{id}|bed:{id}"
    Returns dict with scout_id, partition_id, bed_id (None if missing or invalid).
    """
    result: dict[str, int | None] = {
        "scout_id": None,
        "partition_id": None,
        "bed_id": None,
    }

    if "|" not in token:
        return result

    for segment in token.split("|"):
        if ":" not in segment:
            continue
        key, _, value = segment.partition(":")
        key = key.strip()
        try:
            int_val = int(value.strip())
        except ValueError:
            continue

        if key == "scout":
            result["scout_id"] = int_val
        elif key == "partition":
            result["partition_id"] = int_val
        elif key == "bed":
            result["bed_id"] = int_val

    return result
