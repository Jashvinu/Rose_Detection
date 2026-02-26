"""Centralized pipeline settings, overridable via ROSE_ env vars."""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "ROSE_", "protected_namespaces": ("settings_",)}

    # Paths
    model_dir: Path = Path(__file__).parent / "model"
    model_filename: str = "rose_disease_model.tflite"
    labels_filename: str = "labels.txt"

    confidence_threshold: float = 0.5

    # NMS
    nms_iou_threshold: float = 0.45

    # Tracker
    match_iou_threshold: float = 0.3
    max_age: int = 10
    min_hits: int = 3

    # Counter
    class_agreement_ratio: float = 0.7

    # Motion compensation
    motion_compensation: bool = True
    motion_max_features: int = 500
    motion_min_matches: int = 10

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_filename

    @property
    def labels_path(self) -> Path:
        return self.model_dir / self.labels_filename


@lru_cache
def get_settings() -> Settings:
    return Settings()
