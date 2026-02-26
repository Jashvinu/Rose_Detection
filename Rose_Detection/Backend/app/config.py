from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "ROSE_", "protected_namespaces": ("settings_",)}

    model_dir: Path = Path(__file__).parent / "model"
    model_filename: str = "rose_disease_model.tflite"
    labels_filename: str = "labels.txt"

    confidence_threshold: float = 0.5
    nms_iou_threshold: float = 0.45

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_filename

    @property
    def labels_path(self) -> Path:
        return self.model_dir / self.labels_filename


@lru_cache
def get_settings() -> Settings:
    return Settings()
