from pydantic import BaseSettings
from typing import Optional

class Config(BaseSettings):
    """Конфигурация приложения"""
    
    # Размеры изображений
    TARGET_SIZE: int = 1280
    
    # Пороги детекции
    DETECTION_CONFIDENCE: float = 0.45
    DETECTION_IOU: float = 0.5
    SEGMENTATION_CONFIDENCE: float = 0.25
    SEGMENTATION_IOU: float = 0.5
    
    # Пути к моделям (по умолчанию)
    DEFAULT_DETECT_MODEL: str = "models/v1/yolo_detect_corpus/detect_n_model/weights/best.pt"
    DEFAULT_SEGMENT_MODEL: str = "models/v1/yolo_segment_lines/segment_n_model/weights/best.pt"
    
    class Config:
        env_file = ".env"

config = Config()
