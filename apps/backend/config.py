"""
Yapılandırma dosyası
"""

import os
from typing import Optional


class Config:
    """Uygulama yapılandırması"""

    # API Ayarları
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Model Ayarları
    MODEL_CACHE_DIR: str = os.getenv(
        "MODEL_CACHE_DIR",
        os.path.expanduser("~/.cache/huggingface")
    )

    # CLIP Model
    CLIP_MODEL_NAME: str = os.getenv(
        "CLIP_MODEL_NAME",
        "openai/clip-vit-base-patch32"
    )

    # Stable Diffusion Model
    SD_MODEL_NAME: str = os.getenv(
        "SD_MODEL_NAME",
        "runwayml/stable-diffusion-v1-5"
    )

    # DIRE Ayarları
    DIRE_NUM_STEPS: int = int(os.getenv("DIRE_NUM_STEPS", "50"))
    DIRE_DEVICE: Optional[str] = os.getenv("DIRE_DEVICE", None)  # None = auto

    # Ensemble Ayarları
    ENSEMBLE_THRESHOLD: float = float(os.getenv("ENSEMBLE_THRESHOLD", "50.0"))

    # Node Ağırlıkları
    WATERMARK_WEIGHT: float = float(os.getenv("WATERMARK_WEIGHT", "1.0"))
    FREQUENCY_WEIGHT: float = float(os.getenv("FREQUENCY_WEIGHT", "1.0"))
    CLIP_WEIGHT: float = float(os.getenv("CLIP_WEIGHT", "1.0"))
    DIRE_WEIGHT: float = float(os.getenv("DIRE_WEIGHT", "1.0"))

    # Upload Ayarları
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", "10485760"))  # 10MB
    ALLOWED_IMAGE_FORMATS: list = [
        ".jpg", ".jpeg", ".png", ".webp", ".bmp"
    ]

    # HuggingFace Token (opsiyonel, private modeller için)
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN", None)


class DevelopmentConfig(Config):
    """Development ortamı yapılandırması"""
    DEBUG: bool = True


class ProductionConfig(Config):
    """Production ortamı yapılandırması"""
    DEBUG: bool = False


def get_config() -> Config:
    """
    Ortama göre yapılandırma döndürür.

    Returns:
        Config: Yapılandırma örneği
    """
    env = os.getenv("ENV", "development")

    if env == "production":
        return ProductionConfig()
    else:
        return DevelopmentConfig()


# Global config instance
config = get_config()
