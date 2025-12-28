"""
API Pydantic modelleri
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class AnalyzeRequest(BaseModel):
    """Görsel analiz isteği modeli"""
    check_metadata: bool = Field(
        default=True,
        description="Metadata analizi yapılıp yapılmayacağı"
    )
    return_details: bool = Field(
        default=True,
        description="Detaylı node sonuçlarının döndürülüp döndürülmeyeceği"
    )
    enable_nodes: Optional[list[str]] = Field(
        default=None,
        description="Aktif edilecek node'lar (None = tümü)"
    )


class NodeResultResponse(BaseModel):
    """Node sonucu response modeli"""
    score: float
    verdict: str
    confidence: float
    node_name: str
    processing_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    """Görsel analizi response modeli"""
    final_score: float
    verdict: str
    processing_time: float
    timestamp: datetime
    details: Optional[Dict[str, NodeResultResponse]] = None


class HealthResponse(BaseModel):
    """Sağlık kontrolü response modeli"""
    status: str
    version: str
    models_loaded: Dict[str, bool]
    gpu_available: bool


class ErrorResponse(BaseModel):
    """Hata response modeli"""
    error: str
    detail: Optional[str] = None
