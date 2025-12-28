"""
Veri modelleri - NodeResult, EnsembleResult
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class NodeResult:
    """
    Bir analiz node'unun sonucunu temsil eder.

    Attributes:
        score: 0-100 arası AI olasılık skoru
        verdict: Karar ("REAL", "FAKE", "UNCERTAIN")
        metadata: Node'a özgü ek bilgiler
        confidence: Güven seviyesi (0-1)
        node_name: Node adı
        processing_time: İşleme süresi (saniye)
    """
    score: float
    verdict: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    node_name: str = "unknown"
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Sözlük formatına çevirir"""
        return {
            "score": self.score,
            "verdict": self.verdict,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "node_name": self.node_name,
            "processing_time": self.processing_time
        }


@dataclass
class EnsembleResult:
    """
    Tüm node'ların birleşik sonucunu temsil eder.

    Attributes:
        final_score: 0-100 arası final AI olasılık skoru
        verdict: Final karar ("REAL", "FAKE", "UNCERTAIN")
        node_results: Her bir node'un ayrıntılı sonuçları
        processing_time: Toplam işleme süresi (saniye)
        timestamp: Analiz zamanı
    """
    final_score: float
    verdict: str
    node_results: Dict[str, NodeResult] = field(default_factory=dict)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Sözlük formatına çevirir"""
        return {
            "final_score": self.final_score,
            "verdict": self.verdict,
            "details": {
                name: result.to_dict()
                for name, result in self.node_results.items()
            },
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ImageMetadata:
    """
    Görsel metadatası için veri modeli

    Attributes:
        format: Görsel formatı (PNG, JPG, vb.)
        size: Görsel boyutu (width, height)
        mode: Görsel modu (RGB, RGBA, vb.)
        exif: EXIF verileri (varsa)
        file_size: Dosya boyutu (byte)
    """
    format: Optional[str] = None
    size: tuple = (0, 0)
    mode: Optional[str] = None
    exif: Dict[str, Any] = field(default_factory=dict)
    file_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Sözlük formatına çevirir"""
        return {
            "format": self.format,
            "size": self.size,
            "mode": self.mode,
            "exif": self.exif,
            "file_size": self.file_size
        }
