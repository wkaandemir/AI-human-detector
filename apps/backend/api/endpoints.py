"""
FastAPI endpoint'leri
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path

# Backend paketini path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ensemble import EnsembleEngine
from nodes.watermark import WatermarkNode
from nodes.frequency import FrequencyNode
from nodes.clip import CLIPNode
from nodes.dire import DIRENode
from api.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    NodeResultResponse,
    HealthResponse,
    ErrorResponse
)

router = APIRouter()

# Global ensemble engine (lazy initialization)
_ensemble_engine = None


def get_ensemble() -> EnsembleEngine:
    """
    Ensemble engine'i lazy load ile döndürür.

    Returns:
        EnsembleEngine örneği
    """
    global _ensemble_engine

    if _ensemble_engine is None:
        # Node'ları oluştur
        watermark_node = WatermarkNode(weight=1.0)
        frequency_node = FrequencyNode(weight=1.0)
        clip_node = CLIPNode(weight=1.0)
        dire_node = DIRENode(weight=1.0)

        # Ensemble motorunu oluştur
        _ensemble_engine = EnsembleEngine(
            nodes=[watermark_node, frequency_node, clip_node, dire_node],
            threshold=50.0
        )

    return _ensemble_engine


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    API sağlık kontrolü endpoint'i.

    Returns:
        HealthResponse: Sistem durumu
    """
    import torch

    ensemble = get_ensemble()

    # Model durumlarını kontrol et
    models_loaded = {
        "WatermarkNode": True,  # Her zaman hazır
        "FrequencyNode": True,  # Her zaman hazır
        "CLIPNode": CLIPNode.is_available(ensemble.nodes[1]),
        "DIRENode": DIRENode.is_available(ensemble.nodes[3])
    }

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        models_loaded=models_loaded,
        gpu_available=torch.cuda.is_available() if torch else False
    )


@router.get("/models")
async def list_models():
    """
    Kullanılabilir node'ları listeler.

    Returns:
        Node listesi ve durumları
    """
    ensemble = get_ensemble()

    nodes_info = []
    for node in ensemble.nodes:
        info = {
            "name": node.name,
            "enabled": node.enabled,
            "weight": node.weight
        }

        # Node tipine özgü bilgiler
        if hasattr(node, "get_model_info"):
            info["model_info"] = node.get_model_info()

        nodes_info.append(info)

    return {
        "nodes": nodes_info,
        "threshold": ensemble.threshold
    }


@router.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    image: UploadFile = File(..., description="Analiz edilecek görsel"),
    check_metadata: bool = Form(True),
    return_details: bool = Form(True),
    enable_nodes: Optional[str] = Form(None)
):
    """
    Görseli analiz eder ve AI-generated tespiti yapar.

    Args:
        image: Yüklenecek görsel dosyası
        check_metadata: Metadata analizi yapılıp yapılmayacağı
        return_details: Detaylı node sonuçlarının döndürülüp döndürülmeyeceği
        enable_nodes: Aktif edilecek node'lar (virgülle ayrılmış)

    Returns:
        AnalyzeResponse: Analiz sonucu

    Raises:
        HTTPException: Görsel işleme hatası
    """
    # Görseli oku
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))

        # RGB'ye çevir
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # NumPy array'e çevir
        image_array = np.array(pil_image)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Görsel okuma hatası: {str(e)}"
        )

    # Görseli doğrula
    if image_array.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Boş görsel"
        )

    # Ensemble motorunu al
    ensemble = get_ensemble()

    # Node'ları filtrele (opsiyonel)
    if enable_nodes:
        enabled_list = [n.strip() for n in enable_nodes.split(",")]
        for node in ensemble.nodes:
            if node.name not in enabled_list:
                node.enabled = False
            else:
                node.enabled = True

    # Analiz yap
    try:
        result = ensemble.analyze(image_array)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analiz hatası: {str(e)}"
        )

    # Node sonuçlarını hazırla
    details = None
    if return_details:
        details = {}
        for node_name, node_result in result.node_results.items():
            details[node_name] = NodeResultResponse(
                score=node_result.score,
                verdict=node_result.verdict,
                confidence=node_result.confidence,
                node_name=node_result.node_name,
                processing_time=node_result.processing_time,
                metadata=node_result.metadata
            )

    # Response oluştur
    response = AnalyzeResponse(
        final_score=result.final_score,
        verdict=result.verdict,
        processing_time=result.processing_time,
        timestamp=result.timestamp,
        details=details
    )

    return response


@router.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc)
        }
    )
