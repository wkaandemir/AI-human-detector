"""
FastAPI birim ve entegrasyon testleri
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path

# Backend paketini path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app


@pytest.fixture
def client():
    """Test client'ı"""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Örnek görsel byte'ları"""
    image = Image.new("RGB", (512, 512), color="red")
    byte_io = io.BytesIO()
    image.save(byte_io, format="PNG")
    byte_io.seek(0)
    return byte_io.read()


@pytest.fixture
def sample_image_jpeg():
    """Örnek JPEG görsel byte'ları"""
    image = Image.new("RGB", (512, 512), color="blue")
    byte_io = io.BytesIO()
    image.save(byte_io, format="JPEG")
    byte_io.seek(0)
    return byte_io.read()


class TestAPIEndpoints:
    """API endpoint testleri"""

    def test_root_endpoint(self, client):
        """Root endpoint testi"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "AI Human Detector API"
        assert data["version"] == "0.1.0"
        assert "endpoints" in data

    def test_ping_endpoint(self, client):
        """Ping endpoint testi"""
        response = client.get("/ping")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pong"

    def test_health_check(self, client):
        """Sağlık kontrolü testi"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "models_loaded" in data
        assert "gpu_available" in data

    def test_list_models(self, client):
        """Model listesi endpoint testi"""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert isinstance(data["nodes"], list)
        assert "threshold" in data

        # Node'lar listelenmeli
        node_names = [n["name"] for n in data["nodes"]]
        assert "WatermarkNode" in node_names
        assert "FrequencyNode" in node_names
        assert "CLIPNode" in node_names
        assert "DIRENode" in node_names


class TestAnalyzeEndpoint:
    """Analyze endpoint testleri"""

    def test_analyze_with_png(self, client, sample_image_bytes):
        """PNG görsel analizi testi"""
        files = {"image": ("test.png", sample_image_bytes, "image/png")}
        data = {
            "check_metadata": "true",
            "return_details": "true"
        }

        response = client.post("/api/v1/analyze", files=files, data=data)

        assert response.status_code == 200
        result = response.json()
        assert "final_score" in result
        assert "verdict" in result
        assert "processing_time" in result
        assert "timestamp" in result
        assert "details" in result

        # Skor kontrolü
        assert 0 <= result["final_score"] <= 100
        assert result["verdict"] in ["REAL", "FAKE", "UNCERTAIN"]

    def test_analyze_with_jpeg(self, client, sample_image_jpeg):
        """JPEG görsel analizi testi"""
        files = {"image": ("test.jpg", sample_image_jpeg, "image/jpeg")}
        data = {
            "check_metadata": "true",
            "return_details": "true"
        }

        response = client.post("/api/v1/analyze", files=files, data=data)

        assert response.status_code == 200
        result = response.json()
        assert "final_score" in result

    def test_analyze_without_details(self, client, sample_image_bytes):
        """Detaysız analiz testi"""
        files = {"image": ("test.png", sample_image_bytes, "image/png")}
        data = {
            "return_details": "false"
        }

        response = client.post("/api/v1/analyze", files=files, data=data)

        assert response.status_code == 200
        result = response.json()
        # Details olmamalı
        assert result.get("details") is None

    def test_analyze_with_enabled_nodes(self, client, sample_image_bytes):
        """Spesifik node'larla analiz testi"""
        files = {"image": ("test.png", sample_image_bytes, "image/png")}
        data = {
            "enable_nodes": "WatermarkNode,FrequencyNode"
        }

        response = client.post("/api/v1/analyze", files=files, data=data)

        assert response.status_code == 200
        result = response.json()
        assert "final_score" in result

    def test_analyze_without_image(self, client):
        """Görsel olmadan analiz testi (hata)"""
        response = client.post("/api/v1/analyze")

        assert response.status_code == 422  # Unprocessable Entity

    def test_analyze_with_invalid_image(self, client):
        """Geçersiz görsel testi"""
        files = {"image": ("test.txt", b"not an image", "text/plain")}

        response = client.post("/api/v1/analyze", files=files)

        # Hata dönmeli
        assert response.status_code in [400, 422, 500]

    def test_analyze_response_structure(self, client, sample_image_bytes):
        """Analiz response yapısı testi"""
        files = {"image": ("test.png", sample_image_bytes, "image/png")}
        data = {"return_details": "true"}

        response = client.post("/api/v1/analyze", files=files, data=data)

        assert response.status_code == 200
        result = response.json()

        # Ana yapı
        assert "final_score" in result
        assert "verdict" in result
        assert "processing_time" in result
        assert "timestamp" in result
        assert "details" in result

        # Detaylar
        details = result["details"]
        if details:
            # Node sonuçları kontrolü
            for node_name, node_result in details.items():
                assert "score" in node_result
                assert "verdict" in node_result
                assert "confidence" in node_result
                assert "node_name" in node_result
                assert "processing_time" in node_result
                assert "metadata" in node_result


class TestAPIDocs:
    """API dokümantasyon testleri"""

    def test_swagger_docs(self, client):
        """Swagger UI testi"""
        response = client.get("/docs")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_redoc_docs(self, client):
        """ReDoc testi"""
        response = client.get("/redoc")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_openapi_schema(self, client):
        """OpenAPI şeması testi"""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema


@pytest.mark.integration
class TestAPIIntegration:
    """API entegrasyon testleri"""

    def test_full_workflow(self, client, sample_image_bytes):
        """Tam iş akışı testi"""
        # 1. Health check
        health = client.get("/health")
        assert health.status_code == 200

        # 2. Model listesi
        models = client.get("/models")
        assert models.status_code == 200

        # 3. Analiz
        files = {"image": ("test.png", sample_image_bytes, "image/png")}
        analyze = client.post("/api/v1/analyze", files=files)
        assert analyze.status_code == 200
        result = analyze.json()
        assert "final_score" in result

    def test_concurrent_requests(self, client, sample_image_bytes):
        """Eş zamanlı istek testi"""
        import concurrent.futures

        def analyze():
            files = {"image": ("test.png", sample_image_bytes, "image/png")}
            response = client.post("/api/v1/analyze", files=files)
            return response

        # 3 eş zamanlı istek
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(analyze) for _ in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Hepsi başarılı olmalı
        for response in results:
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
