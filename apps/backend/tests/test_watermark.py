"""
WatermarkNode birim testleri
"""

import pytest
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path

# Backend paketini path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from nodes.watermark import WatermarkNode
from core.models import NodeResult


class TestWatermarkNode:
    """WatermarkNode için test sınıfı"""

    @pytest.fixture
    def node(self):
        """Test watermark node'u"""
        return WatermarkNode(weight=1.0, threshold=0.5)

    @pytest.fixture
    def sample_image(self):
        """Örnek RGB görsel (50x50)"""
        return np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

    @pytest.fixture
    def real_photo(self):
        """Gerçek fotoğraf simülasyonu (daha doğal renkler)"""
        # Gradient pattern
        x = np.linspace(0, 255, 100)
        y = np.linspace(0, 255, 100)
        xx, yy = np.meshgrid(x, y)
        r = (xx + yy) / 2
        g = xx
        b = yy
        return np.stack([r, g, b], axis=2).astype(np.uint8)

    @pytest.fixture
    def blank_image(self):
        """Boş (siyah) görsel - edge case"""
        return np.zeros((10, 10, 3), dtype=np.uint8)

    @pytest.fixture
    def single_channel_image(self):
        """Tek kanallı (grayscale) görsel - edge case"""
        return np.random.randint(0, 255, (50, 50), dtype=np.uint8)

    def test_node_initialization(self, node):
        """Node başlatma testi"""
        assert node.name == "WatermarkNode"
        assert node.weight == 1.0
        assert node.threshold == 0.5
        assert node.enabled is True

    def test_analyze_returns_valid_result(self, node, sample_image):
        """Analiz geçerli sonuç döndürmeli"""
        result = node.analyze(sample_image)

        assert isinstance(result, NodeResult)
        assert isinstance(result.score, (int, float))
        assert 0 <= result.score <= 100
        assert result.verdict in ["REAL", "FAKE", "UNCERTAIN"]
        assert result.node_name == "WatermarkNode"
        assert isinstance(result.metadata, dict)

    def test_analyze_real_photo(self, node, real_photo):
        """Gerçek fotoğraf için REAL kararı vermeli"""
        result = node.analyze(real_photo)

        # Watermark yoksa, düşük AI olasılığı vermeli
        assert result.verdict == "REAL"
        assert result.score < 50
        assert result.metadata.get("watermark_detected", False) is False

    def test_invalid_input_empty_image(self, node, blank_image):
        """Boş görsel için hata vermeli"""
        # Sıfır boyutlu görsel
        empty = np.array([], dtype=np.uint8)

        with pytest.raises(ValueError, match="Görsel boş olamaz"):
            node.analyze(empty)

    def test_invalid_input_wrong_type(self, node):
        """Yanlış tip için hata vermeli"""
        with pytest.raises(ValueError, match="NumPy array olmalı"):
            node.analyze("not_an_image")

        with pytest.raises(ValueError, match="NumPy array olmalı"):
            node.analyze(123)

    def test_invalid_input_wrong_dimensions(self, node):
        """Yanlış boyutlar için hata vermeli"""
        # 4D görsel
        image_4d = np.random.randint(0, 255, (10, 10, 3, 2), dtype=np.uint8)

        with pytest.raises(ValueError, match="2D veya 3D olmalı"):
            node.analyze(image_4d)

    def test_single_channel_image(self, node, single_channel_image):
        """Tek kanallı görsel analiz edilebilmeli"""
        # 2D (grayscale) görsel kabul edilmeli
        result = node.analyze(single_channel_image)

        assert isinstance(result, NodeResult)
        assert result.verdict in ["REAL", "FAKE", "UNCERTAIN"]

    def test_metadata_structure(self, node, sample_image):
        """Metadata yapısı kontrolü"""
        result = node.analyze(sample_image)

        # Zorunlu metadata alanları
        assert "check_watermark" in result.metadata
        assert "check_exif" in result.metadata

    def test_weight_boundary(self):
        """Ağırlık sınır testi"""
        # Geçersiz ağırlıklar otomatik sınırlandırılmalı
        node1 = WatermarkNode(weight=-1.0)
        assert node1.weight == 0.0

        node2 = WatermarkNode(weight=2.0)
        assert node2.weight == 1.0

    def test_disabled_node(self, node, sample_image):
        """Pasif node testi"""
        node.enabled = False
        # Node hala analyze metoduna sahip olmalı
        # Ancak EnsembleEngine tarafından kullanılmayacak
        assert node.enabled is False

    def test_different_image_sizes(self, node):
        """Farklı görsel boyutları testi"""
        sizes = [(10, 10), (100, 100), (512, 512), (1080, 1920)]

        for height, width in sizes:
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            result = node.analyze(image)

            assert isinstance(result, NodeResult)
            assert 0 <= result.score <= 100

    def test_float_image_conversion(self, node):
        """Float (0-1) görsel uint8'e çevrilmeli"""
        # Float görsel (0-1 arası)
        float_image = np.random.rand(50, 50, 3).astype(np.float32)

        # Hata vermemeli
        result = node.analyze(float_image)

        assert isinstance(result, NodeResult)

    def test_watermark_signatures_list(self, node):
        """AI yazılım imzaları listesi kontrolü"""
        assert hasattr(node, "AI_SOFTWARE_SIGNATURES")
        assert isinstance(node.AI_SOFTWARE_SIGNATURES, list)
        assert len(node.AI_SOFTWARE_SIGNATURES) > 0

        # Bilinen AI araçları listelenmeli
        signatures_str = " ".join(node.AI_SOFTWARE_SIGNATURES).lower()
        assert "stable diffusion" in signatures_str
        assert "midjourney" in signatures_str
        assert "dall-e" in signatures_str

    def test_preprocess_no_change(self, node, sample_image):
        """Preprocess görseli değiştirmemeli"""
        processed = node.preprocess(sample_image)

        # Aynı array olmalı (değişiklik yok)
        assert np.array_equal(processed, sample_image)


@pytest.mark.integration
class TestWatermarkNodeIntegration:
        """WatermarkNode entegrasyon testleri"""

    @pytest.fixture
    def ensemble(self):
        """EnsembleEngine örneği"""
        from core.ensemble import EnsembleEngine
        from nodes.watermark import WatermarkNode

        watermark_node = WatermarkNode(weight=1.0)
        return EnsembleEngine(nodes=[watermark_node], threshold=50.0)

    @pytest.fixture
    def sample_image(self):
        """Örnek görsel"""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def test_ensemble_with_watermark_node(self, ensemble, sample_image):
        """Ensemble ile WatermarkNode entegrasyonu"""
        result = ensemble.analyze(sample_image)

        assert result is not None
        assert isinstance(result.final_score, (int, float))
        assert 0 <= result.final_score <= 100
        assert result.verdict in ["REAL", "FAKE", "UNCERTAIN"]
        assert "WatermarkNode" in result.node_results

    def test_ensemble_processing_time(self, ensemble, sample_image):
        """İşleme süresi ölçümü"""
        import time

        start = time.time()
        result = ensemble.analyze(sample_image)
        duration = time.time() - start

        assert result.processing_time > 0
        assert duration < 5.0  # 5 saniyeden kısa sürmeli


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
