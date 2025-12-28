"""
CLIPNode birim testleri
"""

import pytest
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Backend paketini path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from nodes.clip import CLIPNode, CLIP_AVAILABLE
from core.models import NodeResult


@pytest.mark.skipif(not CLIP_AVAILABLE, reason="CLIP modeli kurulmadı")
class TestCLIPNode:
    """CLIPNode için test sınıfı"""

    @pytest.fixture
    def node(self):
        """Test CLIP node'u (hafif model)"""
        # Daha hızlı test için küçük model
        return CLIPNode(
            weight=1.0,
            model_name="openai/clip-vit-base-patch32",
            threshold=0.5
        )

    @pytest.fixture
    def sample_image(self):
        """Örnek RGB görsel (224x224 - CLIP input size)"""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    @pytest.fixture
    def natural_image(self):
        """Doğal görüntü"""
        x = np.linspace(0, 255, 224)
        y = np.linspace(0, 255, 224)
        xx, yy = np.meshgrid(x, y)
        r = (xx + yy) / 2
        g = xx
        b = yy
        return np.stack([r, g, b], axis=2).astype(np.uint8)

    def test_node_initialization(self, node):
        """Node başlatma testi"""
        assert node.name == "CLIPNode"
        assert node.weight == 1.0
        assert node.threshold == 0.5
        assert node.enabled is True
        assert node.model_name == "openai/clip-vit-base-patch32"

    def test_is_available(self, node):
        """Model kullanılabilirlik testi"""
        assert node.is_available() is True

    def test_get_model_info(self, node):
        """Model bilgileri testi"""
        info = node.get_model_info()

        assert isinstance(info, dict)
        assert "available" in info
        assert "model_name" in info
        assert "calibrated" in info
        assert info["available"] is True

    def test_analyze_returns_valid_result(self, node, sample_image):
        """Analiz geçerli sonuç döndürmeli"""
        result = node.analyze(sample_image)

        assert isinstance(result, NodeResult)
        assert isinstance(result.score, (int, float))
        assert 0 <= result.score <= 100
        assert result.verdict in ["REAL", "FAKE", "UNCERTAIN"]
        assert result.node_name == "CLIPNode"
        assert isinstance(result.metadata, dict)

    def test_metadata_structure(self, node, sample_image):
        """Metadata yapısı kontrolü"""
        result = node.analyze(sample_image)

        # Zorunlu metadata alanları
        assert "embedding_mean" in result.metadata
        assert "embedding_std" in result.metadata
        assert "embedding_norm" in result.metadata
        assert "model_name" in result.metadata
        assert "available" in result.metadata
        assert result.metadata["available"] is True

    def test_get_embedding(self, node, sample_image):
        """Embedding çıkarma testi"""
        embedding = node._get_embedding(sample_image)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1  # 1D vektör
        assert embedding.size > 0  # Boş değil

    def test_embedding_statistics(self, node, sample_image):
        """Embedding istatistikleri testi"""
        result = node.analyze(sample_image)

        metadata = result.metadata
        assert isinstance(metadata["embedding_mean"], float)
        assert isinstance(metadata["embedding_std"], float)
        assert isinstance(metadata["embedding_norm"], float)
        assert metadata["embedding_norm"] > 0

    def test_not_calibrated_behavior(self, node, sample_image):
        """Kalibre edilmemiş model davranışı"""
        # Model kalibre edilmemişse UNCERTAIN dönmeli
        assert node._is_calibrated is False

        result = node.analyze(sample_image)

        # Kalibre edilmemişse UNCERTAIN veya nötr sonuç
        assert result.metadata["calibrated"] is False

    def test_calibrate_method(self, node):
        """Kalibrasyon metodu testi"""
        real_images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(3)
        ]

        # Kalibre et
        node.calibrate(real_images)

        # Kalibre olmuş olmalı
        assert node._is_calibrated is True
        assert len(node._real_embeddings) == 3

    def test_calibrate_with_fake_images(self, node):
        """Sahte görsellerle kalibrasyon testi"""
        real_images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(2)
        ]
        fake_images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(2)
        ]

        node.calibrate(real_images, fake_images)

        assert len(node._real_embeddings) == 2
        assert len(node._fake_embeddings) == 2

    def test_anomaly_score_computation(self, node):
        """Anomali skoru hesaplama testi"""
        # Önce kalibre et
        real_images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        node.calibrate(real_images)

        # Test embedding'i al
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        embedding = node._get_embedding(test_image)

        # Anomali skoru hesapla
        score = node._compute_anomaly_score(embedding)

        assert isinstance(score, (int, float))
        assert 0 <= score <= 1

    def test_different_thresholds(self):
        """Farklı eşik değerleri testi"""
        node1 = CLIPNode(threshold=0.3)
        assert node1.threshold == 0.3

        node2 = CLIPNode(threshold=1.5)
        assert node2.threshold == 1.0  # Sınırlandırıldı

    def test_invalid_input_empty_image(self, node):
        """Boş görsel için hata vermeli"""
        empty = np.array([], dtype=np.uint8)

        with pytest.raises(ValueError, match="Görsel boş olamaz"):
            node.analyze(empty)

    def test_invalid_input_wrong_type(self, node):
        """Yanlış tip için hata vermeli"""
        with pytest.raises(ValueError, match="NumPy array olmalı"):
            node.analyze("not_an_image")

    def test_different_image_sizes(self, node):
        """Farklı görsel boyutları testi"""
        # CLIP her boyutu kabul eder (resize eder)
        sizes = [(100, 100), (224, 224), (512, 512)]

        for height, width in sizes:
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            result = node.analyze(image)

            assert isinstance(result, NodeResult)

    def test_float_image_conversion(self, node):
        """Float (0-1) görsel uint8'e çevrilmeli"""
        float_image = np.random.rand(224, 224, 3).astype(np.float32)
        result = node.analyze(float_image)

        assert isinstance(result, NodeResult)

    def test_preprocess_no_change(self, node, sample_image):
        """Preprocess görseli değiştirmemeli"""
        processed = node.preprocess(sample_image)
        assert np.array_equal(processed, sample_image)


@pytest.mark.integration
class TestCLIPNodeIntegration:
    """CLIPNode entegrasyon testleri"""

    @pytest.fixture
    def ensemble(self):
        """EnsembleEngine örneği"""
        from core.ensemble import EnsembleEngine
        from nodes.watermark import WatermarkNode
        from nodes.frequency import FrequencyNode
        from nodes.clip import CLIPNode

        watermark_node = WatermarkNode(weight=1.0)
        frequency_node = FrequencyNode(weight=1.0)
        clip_node = CLIPNode(weight=1.0)

        return EnsembleEngine(
            nodes=[watermark_node, frequency_node, clip_node],
            threshold=50.0
        )

    @pytest.fixture
    def sample_image(self):
        """Örnek görsel"""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    @pytest.mark.skipif(not CLIP_AVAILABLE, reason="CLIP modeli kurulmadı")
    def test_ensemble_with_clip_node(self, ensemble, sample_image):
        """Ensemble ile CLIPNode entegrasyonu"""
        result = ensemble.analyze(sample_image)

        assert result is not None
        assert isinstance(result.final_score, (int, float))
        assert 0 <= result.final_score <= 100
        assert "CLIPNode" in result.node_results

    @pytest.mark.skipif(not CLIP_AVAILABLE, reason="CLIP modeli kurulmadı")
    def test_all_nodes_contributed(self, ensemble, sample_image):
        """Her üç node da katkı sağlamalı"""
        result = ensemble.analyze(sample_image)

        assert "WatermarkNode" in result.node_results
        assert "FrequencyNode" in result.node_results
        assert "CLIPNode" in result.node_results


class TestCLIPNodeWithoutModel:
    """CLIP modeli yokken testler"""

    def test_node_initialization_without_model(self):
        """Model yoksa node başlatılmalı"""
        node = CLIPNode(weight=1.0)
        assert node.name == "CLIPNode"
        assert node.is_available() == CLIP_AVAILABLE

    def test_analyze_without_model(self):
        """Model yoksa analyze hata vermemeli,UNCERTAIN dönmeli"""
        if CLIP_AVAILABLE:
            pytest.skip("CLIP modeli kurulu, test atlanıyor")

        node = CLIPNode(weight=1.0)
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        result = node.analyze(image)

        assert result.verdict == "UNCERTAIN"
        assert result.metadata["available"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
