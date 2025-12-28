"""
FrequencyNode birim testleri
"""

import pytest
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Backend paketini path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from nodes.frequency import FrequencyNode
from core.models import NodeResult


class TestFrequencyNode:
    """FrequencyNode için test sınıfı"""

    @pytest.fixture
    def node(self):
        """Test frequency node'u"""
        return FrequencyNode(weight=1.0)

    @pytest.fixture
    def sample_image(self):
        """Örnek RGB görsel (50x50)"""
        return np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

    @pytest.fixture
    def natural_image(self):
        """Doğal görüntü (smooth gradient)"""
        x = np.linspace(0, 255, 100)
        y = np.linspace(0, 255, 100)
        xx, yy = np.meshgrid(x, y)
        r = (xx + yy) / 2
        g = xx
        b = yy
        return np.stack([r, g, b], axis=2).astype(np.uint8)

    @pytest.fixture
    def checkerboard_image(self):
        """Checkerboard pattern (AI artifact simülasyonu)"""
        size = 100
        checkerboard = np.zeros((size, size, 3), dtype=np.uint8)
        square_size = 8

        for i in range(0, size, square_size):
            for j in range(0, size, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    checkerboard[i:i+square_size, j:j+square_size] = 255

        return checkerboard

    def test_node_initialization(self, node):
        """Node başlatma testi"""
        assert node.name == "FrequencyNode"
        assert node.weight == 1.0
        assert node.fft_threshold == 0.7
        assert node.ela_threshold == 20.0
        assert node.ela_quality == 90
        assert node.enabled is True

    def test_analyze_returns_valid_result(self, node, sample_image):
        """Analiz geçerli sonuç döndürmeli"""
        result = node.analyze(sample_image)

        assert isinstance(result, NodeResult)
        assert isinstance(result.score, (int, float))
        assert 0 <= result.score <= 100
        assert result.verdict in ["REAL", "FAKE", "UNCERTAIN"]
        assert result.node_name == "FrequencyNode"
        assert isinstance(result.metadata, dict)

    def test_metadata_structure(self, node, sample_image):
        """Metadata yapısı kontrolü"""
        result = node.analyze(sample_image)

        # FFT metadata
        assert "fft" in result.metadata
        assert "ela" in result.metadata
        assert "fft_score" in result.metadata
        assert "ela_score" in result.metadata
        assert "combined_score" in result.metadata

    def test_fft_analysis(self, node, sample_image):
        """FFT analizi testi"""
        fft_score, fft_metadata = node._analyze_fft(sample_image)

        assert isinstance(fft_score, (int, float))
        assert 0 <= fft_score <= 1
        assert isinstance(fft_metadata, dict)
        assert "high_freq_energy" in fft_metadata
        assert "low_freq_energy" in fft_metadata
        assert "freq_ratio" in fft_metadata

    def test_ela_analysis(self, node, sample_image):
        """ELA analizi testi"""
        ela_score, ela_metadata = node._analyze_ela(sample_image)

        assert isinstance(ela_score, (int, float))
        assert ela_score >= 0
        assert isinstance(ela_metadata, dict)
        assert "ela_mean" in ela_metadata
        assert "ela_std" in ela_metadata
        assert "ela_max" in ela_metadata

    def test_natural_image_lower_score(self, node, natural_image):
        """Doğal görsel için düşük AI olasılığı"""
        result = node.analyze(natural_image)

        # Doğal görsellerde daha düşük skor
        assert result.score < 70

    def test_checkerboard_higher_score(self, node, checkerboard_image):
        """Checkerboard pattern için yüksek AI olasılığı"""
        result = node.analyze(checkerboard_image)

        # Checkerboard artifact yüksek skor vermeli
        assert result.score > 50

    def test_invalid_input_empty_image(self, node):
        """Boş görsel için hata vermeli"""
        empty = np.array([], dtype=np.uint8)

        with pytest.raises(ValueError, match="Görsel boş olamaz"):
            node.analyze(empty)

    def test_invalid_input_wrong_type(self, node):
        """Yanlış tip için hata vermeli"""
        with pytest.raises(ValueError, match="NumPy array olmalı"):
            node.analyze("not_an_image")

    def test_grayscale_image(self, node):
        """Grayscale görsel testi"""
        gray = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        result = node.analyze(gray)

        assert isinstance(result, NodeResult)

    def test_different_thresholds(self):
        """Farklı eşik değerleri testi"""
        node1 = FrequencyNode(fft_threshold=0.5, ela_threshold=10.0)
        assert node1.fft_threshold == 0.5
        assert node1.ela_threshold == 10.0

        # Sınır testleri
        node2 = FrequencyNode(fft_threshold=1.5, ela_threshold=300.0)
        assert node2.fft_threshold == 1.0  # Sınırlandırıldı
        assert node2.ela_threshold == 255.0  # Sınırlandırıldı

    def test_different_ela_quality(self):
        """Farklı JPEG kaliteleri testi"""
        node1 = FrequencyNode(ela_quality=95)
        assert node1.ela_quality == 95

        node2 = FrequencyNode(ela_quality=150)
        assert node2.ela_quality == 100  # Max 100

        node3 = FrequencyNode(ela_quality=0)
        assert node3.ela_quality == 1  # Min 1

    def test_different_image_sizes(self, node):
        """Farklı görsel boyutları testi"""
        sizes = [(10, 10), (100, 100), (512, 512)]

        for height, width in sizes:
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            result = node.analyze(image)

            assert isinstance(result, NodeResult)
            assert 0 <= result.score <= 100

    def test_float_image_conversion(self, node):
        """Float (0-1) görsel uint8'e çevrilmeli"""
        float_image = np.random.rand(50, 50, 3).astype(np.float32)
        result = node.analyze(float_image)

        assert isinstance(result, NodeResult)

    def test_preprocess_no_change(self, node, sample_image):
        """Preprocess görseli değiştirmemeli"""
        processed = node.preprocess(sample_image)
        assert np.array_equal(processed, sample_image)

    def test_combined_score_calculation(self, node, sample_image):
        """Kombine skor hesaplama testi"""
        result = node.analyze(sample_image)

        # Kombine skor hesaplanmalı
        assert "combined_score" in result.metadata
        combined = result.metadata["combined_score"]
        assert isinstance(combined, float)
        assert 0 <= combined <= 1


@pytest.mark.integration
class TestFrequencyNodeIntegration:
    """FrequencyNode entegrasyon testleri"""

    @pytest.fixture
    def ensemble(self):
        """EnsembleEngine örneği"""
        from core.ensemble import EnsembleEngine
        from nodes.watermark import WatermarkNode
        from nodes.frequency import FrequencyNode

        watermark_node = WatermarkNode(weight=1.0)
        frequency_node = FrequencyNode(weight=1.0)

        return EnsembleEngine(nodes=[watermark_node, frequency_node], threshold=50.0)

    @pytest.fixture
    def sample_image(self):
        """Örnek görsel"""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def test_ensemble_with_frequency_node(self, ensemble, sample_image):
        """Ensemble ile FrequencyNode entegrasyonu"""
        result = ensemble.analyze(sample_image)

        assert result is not None
        assert isinstance(result.final_score, (int, float))
        assert 0 <= result.final_score <= 100
        assert "FrequencyNode" in result.node_results

    def test_both_nodes_contributed(self, ensemble, sample_image):
        """Her iki node da katkı sağlamalı"""
        result = ensemble.analyze(sample_image)

        # Hem Watermark hem Frequency sonuçları olmalı
        assert "WatermarkNode" in result.node_results
        assert "FrequencyNode" in result.node_results

        # Her iki node'un da metadata'sı olmalı
        watermark_result = result.node_results["WatermarkNode"]
        frequency_result = result.node_results["FrequencyNode"]

        assert "fft" in frequency_result.metadata
        assert "ela" in frequency_result.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
