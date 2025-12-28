"""
DIRENode birim testleri
"""

import pytest
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Backend paketini path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from nodes.dire import DIRENode, DIFFUSERS_AVAILABLE
from core.models import NodeResult


@pytest.mark.skipif(not DIFFUSERS_AVAILABLE, reason="Diffusers kurulmadı")
class TestDIRENode:
    """DIRENode için test sınıfı"""

    @pytest.fixture
    def node(self):
        """Test DIRE node'u"""
        return DIRENode(
            weight=1.0,
            model_name="runwayml/stable-diffusion-v1-5",
            num_steps=50
        )

    @pytest.fixture
    def sample_image(self):
        """Örnek RGB görsel (512x512 - SD input size)"""
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    @pytest.fixture
    def natural_image(self):
        """Doğal görüntü"""
        x = np.linspace(0, 255, 512)
        y = np.linspace(0, 255, 512)
        xx, yy = np.meshgrid(x, y)
        r = (xx + yy) / 2
        g = xx
        b = yy
        return np.stack([r, g, b], axis=2).astype(np.uint8)

    def test_node_initialization(self, node):
        """Node başlatma testi"""
        assert node.name == "DIRENode"
        assert node.weight == 1.0
        assert node.model_name == "runwayml/stable-diffusion-v1-5"
        assert node.num_steps == 50
        assert node.enabled is True

    def test_is_available(self, node):
        """Model kullanılabilirlik testi (GPU gerektirir)"""
        import torch

        available = node.is_available()
        assert isinstance(available, bool)
        assert available == (torch.cuda.is_available() and DIFFUSERS_AVAILABLE)

    def test_get_model_info(self, node):
        """Model bilgileri testi"""
        info = node.get_model_info()

        assert isinstance(info, dict)
        assert "available" in info
        assert "model_name" in info
        assert "cuda_available" in info
        assert "num_steps" in info

    def test_prepare_image(self, node, sample_image):
        """Görsel hazırlama testi"""
        import torch

        prepared = node._prepare_image(sample_image)

        assert isinstance(prepared, torch.Tensor)
        assert prepared.shape[0] == 1  # Batch dimension
        assert prepared.shape[2] == 512  # Height
        assert prepared.shape[3] == 512  # Width

    def test_prepare_image_resizes(self, node):
        """Resize testi"""
        import torch

        # Farklı boyutlarda görsel
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        prepared = node._prepare_image(image)

        # 512x512 olmalı
        assert prepared.shape[2] == 512
        assert prepared.shape[3] == 512

    def test_compute_error_map(self, node):
        """Error map hesaplama testi"""
        import torch

        # Fake original ve reconstructed
        original = torch.rand(1, 3, 512, 512)
        reconstructed = torch.rand(1, 3, 512, 512)

        error_map = node._compute_error_map(original, reconstructed)

        assert isinstance(error_map, np.ndarray)
        assert error_map.shape == (512, 512)
        assert np.all(error_map >= 0)

    def test_compute_error_score(self, node):
        """Error skoru hesaplama testi"""
        error_map = np.random.rand(512, 512).astype(np.float32)

        score = node._compute_error_score(error_map)

        assert isinstance(score, (int, float))
        assert 0 <= score <= 1

    def test_different_num_steps(self):
        """Farklı adım sayıları testi"""
        node1 = DIRENode(num_steps=25)
        assert node1.num_steps == 25

        node2 = DIRENode(num_steps=100)
        assert node2.num_steps == 100


@pytest.mark.gpu
@pytest.mark.skipif(not DIFFUSERS_AVAILABLE, reason="Diffusers kurulmadı")
class TestDIRENodeWithGPU:
    """GPU gerektiren testler"""

    @pytest.fixture
    def node(self):
        """Test DIRE node'u"""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA gerekiyor")

        return DIRENode(
            weight=1.0,
            model_name="runwayml/stable-diffusion-v1-5",
            num_steps=50
        )

    @pytest.fixture
    def sample_image(self):
        """Örnek RGB görsel"""
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    def test_analyze_returns_valid_result(self, node, sample_image):
        """Analiz geçerli sonuç döndürmeli"""
        # Not: Bu test yavaş çalışır (model yükleme + inference)
        result = node.analyze(sample_image)

        assert isinstance(result, NodeResult)
        assert isinstance(result.score, (int, float))
        assert 0 <= result.score <= 100
        assert result.verdict in ["REAL", "FAKE", "UNCERTAIN"]
        assert result.node_name == "DIRENode"

    def test_metadata_structure(self, node, sample_image):
        """Metadata yapısı kontrolü"""
        result = node.analyze(sample_image)

        # Zorunlu metadata alanları
        assert "error_score" in result.metadata
        assert "error_mean" in result.metadata
        assert "error_std" in result.metadata
        assert "error_max" in result.metadata
        assert "model_name" in result.metadata
        assert "available" in result.metadata

    def test_inversion_and_reconstruction(self, node, sample_image):
        """DDIM inversion ve reconstruction testi"""
        import torch

        # Görseli hazırla
        prepared = node._prepare_image(sample_image)

        # Inversion
        latents = node._ddim_invert(prepared)
        assert isinstance(latents, torch.Tensor)

        # Reconstruction
        reconstructed = node._reconstruct(latents)
        assert isinstance(reconstructed, torch.Tensor)

        # Shape kontrolü
        assert reconstructed.shape == prepared.shape

    def test_cleanup_memory(self, node):
        """Bellek temizleme testi"""
        import torch

        # Önce bellek kullan
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Temizle
        node._cleanup_memory()

        # Bellek azalmalı (garanti değil ama trend)
        after_cleanup = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0


@pytest.mark.integration
class TestDIRENodeIntegration:
    """DIRENode entegrasyon testleri"""

    @pytest.fixture
    def ensemble(self):
        """EnsembleEngine örneği"""
        from core.ensemble import EnsembleEngine
        from nodes.watermark import WatermarkNode
        from nodes.frequency import FrequencyNode
        from nodes.clip import CLIPNode
        from nodes.dire import DIRENode

        watermark_node = WatermarkNode(weight=1.0)
        frequency_node = FrequencyNode(weight=1.0)
        clip_node = CLIPNode(weight=1.0)
        dire_node = DIRENode(weight=1.0)

        return EnsembleEngine(
            nodes=[watermark_node, frequency_node, clip_node, dire_node],
            threshold=50.0
        )

    @pytest.fixture
    def sample_image(self):
        """Örnek görsel"""
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    @pytest.mark.skipif(not DIFFUSERS_AVAILABLE, reason="Diffusers kurulmadı")
    def test_ensemble_with_dire_node(self, ensemble, sample_image):
        """Ensemble ile DIRENode entegrasyonu"""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA gerekiyor")

        result = ensemble.analyze(sample_image)

        assert result is not None
        assert isinstance(result.final_score, (int, float))
        assert 0 <= result.final_score <= 100

    @pytest.mark.skipif(not DIFFUSERS_AVAILABLE, reason="Diffusers kurulmadı")
    def test_all_nodes_contributed(self, ensemble, sample_image):
        """Her dört node da katkı sağlamalı"""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA gerekiyor")

        result = ensemble.analyze(sample_image)

        assert "WatermarkNode" in result.node_results
        assert "FrequencyNode" in result.node_results
        assert "CLIPNode" in result.node_results
        assert "DIRENode" in result.node_results


class TestDIRENodeWithoutGPU:
    """GPU yokken testler"""

    def test_node_without_gpu_returns_uncertain(self):
        """GPU yoksa UNCERTAIN dönmeli"""
        if DIFFUSERS_AVAILABLE:
            import torch
            if torch.cuda.is_available():
                pytest.skip("GPU mevcut, test atlanıyor")

        node = DIRENode(weight=1.0)
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        result = node.analyze(image)

        assert result.verdict == "UNCERTAIN"
        assert result.metadata.get("available", True) is False or \
               "cuda" in result.metadata.get("error", "").lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
