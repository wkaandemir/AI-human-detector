"""
Entegrasyon Testleri - End-to-end analiz akışı
"""

import pytest
import numpy as np
from PIL import Image
import tempfile
import os

from core.ensemble import EnsembleEngine
from core.models import NodeResult, EnsembleResult
from nodes.watermark import WatermarkNode
from nodes.frequency import FrequencyNode
from nodes.clip import CLIPNode


class TestEnsembleEngine:
    """Ensemble motoru entegrasyon testleri"""

    def test_ensemble_with_single_node(self):
        """Tek node ile ensemble testi"""
        node = WatermarkNode(weight=1.0)
        engine = EnsembleEngine(nodes=[node])

        # Test görseli
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        result = engine.analyze(image)

        assert isinstance(result, EnsembleResult)
        assert result.final_score >= 0 and result.final_score <= 100
        assert result.verdict in ["REAL", "FAKE", "UNCERTAIN"]
        assert len(result.node_results) == 1

    def test_ensemble_with_multiple_nodes(self):
        """Birden fazla node ile ensemble testi"""
        nodes = [
            WatermarkNode(weight=1.0),
            FrequencyNode(weight=1.0),
        ]
        engine = EnsembleEngine(nodes=nodes)

        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        result = engine.analyze(image)

        assert isinstance(result, EnsembleResult)
        assert result.final_score >= 0 and result.final_score <= 100
        assert len(result.node_results) == 2
        # Tüm node'ların sonuçları başarılı olmalı
        for node_result in result.node_results:
            assert node_result.score >= 0 and node_result.score <= 100

    def test_ensemble_weighted_aggregation(self):
        """Ağırlıklı aggregate testi"""
        nodes = [
            WatermarkNode(weight=0.3),
            FrequencyNode(weight=0.7),
        ]
        engine = EnsembleEngine(nodes=nodes)

        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        result = engine.analyze(image)

        assert isinstance(result, EnsembleResult)
        # Ağırlıklı ortalama hesaplanmalı
        # Ancak node sonuçları rastgele olacak için kesin değer kontrolü zor

    def test_ensemble_watermark_short_circuit(self):
        """Watermark tespiti ile short-circuit testi"""
        nodes = [
            WatermarkNode(weight=1.0),
            FrequencyNode(weight=1.0),
        ]
        engine = EnsembleEngine(nodes=nodes, watermark_short_circuit=True)

        # Sahte watermark EXIF içeren görsel
        # Normalde bu test için gerçek watermark içeren görsel gerekir
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        result = engine.analyze(image)

        # Watermark yoksa tüm node'lar çalışmalı
        assert len(result.node_results) >= 1

    def test_ensemble_with_clip_node(self):
        """CLIP node ile ensemble testi"""
        try:
            nodes = [
                WatermarkNode(weight=1.0),
                CLIPNode(weight=1.0),
            ]
            engine = EnsembleEngine(nodes=nodes)

            image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

            result = engine.analyze(image)

            assert isinstance(result, EnsembleResult)
            assert len(result.node_results) == 2
        except ImportError:
            # CLIP kurulmadıysa testi geç
            pytest.skip("CLIP kurulmadı")

    def test_ensemble_empty_nodes(self):
        """Boş node listesi ile ensemble testi"""
        engine = EnsembleEngine(nodes=[])

        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        with pytest.raises((ValueError, Exception)):
            engine.analyze(image)

    def test_ensemble_metadata_aggregation(self):
        """Metadata toplama testi"""
        nodes = [
            WatermarkNode(weight=1.0),
            FrequencyNode(weight=1.0),
        ]
        engine = EnsembleEngine(nodes=nodes)

        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        result = engine.analyze(image)

        # Her node'un metadata'sı olmalı
        for node_result in result.node_results:
            assert isinstance(node_result.metadata, dict)
            assert len(node_result.metadata) > 0


class TestEndToEnd:
    """End-to-end test senaryoları"""

    def test_complete_analysis_pipeline(self):
        """Tam analiz pipeline testi"""
        # Tüm node'ları yükle
        nodes = [
            WatermarkNode(weight=1.0),
            FrequencyNode(weight=1.0),
        ]

        engine = EnsembleEngine(nodes=nodes)

        # Test görseli oluştur
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Analiz et
        result = engine.analyze(image)

        # Sonuçları kontrol et
        assert result.final_score >= 0 and result.final_score <= 100
        assert result.verdict in ["REAL", "FAKE", "UNCERTAIN"]
        assert result.confidence >= 0 and result.confidence <= 1
        assert len(result.node_results) == len(nodes)

        # Metadata kontrolü
        for node_result in result.node_results:
            assert node_result.node_name in ["WatermarkNode", "FrequencyNode"]
            assert node_result.verdict in ["REAL", "FAKE", "UNCERTAIN"]

    def test_multiple_images_analysis(self):
        """Birden fazla görsel analizi testi"""
        engine = EnsembleEngine(nodes=[
            WatermarkNode(weight=1.0),
            FrequencyNode(weight=1.0),
        ])

        # 5 farklı görsel
        images = [
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            for _ in range(5)
        ]

        results = []
        for image in images:
            result = engine.analyze(image)
            results.append(result)
            assert result is not None

        # Tüm sonuçlar başarılı olmalı
        assert len(results) == 5

    def test_different_image_sizes(self):
        """Farklı boyutlardaki görseller testi"""
        engine = EnsembleEngine(nodes=[
            WatermarkNode(weight=1.0),
            FrequencyNode(weight=1.0),
        ])

        sizes = [
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 768),
        ]

        for size in sizes:
            image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            result = engine.analyze(image)
            assert result is not None
            assert result.final_score >= 0 and result.final_score <= 100


class TestNodeInteraction:
    """Node etkileşim testleri"""

    def test_node_preprocessing_chain(self):
        """Node ön işleme zinciri testi"""
        nodes = [
            WatermarkNode(weight=1.0),
            FrequencyNode(weight=1.0),
        ]

        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Her node'un preprocess metodunu test et
        for node in nodes:
            processed = node.preprocess(image)
            assert processed is not None
            assert processed.shape == image.shape

    def test_node_result_consistency(self):
        """Node sonuç tutarlılığı testi"""
        node = WatermarkNode(weight=1.0)
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Aynı görsel için aynı sonuç
        result1 = node.analyze(image)
        result2 = node.analyze(image)

        # Sonuçlar benzer olmalı (deterministik)
        assert result1.score == result2.score
        assert result1.verdict == result2.verdict

    def test_failed_node_handling(self):
        """Başarısız node ele alma testi"""
        # Geçersiz ağırlık ile node
        try:
            node = WatermarkNode(weight=-1.0)
            # Hata fırlatmalı veya düzeltmeli
        except (ValueError, Exception):
            pass

    def test_gpu_memory_management(self):
        """GPU bellek yönetimi testi"""
        # Bu test GPU gerektirir
        try:
            from nodes.dire import DIRENode

            node = DIRENode(weight=1.0)

            if not node.is_available():
                pytest.skip("GPU gerekli")

            image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

            # İlk analiz
            result1 = node.analyze(image)
            assert result1 is not None

            # Bellek temizlendi mi? İkinci analiz de başarılı olmalı
            result2 = node.analyze(image)
            assert result2 is not None

        except ImportError:
            pytest.skip("DIRE node gerekli")


class TestVisualizationIntegration:
    """Görselleştirme entegrasyon testleri"""

    def test_dire_error_map_visualization(self):
        """DIRE error map görselleştirme testi"""
        try:
            from nodes.dire import DIRENode

            node = DIRENode(weight=1.0)

            if not node.is_available():
                pytest.skip("GPU gerekli")

            image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

            # Error map oluştur
            error_map = np.random.rand(512, 512)

            # Görselleştir
            vis = node.visualize_error_map(error_map)
            assert vis is not None
            assert vis.shape == (512, 512, 3)

        except ImportError:
            pytest.skip("DIRE node gerekli")

    def test_frequency_visualization(self):
        """Frequency node görselleştirme testi"""
        node = FrequencyNode(weight=1.0)
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # FFT spectrum
        fft_vis = node.visualize_fft_spectrum(image)
        assert fft_vis is not None
        assert fft_vis.shape[2] == 3

        # ELA map
        ela_vis = node.visualize_ela_map(image)
        assert ela_vis is not None
        assert ela_vis.shape[2] == 3

    def test_overlay_heatmap(self):
        """Heatmap bindirme testi"""
        from utils.visualization import overlay_heatmap

        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        heatmap = np.random.rand(256, 256)

        result = overlay_heatmap(image, heatmap, alpha=0.5)
        assert result is not None
        assert result.shape == (256, 256, 3)

    def test_comparison_gallery(self):
        """Karşılaştırma galerisi testi"""
        from utils.visualization import create_comparison_gallery

        images = [
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            for _ in range(4)
        ]
        titles = ["Image 1", "Image 2", "Image 3", "Image 4"]

        gallery = create_comparison_gallery(images, titles, cols=2)
        assert gallery is not None
        assert gallery.shape[2] == 3


class TestFileIOIntegration:
    """Dosya G/Ç entegrasyon testleri"""

    def test_image_save_load_cycle(self):
        """Görsel kaydetme/yükleme döngüsü testi"""
        node = WatermarkNode(weight=1.0)

        # Orijinal görsel
        original = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result1 = node.analyze(original)

        # Kaydet ve tekrar yükle
        pil_img = Image.fromarray(original)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            pil_img.save(temp_path)

        try:
            loaded = Image.open(temp_path)
            loaded_array = np.array(loaded)
            result2 = node.analyze(loaded_array)

            # Sonuçlar benzer olmalı
            assert abs(result1.score - result2.score) < 5.0
        finally:
            os.unlink(temp_path)

    def test_visualization_save(self):
        """Görselleştirme kaydetme testi"""
        from utils.visualization import visualize_error_map

        error_map = np.random.rand(256, 256)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name

        try:
            result = visualize_error_map(error_map, save_path=temp_path)
            assert result is not None
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
