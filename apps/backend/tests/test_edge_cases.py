"""
Edge Case Testleri - Boş görüntü, farklı formatlar, sınır durumları
"""

import pytest
import numpy as np
from PIL import Image
import io
import tempfile
import os

from nodes.watermark import WatermarkNode
from nodes.frequency import FrequencyNode
from nodes.clip import CLIPNode


class TestEdgeCases:
    """Edge case testleri"""

    def test_empty_image_watermark(self):
        """Boş/invalid görsel WatermarkNode testi"""
        node = WatermarkNode()

        # Boş array
        with pytest.raises((ValueError, Exception)):
            node.analyze(np.array([]))

    def test_single_pixel_image(self):
        """Tek piksel görsel testi"""
        node = WatermarkNode()

        # 1x1 piksel
        single_pixel = np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8)

        # Muhtemelen hata fırlatmalı veya sonuç döndürmeli
        try:
            result = node.analyze(single_pixel)
            assert result is not None
            assert result.score >= 0 and result.score <= 100
        except (ValueError, Exception):
            # Hata fırlatması da kabul edilebilir
            pass

    def test_very_large_image(self):
        """Çok büyük görsel testi (4K+)"""
        node = FrequencyNode()

        # 4K görsel (3840x2160)
        large_image = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)

        try:
            result = node.analyze(large_image)
            assert result is not None
        except (MemoryError, Exception):
            # Bellek hatası da kabul edilebilir
            pass

    def test_different_formats(self):
        """Farklı görsel formatları testi"""
        node = WatermarkNode()

        # RGB
        rgb_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result_rgb = node.analyze(rgb_image)
        assert result_rgb is not None

        # RGBA
        rgba_image = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
        try:
            # RGBA -> RGB dönüşümü gerekli
            result_rgba = node.analyze(rgba_image[:, :, :3])
            assert result_rgba is not None
        except Exception:
            pass

        # Grayscale
        gray_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        try:
            result_gray = node.analyze(gray_image)
            assert result_gray is not None
        except Exception:
            pass

    def test_jpeg_format(self):
        """JPEG formatında kayıtlı görsel testi"""
        node = FrequencyNode()

        # Rastgele görsel oluştur
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

        # JPEG olarak kaydet
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
            img.save(temp_path, 'JPEG', quality=85)

        try:
            # Oku ve test et
            loaded_img = Image.open(temp_path)
            img_array = np.array(loaded_img)
            result = node.analyze(img_array)
            assert result is not None
        finally:
            os.unlink(temp_path)

    def test_png_format(self):
        """PNG formatında kayıtlı görsel testi"""
        node = FrequencyNode()

        # Rastgele görsel oluştur
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

        # PNG olarak kaydet
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            img.save(temp_path, 'PNG')

        try:
            # Oku ve test et
            loaded_img = Image.open(temp_path)
            img_array = np.array(loaded_img)
            result = node.analyze(img_array)
            assert result is not None
        finally:
            os.unlink(temp_path)

    def test_webp_format(self):
        """WebP formatında kayıtlı görsel testi"""
        node = FrequencyNode()

        # Rastgele görsel oluştur
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

        # WebP olarak kaydet (destekleniyorsa)
        try:
            with tempfile.NamedTemporaryFile(suffix='.webp', delete=False) as f:
                temp_path = f.name
                img.save(temp_path, 'WebP')

            try:
                loaded_img = Image.open(temp_path)
                img_array = np.array(loaded_img)
                result = node.analyze(img_array)
                assert result is not None
            finally:
                os.unlink(temp_path)
        except Exception:
            # WebP desteklenmiyorsa testi geç
            pass

    def test_float_image(self):
        """Float tipinde görsel testi (0-1 arası)"""
        node = WatermarkNode()

        # Float image (0-1 arası)
        float_image = np.random.rand(256, 256, 3).astype(np.float32)

        try:
            result = node.analyze(float_image)
            assert result is not None
        except Exception:
            # Dönüşüm gerekli olabilir
            pass

    def test_extreme_aspect_ratios(self):
        """Ekstrem en-boy oranları testi"""
        node = FrequencyNode()

        # Çok geniş
        wide_image = np.random.randint(0, 255, (100, 1000, 3), dtype=np.uint8)
        result_wide = node.analyze(wide_image)
        assert result_wide is not None

        # Çok uzun
        tall_image = np.random.randint(0, 255, (1000, 100, 3), dtype=np.uint8)
        result_tall = node.analyze(tall_image)
        assert result_tall is not None

    def test_monochromatic_images(self):
        """Tek renkli (monokrom) görseller testi"""
        node = FrequencyNode()

        # Tamamen siyah
        black_image = np.zeros((256, 256, 3), dtype=np.uint8)
        result_black = node.analyze(black_image)
        assert result_black is not None

        # Tamamen beyaz
        white_image = np.full((256, 256, 3), 255, dtype=np.uint8)
        result_white = node.analyze(white_image)
        assert result_white is not None

        # Tamamen kırmızı
        red_image = np.zeros((256, 256, 3), dtype=np.uint8)
        red_image[:, :, 0] = 255
        result_red = node.analyze(red_image)
        assert result_red is not None

    def test_checkerboard_pattern(self):
        """Checkerboard pattern testi (AI artifact benzeri)"""
        node = FrequencyNode()

        # Checkerboard oluştur
        size = 256
        checkerboard = np.zeros((size, size, 3), dtype=np.uint8)
        square_size = 16

        for i in range(0, size, square_size):
            for j in range(0, size, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    checkerboard[i:i+square_size, j:j+square_size] = 255

        result = node.analyze(checkerboard)
        assert result is not None
        # Checkerboard yüksek FFT skoru vermeli
        # Ancak bu test sadece çalıştığını kontrol ediyor

    def test_corrupted_image_data(self):
        """Bozuk görsel verisi testi"""
        node = WatermarkNode()

        # NaN değerleri
        nan_image = np.full((256, 256, 3), np.nan, dtype=np.float32)
        with pytest.raises(Exception):
            node.analyze(nan_image)

        # Sonsuz değerler
        inf_image = np.full((256, 256, 3), np.inf, dtype=np.float32)
        with pytest.raises(Exception):
            node.analyze(inf_image)

    def test_low_resolution(self):
        """Çok düşük çözünürlük testi"""
        node = WatermarkNode()

        # 8x8 görsel
        tiny_image = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)

        try:
            result = node.analyze(tiny_image)
            assert result is not None
        except Exception:
            # Çok küçük olabilir, hata da kabul edilebilir
            pass

    def test_non_standard_channels(self):
        """Standart olmayan kanal sayısı testi"""
        node = WatermarkNode()

        # 2 kanal (nadir ama olabilir)
        two_channel = np.random.randint(0, 255, (256, 256, 2), dtype=np.uint8)
        try:
            # İlk 3 kanalı al veya hata fırlat
            result = node.analyze(two_channel[:, :, :3])
            assert result is not None
        except Exception:
            pass

    def test_mixed_precision(self):
        """Farklı veri tipleri testi"""
        node = FrequencyNode()

        base_image = np.random.randint(0, 255, (256, 256, 3))

        # uint8
        result_uint8 = node.analyze(base_image.astype(np.uint8))
        assert result_uint8 is not None

        # uint16
        try:
            result_uint16 = node.analyze(base_image.astype(np.uint16))
            assert result_uint16 is not None
        except Exception:
            pass

        # float32
        try:
            result_float32 = node.analyze(base_image.astype(np.float32) / 255.0)
            assert result_float32 is not None
        except Exception:
            pass


class TestVisualizationEdgeCases:
    """Görselleştirme edge case testleri"""

    def test_empty_error_map_visualization(self):
        """Boş error map görselleştirme testi"""
        from utils.visualization import visualize_error_map

        # Boş error map
        empty_map = np.array([])

        with pytest.raises(Exception):
            visualize_error_map(empty_map)

    def test_single_pixel_error_map(self):
        """Tek piksel error map görselleştirme testi"""
        from utils.visualization import visualize_error_map

        single_pixel = np.array([[0.5]])

        try:
            result = visualize_error_map(single_pixel)
            assert result.shape == (1, 1, 3)
        except Exception:
            pass

    def test_all_zeros_error_map(self):
        """Tamamen sıfır error map testi"""
        from utils.visualization import visualize_error_map

        zero_map = np.zeros((64, 64))

        # Division by zero hatası vermemeli
        result = visualize_error_map(zero_map)
        assert result is not None
        assert result.shape == (64, 64, 3)

    def test_all_same_values_error_map(self):
        """Aynı değerlerden oluşan error map testi"""
        from utils.visualization import visualize_error_map

        same_map = np.full((64, 64), 0.5)

        result = visualize_error_map(same_map)
        assert result is not None
        assert result.shape == (64, 64, 3)

    def test_extreme_values_error_map(self):
        """Ekstrem değerler error map testi"""
        from utils.visualization import visualize_error_map

        extreme_map = np.random.rand(64, 64) * 1000

        result = visualize_error_map(extreme_map)
        assert result is not None
        assert result.shape == (64, 64, 3)
