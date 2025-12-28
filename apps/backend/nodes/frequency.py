"""
Frequency & ELA Node - Frekans analizi ve Error Level Analysis
"""

from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import io
from ..core.base_node import BaseNode
from ..core.models import NodeResult


class FrequencyNode(BaseNode):
    """
    Görsellerde frekans anomalileri ve manipülasyon izleri tespit eden node.

    Analizler:
    - FFT (Fast Fourier Transform): Checkerboard artifact tespiti
    - ELA (Error Level Analysis): JPEG kompresyon manipülasyon tespiti
    """

    def __init__(self, weight: float = 1.0,
                 fft_threshold: float = 0.7,
                 ela_threshold: float = 20.0,
                 ela_quality: int = 90):
        """
        FrequencyNode yapıcısı

        Args:
            weight: Node ağırlığı (varsayılan: 1.0)
            fft_threshold: FFT anomalisi eşiği (0-1 arası)
            ela_threshold: ELF skoru eşiği (0-255 arası)
            ela_quality: ELA için JPEG kalitesi (1-100)
        """
        super().__init__(weight=weight, name="FrequencyNode")
        self.fft_threshold = max(0.0, min(1.0, fft_threshold))
        self.ela_threshold = max(0.0, min(255.0, ela_threshold))
        self.ela_quality = max(1, min(100, ela_quality))

    def analyze(self, image: np.ndarray) -> NodeResult:
        """
        Görselde frekans anomalileri ve manipülasyon izleri arar.

        Args:
            image: NumPy array formatında görsel (H, W, C)

        Returns:
            NodeResult: Frekans analizi sonucu
        """
        self.validate_image(image)

        metadata: Dict[str, Any] = {}
        score = 0.0
        verdict = "REAL"
        confidence = 0.5

        # 1. FFT Analizi
        fft_score, fft_metadata = self._analyze_fft(image)
        metadata["fft"] = fft_metadata

        # 2. ELA Analizi
        ela_score, ela_metadata = self._analyze_ela(image)
        metadata["ela"] = ela_metadata

        # Skorları birleştir
        # FFT skoru 0-1 arası, ELA skoru 0-255 arası
        # Normalize et ve ağırlıklandır
        normalized_ela = min(1.0, ela_score / 255.0)

        # AI görselleri genellikle her iki anomaliyi de gösterir
        combined_score = (fft_score * 0.5 + normalized_ela * 0.5)

        if combined_score > 0.5:
            # Anomali tespit edildi
            score = 50.0 + combined_score * 50.0  # 50-100 arası
            verdict = "FAKE"
            confidence = combined_score
        else:
            # Anomali yok
            score = combined_score * 30.0  # 0-30 arası
            verdict = "REAL"
            confidence = 0.7

        metadata["fft_score"] = float(fft_score)
        metadata["ela_score"] = float(ela_score)
        metadata["combined_score"] = float(combined_score)

        return NodeResult(
            score=score,
            verdict=verdict,
            metadata=metadata,
            confidence=confidence,
            node_name=self.name
        )

    def _analyze_fft(self, image: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        2D FFT ile frekans spektrumu analizi yapar.

        Checkerboard artifact'ları tespit eder (AI görsellerinde sık görülür).

        Args:
            image: Analiz edilecek görsel

        Returns:
            (anomaly_score, metadata) tuple
        """
        metadata = {}

        try:
            # RGB'yi grayscale'e çevir
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image

            # 2D FFT uygula
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)

            # Magnitude spektrumu
            magnitude = np.abs(fft_shift)
            magnitude = np.log(magnitude + 1)  # Log scale

            # Checkerboard pattern tespiti
            # AI görsellerinde genellikle yüksek frekanslarda checkerboard olur
            h, w = magnitude.shape

            # Yüksek frekans bölgesini al (köşeler)
            corner_size = min(h, w) // 8
            corners = [
                magnitude[:corner_size, :corner_size],
                magnitude[:corner_size, -corner_size:],
                magnitude[-corner_size:, :corner_size],
                magnitude[-corner_size:, -corner_size:]
            ]

            # Yüksek frekans enerjisi
            high_freq_energy = np.mean([np.mean(c) for c in corners])

            # Merkez (düşük frekans) enerjisi
            center_h, center_w = h // 2, w // 2
            center_size = corner_size
            center_region = magnitude[
                center_h-center_size:center_h+center_size,
                center_w-center_size:center_w+center_size
            ]
            low_freq_energy = np.mean(center_region)

            # Yüksek/düşük frekans oranı
            if low_freq_energy > 0:
                ratio = high_freq_energy / low_freq_energy
            else:
                ratio = 0.0

            # Anomali skoru (checkerboard tespiti)
            # Yüksek oran = daha fazla checkerboard artifact
            anomaly_score = min(1.0, ratio / 2.0)

            metadata["high_freq_energy"] = float(high_freq_energy)
            metadata["low_freq_energy"] = float(low_freq_energy)
            metadata["freq_ratio"] = float(ratio)
            metadata["anomaly_detected"] = anomaly_score > self.fft_threshold

            return anomaly_score, metadata

        except Exception as e:
            print(f"FFT analizi hatası: {str(e)}")
            return 0.0, {"error": str(e)}

    def _analyze_ela(self, image: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Error Level Analysis (ELA) yapar.

        Görseli tekrar sıkıştırır ve orijinal arasındaki farkı analiz eder.
        Manipüle edilmiş görsellerde fark daha yüksektir.

        Args:
            image: Analiz edilecek görsel

        Returns:
            (ela_score, metadata) tuple
        """
        metadata = {}

        try:
            # NumPy array'i PIL Image'a çevir
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            pil_image = Image.fromarray(image)

            # JPEG sıkıştırması uygula
            byte_io = io.BytesIO()
            pil_image.save(byte_io, format='JPEG', quality=self.ela_quality)
            byte_io.seek(0)
            compressed = Image.open(byte_io)

            # Farkı hesapla
            original_arr = np.array(pil_image).astype(np.int16)
            compressed_arr = np.array(compressed).astype(np.int16)

            # Mutlak fark (ELA map)
            ela_map = np.abs(original_arr - compressed_arr)

            # Ortalama ELA skoru
            ela_score = float(np.mean(ela_map))

            # ELA histogramı
            ela_flat = ela_map.flatten()
            ela_std = float(np.std(ela_flat))
            ela_max = float(np.max(ela_flat))

            # Anomali tespiti
            # Yüksek ELA = manipülasyon veya AI artifact
            metadata["ela_mean"] = ela_score
            metadata["ela_std"] = ela_std
            metadata["ela_max"] = ela_max
            metadata["manipulation_detected"] = ela_score > self.ela_threshold

            return ela_score, metadata

        except Exception as e:
            print(f"ELA analizi hatası: {str(e)}")
            return 0.0, {"error": str(e)}

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Görseli frekans analizi için ön işleme tabii tutar.

        Args:
            image: İşlenecek görsel

        Returns:
            Ön işleme yapılmış görsel
        """
        # Frekans analizi için orijinal boyutu koru
        return image

    def visualize_fft_spectrum(self, image: np.ndarray,
                              save_path: str = None) -> np.ndarray:
        """
        FFT frekans spektrumunu görselleştirir.

        Args:
            image: Görsel (H, W, C) numpy array
            save_path: Kaydedilecek dosya yolu (opsiyonel)

        Returns:
            Görselleştirilmiş frekans spektrumu (H, W, 3) RGB
        """
        from ..utils.visualization import visualize_frequency_spectrum

        # Grayscale'e çevir
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # 2D FFT uygula
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)

        # Magnitude spektrumu
        magnitude = np.abs(fft_shift)

        return visualize_frequency_spectrum(magnitude, log_scale=True, save_path=save_path)

    def visualize_ela_map(self, image: np.ndarray,
                         save_path: str = None) -> np.ndarray:
        """
        ELA map'ini görselleştirir.

        Args:
            image: Görsel (H, W, C) numpy array
            save_path: Kaydedilecek dosya yolu (opsiyonel)

        Returns:
            Görselleştirilmiş ELA map (H, W, 3) RGB
        """
        from ..utils.visualization import visualize_ela_map

        # ELA hesapla
        ela_score, ela_metadata = self._analyze_ela(image)

        # Yeni ELA map'i hesapla (görselleştirme için)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)

        # JPEG sıkıştırması uygula
        byte_io = io.BytesIO()
        pil_image.save(byte_io, format='JPEG', quality=self.ela_quality)
        byte_io.seek(0)
        compressed = Image.open(byte_io)

        # Farkı hesapla
        original_arr = np.array(pil_image).astype(np.int16)
        compressed_arr = np.array(compressed).astype(np.int16)

        # Mutlak fark (ELA map)
        ela_map = np.abs(original_arr - compressed_arr)

        return visualize_ela_map(ela_map, save_path=save_path)
