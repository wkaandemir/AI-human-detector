"""
Watermark Node - AI watermark tespiti
"""

from typing import Optional, Dict, Any
import numpy as np
from PIL import Image
import io
from ..core.base_node import BaseNode
from ..core.models import NodeResult

# İsteğe bağlı import'lar (kurulu değilse hata vermez)
try:
    import imwatermark
    WATERMARK_AVAILABLE = True
except ImportError:
    WATERMARK_AVAILABLE = False
    print("Uyarı: imwatermark kurulmadı, watermark tespiti devre dışı")

try:
    import exifread
    EXIF_AVAILABLE = True
except ImportError:
    EXIF_AVAILABLE = False
    print("Uyarı: exifread kurulmadı, EXIF analizi devre dışı")


class WatermarkNode(BaseNode):
    """
    Görsellerde AI watermark'ı tespit eden node.

    Desteklenen watermark türleri:
    - Stable Diffusion (invisible watermark)
    - EXIF metadata analizi
    """

    # AI yazılım imzaları EXIF'te aranacak software isimleri
    AI_SOFTWARE_SIGNATURES = [
        "Stable Diffusion",
        "Midjourney",
        "DALL-E",
        "Adobe Firefly",
        "DreamStudio",
        "Automatic1111"
    ]

    def __init__(self, weight: float = 1.0, threshold: float = 0.5):
        """
        WatermarkNode yapıcısı

        Args:
            weight: Node ağırlığı (varsayılan: 1.0)
            threshold: Watermark tespit eşiği (0-1 arası)
        """
        super().__init__(weight=weight, name="WatermarkNode")
        self.threshold = threshold
        self.check_watermark = WATERMARK_AVAILABLE
        self.check_exif = EXIF_AVAILABLE

    def analyze(self, image: np.ndarray) -> NodeResult:
        """
        Görselde watermark ve AI imzası arar.

        Args:
            image: NumPy array formatında görsel (H, W, C)

        Returns:
            NodeResult: Watermark analizi sonucu
        """
        self.validate_image(image)

        metadata: Dict[str, Any] = {}
        score = 0.0
        verdict = "REAL"
        confidence = 0.5

        # 1. Invisible watermark kontrolü (Stable Diffusion)
        if self.check_watermark:
            watermark_score = self._check_invisible_watermark(image)
            if watermark_score > self.threshold:
                score = 95.0
                verdict = "FAKE"
                confidence = 0.9
                metadata["watermark_detected"] = True
                metadata["watermark_type"] = "invisible"
                metadata["watermark_score"] = float(watermark_score)
                return NodeResult(
                    score=score,
                    verdict=verdict,
                    metadata=metadata,
                    confidence=confidence,
                    node_name=self.name
                )
            metadata["watermark_score"] = float(watermark_score)

        # 2. EXIF metadata kontrolü
        if self.check_exif:
            exif_result = self._check_exif_watermark(image)
            if exif_result["detected"]:
                score = 90.0
                verdict = "FAKE"
                confidence = 0.85
                metadata["exif_detected"] = True
                metadata["exif_software"] = exif_result.get("software", "unknown")
            else:
                metadata["exif_detected"] = False
                metadata["exif_software"] = exif_result.get("software", "none")

        # Final skor
        if score == 0.0:
            # Watermark bulunamadı
            score = 5.0  # Düşük AI olasılığı
            verdict = "REAL"
            confidence = 0.7

        metadata["check_watermark"] = self.check_watermark
        metadata["check_exif"] = self.check_exif

        return NodeResult(
            score=score,
            verdict=verdict,
            metadata=metadata,
            confidence=confidence,
            node_name=self.name
        )

    def _check_invisible_watermark(self, image: np.ndarray) -> float:
        """
        Görünmez watermark kontrolü yapar (Stable Diffusion vb.)

        Args:
            image: Kontrol edilecek görsel

        Returns:
            0-1 arası watermark olasılık skoru
        """
        try:
            # NumPy array'i PIL Image'a çevir
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            pil_image = Image.fromarray(image)

            # imwatermark ile kontrol et
            # Diğer watermark türleri: "dwt", "rivenet"
            score = imwatermark.detect(pil_image, "riva")

            if score is not None:
                return float(score)

        except Exception as e:
            print(f"Watermark kontrol hatası: {str(e)}")

        return 0.0

    def _check_exif_watermark(self, image: np.ndarray) -> Dict[str, Any]:
        """
        EXIF metadata'sında AI imzası arar.

        Args:
            image: Kontrol edilecek görsel

        Returns:
            Tespit sonucu sözlüğü
        """
        result = {
            "detected": False,
            "software": None,
            "all_tags": {}
        }

        try:
            # NumPy array'i byte'lara çevir
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            pil_image = Image.fromarray(image)
            byte_io = io.BytesIO()
            pil_image.save(byte_io, format='JPEG')
            byte_io.seek(0)

            # EXIF oku
            tags = exifread.process_file(byte_io, details=False)

            # Software tag'ini kontrol et
            if 'Image Software' in tags:
                software = str(tags['Image Software'])
                result["software"] = software

                # AI yazılım imzalarını kontrol et
                for ai_signature in self.AI_SOFTWARE_SIGNATURES:
                    if ai_signature.lower() in software.lower():
                        result["detected"] = True
                        break

            # Tüm EXIF tag'lerini kaydet
            for key, value in tags.items():
                if key not in ['JPEGThumbnail', 'TIFFThumbnail']:
                    result["all_tags"][key] = str(value)

        except Exception as e:
            print(f"EXIF kontrol hatası: {str(e)}")

        return result

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Görseli watermark analizi için ön işleme tabii tutar.

        Args:
            image: İşlenecek görsel

        Returns:
            Ön işleme yapılmış görsel
        """
        # Watermark analizi için orijinal görseli koru
        return image
