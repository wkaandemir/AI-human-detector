"""
BaseNode sınıfı - Tüm detection node'ları için temel sınıf
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import time
from .models import NodeResult


class BaseNode(ABC):
    """
    Tüm detection node'ları için temel (base) sınıf.

    Her node bu sınıftan türetilmeli ve analyze metodunu implement etmeli.

    Attributes:
        weight: Node'un ağırlığı (0-1 arası, ensemble için)
        name: Node adı
        enabled: Node'un aktif/pasif durumu
    """

    def __init__(self, weight: float = 1.0, name: Optional[str] = None):
        """
        BaseNode yapıcısı

        Args:
            weight: Node ağırlığı (varsayılan: 1.0)
            name: Node adı (varsayılan: sınıf adı)
        """
        self.weight = max(0.0, min(1.0, weight))  # 0-1 arası sınırlandır
        self.name = name or self.__class__.__name__
        self.enabled = True

    @abstractmethod
    def analyze(self, image: np.ndarray) -> NodeResult:
        """
        Görseli analiz eder ve sonuç döndürür.

        Args:
            image: NumPy array formatında görsel (H, W, C)

        Returns:
            NodeResult: Analiz sonucu

        Raises:
            ValueError: Geçersiz görsel formatı
            RuntimeError: Analiz sırasında hata
        """
        pass

    def validate_image(self, image: np.ndarray) -> None:
        """
        Görselin geçerliliğini kontrol eder.

        Args:
            image: Kontrol edilecek görsel

        Raises:
            ValueError: Görsel geçersizse
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Görsel NumPy array olmalı, {type(image)} alındı")

        if image.ndim not in [2, 3]:
            raise ValueError(f"Görsel 2D veya 3D olmalı, {image.ndim}D alındı")

        if image.size == 0:
            raise ValueError("Görsel boş olamaz")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Görseli ön işleme tabii tutar. Varsayılan olarak değişiklik yapmaz.

        Args:
            image: İşlenecek görsel

        Returns:
            Ön işleme yapılmış görsel
        """
        return image

    def compute_score(self, image: np.ndarray) -> float:
        """
        Görsel için skor hesaplar. analyze metodunun iç mantığını içerir.

        Bu metod alt sınıflar tarafından override edilebilir.

        Args:
            image: Skor hesaplanacak görsel

        Returns:
            0-100 arası skor
        """
        result = self.analyze(image)
        return result.score

    def __repr__(self) -> str:
        """Node'un string temsili"""
        return f"{self.name}(weight={self.weight}, enabled={self.enabled})"
