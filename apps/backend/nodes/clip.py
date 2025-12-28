"""
CLIP Node - Semantic Anomaly Detection with CLIP embeddings
"""

from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image
from ..core.base_node import BaseNode
from ..core.models import NodeResult

# İsteğe bağlı import'lar
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Uyarı: transformers kurulmadı, CLIP analizi devre dışı")


class CLIPNode(BaseNode):
    """
    CLIP embedding'leri ile anlamsal anomali tespiti yapan node.

    Zero-shot detection için pretrained CLIP modeli kullanır.
    AI-generated görseller embedding uzayında farklı pattern'ler gösterir.
    """

    def __init__(self, weight: float = 1.0,
                 model_name: str = "openai/clip-vit-base-patch32",
                 threshold: float = 0.5,
                 device: Optional[str] = None):
        """
        CLIPNode yapıcısı

        Args:
            weight: Node ağırlığı (varsayılan: 1.0)
            model_name: CLIP model adı (HuggingFace)
            threshold: Anomali tespit eşiği (0-1 arası)
            device: CPU veya CUDA (None = auto)
        """
        super().__init__(weight=weight, name="CLIPNode")
        self.model_name = model_name
        self.threshold = max(0.0, min(1.0, threshold))
        self.device = device

        # Model ve processor (yavaş başlatma - ilk kullanımda)
        self._model = None
        self._processor = None

        # Refer embedding'ler (gerçek ve sahte görseller için)
        self._real_embeddings: List[np.ndarray] = []
        self._fake_embeddings: List[np.ndarray] = []
        self._is_calibrated = False

    @property
    def model(self):
        """CLIP modelini lazy load ile döndürür"""
        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP modeli kurulmadı (transformers paketi eksik)")

        if self._model is None:
            self._model = CLIPModel.from_pretrained(self.model_name)
            if self.device:
                self._model.to(self.device)
            else:
                self._model.to("cuda" if torch.cuda.is_available() else "cpu")
            self._model.eval()
        return self._model

    @property
    def processor(self):
        """CLIP processor'ı lazy load ile döndürür"""
        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP processor kurulmadı (transformers paketi eksik)")

        if self._processor is None:
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
        return self._processor

    def analyze(self, image: np.ndarray) -> NodeResult:
        """
        Görselin CLIP embedding'ini analiz eder.

        Args:
            image: NumPy array formatında görsel (H, W, C)

        Returns:
            NodeResult: CLIP analizi sonucu
        """
        self.validate_image(image)

        metadata: Dict[str, Any] = {}
        score = 0.0
        verdict = "REAL"
        confidence = 0.5

        if not CLIP_AVAILABLE:
            # CLIP yoksa, nötr sonuç döndür
            metadata["error"] = "CLIP modeli kurulmadı"
            metadata["available"] = False
            return NodeResult(
                score=50.0,
                verdict="UNCERTAIN",
                metadata=metadata,
                confidence=0.0,
                node_name=self.name
            )

        try:
            # 1. Embedding'i çıkar
            embedding = self._get_embedding(image)

            # 2. Kalibre edilmişse, anomali skoru hesapla
            if self._is_calibrated and len(self._real_embeddings) > 0:
                anomaly_score = self._compute_anomaly_score(embedding)

                if anomaly_score > self.threshold:
                    score = 50.0 + anomaly_score * 50.0
                    verdict = "FAKE"
                    confidence = anomaly_score
                else:
                    score = anomaly_score * 40.0
                    verdict = "REAL"
                    confidence = 1.0 - anomaly_score

                metadata["anomaly_score"] = float(anomaly_score)
                metadata["calibrated"] = True
            else:
                # Kalibre edilmemişse, nötr sonuç
                score = 50.0
                verdict = "UNCERTAIN"
                confidence = 0.5
                metadata["calibrated"] = False
                metadata["message"] = "Model kalibre edilmemiş"

            # 3. Embedding istatistikleri
            metadata["embedding_mean"] = float(np.mean(embedding))
            metadata["embedding_std"] = float(np.std(embedding))
            metadata["embedding_norm"] = float(np.linalg.norm(embedding))
            metadata["model_name"] = self.model_name
            metadata["available"] = True

        except Exception as e:
            metadata["error"] = str(e)
            metadata["available"] = True
            score = 50.0
            verdict = "UNCERTAIN"
            confidence = 0.0

        return NodeResult(
            score=score,
            verdict=verdict,
            metadata=metadata,
            confidence=confidence,
            node_name=self.name
        )

    def _get_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Görselden CLIP embedding'i çıkarır.

        Args:
            image: Görsel

        Returns:
            Embedding vektörü
        """
        # NumPy array'i PIL Image'a çevir
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)

        # CLIP input hazırla
        inputs = self.processor(images=pil_image, return_tensors="pt")

        # GPU/CPU'ya taşı
        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Embedding çıkar (no grad)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # CPU'ya taşı ve numpy'a çevir
        embedding = image_features.cpu().numpy()[0]

        return embedding

    def _compute_anomaly_score(self, embedding: np.ndarray) -> float:
        """
        Embedding'in anomali skoruunu hesaplar.

        Gerçek görsel embedding'lerine olan uzaklığı ölçer.

        Args:
            embedding: Hesaplanacak embedding

        Returns:
            0-1 arası anomali skoru
        """
        if not self._real_embeddings:
            return 0.5  # Bilinmiyor

        # Gerçek embedding'lerle olan ortalama cosine distance
        distances = []
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        for real_emb in self._real_embeddings:
            real_norm = real_emb / (np.linalg.norm(real_emb) + 1e-8)
            # Cosine distance = 1 - cosine similarity
            distance = 1.0 - np.dot(embedding_norm, real_norm)
            distances.append(distance)

        avg_distance = np.mean(distances)

        # Normalize et (0-1 arası)
        # Tipik cosine distance 0-0.5 arası
        anomaly_score = min(1.0, avg_distance * 2.0)

        return anomaly_score

    def calibrate(self, real_images: List[np.ndarray],
                  fake_images: Optional[List[np.ndarray]] = None):
        """
        Modeli gerçek ve sahte görsellerle kalibre eder.

        Args:
            real_images: Gerçek görsel listesi
            fake_images: Sahte görsel listesi (opsiyonel)
        """
        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP modeli kurulmadı")

        print(f"CLIPNode kalibre ediliyor...")

        # Gerçek görsel embedding'lerini topla
        self._real_embeddings = []
        for img in real_images:
            try:
                emb = self._get_embedding(img)
                self._real_embeddings.append(emb)
            except Exception as e:
                print(f"Uyarı: Gerçek görsel embedding hatası: {e}")

        # Sahte görsel embedding'lerini topla (opsiyonel)
        if fake_images:
            self._fake_embeddings = []
            for img in fake_images:
                try:
                    emb = self._get_embedding(img)
                    self._fake_embeddings.append(emb)
                except Exception as e:
                    print(f"Uyarı: Sahte görsel embedding hatası: {e}")

        self._is_calibrated = len(self._real_embeddings) > 0
        print(f"Kalibrasyon tamamlandı: {len(self._real_embeddings)} gerçek, "
              f"{len(self._fake_embeddings)} sahte embedding")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Görseli CLIP için ön işleme tabii tutar.

        Args:
            image: İşlenecek görsel

        Returns:
            Ön işleme yapılmış görsel
        """
        # CLIP processor kendi ön işlemesini yapar
        return image

    def is_available(self) -> bool:
        """CLIP modelinin kullanılabilirliğini kontrol eder"""
        return CLIP_AVAILABLE

    def get_model_info(self) -> Dict[str, Any]:
        """Model bilgilerini döndürür"""
        return {
            "available": CLIP_AVAILABLE,
            "model_name": self.model_name,
            "device": str(self.model.device) if self._model else "not_loaded",
            "calibrated": self._is_calibrated,
            "real_embeddings_count": len(self._real_embeddings),
            "fake_embeddings_count": len(self._fake_embeddings)
        }
