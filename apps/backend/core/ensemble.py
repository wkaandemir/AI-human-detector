"""
Ensemble motoru - Tüm node'ların sonuçlarını birleştirir
"""

from typing import List, Dict, Optional
import numpy as np
import time
from .base_node import BaseNode
from .models import NodeResult, EnsembleResult


class EnsembleEngine:
    """
    Tüm detection node'larını koordine eden ve sonuçları birleştiren motor.

    Her node'un sonucunu ağırlıklandırarak final skor hesaplar.
    """

    def __init__(self, nodes: List[BaseNode], threshold: float = 50.0):
        """
        Ensemble motor yapıcısı

        Args:
            nodes: Analiz node'ları listesi
            threshold: AI/Fake kararı için eşik değeri (0-100)
        """
        self.nodes = nodes
        self.threshold = max(0.0, min(100.0, threshold))

    def analyze(self, image: np.ndarray) -> EnsembleResult:
        """
        Görseli tüm node'lar ile analiz eder ve sonuçları birleştirir.

        Args:
            image: NumPy array formatında görsel (H, W, C)

        Returns:
            EnsembleResult: Birleştirilmiş analiz sonucu
        """
        start_time = time.time()
        node_results: Dict[str, NodeResult] = {}

        # Her node ile analiz yap
        for node in self.nodes:
            if not node.enabled:
                continue

            try:
                node_start = time.time()
                result = node.analyze(image)
                result.processing_time = time.time() - node_start
                node_results[node.name] = result

                # Watermark node pozitif bulursa short-circuit
                if node.name == "WatermarkNode" and result.verdict == "FAKE":
                    # Yüksek güvenle direkt fake döndür
                    final_score = 95.0
                    return EnsembleResult(
                        final_score=final_score,
                        verdict=self._compute_verdict(final_score),
                        node_results=node_results,
                        processing_time=time.time() - start_time
                    )

            except Exception as e:
                # Hata durumunda node'u atla, logla
                print(f"Uyarı: {node.name} başarısız: {str(e)}")
                continue

        # Sonuçları birleştir
        final_score = self._aggregate_scores(node_results)
        verdict = self._compute_verdict(final_score)

        return EnsembleResult(
            final_score=final_score,
            verdict=verdict,
            node_results=node_results,
            processing_time=time.time() - start_time
        )

    def _aggregate_scores(self, results: Dict[str, NodeResult]) -> float:
        """
        Node sonuçlarını ağırlıklı ortalama ile birleştirir.

        Args:
            results: Node sonuçları sözlüğü

        Returns:
            0-100 arası final skor
        """
        if not results:
            return 50.0  # Belirsiz

        total_weight = 0.0
        weighted_sum = 0.0

        for node_name, result in results.items():
            # Node'un ağırlığını bul
            node = self._get_node_by_name(node_name)
            if node:
                weight = node.weight
                weighted_sum += result.score * weight
                total_weight += weight

        if total_weight == 0:
            return 50.0

        final_score = weighted_sum / total_weight
        return max(0.0, min(100.0, final_score))

    def _compute_verdict(self, score: float) -> str:
        """
        Skora göre karar verir.

        Args:
            score: 0-100 arası skor

        Returns:
            "REAL", "FAKE", veya "UNCERTAIN"
        """
        if score >= self.threshold + 10:
            return "FAKE"
        elif score <= self.threshold - 10:
            return "REAL"
        else:
            return "UNCERTAIN"

    def _get_node_by_name(self, name: str) -> Optional[BaseNode]:
        """
        İsmi verilen node'u döndürür.

        Args:
            name: Node adı

        Returns:
            BaseNode veya None
        """
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def add_node(self, node: BaseNode) -> None:
        """
        Motora yeni bir node ekler.

        Args:
            node: Eklenecek node
        """
        self.nodes.append(node)

    def remove_node(self, node_name: str) -> bool:
        """
        İsmi verilen node'u motordan çıkarır.

        Args:
            node_name: Çıkarılacak node adı

        Returns:
            Başarılı ise True, bulunamadıysa False
        """
        for i, node in enumerate(self.nodes):
            if node.name == node_name:
                self.nodes.pop(i)
                return True
        return False

    def enable_node(self, node_name: str) -> bool:
        """
        Node'u aktifleştirir.

        Args:
            node_name: Aktifleştirilecek node adı

        Returns:
            Başarılı ise True, bulunamadıysa False
        """
        node = self._get_node_by_name(node_name)
        if node:
            node.enabled = True
            return True
        return False

    def disable_node(self, node_name: str) -> bool:
        """
        Node'u pasifleştirir.

        Args:
            node_name: Pasifleştirilecek node adı

        Returns:
            Başarılı ise True, bulunamadıysa False
        """
        node = self._get_node_by_name(node_name)
        if node:
            node.enabled = False
            return True
        return False

    def __repr__(self) -> str:
        """Motorun string temsili"""
        enabled_count = sum(1 for n in self.nodes if n.enabled)
        return f"EnsembleEngine(nodes={len(self.nodes)}, enabled={enabled_count})"
