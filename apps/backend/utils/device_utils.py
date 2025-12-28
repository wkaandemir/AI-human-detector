"""
GPU/CPU Fallback ve Cihaz Yönetim Araçları

Bu modül, GPU yoksa veya bellek yetmezise otomatik olarak CPU'ya geçiş
yapan mekanizmaları içerir.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_device(prefer_gpu: bool = True, min_vram_mb: int = 8000) -> str:
    """
    Mevcut donanım için en uygun cihazı döndürür.

    Args:
        prefer_gpu: GPU tercih et (varsa)
        min_vram_mb: Minimum GPU belleği (MB)

    Returns:
        "cuda", "cpu", veya None (hiçbir cihaz uygun değilse)
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch kurulmamış, CPU modunda devam ediliyor")
        return "cpu"

    # GPU kontrolü
    if prefer_gpu and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"✅ {gpu_count} GPU bulundu")

        # Her GPU'yu kontrol et
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            vram_mb = props.total_memory / 1024 / 1024

            logger.info(f"  GPU {i}: {props.name} ({vram_mb:.0f} MB VRAM)")

            if vram_mb >= min_vram_mb:
                logger.info(f"  → GPU {i} seçiliyor (yeterli VRAM)")
                return f"cuda:{i}"

        # Yeterli VRAM yok
        logger.warning(f"⚠️ Hiçbir GPU'da yeterli VRAM yok (min: {min_vram_mb} MB)")
        logger.info("  → CPU moduna geçiliyor")
        return "cpu"

    else:
        if prefer_gpu:
            logger.info("⚠️ GPU kullanılamıyor, CPU modunda devam ediliyor")
        return "cpu"


def get_optimal_batch_size(
    image_size: tuple = (512, 512),
    device: str = "cuda",
    safety_margin: float = 0.8
) -> int:
    """
    GPU belleğine göre optimal batch boyutunu hesaplar.

    Args:
        image_size: (Yükseklik, Genişlik)
        device: Cihaz ("cuda" veya "cpu")
        safety_margin: Güvenlik marjı (0-1)

    Returns:
        Optimal batch boyutu
    """
    if device == "cpu":
        return 1  # CPU'da her zaman 1

    try:
        import torch
    except ImportError:
        return 1

    if not torch.cuda.is_available():
        return 1

    # GPU belleği
    total_memory_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    usable_memory_mb = total_memory_mb * safety_margin

    # Tahmini bellek kullanımı (görsel başına MB)
    # Bu değer model tipine göre değişebilir
    h, w = image_size
    pixels = h * w
    estimated_mb_per_image = (pixels * 3 * 4) / 1024 / 1024  # RGB float32

    # Batch boyutu
    batch_size = max(1, int(usable_memory_mb / estimated_mb_per_image))

    # Maksimum 16 (performans için)
    batch_size = min(batch_size, 16)

    logger.info(f"📊 Optimal batch boyutu: {batch_size} "
               f"(toplam VRAM: {total_memory_mb:.0f} MB)")

    return batch_size


class GPUMemoryMonitor:
    """
    GPU bellek kullanımını izler ve yönetir.
    """

    def __init__(self, device: str = "cuda"):
        """
        GPU bellek monitörü

        Args:
            device: Cihaz ("cuda" veya "cuda:0")
        """
        self.device = device
        self._peak_memory = 0
        self._has_gpu = self._check_gpu()

    def _check_gpu(self) -> bool:
        """GPU'nun kullanılabilir olup olmadığını kontrol eder."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def get_memory_usage(self) -> dict:
        """
        Mevcut bellek kullanımını döndürür.

        Returns:
            Bellek kullanım bilgileri sözlüğü
        """
        if not self._has_gpu:
            return {
                "available": False,
                "allocated_mb": 0,
                "reserved_mb": 0,
                "total_mb": 0
            }

        import torch

        allocated = torch.cuda.memory_allocated(self.device) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(self.device) / 1024 / 1024
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024 / 1024

        # Peak güncelle
        if allocated > self._peak_memory:
            self._peak_memory = allocated

        return {
            "available": True,
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "total_mb": total,
            "free_mb": total - allocated,
            "peak_mb": self._peak_memory,
            "usage_percent": (allocated / total) * 100
        }

    def clear_cache(self):
        """GPU bellek önbelleğini temizler."""
        if self._has_gpu:
            import torch
            torch.cuda.empty_cache()
            logger.info("🧹 GPU bellek önbelleği temizlendi")

    def check_memory_limit(self, threshold_percent: float = 90.0) -> bool:
        """
        Bellek kullanımının eşik değerini aşıp aşmadığını kontrol eder.

        Args:
            threshold_percent: Eşik değeri (yüzde)

        Returns:
            True = Eşik aşıldı, False = Güvende
        """
        usage = self.get_memory_usage()

        if not usage["available"]:
            return False

        return usage["usage_percent"] >= threshold_percent

    def get_peak_memory(self) -> float:
        """
        Tepe bellek kullanımını döndürür.

        Returns:
        Tepe bellek (MB)
        """
        return self._peak_memory

    def reset_peak(self):
        """Tepe bellek sayaçını sıfırlar."""
        self._peak_memory = 0


def safe_gpu_operation(operation_fn, fallback_fn=None, max_retries: int = 2):
    """
    GPU operasyonunu güvenli şekilde çalıştırır, bellek yetmezse
    CPU'ya geçer veya fallback fonksiyonunu çağırır.

    Args:
        operation_fn: GPU operasyonu fonksiyonu
        fallback_fn: CPU fallback fonksiyonu (opsiyonel)
        max_retries: Maksimum deneme sayısı

    Returns:
        Operasyon sonucu

    Example:
        def gpu_task():
            # GPU yoğun işlem
            return result

        def cpu_fallback():
            # CPU versiyonu
            return result

        result = safe_gpu_operation(gpu_task, cpu_fallback)
    """
    import torch

    for attempt in range(max_retries):
        try:
            result = operation_fn()
            return result

        except RuntimeError as e:
            error_msg = str(e).lower()

            # CUDA out of memory hatası
            if "out of memory" in error_msg:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ GPU bellek hatası (deneme {attempt + 1}/{max_retries})")
                    torch.cuda.empty_cache()

                    # Fallback fonksiyonu varsa kullan
                    if fallback_fn is not None:
                        logger.info("  → CPU fallback kullanılıyor")
                        return fallback_fn()
                else:
                    logger.error("❌ GPU bellek hatası, son deneme başarısız")
                    if fallback_fn is not None:
                        logger.info("  → CPU fallback kullanılıyor")
                        return fallback_fn()
                    raise
            else:
                # Bellek hatası değil, yükselt
                raise

    return None


class DeviceContext:
    """
    Cihaz bağlamı yöneticisi - GPU/CPU geçişleri için.

    Example:
        with DeviceContext(prefer_gpu=True) as ctx:
            device = ctx.device
            model.to(device)
            # ... işlem yap
    """

    def __init__(self, prefer_gpu: bool = True, min_vram_mb: int = 8000):
        """
        Cihaz bağlamı

        Args:
            prefer_gpu: GPU tercih et
            min_vram_mb: Minimum VRAM
        """
        self.prefer_gpu = prefer_gpu
        self.min_vram_mb = min_vram_mb
        self.device = None
        self.memory_monitor = None

    def __enter__(self):
        """Bağlam girildiğinde"""
        self.device = get_device(self.prefer_gpu, self.min_vram_mb)

        if self.device.startswith("cuda"):
            self.memory_monitor = GPUMemoryMonitor(self.device)
            logger.info(f"🎮 GPU modu: {self.device}")
        else:
            logger.info("💻 CPU modu")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Bağlam çıkılırken"""
        if self.memory_monitor:
            peak = self.memory_monitor.get_peak_memory()
            logger.info(f"📊 Tepe bellek kullanımı: {peak:.0f} MB")
            self.memory_monitor.clear_cache()

        return False


# Convenience fonksiyonları
def auto_select_device(min_vram_mb: int = 8000) -> str:
    """
    Otomatik cihaz seçimi (kısayol)

    Args:
        min_vram_mb: Minimum VRAM

    Returns:
        "cuda", "cuda:X" veya "cpu"
    """
    return get_device(prefer_gpu=True, min_vram_mb=min_vram_mb)


def get_memory_info() -> dict:
    """
    Mevcut bellek bilgilerini döndürür (kısayol)

    Returns:
        Bellek bilgileri sözlüğü
    """
    monitor = GPUMemoryMonitor()
    return monitor.get_memory_usage()
