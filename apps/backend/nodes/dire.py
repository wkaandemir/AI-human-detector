"""
DIRE Node - Diffusion Reconstruction Error
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
from ..core.base_node import BaseNode
from ..core.models import NodeResult

# İsteğe bağlı import'lar
try:
    import torch
    import torch.nn as nn
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from torchvision.transforms import functional as TF
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Uyarı: diffusers kurulmadı, DIRE analizi devre dışı")


class DIRENode(BaseNode):
    """
    Diffusion Reconstruction Error (DIRE) node'u.

    Görseli diffusion modeli ile tersine çevirir (DDIM inversion),
    sonra tekrar oluşturur ve aradaki hatayı (reconstruction error)
    analiz ederek AI-generated tespiti yapar.

    GPU gerektirir (min 8GB VRAM).
    """

    def __init__(self,
                 weight: float = 1.0,
                 model_name: str = "runwayml/stable-diffusion-v1-5",
                 num_steps: int = 50,
                 device: Optional[str] = None,
                 classifier_path: Optional[str] = None):
        """
        DIRENode yapıcısı

        Args:
            weight: Node ağırlığı (varsayılan: 1.0)
            model_name: Stable Diffusion model adı
            num_steps: DDIM inversion/adım sayısı
            device: CPU veya CUDA (None = auto)
            classifier_path: Error map classifier yolu (opsiyonel)
        """
        super().__init__(weight=weight, name="DIRENode")
        self.model_name = model_name
        self.num_steps = num_steps
        self.device = device

        # Model ve scheduler (lazy loading)
        self._pipeline = None
        self._scheduler = None

        # Classifier (opsiyonel)
        self._classifier = None
        self.classifier_path = classifier_path

    @property
    def pipeline(self):
        """Stable Diffusion pipeline'ını lazy load ile döndürür"""
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("Diffusers kurulmadı")

        if self._pipeline is None:
            print(f"DIRE: {self.model_name} yükleniyor...")
            self._pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
            )

            # Device ayarla
            if self.device:
                target_device = self.device
            else:
                target_device = "cuda" if torch.cuda.is_available() else "cpu"

            self._pipeline.to(target_device)

            # GPU bellek optimizasyonu
            if target_device == "cuda":
                self._pipeline.enable_attention_slicing()
                # Opsiyonel: CPU offloading için daha az VRAM
                # self._pipeline.enable_sequential_cpu_offload()

            self._pipeline.eval()
            print(f"DIRE: Model yüklendi ({target_device})")

        return self._pipeline

    @property
    def scheduler(self):
        """DDIM scheduler'ı lazy load ile döndürür"""
        if self._scheduler is None:
            self._scheduler = DDIMScheduler.from_config(
                self.pipeline.scheduler.config
            )
        return self._scheduler

    def analyze(self, image: np.ndarray) -> NodeResult:
        """
        Görseli DIRE ile analiz eder.

        Args:
            image: NumPy array formatında görsel (H, W, C)

        Returns:
            NodeResult: DIRE analizi sonucu
        """
        self.validate_image(image)

        metadata: Dict[str, Any] = {}
        score = 0.0
        verdict = "REAL"
        confidence = 0.5

        if not DIFFUSERS_AVAILABLE:
            metadata["error"] = "Diffusers kurulmadı"
            metadata["available"] = False
            return NodeResult(
                score=50.0,
                verdict="UNCERTAIN",
                metadata=metadata,
                confidence=0.0,
                node_name=self.name
            )

        # GPU kontrolü
        if not torch.cuda.is_available():
            metadata["error"] = "CUDA gerekiyor (GPU bulunamadı)"
            metadata["available"] = True
            return NodeResult(
                score=50.0,
                verdict="UNCERTAIN",
                metadata=metadata,
                confidence=0.0,
                node_name=self.name
            )

        try:
            # 1. Görseli hazırla (512x512)
            prepared_image = self._prepare_image(image)
            metadata["original_size"] = image.shape[:2]

            # 2. DDIM Inversion
            latents = self._ddim_invert(prepared_image)
            metadata["inversion_completed"] = True

            # 3. Reconstruction
            reconstructed = self._reconstruct(latents)
            metadata["reconstruction_completed"] = True

            # 4. Error Map hesapla
            error_map = self._compute_error_map(prepared_image, reconstructed)

            # 5. Error map'ten skor çıkar
            error_score = self._compute_error_score(error_map)
            metadata["error_score"] = float(error_score)

            # 6. AI / Real kararı
            # Yüksek hata = AI-generated (diffusion reconstruction hatası)
            # Düşük hata = Real photo (iyi reconstruction)
            if error_score > 0.3:
                score = 50.0 + error_score * 50.0
                verdict = "FAKE"
                confidence = min(1.0, error_score)
            else:
                score = error_score * 30.0
                verdict = "REAL"
                confidence = 0.7

            metadata["error_mean"] = float(np.mean(error_map))
            metadata["error_std"] = float(np.std(error_map))
            metadata["error_max"] = float(np.max(error_map))
            metadata["model_name"] = self.model_name
            metadata["num_steps"] = self.num_steps
            metadata["available"] = True

            # Bellek temizle
            self._cleanup_memory()

        except Exception as e:
            metadata["error"] = str(e)
            metadata["available"] = True
            score = 50.0
            verdict = "UNCERTAIN"
            confidence = 0.0
            self._cleanup_memory()

        return NodeResult(
            score=score,
            verdict=verdict,
            metadata=metadata,
            confidence=confidence,
            node_name=self.name
        )

    def _prepare_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Görseli SD input formatına hazırlar (512x512).

        Args:
            image: Hazırlanacak görsel

        Returns:
            Hazırlanmış görsel tensor
        """
        # NumPy -> PIL
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)

        # 512x512 resize (SD default size)
        pil_image = pil_image.resize((512, 512), Image.LANCZOS)

        # RGB kontrolü
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Normalize [-1, 1]
        image_tensor = TF.to_tensor(pil_image)  # [0, 1]
        image_tensor = image_tensor * 2.0 - 1.0  # [-1, 1]

        # Batch dim ekle
        image_tensor = image_tensor.unsqueeze(0)

        # GPU'ya taşı
        return image_tensor.to(self.pipeline.device)

    def _ddim_invert(self, image: torch.Tensor) -> torch.Tensor:
        """
        DDIM inversion ile görseli noise latent'lerine çevirir.

        Args:
            image: Input görsel tensor

        Returns:
            Noise latent'leri
        """
        # VAE encode
        with torch.no_grad():
            latents = self.pipeline.vae.encode(image).latent_dist.sample()
            latents = latents * 0.18215  # SD scaling factor

        # DDIM inversion
        self.scheduler.set_timesteps(self.num_steps)

        # Inversion loop
        for t in self.scheduler.timesteps:
            # Noise prediction
            with torch.no_grad():
                noise_pred = self.pipeline.unet(
                    latents,
                    t,
                    encoder_hidden_states=self.pipeline._encode_prompt(
                        "", self.pipeline.device, 1, False, ""
                    )
                ).sample

            # DDIM inversion step
            latents = self.scheduler.step(
                noise_pred, t, latents, eta=0.0  # eta=0 deterministik
            ).prev_sample

        return latents

    def _reconstruct(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Noise latent'lerinden görsel reconstruct eder.

        Args:
            latents: Noise latent'leri

        Returns:
            Reconstructed görsel tensor
        """
        # DDIM reconstruction
        latent_samples = latents.clone()

        for t in self.scheduler.timesteps[::-1]:  # Tersten
            # Noise prediction
            with torch.no_grad():
                noise_pred = self.pipeline.unet(
                    latent_samples,
                    t,
                    encoder_hidden_states=self.pipeline._encode_prompt(
                        "", self.pipeline.device, 1, False, ""
                    )
                ).sample

            # DDIM step
            latent_samples = self.scheduler.step(
                noise_pred, t, latent_samples, eta=0.0
            ).prev_sample

        # VAE decode
        with torch.no_grad():
            decoded = self.pipeline.vae.decode(
                latent_samples / 0.18215
            ).sample

        # [-1, 1] -> [0, 1]
        decoded = (decoded + 1.0) / 2.0
        decoded = decoded.clamp(0.0, 1.0)

        return decoded

    def _compute_error_map(self, original: torch.Tensor,
                           reconstructed: torch.Tensor) -> np.ndarray:
        """
        Orijinal ve reconstructed arasındaki error map'i hesaplar.

        Args:
            original: Orijinal görsel tensor
            reconstructed: Reconstructed görsel tensor

        Returns:
            Error map (numpy array)
        """
        # CPU'ya taşı
        orig_np = original.detach().cpu()[0].numpy()  # [C, H, W]
        recon_np = reconstructed.detach().cpu()[0].numpy()

        # Kanak bazlı mutlak fark
        error_map = np.abs(orig_np - recon_np)  # [C, H, W]

        # Kanalları ortala
        error_map = np.mean(error_map, axis=0)  # [H, W]

        return error_map

    def _compute_error_score(self, error_map: np.ndarray) -> float:
        """
        Error map'ten AI olasılık skoru hesaplar.

        Args:
            error_map: Error map

        Returns:
            0-1 arası skor
        """
        # Ortalama hata
        mean_error = np.mean(error_map)

        # Normalize et (typical error 0-0.5 arası)
        score = min(1.0, mean_error * 2.0)

        return score

    def _cleanup_memory(self):
        """GPU belleğini temizler"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_available(self) -> bool:
        """DIRE modelinin kullanılabilirliğini kontrol eder"""
        return DIFFUSERS_AVAILABLE and torch.cuda.is_available()

    def get_model_info(self) -> Dict[str, Any]:
        """Model bilgilerini döndürür"""
        cuda_available = torch.cuda.is_available() if DIFFUSERS_AVAILABLE else False

        return {
            "available": self.is_available(),
            "model_name": self.model_name,
            "cuda_available": cuda_available,
            "num_steps": self.num_steps,
            "device": str(self.pipeline.device) if self._pipeline else "not_loaded"
        }

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Görseli ön işleme tabii tutar"""
        return image

    def visualize_error_map(self, error_map: np.ndarray,
                           save_path: str = None) -> np.ndarray:
        """
        Error map'i görselleştirir.

        Args:
            error_map: Error map (H, W) numpy array
            save_path: Kaydedilecek dosya yolu (opsiyonel)

        Returns:
            Görselleştirilmiş error map (H, W, 3) RGB
        """
        from ..utils.visualization import visualize_error_map as vis_error_map
        return vis_error_map(error_map, colormap='jet', save_path=save_path)
