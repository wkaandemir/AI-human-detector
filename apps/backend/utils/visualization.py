"""
Görselleştirme Araçları - Error Map, Frekans Spektrumu vb.
"""

from typing import Optional, Tuple
import numpy as np
from PIL import Image
import io


def visualize_error_map(error_map: np.ndarray,
                       colormap: str = 'jet',
                       save_path: Optional[str] = None) -> np.ndarray:
    """
    Error map'i renkli görselleştirir.

    Args:
        error_map: Error map (H, W) grayscale
        colormap: Renk paleti ('jet', 'hot', 'viridis')
        save_path: Kaydedilecek dosya yolu (opsiyonel)

    Returns:
        Görselleştirilmiş error map (H, W, 3) RGB
    """
    # Error map'i normalize et [0, 255]
    error_normalized = ((error_map - error_map.min()) /
                       (error_map.max() - error_map.min() + 1e-8) * 255).astype(np.uint8)

    # PIL Image'a çevir
    pil_img = Image.fromarray(error_normalized, mode='L')

    # Renkli haritaya çevir (false color)
    if colormap == 'jet':
        # Jet colormap manuel implementasyonu
        colored = _apply_jet_colormap(pil_img)
    elif colormap == 'hot':
        colored = _apply_hot_colormap(pil_img)
    else:  # viridis veya varsayılan
        colored = _apply_viridis_colormap(pil_img)

    # Kaydet
    if save_path:
        Image.fromarray(colored).save(save_path)

    return colored


def _apply_jet_colormap(image: Image.Image) -> np.ndarray:
    """Jet colormap uygular (mavi-kırmızı gradient)"""
    arr = np.array(image).astype(np.float32) / 255.0

    # Jet colormap (matplotlib benzeri)
    # Mavi -> Cyan -> Yeşil -> Sarı -> Kırmızı
    result = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)

    for i in range(4):
        mask = (arr > i * 0.25) & (arr <= (i + 1) * 0.25)
        local_val = (arr[mask] - i * 0.25) * 4.0

        if i == 0:  # Mavi -> Cyan
            result[mask, 2] = 255
            result[mask, 1] = (local_val * 255).astype(np.uint8)
            result[mask, 0] = 0
        elif i == 1:  # Cyan -> Yeşil
            result[mask, 2] = ((1 - local_val) * 255).astype(np.uint8)
            result[mask, 1] = 255
            result[mask, 0] = 0
        elif i == 2:  # Yeşil -> Sarı
            result[mask, 2] = 0
            result[mask, 1] = 255
            result[mask, 0] = (local_val * 255).astype(np.uint8)
        else:  # Sarı -> Kırmızı
            result[mask, 2] = 0
            result[mask, 1] = ((1 - local_val) * 255).astype(np.uint8)
            result[mask, 0] = 255

    return result


def _apply_hot_colormap(image: Image.Image) -> np.ndarray:
    """Hot colormap uygular (siyah-kırmızı-sarı-beyaz)"""
    arr = np.array(image).astype(np.float32) / 255.0
    result = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)

    for i in range(3):
        mask = (arr > i * 0.33) & (arr <= (i + 1) * 0.33)
        local_val = (arr[mask] - i * 0.33) * 3.0

        if i == 0:  # Siyah -> Kırmızı
            result[mask, 0] = (local_val * 255).astype(np.uint8)
            result[mask, 1] = 0
            result[mask, 2] = 0
        elif i == 1:  # Kırmızı -> Sarı
            result[mask, 0] = 255
            result[mask, 1] = (local_val * 255).astype(np.uint8)
            result[mask, 2] = 0
        else:  # Sarı -> Beyaz
            result[mask, 0] = 255
            result[mask, 1] = 255
            result[mask, 2] = (local_val * 255).astype(np.uint8)

    return result


def _apply_viridis_colormap(image: Image.Image) -> np.ndarray:
    """Viridis colormap uygular (mor-sarı gradient)"""
    arr = np.array(image).astype(np.float32) / 255.0
    result = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)

    # Basitleştirilmiş viridis
    result[:, :, 0] = ((0.2 + 0.6 * arr) * 255).astype(np.uint8)  # R
    result[:, :, 1] = ((0.1 + 0.5 * (1 - arr)) * 255).astype(np.uint8)  # G
    result[:, :, 2] = ((0.3 + 0.5 * (1 - arr)) * 255).astype(np.uint8)  # B

    return result


def visualize_frequency_spectrum(magnitude: np.ndarray,
                                 log_scale: bool = True,
                                 save_path: Optional[str] = None) -> np.ndarray:
    """
    Frekans spektrumunu görselleştirir.

    Args:
        magnitude: FFT magnitude spektrumu (H, W)
        log_scale: Logaritmik ölçek kullan
        save_path: Kaydedilecek dosya yolu (opsiyonel)

    Returns:
        Görselleştirilmiş spektrum (H, W, 3) RGB
    """
    # Log scale
    if log_scale:
        spectrum = np.log(magnitude + 1)
    else:
        spectrum = magnitude.copy()

    # Normalize [0, 255]
    spectrum_min, spectrum_max = spectrum.min(), spectrum.max()
    if spectrum_max > spectrum_min:
        spectrum_normalized = ((spectrum - spectrum_min) /
                              (spectrum_max - spectrum_min) * 255).astype(np.uint8)
    else:
        spectrum_normalized = np.zeros_like(spectrum, dtype=np.uint8)

    # Grayscale -> RGB
    if len(spectrum_normalized.shape) == 2:
        spectrum_rgb = np.stack([spectrum_normalized] * 3, axis=2)
    else:
        spectrum_rgb = spectrum_normalized

    # Kaydet
    if save_path:
        Image.fromarray(spectrum_rgb).save(save_path)

    return spectrum_rgb


def visualize_ela_map(ela_map: np.ndarray,
                     save_path: Optional[str] = None) -> np.ndarray:
    """
    ELA (Error Level Analysis) map'ini görselleştirir.

    Args:
        ela_map: ELA map (H, W) veya (H, W, C)
        save_path: Kaydedilecek dosya yolu (opsiyonel)

    Returns:
        Görselleştirilmiş ELA map (H, W, 3) RGB
    """
    # Gri tonlamalı ise
    if len(ela_map.shape) == 2:
        ela_normalized = np.clip(ela_map, 0, 255).astype(np.uint8)
        ela_rgb = np.stack([ela_normalized] * 3, axis=2)
    else:
        # Zaten RGB
        ela_rgb = np.clip(ela_map, 0, 255).astype(np.uint8)

    # Kaydet
    if save_path:
        Image.fromarray(ela_rgb).save(save_path)

    return ela_rgb


def create_comparison_gallery(images: list,
                             titles: list,
                             cols: int = 2,
                             save_path: Optional[str] = None) -> np.ndarray:
    """
    Birden fazla görseli yan yana gösteren galeri oluşturur.

    Args:
        images: Görsel listesi (numpy array'ler)
        titles: Başlık listesi
        cols: Sütun sayısı
        save_path: Kaydedilecek dosya yolu (opsiyonel)

    Returns:
        Galeri görseli (H, W, 3) RGB
    """
    if len(images) != len(titles):
        raise ValueError("Görsel ve başlık sayıları eşleşmeli")

    # Tüm görselleri aynı boyuta getir
    target_size = images[0].shape[:2]
    resized_images = []

    for img in images:
        if img.shape[:2] != target_size:
            pil_img = Image.fromarray(img.astype(np.uint8))
            pil_img = pil_img.resize((target_size[1], target_size[0]), Image.LANCZOS)
            resized_images.append(np.array(pil_img))
        else:
            resized_images.append(img.astype(np.uint8))

    # Grid oluştur
    rows = (len(images) + cols - 1) // cols
    cell_width = target_size[1]
    cell_height = target_size[0]

    # Galeri canvas'ı (başlıklar için ekstra boşluk)
    title_height = 30
    gallery = np.ones((rows * (cell_height + title_height),
                      cols * cell_width, 3), dtype=np.uint8) * 255

    # Görselleri yerleştir
    for idx, (img, title) in enumerate(zip(resized_images, titles)):
        row = idx // cols
        col = idx % cols

        y_start = row * (cell_height + title_height) + title_height
        y_end = y_start + cell_height
        x_start = col * cell_width
        x_end = x_start + cell_width

        gallery[y_start:y_end, x_start:x_end] = img

    # Kaydet
    if save_path:
        Image.fromarray(gallery).save(save_path)

    return gallery


def overlay_heatmap(image: np.ndarray,
                   heatmap: np.ndarray,
                   alpha: float = 0.5,
                   save_path: Optional[str] = None) -> np.ndarray:
    """
    Orijinal görselin üzerine heatmap bindirir.

    Args:
        image: Orijinal görsel (H, W, 3)
        heatmap: Heatmap (H, W) veya (H, W, 3)
        alpha: Şeffaflık (0-1 arası)
        save_path: Kaydedilecek dosya yolu (opsiyonel)

    Returns:
        Heatmap bindirilmiş görsel (H, W, 3)
    """
    # Görselleri float32'ye çevir
    img_float = image.astype(np.float32)

    if len(heatmap.shape) == 2:
        # Grayscale heatmap -> RGB
        heatmap_normalized = ((heatmap - heatmap.min()) /
                             (heatmap.max() - heatmap.min() + 1e-8))
        heatmap_rgb = _apply_jet_colormap(
            Image.fromarray((heatmap_normalized * 255).astype(np.uint8))
        ).astype(np.float32)
    else:
        heatmap_rgb = heatmap.astype(np.float32)

    # Boyut kontrolü
    if img_float.shape != heatmap_rgb.shape:
        pil_img = Image.fromarray(image.astype(np.uint8))
        pil_img = pil_img.resize((heatmap_rgb.shape[1], heatmap_rgb.shape[0]))
        img_float = np.array(pil_img).astype(np.float32)

    # Blend
    result = (img_float * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)

    # Kaydet
    if save_path:
        Image.fromarray(result).save(save_path)

    return result
