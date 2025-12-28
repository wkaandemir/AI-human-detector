# System Architecture & Technical Reference

## 1. Hibrit Analiz Mimarisi (The "Ensemble" Engine)
Bu proje, tek bir modele güvenmek yerine, farklı üretim tekniklerini yakalamak için **4 Katmanlı Hibrit Mimari** kullanır. Her katman ("Node") görsel üzerinde farklı bir analiz çalıştırır ve sonuçlar ağırlıklı ortalama ile birleştirilir.

### Analiz Node'ları (Detection Pipelines)
1.  **Watermark Node (Temel Kontrol):** 
    - **Görev:** Görselin metadata ve piksel verilerinde bilinen AI imzalarını (örn. `imwatermark`) arar.
    - **Teknoloji:** `imwatermark`, `exifread`.
    - **Davranış:** Pozitif sonuç durumunda analizi erken bitirir (Short-circuit).

2.  **DIRE (Diffusion Reconstruction Error):**
    - **Görev:** Görseli bir Diffusion modeli (örn. SD v1.5) ile "tersine" çevirir ve yeniden üretir. Orijinal ile rekonstrüksiyon arasındaki fark (hata haritası), difüzyon modellerinin izini taşır.
    - **Teknoloji:** PyTorch, Diffusers, Stable Diffusion.
    - **Donanım:** GPU gerektirir (Min 8GB VRAM).

3.  **UniversalFakeDetect (CLIP):**
    - **Görev:** Görseli anlamsal özellik uzayına (CLIP embedding) dönüştürür ve eğitim setinde hiç görülmemiş (Zero-shot) anormallikleri arar.
    - **Teknoloji:** OpenAI CLIP, Linear Probe Classifier.

4.  **Frekans & ELA (Low-Level Analysis):**
    - **Görev:** Fourier Transform (FFT) ile frekans anomalilerini (checkerboard artifacts) ve ELA ile sonradan yapılan manipülasyonları tespit eder.

## 2. Sistem Gereksinimleri
**Tamamen çevrimdışı (offline) ve yerel çalışacak şekilde tasarlanmıştır.**

*   **GPU:** CUDA destekli NVIDIA GPU (Minimum 8GB VRAM).
*   **RAM:** Minimum 16GB.
*   **Python:** 3.10+
*   **Depolama:** ~50GB (Model ağırlıkları ve önbellek için).

## 3. Veri Yapıları ve API
Sistem modülerdir. Her node `BaseNode` sınıfından türetilir ve ortak bir arayüz sunar.

### API Endpoint (Hedeflenen)
İleride bir REST API olarak sunulacaktır:

**`POST /api/v1/analyze`**
- **Girdi:** Görsel Dosyası (Multipart).
- **Parametreler:** `check_metadata=True` (Opsiyonel).
- **Çıktı (JSON):**
  ```json
  {
    "final_score": 85.4,  // 0-100 (Yapay Zeka Olasılığı)
    "verdict": "FAKE",
    "details": {
      "dire_score": 92.0,
      "watermark_detected": false
    }
  }
  ```
