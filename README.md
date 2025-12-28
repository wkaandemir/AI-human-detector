# 🕵️‍♂️ AI Human Detector
**(Yapay Zeka Kaynaklı İnsan Görseli Tespiti)**

![Status](https://img.shields.io/badge/Status-Alpha-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

> **Vizyon:** İnsan yüzü ve bedeni içeren görsellerin yapay zeka (AI) tarafından üretilip üretilmediğini tespit eden, **tamamen açık kaynaklı (open-source)** ve **yerel (local)** çalışan bir akademik analiz sistemi.

## 🚀 Nedir ve Nasıl Çalışır?
Mevcut dedektörlerin aksine, "tek bir sihirli değnek" kullanmaz. **4 Katmanlı Hibrit Mimari** kullanır:

1.  **Watermark Check:** Dosya imzasını kontrol eder (imwatermark + EXIF).
2.  **DIRE:** Difüzyon modellerinin matematiksel izini sürer (DDIM inversion).
3.  **CLIP:** Anlamsal gariplikleri yakalar (Zero-shot semantic detection).
4.  **Frekans Analizi:** Piksellerdeki görünmez desenleri bulur (FFT + ELA).

🔗 **Detaylı Teknik Mimari:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## 🛠️ Hızlı Başlangıç

### Gereksinimler
*   **Python:** 3.10+
*   **GPU:** NVIDIA GPU (Min 8GB VRAM - DIRE node için).
*   **OS:** Windows veya Linux.

### Kurulum Adımları
```bash
# 1. Projeyi Klonlayın
git clone https://github.com/wkaandemir/AI-human-detector.git
cd AI-human-detector/apps/backend

# 2. Sanal Ortam Oluşturun
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate # Linux

# 3. Bağımlılıkları Yükleyin
pip install -r requirements.txt

# 4. API'yi Başlatın
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API dokümantasyonu: http://localhost:8000/docs

### Örnek API Kullanımı
```bash
# Görsel Analizi
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "image=@test.jpg" \
  -F "return_details=true"

# Response
{
  "final_score": 85.4,
  "verdict": "FAKE",
  "processing_time": 2.3,
  "timestamp": "2025-12-28T12:00:00",
  "details": {
    "WatermarkNode": {...},
    "DIRENode": {...},
    "CLIPNode": {...},
    "FrequencyNode": {...}
  }
}
```

## 🗺️ Yol Haritası ve Durum
Proje şu anda **Faz 2-4 (AI Modülleri & API)** aşamasındadır.

🔗 **Detaylı Plan:** [docs/ROADMAP.md](docs/ROADMAP.md)

| Faz | Durum | Tamamlanma |
|-----|-------|-----------|
| Faz 1: Çekirdek Kurulum | 🟢 Tamamlandı | 100% |
| Faz 2: AI Modülleri | 🟢 Tamamlandı | 100% |
| Faz 3: Test ve Doğrulama | 🟡 Planlanıyor | 0% |
| Faz 4: API ve Entegrasyon | 🟢 Tamamlandı | 100% |
| Faz 5: Frontend | 🔵 Planlanıyor | 0% |

## 📂 Proje Yapısı
```
apps/backend/
├── core/           # Temel sınıflar (BaseNode, EnsembleEngine)
├── nodes/          # 4 analiz node'u (Watermark, DIRE, CLIP, Frequency)
├── api/            # FastAPI uygulaması ve endpoint'ler
├── utils/          # Yardımcı fonksiyonlar
├── tests/          # Birim ve entegrasyon testleri
└── requirements.txt # Python bağımlılıkları
```

## 📚 Dokümantasyon

- **[📖 Proje İndeksi](docs/INDEX.md)** - Tüm dokümanların ve bileşenlerin kapsamlı indeksi
- **[🏗️ Teknik Mimari](docs/ARCHITECTURE.md)** - 4 katmanlı hibrit analiz mimarisi
- **[🗺️ Yol Haritası](docs/ROADMAP.md)** - Fazlar ve yapılacaklar listesi
- **[🎓 Eğitim Rehberi](docs/TRAINING.md)** - DIRE modeli eğitim ve iyileştirme
- **[📋 TODO Listesi](TODO.md)** - LLM tarafından takip edilebilir görev listesi

## 🧪 Test Çalıştırma

```bash
# Tüm testler
pytest

# Belirli bir test dosyası
pytest tests/test_watermark.py

# Test kapsamı raporu
pytest --cov=.

# Entegrasyon testleri
pytest -m integration
```

## 🤝 Katkıda Bulunma
Her türlü katkıya açığız! Lütfen [TODO.md](TODO.md) dosyasındaki görevleri inceleyin.

### Katkı Adımları
1. Projeyi fork'layın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit'leyin (`git commit -m 'feat: amazing feature eklendi'`)
4. Branch'inizi push'layın (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📝 Lisans
Bu proje MIT lisansı altında lisanslanmıştır. [LICENSE](LICENSE) dosyasına bakın.
