# 🕵️‍♂️ AI Human Detector
**(Yapay Zeka Kaynaklı İnsan Görseli Tespiti)**

![Status](https://img.shields.io/badge/Status-Prototype-yellow)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

> **Vizyon:** İnsan yüzü ve bedeni içeren görsellerin yapay zeka (AI) tarafından üretilip üretilmediğini tespit eden, **tamamen açık kaynaklı (open-source)** ve **yerel (local)** çalışan bir akademik analiz sistemi.

## 🚀 Nedir ve Nasıl Çalışır?
Mevcut dedektörlerin aksine, "tek bir sihirli değnek" kullanmaz. **4 Katmanlı Hibrit Mimari** kullanır:

1.  **Watermark Check:** Dosya imzasını kontrol eder.
2.  **DIRE:** Difüzyon modellerinin matematiksel izini sürer.
3.  **CLIP:** Anlamsal gariplikleri yakalar.
4.  **Frekans Analizi:** Piksellerdeki görünmez desenleri bulur.

🔗 **Detaylı Teknik Mimari:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## 🛠️ Hızlı Başlangıç (Kurulum)

### Gereksinimler
*   **GPU:** NVIDIA GPU (Min 8GB VRAM önerilir).
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

# 3. Bağımlılıkları Yükleyin (Yakında)
# pip install -r requirements.txt
```

## 🗺️ Yol Haritası ve Durum
Proje şu anda **Faz 0 (Yapılandırma)** aşamasındadır. Kodlama süreci başlamak üzeredir.

🔗 **Detaylı Plan:** [docs/ROADMAP.md](docs/ROADMAP.md)

## 📂 Proje Yapısı
```
.
├── apps/
│   ├── backend/   # Python AI Motoru (Buradayız 📍)
│   └── frontend/  # Web Arayüzü (Planlanıyor)
├── docs/          # Dokümantasyon
└── README.md      # Giriş
```

## 📚 Dokümantasyon

- **[📖 Proje İndeksi](docs/INDEX.md)** - Tüm dokümanların ve bileşenlerin kapsamlı indeksi
- **[🏗️ Teknik Mimari](docs/ARCHITECTURE.md)** - 4 katmanlı hibrit analiz mimarisi
- **[🗺️ Yol Haritası](docs/ROADMAP.md)** - Fazlar ve yapılacaklar listesi
- **[🎓 Eğitim Rehberi](docs/TRAINING.md)** - DIRE modeli eğitim ve iyileştirme

## 🤝 Katkıda Bulunma
Her türlü katkıya açığız! Lütfen [docs/ROADMAP.md](docs/ROADMAP.md) dosyasındaki görevleri inceleyin.
