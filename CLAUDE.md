# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proje Hakkında

AI Human Detector, yapay zeka (AI) tarafından üretilen insan görsellerini tespit etmek için **4 katmanlı hibrit analiz mimarisi** kullanan, tamamen açık kaynaklı ve yerel (offline) çalışan bir sistemdir.

### Çalışma Dili ve Kurallar

1. **İletişim Dili**: Tüm yanıtlar **Türkçe** olmalıdır
2. **Git Branch**: Her zaman `main` branch'inde çalışın, başka branch'lere geçmeyin
3. **Commit Mesajları**: Türkçe olmalı, format: `tür(kapsam): açıklama`
   - Örnek: `feat: watermark node eklendi`, `fix: api bağlantı hatası giderildi`
4. **TODO.md İşaretleme**: %100 emin olmadığınız TODO öğelerini işaretlemeyin

## Mimari Genel Bakış

Proje, tek bir model yerine **4 analiz node'u** kullanır ve sonuçları ağırlıklı ortalama ile birleştirir:

1. **Watermark Node**: Görselde AI imzası (imwatermark, EXIF) arar. Pozitif bulursa analizi erken bitirir (short-circuit).
2. **DIRE Node** (En ağır): Diffusion modeli ile görseli tersine çevirir, yeniden üretir ve aradaki hatayı (reconstruction error) hesaplar. GPU gerektirir.
3. **CLIP Node**: Görseli embedding uzayına dönüştürür, anlamsal anormallikleri (zero-shot) arar.
4. **Frequency & ELA Node**: FFT ile frekans anomalilerini (checkerboard artifacts) ve ELA ile manipülasyon izlerini tespit eder.

Tüm node'lar `BaseNode` sınıfından türetilir ve `analyze(image) -> NodeResult` arayüzünü kullanır.

## Proje Yapısı

```
apps/
├── backend/              # Python AI Motoru (Ana çalışma alanı)
│   ├── core/            # BaseNode, NodeResult, EnsembleEngine
│   ├── nodes/           # 4 analiz node'u (watermark, dire, clip, frequency)
│   ├── api/             # FastAPI uygulaması
│   ├── utils/           # Yardımcı fonksiyonlar
│   └── tests/           # Unit ve entegrasyon testleri
└── frontend/            # Web arayüzü (Planlanıyor)

docs/
├── ARCHITECTURE.md      # 4 katmanlı mimarinin detaylı anlatımı
├── TRAINING.md          # DIRE modeli eğitim stratejileri
├── ROADMAP.md           # Faz bazlı ilerleme planı
└── INDEX.md             # Dokümantasyon indeksi

TODO.md                  # LLM tarafından takip edilebilir detaylı görev listesi
```

## Geliştirme Komutları

### Ortam Kurulumu (Faz 1 - Henüz tamamlanmadı)
```bash
cd apps/backend

# Sanal ortam oluştur
python -m venv venv
.\venv\Scripts\activate       # Windows
# source venv/bin/activate    # Linux

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### Test Çalıştırma
```bash
# Tüm testler
pytest

# Belirli bir test dosyası
pytest tests/test_watermark.py

# Test kapsamı raporu
pytest --cov=.
```

### API Çalıştırma (Faz 4 - Planlanıyor)
```bash
cd apps/backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### DIRE Node Eğitimi (İleri Seviye)
DIRE node'u için classifier eğitimi gerektiğinde `docs/TRAINING.md`'deki stratejileri takip edin. Kısaca:
1. Gerçek ve sahte görsellerden error map'leri çıkarın
2. Bu error map'ler üzerinde ResNet50 eğitin
3. Gerekirse difüzyon modelini fine-tune edin (DreamBooth/LoRA)

## Kodlama İlkeleri

### Node Geliştirme
- Her node `core/base_node.py`'deki `BaseNode` sınıfından türetilmeli
- `weight` parametresi ile node'un ağırlığı ayarlanabilmeli
- `NodeResult` döndürmeli: `score`, `verdict`, `metadata`, `confidence`

### GPU Bellek Yönetimi (Kritik)
- DIRE node GPU bellek yoğundur. Batch processing kullanın
- Bellek temizliği için `torch.cuda.empty_cache()` kullanın
- CPU fallback mekanizması ekleyin (opsiyonel)

### Test Yazma
- Her node için ayrı test dosyası (örn: `tests/test_dire.py`)
- Mock görüntüler ile edge case'leri test edin
- TDD prensibi: Kod yazmadan önce test yazın

## API Endpoint (Hedeflenen)

**`POST /api/v1/analyze`**
- Girdi: Görsel dosyası (multipart)
- Çıktı:
  ```json
  {
    "final_score": 85.4,      // 0-100 (Yapay Zeka Olasılığı)
    "verdict": "FAKE",
    "details": {
      "dire_score": 92.0,
      "watermark_detected": false
    }
  }
  ```

## İlerleme Takibi

Proje **Faz 0 (Yapılandırma)** aşamasında. Öncelikli sıra:

1. **Faz 1**: `apps/backend` yapısını oluştur, BaseNode ve Watermark node'u kodla
2. **Faz 2**: DIRE, CLIP, Frequency node'larını entegre et
3. **Faz 3**: Veri seti indir, testleri yaz, performansı doğrula
4. **Faz 4**: FastAPI uygulaması ve endpoint'ler
5. **Faz 5**: Frontend (opsiyonel)

Detaylı görev listesi için `TODO.md` dosyasına bakın.

## Önemli Dokümanlar

- **docs/ARCHITECTURE.md**: 4 katmanlı mimarinin teknik detayları
- **docs/TRAINING.md**: DIRE modeli eğitim stratejileri (classifier eğitimi, fine-tuning)
- **TODO.md**: LLM tarafından takip edilebilir, adım adım görev listesi

## Hedefler

- **Accuracy**: %95+
- **False Positive Rate**: <%2
- **Tamamen offline**: Çevrimdışı çalışmalı, harici API çağrısı olmamalı
- **GPU gereksinimi**: Min 8GB VRAM (DIRE node için)
