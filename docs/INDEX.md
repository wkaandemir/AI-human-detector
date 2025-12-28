# 📚 AI Human Detector - Proje İndeksi

Bu doküman, projenin tüm bileşenlerini, dokümanlarını ve yapılarını kapsamlı bir şekilde indeksler.

---

## 🗂️ İçindekiler

1. [Proje Özeti](#proje-özeti)
2. [Dokümanlar](#dokümanlar)
3. [Mimari Bileşenler](#mimari-bileşenler)
4. [Geliştirme Rehberi](#geliştirme-rehberi)
5. [Proje Yapısı](#proje-yapısı)

---

## 📋 Proje Özeti

### Temel Bilgiler
| Özellik | Değer |
|---------|-------|
| **Proje Adı** | AI Human Detector |
| **Açıklama** | Yapay zeka tarafından üretilen insan görsellerini tespit eden açık kaynak sistem |
| **Durum** | Prototype (Faz 0 - Yapılandırma) |
| **Lisans** | MIT |
| **Python Sürümü** | 3.10+ |
| **GPU Gereksinimi** | NVIDIA (Min 8GB VRAM) |

### Temel Özellikler
- ✅ **Tamamen Açık Kaynak**: Kapalı API'ler yok (Vertex AI, SynthID kaldırıldı)
- ✅ **Yerel Çalışma**: İnternet bağlantısı gerektirmez
- ✅ **4 Katmanlı Hibrit Analiz**: Watermark, DIRE, CLIP, Frekans Analizi
- ✅ **Akademik Araştırma**: %95+ doğruluk hedefi

---

## 📖 Dokümanlar

### Kullanıcı ve Geliştirici Dokümanları

| Doküman | Konum | Amaç |
|---------|-------|------|
| **README.md** | `/` | Proje giriş, hızlı başlangıç ve genel bakış |
| **ARCHITECTURE.md** | `/docs/` | Teknik mimari, sistem gereksinimleri ve API tasarımı |
| **ROADMAP.md** | `/docs/` | Proje yol haritası, fazlar ve yapılacaklar listesi |
| **TRAINING.md** | `/docs/` | DIRE modeli eğitim ve iyileştirme rehberi |

### Proje Yönetimi Dokümanları

| Doküman | Konum | Amaç |
|---------|-------|------|
| **CHANGELOG.md** | `/` | Sürüm geçmişi ve değişiklikler |
| **CONTRIBUTING.md** | `/` | Katkıda bulunma rehberi |
| **CODE_OF_CONDUCT.md** | `/` | Topluluk davranış kuralları |
| **SECURITY.md** | `/` | Güvenlik politikası ve raporlama |

---

## 🏗️ Mimari Bileşenler

### 1. Hibrit Analiz Motoru (4 Katman)

#### 📍 Watermark Node (Temel Kontrol)
- **Görev**: Metadata ve piksel verilerinde AI imzası arama
- **Teknolojiler**: `imwatermark`, `exifread`
- **Davranış**: Pozitif sonuçta analizi erken bitirir (short-circuit)
- **Durum**: ❓ Planlanıyor

#### 📍 DIRE Node (Diffusion Reconstruction Error)
- **Görev**: Görseli tersine çevirip yeniden üretme, hata haritası oluşturma
- **Teknolojiler**: PyTorch, Diffusers, Stable Diffusion v1.5
- **Donanım**: GPU (Min 8GB VRAM)
- **Durum**: ❓ Planlanıyor

#### 📍 CLIP Node (Semantic Anomaly Detection)
- **Görev**: Görseli CLIP embedding uzayına dönüştürme, anormallik tespiti
- **Teknolojiler**: OpenAI CLIP, Linear Probe Classifier
- **Özellik**: Zero-shot learning
- **Durum**: ❓ Planlanıyor

#### 📍 Frekans & ELA Node (Low-Level Analysis)
- **Görev**: Fourier Transform (FFT) ve ELA ile frekans anomalileri tespiti
- **Teknolojiler**: NumPy, OpenCV, FFT/DCT
- **Tespit Edilen**: Checkerboard artifacts, manipülasyon izleri
- **Durum**: ❓ Planlanıyor

---

### 2. Sistem API (Hedeflenen)

#### POST /api/v1/analyze
Görsel analizi endpoint'i

**İstek:**
```http
POST /api/v1/analyze
Content-Type: multipart/form-data

{
  "image": <file>,
  "check_metadata": true  // optional
}
```

**Cevap:**
```json
{
  "final_score": 85.4,      // 0-100 (Yapay Zeka Olasılığı)
  "verdict": "FAKE",        // REAL / FAKE / UNCERTAIN
  "confidence": 0.92,       // Güven skoru
  "details": {
    "watermark": {
      "detected": false,
      "type": null
    },
    "dire": {
      "score": 92.0,
      "error_map_path": "/tmp/dire_abc123.png"
    },
    "clip": {
      "score": 78.5,
      "anomaly_score": 0.65
    },
    "frequency": {
      "fft_score": 88.2,
      "ela_score": 45.0
    }
  },
  "processing_time_ms": 2340
}
```

---

### 3. Veri Yapıları

#### BaseNode Sınıfı
Tüm detection node'ları için temel sınıf

```python
class BaseNode:
    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def analyze(self, image: np.ndarray) -> NodeResult:
        """Görseli analiz et ve sonuç döndür"""
        raise NotImplementedError

    def get_weight(self) -> float:
        """Node ağırlığını döndür"""
        return self.weight
```

#### NodeResult Sınıfı
Analiz sonucu veri yapısı

```python
@dataclass
class NodeResult:
    score: float           # 0-100 arası skor
    verdict: str           # REAL/FAKE/UNCERTAIN
    metadata: dict         # Node'a özel ek veriler
    confidence: float      # Güven seviyesi
```

---

## 🛠️ Geliştirme Rehberi

### Proje Kuralları
- **Çalışma Branch**: Sadece `main` branch'i kullanılır
- **Commit Mesajları**: Türkçe, `tür(kapsam): açıklama` formatında
- **TODO İşaretleme**: %100 emin olunmadan [x] yapılmaz
- **İletişim Dili**: Tüm cevaplar Türkçe

### Yol Haritası Durumu

#### 🔴 Faz 1: Çekirdek Kurulum (Şu Anda)
- [ ] `apps/backend` klasör yapısı
- [ ] Python sanal ortam (`venv`)
- [ ] `BaseNode` sınıfı
- [ ] **Watermark Node** entegrasyonu

#### 🟠 Faz 2: Yapay Zeka Modülleri
- [ ] **DIRE Node** entegrasyonu
- [ ] **CLIP Node** implementasyonu
- [ ] **Frekans Analizi** modülü

#### 🟡 Faz 3: Test ve Doğrulama
- [ ] Veri seti indirme (HuggingFace, Kaggle)
- [ ] `pytest` birim ve entegrasyon testleri
- [ ] GPU bellek yönetimi testleri

### Test Stratejisi
1. **Birim Testler**: Her node izole olarak test edilir
2. **Entegrasyon Testleri**: Görsel yükleme → Analiz → Skorlama akışı
3. **Validasyon**: Ayrılmış test seti üzerinde performans ölçümü

**Hedefler:**
- %95+ Doğruluk (Accuracy)
- <%2 Yanlış Pozitif (False Positive)

---

## 📁 Proje Yapısı

```
AI-human-detector/
├── .agent/
│   └── rules/
│       └── RULES.md              # Proje geliştirme kuralları
├── .claude/
│   └── settings.local.json       # Claude Code konfigürasyonu
├── apps/
│   ├── backend/                  # Python AI Motoru (Yapılandırma aşamasında)
│   │   ├── core/                 # Temel sınıflar (BaseNode)
│   │   ├── nodes/                # Detection node'ları
│   │   │   ├── watermark.py      # Watermark detection
│   │   │   ├── dire.py           # DIRE implementation
│   │   │   ├── clip.py           # CLIP-based detection
│   │   │   └── frequency.py      # FFT & ELA analysis
│   │   ├── api/                  # REST API (FastAPI/Flask)
│   │   ├── utils/                # Yardımcı fonksiyonlar
│   │   └── tests/                # Test dosyaları
│   └── frontend/                 # Web Arayüzü (Planlanıyor)
├── docs/
│   ├── INDEX.md                  # Bu dosya - Proje indeksi
│   ├── ARCHITECTURE.md           # Teknik mimari dokümantasyonu
│   ├── ROADMAP.md                # Yol haritası
│   └── TRAINING.md               # DIRE eğitim rehberi
├── .gitattributes
├── .gitignore
├── CHANGELOG.md                  # Sürüm geçmişi
├── CODE_OF_CONDUCT.md            # Davranış kuralları
├── CONTRIBUTING.md               # Katkı rehberi
├── LICENSE                       # MIT lisansı
├── README.md                     # Proje giriş
└── SECURITY.md                   # Güvenlik politikası
```

---

## 🔗 Kaynaklar ve Bağlantılar

### İç Bağlantılar
- [Proje README](../README.md)
- [Teknik Mimari](ARCHITECTURE.md)
- [Yol Haritası](ROADMAP.md)
- [Eğitim Rehberi](TRAINING.md)
- [Katkıda Bulunma](../CONTRIBUTING.md)

### Dış Kaynaklar
- **Stable Diffusion**: https://stability.ai/
- **Diffusers Library**: https://huggingface.co/docs/diffusers
- **OpenAI CLIP**: https://openai.com/research/clip
- **DIRE Paper**: Diffusion Reconstruction Error araştırması

---

## 📝 Notlar

### Aktif Geliştirme
- Proje şu anda **Faz 0 (Yapılandırma)** aşamasında
- Kodlama süreci başlamak üzere
- Backend iskeleti oluşturuluyor

### İletişim ve Destek
- **Issues**: GitHub Issues üzerinden
- **Discussions**: GitHub Discussions
- **Lisans**: MIT License - Açık kaynak kullanımı serbesttir

---

*Doküman Son Güncelleme: 28 Aralık 2025*
*Proje Durumu: Prototype - Faz 0*
