# 📋 AI Human Detector - TODO Listesi

Bu doküman, projenin tüm yapılacak işlerini LLM tarafından takip edilebilir ve kodlanabilir şekilde detaylandırır.

---

## 🗂️ İçindekiler

1. [Faz 1: Çekirdek Kurulum](#faz-1-çekirdek-kurulum)
2. [Faz 2: Yapay Zeka Modülleri](#faz-2-yapay-zeka-modülleri)
3. [Faz 3: Test ve Doğrulama](#faz-3-test-ve-doğrulama)
4. [Faz 4: API ve Entegrasyon](#faz-4-api-ve-entegrasyon)
5. [Faz 5: Frontend](#faz-5-frontend)

---

## 🔴 Faz 1: Çekirdek Kurulum

### 1.1 Proje Yapısı Oluşturma
- [x] `apps/backend/` dizin yapısını oluştur
  ```bash
  apps/backend/
  ├── core/
  ├── nodes/
  ├── api/
  ├── utils/
  └── tests/
  ```
- [x] `__init__.py` dosyalarını her pakete ekle
- [ ] `.gitkeep` dosyalarını boş dizinlere ekle

### 1.2 Sanal Ortam ve Bağımlılıklar
- [ ] Python sanal ortam oluştur (`python -m venv venv`)
- [x] `requirements.txt` dosyasını oluştur
  ```txt
  torch>=2.0.0
  torchvision>=0.15.0
  diffusers>=0.20.0
  transformers>=4.30.0
  accelerate>=0.20.0
  safetensors>=0.3.0
  pillow>=10.0.0
  numpy>=1.24.0
  opencv-python>=4.8.0
  imwatermark>=0.3.0
  exifread>=3.0.0
  fastapi>=0.100.0
  uvicorn>=0.23.0
  python-multipart>=0.0.6
  pydantic>=2.0.0
  pytest>=7.4.0
  pytest-asyncio>=0.21.0
  ```
- [ ] `pip install -r requirements.txt` komutunu çalıştır

### 1.3 Temel Sınıflar
- [x] `core/base_node.py` - BaseNode sınıfını oluştur
  ```python
  class BaseNode(ABC):
      def __init__(self, weight: float = 1.0)
      @abstractmethod
      def analyze(self, image: np.ndarray) -> NodeResult
  ```
- [x] `core/models.py` - Veri modellerini oluştur
  ```python
  @dataclass
  class NodeResult:
      score: float
      verdict: str
      metadata: dict
      confidence: float
  ```
- [x] `core/ensemble.py` - Ensemble motorunu oluştur
  ```python
  class EnsembleEngine:
      def __init__(self, nodes: List[BaseNode])
      def analyze(self, image: np.ndarray) -> EnsembleResult
      def _aggregate_scores(self, results: List[NodeResult]) -> float
  ```

### 1.4 Watermark Node (İlk Node)
- [x] `nodes/watermark.py` dosyasını oluştur
- [x] `imwatermark` entegrasyonunu yap
- [x] Metadata (EXIF) okuma fonksiyonunu ekle
- [x] Birim testlerini yaz (`tests/test_watermark.py`)

---

## 🟠 Faz 2: Yapay Zeka Modülleri

### 2.1 DIRE Node (Diffusion Reconstruction Error)

#### 2.1.1 Kurulum
- [x] HuggingFace token'ı yapılandır (opsiyonel)
- [x] Stable Diffusion v1.5 modelini indir (lazy loading)
- [x] Model disk cache mekanizmasını implement et (HuggingFace cache)

#### 2.1.2 DIRE Algoritması
- [x] `nodes/dire.py` dosyasını oluştur
- [x] DDIM Inversion fonksiyonunu implement et
  ```python
  def ddim_invert(model, image, num_steps=50)
  ```
- [x] Reconstruction fonksiyonunu implement et
  ```python
  def reconstruct(model, noise_latents, num_steps=50)
  ```
- [x] Error Map hesaplama fonksiyonunu implement et
  ```python
  def compute_error_map(original, reconstructed)
  ```

#### 2.1.3 Classifier
- [ ] ResNet50 classifier modelini yükle (opsiyonel - ileride eklenecek)
- [x] Error Map → Fake/Real classification implement et (basit threshold)
- [x] GPU bellek yönetimini ekle (batch processing, cleanup)

#### 2.1.4 Testler
- [x] `tests/test_dire.py` oluştur
- [ ] 50 gerçek + 50 sahte görsel ile manuel test
- [x] Error Map görselleştirme fonksiyonu ekle

### 2.2 CLIP Node (Semantic Anomaly Detection)

#### 2.2.1 CLIP Model Entegrasyonu
- [x] `nodes/clip.py` dosyasını oluştur
- [x] OpenAI CLIP modelini yükle (ViT-B/32 veya ViT-L/14)
- [x] Görsel embedding fonksiyonunu implement et
  ```python
  def get_clip_embedding(image: np.ndarray) -> np.ndarray
  ```

#### 2.2.2 Anomaly Detection
- [ ] Linear Probe Classifier'ı implement et (kalibrasyon ile)
- [x] Zero-shot anomaly scoring mekanizması ekle
- [x] Embedding distance hesaplama fonksiyonları

#### 2.2.3 Testler
- [x] `tests/test_clip.py` oluştur
- [x] Embedding benzerlik testleri

### 2.3 Frekans & ELA Node (Low-Level Analysis)

#### 2.3.1 FFT Analizi
- [x] `nodes/frequency.py` dosyasını oluştur
- [x] 2D FFT fonksiyonunu implement et
  ```python
  def compute_fft(image: np.ndarray) -> np.ndarray
  ```
- [x] Frequency spectrum analiz fonksiyonu
- [x] Checkerboard artifact detection algoritması

#### 2.3.2 ELA (Error Level Analysis)
- [x] JPEG compression fonksiyonunu implement et
  ```python
  def compress_jpeg(image: np.ndarray, quality: int) -> np.ndarray
  ```
- [x] ELA haritası hesaplama fonksiyonu
  ```python
  def compute_ela_map(original: np.ndarray, compressed: np.ndarray) -> np.ndarray
  ```

#### 2.3.3 Testler
- [x] `tests/test_frequency.py` oluştur
- [x] Frekans spektrumu görselleştirme

---

## 🟡 Faz 3: Test ve Doğrulama

### 3.1 Veri Seti Hazırlığı
- [ ] HuggingFace datasets araştırması
  - [ ] FFHQ (Real faces)
  - [ ] CelebA-HQ (Real faces)
  - [ ] Midjourney-generated faces
  - [ ] DALL-E 3 generated faces
- [ ] Veri indirme scripti oluştur (`scripts/download_dataset.py`)
- [ ] Veri setini train/val/test olarak böl (80/10/10)
- [ ] Data augmentation pipeline'ı kur

### 3.2 Birim Testler
- [x] Her node için ayrı test dosyaları
- [x] Mock görüntüler ile test senaryoları
- [x] Edge case'leri test et
  - Boş görüntü
  - Çok düşük/çok yüksek çözünürlük
  - Farklı formatlar (PNG, JPG, WEBP)

### 3.3 Entegrasyon Testleri
- [x] `tests/test_integration.py` oluştur
- [x] End-to-end analiz akışı testi
- [x] API endpoint testleri
- [x] GPU bellek yönetimi testleri

### 3.4 Performans Testleri
- [ ] İşleme hızı benchmark'ı (saniye başına görüntü)
- [ ] GPU bellek kullanım ölçümü
- [ ] Batch processing optimizasyonu
- [ ] CPU fallback mekanizması

### 3.5 Model Doğrulama
- [ ] Accuracy, Precision, Recall, F1 hesapla
- [ ] ROC curve oluştur
- [ ] Confusion matrix oluştur
- [ ] False positive analizı
- [ ] Cross-validation (5-fold)

**Hedefler:**
- Accuracy: %95+
- False Positive Rate: <%2

---

## 🔵 Faz 4: API ve Entegrasyon

### 4.1 FastAPI Uygulaması
- [x] `api/main.py` - FastAPI uygulamasını oluştur
- [x] `api/models.py` - Pydantic modellerini oluştur
  ```python
  class AnalyzeRequest(BaseModel):
      check_metadata: bool = True
      return_details: bool = True
  ```
- [x] `api/endpoints.py` - API endpoint'lerini implement et
  - `POST /api/v1/analyze`
  - `GET /health`
  - `GET /models`

### 4.2 Middleware ve Hata Yönetimi
- [x] CORS middleware
- [x] Exception handler'lar
- [x] Rate limiting (opsiyonel)
- [x] Request logging
- [x] Security headers middleware

### 4.3 Dokümantasyon
- [x] OpenAPI (Swagger) dokümantasyonu
- [x] API endpoint açıklamaları
- [x] Response/Request örnekleri

### 4.4 Konfigürasyon
- [x] `config.py` - Yapılandırma dosyası
  - Model yolları
  - GPU ayarları
  - Port numarası
  - Debug modu

### 4.5 Testler
- [x] `tests/test_api.py` - API birim ve entegrasyon testleri

---

## 🟣 Faz 5: Frontend (Opsiyonel)

### 5.1 Teknoloji Seçimi
- [ ] Framework araştırması (React/Next.js vs Svelte)
- [ ] UI component library seçimi (shadcn/ui vs Tailwind)

### 5.2 Temel Arayüz
- [ ] Görsel yükleme bileşeni
- [ ] Analiz butonu ve progress bar
- [ ] Sonuç gösterme paneli
  - Final score
  - Detaylı node skorları
  - Error Map görselleştirme

### 5.3 İleri Özellikler
- [ ] Batch processing
- [ ] Sonuç karşılaştırma
- [ ] İndirme butonu
- [ ] Geçmiş (history) paneli

---

## 🔧 Ek Görevler

### Dokümantasyon
- [ ] API dokümantasyonunu güncelle
- [ ] Kullanım örnekleri ekle
- [ ] Video demo hazırla
- [ ] Tutorial yaz

### CI/CD
- [ ] GitHub Actions workflow oluştur
  - Linting (black, flake8)
  - Unit tests
  - Build check
- [ ] Automated release pipeline

### Optimizasyon
- [ ] Model quantization (opsiyonel)
- [ ] ONNX export (opsiyonel)
- [ ] Model caching mekanizması
- [ ] Async processing

---

## 📊 İlerleme Takibi

| Faz | Durum | Tamamlanma |
|-----|-------|-----------|
| Faz 1: Çekirdek Kurulum | 🟢 Tamamlandı | 100% |
| Faz 2: AI Modülleri | 🟢 Tamamlandı | 100% |
| Faz 3: Test ve Doğrulama | 🟡 Devam Ediyor | 40% |
| Faz 4: API ve Entegrasyon | 🟢 Tamamlandı | 100% |
| Faz 5: Frontend | 🔵 Planlanıyor | 0% |

---

## 📝 Notlar

### Öncelik Sırası
1. **✅ Faz 1 tamamlandı** - Temel yapı kuruldu (BaseNode, WatermarkNode)
2. **✅ Faz 2 tamamlandı** - 4 node tamamlandı (Watermark ✅, DIRE ✅, CLIP ✅, Frequency ✅)
3. **🔄 Faz 3 planlanıyor** - Veri seti ve model doğrulama
4. **✅ Faz 4 tamamlandı** - FastAPI uygulaması ve endpoint'ler
5. **Faz 5 planlanıyor** - Frontend geliştirme

### Dikkat Edilmesi Gerekenler
- GPU bellek yönetimi çok önemli
- Her node bağımsız çalışabilmeli
- Hata yönetimi kapsamlı olmalı
- Dokümantasyon kodla birlikte güncellenmeli

---

*Son Güncelleme: 28 Aralık 2025*
*Proje Durumu: Faz 1, 2, 4 Tamamlandı - MVP Ready!*
