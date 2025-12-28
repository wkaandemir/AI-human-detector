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
- [ ] `apps/backend/` dizin yapısını oluştur
  ```bash
  apps/backend/
  ├── core/
  ├── nodes/
  ├── api/
  ├── utils/
  └── tests/
  ```
- [ ] `__init__.py` dosyalarını her pakete ekle
- [ ] `.gitkeep` dosyalarını boş dizinlere ekle

### 1.2 Sanal Ortam ve Bağımlılıklar
- [ ] Python sanal ortam oluştur (`python -m venv venv`)
- [ ] `requirements.txt` dosyasını oluştur
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
- [ ] `core/base_node.py` - BaseNode sınıfını oluştur
  ```python
  class BaseNode(ABC):
      def __init__(self, weight: float = 1.0)
      @abstractmethod
      def analyze(self, image: np.ndarray) -> NodeResult
  ```
- [ ] `core/models.py` - Veri modellerini oluştur
  ```python
  @dataclass
  class NodeResult:
      score: float
      verdict: str
      metadata: dict
      confidence: float
  ```
- [ ] `core/ensemble.py` - Ensemble motorunu oluştur
  ```python
  class EnsembleEngine:
      def __init__(self, nodes: List[BaseNode])
      def analyze(self, image: np.ndarray) -> EnsembleResult
      def _aggregate_scores(self, results: List[NodeResult]) -> float
  ```

### 1.4 Watermark Node (İlk Node)
- [ ] `nodes/watermark.py` dosyasını oluştur
- [ ] `imwatermark` entegrasyonunu yap
- [ ] Metadata (EXIF) okuma fonksiyonunu ekle
- [ ] Birim testlerini yaz (`tests/test_watermark.py`)

---

## 🟠 Faz 2: Yapay Zeka Modülleri

### 2.1 DIRE Node (Diffusion Reconstruction Error)

#### 2.1.1 Kurulum
- [ ] HuggingFace token'ı yapılandır
- [ ] Stable Diffusion v1.5 modelini indir
- [ ] Model disk cache mekanizmasını implement et

#### 2.1.2 DIRE Algoritması
- [ ] `nodes/dire.py` dosyasını oluştur
- [ ] DDIM Inversion fonksiyonunu implement et
  ```python
  def ddim_invert(model, image, num_steps=50)
  ```
- [ ] Reconstruction fonksiyonunu implement et
  ```python
  def reconstruct(model, noise_latents, num_steps=50)
  ```
- [ ] Error Map hesaplama fonksiyonunu implement et
  ```python
  def compute_error_map(original, reconstructed)
  ```

#### 2.1.3 Classifier
- [ ] ResNet50 classifier modelini yükle
- [ ] Error Map → Fake/Real classification implement et
- [ ] GPU bellek yönetimini ekle (batch processing)

#### 2.1.4 Testler
- [ ] `tests/test_dire.py` oluştur
- [ ] 50 gerçek + 50 sahte görsel ile manuel test
- [ ] Error Map görselleştirme fonksiyonu ekle

### 2.2 CLIP Node (Semantic Anomaly Detection)

#### 2.2.1 CLIP Model Entegrasyonu
- [ ] `nodes/clip.py` dosyasını oluştur
- [ ] OpenAI CLIP modelini yükle (ViT-B/32 veya ViT-L/14)
- [ ] Görsel embedding fonksiyonunu implement et
  ```python
  def get_clip_embedding(image: np.ndarray) -> np.ndarray
  ```

#### 2.2.2 Anomaly Detection
- [ ] Linear Probe Classifier'ı implement et
- [ ] Zero-shot anomaly scoring mekanizması ekle
- [ ] Embedding distance hesaplama fonksiyonları

#### 2.2.3 Testler
- [ ] `tests/test_clip.py` oluştur
- [ ] Embedding benzerlik testleri

### 2.3 Frekans & ELA Node (Low-Level Analysis)

#### 2.3.1 FFT Analizi
- [ ] `nodes/frequency.py` dosyasını oluştur
- [ ] 2D FFT fonksiyonunu implement et
  ```python
  def compute_fft(image: np.ndarray) -> np.ndarray
  ```
- [ ] Frequency spectrum analiz fonksiyonu
- [ ] Checkerboard artifact detection algoritması

#### 2.3.2 ELA (Error Level Analysis)
- [ ] JPEG compression fonksiyonunu implement et
  ```python
  def compress_jpeg(image: np.ndarray, quality: int) -> np.ndarray
  ```
- [ ] ELA haritası hesaplama fonksiyonu
  ```python
  def compute_ela_map(original: np.ndarray, compressed: np.ndarray) -> np.ndarray
  ```

#### 2.3.3 Testler
- [ ] `tests/test_frequency.py` oluştur
- [ ] Frekans spektrumu görselleştirme

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
- [ ] Her node için ayrı test dosyaları
- [ ] Mock görüntüler ile test senaryoları
- [ ] Edge case'leri test et
  - Boş görüntü
  - Çok düşük/çok yüksek çözünürlük
  - Farklı formatlar (PNG, JPG, WEBP)

### 3.3 Entegrasyon Testleri
- [ ] `tests/test_integration.py` oluştur
- [ ] End-to-end analiz akışı testi
- [ ] API endpoint testleri
- [ ] GPU bellek yönetimi testleri

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
- [ ] `api/main.py` - FastAPI uygulamasını oluştur
- [ ] `api/models.py` - Pydantic modellerini oluştur
  ```python
  class AnalyzeRequest(BaseModel):
      check_metadata: bool = True
      return_details: bool = True
  ```
- [ ] `api/endpoints.py` - API endpoint'lerini implement et
  - `POST /api/v1/analyze`
  - `GET /api/v1/health`
  - `GET /api/v1/models`

### 4.2 Middleware ve Hata Yönetimi
- [ ] CORS middleware
- [ ] Exception handler'lar
- [ ] Rate limiting (opsiyonel)
- [ ] Request logging

### 4.3 Dokümantasyon
- [ ] OpenAPI (Swagger) dokümantasyonu
- [ ] API endpoint açıklamaları
- [ ] Response/Request örnekleri

### 4.4 Konfigürasyon
- [ ] `config.py` - Yapılandırma dosyası
  - Model yolları
  - GPU ayarları
  - Port numarası
  - Debug modu

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
| Faz 1: Çekirdek Kurulum | 🔵 Başlanmadı | 0% |
| Faz 2: AI Modülleri | 🔵 Başlanmadı | 0% |
| Faz 3: Test ve Doğrulama | 🔵 Başlanmadı | 0% |
| Faz 4: API ve Entegrasyon | 🔵 Başlanmadı | 0% |
| Faz 5: Frontend | 🔵 Başlanmadı | 0% |

---

## 📝 Notlar

### Öncelik Sırası
1. **Önce Faz 1 tamamlanmalı** - Temel yapı olmadan diğerlere geçilmez
2. **Faz 2 sıralı olabilir** - Watermark → DIRE → CLIP → Frekans
3. **Testler her adımda** - Kod yazmadan önce test yaz (TDD)

### Dikkat Edilmesi Gerekenler
- GPU bellek yönetimi çok önemli
- Her node bağımsız çalışabilmeli
- Hata yönetimi kapsamlı olmalı
- Dokümantasyon kodla birlikte güncellenmeli

---

*Son Güncelleme: 28 Aralık 2025*
*Proje Durumu: Faz 0 - Yapılandırma*
