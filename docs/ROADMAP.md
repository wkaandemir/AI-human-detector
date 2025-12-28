# Project Roadmap & Status

## 🗺️ Neredeyiz? (Current Status)
Proje **Faz 0 (Yapılandırma)** aşamasındadır.
- ✅ Mimari tasarım tamamlandı (Hibrit Analiz).
- ✅ "Tamamen Açık Kaynak" stratejisine geçildi (Vertex AI/SynthID kaldırıldı).
- ✅ Dokümantasyon sadeleştirildi.
- 🚧 Kodlama başlıyor (`backend` iskeleti).

## 📅 Yapılacaklar Listesi (TODO)

### 🔴 Faz 1: Çekirdek Kurulum (Hemen Şimdi)
- [ ] `apps/backend` klasör yapısının oluşturulması.
- [ ] Python sanal ortamının (`venv`) kurulması.
- [ ] Temel sınıfların (`BaseNode`) kodlanması.
- [ ] İlk Node: **Watermark Node**'un entegrasyonu (`imwatermark` ile).

### 🟠 Faz 2: Yapay Zeka Modülleri
- [ ] **DIRE Node Entegrasyonu:** Diffusers kütüphanesi ile SD entegrasyonu.
- [ ] **CLIP Node:** UniversalFakeDetect modelinin port edilmesi.
- [ ] **Frekans Analizi:** FFT/DCT fonksiyonlarının yazılması.

### 🟡 Faz 3: Test ve Doğrulama
- [ ] **Veri Seti:** HuggingFace ve Kaggle'dan açık kaynak "Real vs Fake Face" veri setlerinin indirilmesi.
- [ ] **Otomasyon:** `pytest` ile birim ve entegrasyon testlerinin yazılması.
- [ ] **Performans:** 10MB+ görseller ve GPU bellek yönetimi testleri.

## 🧪 Test Stratejisi
Hedefimiz **%95+ Doğruluk (Accuracy)** ve **<%2 Yanlış Pozitif (False Positive)** oranına ulaşmaktır.

1.  **Birim Testler:** Her bir node (DIRE, ELA) izole olarak siyah/beyaz görsellerle test edilir.
2.  **Entegrasyon:** Görsel yükleme -> Analiz -> Skorlama akışı uçtan uca test edilir.
3.  **Validasyon:** Ayrılmış test veri seti üzerinde model performansı ölçülür.
