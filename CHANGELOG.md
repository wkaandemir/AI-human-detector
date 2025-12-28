# Değişim Günlüğü (Changelog)

Bu dosya, projede yapılan tüm önemli değişiklikleri kronolojik olarak listeler.

## [Yayınlanmadı] (Unreleased)

### Eklendi
- **Monorepo Yapısı:** `apps/backend` ve `apps/frontend` ayrımı planlandı.
- **Dokümantasyon:**
    - `PRD.md`: Proje gereksinimleri ve vizyon.
    - `API_REFERENCE.md`: Endpoint tanımları ve veri modelleri.
    - `system_analysis.md`: Maliyet analizi ve teknik gereksinimler.
    - `TRAINING_GUIDE.md`: DIRE modeli eğitim rehberi.
- **Hibrit Analiz Mimarisi:** Watermark Check, DIRE, Frekans Analizi ve ELA modüllerini içeren analiz stratejisi belirlendi.
- **Açık Kaynak Stratejisi:** Google Vertex AI ve SynthID yerine tamamen açık kaynak kütüphaneler (imwatermark vb.) tercih edildi.

### Düzenlendi
- `README.md`: Proje yapısını yansıtacak şekilde tamamen yenilendi.
- `TODO.md`: Tüm geliştirme süreci adım adım planlandı.
