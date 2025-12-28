# DIRE Modeli Eğitim ve İyileştirme Rehberi

DIRE (Diffusion Reconstruction Error) yönteminin başarısı, hem "Yeniden Yapılandırma" (Reconstruction) aşamasındaki difüzyon modelinin kalitesine hem de "Hata Haritası" (Error Map) üzerinden karar veren sınıflandırıcının (Classifier) eğitimine bağlıdır. "Yetersiz" sonuçlar genellikle bu iki aşamadan birindeki uyumsuzluktan kaynaklanır.

Bu rehber, DIRE modelini **kendi veri setinizle** nasıl daha iyi eğitebileceğinizi adım adım anlatır.

## 1. Problemin Kaynağını Belirleme

Önce sorunun nerede olduğunu anlamak gerekir:
1.  **Reconstruction Sorunu:** Eğer DIRE, gerçek fotoğrafları bile çok kötü (yüksek hatayla) yeniden oluşturuyorsa, difüzyon modeli kullandığınız veri setine (örneğin insan yüzlerine) yeterince hakim değildir.
2.  **Detection Sorunu:** Hata haritaları (Error Maps) gözle bakıldığında gerçek ve sahte arasında fark gösteriyor ama model bunu ayırt edemiyorsa, sınıflandırıcı (ResNet) iyi eğitilmemiştir.

## 2. Strateji A: Sınıflandırıcıyı Eğitmek (Önerilen)

DIRE'nin orijinal makalesinde difüzyon modeli genellikle **dondurulur (frozen)**. Yani `Stable Diffusion v1.5` gibi bir model olduğu gibi kullanılır. Biz sadece onun ürettiği hataları tanıyan bir "Hakem" model eğitiriz.

### Adım 1: Veri Seti Hazırlığı
Kendi domaininize uygun güçlü bir veri seti oluşturun.
*   **Gerçek:** FFHQ, CelebA-HQ veya kendi topladığınız yüksek çözünürlüklü portreler.
*   **Sahte:** Midjourney v5, DALL-E 3, FLUX.1 ile üretilmiş görseller.
*   *Hedef:* En az 5.000 Gerçek + 5.000 Sahte görsel.

### Adım 2: Hata Haritalarını (DIRE Representations) Çıkarma
Tüm görselleri önceden eğitilmiş difüzyon modelinden geçirerek `DIRE` görüntülerini kaydedin.

```python
# Pseudo-code
for image in dataset:
    # 1. Fotoğrafı gürültüye çevir (DDIM Inversion)
    noise = diffusers_model.invert(image)
    # 2. Gürültüden tekrar fotoğraf oluştur (Reconstruction)
    recon = diffusers_model.generate(noise)
    # 3. Farkı al (Absolute Difference)
    dire_map = abs(image - recon)
    # 4. Kaydet
    save(dire_map, f"dire_dataset/{label}/{image_name}")
```

### Adım 3: ResNet50 Eğitimi
Artık elinizde orijinal fotoğraflar değil, onların "hata haritaları" var. Şimdi basit bir CNN (ResNet50) modelini bu haritalar üzerinde eğitin.

*   **Girdi:** DIRE Hata Haritası (3 kanal RGB).
*   **Çıktı:** 0 (Gerçek) veya 1 (Sahte).
*   **Eğitim:** Standart `Binary Cross Entropy` kaybı ile 20-50 epoch.

> **İpucu:** Bu yöntem en hızlı ve en etkili sonuç veren yöntemdir. Genellikle sorun difüzyon modelinde değil, son karar vericinin (ResNet) veriyi tanımamasındadır.

---

## 3. Strateji B: Difüzyon Modelini Fine-Tune Etmek (İleri Seviye)

Eğer Strateji A işe yaramazsa, difüzyon modeliniz gerçek fotoğrafları "yeterince iyi" yeniden oluşturamıyor demektir. Bu durumda modele "Bak, gerçek insan yüzü böyledir" diye öğretmeniz gerekir.

### Adım 1: DreamBooth veya LoRA Kullanımı
Stable Diffusion modelini, sadece **GERÇEK** insan yüzleri (FFHQ gibi) ile eğitin (Fine-tuning).
*   **Hedef:** Modelin gerçek fotoğrafları kusursuz yeniden oluşturmasını sağlamak (Reconstruction Error'u sıfıra yaklaştırmak).
*   **Dikkat:** Sahte fotoğraflarla EĞİTMEYİN. Modelin sadece gerçeği çok iyi bilmesini istiyoruz.

### Adım 2: DIRE Sürecini Tekrarlamak
Fine-tune edilmiş modeli kullanarak Strateji A'daki adımları tekrarlayın. Artık model, gerçek bir yüz gördüğünde onu mükemmel yeniden oluşturacak (Düşük Hata), ama sahte (yapay zeka) bir yüz gördüğünde zorlanacaktır (Yüksek Hata). Bu aradaki makas açıldığı için tespit başarısı artacaktır.

---

## 4. Pratik Yol Haritası

1.  **Mevcut Durum Analizi:** 50 gerçek, 50 sahte fotoğraf üzerinde DIRE hata haritalarını görselleştirin. Gözle fark görebiliyor musunuz?
    *   *Evet:* Sınıflandırıcı eğitimi (Strateji A) yapın.
    *   *Hayır:* Difüzyon modeli yetersiz (Strateji B).

2.  **Veri Seti Genişletme:** "Yetersiz" model genellikle "yetersiz veri" demektir. Daha zorlu (hard negative) örnekler toplayın.

3.  **Cross-Validation:** Eğittiğiniz modelin sadece eğitim setindeki Midjourney görsellerini ezberlemediğinden emin olmak için, hiç görmediği bir modelin (örn. DALL-E) çıktılarıyla test edin.
