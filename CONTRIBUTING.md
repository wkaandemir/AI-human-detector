# Katkı Rehberi (Contributing Guide)

Projeye katkıda bulunmak istediğiniz için teşekkür ederiz! Aşağıdaki rehber, projenin tutarlılığını korumak ve işbirliğini kolaylaştırmak için hazırlanmıştır.

## 🤝 Katkıda Bulunma (Contributing)

Öncelikle, projemize katkıda bulunmak istediğiniz için teşekkürler! Açık kaynak topluluğunu harika yapan şey, sizin gibi insanların katkılarıdır.

Lütfen katkıda bulunmadan önce [Code of Conduct](CODE_OF_CONDUCT.md) ve [Security Policy](SECURITY.md)'yi okuyun.

## 🚀 Geliştirme Süreci (Development Workflow)

1.  **Fork**layın: Projeyi kendi hesabınıza fork'layın.
2.  **Clone**layın: Fork'ladığınız projeyi yerel makinenize indirin.
3.  **Branch** Oluşturun: Yeni bir özellik veya düzeltme için yeni bir dal (branch) açın.
    ```bash
    git checkout -b ozellik/yeni-ozellik
    ```
4.  **Kurulum**: `docs/INSTALL.md` dosyasındaki adımları izleyerek ortamı kurun.

## 📝 Commit Standartları (ÖNEMLİ)

Projede **Conventional Commits** yapısı ve **Türkçe** mesajlar kullanılmaktadır.

Format: `tür(kapsam): açıklama`

*   **tür:**
    *   `feat`: Yeni bir özellik.
    *   `fix`: Hata düzeltmesi.
    *   `docs`: Sadece dokümantasyon değişikliği.
    *   `style`: Kodun çalışmasını etkilemeyen format değişiklikleri (boşluk, noktalama vb.).
    *   `refactor`: Hata düzeltmeyen veya özellik eklemeyen kod düzenlemesi.
    *   `test`: Test ekleme veya düzeltme.
    *   `chore`: Derleme süreci veya yardımcı araçlarda yapılan değişiklikler.

*   **Örnekler:**
    *   `feat(auth): giriş ekranı eklendi`
    *   `fix(api): boş gelen veri hatası giderildi`
    *   `docs(readme): kurulum adımları güncellendi`

## 💻 Kodlama Standartları

*   **Python:** PEP 8 standartlarına uyulmalıdır. `black` veya `flake8` kullanılması önerilir.
*   **Dokümantasyon:** Eklenen her yeni fonksiyon veya sınıf için docstring yazılmalıdır.

## 🔄 Pull Request Süreci

1.  Kodunuzu gönderirken testlerin geçtiğinden emin olun.
2.  Pull Request (PR) açarken yaptığınız değişikliği detaylıca açıklayın.
3.  İlgili `Issue` numarasını belirtin (varsa).
