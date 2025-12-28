#!/usr/bin/env python3
"""
AI Human Detector - Gerçek İnsan Fotoğrafları İndirme Scripti

Unsplash API'den gerçek insan fotoğrafları indirir.
Kullanım:
    python scripts/download_real_photos.py --num-samples 1000 --output ./data/datasets
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from typing import Optional
import logging
import time

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """Gerekli kütüphaneleri kontrol eder."""
    try:
        from PIL import Image
        from io import BytesIO
        logger.info("✅ Tüm bağımlılıklar kurulu")
        return True
    except ImportError as e:
        logger.error(f"❌ Eksik bağımlılık: {e}")
        logger.info("Kurulum için: pip install pillow requests")
        return False


def download_pexels_people(
    output_dir: Path,
    api_key: str,
    num_samples: Optional[int] = None,
    per_page: int = 80
) -> Path:
    """
    Pexels'ten gerçek insan fotoğrafları indirir.

    Args:
        output_dir: Çıktı dizini
        api_key: Pexels API key
        num_samples: İndirilecek örnek sayısı (varsayılan: 1000)
        per_page: Her sayfadaki görsel sayısı (max 80)

    Returns:
        İndirilen verisetinin dizini
    """
    try:
        from PIL import Image
        from io import BytesIO

        if num_samples is None:
            num_samples = 1000

        logger.info(f"📥 Pexels'ten {num_samples} gerçek insan fotoğrafı indiriliyor...")

        # Çıktı dizinini oluştur
        real_dir = output_dir / "real" / "pexels_people"
        real_dir.mkdir(parents=True, exist_ok=True)

        # Pexels API endpoint
        base_url = "https://api.pexels.com/v1/search"

        # İnsan odaklı arama terimleri
        search_queries = [
            "person",
            "people",
            "human",
            "portrait",
            "man",
            "woman",
            "children",
            "friends",
            "family",
            "couple"
        ]

        count = 0
        page = 1

        # Her query için döngü
        for query_idx, query in enumerate(search_queries):
            if count >= num_samples:
                break

            logger.info(f"  Arama terimi: '{query}' ({query_idx + 1}/{len(search_queries)})")

            while count < num_samples:
                try:
                    # API isteği
                    params = {
                        "query": query,
                        "per_page": per_page,
                        "page": page,
                        "orientation": "all"  # landscape, portrait, square
                    }

                    headers = {
                        "Authorization": api_key
                    }

                    response = requests.get(
                        base_url,
                        params=params,
                        headers=headers,
                        timeout=30
                    )

                    if response.status_code != 200:
                        logger.warning(f"⚠️ API hatası: {response.status_code}")
                        break

                    data = response.json()

                    if not data.get("photos"):
                        logger.info(f"  '{query}' için daha fazla sonuç yok")
                        break

                    # Görselleri indir
                    for photo in data["photos"]:
                        if count >= num_samples:
                            break

                        try:
                            # Görsel URL'sini al (original veya large)
                            image_url = photo["src"]["large"]  # 1920px genişlik

                            # Görseli indir
                            img_response = requests.get(image_url, timeout=30)
                            if img_response.status_code != 200:
                                continue

                            # PIL Image olarak aç
                            img = Image.open(BytesIO(img_response.content))

                            # RGB'ye çevir (gerekirse)
                            if img.mode != 'RGB':
                                img = img.convert('RGB')

                            # Kaydet
                            image_path = real_dir / f"pexels_{count:06d}.jpg"
                            img.save(image_path, quality=95)
                            count += 1

                            # İlerleme
                            if count % 50 == 0:
                                logger.info(f"  İndirilen: {count}/{num_samples}")

                        except Exception as e:
                            logger.warning(f"⚠️ Görsel indirilemedi: {e}")
                            continue

                    page += 1

                    # Rate limiting - Pexels: 200 requests/hour
                    time.sleep(1)  # 1 saniye bekle

                    # Eğer bu query için sonuçlar bittiyse
                    if len(data["photos"]) < per_page:
                        logger.info(f"  '{query}' için tüm sonuçlar indirildi")
                        break

                except Exception as e:
                    logger.error(f"❌ İndirme hatası: {e}")
                    time.sleep(5)
                    continue

            # Sonraki query için page'i sıfırla
            page = 1

            # Rate limiting için bekle
            time.sleep(1)

        logger.info(f"✅ {count} gerçek insan fotoğrafı indirildi -> {real_dir}")
        return real_dir

    except Exception as e:
        logger.error(f"❌ Unsplash indirme hatası: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="AI Human Detector - Gerçek İnsan Fotoğrafları İndirme Scripti"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/datasets",
        help="Çıktı dizini"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Pexels API Key"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="İndirilecek örnek sayısı (varsayılan: 1000)"
    )

    args = parser.parse_args()

    # Bağımlılıkları kontrol et
    if not check_dependencies():
        sys.exit(1)

    try:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Pexels'ten gerçek insan fotoğrafları indir
        download_pexels_people(
            output_dir=output_dir,
            api_key=args.api_key,
            num_samples=args.num_samples
        )

        logger.info("🎉 İndirme tamamlandı!")

        # Özet bilgi
        real_dir = output_dir / "real" / "pexels_people"
        if real_dir.exists():
            num_images = len(list(real_dir.glob("*.jpg")))
            logger.info(f"📊 Toplam {num_images} gerçek insan fotoğrafı indirildi")

    except Exception as e:
        logger.error(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
