#!/usr/bin/env python3
"""
AI Human Detector - AI Görsellerini Yeniden İsimlendirme Scripti

data/datasets/fake/ klasöründeki AI görsellerini
ai_000001.jpg, ai_000002.jpg formatında yeniden isimlendirir.

Kullanım:
    python scripts/rename_fake_images.py --input ./data/datasets/fake
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List
import logging

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def rename_images(
    input_dir: Path,
    prefix: str = "ai",
    start_index: int = 1
) -> int:
    """
    Dizindeki görselleri sıralı isimlendirir.

    Args:
        input_dir: Görsellerin bulunduğu dizin
        prefix: Dosya adı öneki (varsayılan: "ai")
        start_index: Başlangıç indeksi (varsayılan: 1)

    Returns:
        Yeniden isimlendirilen dosya sayısı
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        logger.error(f"❌ Dizin bulunamadı: {input_dir}")
        return 0

    # Desteklenen uzantılar
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    # Tüm görselleri bul
    image_files: List[Path] = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    # Sırala (dosya ismine göre)
    image_files.sort()

    if not image_files:
        logger.warning(f"⚠️ Görsel bulunamadı: {input_dir}")
        return 0

    logger.info(f"📁 {len(image_files)} görsel bulundu")
    logger.info(f"🔄 Yeniden isimlendirme başlıyor...")

    # Geçici bir dizine taşı (isim çakışmasını önlemek için)
    temp_dir = input_dir / "temp_rename"
    temp_dir.mkdir(exist_ok=True)

    renamed_count = 0

    # Önce tüm dosyaları temp dizinine taşı
    for idx, old_path in enumerate(image_files):
        try:
            # Yeni dosya adı
            new_name = f"{prefix}_{idx + start_index:06d}{old_path.suffix}"
            temp_path = temp_dir / new_name

            # Temp dizinine taşı
            old_path.rename(temp_path)
            renamed_count += 1

            if (idx + 1) % 100 == 0:
                logger.info(f"  İlerleme: {idx + 1}/{len(image_files)}")

        except Exception as e:
            logger.warning(f"⚠️ Dosya taşınamadı {old_path.name}: {e}")
            continue

    # Şimdi temp dizininden ana dizine taşı
    temp_files = sorted(temp_dir.glob("*"))
    for temp_path in temp_files:
        try:
            final_path = input_dir / temp_path.name
            temp_path.rename(final_path)

        except Exception as e:
            logger.warning(f"⚠️ Dosya taşınamadı {temp_path.name}: {e}")
            continue

    # Temp dizinini sil
    try:
        temp_dir.rmdir()
    except:
        pass

    logger.info(f"✅ {renamed_count} görsel yeniden isimlendirildi")
    return renamed_count


def main():
    parser = argparse.ArgumentParser(
        description="AI Human Detector - AI Görsellerini Yeniden İsimlendirme Scripti"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/datasets/fake",
        help="Görsellerin bulunduğu dizin"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="ai",
        help="Dosya adı öneki (varsayılan: ai)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Başlangıç indeksi (varsayılan: 1)"
    )

    args = parser.parse_args()

    try:
        input_dir = Path(args.input)

        # Yedekleme uyarısı
        logger.info("⚠️  Bu işlem dosya isimlerini kalıcı olarak değiştirecek!")
        logger.info(f"   Dizin: {input_dir}")
        logger.info("   Devam etmek için ENTER'a bas...")

        # İsimlendirme işlemi
        count = rename_images(
            input_dir=input_dir,
            prefix=args.prefix,
            start_index=args.start_index
        )

        if count > 0:
            logger.info("🎉 Yeniden isimlendirme tamamlandı!")

            # Özet bilgi
            extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            total_images = 0
            for ext in extensions:
                total_images += len(list(input_dir.glob(f"*{ext}")))
                total_images += len(list(input_dir.glob(f"*{ext.upper()}")))

            logger.info(f"📊 Toplam {total_images} görsel işlendi")

    except Exception as e:
        logger.error(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
