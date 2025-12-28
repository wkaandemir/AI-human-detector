#!/usr/bin/env python3
"""
AI Human Detector - COCO Dataset İndirme Scripti

COCO dataset'ten sadece insan (person) içeren görselleri indirir.
Kullanım:
    python scripts/download_coco_people.py --num-samples 1000 --output ./data/datasets
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import logging

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """Gerekli kütüphaneleri kontrol eder."""
    try:
        from datasets import load_dataset
        from PIL import Image
        from tqdm import tqdm
        logger.info("✅ Tüm bağımlılıklar kurulu")
        return True
    except ImportError as e:
        logger.error(f"❌ Eksik bağımlılık: {e}")
        logger.info("Kurulum için: pip install datasets pillow tqdm")
        return False


def download_coco_people(
    output_dir: Path,
    num_samples: Optional[int] = None,
    split: str = "train"
) -> Path:
    """
    Flickr30k dataset'ten insan odaklı görselleri indirir.

    Args:
        output_dir: Çıktı dizini
        num_samples: İndirilecek örnek sayısı (None = tümü)
        split: Veri seti bölmesi (train, validation, test)

    Returns:
        İndirilen verisetinin dizini
    """
    try:
        from datasets import load_dataset
        from tqdm import tqdm

        logger.info("📥 Flickr30k dataset indiriliyor (Human-focused real photos)...")

        # Flickr30k dataset'i yükle
        # İnsan odaklı gerçek fotoğraflar içerir
        dataset = load_dataset(
            "michelecafagna31/flickr30k",
            split=split,
            streaming=True
        )

        # Çıktı dizinini oluştur
        flickr_dir = output_dir / "real" / "flickr30k"
        flickr_dir.mkdir(parents=True, exist_ok=True)

        count = 0

        # Görselleri indir (Flickr30k zaten insan odaklı)
        for example in tqdm(dataset, desc=f"Flickr30k {split} indiriliyor", total=num_samples):
            if num_samples and count >= num_samples:
                break

            try:
                # Flickr30k dataset yapısı:
                # - image: PIL Image
                # - caption: str
                # - split: str
                # - img_path: str

                image = example.get("image")

                if image is None:
                    continue

                # Görseli kaydet
                image_path = flickr_dir / f"flickr30k_{count:06d}.jpg"
                image.save(image_path, quality=95)
                count += 1

                # Her 100 görselde bir log
                if count % 100 == 0:
                    logger.info(f"  İlerleme: {count} görsel indirildi")

            except Exception as e:
                logger.warning(f"⚠️ Görsel işlenemedi: {e}")
                continue

        logger.info(f"✅ Flickr30k: {count} gerçek insan görseli indirildi -> {flickr_dir}")
        return flickr_dir

    except Exception as e:
        logger.error(f"❌ COCO indirme hatası: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="AI Human Detector - Flickr30k Dataset İndirme Scripti"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/datasets",
        help="Çıktı dizini"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="İndirilecek örnek sayısı (varsayılan: 1000)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Veri seti bölmesi"
    )

    args = parser.parse_args()

    # Bağımlılıkları kontrol et
    if not check_dependencies():
        sys.exit(1)

    try:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Flickr30k dataset'ini indir
        download_coco_people(
            output_dir=output_dir,
            num_samples=args.num_samples,
            split=args.split
        )

        logger.info("🎉 İndirme tamamlandı!")

        # Özet bilgi
        real_dir = output_dir / "real" / "flickr30k"
        if real_dir.exists():
            num_images = len(list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png")))
            logger.info(f"📊 Toplam {num_images} gerçek insan görseli indirildi")

    except Exception as e:
        logger.error(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
