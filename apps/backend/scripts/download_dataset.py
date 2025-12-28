#!/usr/bin/env python3
"""
AI Human Detector - Veri Seti İndirme Scripti

Bu script, HuggingFace'ten gerçek ve AI üretilmiş yüz verisetlerini indirir.
Kullanım:
    python scripts/download_dataset.py --dataset all --output ./data
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple
import logging

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """
    Gerekli kütüphanelerin kurulu olup olmadığını kontrol eder.
    """
    try:
        import datasets
        import PIL
        import tqdm
        logger.info("✅ Tüm bağımlılıklar kurulu")
        return True
    except ImportError as e:
        logger.error(f"❌ Eksik bağımlılık: {e}")
        logger.info("Kurulum için: pip install datasets pillow tqdm")
        return False


def download_celeba_hq(
    output_dir: Path,
    num_samples: Optional[int] = None,
    split: str = "train"
) -> Path:
    """
    CelebA-HQ verisetini HuggingFace'ten indirir.

    Args:
        output_dir: Çıktı dizini
        num_samples: İndirilecek örnek sayısı (None = tümü)
        split: Veri seti bölmesi

    Returns:
        İndirilen verisetinin dizini
    """
    try:
        from datasets import load_dataset
        from tqdm import tqdm

        logger.info("📥 CelebA-HQ veriseti indiriliyor...")

        # Verisetini yükle (streaming modu)
        dataset = load_dataset(
            "mattymchen/celeba-hq",
            split=split,
            streaming=True
        )

        # Çıktı dizinini oluştur
        celeba_dir = output_dir / "real" / "celeba_hq"
        celeba_dir.mkdir(parents=True, exist_ok=True)

        # Görselleri indir ve kaydet
        count = 0
        for example in tqdm(dataset, desc="CelebA-HQ indiriliyor"):
            if num_samples and count >= num_samples:
                break

            # Görseli kaydet
            image = example["image"]
            image_path = celeba_dir / f"celeba_{count:05d}.png"
            image.save(image_path)
            count += 1

        logger.info(f"✅ CelebA-HQ: {count} görsel indirildi -> {celeba_dir}")
        return celeba_dir

    except Exception as e:
        logger.error(f"❌ CelebA-HQ indirme hatası: {e}")
        raise


def download_coco_ai(
    output_dir: Path,
    num_samples: Optional[int] = None,
    split: str = "train"
) -> Path:
    """
    COCO_AI verisetini (AI üretilmiş görseller) HuggingFace'ten indirir.

    Args:
        output_dir: Çıktı dizini
        num_samples: İndirilecek örnek sayısı (None = tümü)
        split: Veri seti bölmesi

    Returns:
        İndirilen verisetinin dizini
    """
    try:
        from datasets import load_dataset
        from tqdm import tqdm

        logger.info("📥 COCO_AI veriseti indiriliyor...")

        # Verisetini yükle
        dataset = load_dataset(
            "NasrinImp/COCO_AI",
            split=split,
            streaming=True
        )

        # Çıktı dizinini oluştur
        coco_dir = output_dir / "fake" / "coco_ai"
        coco_dir.mkdir(parents=True, exist_ok=True)

        # Görselleri indir ve kaydet
        count = 0
        for example in tqdm(dataset, desc="COCO_AI indiriliyor"):
            if num_samples and count >= num_samples:
                break

            # Görseli kaydet (genellikle 'image' anahtarı)
            if "image" in example:
                image = example["image"]
            elif "jpg" in example:
                image = example["jpg"]
            else:
                logger.warning(f"⚠️ Beklenmeyen veri formatı: {example.keys()}")
                continue

            image_path = coco_dir / f"coco_ai_{count:05d}.png"
            image.save(image_path)
            count += 1

        logger.info(f"✅ COCO_AI: {count} görsel indirildi -> {coco_dir}")
        return coco_dir

    except Exception as e:
        logger.error(f"❌ COCO_AI indirme hatası: {e}")
        raise


def download_dataset(
    dataset_name: str,
    output_dir: Path,
    num_samples: Optional[int] = None
) -> None:
    """
    Belirtilen verisetini indirir.

    Args:
        dataset_name: Veriseti adı ('celeba_hq', 'coco_ai', veya 'all')
        output_dir: Çıktı dizini
        num_samples: İndirilecek örnek sayısı
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name in ["celeba_hq", "all"]:
        download_celeba_hq(output_dir, num_samples)

    if dataset_name in ["coco_ai", "all"]:
        download_coco_ai(output_dir, num_samples)


def create_sample_dataset(
    real_dir: Path,
    fake_dir: Path,
    output_dir: Path,
    num_real: int = 50,
    num_fake: int = 50
) -> Tuple[Path, Path]:
    """
    Manuel test için küçük bir örnek veriseti oluşturur.

    Args:
        real_dir: Gerçek görseller dizini
        fake_dir: Sahte görseller dizini
        output_dir: Çıktı dizini
        num_real: Örnek gerçek görsel sayısı
        num_fake: Örnek sahte görsel sayısı

    Returns:
        (real_sample_dir, fake_sample_dir)
    """
    import shutil
    import random

    logger.info(f"📦 Örnek veriseti oluşturuluyor: {num_real} gerçek, {num_fake} sahte")

    # Dizinleri oluştur
    sample_real = output_dir / "sample_real"
    sample_fake = output_dir / "sample_fake"
    sample_real.mkdir(parents=True, exist_ok=True)
    sample_fake.mkdir(parents=True, exist_ok=True)

    # Gerçek görsellerden örnek al
    real_images = list(real_dir.glob("*.png")) + list(real_dir.glob("*.jpg"))
    if len(real_images) >= num_real:
        selected_real = random.sample(real_images, num_real)
        for img in selected_real:
            shutil.copy2(img, sample_real / img.name)
        logger.info(f"✅ {num_real} gerçek görsel kopyalandı")
    else:
        logger.warning(f"⚠️ Yeterli gerçek görsel yok: {len(real_images)} < {num_real}")

    # Sahte görsellerden örnek al
    fake_images = list(fake_dir.glob("*.png")) + list(fake_dir.glob("*.jpg"))
    if len(fake_images) >= num_fake:
        selected_fake = random.sample(fake_images, num_fake)
        for img in selected_fake:
            shutil.copy2(img, sample_fake / img.name)
        logger.info(f"✅ {num_fake} sahte görsel kopyalandı")
    else:
        logger.warning(f"⚠️ Yeterli sahte görsel yok: {len(fake_images)} < {num_fake}")

    return sample_real, sample_fake


def main():
    parser = argparse.ArgumentParser(
        description="AI Human Detector - Veri Seti İndirme Scripti"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["celeba_hq", "coco_ai", "all"],
        default="all",
        help="İndirilecek veriseti"
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
        default=None,
        help="Her verisetinden indirilecek maksimum örnek sayısı"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Manuel test için 50+50 örnek veriseti oluştur"
    )

    args = parser.parse_args()

    # Bağımlılıkları kontrol et
    if not check_dependencies():
        sys.exit(1)

    try:
        # Verisetini indir
        download_dataset(
            dataset_name=args.dataset,
            output_dir=Path(args.output),
            num_samples=args.num_samples
        )

        # Örnek veriseti oluştur (isteğe bağlı)
        if args.create_sample:
            real_dir = Path(args.output) / "real" / "celeba_hq"
            fake_dir = Path(args.output) / "fake" / "coco_ai"

            if real_dir.exists() and fake_dir.exists():
                sample_output = Path(args.output) / "sample"
                create_sample_dataset(
                    real_dir=real_dir,
                    fake_dir=fake_dir,
                    output_dir=sample_output,
                    num_real=50,
                    num_fake=50
                )
            else:
                logger.warning("⚠️ Örnek veriseti oluşturulamadı: kaynak dizinler yok")

        logger.info("🎉 İndirme tamamlandı!")

    except Exception as e:
        logger.error(f"❌ Hata: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
