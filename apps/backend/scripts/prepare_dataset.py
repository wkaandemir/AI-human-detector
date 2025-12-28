#!/usr/bin/env python3
"""
AI Human Detector - Veri Seti Hazırlama Scripti

Bu script, indirilen verisetlerini train/val/test olarak böler ve
data augmentation uygular.

Kullanım:
    python scripts/prepare_dataset.py --input ./data/datasets --output ./data/processed
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Tuple, List
import logging
import random

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
        import PIL
        import numpy as np
        logger.info("✅ Tüm bağımlılıklar kurulu")
        return True
    except ImportError as e:
        logger.error(f"❌ Eksik bağımlılık: {e}")
        logger.info("Kurulum için: pip install pillow numpy")
        return False


def split_dataset(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Path, Path, Path]:
    """
    Verisetini train/val/test olarak böler.

    Args:
        input_dir: Girdi dizini (real/fake klasörleri)
        output_dir: Çıktı dizini
        train_ratio: Eğitim oranı (varsayılan: 0.8)
        val_ratio: Doğrulama oranı (varsayılan: 0.1)
        test_ratio: Test oranı (varsayılan: 0.1)
        seed: Rastgelelik tohumu

    Returns:
        (train_dir, val_dir, test_dir) demeti
    """
    import numpy as np
    from PIL import Image

    # Oranların toplamı 1 olmalı
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Oranların toplamı 1.0 olmalı"

    logger.info("📊 Veriseti bölünüyor...")

    # Çıktı dizinlerini oluştur
    splits = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio
    }

    split_dirs = {}
    for split_name in splits.keys():
        split_dirs[split_name] = {
            "real": output_dir / split_name / "real",
            "fake": output_dir / split_name / "fake"
        }
        for label_dir in split_dirs[split_name].values():
            label_dir.mkdir(parents=True, exist_ok=True)

    # Her etiket için (real/fake)
    for label in ["real", "fake"]:
        label_input_dir = input_dir / label

        if not label_input_dir.exists():
            logger.warning(f"⚠️ Dizin bulunamadı: {label_input_dir}")
            continue

        # Görselleri bul
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
            image_files.extend(label_input_dir.glob(ext))

        if not image_files:
            logger.warning(f"⚠️ Görsel bulunamadı: {label_input_dir}")
            continue

        logger.info(f"📁 {label.upper()}: {len(image_files)} görsel bulundu")

        # Karıştır
        random.seed(seed)
        np.random.seed(seed)
        random.shuffle(image_files)

        # Böl
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]

        # Kopyala
        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            output_split_dir = split_dirs[split_name][label]
            for img_file in files:
                # Görseli oku ve doğrula
                try:
                    img = Image.open(img_file)
                    img.verify()

                    # Yeniden aç (verify kapıyor)
                    img = Image.open(img_file)

                    # RGB'ye çevir (gerekirse)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Kaydet
                    output_path = output_split_dir / img_file.name
                    img.save(output_path, quality=95)

                except Exception as e:
                    logger.warning(f"⚠️ Görsel atlandı {img_file.name}: {e}")
                    continue

        logger.info(f"✅ {label.upper()} -> Train: {len(train_files)}, "
                   f"Val: {len(val_files)}, Test: {len(test_files)}")

    return (
        split_dirs["train"]["real"],
        split_dirs["val"]["real"],
        split_dirs["test"]["real"]
    )


def create_augmentation_pipeline():
    """
    Data augmentation pipeline'ı oluşturur.

    Returns:
        Augmentation fonksiyonu
    """
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    def augment_image(
        image: Image.Image,
        rotate_range: Tuple[int, int] = (-10, 10),
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1),
        saturation_range: Tuple[float, float] = (0.9, 1.1),
        gaussian_blur: bool = False,
        p_blur: float = 0.3
    ) -> Image.Image:
        """
        Görseli rastgele augment eder.

        Args:
            image: PIL Image
            rotate_range: Döndürme açısı aralığı
            brightness_range: Parlaklık aralığı
            contrast_range: Kontrast aralığı
            saturation_range: Doygunluk aralığı
            gaussian_blur: Gaussian blur uygula
            p_blur: Blur olasılığı

        Returns:
            Augment edilmiş görsel
        """
        img = image.copy()

        # Rastgele döndürme
        if rotate_range:
            angle = random.uniform(*rotate_range)
            img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))

        # Parlaklık
        if brightness_range:
            factor = random.uniform(*brightness_range)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)

        # Kontrast
        if contrast_range:
            factor = random.uniform(*contrast_range)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)

        # Doygunluk
        if saturation_range:
            factor = random.uniform(*saturation_range)
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(factor)

        # Gaussian blur (opsiyonel)
        if gaussian_blur and random.random() < p_blur:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        return img

    return augment_image


def apply_augmentation(
    input_dir: Path,
    output_dir: Path,
    augment_factor: int = 3
) -> Path:
    """
    Train setine data augmentation uygular.

    Args:
        input_dir: Train set dizini
        output_dir: Artırılmış veri çıktı dizini
        augment_factor: Her görsel için kaç augmentation uygulanacağı

    Returns:
        Çıktı dizini
    """
    from PIL import Image

    logger.info(f"🎨 Data augmentation uygulanıyor (x{augment_factor})...")

    # Pipeline oluştur
    augment_fn = create_augmentation_pipeline()

    # Çıktı dizinini oluştur
    output_dir.mkdir(parents=True, exist_ok=True)

    # Görselleri bul
    image_files = []
    for label in ["real", "fake"]:
        label_dir = input_dir / label
        if label_dir.exists():
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                image_files.extend(label_dir.glob(ext))

    logger.info(f"📁 {len(image_files)} görsel işlenecek")

    # Augmentation uygula
    processed = 0
    for img_path in image_files:
        try:
            img = Image.open(img_path)

            # Orijinali kopyala
            label_dir = output_dir / img_path.parent.name
            label_dir.mkdir(parents=True, exist_ok=True)

            # Orijinali kaydet
            img.save(label_dir / img_path.name, quality=95)

            # Augment edilmiş versiyonları oluştur
            for i in range(augment_factor):
                aug_img = augment_fn(img)
                aug_name = f"{img_path.stem}_aug_{i}{img_path.suffix}"
                aug_img.save(label_dir / aug_name, quality=95)

            processed += 1

            if processed % 10 == 0:
                logger.info(f"  İlerleme: {processed}/{len(image_files)}")

        except Exception as e:
            logger.warning(f"⚠️ Görsel işlenemedi {img_path.name}: {e}")
            continue

    logger.info(f"✅ Augmentation tamamlandı: {processed} görsel")
    return output_dir


def create_dataset_info(
    input_dir: Path,
    output_path: Path
) -> None:
    """
    Veriseti hakkında bilgi dosyası oluşturur.

    Args:
        input_dir: Veriseti dizini
        output_path: Çıktı dosyası
    """
    from PIL import Image

    info = {
        "real": {"train": 0, "val": 0, "test": 0},
        "fake": {"train": 0, "val": 0, "test": 0}
    }

    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            label_dir = input_dir / split / label
            if label_dir.exists():
                count = len(list(label_dir.glob("*.png")) +
                          list(label_dir.glob("*.jpg")) +
                          list(label_dir.glob("*.jpeg")))
                info[label][split] = count

    # Dosyaya yaz
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# AI Human Detector - Veri Seti Bilgisi\n\n")
        f.write("## Bölüm Özeti\n\n")
        f.write("| Etiket | Train | Val | Test | Toplam |\n")
        f.write("|--------|-------|-----|------|-------|\n")

        for label in ["real", "fake"]:
            total = sum(info[label].values())
            f.write(f"| {label.upper()} | {info[label]['train']} | "
                   f"{info[label]['val']} | {info[label]['test']} | {total} |\n")

        f.write("\n## Detaylı Bilgi\n\n")
        for split in ["train", "val", "test"]:
            f.write(f"### {split.upper()}\n\n")
            real_count = info["real"][split]
            fake_count = info["fake"][split]
            total = real_count + fake_count
            f.write(f"- Real: {real_count}\n")
            f.write(f"- Fake: {fake_count}\n")
            f.write(f"- Toplam: {total}\n\n")

    logger.info(f"✅ Bilgi dosyası oluşturuldu: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Human Detector - Veri Seti Hazırlama Scripti"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/datasets",
        help="Girdi dizini (indirilen veriseti)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/processed",
        help="Çıktı dizini"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Eğitim oranı (varsayılan: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Doğrulama oranı (varsayılan: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test oranı (varsayılan: 0.1)"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Data augmentation uygula"
    )
    parser.add_argument(
        "--augment-factor",
        type=int,
        default=3,
        help="Augmentation çarpanı (varsayılan: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Rastgelelik tohumu"
    )

    args = parser.parse_args()

    # Bağımlılıkları kontrol et
    if not check_dependencies():
        sys.exit(1)

    try:
        input_dir = Path(args.input)
        output_dir = Path(args.output)

        # 1. Train/Val/Test böl
        logger.info("🔄 ADIM 1: Veriseti bölünüyor...")
        split_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )

        # 2. Data augmentation (opsiyonel)
        if args.augment:
            logger.info("🔄 ADIM 2: Data augmentation uygulanıyor...")
            train_dir = output_dir / "train"
            aug_dir = output_dir / "train_augmented"

            # Real görselleri augment et
            for label in ["real", "fake"]:
                label_input = train_dir / label
                if label_input.exists():
                    apply_augmentation(
                        input_dir=label_input,
                        output_dir=aug_dir / label,
                        augment_factor=args.augment_factor
                    )

        # 3. Bilgi dosyası oluştur
        logger.info("🔄 ADIM 3: Bilgi dosyası oluşturuluyor...")
        create_dataset_info(
            input_dir=output_dir,
            output_path=output_dir / "DATASET_INFO.md"
        )

        logger.info("🎉 Veriseti hazırlama tamamlandı!")

    except Exception as e:
        logger.error(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
