#!/usr/bin/env python3
"""
AI Human Detector - Cross-Validation Script'i

Bu script, modelin k-fold cross-validation ile değerlendirilmesini sağlar.
Modelin farklı veri bölümlerindeki performansını ölçer.

Kullanım:
    python scripts/cross_validate.py --data ./data/processed --k 5 --output ./results/cv
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
import logging
import random

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """Gerekli kütüphaneleri kontrol eder."""
    try:
        import PIL
        import numpy as np
        import sklearn
        logger.info("✅ Tüm bağımlılıklar kurulu")
        return True
    except ImportError as e:
        logger.error(f"❌ Eksik bağımlılık: {e}")
        logger.info("Kurulum için: pip install pillow numpy scikit-learn")
        return False


def load_all_images(data_dir: Path) -> Tuple[List[Tuple[str, np.ndarray, int]], Dict]:
    """
    Tüm gerçek ve sahte görselleri yükler.

    Returns:
        (dosya_adı, görsel, etiket) listesi ve bilgi sözlüğü
    """
    from PIL import Image
    import numpy as np

    logger.info("📊 Tüm veriler yükleniyor...")

    all_images = []
    info = {"real": 0, "fake": 0}

    # Real görseller (train + val + test)
    for split in ["train", "val", "test"]:
        real_dir = data_dir / split / "real"
        if real_dir.exists():
            for img_path in real_dir.glob("*.png"):
                try:
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_array = np.array(img)
                    all_images.append((img_path.name, img_array, 0))  # 0 = REAL
                    info["real"] += 1
                except Exception as e:
                    logger.warning(f"⚠️ Görsel atlandı {img_path.name}: {e}")

    # Fake görseller
    for split in ["train", "val", "test"]:
        fake_dir = data_dir / split / "fake"
        if fake_dir.exists():
            for img_path in fake_dir.glob("*.png"):
                try:
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_array = np.array(img)
                    all_images.append((img_path.name, img_array, 1))  # 1 = FAKE
                    info["fake"] += 1
                except Exception as e:
                    logger.warning(f"⚠️ Görsel atlandı {img_path.name}: {e}")

    logger.info(f"✅ Toplam: {info['real']} REAL, {info['fake']} FAKE")
    return all_images, info


def create_k_folds(
    images: List,
    k: int = 5,
    seed: int = 42
) -> List[Tuple[List, List]]:
    """
    K-fold cross-validation için veri bölümleri oluşturur.

    Args:
        images: (dosya_adı, görsel, etiket) listesi
        k: Fold sayısı
        seed: Rastgelelik tohumu

    Returns:
        (train_images, test_images) çiftlerinin listesi
    """
    import numpy as np

    logger.info(f"🔄 {k}-fold cross-validation için veri bölünüyor...")

    random.seed(seed)
    np.random.seed(seed)

    # Her etiket için ayrı ayrı böl (stratified)
    real_images = [img for img in images if img[2] == 0]
    fake_images = [img for img in images if img[2] == 1]

    # Karıştır
    random.shuffle(real_images)
    random.shuffle(fake_images)

    # Fold'lara böl
    def split_into_folds(img_list, k):
        folds = []
        fold_size = len(img_list) // k
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(img_list)
            folds.append(img_list[start:end])
        return folds

    real_folds = split_into_folds(real_images, k)
    fake_folds = split_into_folds(fake_images, k)

    # Fold çiftleri oluştur
    fold_pairs = []
    for i in range(k):
        test_images = real_folds[i] + fake_folds[i]
        train_images = []
        for j in range(k):
            if j != i:
                train_images.extend(real_folds[j])
                train_images.extend(fake_folds[j])
        fold_pairs.append((train_images, test_images))

        logger.info(f"  Fold {i+1}: Train={len(train_images)}, Test={len(test_images)}")

    return fold_pairs


def initialize_ensemble():
    """Ensemble motorunu başlatır."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from core.ensemble import EnsembleEngine
    from nodes.watermark import WatermarkNode
    from nodes.frequency import FrequencyNode

    try:
        from nodes.clip import CLIPNode
        has_clip = True
    except ImportError:
        has_clip = False

    try:
        from nodes.dire import DIRENode
        has_dire = False  # DIRE node yavaş, CV'de kullanmıyoruz
    except ImportError:
        has_dire = False

    nodes = [
        WatermarkNode(weight=1.0),
        FrequencyNode(weight=1.0),
    ]

    if has_clip:
        nodes.append(CLIPNode(weight=1.0))

    engine = EnsembleEngine(nodes=nodes, threshold=50.0)
    logger.info(f"✅ Ensemble motoru: {len(nodes)} node")

    return engine


def evaluate_fold(
    train_images: List,
    test_images: List,
    fold_idx: int
) -> Dict:
    """
    Tek bir fold'ü değerlendirir.

    Args:
        train_images: Eğitim verisi
        test_images: Test verisi
        fold_idx: Fold indeksi

    Returns:
        Fold metrikleri
    """
    logger.info(f"🔄 Fold {fold_idx + 1} değerlendiriliyor...")

    # Motoru başlat (her fold için yeni motor)
    engine = initialize_ensemble()

    # Test seti üzerinde değerlendir
    y_true = []
    y_pred = []
    y_scores = []

    for filename, image, true_label in test_images:
        try:
            result = engine.analyze(image)
            y_true.append(true_label)
            y_scores.append(result.final_score)
            y_pred.append(1 if result.final_score >= 50.0 else 0)
        except Exception as e:
            logger.warning(f"⚠️ Görsel işlenemedi {filename}: {e}")
            continue

    # Metrikleri hesapla
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    metrics = {
        'fold': fold_idx + 1,
        'train_size': len(train_images),
        'test_size': len(test_images),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }

    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}, "
               f"F1: {metrics['f1_score']:.4f}")

    return metrics


def aggregate_fold_results(fold_metrics: List[Dict]) -> Dict:
    """
    Tüm fold sonuçlarını birleştirir.

    Args:
        fold_metrics: Fold metrikleri listesi

    Returns:
        Birleştirilmiş sonuçlar
    """
    import numpy as np

    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    aggregated = {}

    for metric_name in metrics_names:
        values = [fold[metric_name] for fold in fold_metrics]
        aggregated[metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'values': values
        }

    aggregated['num_folds'] = len(fold_metrics)
    aggregated['fold_details'] = fold_metrics

    return aggregated


def plot_cv_results(results: Dict, output_path: Path):
    """
    Cross-validation sonuçlarını görselleştirir.

    Args:
        results: CV sonuçları
        output_path: Çıktı dosyası
    """
    import matplotlib.pyplot as plt

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    fold_indices = [fold['fold'] for fold in results['fold_details']]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        values = [fold[metric] for fold in results['fold_details']]
        mean_val = results[metric]['mean']
        std_val = results[metric]['std']

        axes[idx].plot(fold_indices, values, marker='o', label='Fold Skoru', color='steelblue')
        axes[idx].axhline(y=mean_val, color='red', linestyle='--', label=f'Ortalama: {mean_val:.4f}')
        axes[idx].fill_between(fold_indices,
                              mean_val - std_val,
                              mean_val + std_val,
                              alpha=0.2, color='red', label=f'±Std: {std_val:.4f}')
        axes[idx].set_xlabel('Fold')
        axes[idx].set_ylabel(metric.capitalize())
        axes[idx].set_title(f'{metric.capitalize()} - Cross-Validation')
        axes[idx].set_xticks(fold_indices)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"✅ CV sonuçları görselleştirildi: {output_path}")


def generate_cv_report(results: Dict, output_dir: Path):
    """
    Cross-validation raporu oluşturur.

    Args:
        results: CV sonuçları
        output_dir: Çıktı dizini
    """
    report_path = output_dir / "cross_validation_report.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# AI Human Detector - Cross-Validation Raporu\n\n")

        f.write("## 📊 Genel Sonuçlar\n\n")
        f.write(f"**Fold Sayısı**: {results['num_folds']}\n\n")

        f.write("| Metrik | Ortalama | Std | Min | Max |\n")
        f.write("|--------|----------|-----|-----|-----|\n")

        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            data = results[metric]
            f.write(f"| {metric.capitalize()} | {data['mean']:.4f} | "
                   f"{data['std']:.4f} | {data['min']:.4f} | {data['max']:.4f} |\n")

        f.write("\n## 📈 Detaylı Fold Sonuçları\n\n")
        f.write("| Fold | Accuracy | Precision | Recall | F1 Score | Train Size | Test Size |\n")
        f.write("|------|----------|-----------|--------|----------|------------|-----------|\n")

        for fold in results['fold_details']:
            f.write(f"| {fold['fold']} | {fold['accuracy']:.4f} | "
                   f"{fold['precision']:.4f} | {fold['recall']:.4f} | "
                   f"{fold['f1_score']:.4f} | {fold['train_size']} | {fold['test_size']} |\n")

        f.write("\n## 🎯 Değerlendirme\n\n")

        mean_acc = results['accuracy']['mean']
        std_acc = results['accuracy']['std']

        if mean_acc >= 0.95:
            f.write("✅ **Mükemmel**: Model %95+ accuracy hedefine ulaştı.\n\n")
        elif mean_acc >= 0.90:
            f.write("⚠️ **İyi**: Model %90+ accuracy ile hedefe yakın.\n\n")
        else:
            f.write("❌ **Geliştirme Gerekli**: Model hedefin altında.\n\n")

        # Std yorumu
        if std_acc < 0.02:
            f.write("✅ **Stabil**: Fold'lar arası varyans çok düşük (%2).\n\n")
        elif std_acc < 0.05:
            f.write("⚠️ **Orta**: Fold'lar arası varyans kabul edilebilir (%5).\n\n")
        else:
            f.write("❌ **İstikrarsız**: Fold'lar arası varyans yüksek (%5+).\n\n")

        f.write("## 📈 Görseller\n\n")
        f.write("Aşağıdaki görsel `results/cv/` dizininde:\n")
        f.write("- `cv_results.png` - Fold bazlı performans grafiği\n")

    logger.info(f"✅ CV raporu kaydedildi: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Human Detector - Cross-Validation Script'i"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./data/processed",
        help="Veri seti dizini (train/val/test klasörleri)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/cv",
        help="Çıktı dizini"
    )
    parser.add_argument(
        "-k",
        "--folds",
        type=int,
        default=5,
        help="Fold sayısı (varsayılan: 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Rastgelelik tohumu"
    )

    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    try:
        # Çıktı dizini
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Verileri yükle
        logger.info("🔄 ADIM 1: Veriler yükleniyor...")
        data_dir = Path(args.data)
        all_images, info = load_all_images(data_dir)

        if not all_images:
            logger.error("❌ Veri seti boş veya bulunamadı")
            sys.exit(1)

        # 2. K-fold böl
        logger.info("🔄 ADIM 2: K-fold bölünüyor...")
        fold_pairs = create_k_folds(all_images, k=args.folds, seed=args.seed)

        # 3. Her fold'ü değerlendir
        logger.info("🔄 ADIM 3: Fold'ler değerlendiriliyor...")
        fold_metrics = []

        for fold_idx, (train_images, test_images) in enumerate(fold_pairs):
            metrics = evaluate_fold(train_images, test_images, fold_idx)
            fold_metrics.append(metrics)

        # 4. Sonuçları birleştir
        logger.info("🔄 ADIM 4: Sonuçlar birleştiriliyor...")
        results = aggregate_fold_results(fold_metrics)

        # 5. Görselleştir
        logger.info("🔄 ADIM 5: Görselleştirmeler oluşturuluyor...")
        plot_cv_results(results, output_dir / "cv_results.png")

        # 6. Rapor
        logger.info("🔄 ADIM 6: Rapor oluşturuluyor...")
        generate_cv_report(results, output_dir)

        # JSON
        with open(output_dir / "cv_metrics.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Özet
        logger.info("\n" + "="*50)
        logger.info("🎉 Cross-validation tamamlandı!")
        logger.info("="*50)
        logger.info(f"✅ Ortalama Accuracy: {results['accuracy']['mean']:.4f} ± {results['accuracy']['std']:.4f}")
        logger.info(f"✅ Ortalama Precision: {results['precision']['mean']:.4f}")
        logger.info(f"✅ Ortalama Recall: {results['recall']['mean']:.4f}")
        logger.info(f"✅ Ortalama F1 Score: {results['f1_score']['mean']:.4f}")
        logger.info("="*50)
        logger.info(f"📁 Sonuçlar: {output_dir}")

    except Exception as e:
        logger.error(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
