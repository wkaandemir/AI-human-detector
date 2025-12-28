#!/usr/bin/env python3
"""
AI Human Detector - Model Doğrulama Script'i

Bu script, eğitilmiş modeli test seti üzerinde değerlendirir ve:
- Accuracy, Precision, Recall, F1 hesaplar
- ROC curve oluşturur
- Confusion matrix oluşturur
- False positive analizi yapar

Kullanım:
    python scripts/validate_model.py --data ./data/processed/test --output ./results
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
import logging
import time

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
        import matplotlib
        logger.info("✅ Tüm bağımlılıklar kurulu")
        return True
    except ImportError as e:
        logger.error(f"❌ Eksik bağımlılık: {e}")
        logger.info("Kurulum için: pip install pillow numpy matplotlib scikit-learn")
        return False


def load_images_from_directory(directory: Path, max_images: int = None) -> List[Tuple[Path, int]]:
    """
    Dizinden görselleri yükler.

    Args:
        directory: Görsel dizini
        max_images: Maksimum görsel sayısı (None = tümü)

    Returns:
        (dosya_yolu, etiket) listesi
        Etiket: 0 = REAL, 1 = FAKE
    """
    from PIL import Image

    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        image_files.extend(directory.glob(ext))

    if max_images:
        image_files = image_files[:max_images]

    return [(f, 0) for f in image_files]  # 0 = REAL (default, will be overridden)


def load_test_dataset(test_dir: Path, max_per_class: int = None) -> Tuple[List[Tuple[str, 'np.ndarray', int]], Dict]:
    """
    Test veri setini yükler.

    Args:
        test_dir: Test dizini (real/fake klasörleri)
        max_per_class: Her sınıf için maksimum görsel sayısı

    Returns:
        (dosya_adı, görsel, etiket) listesi ve bilgi sözlüğü
    """
    from PIL import Image
    import numpy as np

    logger.info("📊 Test veri seti yükleniyor...")

    dataset = []
    info = {"real": 0, "fake": 0}

    # Real görseller
    real_dir = test_dir / "real"
    if real_dir.exists():
        for img_path in real_dir.glob("*.png"):
            if max_per_class and info["real"] >= max_per_class:
                break
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_array = np.array(img)
                dataset.append((img_path.name, img_array, 0))  # 0 = REAL
                info["real"] += 1
            except Exception as e:
                logger.warning(f"⚠️ Görsel atlandı {img_path.name}: {e}")

    # Fake görseller
    fake_dir = test_dir / "fake"
    if fake_dir.exists():
        for img_path in fake_dir.glob("*.png"):
            if max_per_class and info["fake"] >= max_per_class:
                break
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_array = np.array(img)
                dataset.append((img_path.name, img_array, 1))  # 1 = FAKE
                info["fake"] += 1
            except Exception as e:
                logger.warning(f"⚠️ Görsel atlandı {img_path.name}: {e}")

    logger.info(f"✅ Veri seti yüklendi: {info['real']} REAL, {info['fake']} FAKE")
    return dataset, info


def initialize_ensemble():
    """
    Ensemble motorunu başlatır.

    Returns:
        EnsembleEngine
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from core.ensemble import EnsembleEngine
    from nodes.watermark import WatermarkNode
    from nodes.frequency import FrequencyNode

    try:
        from nodes.clip import CLIPNode
        has_clip = True
    except ImportError:
        logger.warning("⚠️ CLIP node kurulmamış, atlanıyor")
        has_clip = False

    try:
        from nodes.dire import DIRENode
        has_dire = True
    except ImportError:
        logger.warning("⚠️ DIRE node kurulmamış, atlanıyor")
        has_dire = False

    nodes = [
        WatermarkNode(weight=1.0),
        FrequencyNode(weight=1.0),
    ]

    if has_clip:
        nodes.append(CLIPNode(weight=1.0))

    if has_dire:
        nodes.append(DIRENode(weight=1.5))

    engine = EnsembleEngine(nodes=nodes, threshold=50.0)
    logger.info(f"✅ Ensemble motoru başlatıldı: {len(nodes)} node")

    return engine


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_scores: List[float]
) -> Dict[str, float]:
    """
    Sınıflandırma metriklerini hesaplar.

    Args:
        y_true: Gerçek etiketler (0=REAL, 1=FAKE)
        y_pred: Tahmin edilen etiketler
        y_scores: Tahmin skorları (0-100)

    Returns:
        Metrik sözlüğü
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_curve,
        auc
    )

    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel()
    metrics['true_negative'] = int(tn)
    metrics['false_positive'] = int(fp)
    metrics['false_negative'] = int(fn)
    metrics['true_positive'] = int(tp)

    # False Positive Rate
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # ROC AUC
    fpr, tpr, thresholds = roc_curve(y_true, [s / 100.0 for s in y_scores])
    metrics['roc_auc'] = auc(fpr, tpr)
    metrics['roc_curve'] = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist()
    }

    return metrics


def plot_confusion_matrix(cm, output_path: Path):
    """
    Confusion matrix görselleştirir.

    Args:
        cm: Confusion matrix
        output_path: Çıktı dosyası
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['REAL', 'FAKE'],
                yticklabels=['REAL', 'FAKE'])
    plt.title('Confusion Matrix')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"✅ Confusion matrix kaydedildi: {output_path}")


def plot_roc_curve(fpr, tpr, roc_auc, output_path: Path):
    """
    ROC curve görselleştirir.

    Args:
        fpr: False positive rate
        tpr: True positive rate
        roc_auc: AUC skoru
        output_path: Çıktı dosyası
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - AI Detection')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"✅ ROC curve kaydedildi: {output_path}")


def plot_score_distribution(y_scores_real, y_scores_fake, output_path: Path):
    """
    Skor dağılımını görselleştirir.

    Args:
        y_scores_real: Real görsellerin skorları
        y_scores_fake: Fake görsellerin skorları
        output_path: Çıktı dosyası
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.hist(y_scores_real, bins=30, alpha=0.5, label='REAL', color='green')
    plt.hist(y_scores_fake, bins=30, alpha=0.5, label='FAKE', color='red')
    plt.axvline(x=50, color='black', linestyle='--', label='Threshold (50)')
    plt.xlabel('AI Olasılık Skoru (0-100)')
    plt.ylabel('Görsel Sayısı')
    plt.title('Skor Dağılımı')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"✅ Skor dağılımı kaydedildi: {output_path}")


def analyze_false_positives(dataset, y_true, y_pred, y_scores, output_dir: Path):
    """
    False positive örneklerini analiz eder.

    Args:
        dataset: (dosya_adı, görsel, etiket) listesi
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        y_scores: Tahmin skorları
        output_dir: Çıktı dizini
    """
    import numpy as np

    false_positives = []
    false_negatives = []

    for i, (filename, _, true_label) in enumerate(dataset):
        pred_label = y_pred[i]
        score = y_scores[i]

        if true_label == 0 and pred_label == 1:
            # Real -> Fake tahmini (False Positive)
            false_positives.append((filename, score))
        elif true_label == 1 and pred_label == 0:
            # Fake -> Real tahmini (False Negative)
            false_negatives.append((filename, score))

    # Rapor oluştur
    report_path = output_dir / "false_positive_analysis.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# False Positive Analizi\n\n")

        f.write("## Özet\n\n")
        f.write(f"- **False Positive (Real → Fake)**: {len(false_positives)} görsel\n")
        f.write(f"- **False Negative (Fake → Real)**: {len(false_negatives)} görsel\n\n")

        if false_positives:
            f.write("## False Positive Örnekleri (Real → Fake)\n\n")
            f.write("| Dosya Adı | Skor |\n")
            f.write("|----------|------|\n")
            for filename, score in sorted(false_positives, key=lambda x: x[1], reverse=True)[:20]:
                f.write(f"| {filename} | {score:.2f} |\n")
            f.write("\n")

        if false_negatives:
            f.write("## False Negative Örnekleri (Fake → Real)\n\n")
            f.write("| Dosya Adı | Skor |\n")
            f.write("|----------|------|\n")
            for filename, score in sorted(false_negatives, key=lambda x: x[1])[:20]:
                f.write(f"| {filename} | {score:.2f} |\n")
            f.write("\n")

    logger.info(f"✅ False positive analizi kaydedildi: {report_path}")


def generate_report(metrics: Dict, output_dir: Path):
    """
    Detaylı doğrulama raporu oluşturur.

    Args:
        metrics: Metrik sözlüğü
        output_dir: Çıktı dizini
    """
    report_path = output_dir / "validation_report.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# AI Human Detector - Model Doğrulama Raporu\n\n")

        f.write("## 📊 Sınıflandırma Metrikleri\n\n")
        f.write("| Metrik | Değer |\n")
        f.write("|--------|-------|\n")
        f.write(f"| **Accuracy** | {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%) |\n")
        f.write(f"| **Precision** | {metrics['precision']:.4f} |\n")
        f.write(f"| **Recall** | {metrics['recall']:.4f} |\n")
        f.write(f"| **F1 Score** | {metrics['f1_score']:.4f} |\n")
        f.write(f"| **ROC AUC** | {metrics['roc_auc']:.4f} |\n")
        f.write(f"| **False Positive Rate** | {metrics['false_positive_rate']:.4f} ({metrics['false_positive_rate']*100:.2f}%) |\n\n")

        f.write("## 🎯 Confusion Matrix\n\n")
        cm = metrics['confusion_matrix']
        f.write("| | Tahmin: REAL | Tahmin: FAKE |\n")
        f.write("|----------|-------------|--------------|\n")
        f.write(f"| **Gerçek: REAL** | {cm[0][0]} (TN) | {cm[0][1]} (FP) |\n")
        f.write(f"| **Gerçek: FAKE** | {cm[1][0]} (FN) | {cm[1][1]} (TP) |\n\n")

        f.write("### Detaylar\n\n")
        f.write(f"- **True Negative (TN)**: {metrics['true_negative']} - Real doğru tanımlandı\n")
        f.write(f"- **False Positive (FP)**: {metrics['false_positive']} - Real yanlış fake olarak tanımlandı\n")
        f.write(f"- **False Negative (FN)**: {metrics['false_negative']} - Fake yanlış real olarak tanımlandı\n")
        f.write(f"- **True Positive (TP)**: {metrics['true_positive']} - Fake doğru tanımlandı\n\n")

        f.write("## 📈 Görseller\n\n")
        f.write("Aşağıdaki görseller `results/` dizininde kaydedilmiştir:\n\n")
        f.write("- `confusion_matrix.png` - Confusion matrix görselleştirmesi\n")
        f.write("- `roc_curve.png` - ROC curve\n")
        f.write("- `score_distribution.png` - Skor dağılımı\n\n")

        f.write("## 🎯 Hedefler vs Gerçekçek\n\n")
        f.write("| Metrik | Hedef | Gerçekçek | Durum |\n")
        f.write("|--------|-------|-----------|-------|\n")
        accuracy_status = "✅" if metrics['accuracy'] >= 0.95 else "⚠️"
        f.write(f"| Accuracy | %95+ | %{metrics['accuracy']*100:.2f} | {accuracy_status} |\n")
        fpr_status = "✅" if metrics['false_positive_rate'] <= 0.02 else "⚠️"
        f.write(f"| FPR | <%2 | %{metrics['false_positive_rate']*100:.2f} | {fpr_status} |\n\n")

    logger.info(f"✅ Doğrulama raporu kaydedildi: {report_path}")


def save_metrics_json(metrics: Dict, output_path: Path):
    """
    Metrikleri JSON formatında kaydeder.

    Args:
        metrics: Metrik sözlüğü
        output_path: Çıktı dosyası
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✅ Metrikler kaydedildi: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Human Detector - Model Doğrulama Script'i"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./data/processed/test",
        help="Test veri seti dizini"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/validation",
        help="Çıktı dizini"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Her sınıf için maksimum görsel sayısı"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Sınıflandırma eşiği (0-100)"
    )

    args = parser.parse_args()

    # Bağımlılıkları kontrol et
    if not check_dependencies():
        sys.exit(1)

    try:
        # Çıktı dizinini oluştur
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Veri setini yükle
        logger.info("🔄 ADIM 1: Veri seti yükleniyor...")
        test_dir = Path(args.data)
        dataset, info = load_test_dataset(test_dir, max_per_class=args.max_images)

        if not dataset:
            logger.error("❌ Veri seti boş veya bulunamadı")
            sys.exit(1)

        # 2. Ensemble motorunu başlat
        logger.info("🔄 ADIM 2: Ensemble motoru başlatılıyor...")
        engine = initialize_ensemble()
        engine.threshold = args.threshold

        # 3. Analiz yap
        logger.info("🔄 ADIM 3: Görseller analiz ediliyor...")
        y_true = []
        y_pred = []
        y_scores = []
        processing_times = []

        for filename, image, true_label in dataset:
            start = time.time()
            result = engine.analyze(image)
            proc_time = time.time() - start

            y_true.append(true_label)
            y_scores.append(result.final_score)
            # Score >= threshold → FAKE (1), otherwise REAL (0)
            y_pred.append(1 if result.final_score >= args.threshold else 0)
            processing_times.append(proc_time)

            if len(y_true) % 10 == 0:
                logger.info(f"  İlerleme: {len(y_true)}/{len(dataset)}")

        # 4. Metrikleri hesapla
        logger.info("🔄 ADIM 4: Metrikler hesaplanıyor...")
        metrics = compute_metrics(y_true, y_pred, y_scores)

        # Ortalama işlem süresi
        metrics['avg_processing_time'] = sum(processing_times) / len(processing_times)
        metrics['total_images'] = len(dataset)
        metrics['threshold'] = args.threshold

        # 5. Görselleştirmeler
        logger.info("🔄 ADIM 5: Görselleştirmeler oluşturuluyor...")

        # Confusion Matrix
        cm = metrics['confusion_matrix']
        import numpy as np
        plot_confusion_matrix(np.array(cm), output_dir / "confusion_matrix.png")

        # ROC Curve
        roc_data = metrics['roc_curve']
        plot_roc_curve(
            roc_data['fpr'],
            roc_data['tpr'],
            metrics['roc_auc'],
            output_dir / "roc_curve.png"
        )

        # Skor dağılımı
        y_scores_real = [y_scores[i] for i in range(len(y_true)) if y_true[i] == 0]
        y_scores_fake = [y_scores[i] for i in range(len(y_true)) if y_true[i] == 1]
        plot_score_distribution(y_scores_real, y_scores_fake, output_dir / "score_distribution.png")

        # 6. False Positive Analizi
        logger.info("🔄 ADIM 6: False positive analizi yapılıyor...")
        analyze_false_positives(dataset, y_true, y_pred, y_scores, output_dir)

        # 7. Raporları kaydet
        logger.info("🔄 ADIM 7: Raporlar kaydediliyor...")
        generate_report(metrics, output_dir)
        save_metrics_json(metrics, output_dir / "metrics.json")

        # 8. Özet
        logger.info("\n" + "="*50)
        logger.info("🎉 Doğrulama tamamlandı!")
        logger.info("="*50)
        logger.info(f"✅ Accuracy: %{metrics['accuracy']*100:.2f}")
        logger.info(f"✅ Precision: {metrics['precision']:.4f}")
        logger.info(f"✅ Recall: {metrics['recall']:.4f}")
        logger.info(f"✅ F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"✅ ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"✅ FPR: %{metrics['false_positive_rate']*100:.2f}")
        logger.info(f"✅ Ortalama işlem süresi: {metrics['avg_processing_time']:.3f}s")
        logger.info("="*50)
        logger.info(f"📁 Sonuçlar: {output_dir}")

    except Exception as e:
        logger.error(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
