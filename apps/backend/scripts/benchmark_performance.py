#!/usr/bin/env python3
"""
AI Human Detector - Performans Benchmark Script'i

Bu script, modelin işleme hızını ve kaynak kullanımını ölçer:
- İşleme hızı (görsel/saniye)
- GPU bellek kullanımı
- CPU kullanımı
- Batch processing optimizasyonu

Kullanım:
    python scripts/benchmark_performance.py --data ./data/processed/test --output ./results/benchmark
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict
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
        import PIL
        import numpy as np
        import psutil
        logger.info("✅ Tüm bağımlılıklar kurulu")
        return True
    except ImportError as e:
        logger.error(f"❌ Eksik bağımlılık: {e}")
        logger.info("Kurulum için: pip install pillow numpy psutil")
        return False


def load_test_images(test_dir: Path, max_images: int = 50) -> List:
    """Test görsellerini yükler."""
    from PIL import Image
    import numpy as np

    images = []
    count = 0

    for label_dir in [test_dir / "real", test_dir / "fake"]:
        if not label_dir.exists():
            continue

        for img_path in label_dir.glob("*.png"):
            if count >= max_images:
                break
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_array = np.array(img)
                images.append((img_path.name, img_array))
                count += 1
            except Exception as e:
                logger.warning(f"⚠️ Görsel atlandı {img_path.name}: {e}")

    logger.info(f"✅ {len(images)} görsel yüklendi")
    return images


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
        has_dire = True
    except ImportError:
        has_dire = False

    nodes = [WatermarkNode(weight=1.0), FrequencyNode(weight=1.0)]

    if has_clip:
        nodes.append(CLIPNode(weight=1.0))

    if has_dire:
        nodes.append(DIRENode(weight=1.5))

    engine = EnsembleEngine(nodes=nodes)
    logger.info(f"✅ Ensemble motoru: {len(nodes)} node")

    return engine


def benchmark_single_image_processing(engine, images: List, warmup_runs: int = 3) -> Dict:
    """
    Tekil görsel işleme performansını ölçer.

    Args:
        engine: Ensemble motoru
        images: Test görselleri
        warmup_runs: Isınma turu sayısı

    Returns:
        Performans metrikleri
    """
    logger.info("🔄 Tekil görsel işleme benchmark'ı...")

    # Isınma turları
    for i in range(min(warmup_runs, len(images))):
        _, img = images[i]
        engine.analyze(img)

    # Ölçüm
    processing_times = []
    memory_usage = []

    import psutil
    process = psutil.Process()

    for filename, img in images:
        # Başlangıç belleği
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # İşlem
        start = time.time()
        result = engine.analyze(img)
        elapsed = time.time() - start

        # Bitiş belleği
        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        processing_times.append(elapsed)
        memory_usage.append(mem_after - mem_before)

    # İstatistikler
    import numpy as np

    metrics = {
        'single_image': {
            'avg_time': float(np.mean(processing_times)),
            'min_time': float(np.min(processing_times)),
            'max_time': float(np.max(processing_times)),
            'std_time': float(np.std(processing_times)),
            'median_time': float(np.median(processing_times)),
            'throughput': 1.0 / np.mean(processing_times),  # images per second
            'avg_memory_mb': float(np.mean([m for m in memory_usage if m > 0])),
            'max_memory_mb': float(np.max(memory_usage)),
            'total_images': len(images)
        }
    }

    logger.info(f"  Ortalama: {metrics['single_image']['avg_time']:.3f}s/görsel")
    logger.info(f"  Throughput: {metrics['single_image']['throughput']:.2f} görsel/s")
    logger.info(f"  Ortalama bellek: {metrics['single_image']['avg_memory_mb']:.2f} MB")

    return metrics


def benchmark_batch_processing(engine, images: List, batch_sizes: List[int] = None) -> Dict:
    """
    Batch processing performansını ölçer.

    Args:
        engine: Ensemble motoru
        images: Test görselleri
        batch_sizes: Denenecek batch boyutları

    Returns:
        Batch performans metrikleri
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16]

    logger.info("🔄 Batch processing benchmark'ı...")

    batch_results = {}

    for batch_size in batch_sizes:
        if batch_size > len(images):
            continue

        logger.info(f"  Batch boyutu: {batch_size}")

        times = []
        num_batches = len(images) // batch_size

        for i in range(num_batches):
            batch = images[i * batch_size:(i + 1) * batch_size]

            start = time.time()
            for _, img in batch:
                engine.analyze(img)
            elapsed = time.time() - start

            times.append(elapsed)

        import numpy as np

        batch_results[batch_size] = {
            'avg_time_per_batch': float(np.mean(times)),
            'avg_time_per_image': float(np.mean(times) / batch_size),
            'throughput': float(batch_size / np.mean(times)),
            'total_batches': num_batches
        }

        logger.info(f"    Ortalama: {batch_results[batch_size]['avg_time_per_image']:.3f}s/görsel")
        logger.info(f"    Throughput: {batch_results[batch_size]['throughput']:.2f} görsel/s")

    return {'batch_processing': batch_results}


def benchmark_gpu_memory(engine, images: List) -> Dict:
    """
    GPU bellek kullanımını ölçer.

    Args:
        engine: Ensemble motoru
        images: Test görselleri

    Returns:
        GPU bellek metrikleri
    """
    logger.info("🔄 GPU bellek kullanımı ölçülüyor...")

    try:
        import torch

        if not torch.cuda.is_available():
            logger.info("  GPU kullanılamıyor, CPU modunda ölçülüyor")
            return {'gpu_memory': {'available': False}}

        # Başlangıç belleği
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB

        # Test görsellerini işle
        for _, img in images[:10]:  # İlk 10 görsel
            engine.analyze(img)

        # Maksimum bellek
        mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        # Temizle
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB

        # GPU bilgisi
        gpu_info = {
            'available': True,
            'device_name': torch.cuda.get_device_name(0),
            'total_memory_mb': torch.cuda.get_device_properties(0).total_memory / 1024 / 1024,
            'memory_before_mb': mem_before,
            'memory_peak_mb': mem_peak,
            'memory_after_mb': mem_after,
            'memory_used_mb': mem_peak - mem_before
        }

        logger.info(f"  GPU: {gpu_info['device_name']}")
        logger.info(f"  Toplam bellek: {gpu_info['total_memory_mb']:.0f} MB")
        logger.info(f"  Tepe kullanım: {gpu_info['memory_peak_mb']:.0f} MB")

        return {'gpu_memory': gpu_info}

    except ImportError:
        logger.info("  PyTorch kurulmamış")
        return {'gpu_memory': {'available': False, 'reason': 'PyTorch not installed'}}


def benchmark_node_performance(engine, images: List) -> Dict:
    """
    Her node'un performansını ayrı ayrı ölçer.

    Args:
        engine: Ensemble motoru
        images: Test görselleri

    Returns:
        Node performans metrikleri
    """
    logger.info("🔄 Node performansı ölçülüyor...")

    import numpy as np

    node_times = {node.name: [] for node in engine.nodes if node.enabled}

    for _, img in images[:20]:  # İlk 20 görsel
        start_total = time.time()
        for node in engine.nodes:
            if not node.enabled:
                continue
            start = time.time()
            try:
                node.analyze(img)
            except:
                pass
            elapsed = time.time() - start
            node_times[node.name].append(elapsed)

    node_stats = {}
    for node_name, times in node_times.items():
        if times:
            node_stats[node_name] = {
                'avg_time_ms': float(np.mean(times) * 1000),
                'min_time_ms': float(np.min(times) * 1000),
                'max_time_ms': float(np.max(times) * 1000),
                'std_time_ms': float(np.std(times) * 1000)
            }
            logger.info(f"  {node_name}: {node_stats[node_name]['avg_time_ms']:.2f} ms")

    return {'node_performance': node_stats}


def plot_performance_benchmark(metrics: Dict, output_dir: Path):
    """Performans grafikleri oluşturur."""
    import matplotlib.pyplot as plt

    # 1. Processing time distribution
    if 'single_image' in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Throughput
        throughput = metrics['single_image']['throughput']
        axes[0].bar(['Tekli İşlem'], [throughput], color='steelblue')
        axes[0].set_ylabel('Görsel/Saniye')
        axes[0].set_title('İşleme Hızı')
        axes[0].grid(axis='y', alpha=0.3)

        # Memory usage
        mem = metrics['single_image']['avg_memory_mb']
        axes[1].bar(['Ortalama Bellek'], [mem], color='coral')
        axes[1].set_ylabel('Bellek (MB)')
        axes[1].set_title('Bellek Kullanımı')
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "single_performance.png", dpi=150)
        plt.close()
        logger.info("✅ Tekli performans grafiği kaydedildi")

    # 2. Batch processing comparison
    if 'batch_processing' in metrics:
        batch_data = metrics['batch_processing']

        if batch_data:
            batch_sizes = sorted(batch_data.keys())
            throughputs = [batch_data[bs]['throughput'] for bs in batch_sizes]
            times = [batch_data[bs]['avg_time_per_image'] for bs in batch_sizes]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].plot(batch_sizes, throughputs, marker='o', color='steelblue')
            axes[0].set_xlabel('Batch Boyutu')
            axes[0].set_ylabel('Görsel/Saniye')
            axes[0].set_title('Batch Processing - Throughput')
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(batch_sizes, times, marker='o', color='coral')
            axes[1].set_xlabel('Batch Boyutu')
            axes[1].set_ylabel('Süre/Görsel (s)')
            axes[1].set_title('Batch Processing - İşleme Süresi')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "batch_performance.png", dpi=150)
            plt.close()
            logger.info("✅ Batch performans grafiği kaydedildi")

    # 3. Node performance comparison
    if 'node_performance' in metrics:
        node_data = metrics['node_performance']

        if node_data:
            nodes = list(node_data.keys())
            times = [node_data[node]['avg_time_ms'] for node in nodes]

            plt.figure(figsize=(10, 6))
            plt.barh(nodes, times, color='steelblue')
            plt.xlabel('Ortalama İşleme Süresi (ms)')
            plt.title('Node Performans Karşılaştırması')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "node_performance.png", dpi=150)
            plt.close()
            logger.info("✅ Node performans grafiği kaydedildi")


def generate_benchmark_report(metrics: Dict, output_dir: Path):
    """Benchmark raporu oluşturur."""
    report_path = output_dir / "benchmark_report.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# AI Human Detector - Performans Benchmark Raporu\n\n")

        # Tekli Görsel İşleme
        if 'single_image' in metrics:
            f.write("## 📊 Tekli Görsel İşleme\n\n")
            si = metrics['single_image']
            f.write(f"- **Ortalama Süre**: {si['avg_time']:.3f} s/görsel\n")
            f.write(f"- **Min Süre**: {si['min_time']:.3f} s\n")
            f.write(f"- **Max Süre**: {si['max_time']:.3f} s\n")
            f.write(f"- **Standart Sapma**: {si['std_time']:.3f} s\n")
            f.write(f"- **Medyan**: {si['median_time']:.3f} s\n")
            f.write(f"- **Throughput**: {si['throughput']:.2f} görsel/s\n")
            f.write(f"- **Ortalama Bellek**: {si['avg_memory_mb']:.2f} MB\n\n")

        # Batch Processing
        if 'batch_processing' in metrics:
            f.write("## 🔄 Batch Processing\n\n")
            f.write("| Batch Boyutu | Süre/Batch (s) | Süre/Görsel (s) | Throughput (img/s) |\n")
            f.write("|-------------|----------------|-----------------|---------------------|\n")

            for batch_size, data in sorted(metrics['batch_processing'].items()):
                f.write(f"| {batch_size} | {data['avg_time_per_batch']:.3f} | "
                       f"{data['avg_time_per_image']:.3f} | {data['throughput']:.2f} |\n")
            f.write("\n")

        # GPU Bellek
        if 'gpu_memory' in metrics:
            f.write("## 🎮 GPU Bellek Kullanımı\n\n")
            gpu = metrics['gpu_memory']

            if gpu.get('available'):
                f.write(f"- **GPU**: {gpu['device_name']}\n")
                f.write(f"- **Toplam Bellek**: {gpu['total_memory_mb']:.0f} MB\n")
                f.write(f"- **Tepe Kullanım**: {gpu['memory_peak_mb']:.0f} MB\n")
                f.write(f"- **Kullanılan Bellek**: {gpu['memory_used_mb']:.0f} MB\n\n")
            else:
                reason = gpu.get('reason', 'GPU kullanılamıyor')
                f.write(f"⚠️ {reason}\n\n")

        # Node Performansı
        if 'node_performance' in metrics:
            f.write("## 🧩 Node Performansı\n\n")
            f.write("| Node | Ortalama (ms) | Min (ms) | Max (ms) | Std (ms) |\n")
            f.write("|------|--------------|----------|----------|----------|\n")

            for node_name, data in metrics['node_performance'].items():
                f.write(f"| {node_name} | {data['avg_time_ms']:.2f} | "
                       f"{data['min_time_ms']:.2f} | {data['max_time_ms']:.2f} | "
                       f"{data['std_time_ms']:.2f} |\n")
            f.write("\n")

        # Öneriler
        f.write("## 💡 Optimizasyon Önerileri\n\n")

        if 'batch_processing' in metrics:
            batch_data = metrics['batch_processing']
            if batch_data:
                best_batch = max(batch_data.items(), key=lambda x: x[1]['throughput'])
                f.write(f"- **En iyi batch boyutu**: {best_batch[0]} "
                       f"({best_batch[1]['throughput']:.2f} görsel/s)\n")

        if 'single_image' in metrics:
            throughput = metrics['single_image']['throughput']
            if throughput < 1.0:
                f.write("- ⚠️ İşleme hızı düşük, GPU kullanımı考虑\n")
            else:
                f.write("- ✅ İşleme hızı iyi düzeyde\n")

        f.write("\n## 📈 Görseller\n\n")
        f.write("Aşağıdaki görseller `results/benchmark/` dizininde:\n")
        f.write("- `single_performance.png` - Tekli işlem performansı\n")
        f.write("- `batch_performance.png` - Batch processing karşılaştırması\n")
        f.write("- `node_performance.png` - Node performansı\n")

    logger.info(f"✅ Benchmark raporu kaydedildi: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Human Detector - Performans Benchmark Script'i"
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
        default="./results/benchmark",
        help="Çıktı dizini"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Test edilecek maksimum görsel sayısı"
    )

    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    try:
        # Çıktı dizini
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Görselleri yükle
        logger.info("🔄 Test görselleri yükleniyor...")
        images = load_test_images(Path(args.data), max_images=args.max_images)

        if not images:
            logger.error("❌ Görsel bulunamadı")
            sys.exit(1)

        # Motoru başlat
        logger.info("🔄 Ensemble motoru başlatılıyor...")
        engine = initialize_ensemble()

        # Benchmark'lar
        all_metrics = {}

        # 1. Tekli işlem
        single_metrics = benchmark_single_image_processing(engine, images)
        all_metrics.update(single_metrics)

        # 2. Batch processing
        batch_metrics = benchmark_batch_processing(engine, images)
        all_metrics.update(batch_metrics)

        # 3. GPU bellek
        gpu_metrics = benchmark_gpu_memory(engine, images)
        all_metrics.update(gpu_metrics)

        # 4. Node performansı
        node_metrics = benchmark_node_performance(engine, images)
        all_metrics.update(node_metrics)

        # 5. Görselleştirmeler
        logger.info("🔄 Görselleştirmeler oluşturuluyor...")
        plot_performance_benchmark(all_metrics, output_dir)

        # 6. Raporlar
        logger.info("🔄 Raporlar oluşturuluyor...")
        generate_benchmark_report(all_metrics, output_dir)

        # JSON
        with open(output_dir / "benchmark_metrics.json", 'w') as f:
            json.dump(all_metrics, f, indent=2)

        logger.info("\n" + "="*50)
        logger.info("🎉 Benchmark tamamlandı!")
        logger.info("="*50)
        logger.info(f"📁 Sonuçlar: {output_dir}")

    except Exception as e:
        logger.error(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
