"""
Benchmark script to compare sequential vs parallel detection performance
"""

import time
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.detection.detect_pipeline import PLCDetectionPipeline
from src.detection.detect_pipeline_parallel import ParallelPLCDetectionPipeline
from src.config import get_config


def benchmark_sequential(model_path, images_folder, batch_size=1):
    """Benchmark sequential detection"""
    print("\n" + "="*60)
    print("BENCHMARKING SEQUENTIAL DETECTION")
    print("="*60)
    
    pipeline = PLCDetectionPipeline(model_path=model_path)
    
    start_time = time.time()
    
    # Run detection only (skip PDF conversion)
    detection_results = pipeline._detect_on_snippets(images_folder)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Count total detections
    total_detections = sum(d.get("detection_count", 0) for d in detection_results.values())
    images_processed = len(detection_results)
    
    print(f"\nSequential Results:")
    print(f"  - Time: {elapsed_time:.2f} seconds")
    print(f"  - Images processed: {images_processed}")
    print(f"  - Total detections: {total_detections}")
    print(f"  - Speed: {images_processed/elapsed_time:.2f} images/second")
    
    return elapsed_time, images_processed, total_detections


def benchmark_parallel(model_path, images_folder, batch_size=32, num_workers=4):
    """Benchmark parallel detection with GPU batching"""
    print("\n" + "="*60)
    print(f"BENCHMARKING PARALLEL DETECTION (batch_size={batch_size})")
    print("="*60)
    
    pipeline = ParallelPLCDetectionPipeline(
        model_path=model_path,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    start_time = time.time()
    
    # Run batch detection
    detection_results = pipeline._detect_on_snippets_batch(images_folder)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Count total detections
    total_detections = sum(d.get("detection_count", 0) for d in detection_results.values())
    images_processed = len(detection_results)
    
    print(f"\nParallel Results (batch_size={batch_size}):")
    print(f"  - Time: {elapsed_time:.2f} seconds")
    print(f"  - Images processed: {images_processed}")
    print(f"  - Total detections: {total_detections}")
    print(f"  - Speed: {images_processed/elapsed_time:.2f} images/second")
    
    return elapsed_time, images_processed, total_detections


def main():
    parser = argparse.ArgumentParser(description='Benchmark detection performance')
    parser.add_argument('--model', '-m', default=None,
                       help='Path to YOLO model (default: auto-detect)')
    parser.add_argument('--images', '-i', default=None,
                       help='Path to images folder (default: from config)')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 8, 16, 32, 64],
                       help='Batch sizes to test (default: 1 8 16 32 64)')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--skip-sequential', action='store_true',
                       help='Skip sequential benchmark')
    
    args = parser.parse_args()
    
    # Get config
    config = get_config()
    
    # Set images folder
    if args.images:
        images_folder = Path(args.images)
    else:
        data_root = Path(config.config['data_root'])
        images_folder = data_root / "processed" / "images"
    
    if not images_folder.exists():
        print(f"Error: Images folder not found: {images_folder}")
        return 1
    
    # Check for images
    image_count = len(list(images_folder.glob("*.png")))
    if image_count == 0:
        print(f"Error: No images found in {images_folder}")
        return 1
    
    print(f"Found {image_count} images to process")
    
    # Find model
    if args.model is None:
        # Auto-detect best model
        runs_dir = config.get_run_path('train')
        if runs_dir.exists():
            train_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "plc_symbol_detector" in d.name]
            if train_dirs:
                latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
                model_path = latest_dir / "weights" / "best.pt"
                if model_path.exists():
                    print(f"Using model: {model_path}")
                else:
                    print("Error: No trained model found")
                    return 1
            else:
                print("Error: No trained model found")
                return 1
        else:
            print("Error: No trained model found")
            return 1
    else:
        model_path = Path(args.model)
    
    results = {}
    
    # Run sequential benchmark
    if not args.skip_sequential:
        seq_time, seq_images, seq_detections = benchmark_sequential(model_path, images_folder)
        results['sequential'] = {
            'time': seq_time,
            'images': seq_images,
            'detections': seq_detections,
            'speed': seq_images / seq_time
        }
    
    # Run parallel benchmarks with different batch sizes
    for batch_size in args.batch_sizes:
        if batch_size == 1 and not args.skip_sequential:
            continue  # Skip batch_size=1 if we already did sequential
        
        par_time, par_images, par_detections = benchmark_parallel(
            model_path, images_folder, batch_size, args.workers
        )
        results[f'parallel_batch_{batch_size}'] = {
            'time': par_time,
            'images': par_images,
            'detections': par_detections,
            'speed': par_images / par_time
        }
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    if 'sequential' in results:
        baseline_time = results['sequential']['time']
        baseline_speed = results['sequential']['speed']
        
        print(f"\nBaseline (Sequential):")
        print(f"  - Time: {baseline_time:.2f}s")
        print(f"  - Speed: {baseline_speed:.2f} images/s")
        
        print("\nSpeedup vs Sequential:")
        for key, result in results.items():
            if key != 'sequential':
                speedup = baseline_time / result['time']
                speed_increase = result['speed'] / baseline_speed
                print(f"  - {key}: {speedup:.2fx} faster ({speed_increase:.2fx} throughput)")
    
    # Find optimal batch size
    best_config = min(results.items(), key=lambda x: x[1]['time'])
    print(f"\nOptimal configuration: {best_config[0]}")
    print(f"  - Processing time: {best_config[1]['time']:.2f}s")
    print(f"  - Throughput: {best_config[1]['speed']:.2f} images/s")
    
    return 0


if __name__ == "__main__":
    exit(main())
