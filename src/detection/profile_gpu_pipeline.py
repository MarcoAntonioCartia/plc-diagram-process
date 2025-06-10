"""
Profile GPU-optimized pipeline to identify bottlenecks
"""

import time
import torch
from pathlib import Path
import sys
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.detection.detect_pipeline_gpu_optimized import GPUOptimizedPLCDetectionPipeline, ImageDataset
from torch.utils.data import DataLoader
from src.config import get_config


def profile_gpu_detection(model_path, images_folder, batch_size=32, num_workers=4):
    """Profile GPU detection to identify bottlenecks"""
    
    print(f"\nPROFILING GPU DETECTION PIPELINE")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {num_workers}")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = GPUOptimizedPLCDetectionPipeline(
        model_path=model_path,
        batch_size=batch_size,
        num_workers=num_workers,
        use_amp=True
    )
    
    # Find images
    image_files = [f for f in Path(images_folder).glob("*.png") if not f.name.endswith("_metadata.json")]
    print(f"Found {len(image_files)} images to process")
    
    # Create dataset and dataloader
    dataset = ImageDataset(image_files, pipeline.img_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(pipeline.device != 'cpu'),
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Timing breakdown
    timings = defaultdict(list)
    
    # Process batches with detailed timing
    total_start = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, paths, names) in enumerate(dataloader):
            batch_start = time.time()
            
            # Time data transfer
            transfer_start = time.time()
            images = images.to(pipeline.device, non_blocking=True)
            if pipeline.device != 'cpu':
                torch.cuda.synchronize()
            timings['data_transfer'].append(time.time() - transfer_start)
            
            # Time inference
            inference_start = time.time()
            if pipeline.use_amp and pipeline.device != 'cpu':
                with torch.amp.autocast('cuda'):
                    outputs = pipeline.model(images)
            else:
                outputs = pipeline.model(images)
            if pipeline.device != 'cpu':
                torch.cuda.synchronize()
            timings['inference'].append(time.time() - inference_start)
            
            # Time post-processing
            post_start = time.time()
            detection_count = 0
            for output in outputs:
                if output.boxes is not None:
                    boxes = output.boxes
                    conf_mask = boxes.conf >= pipeline.confidence_threshold
                    detection_count += conf_mask.sum().item()
            timings['post_process'].append(time.time() - post_start)
            
            # Total batch time
            timings['total_batch'].append(time.time() - batch_start)
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_batch_time = sum(timings['total_batch']) / len(timings['total_batch'])
                print(f"Batch {batch_idx + 1}/{len(dataloader)}: "
                      f"avg {avg_batch_time*1000:.1f}ms/batch, "
                      f"{batch_size/avg_batch_time:.1f} img/s")
    
    total_time = time.time() - total_start
    
    # Calculate statistics
    print("\n" + "=" * 60)
    print("TIMING BREAKDOWN (per batch):")
    print("=" * 60)
    
    for component, times in timings.items():
        if times:
            avg_time = sum(times) / len(times) * 1000  # Convert to ms
            percentage = (sum(times) / sum(timings['total_batch'])) * 100 if component != 'total_batch' else 100
            print(f"{component:15s}: {avg_time:6.1f}ms ({percentage:5.1f}%)")
    
    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE:")
    print("=" * 60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Images processed: {len(image_files)}")
    print(f"Speed: {len(image_files)/total_time:.2f} img/s")
    
    # Additional analysis
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS:")
    print("=" * 60)
    
    # Calculate time spent in each component
    total_inference = sum(timings['inference'])
    total_transfer = sum(timings['data_transfer'])
    total_post = sum(timings['post_process'])
    total_other = total_time - total_inference - total_transfer - total_post
    
    print(f"Inference: {total_inference:.2f}s ({total_inference/total_time*100:.1f}%)")
    print(f"Data transfer: {total_transfer:.2f}s ({total_transfer/total_time*100:.1f}%)")
    print(f"Post-processing: {total_post:.2f}s ({total_post/total_time*100:.1f}%)")
    print(f"Data loading/other: {total_other:.2f}s ({total_other/total_time*100:.1f}%)")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    
    if total_other / total_time > 0.3:
        print("- Data loading is a bottleneck. Consider:")
        print("  * Increasing num_workers")
        print("  * Using faster storage (SSD)")
        print("  * Pre-loading data into memory")
    
    if total_transfer / total_time > 0.2:
        print("- Data transfer is a bottleneck. Consider:")
        print("  * Ensuring pin_memory is enabled")
        print("  * Using non_blocking transfers")
        print("  * Reducing image size if possible")
    
    if total_inference / total_time < 0.5:
        print("- GPU is underutilized. Consider:")
        print("  * Increasing batch size")
        print("  * Using multiple GPUs")
        print("  * Optimizing data pipeline")


def main():
    # Get config
    config = get_config()
    data_root = Path(config.config['data_root'])
    images_folder = data_root / "processed" / "images"
    
    # Find model
    runs_dir = config.get_run_path('train')
    model_path = None
    if runs_dir.exists():
        train_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "plc_symbol_detector" in d.name]
        if train_dirs:
            latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
            model_path = latest_dir / "weights" / "best.pt"
    
    if model_path is None:
        print("Error: No trained model found")
        return 1
    
    print(f"Using model: {model_path}")
    
    # Test different configurations
    configurations = [
        (8, 4),    # batch_size, num_workers
        (16, 4),
        (32, 4),
        (64, 4),
        (32, 2),
        (32, 8),
    ]
    
    for batch_size, num_workers in configurations:
        profile_gpu_detection(model_path, images_folder, batch_size, num_workers)
        print("\n" + "#" * 80 + "\n")
        
        # Clear GPU cache between tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    exit(main())
