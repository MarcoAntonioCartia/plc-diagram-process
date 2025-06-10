"""
Comprehensive benchmark comparing original vs optimized GPU pipelines
Demonstrates the performance improvements and identifies bottlenecks
"""

import json
import time
import argparse
from pathlib import Path
import sys
import torch
import gc
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.config import get_config


class GPUPipelineBenchmark:
    """
    Comprehensive benchmarking suite for GPU pipeline optimization analysis
    """
    
    def __init__(self, model_path=None, images_folder=None):
        """
        Initialize benchmark suite
        
        Args:
            model_path: Path to YOLO model (None for auto-detect)
            images_folder: Path to images folder (None for auto-detect)
        """
        self.config = get_config()
        self.model_path = self._resolve_model_path(model_path)
        self.images_folder = self._resolve_images_folder(images_folder)
        
        print(f"Benchmark Configuration:")
        print(f"  Model: {self.model_path}")
        print(f"  Images: {self.images_folder}")
        print(f"  GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print()
    
    def _resolve_model_path(self, model_path):
        """Resolve model path automatically if not provided"""
        if model_path and Path(model_path).exists():
            return model_path
        
        # Auto-detect latest trained model
        runs_dir = self.config.get_run_path('train')
        if runs_dir.exists():
            train_dirs = [d for d in runs_dir.iterdir() 
                         if d.is_dir() and "plc_symbol_detector" in d.name]
            if train_dirs:
                latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
                best_model = latest_dir / "weights" / "best.pt"
                if best_model.exists():
                    return str(best_model)
        
        # Fallback to pretrained model
        pretrained_model = self.config.get_model_path('yolo11m.pt', 'pretrained')
        if pretrained_model.exists():
            return str(pretrained_model)
        
        raise FileNotFoundError("No suitable model found for benchmarking")
    
    def _resolve_images_folder(self, images_folder):
        """Resolve images folder automatically if not provided"""
        if images_folder and Path(images_folder).exists():
            return Path(images_folder)
        
        # Auto-detect from config
        data_root = Path(self.config.config['data_root'])
        images_path = data_root / "processed" / "images"
        
        if images_path.exists() and list(images_path.glob("*.png")):
            return images_path
        
        raise FileNotFoundError(f"No images found in {images_path}")
    
    def benchmark_original_gpu_pipeline(self, batch_sizes=[8, 16, 32, 64]):
        """Benchmark the original GPU pipeline implementation"""
        print("=" * 60)
        print("BENCHMARKING ORIGINAL GPU PIPELINE")
        print("=" * 60)
        
        try:
            from src.detection.detect_pipeline_gpu_optimized import GPUOptimizedPLCDetectionPipeline
            
            results = {}
            
            for batch_size in batch_sizes:
                print(f"\nTesting original pipeline with batch_size={batch_size}")
                
                try:
                    # Initialize original pipeline
                    pipeline = GPUOptimizedPLCDetectionPipeline(
                        model_path=self.model_path,
                        batch_size=batch_size,
                        num_workers=4,
                        use_amp=True
                    )
                    
                    # Run detection
                    start_time = time.time()
                    detection_results = pipeline._detect_on_snippets_gpu_optimized(self.images_folder)
                    end_time = time.time()
                    
                    elapsed_time = end_time - start_time
                    image_count = len(detection_results)
                    speed = image_count / elapsed_time if elapsed_time > 0 else 0
                    
                    results[f'original_batch_{batch_size}'] = {
                        'pipeline': 'original',
                        'batch_size': batch_size,
                        'time': elapsed_time,
                        'images': image_count,
                        'speed': speed,
                        'success': True
                    }
                    
                    print(f"  Time: {elapsed_time:.2f}s")
                    print(f"  Speed: {speed:.2f} img/s")
                    
                    # Cleanup
                    del pipeline
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results[f'original_batch_{batch_size}'] = {
                        'pipeline': 'original',
                        'batch_size': batch_size,
                        'error': str(e),
                        'success': False
                    }
            
            return results
            
        except ImportError as e:
            print(f"Could not import original pipeline: {e}")
            return {}
    
    def benchmark_optimized_gpu_pipeline(self, batch_sizes=[8, 16, 32, 64]):
        """Benchmark the optimized GPU pipeline implementation"""
        print("=" * 60)
        print("BENCHMARKING OPTIMIZED GPU PIPELINE")
        print("=" * 60)
        
        try:
            from src.detection.detect_pipeline_gpu_optimized_fixed import OptimizedGPUPipeline
            
            results = {}
            
            for batch_size in batch_sizes:
                print(f"\nTesting optimized pipeline with max_batch_size={batch_size}")
                
                try:
                    # Initialize optimized pipeline
                    pipeline = OptimizedGPUPipeline(
                        model_path=self.model_path,
                        max_batch_size=batch_size,
                        num_workers=4
                    )
                    
                    # Run detection
                    start_time = time.time()
                    detection_results = pipeline._detect_on_snippets_optimized(self.images_folder)
                    end_time = time.time()
                    
                    elapsed_time = end_time - start_time
                    image_count = len(detection_results)
                    speed = image_count / elapsed_time if elapsed_time > 0 else 0
                    
                    results[f'optimized_batch_{batch_size}'] = {
                        'pipeline': 'optimized',
                        'requested_batch_size': batch_size,
                        'actual_batch_size': pipeline.optimal_batch_size,
                        'time': elapsed_time,
                        'images': image_count,
                        'speed': speed,
                        'success': True
                    }
                    
                    print(f"  Actual batch size: {pipeline.optimal_batch_size}")
                    print(f"  Time: {elapsed_time:.2f}s")
                    print(f"  Speed: {speed:.2f} img/s")
                    
                    # Cleanup
                    del pipeline
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results[f'optimized_batch_{batch_size}'] = {
                        'pipeline': 'optimized',
                        'requested_batch_size': batch_size,
                        'error': str(e),
                        'success': False
                    }
            
            return results
            
        except ImportError as e:
            print(f"Could not import optimized pipeline: {e}")
            return {}
    
    def benchmark_baseline_pipelines(self):
        """Benchmark baseline pipelines for comparison"""
        print("=" * 60)
        print("BENCHMARKING BASELINE PIPELINES")
        print("=" * 60)
        
        results = {}
        
        # Test sequential pipeline
        try:
            print("\nTesting sequential pipeline...")
            from src.detection.detect_pipeline import PLCDetectionPipeline
            
            pipeline = PLCDetectionPipeline(
                model_path=self.model_path,
                confidence_threshold=0.25
            )
            
            start_time = time.time()
            # Note: This would need adaptation to work with just images folder
            # For now, we'll simulate or skip this test
            print("  Skipping sequential test (requires full pipeline setup)")
            
        except Exception as e:
            print(f"  Sequential pipeline error: {e}")
        
        # Test parallel pipeline
        try:
            print("\nTesting parallel pipeline...")
            from src.detection.detect_pipeline_parallel import ParallelPLCDetectionPipeline
            
            pipeline = ParallelPLCDetectionPipeline(
                model_path=self.model_path,
                batch_size=32,
                num_workers=4
            )
            
            print("  Skipping parallel test (requires full pipeline setup)")
            
        except Exception as e:
            print(f"  Parallel pipeline error: {e}")
        
        return results
    
    def run_comprehensive_benchmark(self, batch_sizes=[8, 16, 32, 64]):
        """Run comprehensive benchmark comparing all implementations"""
        print("COMPREHENSIVE GPU PIPELINE BENCHMARK")
        print("=" * 80)
        
        all_results = {}
        
        # Benchmark original implementation
        original_results = self.benchmark_original_gpu_pipeline(batch_sizes)
        all_results.update(original_results)
        
        # Benchmark optimized implementation
        optimized_results = self.benchmark_optimized_gpu_pipeline(batch_sizes)
        all_results.update(optimized_results)
        
        # Benchmark baseline pipelines
        baseline_results = self.benchmark_baseline_pipelines()
        all_results.update(baseline_results)
        
        return all_results
    
    def analyze_results(self, results: Dict):
        """Analyze and compare benchmark results"""
        print("\n" + "=" * 80)
        print("BENCHMARK ANALYSIS")
        print("=" * 80)
        
        # Separate successful results
        successful_results = {k: v for k, v in results.items() if v.get('success', False)}
        
        if not successful_results:
            print("No successful benchmark results to analyze")
            return
        
        # Create comparison table
        comparison_data = []
        for name, result in successful_results.items():
            comparison_data.append({
                'Test': name,
                'Pipeline': result.get('pipeline', 'unknown'),
                'Batch Size': result.get('actual_batch_size', result.get('batch_size', 'N/A')),
                'Time (s)': result.get('time', 0),
                'Images': result.get('images', 0),
                'Speed (img/s)': result.get('speed', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        print("\nPerformance Comparison:")
        print(df.to_string(index=False, float_format='%.2f'))
        
        # Calculate improvements
        print("\n" + "=" * 60)
        print("PERFORMANCE IMPROVEMENTS")
        print("=" * 60)
        
        # Compare optimized vs original for each batch size
        for batch_size in [8, 16, 32, 64]:
            original_key = f'original_batch_{batch_size}'
            optimized_key = f'optimized_batch_{batch_size}'
            
            if original_key in successful_results and optimized_key in successful_results:
                original_speed = successful_results[original_key]['speed']
                optimized_speed = successful_results[optimized_key]['speed']
                
                if original_speed > 0:
                    improvement = (optimized_speed / original_speed - 1) * 100
                    speedup = optimized_speed / original_speed
                    
                    print(f"\nBatch Size {batch_size}:")
                    print(f"  Original:  {original_speed:.2f} img/s")
                    print(f"  Optimized: {optimized_speed:.2f} img/s")
                    print(f"  Improvement: {improvement:+.1f}% ({speedup:.2f}x speedup)")
        
        # Find best performing configuration
        best_result = max(successful_results.values(), key=lambda x: x.get('speed', 0))
        best_name = [k for k, v in successful_results.items() if v == best_result][0]
        
        print(f"\n" + "=" * 60)
        print("BEST PERFORMANCE")
        print("=" * 60)
        print(f"Configuration: {best_name}")
        print(f"Pipeline: {best_result['pipeline']}")
        print(f"Speed: {best_result['speed']:.2f} img/s")
        print(f"Time: {best_result['time']:.2f}s for {best_result['images']} images")
        
        return df
    
    def save_results(self, results: Dict, output_file: str = None):
        """Save benchmark results to file"""
        if output_file is None:
            output_file = f"gpu_benchmark_results_{int(time.time())}.json"
        
        output_path = Path(output_file)
        
        # Add metadata
        results_with_metadata = {
            'metadata': {
                'timestamp': time.time(),
                'model_path': str(self.model_path),
                'images_folder': str(self.images_folder),
                'gpu_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None
            },
            'results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        return output_path
    
    def create_performance_plot(self, results: Dict, output_file: str = None):
        """Create performance comparison plots"""
        try:
            import matplotlib.pyplot as plt
            
            # Filter successful results
            successful_results = {k: v for k, v in results.items() if v.get('success', False)}
            
            if not successful_results:
                print("No successful results to plot")
                return
            
            # Prepare data for plotting
            original_data = []
            optimized_data = []
            batch_sizes = []
            
            # Only include batch sizes that have data for both pipelines
            for batch_size in [8, 16, 32, 64]:
                original_key = f'original_batch_{batch_size}'
                optimized_key = f'optimized_batch_{batch_size}'
                
                if original_key in successful_results and optimized_key in successful_results:
                    original_data.append(successful_results[original_key]['speed'])
                    optimized_data.append(successful_results[optimized_key]['speed'])
                    batch_sizes.append(batch_size)
            
            # If no matching data, skip plotting
            if not batch_sizes:
                print("No matching data for both pipelines, skipping plot")
                return None
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Performance comparison
            x = range(len(batch_sizes))
            width = 0.35
            
            ax1.bar([i - width/2 for i in x], original_data, width, label='Original GPU Pipeline', alpha=0.8)
            ax1.bar([i + width/2 for i in x], optimized_data, width, label='Optimized GPU Pipeline', alpha=0.8)
            
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Speed (images/second)')
            ax1.set_title('GPU Pipeline Performance Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(batch_sizes)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Speedup ratio
            speedup_ratios = []
            for orig, opt in zip(original_data, optimized_data):
                if orig > 0:
                    speedup_ratios.append(opt / orig)
                else:
                    speedup_ratios.append(0)
            
            ax2.bar(x, speedup_ratios, alpha=0.8, color='green')
            ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No improvement')
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Speedup Ratio (Optimized/Original)')
            ax2.set_title('Performance Improvement Ratio')
            ax2.set_xticks(x)
            ax2.set_xticklabels(batch_sizes)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_file is None:
                output_file = f"gpu_benchmark_plot_{int(time.time())}.png"
            
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Performance plot saved to: {output_file}")
            
            return output_file
            
        except ImportError:
            print("Matplotlib not available, skipping plot generation")
            return None


def main():
    parser = argparse.ArgumentParser(description='GPU Pipeline Benchmark Comparison')
    parser.add_argument('--model', '-m', default=None,
                       help='Path to YOLO model (default: auto-detect)')
    parser.add_argument('--images', '-i', default=None,
                       help='Path to images folder (default: auto-detect)')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[8, 16, 32, 64],
                       help='Batch sizes to test (default: 8 16 32 64)')
    parser.add_argument('--output', '-o', default=None,
                       help='Output file for results (default: auto-generated)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate performance plots')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with fewer batch sizes')
    
    args = parser.parse_args()
    
    if args.quick:
        args.batch_sizes = [16, 32]
    
    try:
        # Initialize benchmark
        benchmark = GPUPipelineBenchmark(
            model_path=args.model,
            images_folder=args.images
        )
        
        # Run comprehensive benchmark
        results = benchmark.run_comprehensive_benchmark(args.batch_sizes)
        
        # Analyze results
        df = benchmark.analyze_results(results)
        
        # Save results
        output_file = benchmark.save_results(results, args.output)
        
        # Create plots if requested
        if args.plot:
            plot_file = benchmark.create_performance_plot(results)
        
        print(f"\nBenchmark completed successfully!")
        print(f"Results saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
