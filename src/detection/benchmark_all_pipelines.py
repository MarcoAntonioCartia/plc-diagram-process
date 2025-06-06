"""
Comprehensive benchmark script to compare all pipeline implementations
"""

import time
import argparse
from pathlib import Path
import sys
import json
from tabulate import tabulate

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.detection.detect_pipeline import PLCDetectionPipeline
from src.detection.detect_pipeline_parallel import ParallelPLCDetectionPipeline
from src.detection.unified_parallel_pipeline import UnifiedParallelPipeline
from src.preprocessing.preprocessing_parallel import ParallelPDFProcessor
from src.preprocessing.SnipPdfToPng import process_pdf_folder, find_poppler_path
from src.config import get_config


def benchmark_preprocessing(input_folder, output_folder, snippet_size, overlap, num_workers=None):
    """Benchmark different preprocessing methods."""
    results = {}
    
    # Clear output folder for fair comparison
    import shutil
    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Test 1: Sequential preprocessing
    print("\n" + "="*60)
    print("BENCHMARKING SEQUENTIAL PREPROCESSING")
    print("="*60)
    
    start_time = time.time()
    poppler_path = find_poppler_path()
    process_pdf_folder(input_folder, output_folder, snippet_size, overlap, poppler_path)
    seq_time = time.time() - start_time
    
    # Count results
    image_count = len(list(output_folder.glob("*.png")))
    pdf_count = len(list(input_folder.glob("*.pdf")))
    
    results['sequential'] = {
        'time': seq_time,
        'pdfs': pdf_count,
        'images': image_count,
        'speed': pdf_count / seq_time if seq_time > 0 else 0
    }
    
    print(f"Sequential preprocessing: {seq_time:.2f}s ({pdf_count} PDFs, {image_count} images)")
    
    # Clear for next test
    shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Test 2: Parallel preprocessing
    print("\n" + "="*60)
    print(f"BENCHMARKING PARALLEL PREPROCESSING ({num_workers} workers)")
    print("="*60)
    
    processor = ParallelPDFProcessor(
        num_workers=num_workers,
        snippet_size=snippet_size,
        overlap=overlap
    )
    
    start_time = time.time()
    processor.process_pdf_folder(input_folder, output_folder, show_progress=True)
    par_time = time.time() - start_time
    
    results['parallel'] = {
        'time': par_time,
        'pdfs': pdf_count,
        'images': image_count,
        'speed': pdf_count / par_time if par_time > 0 else 0,
        'speedup': seq_time / par_time if par_time > 0 else 0
    }
    
    print(f"Parallel preprocessing: {par_time:.2f}s (speedup: {results['parallel']['speedup']:.2f}x)")
    
    return results


def benchmark_detection_pipelines(diagrams_folder, output_folder, model_path, 
                                 snippet_size, overlap, batch_size=32, num_workers=4):
    """Benchmark all detection pipeline implementations."""
    results = {}
    
    # Ensure images exist
    images_folder = output_folder / "images"
    if not images_folder.exists() or len(list(images_folder.glob("*.png"))) == 0:
        print("No images found. Running preprocessing first...")
        processor = ParallelPDFProcessor(
            num_workers=num_workers,
            snippet_size=snippet_size,
            overlap=overlap
        )
        processor.process_pdf_folder(diagrams_folder, images_folder, show_progress=True)
    
    image_count = len(list(images_folder.glob("*.png")))
    print(f"\nFound {image_count} images to process")
    
    # Test 1: Sequential pipeline
    print("\n" + "="*60)
    print("BENCHMARKING SEQUENTIAL DETECTION PIPELINE")
    print("="*60)
    
    pipeline = PLCDetectionPipeline(
        model_path=model_path,
        confidence_threshold=0.25
    )
    
    start_time = time.time()
    pipeline.process_pdf_folder(
        diagrams_folder=diagrams_folder,
        output_folder=output_folder,
        snippet_size=snippet_size,
        overlap=overlap,
        skip_pdf_conversion=True
    )
    seq_time = time.time() - start_time
    
    results['sequential'] = {
        'time': seq_time,
        'images': image_count,
        'speed': image_count / seq_time if seq_time > 0 else 0
    }
    
    print(f"Sequential detection: {seq_time:.2f}s ({results['sequential']['speed']:.2f} img/s)")
    
    # Test 2: Parallel pipeline
    print("\n" + "="*60)
    print(f"BENCHMARKING PARALLEL DETECTION PIPELINE (batch={batch_size})")
    print("="*60)
    
    pipeline = ParallelPLCDetectionPipeline(
        model_path=model_path,
        confidence_threshold=0.25,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    start_time = time.time()
    pipeline.process_pdf_folder(
        diagrams_folder=diagrams_folder,
        output_folder=output_folder,
        snippet_size=snippet_size,
        overlap=overlap,
        skip_pdf_conversion=True
    )
    par_time = time.time() - start_time
    
    results['parallel'] = {
        'time': par_time,
        'images': image_count,
        'speed': image_count / par_time if par_time > 0 else 0,
        'speedup': seq_time / par_time if par_time > 0 else 0
    }
    
    print(f"Parallel detection: {par_time:.2f}s (speedup: {results['parallel']['speedup']:.2f}x)")
    
    # Test 3: Unified pipeline
    print("\n" + "="*60)
    print("BENCHMARKING UNIFIED PARALLEL PIPELINE")
    print("="*60)
    
    pipeline = UnifiedParallelPipeline(
        model_path=model_path,
        confidence_threshold=0.25,
        batch_size=batch_size,
        pdf_workers=num_workers,
        detection_workers=2,
        streaming_mode=False
    )
    
    start_time = time.time()
    pipeline.process_pdf_folder(
        diagrams_folder=diagrams_folder,
        output_folder=output_folder,
        snippet_size=snippet_size,
        overlap=overlap,
        skip_pdf_conversion=True
    )
    unified_time = time.time() - start_time
    
    results['unified'] = {
        'time': unified_time,
        'images': image_count,
        'speed': image_count / unified_time if unified_time > 0 else 0,
        'speedup': seq_time / unified_time if unified_time > 0 else 0
    }
    
    print(f"Unified pipeline: {unified_time:.2f}s (speedup: {results['unified']['speedup']:.2f}x)")
    
    return results


def benchmark_end_to_end(diagrams_folder, output_folder, model_path, 
                        snippet_size, overlap, batch_size=32, num_workers=4):
    """Benchmark complete end-to-end pipeline."""
    results = {}
    
    # Clear output folder
    import shutil
    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    pdf_count = len(list(diagrams_folder.glob("*.pdf")))
    
    # Test 1: Sequential end-to-end
    print("\n" + "="*60)
    print("BENCHMARKING SEQUENTIAL END-TO-END PIPELINE")
    print("="*60)
    
    pipeline = PLCDetectionPipeline(
        model_path=model_path,
        confidence_threshold=0.25
    )
    
    start_time = time.time()
    pipeline.process_pdf_folder(
        diagrams_folder=diagrams_folder,
        output_folder=output_folder,
        snippet_size=snippet_size,
        overlap=overlap,
        skip_pdf_conversion=False
    )
    seq_time = time.time() - start_time
    
    results['sequential'] = {
        'time': seq_time,
        'pdfs': pdf_count,
        'speed': pdf_count / seq_time if seq_time > 0 else 0
    }
    
    print(f"Sequential end-to-end: {seq_time:.2f}s ({results['sequential']['speed']:.2f} PDFs/s)")
    
    # Clear for next test
    shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Test 2: Unified pipeline
    print("\n" + "="*60)
    print("BENCHMARKING UNIFIED END-TO-END PIPELINE")
    print("="*60)
    
    pipeline = UnifiedParallelPipeline(
        model_path=model_path,
        confidence_threshold=0.25,
        batch_size=batch_size,
        pdf_workers=num_workers,
        detection_workers=2,
        streaming_mode=False
    )
    
    start_time = time.time()
    pipeline.process_pdf_folder(
        diagrams_folder=diagrams_folder,
        output_folder=output_folder,
        snippet_size=snippet_size,
        overlap=overlap,
        skip_pdf_conversion=False
    )
    unified_time = time.time() - start_time
    
    results['unified'] = {
        'time': unified_time,
        'pdfs': pdf_count,
        'speed': pdf_count / unified_time if unified_time > 0 else 0,
        'speedup': seq_time / unified_time if unified_time > 0 else 0
    }
    
    print(f"Unified end-to-end: {unified_time:.2f}s (speedup: {results['unified']['speedup']:.2f}x)")
    
    # Clear for next test
    shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Test 3: Streaming pipeline
    print("\n" + "="*60)
    print("BENCHMARKING STREAMING END-TO-END PIPELINE")
    print("="*60)
    
    pipeline = UnifiedParallelPipeline(
        model_path=model_path,
        confidence_threshold=0.25,
        batch_size=batch_size,
        pdf_workers=num_workers,
        detection_workers=2,
        streaming_mode=True
    )
    
    start_time = time.time()
    pipeline.process_pdf_folder(
        diagrams_folder=diagrams_folder,
        output_folder=output_folder,
        snippet_size=snippet_size,
        overlap=overlap,
        skip_pdf_conversion=False
    )
    streaming_time = time.time() - start_time
    
    results['streaming'] = {
        'time': streaming_time,
        'pdfs': pdf_count,
        'speed': pdf_count / streaming_time if streaming_time > 0 else 0,
        'speedup': seq_time / streaming_time if streaming_time > 0 else 0
    }
    
    print(f"Streaming end-to-end: {streaming_time:.2f}s (speedup: {results['streaming']['speedup']:.2f}x)")
    
    return results


def print_summary(all_results):
    """Print a comprehensive summary of all benchmarks."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Preprocessing results
    if 'preprocessing' in all_results:
        print("\nPREPROCESSING PERFORMANCE:")
        data = []
        for method, results in all_results['preprocessing'].items():
            row = [
                method.capitalize(),
                f"{results['time']:.2f}s",
                f"{results['speed']:.2f} PDFs/s",
                f"{results.get('speedup', 1.0):.2f}x"
            ]
            data.append(row)
        
        headers = ["Method", "Time", "Speed", "Speedup"]
        print(tabulate(data, headers=headers, tablefmt="grid"))
    
    # Detection results
    if 'detection' in all_results:
        print("\nDETECTION PERFORMANCE:")
        data = []
        for method, results in all_results['detection'].items():
            row = [
                method.capitalize(),
                f"{results['time']:.2f}s",
                f"{results['speed']:.2f} img/s",
                f"{results.get('speedup', 1.0):.2f}x"
            ]
            data.append(row)
        
        headers = ["Method", "Time", "Speed", "Speedup"]
        print(tabulate(data, headers=headers, tablefmt="grid"))
    
    # End-to-end results
    if 'end_to_end' in all_results:
        print("\nEND-TO-END PERFORMANCE:")
        data = []
        for method, results in all_results['end_to_end'].items():
            row = [
                method.capitalize(),
                f"{results['time']:.2f}s",
                f"{results['speed']:.2f} PDFs/s",
                f"{results.get('speedup', 1.0):.2f}x"
            ]
            data.append(row)
        
        headers = ["Method", "Time", "Speed", "Speedup"]
        print(tabulate(data, headers=headers, tablefmt="grid"))
    
    # Save results to JSON
    results_file = Path("benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark all pipeline implementations')
    parser.add_argument('--diagrams', '-d', type=str, default=None,
                       help='Folder containing PDF diagrams')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output folder for results')
    parser.add_argument('--model', '-m', type=str, default=None,
                       help='Path to YOLO model (default: auto-detect)')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size for GPU inference (default: 32)')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--snippet-size', nargs=2, type=int, default=[1500, 1200],
                       help='Snippet size as width height (default: 1500 1200)')
    parser.add_argument('--overlap', type=int, default=500,
                       help='Overlap between snippets (default: 500)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing benchmark')
    parser.add_argument('--skip-detection', action='store_true',
                       help='Skip detection benchmark')
    parser.add_argument('--skip-end-to-end', action='store_true',
                       help='Skip end-to-end benchmark')
    
    args = parser.parse_args()
    
    # Get config
    config = get_config()
    data_root = Path(config.config['data_root'])
    
    # Set defaults from config
    if args.diagrams is None:
        args.diagrams = data_root / "raw" / "pdfs"
    else:
        args.diagrams = Path(args.diagrams)
        
    if args.output is None:
        args.output = data_root / "processed"
    else:
        args.output = Path(args.output)
    
    # Find model
    if args.model is None:
        runs_dir = config.get_run_path('train')
        if runs_dir.exists():
            train_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "plc_symbol_detector" in d.name]
            if train_dirs:
                latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
                args.model = latest_dir / "weights" / "best.pt"
                print(f"Using model: {args.model}")
            else:
                print("Error: No trained model found")
                return 1
        else:
            print("Error: No trained model found")
            return 1
    else:
        args.model = Path(args.model)
    
    # Run benchmarks
    all_results = {}
    
    try:
        # Preprocessing benchmark
        if not args.skip_preprocessing:
            print("\n" + "#"*80)
            print("# PREPROCESSING BENCHMARKS")
            print("#"*80)
            all_results['preprocessing'] = benchmark_preprocessing(
                args.diagrams,
                args.output / "images",
                tuple(args.snippet_size),
                args.overlap,
                args.workers
            )
        
        # Detection benchmark
        if not args.skip_detection:
            print("\n" + "#"*80)
            print("# DETECTION BENCHMARKS")
            print("#"*80)
            all_results['detection'] = benchmark_detection_pipelines(
                args.diagrams,
                args.output,
                args.model,
                tuple(args.snippet_size),
                args.overlap,
                args.batch_size,
                args.workers
            )
        
        # End-to-end benchmark
        if not args.skip_end_to_end:
            print("\n" + "#"*80)
            print("# END-TO-END BENCHMARKS")
            print("#"*80)
            all_results['end_to_end'] = benchmark_end_to_end(
                args.diagrams,
                args.output,
                args.model,
                tuple(args.snippet_size),
                args.overlap,
                args.batch_size,
                args.workers
            )
        
        # Print summary
        print_summary(all_results)
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
