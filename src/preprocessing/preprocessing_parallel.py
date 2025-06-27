"""
Parallel PDF to Image Preprocessing with Multi-level Parallelism
Optimized for speed using multiprocessing and efficient memory management
"""

import os
import json
import platform
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import multiprocessing as mp
from multiprocessing import Pool, Manager, Queue
from functools import partial
import time
from tqdm import tqdm
import queue
import threading

# Try to import WSL wrapper
try:
    from .pdf_to_image_wsl import convert_from_path_wsl, test_wsl_poppler
except ImportError:
    try:
        from pdf_to_image_wsl import convert_from_path_wsl, test_wsl_poppler
    except ImportError:
        convert_from_path_wsl = None
        test_wsl_poppler = None


def find_poppler_path():
    """
    Determine the appropriate poppler path based on the platform.
    """
    # Check environment variable first
    env_path = os.environ.get("POPPLER_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    if platform.system() == "Windows":
        # Look for poppler in project root/bin/poppler
        project_root = Path(__file__).resolve().parent.parent.parent
        poppler_locations = [
            project_root / "bin" / "poppler" / "Library" / "bin",
            project_root / "bin" / "poppler"
        ]
        
        for location in poppler_locations:
            if location.exists():
                if ((location / "pdftoppm.exe").exists() or 
                    (location / "pdftoppm").exists()):
                    return str(location)
        return None
    else:
        return None


def process_single_pdf_worker(args):
    """
    Worker function to process a single PDF file.
    This function is designed to be picklable for multiprocessing.
    """
    pdf_path, output_folder, snippet_size, overlap, poppler_path, use_wsl = args
    
    try:
        # Convert PDF to images
        if use_wsl and platform.system() == "Windows":
            try:
                images = convert_from_path_wsl(str(pdf_path))
            except Exception as e:
                # Fallback to standard method
                images = convert_from_path(str(pdf_path), poppler_path=poppler_path)
        else:
            images = convert_from_path(str(pdf_path), poppler_path=poppler_path)
        
        if not images:
            return None, f"No images extracted from {pdf_path}"
        
        base_name = Path(pdf_path).stem
        metadata = {
            "original_pdf": base_name,
            "snippet_size": snippet_size,
            "overlap": overlap,
            "pages": []
        }
        
        # Process each page
        for page_num, image in enumerate(images, start=1):
            page_metadata = process_single_page(
                image, page_num, base_name, output_folder, snippet_size, overlap
            )
            metadata["pages"].append(page_metadata)
        
        # Save metadata
        metadata_path = output_folder / f"{base_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return metadata, None
        
    except Exception as e:
        return None, f"Error processing {pdf_path}: {str(e)}"


def process_single_page(image, page_num, base_name, output_folder, snippet_size, overlap):
    """
    Process a single page into snippets.
    """
    # Convert PIL image to OpenCV format
    img = np.array(image.convert("RGB"))
    height, width = img.shape[:2]
    snip_w, snip_h = snippet_size
    
    # Step size is reduced by overlap
    step_w = snip_w - overlap
    step_h = snip_h - overlap
    
    # Calculate grid dimensions
    cols = max(1, (width - overlap) // step_w)
    rows = max(1, (height - overlap) // step_h)
    
    if width > cols * step_w:
        cols += 1
    if height > rows * step_h:
        rows += 1
    
    page_data = {
        "page_num": page_num,
        "original_width": width,
        "original_height": height,
        "rows": rows,
        "cols": cols,
        "snippets": []
    }
    
    # Process snippets
    for row in range(rows):
        for col in range(cols):
            # Compute snip coordinates
            x1 = min(col * step_w, width - snip_w)
            y1 = min(row * step_h, height - snip_h)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(x1 + snip_w, width)
            y2 = min(y1 + snip_h, height)
            
            snippet = img[y1:y2, x1:x2]
            s_h, s_w = snippet.shape[:2]
            
            snippet_name = f"{base_name}_p{page_num}_r{row}_c{col}.png"
            snippet_path = output_folder / snippet_name
            cv2.imwrite(str(snippet_path), snippet)
            
            # Store metadata
            snippet_info = {
                "filename": snippet_name,
                "row": row,
                "col": col,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "width": int(s_w),
                "height": int(s_h)
            }
            
            page_data["snippets"].append(snippet_info)
    
    return page_data


class ParallelPDFProcessor:
    """
    Parallel PDF processor with progress tracking and resource management.
    """
    
    def __init__(self, num_workers=None, snippet_size=(1500, 1200), overlap=500):
        """
        Initialize the parallel PDF processor.
        
        Args:
            num_workers: Number of parallel workers (None for auto)
            snippet_size: Size of image snippets
            overlap: Overlap between snippets
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.snippet_size = snippet_size
        self.overlap = overlap
        self.poppler_path = find_poppler_path()
        self.use_wsl = (platform.system() == "Windows" and 
                       test_wsl_poppler and test_wsl_poppler())
        
        if self.use_wsl:
            print("WSL poppler detected and will be used for PDF conversion")
        elif self.poppler_path:
            print(f"Using poppler from: {self.poppler_path}")
        else:
            print("Using system poppler")
    
    def process_pdf_folder(self, input_folder, output_folder, show_progress=True):
        """
        Process all PDFs in a folder using parallel processing.
        
        Args:
            input_folder: Folder containing PDFs
            output_folder: Output folder for images
            show_progress: Show progress bar
            
        Returns:
            dict: Combined metadata for all processed PDFs
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Find all PDF files
        pdf_files = list(input_folder.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {input_folder}")
            return {}
        
        print(f"Found {len(pdf_files)} PDFs to process")
        print(f"Using {self.num_workers} parallel workers")
        
        # Prepare arguments for workers
        worker_args = [
            (pdf_path, output_folder, self.snippet_size, self.overlap, 
             self.poppler_path, self.use_wsl)
            for pdf_path in pdf_files
        ]
        
        # Process PDFs in parallel
        all_metadata = {}
        errors = []
        
        # Use multiprocessing Pool for true parallelism
        with Pool(processes=self.num_workers) as pool:
            # Create async results
            if show_progress:
                # Use imap for progress tracking
                results = list(tqdm(
                    pool.imap(process_single_pdf_worker, worker_args),
                    total=len(pdf_files),
                    desc="Processing PDFs"
                ))
            else:
                results = pool.map(process_single_pdf_worker, worker_args)
        
        # Collect results
        for pdf_file, (metadata, error) in zip(pdf_files, results):
            if error:
                errors.append(error)
                print(f"X Error: {error}")
                failed_files.append(pdf_file.name)
            elif metadata:
                all_metadata[metadata["original_pdf"]] = metadata
                print(f"V Processed: {pdf_file.name}")
                processed_files.append(pdf_file.name)
        
        # Save combined metadata
        if all_metadata:
            combined_meta_path = output_folder / "all_pdfs_metadata.json"
            with open(combined_meta_path, "w") as f:
                json.dump(all_metadata, f, indent=2)
            print(f"\nCombined metadata saved to: {combined_meta_path}")
        
        # Report summary
        print(f"\nProcessing complete:")
        print(f"  - Successful: {len(all_metadata)}/{len(pdf_files)}")
        if errors:
            print(f"  - Errors: {len(errors)}")
        
        return all_metadata


class StreamingPDFProcessor:
    """
    Advanced processor that can start detection while still preprocessing.
    Uses a producer-consumer pattern for maximum efficiency.
    """
    
    def __init__(self, num_pdf_workers=4, snippet_size=(1500, 1200), overlap=500):
        """
        Initialize the streaming processor.
        """
        self.num_pdf_workers = num_pdf_workers
        self.snippet_size = snippet_size
        self.overlap = overlap
        self.poppler_path = find_poppler_path()
        self.use_wsl = (platform.system() == "Windows" and 
                       test_wsl_poppler and test_wsl_poppler())
        
        # Queue for produced images
        self.image_queue = Queue(maxsize=100)  # Limit memory usage
        self.metadata_dict = {}
        self.stop_signal = threading.Event()
    
    def process_pdfs_streaming(self, pdf_files, output_folder):
        """
        Process PDFs in streaming mode, allowing consumers to start
        processing images as soon as they're ready.
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Start producer thread
        producer_thread = threading.Thread(
            target=self._produce_images,
            args=(pdf_files, output_folder)
        )
        producer_thread.start()
        
        return producer_thread
    
    def _produce_images(self, pdf_files, output_folder):
        """
        Producer thread that processes PDFs and puts images in queue.
        """
        worker_args = [
            (pdf_path, output_folder, self.snippet_size, self.overlap, 
             self.poppler_path, self.use_wsl)
            for pdf_path in pdf_files
        ]
        
        with Pool(processes=self.num_pdf_workers) as pool:
            for pdf_file, (metadata, error) in zip(
                pdf_files, 
                pool.imap(process_single_pdf_worker, worker_args)
            ):
                if metadata:
                    self.metadata_dict[metadata["original_pdf"]] = metadata
                    
                    # Put image paths in queue for detection
                    for page in metadata["pages"]:
                        for snippet in page["snippets"]:
                            image_path = output_folder / snippet["filename"]
                            self.image_queue.put(image_path)
        
        # Signal completion
        self.stop_signal.set()
    
    def get_image_batch(self, batch_size, timeout=1.0):
        """
        Get a batch of images from the queue.
        """
        batch = []
        deadline = time.time() + timeout
        
        while len(batch) < batch_size and time.time() < deadline:
            try:
                remaining_time = deadline - time.time()
                if remaining_time > 0:
                    image_path = self.image_queue.get(timeout=remaining_time)
                    batch.append(image_path)
            except queue.Empty:
                if self.stop_signal.is_set():
                    break
        
        return batch


def main():
    """
    Standalone parallel preprocessing script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Parallel PDF to Image Preprocessing')
    parser.add_argument('--input', '-i', type=str, default=None,
                       help='Input folder containing PDFs')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output folder for images')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--snippet-size', nargs=2, type=int, default=[1500, 1200],
                       help='Snippet size as width height (default: 1500 1200)')
    parser.add_argument('--overlap', type=int, default=500,
                       help='Overlap between snippets (default: 500)')
    
    args = parser.parse_args()
    
    # Get paths from config if not provided
    if args.input is None or args.output is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        import sys
        sys.path.append(str(project_root))
        
        try:
            from src.config import get_config
            config = get_config()
            data_root = Path(config.config['data_root'])
            
            if args.input is None:
                args.input = data_root / "raw" / "pdfs"
            if args.output is None:
                args.output = data_root / "processed" / "images"
        except:
            print("Error: Could not load config. Please specify --input and --output")
            return 1
    
    # Create processor
    processor = ParallelPDFProcessor(
        num_workers=args.workers,
        snippet_size=tuple(args.snippet_size),
        overlap=args.overlap
    )
    
    # Process PDFs
    start_time = time.time()
    processor.process_pdf_folder(args.input, args.output)
    elapsed_time = time.time() - start_time
    
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")
    
    return 0


if __name__ == "__main__":
    exit(main())
