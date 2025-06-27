"""
Complete PLC Pipeline with Text Extraction
Runs the entire pipeline: Training → Detection → Text Extraction
"""

# Standard libs
import os
import sys
import time
import json
import argparse
from pathlib import Path

# ------------------------------------------------------------------
# OPTIONAL TORCH STUB
# ------------------------------------------------------------------
# `ultralytics` and several detection helpers hard-import torch.  When the
# user runs the pipeline with `--skip-detection`, we don't actually need
# torch, but those import statements still execute.  To avoid a hard crash
# we create a minimal stub that satisfies the attribute look-ups used during
# module import (mainly `torch.cuda.is_available`).  The real detection code
# will fail loudly later if someone tries to *use* torch without installing
# it.

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – optional dependency
    class _CudaStub:
        @staticmethod
        def is_available() -> bool:  # noqa: D401
            return False

    class _TorchStub:
        __version__ = "0.0.0-stub"
        cuda = _CudaStub()

        def __getattr__(self, name):
            # Always return a harmless no-op callable or placeholder so that
            # import-time side-effects inside external libraries don't crash.
            if name.isupper():
                return name  # dtype constant placeholders

            def _noop(*_a, **_kw):  # noqa: D401
                # Accessing most torch APIs without torch installed is an error
                # at *runtime*, but during --skip-detection startup we tolerate it.
                return None

            # Provide a minimal torch.distributed submodule when requested
            if name == "distributed":
                import types, sys as _sys
                import importlib.machinery as _mach
                dist_mod = types.ModuleType("torch.distributed")
                dist_mod.is_available = lambda: False  # type: ignore
                dist_mod.is_initialized = lambda: False  # type: ignore
                dist_mod.init_process_group = _noop  # type: ignore
                dist_mod.destroy_process_group = _noop  # type: ignore
                dist_mod.__path__ = []  # type: ignore
                dist_mod.__spec__ = _mach.ModuleSpec(name="torch.distributed", loader=None)
                _sys.modules["torch.distributed"] = dist_mod
                return dist_mod

            return _noop

    sys.modules["torch"] = _TorchStub()  # type: ignore

# ------------------------------------------------------------------
# Monkey-patch importlib.metadata.version early so any subsequent imports
# (e.g., ultralytics) that query torchvision / torchaudio versions do not
# raise PackageNotFoundError when those packages are absent.
# ------------------------------------------------------------------
import importlib.metadata as _ilm

_orig_version = _ilm.version

def _safe_version(package: str):  # noqa: D401
    """Safe wrapper around importlib.metadata.version.

    We only spoof version strings for extremely common heavy packages that we
    purposefully stub out (torch, torchvision, torchaudio). For any *other*
    package we simply defer to the original implementation and allow it to
    raise PackageNotFoundError if the package is missing.  Down-stream code
    (like PaddleXʼs dependency helper) expects that behaviour to reliably
    detect absent optional dependencies.  Swallowing *all* errors (the old
    behaviour) inadvertently convinced PaddleX that the `soundfile` package
    was installed even when it was not, leading to an import failure further
    down the line.
    """

    if package in {"torch", "torchvision", "torchaudio"}:
        # These packages are intentionally stubbed, so we pretend they exist.
        return "0.0.0-stub"

    # For every other package, use the original implementation and propagate
    # the PackageNotFoundError so callers can detect missing dependencies.
    return _orig_version(package)

_ilm.version = _safe_version  # type: ignore

# --- BEGIN GPU PATH FIX ---
# This block fixes a common issue on Windows where a system-wide CUDA installation
# conflicts with Paddle's bundled CUDA libraries. It works by finding the correct
# library paths inside the virtual environment and prepending them to the system's
# PATH, ensuring they are loaded first.
def _apply_gpu_path_fix():
    """Dynamically finds and prepends bundled GPU library paths to the system PATH."""
    try:
        if sys.platform != "win32":
            return # This fix is only for Windows

        # Find the root of the virtual environment
        venv_path = Path(sys.executable).parent.parent
        site_packages = venv_path / "Lib" / "site-packages"

        if not site_packages.is_dir():
            return

        # Define the essential paths for Paddle's bundled libraries
        bundled_paths_to_add = [
            site_packages / "nvidia" / "cuda_runtime" / "bin",
            site_packages / "nvidia" / "cudnn" / "bin",
            site_packages / "nvidia" / "cublas" / "bin",
            site_packages / "nvidia" / "cuda_nvrtc" / "bin",
            site_packages / "nvidia" / "cusparse" / "bin",
            site_packages / "nvidia" / "cusparse" / "lib",
            site_packages / "nvidia" / "cusolver" / "bin",
            site_packages / "nvidia" / "cusolver" / "lib",
            site_packages / "paddle" / "libs",
        ]
        
        found_paths = [str(p) for p in bundled_paths_to_add if p.is_dir()]

        if not found_paths:
            return

        # Prepend the found paths to the system's PATH environment variable
        original_path = os.environ.get("PATH", "")
        new_path = os.pathsep.join(found_paths) + os.pathsep + original_path
        os.environ["PATH"] = new_path
        
        # Optional: uncomment the line below for verification during startup
        # print("GPU Path Fix: Successfully prioritized bundled CUDA libraries.")

    except Exception:
        # Silently ignore errors in the patch, as it's an enhancement
        pass

# Apply the fix at the very start of the application, before any other imports
_apply_gpu_path_fix()
# --- END GPU PATH FIX ---

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# ------------------------------------------------------------------
# Runtime flag helpers – must come *before* any heavy imports.
# ------------------------------------------------------------------

import importlib.metadata as _ilm
from src.utils.runtime_flags import skip_detection_requested, multi_env_active

_SKIP_DETECTION = skip_detection_requested()

# -------------------------------------------------------------
# Lazy / conditional import to avoid ultralytics if detection is skipped
# -------------------------------------------------------------

if _SKIP_DETECTION:
    class _DummyRunner:  # pragma: no cover
        def __init__(self, *a, **kw):
            """Light-weight stand-in for CompletePipelineRunner when detection is skipped.

            It sets up just enough state so that the text-extraction layer that
            derives from it can run without hitting attribute errors.  We avoid
            any heavyweight imports (Ultralytics / torch) and do **not** perform
            dataset validation or training.
            """

            # Project & config
            from pathlib import Path  # local import to keep top-level pristine
            from src.config import get_config  # deferred; cheap and torch-free

            self.project_root = Path(__file__).resolve().parent.parent

            # Basic config paths (match the ones in the real runner)
            cfg = get_config()
            data_root = Path(cfg.config["data_root"])
            self.diagrams_folder = data_root / "raw" / "pdfs"
            self.images_folder = data_root / "processed" / "images"
            self.detdiagrams_folder = data_root / "processed" / "detdiagrams"

            # Ensure output directories exist so later code can write to them
            self.images_folder.mkdir(parents=True, exist_ok=True)
            self.detdiagrams_folder.mkdir(parents=True, exist_ok=True)

            # Minimal results skeleton expected by downstream methods
            self.results = {
                "training": {},
                "detection": {},
                "reconstruction": {},
                "summary": {},
            }

        # The training / detection phases are intentionally unavailable in
        # skip-detection mode.
        def run_complete_pipeline(self):  # noqa: D401
            print("Detection skipped; full pipeline not available without torch.")
            return False

        def run_text_extraction_only(self):
            print("Detection skipped (dummy runner). No detection pipeline available without torch.")
            return False

    CompletePipelineRunner = _DummyRunner  # type: ignore
else:
    # Defer the import decision - we'll import the right runner later
    CompletePipelineRunner = None  # type: ignore

# ------------------------------------------------------------------
# Stub out optional heavy dependencies that PaddleOCR -> PaddleX may try
# to import (pycocotools) when those packages are not installed.
# ------------------------------------------------------------------

import types as _types, sys as _sys

if "pycocotools" not in _sys.modules:
    _pc_root = _types.ModuleType("pycocotools")
    _sys.modules["pycocotools"] = _pc_root

    _pc_coco = _types.ModuleType("pycocotools.coco")
    class _DummyCOCO:  # noqa: D401
        def __init__(self, *a, **kw):
            raise RuntimeError("pycocotools is not installed; COCO functionality unavailable.")

    _pc_coco.COCO = _DummyCOCO  # type: ignore
    _sys.modules["pycocotools.coco"] = _pc_coco

from src.ocr.text_extraction_pipeline import TextExtractionPipeline
from src.ocr.coordinate_calibration import CoordinateCalibrator
from src.utils.detection_text_extraction_pdf_creator import DetectionPDFCreator
from src.config import get_config

def get_pipeline_runner_class():
    """Get the appropriate pipeline runner class based on runtime conditions."""
    global CompletePipelineRunner
    
    # If already resolved, return it
    if CompletePipelineRunner is not None:
        return CompletePipelineRunner
    
    # Check if we should skip detection entirely
    if skip_detection_requested():
        CompletePipelineRunner = _DummyRunner
    # Check if we're in multi-env mode
    elif multi_env_active():
        from src.detection.lightweight_pipeline_runner import LightweightPipelineRunner
        CompletePipelineRunner = LightweightPipelineRunner
    else:
        # Standard mode - import the heavy runner
        from src.detection.run_complete_pipeline import CompletePipelineRunner as StandardRunner
        CompletePipelineRunner = StandardRunner
    
    return CompletePipelineRunner

class CompleteTextPipelineRunner:
    """Extended pipeline runner that includes text extraction"""
    
    def __new__(cls, *args, **kwargs):
        """Dynamically create instance with the right base class."""
        # Get the appropriate base class
        base_class = get_pipeline_runner_class()
        
        # Create a new class that inherits from the right base
        dynamic_class = type(
            'CompleteTextPipelineRunner',
            (base_class,),
            dict(cls.__dict__)
        )
        
        # Create and return instance
        instance = object.__new__(dynamic_class)
        return instance
    
    def __init__(self, epochs=10, confidence_threshold=0.25, snippet_size=(1500, 1200), 
                 overlap=500, model_name=None, device=None, ocr_confidence=0.7, ocr_lang="en",
                 pdf_confidence_threshold=0.8, create_enhanced_pdf=False, enhanced_pdf_version='short', detection_folder=None):
        """
        Initialize the complete pipeline runner with text extraction
        
        Args:
            epochs: Number of training epochs
            confidence_threshold: Detection confidence threshold
            snippet_size: Size of image snippets for PDF processing
            overlap: Overlap between snippets
            model_name: YOLO model to use (None for auto-detection)
            device: Device to use for training ('auto', 'cpu', '0', '1', etc.)
            ocr_confidence: OCR confidence threshold
            ocr_lang: OCR language
            pdf_confidence_threshold: Confidence threshold for PDF enhancement
            create_enhanced_pdf: Whether to create enhanced PDFs
            enhanced_pdf_version: 'short' (1 page) or 'long' (4 pages) for troubleshooting
            detection_folder: Custom detection folder path (None for auto-detection)
        """
        super().__init__(epochs, confidence_threshold, snippet_size, overlap, model_name, device)
        
        self.ocr_confidence = ocr_confidence
        self.ocr_lang = ocr_lang
        self.pdf_confidence_threshold = pdf_confidence_threshold
        self.create_enhanced_pdf = create_enhanced_pdf
        self.enhanced_pdf_version = enhanced_pdf_version
        self.custom_detection_folder = Path(detection_folder) if detection_folder else None
        
        # Add text extraction results tracking
        self.results["text_extraction"] = {}
        self.results["pdf_enhancement"] = {}
    
    def _get_effective_detection_folder(self):
        """Get the effective detection folder (custom or default)"""
        if self.custom_detection_folder:
            if self.custom_detection_folder.exists():
                print(f"Using custom detection folder: {self.custom_detection_folder}")
                return self.custom_detection_folder
            else:
                print(f"Warning: Custom detection folder not found: {self.custom_detection_folder}")
                print(f"Falling back to default: {self.detdiagrams_folder}")
                return self.detdiagrams_folder
        else:
            return self.detdiagrams_folder
    
    def _check_text_extraction_exists(self):
        """Check if text extraction results already exist"""
        detection_folder = self._get_effective_detection_folder()
        text_extraction_folder = detection_folder.parent / "text_extraction"
        
        if not text_extraction_folder.exists():
            return False
        
        # Check if we have text extraction files for all detection files
        detection_files = list(detection_folder.glob("*_detections.json"))
        text_files = list(text_extraction_folder.glob("*_text_extraction.json"))
        
        if len(text_files) == 0:
            return False
        
        # Check if we have text extraction for each detection file
        detection_names = {f.stem.replace("_detections", "") for f in detection_files}
        text_names = {f.stem.replace("_text_extraction", "") for f in text_files}
        
        missing_text = detection_names - text_names
        if missing_text:
            print(f"Missing text extraction for: {', '.join(missing_text)}")
            return False
        
        print(f"Found existing text extraction results for {len(text_files)} files")
        return True
    
    def run_pdf_only_mode(self):
        """Run only enhanced PDF creation (skip detection and text extraction)"""
        print("Running Enhanced PDF Creation Only (Detection and Text Extraction Skipped)")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            # Check if both detection and text extraction results exist
            detection_folder = self._get_effective_detection_folder()
            text_extraction_folder = detection_folder.parent / "text_extraction"
            
            if not detection_folder.exists():
                print(f"Error: Detection folder not found: {detection_folder}")
                return False
            
            if not text_extraction_folder.exists():
                print(f"Error: Text extraction folder not found: {text_extraction_folder}")
                return False
            
            detection_files = list(detection_folder.glob("*_detections.json"))
            text_files = list(text_extraction_folder.glob("*_text_extraction.json"))
            
            if not detection_files:
                print(f"Error: No detection files found in {detection_folder}")
                return False
            
            if not text_files:
                print(f"Error: No text extraction files found in {text_extraction_folder}")
                return False
            
            print(f"Found {len(detection_files)} detection files and {len(text_files)} text extraction files")
            
            # Initialize results for PDF-only mode
            self.results["training"] = {"training_time": 0, "epochs": "skipped"}
            self.results["detection"] = {"detection_time": 0, "status": "skipped"}
            self.results["text_extraction"] = {"extraction_time": 0, "status": "skipped"}
            
            # Create enhanced PDFs
            if self.create_enhanced_pdf:
                print(f"\nPhase 1: Enhanced PDF Creation")
                print("-" * 30)
                self._run_enhanced_pdf_creation()
            else:
                print("Warning: --create-enhanced-pdf not specified, nothing to do")
                return False
            
            # Generate summary report
            print(f"\nPhase 2: Generating Summary")
            print("-" * 30)
            self._generate_pdf_only_summary_report()
            
            total_time = time.time() - start_time
            
            print(f"\nEnhanced PDF creation finished successfully!")
            print(f"Total execution time: {total_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"\nPDF creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_pdf_only_summary_report(self):
        """Generate summary report for PDF-only mode"""
        
        # Create summary for PDF-only mode
        summary = {
            "pipeline_mode": "pdf_creation_only",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pdf_enhancement_results": self.results.get("pdf_enhancement", {}),
            "performance_metrics": {
                "pdf_enhancement_time": self.results.get("pdf_enhancement", {}).get("enhancement_time", 0),
                "total_processing_time": self.results.get("pdf_enhancement", {}).get("enhancement_time", 0)
            }
        }
        
        # Save summary report
        summary_file = self.detdiagrams_folder / "pdf_creation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\nPDF Creation Pipeline Summary:")
        print("-" * 50)
        if "pdf_enhancement" in self.results and self.results["pdf_enhancement"]:
            print(f"PDF enhancement time: {self.results['pdf_enhancement']['enhancement_time']:.2f}s")
            print(f"Enhanced PDFs created: {self.results['pdf_enhancement']['processed_files']}")
            print(f"Success rate: {self.results['pdf_enhancement']['success_rate']:.1f}%")
        
        print(f"Summary saved to: {summary_file}")
        
        self.results["summary"] = summary
    
    def _filter_detections_by_confidence(self, detection_file: Path, min_confidence: float) -> dict:
        """Filter detection JSON file by confidence threshold"""
        try:
            with open(detection_file, 'r') as f:
                detection_data = json.load(f)
            
            original_count = 0
            filtered_count = 0
            
            # Filter detections in each page
            for page_data in detection_data.get("pages", []):
                original_detections = page_data.get("detections", [])
                original_count += len(original_detections)
                
                # Filter by confidence
                filtered_detections = [
                    det for det in original_detections 
                    if det.get("confidence", 0.0) >= min_confidence
                ]
                
                page_data["detections"] = filtered_detections
                filtered_count += len(filtered_detections)
            
            print(f"  Confidence filtering: {original_count} → {filtered_count} detections (≥{min_confidence:.1f})")
            
            return detection_data
            
        except Exception as e:
            print(f"  Error filtering detections: {e}")
            # Return original data if filtering fails
            with open(detection_file, 'r') as f:
                return json.load(f)
    
    def run_complete_pipeline_with_text(self):
        """
        Run the complete pipeline including text extraction
        """
        print("Starting Complete PLC Pipeline with Text Extraction")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            # Step 1-3: Run the standard detection pipeline
            print("Phase 1: Detection Pipeline")
            print("-" * 30)
            
            detection_success = self.run_complete_pipeline()
            
            if not detection_success:
                print("Detection pipeline failed, aborting text extraction")
                return False
            
            # Step 4: Run text extraction
            print(f"\nPhase 2: Text Extraction Pipeline")
            print("-" * 30)
            text_results = self._run_text_extraction()
            
            # Step 5: Generate combined summary report
            print(f"\nPhase 3: Generating Combined Summary")
            print("-" * 30)
            self._generate_combined_summary_report()
            
            total_time = time.time() - start_time
            
            print(f"\nComplete pipeline with text extraction finished successfully!")
            print(f"Total execution time: {total_time:.2f} seconds")
            print(f"Results saved to: {self.detdiagrams_folder}")
            
            return True
            
        except Exception as e:
            print(f"\nPipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_text_extraction(self):
        """Run text extraction on detection results with custom folder and confidence filtering"""
        
        print(f"Running text extraction with OCR confidence: {self.ocr_confidence}")
        print(f"Detection confidence threshold: {self.pdf_confidence_threshold}")
        
        # Get effective detection folder (custom or default)
        detection_folder = self._get_effective_detection_folder()
        
        # Initialize text extraction pipeline
        text_pipeline = TextExtractionPipeline(
            confidence_threshold=self.ocr_confidence,
            ocr_lang=self.ocr_lang
        )
        
        # Set up paths
        pdf_folder = self.diagrams_folder
        text_output_folder = detection_folder.parent / "text_extraction"
        
        # Create a temporary folder for filtered detection files
        filtered_detection_folder = detection_folder.parent / "filtered_detections_temp"
        filtered_detection_folder.mkdir(exist_ok=True)
        
        text_start = time.time()
        
        try:
            # Filter detection files by confidence threshold
            detection_files = list(detection_folder.glob("*_detections.json"))
            print(f"Processing {len(detection_files)} detection files...")
            
            for detection_file in detection_files:
                print(f"Processing {detection_file.name}...")
                
                # Filter detections by confidence
                filtered_data = self._filter_detections_by_confidence(
                    detection_file, self.pdf_confidence_threshold
                )
                
                # Save filtered data to temporary folder
                filtered_file = filtered_detection_folder / detection_file.name
                with open(filtered_file, 'w') as f:
                    json.dump(filtered_data, f, indent=2)
            
            # Run text extraction on filtered detection files
            text_summary = text_pipeline.process_detection_folder(
                filtered_detection_folder, pdf_folder, text_output_folder
            )
            
            # Apply coordinate calibration to fix transformation errors
            print(f"Applying coordinate calibration...")
            calibrator = CoordinateCalibrator()
            
            calibration_results = []
            text_files = list(text_output_folder.glob("*_text_extraction.json"))
            
            for text_file in text_files:
                try:
                    corrected_file = text_file.parent / f"{text_file.stem}_corrected.json"
                    result = calibrator.calibrate_text_extraction_file(text_file, corrected_file)
                    calibration_results.append({
                        "file": text_file.name,
                        "correction_applied": result["correction"]["description"],
                        "status": "success"
                    })
                    
                    # Replace original with corrected version
                    import shutil
                    shutil.move(str(corrected_file), str(text_file))
                    print(f"  ✓ Calibrated {text_file.name}")
                    
                except Exception as e:
                    print(f"  ✗ Failed to calibrate {text_file.name}: {e}")
                    calibration_results.append({
                        "file": text_file.name,
                        "error": str(e),
                        "status": "failed"
                    })
            
            print(f"Coordinate calibration completed for {len(text_files)} files")
            
            text_time = time.time() - text_start
            
            # Store text extraction results
            self.results["text_extraction"] = {
                "extraction_time": text_time,
                "ocr_confidence": self.ocr_confidence,
                "ocr_language": self.ocr_lang,
                "detection_confidence_threshold": self.pdf_confidence_threshold,
                "custom_detection_folder": str(detection_folder) if self.custom_detection_folder else None,
                "processed_files": text_summary["processed_files"],
                "total_text_regions": text_summary["total_text_regions"],
                "output_folder": str(text_output_folder),
                "summary": text_summary
            }
            
            print(f"Text extraction completed in {text_time:.2f} seconds")
            print(f"Processed {text_summary['processed_files']} files")
            print(f"Extracted {text_summary['total_text_regions']} text regions")
            
        finally:
            # Clean up temporary filtered detection files
            try:
                import shutil
                shutil.rmtree(filtered_detection_folder)
                print(f"Cleaned up temporary folder: {filtered_detection_folder}")
            except Exception as e:
                print(f"Warning: Could not clean up temporary folder: {e}")
        
        return text_summary
    
    def _run_enhanced_pdf_creation(self):
        """Create enhanced PDFs with detection boxes and text extraction using original PDFs"""
        
        if not self.create_enhanced_pdf:
            return
        
        print(f"Creating enhanced PDFs from original PDFs (version: {self.enhanced_pdf_version})...")
        
        # Initialize PDF creator
        pdf_creator = DetectionPDFCreator(font_size=10, min_conf=self.pdf_confidence_threshold)
        
        # Get effective detection folder (custom or default)
        detection_folder = self._get_effective_detection_folder()
        
        # Set up paths
        text_extraction_folder = detection_folder.parent / "text_extraction"
        enhanced_pdf_folder = detection_folder.parent / "enhanced_pdfs"
        enhanced_pdf_folder.mkdir(parents=True, exist_ok=True)
        
        pdf_start = time.time()
        
        # Find detection JSON files
        detection_files = list(detection_folder.glob("*_detections.json"))
        
        if not detection_files:
            print("Warning: No detection files found")
            return {"processed_files": 0, "error": "No detection files found"}
        
        print(f"Found {len(detection_files)} detection files to process")
        
        processed_count = 0
        enhancement_results = []
        
        # Process each detection file
        for detection_file in detection_files:
            pdf_name = detection_file.name.replace("_detections.json", "")
            print(f"Processing {pdf_name}...")
            
            # Find corresponding files
            original_pdf = self.diagrams_folder / f"{pdf_name}.pdf"
            text_file = text_extraction_folder / f"{pdf_name}_text_extraction.json"
            
            if not original_pdf.exists():
                print(f"  Warning: Original PDF not found: {original_pdf}")
                continue
            
            if not text_file.exists():
                print(f"  Warning: Text file not found: {text_file}")
                continue
            
            # Create enhanced PDF
            output_file = enhanced_pdf_folder / f"{pdf_name}_enhanced.pdf"
            
            try:
                # Use original PDF instead of pre-annotated PNG
                result_pdf = pdf_creator.create_enhanced_pdf_from_original(
                    original_pdf=original_pdf,
                    detections_file=detection_file,
                    text_file=text_file,
                    output_file=output_file,
                    version=self.enhanced_pdf_version
                )
                
                enhancement_results.append({
                    "pdf_name": pdf_name,
                    "original_pdf": str(original_pdf),
                    "detection_file": str(detection_file),
                    "text_file": str(text_file),
                    "enhanced_pdf": str(result_pdf),
                    "version": self.enhanced_pdf_version,
                    "status": "success"
                })
                
                processed_count += 1
                print(f"  ✓ Created: {output_file.name} ({self.enhanced_pdf_version} version)")
                
            except Exception as e:
                print(f"  ✗ Failed to create {output_file.name}: {e}")
                enhancement_results.append({
                    "pdf_name": pdf_name,
                    "status": "failed",
                    "error": str(e)
                })
                continue
        
        pdf_time = time.time() - pdf_start
        
        # Store PDF enhancement results
        self.results["pdf_enhancement"] = {
            "enhancement_time": pdf_time,
            "method": "original_pdf",
            "version": self.enhanced_pdf_version,
            "processed_files": processed_count,
            "total_files": len(detection_files),
            "success_rate": (processed_count / len(detection_files) * 100) if detection_files else 0,
            "output_folder": str(enhanced_pdf_folder),
            "results": enhancement_results
        }
        
        print(f"Enhanced PDF creation completed in {pdf_time:.2f} seconds")
        print(f"Created {processed_count}/{len(detection_files)} enhanced PDFs")
        print(f"Success rate: {self.results['pdf_enhancement']['success_rate']:.1f}%")
        print(f"Enhanced PDFs saved to: {enhanced_pdf_folder}")
        
        return self.results["pdf_enhancement"]
    
    def run_text_extraction_only(self):
        """Run only text extraction and PDF enhancement (skip detection)"""
        print("Running Text Extraction Pipeline (Detection Skipped)")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Check if detection results exist
            if not self.detdiagrams_folder.exists():
                print(f"Error: Detection folder not found: {self.detdiagrams_folder}")
                print("Please run detection first or use a different mode.")
                return False
            
            detection_files = list(self.detdiagrams_folder.glob("*_detections.json"))
            if not detection_files:
                print(f"Error: No detection files found in {self.detdiagrams_folder}")
                return False
            
            print(f"Found {len(detection_files)} detection files")
            
            # Initialize results for text-only mode
            self.results["training"] = {"training_time": 0, "epochs": "skipped"}
            self.results["detection"] = {"detection_time": 0, "status": "skipped"}
            
            # Step 1: Run text extraction
            print(f"\nPhase 1: Text Extraction Pipeline")
            print("-" * 30)
            text_results = self._run_text_extraction()
            
            # Step 2: Create enhanced PDFs if requested
            if self.create_enhanced_pdf:
                print(f"\nPhase 2: Enhanced PDF Creation")
                print("-" * 30)
                self._run_enhanced_pdf_creation()
            
            # Step 3: Generate summary report
            print(f"\nPhase 3: Generating Summary")
            print("-" * 30)
            self._generate_text_only_summary_report()
            
            total_time = time.time() - start_time
            
            print(f"\nText extraction pipeline finished successfully!")
            print(f"Total execution time: {total_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"\nText extraction pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_text_only_summary_report(self):
        """Generate summary report for text-only mode"""
        
        # Create summary for text-only mode
        summary = {
            "pipeline_mode": "text_extraction_only",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "text_extraction_results": self.results["text_extraction"],
            "performance_metrics": {
                "text_extraction_time": self.results["text_extraction"]["extraction_time"],
                "total_processing_time": self.results["text_extraction"]["extraction_time"]
            }
        }
        
        # Add PDF enhancement results if available
        if "pdf_enhancement" in self.results and self.results["pdf_enhancement"]:
            summary["pdf_enhancement_results"] = self.results["pdf_enhancement"]
            summary["performance_metrics"]["pdf_enhancement_time"] = self.results["pdf_enhancement"]["enhancement_time"]
            summary["performance_metrics"]["total_processing_time"] += self.results["pdf_enhancement"]["enhancement_time"]
        
        # Add text extraction specific metrics
        if self.results["text_extraction"]["processed_files"] > 0:
            avg_texts_per_file = (
                self.results["text_extraction"]["total_text_regions"] / 
                self.results["text_extraction"]["processed_files"]
            )
            summary["performance_metrics"]["average_text_regions_per_file"] = avg_texts_per_file
        
        # Save summary report
        summary_file = self.detdiagrams_folder / "text_extraction_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\nText Extraction Pipeline Summary:")
        print("-" * 50)
        print(f"Text extraction time: {self.results['text_extraction']['extraction_time']:.2f}s")
        if "pdf_enhancement" in self.results and self.results["pdf_enhancement"]:
            print(f"PDF enhancement time: {self.results['pdf_enhancement']['enhancement_time']:.2f}s")
        print(f"Total time: {summary['performance_metrics']['total_processing_time']:.2f}s")
        print(f"Text regions extracted: {self.results['text_extraction']['total_text_regions']}")
        
        if self.results["text_extraction"]["processed_files"] > 0:
            avg_texts = self.results["text_extraction"]["total_text_regions"] / self.results["text_extraction"]["processed_files"]
            print(f"Avg text regions/PDF: {avg_texts:.1f}")
        
        if "pdf_enhancement" in self.results and self.results["pdf_enhancement"]:
            print(f"Enhanced PDFs created: {self.results['pdf_enhancement']['processed_files']}")
        
        print(f"Summary saved to: {summary_file}")
        
        self.results["summary"] = summary
    
    def _generate_combined_summary_report(self):
        """Generate a comprehensive summary report including text extraction"""
        
        # Get the base summary from parent class
        super()._generate_summary_report()
        
        # Enhance with text extraction data
        combined_summary = self.results["summary"].copy()
        combined_summary["text_extraction_results"] = self.results["text_extraction"]
        
        # Calculate combined metrics
        total_processing_time = (
            self.results["training"]["training_time"] + 
            self.results["detection"]["detection_time"] +
            self.results["text_extraction"]["extraction_time"]
        )
        
        combined_summary["performance_metrics"]["total_processing_time_with_text"] = total_processing_time
        
        # Add text extraction specific metrics
        if self.results["text_extraction"]["processed_files"] > 0:
            avg_texts_per_file = (
                self.results["text_extraction"]["total_text_regions"] / 
                self.results["text_extraction"]["processed_files"]
            )
            combined_summary["performance_metrics"]["average_text_regions_per_file"] = avg_texts_per_file
        
        # Save enhanced summary report
        summary_file = self.detdiagrams_folder / "complete_pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(combined_summary, f, indent=2)
        
        # Print enhanced summary
        print("\nComplete Pipeline Summary:")
        print("-" * 50)
        print(f"Training time: {self.results['training']['training_time']:.2f}s")
        print(f"Detection time: {self.results['detection']['detection_time']:.2f}s")
        print(f"Text extraction time: {self.results['text_extraction']['extraction_time']:.2f}s")
        print(f"Total time: {total_processing_time:.2f}s")
        print(f"PDFs processed: {self.results['detection']['statistics']['total_pdfs_processed']}")
        print(f"Symbols detected: {self.results['detection']['statistics']['total_detections']}")
        print(f"Text regions extracted: {self.results['text_extraction']['total_text_regions']}")
        
        if self.results["text_extraction"]["processed_files"] > 0:
            avg_texts = self.results["text_extraction"]["total_text_regions"] / self.results["text_extraction"]["processed_files"]
            print(f"Avg text regions/PDF: {avg_texts:.1f}")
        
        print(f"Complete summary saved to: {summary_file}")
        
        self.results["summary"] = combined_summary

def main():
    parser = argparse.ArgumentParser(description='Run Complete PLC Pipeline with Text Extraction')
    
    # Detection arguments (inherited from parent)
    parser.add_argument('--model', '-m', default=None,
                       help='YOLO model to use (e.g., yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)')
    parser.add_argument('--list-models', action='store_true',
                       help='List all available models and exit')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                       help='Detection confidence threshold (default: 0.25)')
    parser.add_argument('--snippet-size', nargs=2, type=int, default=[1500, 1200],
                       help='Snippet size as width height (default: 1500 1200)')
    parser.add_argument('--overlap', type=int, default=500,
                       help='Overlap between snippets (default: 500)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and use existing best model')
    parser.add_argument('--device', '-d', default='auto',
                       help='Device to use: auto, cpu, cuda, 0, 1, etc. (default: auto)')
    
    # Text extraction arguments
    parser.add_argument('--ocr-confidence', type=float, default=0.7,
                       help='OCR confidence threshold (default: 0.7)')
    parser.add_argument('--ocr-lang', type=str, default='en',
                       help='OCR language (default: en)')
    parser.add_argument('--skip-text-extraction', action='store_true',
                       help='Skip text extraction phase (use existing text extraction results)')
    parser.add_argument('--skip-detection', action='store_true',
                       help='Skip detection and only run text extraction (requires existing detection results)')
    parser.add_argument('--auto-skip-existing', action='store_true',
                       help='Automatically skip phases if results already exist')
    
    # Enhanced PDF arguments
    parser.add_argument('--create-enhanced-pdf', action='store_true',
                       help='Create enhanced PDFs with detection boxes and text extraction')
    parser.add_argument('--pdf-confidence-threshold', type=float, default=0.8,
                       help='Confidence threshold for showing detection boxes in enhanced PDFs (default: 0.8)')
    parser.add_argument('--enhanced-pdf-version', type=str, choices=['short', 'long'], 
                       default='short', help='Enhanced PDF version: short (1 page) or long (4 pages) for troubleshooting')
    parser.add_argument('--detection-folder', type=str, default=None,
                       help='Custom detection folder path (default: auto-detected based on pipeline)')
    
    # Multi-environment switch
    parser.add_argument('--mode', choices=['single', 'multi'], default='single',
                       help='Execution mode: single (default) = legacy in-process, multi = use isolated detection/ocr environments.')
    
    args = parser.parse_args()
    
    # Set multi-env flag EARLY before any class instantiation
    if args.mode == 'multi':
        os.environ["PLCDP_MULTI_ENV"] = "1"
    
    # Handle list-models command (delegate to parent)
    if args.list_models:
        from src.detection.run_complete_pipeline import main as parent_main
        return parent_main()
    
    try:
        # ------------------------------------------------------------------
        # Multi-environment short-circuit (first draft)
        # ------------------------------------------------------------------
        if args.mode == 'multi':
            from pathlib import Path as _P
            from src.utils.multi_env_manager import MultiEnvironmentManager  # noqa: WPS433 – runtime import by design

            mgr = MultiEnvironmentManager(_P(__file__).resolve().parent.parent)
            print("[Multi-Env] Ensuring detection / OCR environments …")
            if not (mgr.setup() and mgr.health_check()):
                print("❌ Multi-environment setup failed – aborting.")
                return 1

            # ------------------------------------------------------------------
            # Discover input PDFs using the same config conventions as the legacy
            # single-process runner: ${data_root}/raw/pdfs/*.pdf
            # ------------------------------------------------------------------
            from src.config import Config  # noqa: WPS433 – runtime import

            cfg = Config()
            data_root = _P(cfg.config["data_root"])
            pdf_folder = data_root / "raw" / "pdfs"
            if not pdf_folder.exists():
                print(f"❌ PDF input folder not found: {pdf_folder}")
                return 1

            out_root = data_root / "processed" / "detdiagrams"
            out_root.mkdir(parents=True, exist_ok=True)

            pdf_files = sorted(pdf_folder.glob("*.pdf"))
            if not pdf_files:
                print(f"⚠️  No PDFs found in {pdf_folder}")
                return 0  # Nothing to do, but not an error

            print(f"[Multi-Env] Processing {len(pdf_files)} PDFs …")

            try:
                from tqdm import tqdm  # type: ignore
                iterator = tqdm(pdf_files, unit="pdf")
            except ImportError:
                iterator = pdf_files  # fallback

            combined_stats = {"processed": 0, "errors": 0}

            for pdf_path in iterator:
                if isinstance(iterator, list):
                    print(f"→ {pdf_path.name}")

                per_pdf_out = out_root / pdf_path.stem
                per_pdf_out.mkdir(parents=True, exist_ok=True)

                result = mgr.run_complete_pipeline(
                    pdf_path=pdf_path,
                    output_dir=per_pdf_out,
                    detection_conf=args.conf,
                    ocr_conf=args.ocr_confidence,
                    lang=args.ocr_lang,
                )

                log_path = per_pdf_out / "run.log"
                with open(log_path, "w", encoding="utf-8") as log_f:
                    json.dump(result, log_f, indent=2)

                combined_stats["processed"] += 1
                if result.get("status") != "success":
                    combined_stats["errors"] += 1
                    if isinstance(iterator, list):
                        print(f"   ❌ Failed at {result.get('stage')} stage → see {log_path}")
                else:
                    if isinstance(iterator, list):
                        print("   ✅ Success")

            print("\n[Multi-Env] Summary")
            print("------------------")
            print(f"Total PDFs: {combined_stats['processed']}")
            print(f"Successes : {combined_stats['processed'] - combined_stats['errors']}")
            print(f"Errors    : {combined_stats['errors']}")

            return 0 if combined_stats["errors"] == 0 else 1
        
        # Initialize pipeline runner
        runner = CompleteTextPipelineRunner(
            epochs=args.epochs,
            confidence_threshold=args.conf,
            snippet_size=tuple(args.snippet_size),
            overlap=args.overlap,
            model_name=args.model,
            device=args.device,
            ocr_confidence=args.ocr_confidence,
            ocr_lang=args.ocr_lang,
            pdf_confidence_threshold=args.pdf_confidence_threshold,
            create_enhanced_pdf=args.create_enhanced_pdf,
            enhanced_pdf_version=args.enhanced_pdf_version,
            detection_folder=args.detection_folder
        )
        
        # Handle skip modes
        if args.skip_detection and args.skip_text_extraction:
            print("Both detection and text extraction will be skipped, running PDF creation only...")
            success = runner.run_pdf_only_mode()
            
            if success:
                print("\nPDF creation completed successfully!")
                print("Results:")
                print(f"- Enhanced PDFs: {runner.detdiagrams_folder.parent / 'enhanced_pdfs'}")
                print(f"- Summary: {runner.detdiagrams_folder / 'pdf_creation_summary.json'}")
            return 0 if success else 1
        
        elif args.skip_detection:
            # Check if we should auto-skip text extraction
            if args.auto_skip_existing and runner._check_text_extraction_exists():
                print("Text extraction results already exist, skipping to PDF creation...")
                success = runner.run_pdf_only_mode()
                
                if success:
                    print("\nPDF creation completed successfully!")
                    print("Results:")
                    print(f"- Enhanced PDFs: {runner.detdiagrams_folder.parent / 'enhanced_pdfs'}")
                    print(f"- Summary: {runner.detdiagrams_folder / 'pdf_creation_summary.json'}")
                return 0 if success else 1
            else:
                print("Detection will be skipped, running text extraction only...")
                success = runner.run_text_extraction_only()
                
                if success:
                    print("\nText extraction pipeline completed successfully!")
                    print("Results:")
                    print(f"- Text extraction results: {runner.detdiagrams_folder.parent / 'text_extraction'}")
                    if args.create_enhanced_pdf:
                        print(f"- Enhanced PDFs: {runner.detdiagrams_folder.parent / 'enhanced_pdfs'}")
                    print(f"- Summary: {runner.detdiagrams_folder / 'text_extraction_summary.json'}")
                return 0 if success else 1
        
        if args.skip_training:
            print("Skipping training, using existing model...")
            # Find existing best model using config
            runs_dir = runner.config.get_run_path('train')
            if runs_dir.exists():
                train_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "plc_symbol_detector" in d.name]
                if train_dirs:
                    latest_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
                    best_model = latest_dir / "weights" / "best.pt"
                    if best_model.exists():
                        print(f"Using existing model: {best_model}")
                        runner.results["training"] = {"model_path": str(best_model), "epochs": "skipped", "training_time": 0}
                        
                        # Run detection pipeline
                        runner._run_detection_pipeline(best_model)
                        
                        # Run text extraction if not skipped
                        if not args.skip_text_extraction:
                            runner._run_text_extraction()
                        
                        # Create enhanced PDFs if requested
                        if args.create_enhanced_pdf:
                            runner._run_enhanced_pdf_creation()
                        
                        runner._generate_combined_summary_report()
                        return 0
            
            print("No existing trained model found, running full pipeline...")
        
        # Run complete pipeline
        if args.skip_text_extraction:
            print("Text extraction will be skipped")
            success = runner.run_complete_pipeline()
        else:
            success = runner.run_complete_pipeline_with_text()
            
            # Create enhanced PDFs if requested and pipeline succeeded
            if success and args.create_enhanced_pdf:
                print(f"\nPhase 4: Enhanced PDF Creation")
                print("-" * 30)
                runner._run_enhanced_pdf_creation()
        
        if success:
            print("\nPipeline completed successfully!")
            print("Next steps:")
            print("1. Review detection results in processed/detdiagrams/")
            if not args.skip_text_extraction:
                print("2. Review text extraction results in processed/text_extraction/")
                if args.create_enhanced_pdf:
                    print("3. Review enhanced PDFs in processed/enhanced_pdfs/")
                    print("4. Check complete_pipeline_summary.json for detailed metrics")
                    print("5. Proceed to data structuring stage (LayoutLM)")
                else:
                    print("3. Check complete_pipeline_summary.json for detailed metrics")
                    print("4. Proceed to data structuring stage (LayoutLM)")
            else:
                print("2. Run text extraction: python src/run_complete_pipeline_with_text.py --skip-detection")
                print("3. Proceed to data structuring stage")
            return 0
        else:
            print("\nPipeline failed!")
            return 1
            
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
