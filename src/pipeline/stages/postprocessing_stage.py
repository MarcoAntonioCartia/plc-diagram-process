"""
Postprocessing Stage for PLC Pipeline
Creates CSV output and enhanced PDFs in core environment
"""

import os
from pathlib import Path
from typing import Dict, Any, List

from ..base_stage import BaseStage


class PostprocessingStage(BaseStage):
    """Stage 5: Postprocessing - Create CSV output and enhanced PDFs"""
    
    def __init__(self, name: str = "postprocessing", 
                 description: str = "Create CSV output and enhanced PDFs",
                 required_env: str = "core", dependencies: list = None):
        super().__init__(name, description, required_env, dependencies or ["ocr"])
    
    def execute(self) -> Dict[str, Any]:
        """Execute postprocessing stage"""
        print("X Starting postprocessing stage...")
        
        # Get configuration
        from src.config import get_config
        config = get_config()
        
        # Import postprocessing modules
        self._import_dependencies()
        
        # Find OCR results to process
        data_root = Path(config.config["data_root"])
        ocr_dir = data_root / "processed" / "text_extraction"
        csv_output_dir = data_root / "processed" / "csv_output"
        pdf_output_dir = data_root / "processed" / "enhanced_pdfs"
        
        if not ocr_dir.exists():
            return {
                'status': 'error',
                'error': f"OCR results directory not found: {ocr_dir}",
                'environment': 'core'
            }
        
        # Find text extraction files
        text_files = list(ocr_dir.glob("*_text_extraction.json"))
        if not text_files:
            return {
                'status': 'error',
                'error': f"No text extraction files found in: {ocr_dir}",
                'environment': 'core'
            }
        
        # Create CSV output
        csv_results = self._create_csv_output(text_files, csv_output_dir, config)
        
        # Create enhanced PDFs
        pdf_results = self._create_enhanced_pdfs(text_files, pdf_output_dir, config)
        
        return {
            'status': 'success',
            'csv_results': csv_results,
            'pdf_results': pdf_results,
            'environment': 'core',
            'output_directories': {
                'csv': str(csv_output_dir),
                'pdfs': str(pdf_output_dir)
            }
        }
    
    def execute_ci_safe(self) -> Dict[str, Any]:
        """CI-safe execution without heavy dependencies"""
        print("X Running postprocessing stage in CI-safe mode")
        
        return {
            'status': 'ci_mock',
            'csv_results': {
                'files_created': 0,
                'total_regions': 0,
                'mock_mode': True
            },
            'pdf_results': {
                'files_created': 0,
                'mock_mode': True
            },
            'environment': self.required_env,
            'ci_mode': True
        }
    
    def _import_dependencies(self):
        """Import postprocessing dependencies"""
        try:
            # Import CSV formatter (should always work)
            from src.postprocessing.csv_formatter import CSVFormatter
            from src.postprocessing.area_grouper import AreaGrouper
            self._csv_formatter = CSVFormatter
            self._area_grouper = AreaGrouper
            
            # Try to import PDF enhancement modules
            try:
                from src.utils.pdf_annotator import PDFAnnotator
                self._pdf_creator = PDFAnnotator
                self._pdf_available = True
                print("  ✓ Postprocessing modules available (including PDF)")
            except ImportError as pdf_error:
                # PDF creation not available, but CSV should still work
                self._pdf_creator = MockPDFCreator
                self._pdf_available = False
                print(f"    PDF enhancement not available: {pdf_error}")
                print("  ✓ CSV output will still be created")
            
        except ImportError as e:
            if self.is_ci:
                # In CI, use mock
                self._csv_formatter = MockCSVFormatter
                self._area_grouper = MockAreaGrouper
                self._pdf_creator = MockPDFCreator
                self._pdf_available = False
                print("  🔧 Using mock postprocessing modules for CI")
            else:
                raise ImportError(f"Postprocessing modules not available: {e}")
    
    def _create_csv_output(self, text_files: List[Path], output_dir: Path, config) -> Dict[str, Any]:
        """Create CSV output from text extraction results"""
        print("  X Creating CSV output...")
        
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure CSV formatter
            area_grouping = self.config.get('area_grouping', True)
            alphanumeric_sort = self.config.get('alphanumeric_sort', True)
            
            formatter = self._csv_formatter(
                area_grouping=area_grouping,
                alphanumeric_sort=alphanumeric_sort
            )
            
            # Create combined CSV for all files
            combined_csv = output_dir / "combined_text_extraction.csv"
            result = formatter.format_text_extraction_results(text_files, combined_csv)
            
            # Create individual CSV files for each document
            individual_results = []
            for text_file in text_files:
                # Extract document name from filename (e.g., "1150_text_extraction.json" -> "1150")
                document_name = text_file.stem.replace("_text_extraction", "")
                individual_csv = output_dir / f"{document_name}_text_extraction.csv"
                
                individual_result = formatter.format_text_extraction_results(
                    [text_file], individual_csv
                )
                individual_results.append({
                    'document': document_name,
                    'file': str(individual_csv),
                    'regions': individual_result.get('total_regions', 0)
                })
            
            return {
                'status': 'success',
                'combined_csv': str(combined_csv),
                'combined_regions': result.get('total_regions', 0),
                'individual_files': individual_results,
                'total_files': len(individual_results),
                'area_grouping': area_grouping,
                'alphanumeric_sort': alphanumeric_sort
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f"CSV creation failed: {str(e)}"
            }
    
    def _create_enhanced_pdfs(self, text_files: List[Path], output_dir: Path, config) -> Dict[str, Any]:
        """Create enhanced PDFs with detection and text overlays"""
        print("  X Creating enhanced PDFs...")
        
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure PDF creator
            pdf_version = self.config.get('enhanced_pdf_version', 'short')
            detection_threshold = self.config.get('detection_threshold', 0.8)
            text_threshold = self.config.get('text_threshold', 0.5)
            
            creator = self._pdf_creator(
                detection_confidence_threshold=detection_threshold,
                text_confidence_threshold=text_threshold
            )
            
            # Find corresponding detection files and original PDFs
            data_root = Path(config.config["data_root"])
            detection_dir = data_root / "processed" / "detdiagrams"
            pdf_dir = data_root / "raw" / "pdfs"
            
            enhanced_pdfs = []
            
            for text_file in text_files:
                try:
                    # Extract document name from filename (e.g., "1150_text_extraction.json" -> "1150")
                    document_name = text_file.stem.replace("_text_extraction", "")
                    
                    # Find corresponding files
                    detection_file = detection_dir / f"{document_name}_detections.json"
                    original_pdf = pdf_dir / f"{document_name}.pdf"
                    
                    if not detection_file.exists():
                        print(f"    X Detection file not found: {detection_file}")
                        continue
                    
                    if not original_pdf.exists():
                        print(f"    X Original PDF not found: {original_pdf}")
                        continue
                    
                    # Create enhanced PDF
                    enhanced_pdf = output_dir / f"{document_name}_enhanced.pdf"
                    
                    result_pdf = creator.create_enhanced_pdf(
                        detection_file=detection_file,
                        text_extraction_file=text_file,
                        pdf_file=original_pdf,
                        output_file=enhanced_pdf,
                        version=pdf_version
                    )
                    
                    enhanced_pdfs.append({
                        'document': document_name,
                        'enhanced_pdf': str(result_pdf),
                        'version': pdf_version
                    })
                    
                    print(f"    V Created enhanced PDF: {document_name}")
                    
                except Exception as e:
                    print(f"    X Error creating PDF for {document_name}: {e}")
                    continue
            
            return {
                'status': 'success',
                'enhanced_pdfs': enhanced_pdfs,
                'total_files': len(enhanced_pdfs),
                'version': pdf_version,
                'detection_threshold': detection_threshold,
                'text_threshold': text_threshold
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Enhanced PDF creation failed: {str(e)}"
            }
    
    def validate_inputs(self) -> bool:
        """Validate stage inputs"""
        # Check if OCR stage completed successfully
        if self.state_file:
            ocr_state_file = self.state_file.parent / "ocr_state.json"
            if not ocr_state_file.exists():
                print("X OCR stage not completed")
                return False
        
        return True


class MockCSVFormatter:
    """Mock CSV formatter for CI testing"""
    
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def format_text_extraction_results(self, text_files, output_file):
        """Mock CSV formatting"""
        return {
            'status': 'success',
            'total_regions': 10,  # Mock count
            'areas_found': 3,
            'output_file': str(output_file)
        }


class MockAreaGrouper:
    """Mock area grouper for CI testing"""
    
    def __init__(self, **kwargs):
        self.config = kwargs


class MockPDFCreator:
    """Mock PDF creator for CI testing"""
    
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def create_enhanced_pdf(self, **kwargs):
        """Mock PDF creation"""
        return kwargs.get('output_file', 'mock_enhanced.pdf')
