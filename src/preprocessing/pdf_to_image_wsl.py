"""
WSL wrapper for pdf2image to use poppler from WSL on Windows.

This module provides a fallback method for Windows users who don't have
native poppler installed but have it available in WSL (Windows Subsystem for Linux).

Functions:
    test_wsl_poppler(): Check if WSL and poppler are available
    convert_from_path_wsl(): Convert PDF to images using WSL poppler
"""
import subprocess
import tempfile
import shutil
from pathlib import Path
from PIL import Image


def convert_from_path_wsl(pdf_path, dpi=200, fmt='png'):
    """
    Convert PDF to images using WSL poppler
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion
        fmt: Output format (png, jpg, etc.)
    
    Returns:
        List of PIL Image objects
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Convert Windows path to WSL path
        win_path = str(pdf_path.absolute()).replace('\\', '/')
        if win_path[1] == ':':
            # Convert C:/ to /mnt/c/
            wsl_pdf_path = f"/mnt/{win_path[0].lower()}/{win_path[3:]}"
        else:
            wsl_pdf_path = win_path
        
        # Output path in WSL format
        win_temp = str(temp_path.absolute()).replace('\\', '/')
        if win_temp[1] == ':':
            wsl_temp_path = f"/mnt/{win_temp[0].lower()}/{win_temp[3:]}"
        else:
            wsl_temp_path = win_temp
        
        # Build WSL command
        output_pattern = f"{wsl_temp_path}/page"
        cmd = [
            'wsl', '-e', 'pdftoppm',
            '-r', str(dpi),
            '-png' if fmt == 'png' else f'-{fmt}',
            wsl_pdf_path,
            output_pattern
        ]
        
        # Run conversion
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"PDF conversion failed: {e.stderr}")
        
        # Load generated images
        images = []
        image_files = sorted(temp_path.glob(f"page-*.{fmt}"))
        
        if not image_files:
            # Try without hyphen (different poppler versions)
            image_files = sorted(temp_path.glob(f"page*.{fmt}"))
        
        for img_file in image_files:
            images.append(Image.open(img_file).copy())
        
        if not images:
            raise RuntimeError("No images generated from PDF")
        
        return images


def test_wsl_poppler():
    """Test if WSL poppler is available"""
    try:
        result = subprocess.run(['wsl', '-e', 'which', 'pdftoppm'], 
                              capture_output=True, text=True)
        return result.returncode == 0 and result.stdout.strip()
    except:
        return False
