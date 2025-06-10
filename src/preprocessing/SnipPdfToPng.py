import os
import json
import platform
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

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
    
    Detection order:
    1. Environment variable POPPLER_PATH (all platforms)
    2. Local bin/poppler directory (Windows)
    3. System-wide installation (Linux/macOS)
    4. WSL fallback (Windows only, handled by caller)
    
    Returns:
        str or None: Path to poppler binaries, or None if not found
    """
    # Check environment variable first (works on all platforms)
    env_path = os.environ.get("POPPLER_PATH")
    if env_path and Path(env_path).exists():
        print(f"Using poppler from POPPLER_PATH: {env_path}")
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
                # Check if it contains poppler executables
                if ((location / "pdftoppm.exe").exists() or 
                    (location / "pdftoppm").exists()):
                    print(f"Using local poppler: {location}")
                    return str(location)

        # Return None to indicate poppler not found
        # The caller will try WSL as a fallback
        return None
    else:
        # On Unix-like systems, system-wide installation is expected
        # pdf2image will use system poppler automatically when path is None
        return None


def snip_pdf_to_images(pdf_path, output_folder, snippet_size=(800, 600), overlap=0, poppler_path=None):
    """
    Convert PDF to image(s), then snip each image into overlapping chunks.
    """
    # Try WSL conversion first if available on Windows
    if platform.system() == "Windows" and test_wsl_poppler and test_wsl_poppler():
        try:
            print("Using WSL poppler for PDF conversion...")
            images = convert_from_path_wsl(str(pdf_path))
        except Exception as e:
            print(f"WSL conversion failed: {e}, falling back to standard method")
            images = convert_from_path(str(pdf_path), poppler_path=poppler_path)
    else:
        # Standard conversion
        images = convert_from_path(str(pdf_path), poppler_path=poppler_path)
    
    if not images:
        return

    base_name = pdf_path.stem
    metadata = {
        "original_pdf": base_name,
        "snippet_size": snippet_size,
        "overlap": overlap,
        "pages": []
    }

    for page_num, image in enumerate(images, start=1):
        # Convert PIL image to OpenCV format (RGB to BGR array)
        img = np.array(image.convert("RGB"))
        height, width = img.shape[:2]
        snip_w, snip_h = snippet_size

        # Step size is reduced by overlap
        step_w = snip_w - overlap
        step_h = snip_h - overlap

        # Estimate number of snips in each direction
        cols = max(1, (width - overlap) // step_w)
        rows = max(1, (height - overlap) // step_h)

        # Add last column/row if there's leftover area
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

        for row in range(rows):
            for col in range(cols):
                # Compute snip coordinates with bounds checking
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

                print(f"✅ Saved {snippet_name} — {s_w}x{s_h}")

                # Store metadata about this snippet
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

        metadata["pages"].append(page_data)

    output_folder.mkdir(parents=True, exist_ok=True)

    # Save metadata per PDF
    metadata_path = output_folder / f"{base_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def process_pdf_folder(input_folder, output_folder, snippet_size=(800, 600), overlap=0, poppler_path=None):
    """
    Process all PDFs in a folder and generate snipped image chunks for each.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    all_metadata = {}

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.pdf'):
            print(f"📄 Processing: {filename}")
            path = input_folder / filename
            metadata = snip_pdf_to_images(path, output_folder, snippet_size, overlap, poppler_path)
            if metadata:
                all_metadata[metadata["original_pdf"]] = metadata

    # Save combined metadata for all PDFs
    combined_meta_path = output_folder / "all_pdfs_metadata.json"
    with open(combined_meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)


if __name__ == "__main__":
    # 📂 Get project root and data paths
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Try to load config for paths
    import sys
    sys.path.append(str(project_root))
    
    try:
        from src.config import get_config
        config = get_config()
        data_root = Path(config.config['data_root'])
        input_dir = data_root / "raw" / "pdfs"
        output_dir = data_root / "processed" / "images"
    except:
        # Fallback to default paths
        data_root = project_root.parent / "plc-data"
        input_dir = data_root / "raw" / "pdfs"
        output_dir = data_root / "processed" / "images"

    # 🧩 Snippet settings
    snippet_size = (1500, 1200)
    overlap = 500

    # 🔍 Find platform-specific Poppler path
    poppler_dir = find_poppler_path()

    # 🏃 Process all PDFs
    process_pdf_folder(input_dir, output_dir, snippet_size, overlap, poppler_path=poppler_dir)

    print("✅ Snipping completed.")
