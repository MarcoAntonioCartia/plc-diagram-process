import os
import json
import platform
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path


def find_poppler_path():
    """
    Determine the appropriate poppler path based on the platform.
    - On Linux/macOS: assumes poppler-utils is installed system-wide.
    - On Windows: tries to locate poppler in a local directory or via env variable.
    """
    if platform.system() == "Windows":
        # Check environment variable first
        env_path = os.environ.get("POPPLER_PATH")
        if env_path and Path(env_path).exists():
            return env_path

        # Default fallback path (e.g. project_root/bin/poppler/Library/bin)
        fallback = Path(__file__).resolve().parent.parent / "bin" / "poppler" / "Library" / "bin"
        if fallback.exists():
            return str(fallback)

        raise RuntimeError(
            "Poppler not found. Please set the POPPLER_PATH env variable or place poppler binaries in ./bin/poppler/Library/bin"
        )
    else:
        # On Unix-like systems, system-wide installation is expected
        return None


def snip_pdf_to_images(pdf_path, output_folder, snippet_size=(800, 600), overlap=0, poppler_path=None):
    """
    Convert PDF to image(s), then snip each image into overlapping chunks.
    """
    # Convert PDF pages to images
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
    # 📂 Resolve base directory relative to this script
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "dataset" / "test" / "diagrams"
    output_dir = base_dir / "dataset" / "test" / "images"

    # 🧩 Snippet settings
    snippet_size = (1500, 1200)
    overlap = 500

    # 🔍 Find platform-specific Poppler path
    poppler_dir = find_poppler_path()

    # 🏃 Process all PDFs
    process_pdf_folder(input_dir, output_dir, snippet_size, overlap, poppler_path=poppler_dir)

    print("✅ Snipping completed.")
