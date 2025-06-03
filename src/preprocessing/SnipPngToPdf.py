import os
import json
import cv2
import numpy as np
from PIL import Image
import glob
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
from pdf2image import convert_from_path
import argparse

def detect_and_remove_qr_stripe(img, threshold=30):
    """Detects and removes vertical QR stripe from right side of image."""
    h, w = img.shape[:2]
    
    # Scan from right to left for consistent low-variance column
    for x in range(w-1, int(w*0.6), -1):
        if np.std(img[:, x]) < threshold:
            # Found potential QR stripe - verify with neighboring columns
            if np.std(img[:, x-1]) < threshold and np.std(img[:, x-2]) < threshold:
                print(f"🔍 Detected QR stripe at x={x}")
                return img[:, :x]
    
    return img

def compare_images(original_img, reconstructed_img, output_diff_path):
    h1, w1 = original_img.shape[:2]
    h2, w2 = reconstructed_img.shape[:2]
    if (h1, w1) != (h2, w2):
        reconstructed_img = cv2.resize(reconstructed_img, (w1, h1))
    diff = cv2.absdiff(original_img, reconstructed_img)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
    diff_mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(original_img, 0.7, diff_mask, 0.3, 0)
    cv2.imwrite(output_diff_path, overlay)
    nonzero = np.count_nonzero(thresh)
    total = thresh.size
    similarity = 100 * (1 - nonzero / total)
    print(f"✅ Match: {similarity:.2f}% — Diff saved: {output_diff_path}")
    return similarity

def create_pdf_from_images(images, pdf_path, dpi=300):
    h, w = images[0].shape[:2]
    pdf_w = w * 72.0 / dpi
    pdf_h = h * 72.0 / dpi
    c = canvas.Canvas(pdf_path, pagesize=(pdf_w, pdf_h))
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        bio = BytesIO()
        pil_img.save(bio, format="PNG")
        bio.seek(0)
        c.drawImage(ImageReader(bio), 0, 0, pdf_w, pdf_h)
        c.showPage()
    c.save()

def reconstruct_pdf_from_metadata(snippet_folder, metadata_path, output_folder, original_pdf=None, poppler_path=None):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    base_name = metadata["original_pdf"]
    dpi = metadata.get("dpi", 300)
    remove_qr = metadata.get("remove_qr", True)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    original_pages = []
    if original_pdf and os.path.exists(original_pdf):
        try:
            pil_pages = convert_from_path(original_pdf, poppler_path=poppler_path)
            original_pages = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pil_pages]
            if remove_qr:
                original_pages = [detect_and_remove_qr_stripe(p) for p in original_pages]
        except Exception as e:
            print(f"⚠️ Failed to load original PDF: {e}")

    reconstructed_images = []

    for page in metadata["pages"]:
        width = page["original_width"]
        height = page["original_height"]
        page_num = page["page_num"]

        canvas_img = np.ones((height, width, 3), dtype=np.uint8) * 255

        for snippet_info in page["snippets"]:
            path = os.path.join(snippet_folder, snippet_info["filename"])
            x1, y1 = int(snippet_info["x1"]), int(snippet_info["y1"])
            x2, y2 = int(snippet_info["x2"]), int(snippet_info["y2"])
            expected_w = x2 - x1
            expected_h = y2 - y1

            snippet = cv2.imread(path)
            if snippet is None:
                raise ValueError(f"❌ Could not read image: {path}")
            
            # Remove QR stripe if not already removed during snipping
            if not remove_qr:
                snippet = detect_and_remove_qr_stripe(snippet)
            
            s_h, s_w = snippet.shape[:2]

            print(f"🧩 {snippet_info['filename']}")
            print(f"   Metadata coords: ({x1},{y1}) → ({x2},{y2}) = {expected_w}x{expected_h}")
            print(f"   Image shape:     {s_w}x{s_h}")

            if s_h != expected_h:
                raise ValueError(f"❌ Height mismatch for {snippet_info['filename']}: expected {expected_h}, got {s_h}")
            
            if s_w != expected_w:
                print(f"⚠️ Width difference detected. Adjusting placement.")
                x_offset = (expected_w - s_w) // 2
                x1_adjusted = x1 + x_offset
                x2_adjusted = x1_adjusted + s_w
                
                if x2_adjusted > width:
                    x2_adjusted = width
                    x1_adjusted = x2_adjusted - s_w
                
                canvas_img[y1:y2, x1_adjusted:x2_adjusted] = snippet
            else:
                canvas_img[y1:y2, x1:x2] = snippet

        img_path = os.path.join(output_folder, f"{base_name}_page_{page_num}.png")
        cv2.imwrite(img_path, canvas_img)
        reconstructed_images.append(canvas_img)

        if original_pages and page_num <= len(original_pages):
            diff_path = os.path.join(output_folder, f"{base_name}_page_{page_num}_diff.png")
            compare_images(original_pages[page_num - 1], canvas_img, diff_path)

    pdf_path = os.path.join(output_folder, f"{base_name}_reconstructed.pdf")
    create_pdf_from_images(reconstructed_images, pdf_path, dpi)
    print(f"📄 PDF saved: {pdf_path}")
    return pdf_path

def find_original_pdf(base_name, pdf_folder):
    matches = glob.glob(os.path.join(pdf_folder, f"{base_name}*.pdf"))
    return matches[0] if matches else None

def main():
    parser = argparse.ArgumentParser(description="Reconstruct PDFs using metadata.")
    parser.add_argument("--input", "-i", required=True, help="Folder with PNG snippets")
    parser.add_argument("--output", "-o", required=True, help="Output folder for results")
    parser.add_argument("--pdfs", required=True, help="Folder containing original PDFs")
    parser.add_argument("--poppler", help="Poppler path for pdf2image (Windows)")

    args = parser.parse_args()
    metadata_files = glob.glob(os.path.join(args.input, "*_metadata.json"))
    if not metadata_files:
        print("❌ No metadata files found.")
        return

    for meta_path in metadata_files:
        if "all_pdfs_metadata.json" in meta_path:
            continue
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        base_name = metadata.get("original_pdf")
        print(f"\n🔁 Reconstructing {base_name}")
        original_pdf = find_original_pdf(base_name, args.pdfs)
        if original_pdf:
            print(f"🔎 Matching original PDF found: {original_pdf}")
        else:
            print(f"⚠️ No matching original PDF found for validation.")

        reconstruct_pdf_from_metadata(
            snippet_folder=args.input,
            metadata_path=meta_path,
            output_folder=args.output,
            original_pdf=original_pdf,
            poppler_path=args.poppler
        )

if __name__ == "__main__":
    main()