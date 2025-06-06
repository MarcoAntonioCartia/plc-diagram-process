"""
PDF Reconstruction with Detection Overlays
Reconstructs PDFs from snippets and overlays detection results with labels
"""

import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import red, blue, green, orange
from io import BytesIO
from pdf2image import convert_from_path
import os

from coordinate_transform import save_coordinate_mapping, validate_coordinates, get_detection_statistics

def reconstruct_pdf_with_detections(metadata, global_detections, images_folder, output_folder, original_pdf=None):
    """
    Reconstruct PDF with detection overlays and coordinate mapping
    
    Args:
        metadata: PDF metadata from snipping process
        global_detections: Detection results with global coordinates
        images_folder: Folder containing image snippets
        output_folder: Output folder for results
        original_pdf: Path to original PDF for comparison (optional)
    
    Returns:
        List of generated output files
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    pdf_name = metadata["original_pdf"]
    output_files = []
    
    print(f"Reconstructing {pdf_name} with detections...")
    
    # Reconstruct each page
    reconstructed_images = []
    
    for page_info in metadata["pages"]:
        page_num = page_info["page_num"]
        print(f"  Processing page {page_num}...")
        
        # Find corresponding page detections
        page_detections = None
        for page_det in global_detections["pages"]:
            if page_det["page_num"] == page_num:
                page_detections = page_det["detections"]
                break
        
        if page_detections is None:
            page_detections = []
        
        # Reconstruct page image
        page_image = reconstruct_page_with_detections(
            page_info, page_detections, images_folder
        )
        
        reconstructed_images.append(page_image)
        
        # Save individual page image
        page_image_path = output_folder / f"{pdf_name}_page_{page_num}_detected.png"
        cv2.imwrite(str(page_image_path), page_image)
        output_files.append(page_image_path)
    
    # Create PDF from reconstructed images
    pdf_path = output_folder / f"{pdf_name}_detected.pdf"
    create_pdf_from_images(reconstructed_images, pdf_path)
    output_files.append(pdf_path)
    
    # Save detection results as JSON
    json_path = output_folder / f"{pdf_name}_detections.json"
    with open(json_path, 'w') as f:
        json.dump(global_detections, f, indent=2)
    output_files.append(json_path)
    
    # Save coordinate mapping as text
    txt_path = output_folder / f"{pdf_name}_coordinates.txt"
    save_coordinate_mapping(global_detections, txt_path)
    output_files.append(txt_path)
    
    # Generate statistics
    stats = get_detection_statistics(global_detections)
    stats_path = output_folder / f"{pdf_name}_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    output_files.append(stats_path)
    
    # Validate coordinates
    valid = validate_coordinates(global_detections)
    if not valid:
        print(f"  Warning: Some detections have invalid coordinates")
    
    print(f"  Generated {len(output_files)} output files")
    print(f"  Total detections: {stats['total_detections']}")
    
    return output_files

def reconstruct_page_with_detections(page_info, page_detections, images_folder):
    """
    Reconstruct a single page with detection overlays
    
    Args:
        page_info: Page metadata
        page_detections: List of detections for this page
        images_folder: Folder containing image snippets
    
    Returns:
        Reconstructed page image with detection overlays
    """
    width = page_info["original_width"]
    height = page_info["original_height"]
    
    # Create blank canvas
    canvas_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Place all snippets
    for snippet_info in page_info["snippets"]:
        snippet_path = Path(images_folder) / snippet_info["filename"]
        
        if not snippet_path.exists():
            print(f"    Warning: Snippet not found: {snippet_path}")
            continue
        
        # Load snippet
        snippet = cv2.imread(str(snippet_path))
        if snippet is None:
            print(f"    Warning: Could not load snippet: {snippet_path}")
            continue
        
        # Place snippet on canvas
        x1, y1 = snippet_info["x1"], snippet_info["y1"]
        x2, y2 = snippet_info["x2"], snippet_info["y2"]
        s_h, s_w = snippet.shape[:2]
        
        # Ensure snippet fits
        x2 = min(x1 + s_w, width)
        y2 = min(y1 + s_h, height)
        
        canvas_img[y1:y2, x1:x2] = snippet[:y2-y1, :x2-x1]
    
    # Overlay detections
    canvas_img = overlay_detections(canvas_img, page_detections)
    
    return canvas_img

def overlay_detections(image, detections):
    """
    Overlay detection bounding boxes and labels on image
    
    Args:
        image: Input image (numpy array)
        detections: List of detections with global coordinates
    
    Returns:
        Image with detection overlays
    """
    if not detections:
        return image
    
    # Convert to PIL for better text rendering
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Color mapping for different classes
    colors = {
        'PLC': (255, 0, 0),      # Red
        'input_sensor': (0, 255, 0),    # Green
        'output_valve': (0, 0, 255),    # Blue
        'relay': (255, 165, 0),         # Orange
        'contactor': (255, 0, 255)      # Magenta
    }
    
    for i, detection in enumerate(detections):
        bbox = detection["bbox_global"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        
        # Get color for this class
        color = colors.get(class_name, (255, 255, 0))  # Default to yellow
        
        # Draw bounding box
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Create label
        label = f"{class_name} ({confidence:.2f})"
        
        # Calculate label position
        if font:
            bbox_font = draw.textbbox((0, 0), label, font=font)
            label_width = bbox_font[2] - bbox_font[0]
            label_height = bbox_font[3] - bbox_font[1]
        else:
            label_width = len(label) * 10
            label_height = 15
        
        # Position label above bounding box
        label_x = x1
        label_y = max(0, y1 - label_height - 5)
        
        # Draw label background
        draw.rectangle([label_x, label_y, label_x + label_width + 10, label_y + label_height + 5], 
                      fill=color, outline=color)
        
        # Draw label text
        if font:
            draw.text((label_x + 5, label_y + 2), label, fill=(255, 255, 255), font=font)
        else:
            draw.text((label_x + 5, label_y + 2), label, fill=(255, 255, 255))
        
        # Add detection number
        number_label = str(i + 1)
        if font:
            draw.text((x1 + 5, y1 + 5), number_label, fill=color, font=font)
        else:
            draw.text((x1 + 5, y1 + 5), number_label, fill=color)
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def create_pdf_from_images(images, pdf_path, dpi=300):
    """
    Create PDF from list of images
    
    Args:
        images: List of images (numpy arrays)
        pdf_path: Output PDF path
        dpi: DPI for PDF creation
    """
    if not images:
        return
    
    # Get dimensions from first image
    h, w = images[0].shape[:2]
    pdf_w = w * 72.0 / dpi
    pdf_h = h * 72.0 / dpi
    
    c = canvas.Canvas(str(pdf_path), pagesize=(pdf_w, pdf_h))
    
    for img in images:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Save to BytesIO
        bio = BytesIO()
        pil_img.save(bio, format="PNG")
        bio.seek(0)
        
        # Add to PDF
        c.drawImage(ImageReader(bio), 0, 0, pdf_w, pdf_h)
        c.showPage()
    
    c.save()
    print(f"    PDF saved: {pdf_path}")

def compare_with_original(reconstructed_path, original_pdf_path, output_folder, poppler_path=None):
    """
    Compare reconstructed PDF with original for validation
    
    Args:
        reconstructed_path: Path to reconstructed PDF
        original_pdf_path: Path to original PDF
        output_folder: Output folder for comparison results
        poppler_path: Path to poppler binaries (Windows)
    
    Returns:
        Similarity percentage
    """
    try:
        # Convert both PDFs to images
        original_images = convert_from_path(str(original_pdf_path), poppler_path=poppler_path)
        reconstructed_images = convert_from_path(str(reconstructed_path), poppler_path=poppler_path)
        
        if len(original_images) != len(reconstructed_images):
            print(f"    Warning: Page count mismatch - Original: {len(original_images)}, Reconstructed: {len(reconstructed_images)}")
        
        similarities = []
        
        for i, (orig, recon) in enumerate(zip(original_images, reconstructed_images)):
            # Convert to OpenCV format
            orig_cv = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
            recon_cv = cv2.cvtColor(np.array(recon), cv2.COLOR_RGB2BGR)
            
            # Resize if needed
            if orig_cv.shape != recon_cv.shape:
                recon_cv = cv2.resize(recon_cv, (orig_cv.shape[1], orig_cv.shape[0]))
            
            # Calculate difference
            diff = cv2.absdiff(orig_cv, recon_cv)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
            
            # Calculate similarity
            nonzero = np.count_nonzero(thresh)
            total = thresh.size
            similarity = 100 * (1 - nonzero / total)
            similarities.append(similarity)
            
            # Save difference image
            diff_path = Path(output_folder) / f"page_{i+1}_comparison.png"
            diff_overlay = cv2.addWeighted(orig_cv, 0.7, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), 0.3, 0)
            cv2.imwrite(str(diff_path), diff_overlay)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        print(f"    Average similarity: {avg_similarity:.2f}%")
        
        return avg_similarity
        
    except Exception as e:
        print(f"    Error comparing with original: {e}")
        return 0

if __name__ == "__main__":
    # Example usage for testing
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PDF reconstruction with detections')
    parser.add_argument('--metadata', required=True, help='Metadata JSON file')
    parser.add_argument('--detections', required=True, help='Global detections JSON file')
    parser.add_argument('--images', required=True, help='Images folder')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--original', help='Original PDF for comparison')
    
    args = parser.parse_args()
    
    # Load metadata
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)
    
    # Load detections
    with open(args.detections, 'r') as f:
        global_detections = json.load(f)
    
    # Reconstruct PDF
    output_files = reconstruct_pdf_with_detections(
        metadata=metadata,
        global_detections=global_detections,
        images_folder=args.images,
        output_folder=args.output,
        original_pdf=args.original
    )
    
    print(f"Reconstruction completed")
    print(f"Generated files: {len(output_files)}")
    for file_path in output_files:
        print(f"  {file_path}")
