import fitz  # PyMuPDF
from pathlib import Path
import json

# --- Configuration ---
PDF_FILE = Path("D:/MarMe/github/0.3/plc-data/raw/pdfs/1150.pdf")
TEXT_JSON = Path("D:/MarMe/github/0.3/plc-data/processed/text_extraction/1150_text_extraction.json")
DETECTION_JSON = Path("D:/MarMe/github/0.3/plc-data/processed/detdiagrams2/1150_detections.json")
# ---

# 1. Load the original PDF
original_doc = fitz.open(str(PDF_FILE))
original_page = original_doc[0]  # Assuming one-page PDF

# 2. Print page geometry information
print("--- Original Page Geometry ---")
print(f"Rotation: {original_page.rotation}")
print(f"Rect (WxH): {original_page.rect.width} x {original_page.rect.height}")
print(f"MediaBox (WxH): {original_page.mediabox.width} x {original_page.mediabox.height}")
print(f"CropBox (WxH): {original_page.cropbox.width} x {original_page.cropbox.height}")
print("-" * 30)

# Expected output for a landscape PDF:
# Rotation: 90 or 270
# Rect (WxH): [Height] x [Width] (e.g., 612.0 x 792.0 for letter size)
# This shows fitz reports dimensions for a portrait page and applies rotation.

# 3. (Optional) Load one text box and one detection box for later testing
# with open(TEXT_JSON, 'r') as f:
#     text_data = json.load(f)
#     first_text_box = text_data['text_regions'][0]['bbox']
#     print(f"Sample Text Box: {first_text_box}")

original_doc.close()