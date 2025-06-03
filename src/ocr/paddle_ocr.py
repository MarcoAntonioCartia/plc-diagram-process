# https://paddlepaddle.github.io/PaddleOCR/main/en/ppocr/quick_start.html
from paddleocr import PaddleOCR
import cv2

# initialize PP-OCRv4 (best trade-off)
ocr = PaddleOCR(ocr_version="PP-OCRv4", lang="en")

def extract_text(image_path):
    img = cv2.imread(image_path)
    result = ocr.ocr(img)
    # flatten results
    texts = []
    for line in result:
        for box, (txt, score) in line:
            texts.append({"text": txt, "score": float(score), "box": box})
    return texts

if __name__ == "__main__":
    import json
    from pathlib import Path
    
    # Example usage with test images
    project_root = Path(__file__).resolve().parent.parent.parent
    test_images_dir = project_root / "data" / "dataset" / "test" / "images"
    
    if test_images_dir.exists():
        # Get first image from test set
        image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
        if image_files:
            test_image = image_files[0]
            print(f"Processing test image: {test_image}")
            texts = extract_text(str(test_image))
            print(json.dumps(texts, indent=2))
        else:
            print("No test images found")
    else:
        print(f"Test images directory not found: {test_images_dir}")
