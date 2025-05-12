# https://paddlepaddle.github.io/PaddleOCR/main/en/ppocr/quick_start.html
from paddleocr import PaddleOCR
import cv2

# initialize PP-OCRv4 (best trade-off):contentReference[oaicite:2]{index=2}
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
    texts = extract_text("../../data/raw/diagram1.png")
    print(json.dumps(texts, indent=2))