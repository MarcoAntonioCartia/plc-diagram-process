import sys
import traceback

print("Python Executable:", sys.executable)

try:
    import paddle
    print("Paddle Version:", paddle.__version__)
    
    import paddleocr
    print("PaddleOCR Version:", paddleocr.__version__)
    
    from paddleocr import PaddleOCR
    
    print("Attempting to initialize PaddleOCR...")
    ocr = PaddleOCR(lang='en')
    print("OCR Initialized Successfully")

except Exception as e:
    print("Error:", e)
    traceback.print_exc()
    sys.exit(1)
