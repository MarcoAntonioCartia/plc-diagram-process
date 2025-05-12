# https://ubiai.tools/fine-tuning-layoutlmv3-customizing-layout-recognition-for-diverse-document-types/
import cv2
from segment_anything import build_sam, SamPredictor

def overlay_mask(image_path, masks):
    img = cv2.imread(image_path)
    for mask in masks:
        img[mask > 0] = (0, 0, 255)  # highlight error regions
    cv2.imwrite("overlay.png", img)

# usage:
# sam = build_sam(checkpoint="sam_vit_h.pth")
# predictor = SamPredictor(sam)
# masks = predictor.predict(...)