## Debugging and Findings: OCR Pipeline and Detection Issues

### What Was Added
- **ROI Debugging:** The pipeline now saves every region of interest (ROI) image that is passed to OCR into a `debug_rois` folder. Each ROI is named with its page, class, confidence, and coordinates.
- **Logging:** Added print statements to log ROI details for each detection, including coordinates, class, and confidence.

### What Was Found
- **Not the Whole PDF:** The OCR is not being run on the entire PDF page, but only on the cropped detection box (ROI) as intended.
- **Blank/Irrelevant ROIs:** Many of the saved ROIs are blank or contain only lines/arrows, not text. This means the detection model (YOLO) is producing boxes in non-text regions, even at high confidence thresholds.
- **Missed Text:** Some text regions are not being detected at all, indicating the detection model is missing relevant areas.

### Current Problems
- **Detection Model Quality:** The main issue is with the detection model, not the OCR or cropping logic. YOLO is not reliably localizing text regions.
- **Filtering Needed:** The pipeline should filter detections by class, area, and aspect ratio to avoid running OCR on irrelevant regions.
- **Model Improvement:** Consider retraining or refining the detection model with more accurate text/symbol labels to improve localization.

### Next Steps
- Implement class-based and area/aspect ratio filtering in the pipeline.
- Retrain or fine-tune the detection model for better text region detection.
- Continue using ROI debug images to validate improvements.