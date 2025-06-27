"""
Pipeline Stages for PLC Diagram Processor
Individual stage implementations with CI safety
"""

# Import stages with CI safety
try:
    from .preparation_stage import PreparationStage
    from .training_stage import TrainingStage
    from .detection_stage import DetectionStage
    from .ocr_stage import OcrStage
    from .enhancement_stage import EnhancementStage
    
    __all__ = [
        'PreparationStage',
        'TrainingStage', 
        'DetectionStage',
        'OcrStage',
        'EnhancementStage'
    ]
except ImportError:
    # In CI or when dependencies are missing, provide empty list
    __all__ = []
