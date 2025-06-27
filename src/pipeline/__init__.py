"""
PLC Pipeline Stage-Based Architecture
Provides stage-based execution with CI compatibility and webapp API support
"""

from .stage_manager import StageManager
from .base_stage import BaseStage

__all__ = ['StageManager', 'BaseStage']
