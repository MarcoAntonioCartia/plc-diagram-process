"""
Setup package for PLC Diagram Processor
Contains enhanced setup utilities for automatic installation and configuration
"""

# Make setup modules available
from .gpu_detector import GPUDetector
from .build_tools_installer import BuildToolsInstaller
from .package_installer import RobustPackageInstaller

__all__ = [
    'GPUDetector',
    'BuildToolsInstaller', 
    'RobustPackageInstaller'
]
