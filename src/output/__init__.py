"""
Output Processing for PLC Pipeline
Handles CSV formatting and area-based text grouping
"""

from .csv_formatter import CSVFormatter
from .area_grouper import AreaGrouper

__all__ = ['CSVFormatter', 'AreaGrouper']
