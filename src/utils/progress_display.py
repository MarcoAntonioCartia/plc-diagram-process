"""
Progress Display Utility
Provides in-place terminal updates for long-running operations
"""

import sys
import threading
import time
from typing import List, Optional


class ProgressDisplay:
    """
    Terminal progress display that updates lines in-place instead of scrolling
    """
    
    def __init__(self, max_lines: int = 5):
        self.max_lines = max_lines
        self.lines: List[str] = []
        self.lock = threading.Lock()
        self.enabled = sys.stdout.isatty()  # Only enable for interactive terminals
    
    def add_line(self, text: str) -> int:
        """Add a new line and return its index"""
        with self.lock:
            if not self.enabled:
                print(text)
                return len(self.lines)
            
            self.lines.append(text)
            print(text)
            
            # Keep only the last max_lines
            if len(self.lines) > self.max_lines:
                self.lines = self.lines[-self.max_lines:]
            
            return len(self.lines) - 1
    
    def update_line(self, line_index: int, text: str):
        """Update a specific line in-place"""
        with self.lock:
            if not self.enabled:
                print(text)
                return
            
            if line_index < 0 or line_index >= len(self.lines):
                # If line doesn't exist, just add it
                self.add_line(text)
                return
            
            # Calculate how many lines to move up
            lines_to_move_up = len(self.lines) - line_index
            
            # Move cursor up
            if lines_to_move_up > 0:
                sys.stdout.write(f'\033[{lines_to_move_up}A')
            
            # Clear the line and write new content
            sys.stdout.write('\r\033[K' + text)
            
            # Move cursor back down to the bottom
            if lines_to_move_up > 0:
                sys.stdout.write(f'\033[{lines_to_move_up}B')
            
            # Update our internal state
            self.lines[line_index] = text
            
            # Ensure output is flushed
            sys.stdout.flush()
    
    def update_last_line(self, text: str):
        """Update the last line (most common use case)"""
        if self.lines:
            self.update_line(len(self.lines) - 1, text)
        else:
            self.add_line(text)
    
    def clear_lines(self, count: int = None):
        """Clear the last 'count' lines (or all if count is None)"""
        with self.lock:
            if not self.enabled:
                return
            
            if count is None:
                count = len(self.lines)
            
            count = min(count, len(self.lines))
            
            if count > 0:
                # Move up and clear each line
                for i in range(count):
                    if i > 0:
                        sys.stdout.write('\033[A')  # Move up
                    sys.stdout.write('\r\033[K')   # Clear line
                
                # Remove cleared lines from our state
                self.lines = self.lines[:-count]
                sys.stdout.flush()
    
    def finalize_line(self, line_index: int, final_text: str):
        """Finalize a line (no more updates) and move to next"""
        with self.lock:
            if not self.enabled:
                print(final_text)
                return
            
            self.update_line(line_index, final_text)
            # Add a newline to "lock in" this line
            sys.stdout.write('\n')
            sys.stdout.flush()
    
    def finalize_last_line(self, final_text: str):
        """Finalize the last line"""
        if self.lines:
            self.finalize_line(len(self.lines) - 1, final_text)
        else:
            self.add_line(final_text)


class StageProgressDisplay:
    """
    Specialized progress display for pipeline stages
    """
    
    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.display = ProgressDisplay(max_lines=3)
        self.current_file_line = None
        self.current_progress_line = None
    
    def start_stage(self, message: str):
        """Start the stage with an initial message"""
        self.display.add_line(f"X Starting {self.stage_name}...")
        if message:
            self.display.add_line(f"  {message}")
    
    def start_file(self, filename: str):
        """Start processing a new file"""
        text = f"  → Processing: {filename}"
        if self.current_file_line is not None:
            self.display.update_line(self.current_file_line, text)
        else:
            self.current_file_line = self.display.add_line(text)
    
    def update_progress(self, message: str):
        """Update the progress message"""
        text = f"    {message}"
        if self.current_progress_line is not None:
            self.display.update_line(self.current_progress_line, text)
        else:
            self.current_progress_line = self.display.add_line(text)
    
    def complete_file(self, filename: str, result: str):
        """Complete processing of a file"""
        if self.current_file_line is not None:
            self.display.finalize_line(self.current_file_line, f"  ✓ Completed: {filename} - {result}")
        else:
            self.display.add_line(f"  ✓ Completed: {filename} - {result}")
        
        # Reset line tracking for next file
        self.current_file_line = None
        self.current_progress_line = None
    
    def error_file(self, filename: str, error: str):
        """Mark file as failed"""
        if self.current_file_line is not None:
            self.display.finalize_line(self.current_file_line, f"  ✗ Failed: {filename} - {error}")
        else:
            self.display.add_line(f"  ✗ Failed: {filename} - {error}")
        
        # Reset line tracking for next file
        self.current_file_line = None
        self.current_progress_line = None
    
    def complete_stage(self, summary: str):
        """Complete the stage"""
        self.display.add_line(f"✓ {self.stage_name} completed: {summary}")


# Global progress display instance for easy access
_global_progress = None

def get_progress_display() -> ProgressDisplay:
    """Get the global progress display instance"""
    global _global_progress
    if _global_progress is None:
        _global_progress = ProgressDisplay()
    return _global_progress

def create_stage_progress(stage_name: str) -> StageProgressDisplay:
    """Create a new stage progress display"""
    return StageProgressDisplay(stage_name)
