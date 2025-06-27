"""
Base Stage Class for PLC Pipeline
Provides common functionality for all pipeline stages with CI safety
"""

import os
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class StageResult:
    """Result object for stage execution"""
    
    def __init__(self, success: bool, data: Optional[Dict[str, Any]] = None, 
                 error: Optional[str] = None, duration: float = 0.0):
        self.success = success
        self.data = data or {}
        self.error = error
        self.duration = duration
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'duration': self.duration,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StageResult':
        """Create from dictionary"""
        return cls(
            success=data.get('success', False),
            data=data.get('data', {}),
            error=data.get('error'),
            duration=data.get('duration', 0.0)
        )


class BaseStage(ABC):
    """Abstract base class for all pipeline stages"""
    
    def __init__(self, name: str, description: str = "", 
                 required_env: str = "core", dependencies: Optional[List[str]] = None):
        """
        Initialize base stage
        
        Args:
            name: Stage name (e.g., 'preparation', 'detection')
            description: Human-readable description
            required_env: Required environment ('core', 'yolo_env', 'ocr_env')
            dependencies: List of stage names this stage depends on
        """
        self.name = name
        self.description = description
        self.required_env = required_env
        self.dependencies = dependencies or []
        self.state_file = None
        self.config = {}
        self.is_ci = self._detect_ci_environment()
        
    def _detect_ci_environment(self) -> bool:
        """Detect if running in CI environment"""
        return (
            os.getenv('CI') == 'true' or 
            os.getenv('GITHUB_ACTIONS') == 'true' or
            os.getenv('PYTEST_CURRENT_TEST') is not None
        )
    
    def setup(self, config: Dict[str, Any], state_dir: Path) -> None:
        """
        Setup stage with configuration and state directory
        
        Args:
            config: Stage configuration
            state_dir: Directory for state files
        """
        self.config = config
        self.state_file = state_dir / f"{self.name}_state.json"
        
        # Ensure state directory exists
        state_dir.mkdir(parents=True, exist_ok=True)
    
    def is_completed(self) -> bool:
        """Check if stage was already completed successfully"""
        if not self.state_file or not self.state_file.exists():
            return False
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            result = StageResult.from_dict(state)
            return result.success
        except Exception:
            return False
    
    def get_state(self) -> Optional[StageResult]:
        """Load previous stage state"""
        if not self.state_file or not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            return StageResult.from_dict(state)
        except Exception as e:
            print(f"Warning: Could not load state for {self.name}: {e}")
            return None
    
    def save_state(self, result: StageResult) -> None:
        """Save stage state"""
        if not self.state_file:
            return
        
        try:
            with open(self.state_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save state for {self.name}: {e}")
    
    def clear_state(self) -> None:
        """Clear stage state (for force re-run)"""
        if self.state_file and self.state_file.exists():
            try:
                self.state_file.unlink()
            except Exception as e:
                print(f"Warning: Could not clear state for {self.name}: {e}")
    
    def run(self) -> StageResult:
        """
        Run the stage with timing and error handling
        
        Returns:
            StageResult: Result of stage execution
        """
        print(f"\n{'='*50}")
        print(f"Stage {self.name.upper()}: {self.description}")
        print(f"Environment: {self.required_env}")
        print(f"CI Mode: {self.is_ci}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        try:
            # Check dependencies
            if not self._check_dependencies():
                return StageResult(
                    success=False,
                    error="Stage dependencies not met",
                    duration=time.time() - start_time
                )
            
            # Execute stage
            if self.is_ci:
                result_data = self.execute_ci_safe()
            else:
                result_data = self.execute()
            
            duration = time.time() - start_time
            result = StageResult(success=True, data=result_data, duration=duration)
            
            # Save state
            self.save_state(result)
            
            print(f"V Stage {self.name} completed successfully in {duration:.2f}s")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Stage {self.name} failed: {str(e)}"
            print(f"X {error_msg}")
            
            result = StageResult(success=False, error=error_msg, duration=duration)
            self.save_state(result)
            return result
    
    def _check_dependencies(self) -> bool:
        """Check if all dependencies are satisfied"""
        if not self.dependencies:
            return True
        
        state_dir = self.state_file.parent if self.state_file else Path(".")
        
        for dep_name in self.dependencies:
            dep_state_file = state_dir / f"{dep_name}_state.json"
            
            if not dep_state_file.exists():
                print(f"X Dependency {dep_name} not completed")
                return False
            
            try:
                with open(dep_state_file, 'r') as f:
                    dep_state = json.load(f)
                
                if not dep_state.get('success', False):
                    print(f"X Dependency {dep_name} failed")
                    return False
            except Exception as e:
                print(f"X Could not check dependency {dep_name}: {e}")
                return False
        
        return True
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute the stage logic (full implementation)
        
        Returns:
            Dict[str, Any]: Stage execution results
        """
        pass
    
    def execute_ci_safe(self) -> Dict[str, Any]:
        """
        Execute stage in CI-safe mode (lightweight, no heavy dependencies)
        
        Returns:
            Dict[str, Any]: Mock/lightweight execution results
        """
        print(f"X Running {self.name} in CI-safe mode")
        
        # Default CI-safe implementation - can be overridden by subclasses
        return {
            'status': 'ci_mock',
            'message': f'Stage {self.name} executed in CI-safe mode',
            'environment': self.required_env,
            'ci_mode': True
        }
    
    def validate_inputs(self) -> bool:
        """
        Validate stage inputs
        
        Returns:
            bool: True if inputs are valid
        """
        # Default implementation - can be overridden
        return True
    
    def cleanup(self) -> None:
        """Cleanup stage resources"""
        # Default implementation - can be overridden
        pass
    
    def get_progress(self) -> float:
        """
        Get stage progress (0.0 to 1.0)
        
        Returns:
            float: Progress percentage
        """
        # Default implementation - can be overridden
        if self.is_completed():
            return 1.0
        return 0.0
    
    def __str__(self) -> str:
        return f"Stage({self.name}, env={self.required_env})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', env='{self.required_env}')>"
