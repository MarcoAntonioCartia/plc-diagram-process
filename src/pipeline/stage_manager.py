"""
Stage Manager for PLC Pipeline
Orchestrates stage execution with CI safety and webapp API support
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .base_stage import BaseStage, StageResult


class StageManager:
    """Manages pipeline stage execution and state"""
    
    def __init__(self, project_root: Optional[Path] = None, state_dir: Optional[Path] = None):
        """
        Initialize stage manager
        
        Args:
            project_root: Project root directory
            state_dir: Directory for state files
        """
        self.project_root = project_root or self._find_project_root()
        self.state_dir = state_dir or (self.project_root / ".pipeline_state")
        self.stages: Dict[str, BaseStage] = {}
        self.execution_order: List[str] = []
        self.is_ci = self._detect_ci_environment()
        
        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Register default stages
        self._register_default_stages()
    
    def _find_project_root(self) -> Path:
        """Find project root by looking for key files"""
        current = Path(__file__).resolve()
        
        # Look for project markers
        markers = ['setup.py', 'requirements.txt', '.git', 'src']
        
        for parent in current.parents:
            if any((parent / marker).exists() for marker in markers):
                return parent
        
        # Fallback to current directory
        return Path.cwd()
    
    def _detect_ci_environment(self) -> bool:
        """Detect if running in CI environment"""
        return (
            os.getenv('CI') == 'true' or 
            os.getenv('GITHUB_ACTIONS') == 'true' or
            os.getenv('PYTEST_CURRENT_TEST') is not None
        )
    
    def _register_default_stages(self) -> None:
        """Register default pipeline stages with lazy imports for CI safety"""
        # Import stages lazily to avoid heavy dependencies in CI
        stage_configs = [
            ('preparation', 'Validate inputs and setup directories', 'core', []),
            ('training', 'Train or validate YOLO models', 'yolo_env', ['preparation']),
            ('detection', 'Run YOLO object detection', 'yolo_env', ['training']),
            ('ocr', 'Extract text from detected regions', 'ocr_env', ['detection']),
            ('postprocessing', 'Create CSV output and enhanced PDFs', 'core', ['ocr'])
        ]
        
        for name, description, env, deps in stage_configs:
            try:
                stage_class = self._get_stage_class(name)
                stage = stage_class(name, description, env, deps)
                self.register_stage(stage)
            except Exception as e:
                if not self.is_ci:
                    print(f"Warning: Could not register stage {name}: {e}")
                # In CI, create a mock stage
                else:
                    stage = MockStage(name, description, env, deps)
                    self.register_stage(stage)
    
    def _get_stage_class(self, stage_name: str):
        """Get stage class with lazy import for CI safety"""
        if self.is_ci:
            # In CI, return mock stage class
            return MockStage
        
        # Dynamic import to avoid heavy dependencies
        stage_module_map = {
            'preparation': 'src.pipeline.stages.preparation_stage',
            'training': 'src.pipeline.stages.training_stage',
            'detection': 'src.pipeline.stages.detection_stage',
            'ocr': 'src.pipeline.stages.ocr_stage',
            'postprocessing': 'src.pipeline.stages.postprocessing_stage'
        }
        
        module_name = stage_module_map.get(stage_name)
        if not module_name:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        try:
            import importlib
            module = importlib.import_module(module_name)
            class_name = ''.join(word.capitalize() for word in stage_name.split('_')) + 'Stage'
            return getattr(module, class_name)
        except ImportError as e:
            if self.is_ci:
                return MockStage
            raise ImportError(f"Could not import stage {stage_name}: {e}")
    
    def register_stage(self, stage: BaseStage) -> None:
        """Register a stage"""
        self.stages[stage.name] = stage
        
        # Update execution order based on dependencies
        self._update_execution_order()
    
    def _update_execution_order(self) -> None:
        """Update execution order based on stage dependencies"""
        # Topological sort of stages based on dependencies
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(stage_name: str):
            if stage_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {stage_name}")
            if stage_name in visited:
                return
            
            temp_visited.add(stage_name)
            
            stage = self.stages.get(stage_name)
            if stage:
                for dep in stage.dependencies:
                    if dep in self.stages:
                        visit(dep)
            
            temp_visited.remove(stage_name)
            visited.add(stage_name)
            order.append(stage_name)
        
        for stage_name in self.stages:
            if stage_name not in visited:
                visit(stage_name)
        
        self.execution_order = order
    
    def list_stages(self) -> List[Dict[str, Any]]:
        """List all registered stages"""
        return [
            {
                'name': stage.name,
                'description': stage.description,
                'environment': stage.required_env,
                'dependencies': stage.dependencies,
                'completed': stage.is_completed() if hasattr(stage, 'is_completed') else False
            }
            for stage in self.stages.values()
        ]
    
    def get_stage_status(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific stage"""
        stage = self.stages.get(stage_name)
        if not stage:
            return None
        
        state = stage.get_state() if hasattr(stage, 'get_state') else None
        
        return {
            'name': stage.name,
            'description': stage.description,
            'environment': stage.required_env,
            'dependencies': stage.dependencies,
            'completed': stage.is_completed() if hasattr(stage, 'is_completed') else False,
            'state': state.to_dict() if state else None,
            'progress': stage.get_progress() if hasattr(stage, 'get_progress') else 0.0
        }
    
    def run_stages(self, stage_names: Optional[List[str]] = None, 
                   config: Optional[Dict[str, Any]] = None,
                   skip_completed: bool = True,
                   force_stages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run specified stages or all stages
        
        Args:
            stage_names: List of stage names to run (None for all)
            config: Configuration for stages
            skip_completed: Skip already completed stages
            force_stages: List of stages to force re-run
            
        Returns:
            Dict with execution results
        """
        config = config or {}
        force_stages = force_stages or []
        
        # Determine which stages to run
        if stage_names is None:
            stages_to_run = self.execution_order
        else:
            # Validate stage names
            invalid_stages = [name for name in stage_names if name not in self.stages]
            if invalid_stages:
                return {
                    'success': False,
                    'error': f"Invalid stages: {invalid_stages}",
                    'available_stages': list(self.stages.keys())
                }
            stages_to_run = stage_names
        
        minimal_mode = (
            os.environ.get("PLCDP_MINIMAL_OUTPUT", "0") == "1" or
            os.environ.get("PLCDP_QUIET", "0") == "1"
        )
        
        if not minimal_mode:
            print(f"\nX Starting pipeline execution")
            print(f"Stages to run: {stages_to_run}")
            print(f"CI Mode: {self.is_ci}")
            print(f"Skip completed: {skip_completed}")
            print(f"Force re-run: {force_stages}")
        
        results = {}
        overall_success = True
        start_time = datetime.now()
        
        for stage_name in stages_to_run:
            stage = self.stages[stage_name]
            
            if not minimal_mode:
                print(f"\nX Preparing to run stage: {stage_name}")
                print(f"  Environment: {stage.required_env}")
                print(f"  Dependencies: {stage.dependencies}")
            
            # Setup stage
            stage.setup(config.get(stage_name, {}), self.state_dir)
            
            # Check if should skip
            if (skip_completed and stage.is_completed() and 
                stage_name not in force_stages):
                if not minimal_mode:
                    print(f"X Skipping completed stage: {stage_name}")
                results[stage_name] = {
                    'skipped': True,
                    'reason': 'already_completed'
                }
                continue
            
            # Clear state if forcing re-run
            if stage_name in force_stages:
                if not minimal_mode:
                    print(f"X Forcing re-run of stage: {stage_name}")
                stage.clear_state()
            
            # Run stage
            if not minimal_mode:
                print(f"X Starting execution of stage: {stage_name}")
            
            result = stage.run()
            results[stage_name] = result.to_dict()
            
            if not minimal_mode:
                if result.success:
                    print(f"V Stage {stage_name} completed successfully")
                    # Show key results if available
                    if result.data:
                        if 'output_directory' in result.data:
                            print(f"  Output directory: {result.data['output_directory']}")
                        if 'total_detections' in result.data:
                            print(f"  Total detections: {result.data['total_detections']}")
                        if 'total_text_regions' in result.data:
                            print(f"  Total text regions: {result.data['total_text_regions']}")
                        if 'files_processed' in result.data:
                            print(f"  Files processed: {result.data['files_processed']}")
                else:
                    print(f"X Stage {stage_name} failed: {result.error}")
            
            if not result.success:
                overall_success = False
                print(f"X Pipeline failed at stage: {stage_name}")
                print(f"  Error: {result.error}")
                break
            
            if not minimal_mode:
                print(f"X Stage {stage_name} completed successfully")
                print(f"X DEBUG: About to proceed to next stage...")
                print(f"X DEBUG: Current stage index: {stages_to_run.index(stage_name)}")
                print(f"X DEBUG: Total stages: {len(stages_to_run)}")
                print(f"X DEBUG: Remaining stages: {stages_to_run[stages_to_run.index(stage_name)+1:]}")
                print(f"X Proceeding to next stage...")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            'success': overall_success,
            'stages_run': len([r for r in results.values() if not r.get('skipped', False)]),
            'stages_skipped': len([r for r in results.values() if r.get('skipped', False)]),
            'total_duration': duration,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'results': results,
            'ci_mode': self.is_ci
        }
        
        # Save execution summary
        self._save_execution_summary(summary)
        
        if not minimal_mode:
            print(f"\nX Pipeline execution completed")
            print(f"Success: {overall_success}")
            print(f"Duration: {duration:.2f}s")
            print(f"Stages run: {summary['stages_run']}")
            print(f"Stages skipped: {summary['stages_skipped']}")
        
        return summary
    
    def run_single_stage(self, stage_name: str, config: Optional[Dict[str, Any]] = None,
                        force: bool = False) -> StageResult:
        """
        Run a single stage
        
        Args:
            stage_name: Name of stage to run
            config: Stage configuration
            force: Force re-run even if completed
            
        Returns:
            StageResult: Result of stage execution
        """
        if stage_name not in self.stages:
            return StageResult(
                success=False,
                error=f"Stage '{stage_name}' not found. Available: {list(self.stages.keys())}"
            )
        
        stage = self.stages[stage_name]
        stage.setup(config or {}, self.state_dir)
        
        if force:
            stage.clear_state()
        
        return stage.run()
    
    def reset_pipeline(self, stage_names: Optional[List[str]] = None) -> None:
        """
        Reset pipeline state (clear all or specified stage states)
        
        Args:
            stage_names: List of stage names to reset (None for all)
        """
        stages_to_reset = stage_names or list(self.stages.keys())
        
        for stage_name in stages_to_reset:
            if stage_name in self.stages:
                stage = self.stages[stage_name]
                stage.setup({}, self.state_dir)  # Ensure state_file is set
                stage.clear_state()
                print(f"X Reset stage: {stage_name}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get overall pipeline status"""
        stages_status = []
        
        for stage_name in self.execution_order:
            if stage_name in self.stages:
                status = self.get_stage_status(stage_name)
                if status:
                    stages_status.append(status)
        
        completed_stages = sum(1 for s in stages_status if s['completed'])
        total_stages = len(stages_status)
        overall_progress = completed_stages / total_stages if total_stages > 0 else 0.0
        
        return {
            'total_stages': total_stages,
            'completed_stages': completed_stages,
            'overall_progress': overall_progress,
            'execution_order': self.execution_order,
            'stages': stages_status,
            'ci_mode': self.is_ci,
            'state_dir': str(self.state_dir)
        }
    
    def _save_execution_summary(self, summary: Dict[str, Any]) -> None:
        """Save execution summary to file"""
        try:
            summary_file = self.state_dir / "execution_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save execution summary: {e}")


class MockStage(BaseStage):
    """Mock stage for CI testing"""
    
    def __init__(self, name: str, description: str = "", 
                 required_env: str = "core", dependencies: Optional[List[str]] = None):
        super().__init__(name, description, required_env, dependencies)
    
    def execute(self) -> Dict[str, Any]:
        """Mock execution for CI"""
        return {
            'status': 'mock_success',
            'message': f'Mock execution of {self.name} stage',
            'environment': self.required_env,
            'mock_mode': True
        }
    
    def execute_ci_safe(self) -> Dict[str, Any]:
        """CI-safe execution"""
        return self.execute()
