"""
Stage-Based PLC Pipeline Runner
Modern replacement for the monolithic pipeline with stage management
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.stage_manager import StageManager
from src.config import get_config


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="PLC Diagram Processor - Stage-Based Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python src/run_pipeline.py --run-all
  
  # Run specific stages
  python src/run_pipeline.py --stages preparation training detection
  
  # Show pipeline status
  python src/run_pipeline.py --status
  
  # Reset pipeline state
  python src/run_pipeline.py --reset
  
  # Force re-run specific stages
  python src/run_pipeline.py --stages detection ocr --force
  
  # Run with custom configuration
  python src/run_pipeline.py --run-all --config custom_config.json
        """
    )
    
    # Main actions
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--run-all", 
        action="store_true",
        help="Run all pipeline stages in order"
    )
    action_group.add_argument(
        "--stages", 
        nargs="+",
        help="Run specific stages (e.g., preparation training detection)"
    )
    action_group.add_argument(
        "--status", 
        action="store_true",
        help="Show pipeline status and stage information"
    )
    action_group.add_argument(
        "--list-stages", 
        action="store_true",
        help="List all available stages"
    )
    action_group.add_argument(
        "--reset", 
        nargs="*",
        help="Reset pipeline state (all stages or specific ones)"
    )
    
    # Configuration options
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to custom configuration file"
    )
    parser.add_argument(
        "--data-root", 
        type=str,
        help="Override data root directory"
    )
    parser.add_argument(
        "--state-dir", 
        type=str,
        help="Override pipeline state directory"
    )
    
    # Execution options
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force re-run stages even if completed"
    )
    parser.add_argument(
        "--skip-completed", 
        action="store_true", 
        default=True,
        help="Skip already completed stages (default)"
    )
    parser.add_argument(
        "--no-skip-completed", 
        action="store_false", 
        dest="skip_completed",
        help="Re-run all stages regardless of completion status"
    )
    
    # Environment options
    parser.add_argument(
        "--multi-env", 
        action="store_true",
        help="Enable multi-environment mode (separate envs for YOLO/OCR)"
    )
    parser.add_argument(
        "--single-env", 
        action="store_true",
        help="Force single environment mode"
    )
    
    # Training options
    parser.add_argument(
        "--model", 
        type=str,
        help="YOLO model to use (e.g., yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)"
    )
    parser.add_argument(
        "--epochs", 
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int,
        help="Training batch size"
    )
    
    # Detection options
    parser.add_argument(
        "--detection-conf", 
        type=float,
        help="Detection confidence threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--detection-threshold", 
        type=float,
        help="Alias for detection confidence threshold"
    )
    
    # OCR options
    parser.add_argument(
        "--ocr-conf", 
        type=float,
        help="OCR confidence threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--ocr-threshold", 
        type=float,
        help="Alias for OCR confidence threshold"
    )
    
    # Output options
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet", "-q", 
        action="store_true",
        help="Suppress non-essential output"
    )
    parser.add_argument(
        "--json-output", 
        action="store_true",
        help="Output results in JSON format"
    )
    parser.add_argument(
        "--save-results", 
        type=str,
        help="Save execution results to file"
    )
    
    return parser


def setup_environment(args) -> None:
    """Setup environment variables based on arguments"""
    if args.multi_env:
        os.environ["PLCDP_MULTI_ENV"] = "1"
    elif args.single_env:
        os.environ["PLCDP_MULTI_ENV"] = "0"
    
    if args.verbose:
        os.environ["PLCDP_VERBOSE"] = "1"
    elif args.quiet:
        os.environ["PLCDP_QUIET"] = "1"


def load_custom_config(config_path: str) -> Dict[str, Any]:
    """Load custom configuration from file"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        if config_file.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_file.suffix}")


def print_stage_status(stage_status: Dict[str, Any], verbose: bool = False) -> None:
    """Print formatted stage status"""
    name = stage_status['name']
    completed = "V" if stage_status['completed'] else "○"
    env = stage_status['environment']
    
    print(f"  {completed} {name:<12} ({env})")
    
    if verbose:
        deps = stage_status.get('dependencies', [])
        if deps:
            print(f"    Dependencies: {', '.join(deps)}")
        
        progress = stage_status.get('progress', 0.0)
        if progress > 0:
            print(f"    Progress: {progress:.1%}")


def print_pipeline_status(status: Dict[str, Any], verbose: bool = False) -> None:
    """Print formatted pipeline status"""
    print(f"\nX Pipeline Status")
    print(f"Total stages: {status['total_stages']}")
    print(f"Completed: {status['completed_stages']}")
    print(f"Progress: {status['overall_progress']:.1%}")
    print(f"CI Mode: {status['ci_mode']}")
    print(f"State directory: {status['state_dir']}")
    
    print(f"\nX Stages (execution order):")
    for stage_status in status['stages']:
        print_stage_status(stage_status, verbose)


def print_execution_summary(summary: Dict[str, Any], json_output: bool = False) -> None:
    """Print execution summary"""
    if json_output:
        print(json.dumps(summary, indent=2))
        return
    
    success = "V" if summary['success'] else "X"
    print(f"\n{success} Pipeline Execution Summary")
    print(f"Success: {summary['success']}")
    print(f"Duration: {summary['total_duration']:.2f}s")
    print(f"Stages run: {summary['stages_run']}")
    print(f"Stages skipped: {summary['stages_skipped']}")
    print(f"CI Mode: {summary['ci_mode']}")
    
    if not summary['success']:
        print(f"\nX Failed stages:")
        for stage_name, result in summary['results'].items():
            if not result.get('skipped', False) and not result.get('success', True):
                error = result.get('error', 'Unknown error')
                print(f"  - {stage_name}: {error}")


def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup environment
    setup_environment(args)
    
    try:
        # Initialize stage manager
        project_root_path = project_root
        state_dir = Path(args.state_dir) if args.state_dir else None
        
        manager = StageManager(
            project_root=project_root_path,
            state_dir=state_dir
        )
        
        # Load configuration
        config = {}
        if args.config:
            config.update(load_custom_config(args.config))
        
        # Override config with command line arguments
        if args.data_root:
            config['data_root'] = args.data_root
        
        # Add stage-specific configurations from command line arguments
        if hasattr(args, 'model') and args.model:
            config.setdefault('training', {})['model'] = args.model
        if hasattr(args, 'epochs') and args.epochs:
            config.setdefault('training', {})['epochs'] = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size:
            config.setdefault('training', {})['batch_size'] = args.batch_size
        
        # Detection configuration
        detection_conf = args.detection_conf or args.detection_threshold
        if detection_conf:
            config.setdefault('detection', {})['confidence_threshold'] = detection_conf
        
        # OCR configuration
        ocr_conf = args.ocr_conf or args.ocr_threshold
        if ocr_conf:
            config.setdefault('ocr', {})['confidence_threshold'] = ocr_conf
        
        # Handle different actions
        if args.status:
            status = manager.get_pipeline_status()
            print_pipeline_status(status, args.verbose)
            
        elif args.list_stages:
            stages = manager.list_stages()
            print(f"\nX Available Stages ({len(stages)} total):")
            for stage in stages:
                completed = "V" if stage['completed'] else "○"
                deps = f" (deps: {', '.join(stage['dependencies'])})" if stage['dependencies'] else ""
                print(f"  {completed} {stage['name']:<12} - {stage['description']} ({stage['environment']}){deps}")
                
        elif args.reset is not None:
            stages_to_reset = args.reset if args.reset else None
            manager.reset_pipeline(stages_to_reset)
            if stages_to_reset:
                print(f"V Reset stages: {', '.join(stages_to_reset)}")
            else:
                print("V Reset all pipeline stages")
                
        elif args.run_all:
            print("X Running complete pipeline...")
            
            force_stages = None
            if args.force:
                # Force all stages if --force is used with --run-all
                force_stages = list(manager.stages.keys())
            
            summary = manager.run_stages(
                stage_names=None,  # Run all stages
                config=config,
                skip_completed=args.skip_completed,
                force_stages=force_stages
            )
            
            print_execution_summary(summary, args.json_output)
            
            # Save results if requested
            if args.save_results:
                with open(args.save_results, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"📄 Results saved to: {args.save_results}")
            
            # Exit with error code if pipeline failed
            if not summary['success']:
                sys.exit(1)
                
        elif args.stages:
            print(f"X Running stages: {', '.join(args.stages)}")
            
            force_stages = args.stages if args.force else None
            
            summary = manager.run_stages(
                stage_names=args.stages,
                config=config,
                skip_completed=args.skip_completed,
                force_stages=force_stages
            )
            
            print_execution_summary(summary, args.json_output)
            
            # Save results if requested
            if args.save_results:
                with open(args.save_results, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"📄 Results saved to: {args.save_results}")
            
            # Exit with error code if pipeline failed
            if not summary['success']:
                sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nX Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nX Pipeline error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
