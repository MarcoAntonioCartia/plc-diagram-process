#!/usr/bin/env python3
"""
Storage Cleanup Utility for PLC Diagram Processor
Identifies and cleans up large temporary files to prevent disk space issues
"""

import os
import shutil
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

def get_directory_size(path):
    """Get the total size of a directory in bytes"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except (OSError, FileNotFoundError):
        pass
    return total_size

def format_size(size_bytes):
    """Format bytes to human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def audit_storage():
    """Audit current storage usage"""
    print("PLC Storage Audit")
    print("=" * 50)
    
    project_root = Path(__file__).resolve().parent
    plc_data_path = project_root.parent / "plc-data"
    
    if not plc_data_path.exists():
        print(f"❌ plc-data directory not found at: {plc_data_path}")
        return {}
    
    storage_info = {}
    
    # Check main directories
    main_dirs = ['runs', 'models', 'datasets', 'processed', 'raw']
    
    for dir_name in main_dirs:
        dir_path = plc_data_path / dir_name
        if dir_path.exists():
            size = get_directory_size(dir_path)
            storage_info[dir_name] = {
                'path': str(dir_path),
                'size_bytes': size,
                'size_formatted': format_size(size)
            }
            print(f"{dir_name:12}: {format_size(size):>10} ({dir_path})")
        else:
            print(f"{dir_name:12}: {'Not found':>10}")
    
    # Check environments
    env_path = project_root / "environments"
    if env_path.exists():
        size = get_directory_size(env_path)
        storage_info['environments'] = {
            'path': str(env_path),
            'size_bytes': size,
            'size_formatted': format_size(size)
        }
        print(f"{'environments':12}: {format_size(size):>10} ({env_path})")
    
    # Check temp directories
    temp_dirs = []
    try:
        import tempfile
        temp_root = Path(tempfile.gettempdir())
        temp_dirs = list(temp_root.glob("plc_worker_*"))
        if temp_dirs:
            temp_size = sum(get_directory_size(d) for d in temp_dirs)
            storage_info['temp_workers'] = {
                'path': str(temp_root),
                'size_bytes': temp_size,
                'size_formatted': format_size(temp_size),
                'count': len(temp_dirs)
            }
            print(f"{'temp_workers':12}: {format_size(temp_size):>10} ({len(temp_dirs)} directories)")
    except Exception:
        pass
    
    # Total size
    total_size = sum(info['size_bytes'] for info in storage_info.values())
    print("-" * 50)
    print(f"{'TOTAL':12}: {format_size(total_size):>10}")
    
    return storage_info

def cleanup_training_runs(keep_latest=2, dry_run=True):
    """Clean up old training runs, keeping only the latest N"""
    print(f"\nCleaning Training Runs (keep latest {keep_latest})")
    print("-" * 40)
    
    project_root = Path(__file__).resolve().parent
    runs_path = project_root.parent / "plc-data" / "runs" / "train"
    
    if not runs_path.exists():
        print("No training runs directory found")
        return 0
    
    # Get all training run directories
    run_dirs = [d for d in runs_path.iterdir() if d.is_dir()]
    
    if len(run_dirs) <= keep_latest:
        print(f"Only {len(run_dirs)} runs found, nothing to clean")
        return 0
    
    # Sort by modification time (newest first)
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Keep the latest N, mark the rest for deletion
    to_keep = run_dirs[:keep_latest]
    to_delete = run_dirs[keep_latest:]
    
    total_freed = 0
    
    print("Keeping:")
    for run_dir in to_keep:
        size = get_directory_size(run_dir)
        print(f"  ✅ {run_dir.name} ({format_size(size)})")
    
    print("\nRemoving:")
    for run_dir in to_delete:
        size = get_directory_size(run_dir)
        total_freed += size
        
        if dry_run:
            print(f"  🗑️  {run_dir.name} ({format_size(size)}) [DRY RUN]")
        else:
            try:
                shutil.rmtree(run_dir)
                print(f"  ✅ {run_dir.name} ({format_size(size)}) [DELETED]")
            except Exception as e:
                print(f"  ❌ {run_dir.name} ({format_size(size)}) [ERROR: {e}]")
                total_freed -= size
    
    print(f"\nTotal space that would be freed: {format_size(total_freed)}")
    return total_freed

def cleanup_cache_files(dry_run=True):
    """Clean up YOLO cache files"""
    print(f"\nCleaning Cache Files")
    print("-" * 20)
    
    project_root = Path(__file__).resolve().parent
    plc_data_path = project_root.parent / "plc-data"
    
    cache_files = []
    
    # Find all .cache files
    if plc_data_path.exists():
        cache_files.extend(plc_data_path.rglob("*.cache"))
    
    total_freed = 0
    
    for cache_file in cache_files:
        try:
            size = cache_file.stat().st_size
            total_freed += size
            
            if dry_run:
                print(f"  🗑️  {cache_file.name} ({format_size(size)}) [DRY RUN]")
            else:
                cache_file.unlink()
                print(f"  ✅ {cache_file.name} ({format_size(size)}) [DELETED]")
        except Exception as e:
            print(f"  ❌ {cache_file.name} [ERROR: {e}]")
            total_freed -= size
    
    if not cache_files:
        print("  No cache files found")
    
    print(f"\nTotal cache space that would be freed: {format_size(total_freed)}")
    return total_freed

def cleanup_temp_workers(dry_run=True):
    """Clean up temporary worker directories"""
    print(f"\nCleaning Temporary Worker Files")
    print("-" * 35)
    
    try:
        import tempfile
        temp_root = Path(tempfile.gettempdir())
        temp_dirs = list(temp_root.glob("plc_worker_*"))
        
        total_freed = 0
        
        for temp_dir in temp_dirs:
            try:
                size = get_directory_size(temp_dir)
                total_freed += size
                
                if dry_run:
                    print(f"  🗑️  {temp_dir.name} ({format_size(size)}) [DRY RUN]")
                else:
                    shutil.rmtree(temp_dir)
                    print(f"  ✅ {temp_dir.name} ({format_size(size)}) [DELETED]")
            except Exception as e:
                print(f"  ❌ {temp_dir.name} [ERROR: {e}]")
                total_freed -= size
        
        if not temp_dirs:
            print("  No temporary worker directories found")
        
        print(f"\nTotal temp space that would be freed: {format_size(total_freed)}")
        return total_freed
        
    except Exception as e:
        print(f"Error cleaning temp files: {e}")
        return 0

def main():
    """Main cleanup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PLC Storage Cleanup Utility')
    parser.add_argument('--audit', action='store_true', help='Audit current storage usage')
    parser.add_argument('--cleanup', action='store_true', help='Perform cleanup operations')
    parser.add_argument('--keep-runs', type=int, default=2, help='Number of training runs to keep (default: 2)')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Show what would be deleted without actually deleting')
    parser.add_argument('--force', action='store_true', help='Actually perform deletions (overrides --dry-run)')
    
    args = parser.parse_args()
    
    if args.force:
        args.dry_run = False
    
    if args.audit or not args.cleanup:
        storage_info = audit_storage()
        
        # Check if we're running low on space
        total_size = sum(info['size_bytes'] for info in storage_info.values())
        if total_size > 10 * 1024 * 1024 * 1024:  # 10GB
            print(f"\n⚠️  WARNING: Large storage usage detected ({format_size(total_size)})")
            print("Consider running cleanup operations")
    
    if args.cleanup:
        print(f"\nStarting Cleanup Operations {'(DRY RUN)' if args.dry_run else '(LIVE)'}")
        print("=" * 60)
        
        total_freed = 0
        
        # Clean training runs
        total_freed += cleanup_training_runs(keep_latest=args.keep_runs, dry_run=args.dry_run)
        
        # Clean cache files
        total_freed += cleanup_cache_files(dry_run=args.dry_run)
        
        # Clean temp workers
        total_freed += cleanup_temp_workers(dry_run=args.dry_run)
        
        print(f"\n{'=' * 60}")
        print(f"Total space that would be freed: {format_size(total_freed)}")
        
        if args.dry_run:
            print("\nTo actually perform cleanup, run with --force")
        else:
            print("\nCleanup completed!")

if __name__ == "__main__":
    main()
