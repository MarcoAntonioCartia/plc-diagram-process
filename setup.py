#!/usr/bin/env python3
"""
Complete setup script for PLC Diagram Processor.
This script handles:
1. Data directory structure creation
2. Data migration from old structure
3. Virtual environment creation
4. System dependency installation
5. Python dependency installation (correctly in venv)
6. Project configuration

Run with: python setup.py
"""

import os
import sys
import shutil
import subprocess
import argparse
import platform
from pathlib import Path
from typing import List, Tuple, Optional

class PLCSetup:
    def __init__(self, data_root: Optional[str] = None, dry_run: bool = False):
        self.project_root = Path(__file__).parent.absolute()
        self.data_root = Path(data_root).absolute() if data_root else self.project_root.parent / 'plc-data'
        self.venv_name = 'yolovenv'
        self.venv_path = self.project_root / self.venv_name
        self.dry_run = dry_run
        self.system = platform.system().lower()
        
        # Set up virtual environment paths
        if self.system == 'windows':
            self.venv_python = self.venv_path / 'Scripts' / 'python.exe'
            self.venv_pip = self.venv_path / 'Scripts' / 'pip.exe'
            self.venv_activate = self.venv_path / 'Scripts' / 'activate.bat'
        else:
            self.venv_python = self.venv_path / 'bin' / 'python'
            self.venv_pip = self.venv_path / 'bin' / 'pip'
            self.venv_activate = self.venv_path / 'bin' / 'activate'
        
        print(f"Project root: {self.project_root}")
        print(f"Data root: {self.data_root}")
        print(f"Virtual environment: {self.venv_path}")
        print(f"Virtual environment Python: {self.venv_python}")
        print(f"Virtual environment pip: {self.venv_pip}")
        print(f"System: {self.system}")
        if self.dry_run:
            print("DRY RUN MODE - No actual changes will be made")
        print()

    def run_command(self, command: List[str], description: str, shell: bool = False, use_venv: bool = False) -> bool:
        """Run a system command with error handling."""
        print(f"Running: {description}")
        
        # If use_venv is True and we have a venv, use the venv executables
        if use_venv and self.venv_path.exists():
            if command[0] == 'python' or command[0] == 'python3':
                command[0] = str(self.venv_python)
            elif command[0] == 'pip':
                command[0] = str(self.venv_pip)
        
        if self.dry_run:
            print(f"  DRY RUN: Would execute: {' '.join(command) if isinstance(command, list) else command}")
            if use_venv:
                print(f"  DRY RUN: Using virtual environment at: {self.venv_path}")
            return True
        
        try:
            if shell:
                result = subprocess.run(' '.join(command), shell=True, check=True, 
                                      capture_output=True, text=True)
            else:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
            
            if result.stdout:
                print(f"  Output: {result.stdout.strip()}")
            print(f"  Success: {description}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: {e}")
            if e.stderr:
                print(f"  Error details: {e.stderr.strip()}")
            return False
        except Exception as e:
            print(f"  ERROR: {e}")
            return False

    def install_system_dependencies(self) -> bool:
        """Install system-level dependencies based on the platform."""
        print("=== Installing System Dependencies ===")
        
        if self.system == 'linux':
            return self._install_linux_dependencies()
        elif self.system == 'darwin':
            return self._install_macos_dependencies()
        elif self.system == 'windows':
            return self._install_windows_dependencies()
        else:
            print(f"Unsupported system: {self.system}")
            return False

    def _install_linux_dependencies(self) -> bool:
        """Install dependencies on Linux."""
        print("Linux detected")
        
        # Detect package manager
        if shutil.which('apt'):
            print("Using APT package manager (Ubuntu/Debian)")
            commands = [
                (['sudo', 'apt', 'update'], "Updating package list"),
                (['sudo', 'apt', 'install', '-y', 'python3-dev', 'python3-pip', 'python3-venv', 'build-essential'], "Installing Python dev tools"),
                (['sudo', 'apt', 'install', '-y', 'poppler-utils'], "Installing Poppler utilities"),
                (['sudo', 'apt', 'install', '-y', 'libglib2.0-0', 'libsm6', 'libxrender1', 'libxext6'], "Installing OpenCV dependencies"),
            ]
        elif shutil.which('yum'):
            print("Using YUM package manager (CentOS/RHEL)")
            commands = [
                (['sudo', 'yum', 'install', '-y', 'python3-devel', 'python3-pip', 'gcc', 'gcc-c++', 'make'], "Installing Python dev tools"),
                (['sudo', 'yum', 'install', '-y', 'poppler-utils'], "Installing Poppler utilities"),
                (['sudo', 'yum', 'install', '-y', 'glib2-devel', 'libSM-devel', 'libXrender-devel', 'libXext-devel'], "Installing OpenCV dependencies"),
            ]
        elif shutil.which('dnf'):
            print("Using DNF package manager (Fedora)")
            commands = [
                (['sudo', 'dnf', 'install', '-y', 'python3-devel', 'python3-pip', 'gcc', 'gcc-c++', 'make'], "Installing Python dev tools"),
                (['sudo', 'dnf', 'install', '-y', 'poppler-utils'], "Installing Poppler utilities"),
                (['sudo', 'dnf', 'install', '-y', 'glib2-devel', 'libSM-devel', 'libXrender-devel', 'libXext-devel'], "Installing OpenCV dependencies"),
            ]
        else:
            print("Unknown Linux distribution. Please install manually:")
            print("- Python development headers (python3-dev/python3-devel)")
            print("- Build tools (build-essential/gcc/gcc-c++/make)")
            print("- Poppler utilities")
            print("- OpenCV system dependencies")
            return False
        
        for command, description in commands:
            if not self.run_command(command, description):
                return False
        return True

    def _install_macos_dependencies(self) -> bool:
        """Install dependencies on macOS."""
        print("macOS detected")
        
        # Check for Homebrew
        if not shutil.which('brew'):
            print("Installing Homebrew...")
            install_brew = ['/bin/bash', '-c', 
                          "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"]
            if not self.run_command(install_brew, "Installing Homebrew"):
                return False
        
        commands = [
            (['brew', 'install', 'poppler'], "Installing Poppler"),
            (['brew', 'install', 'opencv'], "Installing OpenCV"),
        ]
        
        # Check for Xcode Command Line Tools
        if not shutil.which('gcc'):
            commands.append((['xcode-select', '--install'], "Installing Xcode Command Line Tools"))
        
        for command, description in commands:
            if not self.run_command(command, description):
                return False
        return True

    def _install_windows_dependencies(self) -> bool:
        """Install dependencies on Windows."""
        print("Windows detected")
        print("Please ensure the following are installed manually:")
        print("1. Visual Studio Build Tools for C++ compilation")
        print("2. Poppler from: https://github.com/oschwartz10612/poppler-windows/releases")
        print("   Extract to: bin/poppler/Library/bin/ and add to PATH")
        
        if not self.dry_run:
            response = input("Have you installed the above dependencies? (y/n): ")
            if response.lower() != 'y':
                print("Please install the dependencies and run the setup again.")
                return False
        return True

    def create_virtual_environment(self) -> bool:
        """Create and setup the virtual environment."""
        print("=== Setting up Python Virtual Environment ===")
        
        if self.venv_path.exists():
            print(f"Virtual environment already exists at: {self.venv_path}")
            if not self.dry_run:
                response = input("Do you want to recreate it? (y/n): ")
                if response.lower() == 'y':
                    print("Removing existing virtual environment...")
                    shutil.rmtree(self.venv_path)
                else:
                    print("Using existing virtual environment")
                    return True
        
        # Create virtual environment using current Python interpreter
        if not self.run_command([sys.executable, '-m', 'venv', str(self.venv_path)], 
                               "Creating virtual environment"):
            return False
        
        # Verify the virtual environment was created correctly
        if not self.dry_run:
            if not self.venv_python.exists():
                print(f"ERROR: Virtual environment Python not found at {self.venv_python}")
                return False
            if not self.venv_pip.exists():
                print(f"ERROR: Virtual environment pip not found at {self.venv_pip}")
                return False
            
            print(f"✓ Virtual environment created successfully")
            print(f"  Python: {self.venv_python}")
            print(f"  Pip: {self.venv_pip}")
        
        return True

    def upgrade_venv_tools(self) -> bool:
        """Upgrade pip, setuptools, and wheel in the virtual environment."""
        print("=== Upgrading Virtual Environment Tools ===")
        
        if not self.venv_path.exists():
            print("ERROR: Virtual environment does not exist. Create it first.")
            return False
        
        # Use the virtual environment's pip directly
        commands = [
            ([str(self.venv_pip), 'install', '--upgrade', 'pip'], "Upgrading pip in venv"),
            ([str(self.venv_pip), 'install', '--upgrade', 'setuptools'], "Upgrading setuptools in venv"),
            ([str(self.venv_pip), 'install', '--upgrade', 'wheel'], "Upgrading wheel in venv"),
        ]
        
        for command, description in commands:
            if not self.run_command(command, description):
                return False
        
        # Verify pip version in venv
        if not self.dry_run:
            result = subprocess.run([str(self.venv_pip), '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Virtual environment pip version: {result.stdout.strip()}")
            else:
                print("WARNING: Could not verify pip version in virtual environment")
        
        return True

    def install_python_dependencies(self) -> bool:
        """Install Python dependencies from requirements.txt in the virtual environment."""
        print("=== Installing Python Dependencies in Virtual Environment ===")
        
        if not self.venv_path.exists():
            print("ERROR: Virtual environment does not exist. Create it first.")
            return False
        
        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            print(f"ERROR: requirements.txt not found at {requirements_file}")
            return False
        
        print(f"Installing dependencies from: {requirements_file}")
        print(f"Using pip from virtual environment: {self.venv_pip}")
        
        # Install requirements using the virtual environment's pip
        success = self.run_command([str(self.venv_pip), 'install', '-r', str(requirements_file)], 
                                 "Installing Python dependencies in virtual environment")
        
        if success and not self.dry_run:
            # Verify some key packages were installed in the venv
            print("Verifying installation in virtual environment...")
            test_packages = ['pandas', 'torch', 'ultralytics']
            
            for package in test_packages:
                result = subprocess.run([str(self.venv_python), '-c', f'import {package}; print(f"{package} imported successfully")'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  ✓ {package}: {result.stdout.strip()}")
                else:
                    print(f"  ✗ {package}: Failed to import")
        
        return success

    def create_data_structure(self) -> None:
        """Create the required directory structure in the data root."""
        print("=== Creating Data Directory Structure ===")
        
        directories = [
            'datasets',
            'models/pretrained',
            'models/custom',
            'raw/pdfs',
            'processed/images',
            'runs/detect'
        ]
        
        print(f"Creating data structure in: {self.data_root}")
        
        for directory in directories:
            dir_path = self.data_root / directory
            if not self.dry_run:
                dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  {'Would create' if self.dry_run else 'Created'}: {dir_path}")

    def find_existing_data(self) -> List[Tuple[Path, str]]:
        """Find existing data files that need to be migrated."""
        data_to_migrate = []
        
        # Check for data directory
        old_data_dir = self.project_root / 'data'
        if old_data_dir.exists():
            for item in old_data_dir.rglob('*'):
                try:
                    if item.is_file():
                        # Determine target location based on file type/location
                        relative_path = item.relative_to(old_data_dir)
                        
                        if 'dataset' in str(relative_path).lower():
                            target = f'datasets/{relative_path}'
                        elif item.suffix in ['.pt', '.pth', '.onnx']:
                            if 'yolo' in item.name.lower() and any(x in item.name.lower() for x in ['11n', '11s', '11m', '11l', '11x']):
                                target = f'models/pretrained/{item.name}'
                            else:
                                target = f'models/custom/{item.name}'
                        elif item.suffix in ['.pdf']:
                            target = f'raw/pdfs/{item.name}'
                        elif item.suffix in ['.png', '.jpg', '.jpeg']:
                            target = f'processed/images/{item.name}'
                        else:
                            # Keep original structure for other files
                            target = f'datasets/{relative_path}'
                        
                        data_to_migrate.append((item, target))
                except (OSError, PermissionError) as e:
                    # Skip files that can't be accessed (like broken symlinks)
                    print(f"    Warning: Skipping {item}: {e}")
                    continue
        
        # Check for runs directory
        old_runs_dir = self.project_root / 'runs'
        if old_runs_dir.exists():
            for item in old_runs_dir.rglob('*'):
                if item.is_file():
                    relative_path = item.relative_to(old_runs_dir)
                    target = f'runs/{relative_path}'
                    data_to_migrate.append((item, target))
        
        return data_to_migrate

    def migrate_data(self, data_to_migrate: List[Tuple[Path, str]]) -> None:
        """Migrate existing data to the new structure."""
        if not data_to_migrate:
            print("No data found to migrate.")
            return
        
        print(f"=== {'DRY RUN: ' if self.dry_run else ''}Migrating {len(data_to_migrate)} files ===")
        
        for source_path, target_relative in data_to_migrate:
            target_path = self.data_root / target_relative
            
            print(f"  {source_path} -> {target_path}")
            
            if not self.dry_run:
                # Create target directory if it doesn't exist
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file (don't move yet, in case something goes wrong)
                try:
                    shutil.copy2(source_path, target_path)
                except Exception as e:
                    print(f"    ERROR: Failed to copy {source_path}: {e}")

    def cleanup_old_data(self) -> None:
        """Remove old data directories after successful migration."""
        print("=== Cleaning up old data directories ===")
        
        old_dirs = [
            self.project_root / 'data',
            self.project_root / 'runs'
        ]
        
        for old_dir in old_dirs:
            if old_dir.exists():
                print(f"{'DRY RUN: ' if self.dry_run else ''}Removing old directory: {old_dir}")
                if not self.dry_run:
                    try:
                        shutil.rmtree(old_dir)
                        print(f"  Removed: {old_dir}")
                    except Exception as e:
                        print(f"  ERROR: Failed to remove {old_dir}: {e}")

    def create_activation_script(self) -> None:
        """Create convenience scripts for activating the environment."""
        print("=== Creating activation scripts ===")
        
        if self.system == 'windows':
            activate_script = self.project_root / 'activate.bat'
            content = f'''@echo off
echo Activating PLC Diagram Processor environment...
call "{self.venv_activate}"
echo Virtual environment activated: {self.venv_path}
echo Python executable: {self.venv_python}
echo Current directory: %cd%
'''
        else:
            activate_script = self.project_root / 'activate.sh'
            content = f'''#!/bin/bash
echo "Activating PLC Diagram Processor environment..."
source "{self.venv_activate}"
echo "Virtual environment activated: {self.venv_path}"
echo "Python executable: {self.venv_python}"
echo "Current directory: $(pwd)"
'''
        
        if not self.dry_run:
            with open(activate_script, 'w') as f:
                f.write(content)
            if self.system != 'windows':
                os.chmod(activate_script, 0o755)
        
        print(f"  {'Would create' if self.dry_run else 'Created'}: {activate_script}")

    def run_setup(self, migrate: bool = True, cleanup: bool = False) -> bool:
        """Run the complete setup process."""
        print("=== PLC Diagram Processor Complete Setup ===")
        print()
        
        steps = [
            ("Installing system dependencies", self.install_system_dependencies),
            ("Creating data directory structure", lambda: (self.create_data_structure(), True)[1]),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Upgrading virtual environment tools", self.upgrade_venv_tools),
            ("Installing Python dependencies in virtual environment", self.install_python_dependencies),
            ("Creating activation scripts", lambda: (self.create_activation_script(), True)[1]),
        ]
        
        if migrate:
            data_to_migrate = self.find_existing_data()
            if data_to_migrate:
                steps.insert(-2, ("Migrating existing data", lambda: (self.migrate_data(data_to_migrate), True)[1]))
                if cleanup:
                    steps.insert(-1, ("Cleaning up old directories", lambda: (self.cleanup_old_data(), True)[1]))
        
        for step_name, step_func in steps:
            print(f"\n{'='*60}")
            print(f"Step: {step_name}")
            print('='*60)
            
            if not step_func():
                print(f"ERROR: Failed at step: {step_name}")
                return False
        
        print("\n" + "="*60)
        print("SETUP COMPLETE!")
        print("="*60)
        print(f"Project root: {self.project_root}")
        print(f"Data directory: {self.data_root}")
        print(f"Virtual environment: {self.venv_path}")
        print(f"Virtual environment Python: {self.venv_python}")
        print(f"Virtual environment pip: {self.venv_pip}")
        print()
        print("To activate the environment:")
        if self.system == 'windows':
            print(f"  {self.project_root}\\activate.bat")
            print("  OR")
            print(f"  {self.venv_activate}")
        else:
            print(f"  source {self.project_root}/activate.sh")
            print("  OR")
            print(f"  source {self.venv_activate}")
        print()
        print("To verify the installation:")
        print(f"  {self.venv_python} -c \"import torch, ultralytics, pandas; print('All packages installed correctly!')\"")
        print()
        print("Your project is now ready for training and testing!")
        print("Next steps:")
        print("1. Activate the virtual environment")
        print("2. Run training: python src/detection/yolo11_train.py")
        print("3. Run inference: python src/detection/yolo11_infer.py")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Complete PLC Diagram Processor Setup')
    parser.add_argument('--migrate', action='store_true', default=True,
                       help='Migrate existing data from project directory (default: True)')
    parser.add_argument('--no-migrate', action='store_true',
                       help='Skip data migration')
    parser.add_argument('--cleanup', action='store_true',
                       help='Remove old data directories after migration')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually doing it')
    parser.add_argument('--data-root', type=str,
                       help='Custom data root directory (default: ../plc-data)')
    
    args = parser.parse_args()
    
    # Handle migrate logic
    migrate = args.migrate and not args.no_migrate
    
    setup = PLCSetup(data_root=args.data_root, dry_run=args.dry_run)
    
    success = setup.run_setup(migrate=migrate, cleanup=args.cleanup)
    
    if not success:
        print("\nSetup failed. Please check the errors above and try again.")
        sys.exit(1)

if __name__ == '__main__':
    main()