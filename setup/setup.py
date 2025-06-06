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
import threading
import queue
import time
import yaml
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

class PLCSetup:
    def __init__(self, data_root: Optional[str] = None, dry_run: bool = False, parallel_jobs: int = 4):
        self.project_root = Path(__file__).parent.absolute()
        self.data_root = Path(data_root).absolute() if data_root else self.project_root.parent / 'plc-data'
        self.venv_name = 'yolovenv'
        self.venv_path = self.project_root / self.venv_name
        self.dry_run = dry_run
        self.system = platform.system().lower()
        self.parallel_jobs = max(1, min(parallel_jobs, 8))  # Limit between 1-8 jobs
        
        # Thread-safe progress tracking
        self.progress_lock = threading.Lock()
        self.completed_packages = 0
        self.total_packages = 0
        self.current_packages = set()
        
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
        print(f"Parallel installation jobs: {self.parallel_jobs}")
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
            print(f"  ✓ Success: {description}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ✗ ERROR: {e}")
            if e.stderr:
                print(f"  Error details: {e.stderr.strip()}")
            return False
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
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
        
        # Check if virtual environment already exists
        if self.venv_path.exists():
            print(f"Virtual environment already exists at: {self.venv_path}")
            
            # Verify it's a valid virtual environment
            if self.venv_python.exists() and self.venv_pip.exists():
                print("Existing virtual environment appears to be valid.")
                if not self.dry_run:
                    response = input("Do you want to recreate it? (y/n): ")
                    if response.lower() != 'y':
                        print("Using existing virtual environment")
                        return True
                else:
                    print("DRY RUN: Would use existing virtual environment")
                    return True
            else:
                print("Existing virtual environment appears to be corrupted.")
                print("Will remove and recreate it.")
            
            # Remove existing environment
            print("Removing existing virtual environment...")
            if not self.dry_run:
                try:
                    # On Windows, sometimes files are locked, so try multiple times
                    for attempt in range(3):
                        try:
                            shutil.rmtree(self.venv_path)
                            break
                        except (OSError, PermissionError) as e:
                            if attempt == 2:  # Last attempt
                                raise e
                            print(f"  Attempt {attempt + 1} failed, retrying...")
                            time.sleep(1)
                    print("  ✓ Old environment removed successfully")
                except Exception as e:
                    print(f"  ✗ ERROR: Could not remove existing environment: {e}")
                    print("  Please manually delete the directory and try again.")
                    return False
        
        # Create new virtual environment
        print("Creating new virtual environment...")
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
            
            print("✓ Virtual environment created successfully")
            print(f"  Python: {self.venv_python}")
            print(f"  Pip: {self.venv_pip}")
        
        return True

    def upgrade_venv_tools(self) -> bool:
        """Upgrade pip, setuptools, and wheel in the virtual environment."""
        print("=== Upgrading Virtual Environment Tools ===")
        
        if not self.venv_path.exists():
            print("ERROR: Virtual environment does not exist. Create it first.")
            return False
        
        # Upgrade tools one by one with clear output
        tools = [
            ('pip', 'pip package manager'),
            ('setuptools', 'Python setuptools'),
            ('wheel', 'wheel package format support')
        ]
        
        print(f"Upgrading {len(tools)} essential tools in virtual environment...")
        print()
        
        for i, (tool, description) in enumerate(tools, 1):
            print(f"[{i}/{len(tools)}] Upgrading {tool} ({description})...")
            
            success = self.run_command([str(self.venv_pip), 'install', '--upgrade', tool], 
                                     f"Upgrading {tool} in virtual environment")
            
            if success:
                print(f"  ✓ Successfully upgraded {tool}")
            else:
                print(f"  ✗ Failed to upgrade {tool}")
                return False
            print()
        
        # Verify final pip version
        if not self.dry_run:
            print("Verifying upgraded tools...")
            try:
                result = subprocess.run([str(self.venv_pip), '--version'], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print(f"✓ Final pip version: {result.stdout.strip()}")
                else:
                    print("⚠ Warning: Could not verify pip version")
            except subprocess.TimeoutExpired:
                print("⚠ Warning: Pip version check timed out")
            except Exception as e:
                print(f"⚠ Warning: Error checking pip version: {e}")
        
        print("✓ All virtual environment tools upgraded successfully")
        return True

    def parse_requirements(self, requirements_file: Path) -> List[str]:
        """Parse requirements.txt and extract package names."""
        packages = []
        try:
            with open(requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name (before ==, >=, <=, etc.)
                        package_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].split('>')[0].split('<')[0].split('[')[0]
                        packages.append(package_name.strip())
        except Exception as e:
            print(f"Warning: Could not parse requirements file: {e}")
        return packages

    def get_package_timeout(self, package: str) -> int:
        """Get appropriate timeout for different package types."""
        base_name = package.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].split('>')[0].split('<')[0].split('[')[0].lower()
        
        # Extended timeouts for heavy packages (in seconds)
        heavy_timeouts = {
            'torch': 3600,           # 60 minutes for PyTorch
            'torchvision': 2400,     # 40 minutes
            'torchaudio': 1800,      # 30 minutes
            'tensorflow': 3600,      # 60 minutes
            'ultralytics': 2400,     # 40 minutes for Ultralytics
            'opencv-python': 1800,   # 30 minutes
            'opencv-contrib-python': 2100,  # 35 minutes
            'paddleocr': 1800,       # 30 minutes
            'scipy': 1200,           # 20 minutes
            'numpy': 900,            # 15 minutes
            'pandas': 900,           # 15 minutes
            'pillow': 600,           # 10 minutes
            'matplotlib': 900,       # 15 minutes
            'scikit-learn': 1200,    # 20 minutes
        }
        
        # Check if this package needs extended timeout
        for heavy_pkg, timeout in heavy_timeouts.items():
            if heavy_pkg in base_name:
                return timeout
        
        # Default timeout for other packages
        return 600  # 10 minutes

    def install_single_package(self, package: str) -> Tuple[str, bool, str]:
        """Install a single package with appropriate timeout."""
        timeout = self.get_package_timeout(package)
        base_name = package.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].split('>')[0].split('<')[0].split('[')[0]
        
        print(f"Installing {base_name} (timeout: {timeout//60} minutes)...")
        
        try:
            # For very large packages, show live output instead of capturing it
            if timeout > 1800:  # > 30 minutes
                print(f"  Large package detected - showing live installation progress...")
                result = subprocess.run(
                    [str(self.venv_pip), 'install', package, '--no-cache-dir', '--timeout', '1200', '--verbose'],
                    timeout=timeout,
                    text=True
                    # Don't capture output for large packages - show it live
                )
                success = result.returncode == 0
                error_msg = f"Exit code: {result.returncode}" if not success else ""
            else:
                # Regular installation with captured output
                result = subprocess.run(
                    [str(self.venv_pip), 'install', package, '--no-cache-dir', '--timeout', '600'],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                success = result.returncode == 0
                error_msg = result.stderr.strip() if result.stderr else ""
            
            return package, success, error_msg
            
        except subprocess.TimeoutExpired:
            return package, False, f"Installation timed out ({timeout//60} minutes)"
        except Exception as e:
            return package, False, str(e)

    def display_progress_bar(self, current: int, total: int, package_name: str = "", width: int = 50):
        """Display a colored progress bar with package name."""
        if total == 0:
            return
        
        progress = current / total
        filled = int(width * progress)
        remaining = width - filled
        
        # Use simple ASCII characters that work on all terminals
        filled_char = '#'  # Green part (completed)
        empty_char = '-'   # Red part (remaining)
        
        bar = filled_char * filled + empty_char * remaining
        percentage = progress * 100
        
        if package_name:
            print(f"\rInstalling: {package_name:<30} [{bar}] {current}/{total} ({percentage:.1f}%)", end='', flush=True)
        else:
            print(f"\r[{bar}] {current}/{total} ({percentage:.1f}%)", end='', flush=True)

    def display_operation_progress(self, operation: str, step: int, total_steps: int, current_action: str = "", width: int = 50):
        """Display progress for general operations like environment creation/removal."""
        if total_steps == 0:
            return
        
        progress = step / total_steps
        filled = int(width * progress)
        remaining = width - filled
        
        filled_char = '#'  # Completed
        empty_char = '-'   # Remaining
        
        bar = filled_char * filled + empty_char * remaining
        percentage = progress * 100
        
        if current_action:
            print(f"\r{operation}: {current_action:<35} [{bar}] {step}/{total_steps} ({percentage:.1f}%)", end='', flush=True)
        else:
            print(f"\r{operation}: [{bar}] {step}/{total_steps} ({percentage:.1f}%)", end='', flush=True)

    def update_progress_display(self, package: str, completed: bool = False):
        """Thread-safe progress display update for parallel installation."""
        with self.progress_lock:
            if completed:
                self.completed_packages += 1
                self.current_packages.discard(package)
            else:
                self.current_packages.add(package)
            
            # Show currently installing packages (up to 2 to avoid clutter)
            current_list = list(self.current_packages)[:2]
            current_display = ", ".join(current_list)
            if len(self.current_packages) > 2:
                current_display += f" +{len(self.current_packages) - 2} more"
            
            # Display parallel progress
            self.display_progress_bar(self.completed_packages, self.total_packages, current_display)
            
            if completed:
                print(f"\nCompleted: {package}")

    def categorize_packages(self, packages: List[str]) -> Dict[str, List[str]]:
        """Categorize packages by installation complexity and dependencies."""
        # Ultra-heavy packages that need sequential installation with long timeouts
        ultra_heavy = {
            'torch', 'tensorflow', 'tensorflow-gpu'
        }
        
        # Heavy packages that should be installed sequentially
        heavy_packages = {
            'torchvision', 'torchaudio', 'ultralytics', 'paddleocr',
            'opencv-python', 'opencv-contrib-python', 'scipy', 
            'numpy', 'pandas', 'matplotlib', 'scikit-learn'
        }
        
        ultra_heavy_list = []
        heavy_list = []
        parallel_packages = []
        
        for package in packages:
            base_name = package.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].split('>')[0].split('<')[0].split('[')[0].lower()
            
            if any(ultra in base_name for ultra in ultra_heavy):
                ultra_heavy_list.append(package)
            elif any(heavy in base_name for heavy in heavy_packages):
                heavy_list.append(package)
            else:
                parallel_packages.append(package)
        
        return {
            'ultra_heavy': ultra_heavy_list,
            'heavy': heavy_list,
            'parallel': parallel_packages
        }

    def install_python_dependencies(self) -> bool:
        """Install Python dependencies with intelligent timeout management."""
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
        print()
        
        if self.dry_run:
            print("DRY RUN: Would install Python dependencies with extended timeouts")
            return True
        
        # Parse and categorize packages
        all_packages = self.parse_requirements(requirements_file)
        if not all_packages:
            print("Warning: Could not parse package list, using fallback installation")
            return self.run_command([str(self.venv_pip), 'install', '-r', str(requirements_file)], 
                                   "Installing Python dependencies in virtual environment")
        
        categorized = self.categorize_packages(all_packages)
        ultra_heavy_packages = categorized['ultra_heavy']
        heavy_packages = categorized['heavy']
        parallel_packages = categorized['parallel']
        
        print(f"Found {len(all_packages)} packages to install:")
        print(f"  - {len(ultra_heavy_packages)} ultra-heavy packages (PyTorch, TensorFlow - up to 60min each)")
        print(f"  - {len(heavy_packages)} heavy packages (computer vision, ML libs - up to 40min each)")
        print(f"  - {len(parallel_packages)} light packages (parallel installation)")
        print("=" * 80)
        
        failed_packages = []
        
        try:
            # Phase 1: Install ultra-heavy packages one by one with maximum care
            if ultra_heavy_packages:
                print(f"\nPhase 1: Installing {len(ultra_heavy_packages)} ultra-heavy packages (PyTorch/TensorFlow)...")
                print("These may take 30-60 minutes each. Please be patient...")
                print()
                
                for i, package in enumerate(ultra_heavy_packages, 1):
                    print(f"[{i}/{len(ultra_heavy_packages)}] Installing {package}...")
                    print("=" * 60)
                    
                    package_name, success, error_msg = self.install_single_package(package)
                    
                    if success:
                        print(f"✓ Successfully installed: {package}")
                    else:
                        print(f"✗ Failed to install: {package}")
                        print(f"  Error: {error_msg}")
                        failed_packages.append(package)
                    print()
                
                print("Phase 1 (ultra-heavy packages) complete.\n")
            
            # Phase 2: Install heavy packages sequentially
            if heavy_packages:
                print(f"Phase 2: Installing {len(heavy_packages)} heavy packages...")
                print("These may take 10-40 minutes each...")
                print()
                
                for i, package in enumerate(heavy_packages, 1):
                    print(f"[{i}/{len(heavy_packages)}] Installing {package}...")
                    
                    package_name, success, error_msg = self.install_single_package(package)
                    
                    if success:
                        print(f"✓ Successfully installed: {package}")
                    else:
                        print(f"✗ Failed to install: {package}")
                        print(f"  Error: {error_msg}")
                        failed_packages.append(package)
                    print()
                
                print("Phase 2 (heavy packages) complete.\n")
            
            # Phase 3: Install light packages in parallel
            if parallel_packages:
                print(f"Phase 3: Installing {len(parallel_packages)} light packages in parallel...")
                
                self.total_packages = len(parallel_packages)
                self.completed_packages = 0
                self.current_packages = set()
                
                with ThreadPoolExecutor(max_workers=self.parallel_jobs) as executor:
                    # Submit all parallel package installations
                    future_to_package = {
                        executor.submit(self.install_single_package, package): package 
                        for package in parallel_packages
                    }
                    
                    # Process completed installations
                    for future in as_completed(future_to_package):
                        package_name, success, error_msg = future.result()
                        
                        if success:
                            self.update_progress_display(package_name, completed=True)
                        else:
                            print(f"\n✗ Failed: {package_name} - {error_msg}")
                            failed_packages.append(package_name)
                            self.update_progress_display(package_name, completed=True)
                
                print("\nPhase 3 (light packages) complete.")
            
            print("\n" + "=" * 80)
            
            # Handle failed packages with extended timeout bulk installation
            if failed_packages:
                print(f"Warning: {len(failed_packages)} packages failed individual installation:")
                for pkg in failed_packages:
                    print(f"  - {pkg}")
                
                print("\nAttempting bulk installation of failed packages with extended timeout...")
                
                # Create requirements file for failed packages
                temp_req_file = self.project_root / 'failed_requirements.txt'
                try:
                    with open(temp_req_file, 'w') as f:
                        for pkg in failed_packages:
                            f.write(f"{pkg}\n")
                    
                    # Try bulk installation with very long timeout
                    print("Running bulk installation (this may take up to 2 hours)...")
                    print("Live output will be shown...")
                    
                    bulk_result = subprocess.run(
                        [str(self.venv_pip), 'install', '-r', str(temp_req_file), '--verbose', '--timeout', '1800'],
                        text=True,
                        timeout=7200  # 2 hour timeout for bulk installation
                    )
                    
                    temp_req_file.unlink()
                    
                    if bulk_result.returncode == 0:
                        print("✓ Bulk installation completed successfully")
                        failed_packages = []  # Clear failed packages list
                    else:
                        print("✗ Bulk installation failed. Some packages may need manual installation.")
                        
                except subprocess.TimeoutExpired:
                    print("⚠ Bulk installation timed out after 2 hours. Some packages may need manual installation.")
                except Exception as e:
                    print(f"Error in bulk installation attempt: {e}")
            else:
                print("✓ All Python dependencies installed successfully!")
            
        except KeyboardInterrupt:
            print("\n\nInstallation interrupted by user.")
            return False
        except Exception as e:
            print(f"\nError during installation: {e}")
            return False
        
        # Verify key packages
        print("\nVerifying installation in virtual environment...")
        test_packages = ['pandas', 'torch', 'ultralytics']
        
        for package in test_packages:
            try:
                result = subprocess.run([str(self.venv_python), '-c', f'import {package}; print(f"{package} imported successfully")'], 
                                      capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print(f"  ✓ SUCCESS: {package}")
                else:
                    print(f"  ✗ FAILED: {package} - could not import")
            except subprocess.TimeoutExpired:
                print(f"  ⚠ TIMEOUT: {package} - import test timed out")
            except Exception as e:
                print(f"  ✗ ERROR: {package} - {e}")
        
        # Return success if no packages failed or less than 25% failed
        success_rate = (len(all_packages) - len(failed_packages)) / len(all_packages)
        
        if success_rate >= 0.75:
            print(f"\n✓ Installation completed successfully! ({success_rate*100:.1f}% success rate)")
            return True
        else:
            print(f"\n⚠ Installation completed with issues. ({success_rate*100:.1f}% success rate)")
            print("Consider running the installation again or installing failed packages manually.")
            return False

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

    def load_download_config(self) -> Dict:
        """Load download configuration from config/download_config.yaml"""
        config_path = self.project_root / 'setup' / 'config' / 'download_config.yaml'
        
        if not config_path.exists():
            print(f"Warning: Download config not found at {config_path}")
            print("Using default configuration")
            return self._create_default_download_config()
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading download config: {e}")
            print("Using default configuration")
            return self._create_default_download_config()
    
    def _create_default_download_config(self) -> Dict:
        """Create default download configuration"""
        return {
            'onedrive': {
                'base_url': '',
                'dataset': {
                    'auto_select_latest': True,
                    'keep_old_versions': True
                }
            },
            'models': {
                'default_model': 'yolo11m.pt',
                'download_multiple': True,
                'verify_downloads': True
            },
            'setup': {
                'auto_download_dataset': False,
                'auto_download_models': False,
                'prompt_for_downloads': True,
                'show_empty_folder_warnings': True
            }
        }
    
    def setup_data_and_models(self) -> bool:
        """Setup data and model downloads"""
        print("=== Data and Model Setup ===")
        
        # Load download configuration
        config = self.load_download_config()
        
        if config['setup']['prompt_for_downloads']:
            return self.interactive_download_prompt(config)
        elif config['setup']['auto_download_dataset'] or config['setup']['auto_download_models']:
            return self.automatic_download(config)
        else:
            # Just show warnings about empty folders
            if config['setup']['show_empty_folder_warnings']:
                self.check_and_warn_empty_folders()
            return True
    
    def interactive_download_prompt(self, config: Dict) -> bool:
        """Interactive prompts for downloading data and models"""
        storage_backend = config.get('storage_backend', 'network_drive')
        backend_name = "Network Drive" if storage_backend == 'network_drive' else "OneDrive"
        
        print("\nData and Model Download Options:")
        print(f"1. Download datasets from {backend_name}")
        print("2. Download YOLO models")
        print("3. Both datasets and models")
        print("4. Skip downloads (setup empty folders only)")
        
        if self.dry_run:
            print("DRY RUN: Would prompt user for download options")
            return True
        
        while True:
            try:
                choice = input("\nChoose option (1-4): ").strip()
                
                if choice == "1":
                    return self.download_datasets_interactive(config)
                elif choice == "2":
                    return self.download_models_interactive(config)
                elif choice == "3":
                    dataset_success = self.download_datasets_interactive(config)
                    model_success = self.download_models_interactive(config)
                    return dataset_success and model_success
                elif choice == "4":
                    print("Skipping downloads...")
                    if config['setup']['show_empty_folder_warnings']:
                        self.check_and_warn_empty_folders()
                    return True
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
                    
            except KeyboardInterrupt:
                print("\nSetup interrupted by user.")
                return False
    
    def download_datasets_interactive(self, config: Dict) -> bool:
        """Interactive dataset download"""
        try:
            # Import managers here to avoid import issues during setup
            sys.path.append(str(self.project_root / 'src'))
            from utils.dataset_manager import DatasetManager
            
            # Import appropriate storage backend
            storage_backend = config.get('storage_backend', 'network_drive')
            
            if storage_backend == 'network_drive':
                from utils.network_drive_manager import NetworkDriveManager
                storage_manager = NetworkDriveManager(config)
                backend_name = "Network Drive"
            else:  # Legacy OneDrive support
                from utils.onedrive_manager import OneDriveManager
                storage_manager = OneDriveManager(config)
                backend_name = "OneDrive"
            
            print(f"\n=== Dataset Download ({backend_name}) ===")
            
            dataset_manager = DatasetManager(config)
            
            # List available datasets
            print(f"Checking available datasets on {backend_name}...")
            available_datasets = storage_manager.list_available_datasets()
            
            if not available_datasets:
                print("No datasets found")
                return False
            
            print(f"\nFound {len(available_datasets)} datasets:")
            for i, dataset in enumerate(available_datasets, 1):
                print(f"  {i}. {dataset['name']} ({dataset.get('date', 'Unknown date')})")
            
            print(f"  {len(available_datasets) + 1}. Download latest automatically")
            print(f"  {len(available_datasets) + 2}. Skip dataset download")
            
            while True:
                try:
                    choice = input(f"\nSelect dataset (1-{len(available_datasets) + 2}): ").strip()
                    choice_num = int(choice)
                    
                    if 1 <= choice_num <= len(available_datasets):
                        # Download specific dataset
                        selected_dataset = available_datasets[choice_num - 1]
                        dataset_path = storage_manager.download_dataset(selected_dataset['name'], use_latest=False)
                        
                        if dataset_path:
                            # Activate the dataset
                            dataset_name = Path(dataset_path).name
                            activation_method = dataset_manager.activate_dataset(dataset_name)
                            print(f"Dataset activated using {activation_method}")
                            return True
                        else:
                            print("Dataset download failed")
                            return False
                            
                    elif choice_num == len(available_datasets) + 1:
                        # Download latest
                        dataset_path = storage_manager.download_dataset(use_latest=True)
                        
                        if dataset_path:
                            # Activate the dataset
                            dataset_name = Path(dataset_path).name
                            activation_method = dataset_manager.activate_dataset(dataset_name)
                            print(f"Dataset activated using {activation_method}")
                            return True
                        else:
                            print("Dataset download failed")
                            return False
                            
                    elif choice_num == len(available_datasets) + 2:
                        # Skip
                        print("Skipping dataset download")
                        return True
                    else:
                        print(f"Invalid choice. Please enter 1-{len(available_datasets) + 2}")
                        
                except ValueError:
                    print("Please enter a valid number")
                except KeyboardInterrupt:
                    print("\nDataset download interrupted")
                    return False
                    
        except ImportError as e:
            print(f"Error importing dataset managers: {e}")
            print("Make sure the virtual environment is activated and dependencies are installed")
            return False
        except Exception as e:
            print(f"Error during dataset download: {e}")
            return False
    
    def download_models_interactive(self, config: Dict) -> bool:
        """Interactive model download"""
        try:
            # Import managers here to avoid import issues during setup
            sys.path.append(str(self.project_root / 'src'))
            from utils.model_manager import ModelManager
            
            print("\n=== Model Download ===")
            
            model_manager = ModelManager(config)
            
            # Interactive model selection
            selected_models = model_manager.interactive_model_selection()
            
            if not selected_models:
                print("No models selected")
                return True
            
            # Download selected models
            results = model_manager.download_multiple_models(selected_models)
            
            # Check results
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            
            if successful == total:
                print(f"All {total} models downloaded successfully!")
                return True
            elif successful > 0:
                print(f"{successful}/{total} models downloaded successfully")
                return True
            else:
                print("No models were downloaded successfully")
                return False
                
        except ImportError as e:
            print(f"Error importing model manager: {e}")
            print("Make sure the virtual environment is activated and dependencies are installed")
            return False
        except Exception as e:
            print(f"Error during model download: {e}")
            return False
    
    def automatic_download(self, config: Dict) -> bool:
        """Automatic download based on configuration"""
        success = True
        
        if config['setup']['auto_download_dataset']:
            print("Auto-downloading dataset...")
            success &= self.download_datasets_interactive(config)
        
        if config['setup']['auto_download_models']:
            print("Auto-downloading models...")
            success &= self.download_models_interactive(config)
        
        return success
    
    def check_and_warn_empty_folders(self):
        """Check for empty folders and warn user"""
        print("\n=== Checking Data Folders ===")
        
        folders_to_check = [
            ('datasets', 'No datasets found. Use scripts/manage_datasets.py to download datasets.'),
            ('models/pretrained', 'No pretrained models found. Use scripts/manage_models.py to download models.'),
            ('raw/pdfs', 'No input PDFs found. Place your PDF diagrams here for processing.'),
        ]
        
        empty_folders = []
        
        for folder, message in folders_to_check:
            folder_path = self.data_root / folder
            
            if not folder_path.exists() or not any(folder_path.iterdir()):
                empty_folders.append((folder, message))
        
        if empty_folders:
            print("Warning: The following folders are empty:")
            for folder, message in empty_folders:
                print(f"  - {folder}: {message}")
            
            print("\nTo download data later, use:")
            print("  python scripts/manage_datasets.py --interactive")
            print("  python scripts/manage_models.py --interactive")
        else:
            print("All data folders contain files")

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
            ("Setting up data and models", self.setup_data_and_models),
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
        
        print("\n" + "=" * 60)
        print("SETUP COMPLETE!")
        print("=" * 60)
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
    parser.add_argument('--parallel-jobs', type=int, default=4,
                       help='Number of parallel installation jobs (default: 4, max: 8)')
    
    args = parser.parse_args()
    
    # Handle migrate logic
    migrate = args.migrate and not args.no_migrate
    
    setup = PLCSetup(data_root=args.data_root, dry_run=args.dry_run, parallel_jobs=args.parallel_jobs)
    
    success = setup.run_setup(migrate=migrate, cleanup=args.cleanup)
    
    if not success:
        print("\nSetup failed. Please check the errors above and try again.")
        sys.exit(1)

if __name__ == '__main__':
    main()
