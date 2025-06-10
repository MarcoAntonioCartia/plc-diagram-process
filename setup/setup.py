#!/usr/bin/env python3
"""
Unified Setup Script for PLC Diagram Processor
Merges the best features from both setup.py and enhanced_setup.py while fixing critical issues.

This script handles:
1. System dependency installation (WSL, poppler, build tools)
2. Virtual environment creation and management
3. Robust package installation with multiple fallback strategies
4. GPU detection and PyTorch installation (CPU-first approach)
5. Data directory structure creation
6. Project configuration and activation scripts

Key improvements:
- Simplified PyTorch installation (CPU-first, GPU optional)
- Proven WSL poppler installation from old setup
- Enhanced modular architecture from enhanced setup
- Robust error handling and recovery strategies
- Better progress reporting and user guidance
"""

import os
import sys
import subprocess
import platform
import argparse
import time
import threading
import queue
import tempfile
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import our enhanced setup modules
setup_dir = Path(__file__).resolve().parent
sys.path.append(str(setup_dir))

try:
    from gpu_detector import GPUDetector
    from build_tools_installer import BuildToolsInstaller
    from package_installer import RobustPackageInstaller
except ImportError as e:
    print(f"Warning: Could not import enhanced modules: {e}")
    print("Falling back to basic functionality...")
    GPUDetector = None
    BuildToolsInstaller = None
    RobustPackageInstaller = None

class UnifiedPLCSetup:
    """Unified setup combining the best of both setup approaches"""
    
    def __init__(self, data_root: Optional[str] = None, dry_run: bool = False, parallel_jobs: int = 4):
        self.project_root = project_root
        self.data_root = Path(data_root).absolute() if data_root else self.project_root.parent / 'plc-data'
        self.venv_name = 'yolovenv'
        self.venv_path = self.project_root / self.venv_name
        self.dry_run = dry_run
        self.system = platform.system().lower()
        self.parallel_jobs = max(1, min(parallel_jobs, 8))
        
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
        
        # Initialize enhanced components if available
        self.gpu_detector = GPUDetector() if GPUDetector else None
        self.build_tools_installer = BuildToolsInstaller() if BuildToolsInstaller else None
        self.package_installer = RobustPackageInstaller() if RobustPackageInstaller else None
        
        print(f"Unified PLC Diagram Processor Setup")
        print(f"Project root: {self.project_root}")
        print(f"Data root: {self.data_root}")
        print(f"Virtual environment: {self.venv_path}")
        print(f"System: {self.system}")
        print(f"Parallel installation jobs: {self.parallel_jobs}")
        if self.dry_run:
            print("DRY RUN MODE - No actual changes will be made")
        print()

    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        min_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version < min_version:
            print(f"✗ Python {min_version[0]}.{min_version[1]}+ required, but {current_version[0]}.{current_version[1]} found")
            return False
        
        print(f"✓ Python {current_version[0]}.{current_version[1]} detected")
        return True

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

    def detect_system_capabilities(self) -> Dict:
        """Detect GPU and build tools capabilities (optional, non-blocking)"""
        print("=== System Capability Detection ===")
        
        capabilities = {
            "gpu_info": {"has_nvidia_gpu": False, "has_cuda": False, "recommended_pytorch": "cpu"},
            "build_tools_status": {"needs_installation": True, "installation_method": "manual"}
        }
        
        # Try GPU detection (non-blocking)
        if self.gpu_detector:
            try:
                print("Detecting GPU capabilities...")
                gpu_info = self.gpu_detector.detect_gpu_capabilities()
                capabilities["gpu_info"] = gpu_info
                print("✓ GPU detection completed")
            except Exception as e:
                print(f"⚠ GPU detection failed (non-critical): {e}")
                print("  Will use CPU-only PyTorch installation")
        else:
            print("⚠ GPU detector not available, using CPU-only PyTorch")
        
        # Try build tools detection (non-blocking)
        if self.build_tools_installer:
            try:
                print("Checking build tools status...")
                build_tools_status = self.build_tools_installer.check_build_tools_status()
                capabilities["build_tools_status"] = build_tools_status
                print("✓ Build tools check completed")
            except Exception as e:
                print(f"⚠ Build tools detection failed (non-critical): {e}")
                print("  May encounter compilation issues with some packages")
        else:
            print("⚠ Build tools installer not available")
        
        return capabilities

    def setup_build_environment(self, capabilities: Dict) -> bool:
        """Set up build environment if needed (non-blocking)"""
        print("\n=== Build Environment Setup ===")
        
        if not self.build_tools_installer:
            print("⚠ Build tools installer not available, skipping...")
            return True
        
        build_tools_status = capabilities["build_tools_status"]
        
        if build_tools_status["needs_installation"]:
            print("Build tools installation recommended...")
            
            if not self.dry_run:
                response = input("Install build tools automatically? (y/n/skip): ")
                if response.lower() == 'y':
                    try:
                        if self.build_tools_installer.install_build_tools():
                            print("✓ Build tools installed successfully")
                        else:
                            print("⚠ Build tools installation failed (non-critical)")
                    except Exception as e:
                        print(f"⚠ Build tools installation error (non-critical): {e}")
                elif response.lower() == 'skip':
                    print("Skipping build tools installation")
                else:
                    print("Build tools installation declined")
                    print("You may encounter compilation issues with some packages")
        else:
            print("✓ Build tools already available")
        
        return True

    # === WSL POPPLER INTEGRATION (FROM OLD SETUP - PROVEN) ===
    def _check_wsl_available(self) -> bool:
        """Check if WSL is available on Windows"""
        try:
            result = subprocess.run(['wsl', '--list'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_wsl_gpu_support(self) -> Dict[str, Any]:
        """Check if GPU is available in WSL for training (informational only)"""
        print("\n=== Checking WSL GPU Support (Informational) ===")
        
        gpu_info = {
            'available': False,
            'cuda_available': False,
            'nvidia_smi': False,
            'driver_version': None,
            'cuda_version': None,
            'gpu_name': None,
            'issues': []
        }
        
        if not self._check_wsl_available():
            gpu_info['issues'].append("WSL is not available")
            return gpu_info
        
        # Check for nvidia-smi in WSL (with timeout)
        print("Checking for NVIDIA GPU in WSL...")
        try:
            result = subprocess.run(
                ['wsl', '-e', 'bash', '-c', 'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                gpu_info['nvidia_smi'] = True
                output_lines = result.stdout.strip().split('\n')
                if output_lines:
                    parts = output_lines[0].split(', ')
                    if len(parts) >= 2:
                        gpu_info['gpu_name'] = parts[0].strip()
                        gpu_info['driver_version'] = parts[1].strip()
                        gpu_info['available'] = True
                        print(f"✓ Found GPU: {gpu_info['gpu_name']}")
                        print(f"  Driver version: {gpu_info['driver_version']}")
            else:
                gpu_info['issues'].append("nvidia-smi not found in WSL")
                print("ℹ NVIDIA GPU not detected in WSL")
        except subprocess.TimeoutExpired:
            gpu_info['issues'].append("nvidia-smi command timed out")
            print("⚠ GPU check timed out")
        except Exception as e:
            gpu_info['issues'].append(f"Error checking GPU: {str(e)}")
            print(f"⚠ Error checking GPU: {e}")
        
        return gpu_info

    def _install_poppler_via_wsl(self) -> bool:
        """Install poppler using WSL on Windows (from old setup - proven logic)"""
        print("\n=== Installing Poppler via WSL ===")
        
        # Check GPU support (informational, don't block installation)
        wsl_gpu_info = self._check_wsl_gpu_support()
        self.wsl_gpu_info = wsl_gpu_info
        
        if wsl_gpu_info['available']:
            print(f"ℹ WSL GPU ready: {wsl_gpu_info['gpu_name']}")
        else:
            print("ℹ WSL GPU not available - training will use CPU")
        
        # Check if poppler is already installed in WSL
        print("\nChecking if poppler is already installed in WSL...")
        check_cmd = ['wsl', '-e', 'bash', '-c', 'which pdftotext 2>/dev/null']
        
        try:
            result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                print(f"✓ Poppler is already installed in WSL at: {result.stdout.strip()}")
                return self._create_wsl_wrappers()
        except subprocess.TimeoutExpired:
            print("⚠ WSL check timed out - WSL might be starting up")
            time.sleep(3)
            # Try once more
            try:
                result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and result.stdout.strip():
                    print("✓ Poppler is already installed in WSL")
                    return self._create_wsl_wrappers()
            except:
                pass
        except Exception as e:
            print(f"⚠ Could not check for existing poppler: {e}")
        
        # Check if we can run commands without sudo
        print("\nChecking WSL sudo requirements...")
        test_sudo_cmd = ['wsl', '-e', 'bash', '-c', 'sudo -n true 2>/dev/null']
        try:
            result = subprocess.run(test_sudo_cmd, capture_output=True, text=True, timeout=5)
            passwordless_sudo = (result.returncode == 0)
        except:
            passwordless_sudo = False
        
        if passwordless_sudo:
            print("✓ Passwordless sudo detected, proceeding with automatic installation...")
            return self._run_wsl_poppler_install()
        
        # Need password authentication - try interactive installation
        print("\n" + "="*60)
        print("WSL Poppler Installation")
        print("="*60)
        print("\n⚠ WSL requires sudo password for package installation.")
        print("Choose installation method:")
        print("1. Automatic installation (will prompt for password)")
        print("2. Manual installation (guided)")
        print("3. Skip poppler installation")
        
        if self.dry_run:
            print("DRY RUN: Would prompt for installation method")
            return True
        
        while True:
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == "1":
                return self._install_with_password_prompt()
            elif choice == "2":
                return self._guide_manual_wsl_installation()
            elif choice == "3":
                print("Skipping poppler installation...")
                return True
            else:
                print("Invalid choice. Please select 1, 2, or 3.")

    def _install_with_password_prompt(self) -> bool:
        """Install poppler with interactive password prompt"""
        try:
            install_script = '''#!/bin/bash
echo "Updating package lists..."
sudo apt-get update
if [ $? -eq 0 ]; then
    echo "Installing poppler-utils..."
    sudo apt-get install -y poppler-utils
    if [ $? -eq 0 ]; then
        echo "SUCCESS"
    else
        echo "FAILED_INSTALL"
    fi
else
    echo "FAILED_UPDATE"
fi
'''
            
            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                f.write(install_script)
                script_path = f.name
            
            # Convert Windows path to WSL path
            wsl_script_path = subprocess.run(
                ['wsl', '-e', 'wslpath', '-u', script_path],
                capture_output=True, text=True
            ).stdout.strip()
            
            # Make script executable and run it
            print("\nPlease enter your WSL password when prompted:")
            print("(Password will not be displayed while typing)")
            print()
            
            # Run the script interactively so user can enter password
            result = subprocess.run(
                ['wsl', '-e', 'bash', wsl_script_path],
                text=True
            )
            
            # Clean up
            os.unlink(script_path)
            
            # Check if installation was successful
            if result.returncode == 0:
                check_cmd = ['wsl', '-e', 'bash', '-c', 'which pdftotext']
                check_result = subprocess.run(check_cmd, capture_output=True, text=True)
                if check_result.returncode == 0:
                    print("\n✓ Poppler installed successfully!")
                    return self._create_wsl_wrappers()
            
            print("\n✗ Poppler installation failed")
            return False
            
        except Exception as e:
            print(f"\nError during installation: {e}")
            return False

    def _run_wsl_poppler_install(self) -> bool:
        """Run the actual WSL poppler installation (for passwordless sudo)"""
        # Update package list
        print("Updating WSL package list...")
        update_cmd = ['wsl', '-e', 'bash', '-c', 'sudo apt-get update']
        try:
            subprocess.run(update_cmd, check=True, timeout=60)
            print("✓ Package list updated successfully")
        except subprocess.CalledProcessError:
            print("✗ Failed to update package list")
            return False
        except subprocess.TimeoutExpired:
            print("⚠ Package update timed out")
            return False
        
        # Install poppler-utils
        print("Installing poppler-utils in WSL...")
        install_cmd = ['wsl', '-e', 'bash', '-c', 'sudo apt-get install -y poppler-utils']
        try:
            subprocess.run(install_cmd, check=True, timeout=120)
            print("✓ Poppler-utils installed successfully")
        except subprocess.CalledProcessError:
            print("✗ Failed to install poppler-utils")
            return False
        except subprocess.TimeoutExpired:
            print("⚠ Poppler installation timed out")
            return False
        
        return self._create_wsl_wrappers()

    def _guide_manual_wsl_installation(self) -> bool:
        """Guide user through manual WSL installation with verification"""
        print("\n" + "="*70)
        print("Manual WSL Poppler Installation Guide")
        print("="*70)
        
        print("\nStep 1: Open a NEW terminal window (not this one)")
        print("Step 2: Enter WSL by typing: wsl")
        print("Step 3: Run these commands:")
        print("\n  sudo apt-get update")
        print("  sudo apt-get install -y poppler-utils")
        print("\nStep 4: Verify installation:")
        print("  which pdftotext")
        print("  (Should show: /usr/bin/pdftotext)")
        print("\nStep 5: Exit WSL:")
        print("  exit")
        print("\n" + "="*70)
        
        print("\n⚠ IMPORTANT: Complete these steps in a SEPARATE terminal window!")
        input("\nPress Enter when you've completed the installation...")
        
        # Verify installation with multiple attempts
        print("\nVerifying poppler installation...")
        
        for attempt in range(3):
            check_cmd = ['wsl', '-e', 'bash', '-c', 'which pdftotext 2>/dev/null']
            try:
                result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    print(f"✓ Poppler successfully installed at: {result.stdout.strip()}")
                    return self._create_wsl_wrappers()
            except:
                pass
            
            if attempt < 2:
                print(f"  Attempt {attempt + 1} failed, retrying...")
                time.sleep(2)
        
        print("\n✗ Could not verify poppler installation")
        print("\nTroubleshooting:")
        print("1. Make sure you completed all steps in WSL")
        print("2. Try running 'wsl --shutdown' and then retry")
        print("3. Check if WSL is properly installed")
        
        retry = input("\nWould you like to try verification again? (y/n): ")
        if retry.lower() == 'y':
            return self._guide_manual_wsl_installation()
        
        return False

    def _create_wsl_wrappers(self) -> bool:
        """Create Windows-accessible poppler wrappers"""
        print("\nCreating Windows-accessible poppler wrappers...")
        wrapper_dir = self.project_root / 'bin' / 'poppler'
        
        if not self.dry_run:
            try:
                wrapper_dir.mkdir(parents=True, exist_ok=True)
                
                # Create wrapper scripts for poppler tools
                poppler_tools = ['pdftotext', 'pdftoppm', 'pdfinfo', 'pdfimages']
                created_wrappers = []
                
                for tool in poppler_tools:
                    wrapper_path = wrapper_dir / f'{tool}.bat'
                    wrapper_content = f'''@echo off
wsl -e {tool} %*
'''
                    try:
                        with open(wrapper_path, 'w') as f:
                            f.write(wrapper_content)
                        created_wrappers.append(wrapper_path)
                        print(f"  ✓ Created wrapper: {wrapper_path.name}")
                    except Exception as e:
                        print(f"  ✗ Failed to create {wrapper_path.name}: {e}")
                
                if not created_wrappers:
                    print("\n✗ Failed to create any wrappers")
                    return False
                
                # Add to PATH for current session
                current_path = os.environ.get('PATH', '')
                if str(wrapper_dir) not in current_path:
                    os.environ['PATH'] = f"{wrapper_dir};{current_path}"
                    print(f"\n✓ Added {wrapper_dir} to PATH for current session")
                
                print("\n" + "="*60)
                print("✓ Poppler Installation Complete!")
                print("="*60)
                print(f"  Wrappers created: {len(created_wrappers)}/{len(poppler_tools)}")
                print(f"  Location: {wrapper_dir}")
                print("\n  To make permanent, add this to your system PATH:")
                print(f"  {wrapper_dir}")
                print("="*60)
                
                return True
                
            except Exception as e:
                print(f"\n✗ Error creating wrappers: {e}")
                return False
        
        return True

    def install_system_dependencies(self) -> bool:
        """Install system-level dependencies based on platform"""
        print("\n=== Installing System Dependencies ===")
        
        if self.system == 'windows':
            return self._install_windows_dependencies()
        elif self.system == 'linux':
            return self._install_linux_dependencies()
        elif self.system == 'darwin':
            return self._install_macos_dependencies()
        else:
            print(f"Unsupported system: {self.system}")
            return False

    def _install_windows_dependencies(self) -> bool:
        """Install dependencies on Windows"""
        print("Windows detected")
        
        # Check for WSL and install poppler
        print("\n1. Setting up Poppler...")
        
        if self._check_wsl_available():
            print("✓ WSL detected - will install poppler automatically")
            
            if not self._install_poppler_via_wsl():
                print("\n⚠ Failed to install poppler via WSL")
                print("Falling back to manual installation instructions...")
                return self._manual_poppler_instructions()
        else:
            print("⚠ WSL not detected")
            print("\nWSL (Windows Subsystem for Linux) is recommended for automatic poppler installation.")
            print("To install WSL:")
            print("  1. Open PowerShell as Administrator")
            print("  2. Run: wsl --install")
            print("  3. Restart your computer")
            print("  4. Run this setup again")
            
            if not self.dry_run:
                response = input("\nDo you want to continue with manual poppler installation? (y/n): ")
                if response.lower() != 'y':
                    print("\nPlease install WSL and run setup again for automatic installation.")
                    return False
            
            return self._manual_poppler_instructions()
        
        return True

    def _install_linux_dependencies(self) -> bool:
        """Install dependencies on Linux"""
        print("Linux detected")
        
        if shutil.which('apt'):
            print("Using APT package manager")
            commands = [
                (['sudo', 'apt', 'update'], "Updating package list"),
                (['sudo', 'apt', 'install', '-y', 'python3-dev', 'python3-pip', 'python3-venv', 'build-essential'], "Installing dev tools"),
                (['sudo', 'apt', 'install', '-y', 'poppler-utils'], "Installing Poppler"),
                (['sudo', 'apt', 'install', '-y', 'libglib2.0-0', 'libsm6', 'libxrender1', 'libxext6'], "Installing OpenCV dependencies"),
            ]
        elif shutil.which('yum'):
            print("Using YUM package manager")
            commands = [
                (['sudo', 'yum', 'install', '-y', 'python3-devel', 'python3-pip', 'gcc', 'gcc-c++', 'make'], "Installing dev tools"),
                (['sudo', 'yum', 'install', '-y', 'poppler-utils'], "Installing Poppler"),
                (['sudo', 'yum', 'install', '-y', 'glib2-devel', 'libSM-devel', 'libXrender-devel', 'libXext-devel'], "Installing dependencies"),
            ]
        elif shutil.which('dnf'):
            print("Using DNF package manager")
            commands = [
                (['sudo', 'dnf', 'install', '-y', 'python3-devel', 'python3-pip', 'gcc', 'gcc-c++', 'make'], "Installing dev tools"),
                (['sudo', 'dnf', 'install', '-y', 'poppler-utils'], "Installing Poppler"),
                (['sudo', 'dnf', 'install', '-y', 'glib2-devel', 'libSM-devel', 'libXrender-devel', 'libXext-devel'], "Installing dependencies"),
            ]
        else:
            print("Unknown Linux distribution - please install dependencies manually")
            return False
        
        return self._run_commands(commands)

    def _install_macos_dependencies(self) -> bool:
        """Install dependencies on macOS"""
        print("macOS detected")
        
        if not shutil.which('brew'):
            print("Installing Homebrew...")
            if not self.run_command(['/bin/bash', '-c', 
                          "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"], 
                          "Installing Homebrew"):
                return False
        
        commands = [
            (['brew', 'install', 'poppler'], "Installing Poppler"),
            (['brew', 'install', 'opencv'], "Installing OpenCV"),
        ]
        
        if not shutil.which('gcc'):
            commands.append((['xcode-select', '--install'], "Installing Xcode Command Line Tools"))
        
        return self._run_commands(commands)

    def _run_commands(self, commands: List[Tuple[List[str], str]]) -> bool:
        """Run multiple commands"""
        for command, description in commands:
            if not self.run_command(command, description):
                return False
        return True

    def _manual_poppler_instructions(self) -> bool:
        """Provide manual poppler installation instructions"""
        print("\n=== Manual Poppler Installation Required ===")
        print("Please install Poppler manually:")
        print("1. Download from: https://github.com/oschwartz10612/poppler-windows/releases")
        print("2. Extract the archive")
        print("3. Add the 'bin' folder to your system PATH")
        print("   Example: C:\\poppler-xx.xx.x\\Library\\bin")
        
        if not self.dry_run:
            response = input("\nHave you installed Poppler manually? (y/n): ")
            if response.lower() != 'y':
                print("Please install Poppler and run the setup again.")
                return False
        
        return True

    def create_virtual_environment(self) -> bool:
        """Create virtual environment"""
        print("\n=== Virtual Environment Setup ===")
        
        if self.venv_path.exists():
            print(f"Virtual environment already exists at: {self.venv_path}")
            
            if self.venv_python.exists():
                print("✓ Existing virtual environment appears valid")
                if not self.dry_run:
                    response = input("Recreate virtual environment? (y/n): ")
                    if response.lower() != 'y':
                        return True
                else:
                    print("DRY RUN: Would use existing virtual environment")
                    return True
            
            print("Removing existing virtual environment...")
            if not self.dry_run:
                try:
                    shutil.rmtree(self.venv_path)
                except Exception as e:
                    print(f"✗ Failed to remove existing environment: {e}")
                    return False
        
        print("Creating virtual environment...")
        try:
            if not self.dry_run:
                subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)], check=True)
            print("✓ Virtual environment created successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create virtual environment: {e}")
            return False

    def upgrade_pip_tools(self) -> bool:
        """Upgrade pip and essential tools in virtual environment"""
        print("\n=== Upgrading Virtual Environment Tools ===")
        
        if not self.venv_python.exists() and not self.dry_run:
            print("✗ Virtual environment not found")
            return False
        
        tools = ["pip", "setuptools", "wheel"]
        
        for tool in tools:
            print(f"Upgrading {tool}...")
            try:
                if self.dry_run:
                    print(f"  DRY RUN: Would upgrade {tool}")
                    continue
                
                if tool == "pip":
                    # Use python -m pip for pip upgrades
                    subprocess.run([
                        str(self.venv_python), "-m", "pip", "install", "--upgrade", "pip"
                    ], check=True, capture_output=True)
                else:
                    subprocess.run([
                        str(self.venv_pip), "install", "--upgrade", tool
                    ], check=True, capture_output=True)
                print(f"  ✓ {tool} upgraded successfully")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to upgrade {tool}: {e}")
                return False
        
        return True

    # === PYTORCH INSTALLATION (CPU-FIRST APPROACH) ===
    def install_pytorch(self, capabilities: Dict) -> bool:
        """Install PyTorch with CPU-first approach, GPU optional"""
        print("\n=== PyTorch Installation (CPU-First Approach) ===")
        
        # Always start with CPU version for reliability
        print("Installing PyTorch CPU version first...")
        
        try:
            if self.dry_run:
                print("  DRY RUN: Would install PyTorch CPU version")
            else:
                subprocess.run([
                    str(self.venv_pip), "install", "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cpu"
                ], check=True, timeout=1800)  # 30 minute timeout
                print("✓ PyTorch CPU version installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ PyTorch CPU installation failed: {e}")
            return False
        except subprocess.TimeoutExpired:
            print("⚠ PyTorch installation timed out")
            return False
        
        # Check if GPU upgrade is available and desired
        gpu_info = capabilities.get("gpu_info", {})
        if gpu_info.get("has_nvidia_gpu") and gpu_info.get("has_cuda"):
            print(f"\n🎯 GPU detected: {gpu_info.get('gpu_models', ['Unknown'])[0]}")
            print(f"   CUDA version: {gpu_info.get('cuda_version', 'Unknown')}")
            
            if not self.dry_run:
                response = input("\nUpgrade to GPU-enabled PyTorch? (y/n): ")
                if response.lower() == 'y':
                    return self._upgrade_pytorch_to_gpu(gpu_info)
            else:
                print("  DRY RUN: Would offer GPU PyTorch upgrade")
        
        return True

    def _upgrade_pytorch_to_gpu(self, gpu_info: Dict) -> bool:
        """Upgrade PyTorch to GPU version"""
        print("\nUpgrading PyTorch to GPU version...")
        
        # Determine CUDA version for PyTorch
        cuda_version = gpu_info.get("cuda_version", "11.8")
        
        if cuda_version.startswith("12"):
            index_url = "https://download.pytorch.org/whl/cu121"
            print("  Using CUDA 12.x PyTorch")
        else:
            index_url = "https://download.pytorch.org/whl/cu118"
            print("  Using CUDA 11.x PyTorch")
        
        try:
            subprocess.run([
                str(self.venv_pip), "install", "--upgrade", "torch", "torchvision", "torchaudio",
                "--index-url", index_url
            ], check=True, timeout=1800)
            
            print("✓ PyTorch GPU version installed successfully")
            
            # Verify GPU availability
            try:
                result = subprocess.run([
                    str(self.venv_python), "-c", 
                    "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print("  GPU verification:")
                    for line in result.stdout.strip().split('\n'):
                        print(f"    {line}")
                else:
                    print("  ⚠ Could not verify GPU functionality")
            except Exception as e:
                print(f"  ⚠ GPU verification failed: {e}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ PyTorch GPU upgrade failed: {e}")
            print("  Continuing with CPU version...")
            return True  # Don't fail setup for GPU upgrade failure
        except subprocess.TimeoutExpired:
            print("⚠ PyTorch GPU upgrade timed out")
            print("  Continuing with CPU version...")
            return True

    # === PACKAGE INSTALLATION (ROBUST APPROACH FROM OLD SETUP) ===
    def parse_requirements(self, requirements_file: Path) -> List[str]:
        """Parse requirements.txt and extract package names"""
        packages = []
        try:
            with open(requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name (before ==, >=, <=, etc.)
                        package_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].split('>')[0].split('<')[0].split('[')[0]
                        packages.append(line)  # Keep full specification
        except Exception as e:
            print(f"Warning: Could not parse requirements file: {e}")
        return packages

    def categorize_packages(self, packages: List[str]) -> Dict[str, List[str]]:
        """Categorize packages by installation complexity"""
        # Exclude PyTorch packages (already installed)
        pytorch_packages = {"torch", "torchvision", "torchaudio"}
        
        # Heavy packages that should be installed sequentially
        heavy_packages = {
            "ultralytics", "paddlepaddle", "paddleocr", "opencv-python", 
            "scipy", "numpy", "pandas", "matplotlib", "transformers"
        }
        
        heavy_list = []
        parallel_packages = []
        
        for package in packages:
            base_name = package.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].split('>')[0].split('<')[0].split('[')[0].lower()
            
            # Skip PyTorch packages
            if base_name in pytorch_packages:
                continue
            
            if any(heavy in base_name for heavy in heavy_packages):
                heavy_list.append(package)
            else:
                parallel_packages.append(package)
        
        return {
            'heavy': heavy_list,
            'parallel': parallel_packages
        }

    def install_single_package(self, package: str) -> Tuple[str, bool, str]:
        """Install a single package with timeout"""
        base_name = package.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].split('>')[0].split('<')[0].split('[')[0]
        
        # Determine timeout based on package
        if base_name.lower() in ['ultralytics', 'paddlepaddle', 'paddleocr', 'transformers']:
            timeout = 1800  # 30 minutes
        elif base_name.lower() in ['opencv-python', 'scipy', 'numpy', 'pandas']:
            timeout = 900   # 15 minutes
        else:
            timeout = 300   # 5 minutes
        
        try:
            if self.dry_run:
                print(f"  DRY RUN: Would install {package}")
                return package, True, ""
            
            result = subprocess.run(
                [str(self.venv_pip), 'install', package, '--no-cache-dir'],
                capture_output=True, text=True, timeout=timeout
            )
            
            success = result.returncode == 0
            error_msg = result.stderr.strip() if result.stderr else ""
            
            return package, success, error_msg
            
        except subprocess.TimeoutExpired:
            return package, False, f"Installation timed out ({timeout//60} minutes)"
        except Exception as e:
            return package, False, str(e)

    def update_progress_display(self, package: str, completed: bool = False):
        """Thread-safe progress display update"""
        with self.progress_lock:
            if completed:
                self.completed_packages += 1
                self.current_packages.discard(package)
            else:
                self.current_packages.add(package)
            
            # Show progress
            if self.total_packages > 0:
                progress = self.completed_packages / self.total_packages
                filled = int(50 * progress)
                bar = '#' * filled + '-' * (50 - filled)
                print(f"\r[{bar}] {self.completed_packages}/{self.total_packages} ({progress*100:.1f}%)", end='', flush=True)
            
            if completed:
                print(f"\n  ✓ {package}")

    def install_other_packages(self) -> bool:
        """Install other packages using robust strategies"""
        print("\n=== Installing Other Dependencies ===")
        
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            print(f"✗ Requirements file not found: {requirements_file}")
            return False
        
        # Parse and categorize packages
        all_packages = self.parse_requirements(requirements_file)
        categorized = self.categorize_packages(all_packages)
        
        heavy_packages = categorized['heavy']
        parallel_packages = categorized['parallel']
        
        print(f"Found {len(all_packages)} packages to install:")
        print(f"  - {len(heavy_packages)} heavy packages (sequential installation)")
        print(f"  - {len(parallel_packages)} light packages (parallel installation)")
        
        failed_packages = []
        
        # Phase 1: Install heavy packages sequentially
        if heavy_packages:
            print(f"\nPhase 1: Installing {len(heavy_packages)} heavy packages...")
            
            for i, package in enumerate(heavy_packages, 1):
                print(f"[{i}/{len(heavy_packages)}] Installing {package}...")
                
                package_name, success, error_msg = self.install_single_package(package)
                
                if success:
                    print(f"  ✓ Successfully installed: {package}")
                else:
                    print(f"  ✗ Failed to install: {package}")
                    if error_msg:
                        print(f"    Error: {error_msg}")
                    failed_packages.append(package)
        
        # Phase 2: Install light packages in parallel
        if parallel_packages:
            print(f"\nPhase 2: Installing {len(parallel_packages)} light packages in parallel...")
            
            self.total_packages = len(parallel_packages)
            self.completed_packages = 0
            self.current_packages = set()
            
            with ThreadPoolExecutor(max_workers=self.parallel_jobs) as executor:
                future_to_package = {
                    executor.submit(self.install_single_package, package): package 
                    for package in parallel_packages
                }
                
                for future in as_completed(future_to_package):
                    package_name, success, error_msg = future.result()
                    
                    if success:
                        self.update_progress_display(package_name, completed=True)
                    else:
                        print(f"\n  ✗ Failed: {package_name} - {error_msg}")
                        failed_packages.append(package_name)
                        self.update_progress_display(package_name, completed=True)
            
            print()  # New line after progress bar
        
        # Summary
        print(f"\nInstallation Summary:")
        print(f"  ✓ Successful: {len(all_packages) - len(failed_packages)}")
        print(f"  ✗ Failed: {len(failed_packages)}")
        
        if failed_packages:
            print(f"\nFailed packages:")
            for pkg in failed_packages:
                print(f"  - {pkg}")
            
            # Try bulk installation of failed packages
            if not self.dry_run:
                response = input("\nAttempt bulk installation of failed packages? (y/n): ")
                if response.lower() == 'y':
                    return self._bulk_install_failed_packages(failed_packages)
        
        # Return success if less than 25% failed
        success_rate = (len(all_packages) - len(failed_packages)) / len(all_packages) if all_packages else 1.0
        return success_rate >= 0.75

    def _bulk_install_failed_packages(self, failed_packages: List[str]) -> bool:
        """Attempt bulk installation of failed packages"""
        print("\nAttempting bulk installation of failed packages...")
        
        try:
            # Create temporary requirements file
            temp_req_file = self.project_root / 'temp_failed_requirements.txt'
            with open(temp_req_file, 'w') as f:
                for pkg in failed_packages:
                    f.write(f"{pkg}\n")
            
            # Try bulk installation with extended timeout
            result = subprocess.run(
                [str(self.venv_pip), 'install', '-r', str(temp_req_file), '--verbose'],
                text=True, timeout=3600  # 1 hour timeout
            )
            
            # Clean up
            temp_req_file.unlink()
            
            if result.returncode == 0:
                print("✓ Bulk installation completed successfully")
                return True
            else:
                print("✗ Bulk installation failed")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠ Bulk installation timed out")
            return False
        except Exception as e:
            print(f"✗ Bulk installation error: {e}")
            return False

    def verify_installation(self) -> bool:
        """Verify that key packages are working"""
        print("\n=== Installation Verification ===")
        
        test_packages = [
            ("torch", "PyTorch"),
            ("cv2", "OpenCV"),
            ("pandas", "Pandas"),
            ("numpy", "NumPy")
        ]
        
        failed_packages = []
        
        for package, description in test_packages:
            try:
                if self.dry_run:
                    print(f"  DRY RUN: Would test {description}")
                    continue
                
                result = subprocess.run([
                    str(self.venv_python), "-c", f"import {package}; print('✓ {description} working')"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print(result.stdout.strip())
                else:
                    print(f"✗ {description} failed to import")
                    failed_packages.append(description)
            except subprocess.TimeoutExpired:
                print(f"⚠ {description} import test timed out")
                failed_packages.append(description)
            except Exception as e:
                print(f"✗ {description} test failed: {e}")
                failed_packages.append(description)
        
        if failed_packages:
            print(f"\n⚠ {len(failed_packages)} packages failed verification:")
            for pkg in failed_packages:
                print(f"  - {pkg}")
            return False
        else:
            print("\n✓ All key packages verified successfully")
            return True

    def setup_data_directories(self) -> bool:
        """Set up data directory structure"""
        print("\n=== Data Directory Setup ===")
        
        directories = [
            self.data_root / "datasets",
            self.data_root / "models" / "pretrained",
            self.data_root / "models" / "custom",
            self.data_root / "processed",
            self.data_root / "raw" / "pdfs",
            self.data_root / "runs"
        ]
        
        print(f"Creating data directories in: {self.data_root}")
        
        for directory in directories:
            try:
                if not self.dry_run:
                    directory.mkdir(parents=True, exist_ok=True)
                print(f"  ✓ {directory}")
            except Exception as e:
                print(f"  ✗ Failed to create {directory}: {e}")
                return False
        
        return True

    def create_activation_scripts(self) -> bool:
        """Create activation scripts"""
        print("\n=== Creating Activation Scripts ===")
        
        if self.system == 'windows':
            activate_script = self.project_root / 'activate.bat'
            content = f'''@echo off
echo Activating PLC Diagram Processor environment...
call "{self.venv_activate}"
echo Virtual environment activated!
echo Python: {self.venv_python}
'''
        else:
            activate_script = self.project_root / 'activate.sh'
            content = f'''#!/bin/bash
echo "Activating PLC Diagram Processor environment..."
source "{self.venv_activate}"
echo "Virtual environment activated!"
echo "Python: {self.venv_python}"
'''
        
        try:
            if not self.dry_run:
                with open(activate_script, 'w') as f:
                    f.write(content)
                
                if self.system != 'windows':
                    os.chmod(activate_script, 0o755)
            
            print(f"  ✓ Created: {activate_script}")
            return True
        except Exception as e:
            print(f"  ✗ Failed to create activation script: {e}")
            return False

    def run_complete_setup(self) -> bool:
        """Run the complete unified setup process"""
        print("Unified PLC Diagram Processor Setup")
        print("=" * 60)
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Detecting system capabilities", lambda: (self.detect_system_capabilities(), True)[1]),
            ("Installing system dependencies", self.install_system_dependencies),
            ("Setting up build environment", lambda: self.setup_build_environment(self.capabilities)),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Upgrading pip tools", self.upgrade_pip_tools),
            ("Installing PyTorch (CPU-first)", lambda: self.install_pytorch(self.capabilities)),
            ("Installing other packages", self.install_other_packages),
            ("Setting up data directories", self.setup_data_directories),
            ("Creating activation scripts", self.create_activation_scripts),
            ("Verifying installation", self.verify_installation),
        ]
        
        # Store capabilities for later steps
        self.capabilities = None
        
        for i, (step_name, step_func) in enumerate(steps, 1):
            print(f"\n[{i}/{len(steps)}] {step_name}")
            print("-" * 50)
            
            if step_name == "Detecting system capabilities":
                self.capabilities = self.detect_system_capabilities()
                continue
            
            try:
                if not step_func():
                    print(f"\n✗ Setup failed at step: {step_name}")
                    return False
            except Exception as e:
                print(f"\n✗ Setup failed at step: {step_name}")
                print(f"Error: {e}")
                return False
        
        print("\n" + "=" * 60)
        print("✓ UNIFIED SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Show summary
        gpu_info = self.capabilities.get("gpu_info", {}) if self.capabilities else {}
        if gpu_info.get("has_nvidia_gpu"):
            print(f"\n🎯 GPU Status: {gpu_info.get('gpu_models', ['Unknown'])[0]} detected")
        else:
            print("\n💻 GPU Status: CPU-only (no CUDA GPU detected)")
        
        # Show WSL GPU status if available
        if hasattr(self, 'wsl_gpu_info') and self.wsl_gpu_info.get('available'):
            print(f"🐧 WSL GPU Status: {self.wsl_gpu_info['gpu_name']} ready for training")
        
        print(f"\nProject ready at: {self.project_root}")
        print(f"Data directory: {self.data_root}")
        print(f"Virtual environment: {self.venv_path}")
        
        print(f"\nTo activate the environment:")
        if self.system == 'windows':
            print(f"  {self.project_root}\\activate.bat")
        else:
            print(f"  source {self.project_root}/activate.sh")
        
        print(f"\nNext steps:")
        print("1. Activate the virtual environment")
        print("2. Test text extraction: python tests/test_text_extraction.py")
        print("3. Run detection pipeline: python src/detection/run_complete_pipeline.py")
        print("4. Run text extraction: python src/ocr/run_text_extraction.py")
        
        return True

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Unified PLC Diagram Processor Setup')
    parser.add_argument('--data-root', type=str,
                       help='Custom data root directory (default: ../plc-data)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually doing it')
    parser.add_argument('--parallel-jobs', type=int, default=4,
                       help='Number of parallel installation jobs (default: 4, max: 8)')
    
    args = parser.parse_args()
    
    setup = UnifiedPLCSetup(data_root=args.data_root, dry_run=args.dry_run, parallel_jobs=args.parallel_jobs)
    
    try:
        success = setup.run_complete_setup()
        
        if success:
            print("\n✓ Setup completed successfully!")
            return 0
        else:
            print("\n✗ Setup failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n✗ Setup failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
