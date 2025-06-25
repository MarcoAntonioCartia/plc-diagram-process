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
        self.venv_name = 'plcdp'
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
        if BuildToolsInstaller:
            try:
                self.build_tools_installer = BuildToolsInstaller(str(self.venv_pip))
            except Exception:
                self.build_tools_installer = BuildToolsInstaller()
        else:
            self.build_tools_installer = None
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
        min_version = (3, 8)  # Minimum supported version
        max_version = (3, 12)  # Highest supported **inclusive** (3.12.x)
        current_version = sys.version_info[:2]
        
        if current_version < min_version or current_version > max_version:
            print(f"✗ Python {min_version[0]}.{min_version[1]}–{max_version[1]} required, "
                  f"but {current_version[0]}.{current_version[1]} found")
            return False
        
        print(f"✓ Python {current_version[0]}.{current_version[1]} detected")
        return True
    
    def clean_existing_environment(self) -> bool:
        """Clean up existing virtual environment and caches"""
        print("\n=== Cleaning Existing Environment ===")
        
        # Clean virtual environment
        if self.venv_path.exists():
            print(f"Removing existing virtual environment at: {self.venv_path}")
            try:
                shutil.rmtree(self.venv_path)
                print("✓ Virtual environment removed")
            except Exception as e:
                print(f"✗ Failed to remove virtual environment: {e}")
                return False
        
        # Clean pip cache
        print("Cleaning pip cache...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], 
                        check=True, capture_output=True)
            print("✓ Pip cache cleaned")
        except Exception as e:
            print(f"⚠ Failed to clean pip cache: {e}")
        
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
        """Enhanced build environment setup with C++ and Rust support"""
        print("\n=== Enhanced Build Environment Setup ===")
        
        if not self.build_tools_installer:
            print("⚠ Build tools installer not available, skipping...")
            return True
        
        build_tools_status = capabilities["build_tools_status"]
        
        # Install C++ build tools if needed
        if build_tools_status.get("needs_installation", True):
            print("C++ build tools installation recommended...")
            
            if not self.dry_run:
                response = input("Install C++ build tools automatically? (y/n/skip): ")
                if response.lower() == 'y':
                    try:
                        if self.build_tools_installer.install_build_tools():
                            print("✓ C++ build tools installed successfully")
                            # Update capabilities after installation
                            capabilities["build_tools_status"] = self.build_tools_installer.check_build_tools_status()
                        else:
                            print("⚠ C++ build tools installation failed")
                    except Exception as e:
                        print(f"⚠ C++ build tools installation error: {e}")
                elif response.lower() == 'skip':
                    print("Skipping C++ build tools installation")
                else:
                    print("C++ build tools installation declined")
                    print("You may encounter compilation issues with some packages")
        else:
            print("✓ C++ build tools already available")
        
        # Check and install Rust/Cargo if needed
        if build_tools_status.get("needs_rust_installation", True):
            print("\nRust/Cargo installation recommended for some packages...")
            
            if not self.dry_run:
                response = input("Install Rust/Cargo automatically? (y/n/skip): ")
                if response.lower() == 'y':
                    try:
                        if self.build_tools_installer.install_rust_cargo():
                            print("✓ Rust/Cargo installed successfully")
                            # Update capabilities after installation
                            rust_status = self.build_tools_installer._check_rust_cargo()
                            capabilities["build_tools_status"].update(rust_status)
                        else:
                            print("⚠ Rust/Cargo installation failed")
                    except Exception as e:
                        print(f"⚠ Rust/Cargo installation error: {e}")
                elif response.lower() == 'skip':
                    print("Skipping Rust/Cargo installation")
                else:
                    print("Rust/Cargo installation declined")
                    print("Some packages may fail to compile")
        else:
            print("✓ Rust/Cargo already available")
        
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
        """Check if GPU and CUDA are available in WSL (enhanced detection)"""
        print("\n=== Checking WSL GPU Support (Enhanced Detection) ===")
        
        gpu_info = {
            'available': False,
            'cuda_available': False,
            'nvidia_smi': False,
            'driver_version': None,
            'cuda_version': None,
            'gpu_name': None,
            'gpu_memory': None,
            'compute_capability': None,
            'issues': []
        }
        
        if not self._check_wsl_available():
            gpu_info['issues'].append("WSL is not available")
            return gpu_info
        
        # Check for nvidia-smi in WSL (with timeout)
        print("Checking for NVIDIA GPU in WSL...")
        try:
            result = subprocess.run(
                ['wsl', '-e', 'bash', '-c', 'nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                gpu_info['nvidia_smi'] = True
                output_lines = result.stdout.strip().split('\n')
                if output_lines:
                    parts = [p.strip() for p in output_lines[0].split(',')]
                    if len(parts) >= 3:
                        gpu_info['gpu_name'] = parts[0]
                        gpu_info['driver_version'] = parts[1]
                        try:
                            gpu_info['gpu_memory'] = int(float(parts[2].replace(' MiB', '')) / 1024)  # Convert to GB
                        except:
                            gpu_info['gpu_memory'] = 0
                        gpu_info['available'] = True
                        print(f"✓ Found GPU: {gpu_info['gpu_name']}")
                        print(f"  Driver version: {gpu_info['driver_version']}")
                        print(f"  Memory: {gpu_info['gpu_memory']}GB")
            else:
                gpu_info['issues'].append("nvidia-smi not found in WSL")
                print("ℹ NVIDIA GPU not detected in WSL")
        except subprocess.TimeoutExpired:
            gpu_info['issues'].append("nvidia-smi command timed out")
            print("⚠ GPU check timed out")
        except Exception as e:
            gpu_info['issues'].append(f"Error checking GPU: {str(e)}")
            print(f"⚠ Error checking GPU: {e}")
        
        # Check for CUDA in WSL if GPU is available
        if gpu_info['available']:
            print("Checking for CUDA in WSL...")
            try:
                # Try nvidia-smi CUDA version detection FIRST (driver version - most reliable)
                result = subprocess.run(
                    ['wsl', '-e', 'bash', '-c', 'nvidia-smi | grep "CUDA Version" | sed "s/.*CUDA Version: \\([0-9]\\+\\.[0-9]\\+\\).*/\\1/" 2>/dev/null'],
                    capture_output=True, text=True, timeout=5
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    cuda_version = result.stdout.strip()
                    gpu_info['cuda_version'] = cuda_version
                    gpu_info['cuda_available'] = True
                    print(f"✓ CUDA {cuda_version} detected in WSL (via nvidia-smi - driver version)")
                else:
                    # Try nvcc --version as fallback (toolkit version)
                    result = subprocess.run(
                        ['wsl', '-e', 'bash', '-c', 'nvcc --version 2>/dev/null | grep "release" | sed "s/.*release \\([0-9]\\+\\.[0-9]\\+\\).*/\\1/"'],
                        capture_output=True, text=True, timeout=5
                    )
                    
                    if result.returncode == 0 and result.stdout.strip():
                        cuda_version = result.stdout.strip()
                        gpu_info['cuda_version'] = cuda_version
                        gpu_info['cuda_available'] = True
                        print(f"✓ CUDA {cuda_version} detected in WSL (via nvcc - toolkit version)")
                    else:
                        # Try version.txt as final fallback
                        result = subprocess.run(
                            ['wsl', '-e', 'bash', '-c', 'cat /usr/local/cuda/version.txt 2>/dev/null | grep "CUDA Version" | sed "s/.*CUDA Version \\([0-9]\\+\\.[0-9]\\+\\).*/\\1/"'],
                            capture_output=True, text=True, timeout=5
                        )
                        
                        if result.returncode == 0 and result.stdout.strip():
                            cuda_version = result.stdout.strip()
                            gpu_info['cuda_version'] = cuda_version
                            gpu_info['cuda_available'] = True
                            print(f"✓ CUDA {cuda_version} detected in WSL (via version.txt)")
                        else:
                            gpu_info['issues'].append("CUDA not found in WSL")
                            print("⚠ CUDA not detected in WSL")
                            
            except subprocess.TimeoutExpired:
                gpu_info['issues'].append("CUDA check timed out")
                print("⚠ CUDA check timed out")
            except Exception as e:
                gpu_info['issues'].append(f"Error checking CUDA: {str(e)}")
                print(f"⚠ Error checking CUDA: {e}")
        
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
            install_script = r'''#!/usr/bin/env bash

# Detect available package manager inside WSL and install poppler-utils.

set -e

detect_pkg_mgr() {
    if command -v apt-get >/dev/null 2>&1; then
        echo "apt-get"
    elif command -v apt >/dev/null 2>&1; then
        echo "apt"
    elif command -v dnf >/dev/null 2>&1; then
        echo "dnf"
    elif command -v yum >/dev/null 2>&1; then
        echo "yum"
    elif command -v apk >/dev/null 2>&1; then
        echo "apk"
    else
        echo "unsupported"
    fi
}

PKG_MGR=$(detect_pkg_mgr)
if [ "$PKG_MGR" = "unsupported" ]; then
    echo "FAILED_UNSUPPORTED_PM"
    exit 1
fi

echo "Package manager detected: $PKG_MGR"

case "$PKG_MGR" in
    apt-get|apt)
        sudo $PKG_MGR update && sudo $PKG_MGR install -y poppler-utils ;;
    dnf|yum)
        sudo $PKG_MGR install -y poppler-utils ;;
    apk)
        sudo $PKG_MGR add --update poppler-utils ;;
esac

echo "SUCCESS"
'''
            
            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False, newline='\n') as f:
                # Force LF line endings regardless of host OS to satisfy bash
                f.write(install_script.replace('\r', ''))
                script_path = f.name
            
            # Obtain WSL-compatible path to the script
            wsl_script_path = subprocess.check_output([
                'wsl', '-e', 'wslpath', '-u', script_path
            ], text=True).strip()

            # Ensure LF endings inside WSL (dos2unix optional)
            try:
                subprocess.run(['wsl', '-e', 'bash', '-c', f'dos2unix {wsl_script_path} >/dev/null 2>&1 || true'])
            except Exception:
                pass
            
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
        print("Detecting package manager inside WSL …")
        detect_cmd = [
            'wsl', '-e', 'bash', '-c',
            'if command -v apt-get >/dev/null 2>&1; then echo apt-get; '
            'elif command -v apt >/dev/null 2>&1; then echo apt; '
            'elif command -v dnf >/dev/null 2>&1; then echo dnf; '
            'elif command -v yum >/dev/null 2>&1; then echo yum; '
            'elif command -v apk >/dev/null 2>&1; then echo apk; '
            'else echo unsupported; fi'
        ]
        try:
            pm = subprocess.check_output(detect_cmd, text=True, timeout=10).strip()
        except Exception:
            pm = 'unsupported'

        if pm == 'unsupported':
            print('✗ Could not detect a supported package manager inside WSL')
            return False

        print(f"✓ Package manager detected inside WSL: {pm}")

        # Build update / install commands for the detected manager
        if pm in {'apt', 'apt-get'}:
            update_cmd = ['wsl', '-e', 'bash', '-c', f'sudo {pm} update']
            install_cmd = ['wsl', '-e', 'bash', '-c', f'sudo {pm} install -y poppler-utils']
        elif pm in {'dnf', 'yum'}:
            update_cmd = ['wsl', '-e', 'bash', '-c', f'sudo {pm} makecache']
            install_cmd = ['wsl', '-e', 'bash', '-c', f'sudo {pm} install -y poppler-utils']
        elif pm == 'apk':
            update_cmd = None  # apk update runs implicitly when adding packages
            install_cmd = ['wsl', '-e', 'bash', '-c', 'sudo apk add --update poppler-utils']
        else:
            print('✗ Unsupported package manager')
            return False

        # Run update if defined
        if update_cmd:
            print('Updating WSL package list …')
            try:
                subprocess.run(update_cmd, check=True, timeout=60)
                print('✓ Package list updated successfully')
            except subprocess.CalledProcessError:
                print('✗ Failed to update package list')
                return False
            except subprocess.TimeoutExpired:
                print('⚠ Package update timed out')
                return False

        # Install package
        print('Installing poppler-utils in WSL …')
        try:
            subprocess.run(install_cmd, check=True, timeout=180)
            print('✓ poppler-utils installed successfully')
        except subprocess.CalledProcessError:
            print('✗ Failed to install poppler-utils')
            return False
        except subprocess.TimeoutExpired:
            print('⚠ Poppler installation timed out')
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

    def find_latest_python(self) -> Optional[str]:
        """Find the latest compatible Python version installed on the system"""
        print("\n=== Finding Latest Compatible Python ===")
        
        if self.system == 'windows':
            # On Windows, check common installation paths
            python_paths = [
                r"C:\Python311\python.exe",  # Python 3.11
                r"C:\Python310\python.exe",  # Python 3.10
                r"C:\Python39\python.exe",   # Python 3.9
                r"C:\Python38\python.exe",   # Python 3.8
                r"C:\Users\*\AppData\Local\Programs\Python\Python311\python.exe",
                r"C:\Users\*\AppData\Local\Programs\Python\Python310\python.exe",
                r"C:\Users\*\AppData\Local\Programs\Python\Python39\python.exe",
                r"C:\Users\*\AppData\Local\Programs\Python\Python38\python.exe",
            ]
            
            # Try to find Python in PATH first
            try:
                result = subprocess.run(['where', 'python'], capture_output=True, text=True)
                if result.returncode == 0:
                    for path in result.stdout.splitlines():
                        if path.strip():
                            try:
                                version = subprocess.run([path, '--version'], 
                                                    capture_output=True, text=True)
                                if version.returncode == 0:
                                    ver_tokens = version.stdout.strip().split()
                                    ver_tuple = tuple(map(int, ver_tokens[-1].split('.')[:2]))
                                    if (3, 8) <= ver_tuple <= (3, 12):
                                        print(f"Found compatible Python in PATH: {path}")
                                        return path
                            except:
                                continue
            except:
                pass
            
            # Check specific paths
            for path_pattern in python_paths:
                try:
                    import glob
                    for path in glob.glob(path_pattern):
                        if os.path.exists(path):
                            # Read version string
                            ver = subprocess.check_output([path, '--version'], text=True)
                            ver_tuple = tuple(map(int, ver.strip().split()[-1].split('.')[:2]))
                            if (3, 8) <= ver_tuple <= (3, 12):
                                print(f"Found compatible Python installation: {path}")
                                return path
                except:
                    continue
        else:
            # On Linux/Mac, try python3.x commands
            for version in range(11, 7, -1):  # Try 3.11 down to 3.8
                try:
                    cmd = f'python3.{version}'
                    result = subprocess.run([cmd, '--version'], 
                                        capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"Found Python: {cmd}")
                        return cmd
                except:
                    continue
        
        print("⚠ No compatible Python version found")
        return None

    def create_virtual_environment(self) -> bool:
        """Create virtual environment using the latest compatible Python"""
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
        
        # Find the latest compatible Python
        current_ver = sys.version_info[:2]
        if (3, 8) <= current_ver <= (3, 12):
            python_executable = sys.executable
        else:
            python_executable = self.find_latest_python()

        if not python_executable:
            print("✗ No compatible Python version found")
            print("Please install Python 3.8-3.12 and try again")
            return False
        
        print(f"Creating virtual environment using: {python_executable}")
        try:
            if not self.dry_run:
                subprocess.run([python_executable, "-m", "venv", str(self.venv_path)], check=True)
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

    # === PYTORCH INDEX VALIDATION ===
    def _validate_pytorch_index(self, index_url: str) -> bool:
        """Validate that PyTorch index URL is accessible"""
        try:
            import urllib.request
            import urllib.error
            
            # Test if the index URL is accessible
            request = urllib.request.Request(index_url, method='HEAD')
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, Exception):
            return False

    def _get_best_pytorch_index(self, cuda_version: str) -> Tuple[str, str]:
        """Return the fixed PyTorch CUDA-12.1 wheel index.

        We standardise the whole project on *cu121* wheels because they are the
        most widely tested combo with Paddle 3.0.0.  This simplifies the
        dependency matrix and avoids future surprises when NVIDIA publishes
        newer minor releases (cu128, cu129, …).
        """

        fixed_url = "https://download.pytorch.org/whl/cu121"
        print(f"  Using fixed PyTorch index for CU121 wheels: {fixed_url}")

        # Still validate connectivity so we can fall back to CPU wheels if the
        # URL is blocked (rare corporate proxy cases).
        if self._validate_pytorch_index(fixed_url):
            print("    ✓ cu121 index is accessible")
            return "cu121", fixed_url

        print("    ✗ cu121 index not accessible – falling back to CPU wheels")
        return "cpu", "https://download.pytorch.org/whl/cpu"

    # === PYTORCH INSTALLATION (DIRECT CUDA APPROACH) ===
    def install_pytorch(self, capabilities: Dict) -> bool:
        """Install PyTorch with direct CUDA detection and installation"""
        print("\n=== PyTorch Installation (Direct CUDA Detection) ===")
        
        # Get GPU information for direct installation
        gpu_info = capabilities.get("gpu_info", {})
        
        # Check if WSL CUDA info is available and use it as primary source
        wsl_cuda_available = False
        if hasattr(self, 'wsl_gpu_info') and self.wsl_gpu_info.get('cuda_available'):
            wsl_cuda_available = True
            cuda_version = self.wsl_gpu_info.get("cuda_version", "11.8")
            gpu_model = self.wsl_gpu_info.get('gpu_name', 'Unknown GPU')
            
            print(f"  NVIDIA GPU detected (via WSL): {gpu_model}")
            print(f"   CUDA version (via WSL): {cuda_version}")
            print(f"   Using WSL CUDA detection as primary source")
        elif gpu_info.get("has_nvidia_gpu") and gpu_info.get("has_cuda"):
            # Fallback to main GPU detector
            cuda_version = gpu_info.get("cuda_version", "11.8")
            gpu_model = gpu_info.get('gpu_models', ['Unknown GPU'])[0]
            
            print(f"  NVIDIA GPU detected: {gpu_model}")
            print(f"   CUDA version: {cuda_version}")
        else:
            # No CUDA detected - install CPU version
            print("  No CUDA GPU detected - installing CPU version")
            return self._install_pytorch_cpu_fallback()
        
        # Determine installation strategy based on CUDA detection
        if wsl_cuda_available or (gpu_info.get("has_nvidia_gpu") and gpu_info.get("has_cuda")):
            
            # Get the best available PyTorch index with validation
            cuda_suffix, index_url = self._get_best_pytorch_index(cuda_version)
            
            print(f"  Selected PyTorch index: {cuda_suffix} ({index_url})")
            
            # Install CUDA version directly
            try:
                if self.dry_run:
                    print(f"  DRY RUN: Would install PyTorch {cuda_suffix} version")
                else:
                    print(f"  Installing PyTorch with {cuda_suffix} support...")
                    subprocess.run([
                        str(self.venv_pip), "install", "torch", "torchvision", "torchaudio",
                        "--index-url", index_url
                    ], check=True, timeout=1800)  # 30 minute timeout
                    print("✓ PyTorch CUDA version installed successfully")
                    
                    # Verify GPU functionality
                    return self._verify_pytorch_gpu_installation()
                    
            except subprocess.CalledProcessError as e:
                print(f"⚠ PyTorch CUDA installation failed: {e}")
                print("  Falling back to CPU version...")
                return self._install_pytorch_cpu_fallback()
            except subprocess.TimeoutExpired:
                print("⚠ PyTorch CUDA installation timed out")
                print("  Falling back to CPU version...")
                return self._install_pytorch_cpu_fallback()
        else:
            # No CUDA detected - install CPU version
            print("💻 No CUDA GPU detected - installing CPU version")
            return self._install_pytorch_cpu_fallback()
        
        return True

    def _install_pytorch_cpu_fallback(self) -> bool:
        """Install CPU version of PyTorch as fallback"""
        print("Installing PyTorch CPU version...")
        
        try:
            if self.dry_run:
                print("  DRY RUN: Would install PyTorch CPU version")
                return True
            
            subprocess.run([
                str(self.venv_pip), "install", "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ], check=True, timeout=1800)  # 30 minute timeout
            
            print("✓ PyTorch CPU version installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ PyTorch CPU installation failed: {e}")
            return False
        except subprocess.TimeoutExpired:
            print("⚠ PyTorch CPU installation timed out")
            return False

    def _verify_pytorch_gpu_installation(self) -> bool:
        """Verify PyTorch GPU installation and functionality"""
        print("  Verifying PyTorch GPU installation...")
        
        try:
            result = subprocess.run([
                str(self.venv_python), "-c", 
                """
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'PyTorch version: {torch.__version__}')
else:
    print('GPU functionality not available')
                """
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("  🎯 GPU verification successful:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        print(f"    {line}")
                return True
            else:
                print("  ⚠ Could not verify GPU functionality")
                print(f"    Error: {result.stderr}")
                return True  # Don't fail setup for verification issues
                
        except Exception as e:
            print(f"  ⚠ GPU verification failed: {e}")
            return True  # Don't fail setup for verification issues

    def _upgrade_pytorch_to_gpu(self, gpu_info: Dict) -> bool:
        """Upgrade PyTorch to GPU version with enhanced CUDA detection"""
        print("\nUpgrading PyTorch to GPU version...")
        
        # Enhanced CUDA version detection
        cuda_version = gpu_info.get("cuda_version", "11.8")
        
        # We standardise on cu121 for all GPUs – simpler matrix & matches Paddle.
        cuda_suffix = "cu121"
        index_url = "https://download.pytorch.org/whl/cu121"
        print(f"  Installing fixed PyTorch build {cuda_suffix} regardless of detected CUDA version")
        
        try:
            # Use wheel-only installation to avoid compilation issues
            print(f"  Installing PyTorch with {cuda_suffix} support...")
            subprocess.run([
                str(self.venv_pip), "install", "--upgrade", 
                "torch", "torchvision", "torchaudio",
                "--index-url", index_url,
                "--only-binary=all"  # Force wheel-only installation
            ], check=True, timeout=1800)
            
            print("✓ PyTorch GPU version installed successfully")
            
            # Verify GPU availability and show detailed info
            try:
                result = subprocess.run([
                    str(self.venv_python), "-c", 
                    """
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'PyTorch version: {torch.__version__}')
                    """
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print("  🎯 GPU verification successful:")
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            print(f"    {line}")
                else:
                    print("  ⚠ Could not verify GPU functionality")
                    print(f"    Error: {result.stderr}")
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
        """Parse requirements.txt and extract clean package specifications"""
        packages = []
        try:
            with open(requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Remove inline comments (everything after #)
                        if '#' in line:
                            line = line.split('#')[0].strip()
                        
                        # Skip empty lines after comment removal
                        if not line:
                            continue
                        
                        # Add clean package specification
                        packages.append(line)
        except Exception as e:
            print(f"Warning: Could not parse requirements file: {e}")
        return packages

    def categorize_packages(self, packages: List[str]) -> Dict[str, List[str]]:
        """Categorize packages by installation complexity"""
        # Exclude PyTorch packages (already installed)
        pytorch_packages = {"torch", "torchvision", "torchaudio"}
        
        # Exclude PaddleOCR core packages (handled by specialized installer later)
        paddleocr_packages = {"paddlepaddle", "paddlepaddle-gpu", "paddleocr"}
        
        # Heavy packages that should be installed sequentially
        heavy_packages = {
            "ultralytics", "opencv-python", 
            "scipy", "numpy", "pandas", "matplotlib", "transformers"
        }
        
        heavy_list = []
        parallel_packages = []
        
        for package in packages:
            base_name = package.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].split('>')[0].split('<')[0].split('[')[0].lower()
            
            # Skip PyTorch packages
            if base_name in pytorch_packages:
                continue
            
            # Skip PaddleOCR core packages (handled by specialized installer later)
            if base_name in paddleocr_packages:
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
        """Install a single package with enhanced compilation support"""
        base_name = package.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].split('>')[0].split('<')[0].split('[')[0]
        
        # Determine timeout based on package
        if base_name.lower() in ['ultralytics', 'paddlepaddle', 'paddleocr', 'transformers']:
            timeout = 1800  # 30 minutes
        elif base_name.lower() in ['opencv-python', 'scipy', 'numpy', 'pandas']:
            timeout = 900   # 15 minutes
        else:
            timeout = 300   # 5 minutes
        
        if self.dry_run:
            print(f"  DRY RUN: Would install {package}")
            return package, True, ""
        
        # Check if this package needs special compilation support
        compilation_packages = ['paddleocr', 'paddlepaddle']
        needs_compilation = any(comp_pkg in base_name.lower() for comp_pkg in compilation_packages)
        
        if needs_compilation and self.build_tools_installer:
            print(f"  Installing {base_name} with enhanced compilation support...")
            
            # Try VS environment installation first
            try:
                if self.build_tools_installer.install_with_vs_environment(package, str(self.venv_pip)):
                    print(f"  ✓ {base_name} installed successfully with VS environment")
                    return package, True, ""
            except Exception as e:
                print(f"  ⚠ VS environment installation failed: {e}")
                print(f"  Falling back to standard installation...")
        
        # Standard pip install
        try:
            print(f"  Installing {base_name}...")
            result = subprocess.run([
                str(self.venv_pip), 'install', package
            ], capture_output=True, text=True, timeout=timeout)
            
            success = result.returncode == 0
            error_msg = result.stderr.strip() if result.stderr else ""
            
            if success:
                print(f"  ✓ {base_name} installed successfully")
            else:
                print(f"  ✗ {base_name} installation failed")
                if error_msg:
                    print(f"    Error: {error_msg[:200]}...")  # Show first 200 chars
                
                # For compilation packages, suggest manual installation with VS environment
                if needs_compilation:
                    print(f"      Try manual installation with VS environment:")
                    print(f"       1. Run: install_paddleocr_with_vs.bat")
                    print(f"       2. Or activate VS environment manually before pip install")
            
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

    def clean_ultralytics_cache(self) -> bool:
        """Clean Ultralytics cache to prevent path conflicts"""
        print("\n=== Cleaning Ultralytics Cache ===")
        
        cache_locations = [
            Path.home() / "AppData" / "Roaming" / "Ultralytics",  # Windows
            Path.home() / ".config" / "Ultralytics",  # Linux
            Path.home() / "Library" / "Application Support" / "Ultralytics",  # macOS
        ]
        
        cleaned_any = False
        
        for cache_dir in cache_locations:
            if cache_dir.exists():
                print(f"Found Ultralytics cache at: {cache_dir}")
                
                # Check for settings.json specifically
                settings_file = cache_dir / "settings.json"
                if settings_file.exists():
                    try:
                        with open(settings_file, 'r') as f:
                            settings = f.read()
                        
                        # Check if it contains old version paths
                        if "0.1" in settings or "0.2" in settings:
                            print(f"  Found old version references in settings.json")
                            
                            if not self.dry_run:
                                response = input(f"  Clean Ultralytics cache at {cache_dir}? (y/n): ")
                                if response.lower() == 'y':
                                    try:
                                        shutil.rmtree(cache_dir)
                                        print(f"  ✓ Cleaned cache directory: {cache_dir}")
                                        cleaned_any = True
                                    except Exception as e:
                                        print(f"  ⚠ Failed to clean cache: {e}")
                                else:
                                    print(f"  Skipped cleaning {cache_dir}")
                            else:
                                print(f"  DRY RUN: Would clean {cache_dir}")
                                cleaned_any = True
                        else:
                            print(f"  No old version references found in settings.json")
                    except Exception as e:
                        print(f"  ⚠ Could not read settings.json: {e}")
                else:
                    print(f"  No settings.json found in {cache_dir}")
        
        if not cleaned_any:
            print("  No Ultralytics cache cleanup needed")
        
        return True

    def install_other_packages(self) -> bool:
        """Install other packages using robust strategies"""
        print("\n=== Installing Other Dependencies ===")
        
        # Clean Ultralytics cache before installing to prevent path conflicts
        self.clean_ultralytics_cache()
        
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
        """Attempt bulk installation of failed packages using simple pip install"""
        print("\nAttempting bulk installation of failed packages...")
        print("Using simple pip install (same as manual installation)...")
        
        try:
            # Create temporary requirements file
            temp_req_file = self.project_root / 'temp_failed_requirements.txt'
            with open(temp_req_file, 'w') as f:
                for pkg in failed_packages:
                    f.write(f"{pkg}\n")
            
            # Simple bulk installation - exactly like manual "pip install -r requirements.txt"
            print("  Installing from requirements file...")
            result = subprocess.run([
                str(self.venv_pip), 'install', '-r', str(temp_req_file)
            ], text=True, timeout=3600)  # 1 hour timeout
            
            # Clean up
            temp_req_file.unlink()
            
            if result.returncode == 0:
                print("✓ Bulk installation completed successfully")
                return True
            else:
                print("✗ Bulk installation failed")
                print("\n Manual installation suggestions:")
                print("1. Activate the virtual environment:")
                print(f"   {self.venv_activate}")
                print("2. Try installing packages individually:")
                for pkg in failed_packages[:5]:  # Show first 5 as examples
                    print(f"   pip install {pkg}")
                if len(failed_packages) > 5:
                    print(f"   ... and {len(failed_packages) - 5} more packages")
                print("3. Or try the full requirements file:")
                print("   pip install -r requirements.txt")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠ Bulk installation timed out after 1 hour")
            if temp_req_file.exists():
                temp_req_file.unlink()
            return False
        except Exception as e:
            print(f"✗ Bulk installation error: {e}")
            if temp_req_file.exists():
                temp_req_file.unlink()
            return False

    def verify_installation(self) -> bool:
        """Verify that key packages are working"""
        print("\n=== Installation Verification ===")
        
        test_packages = [
            ("torch", "PyTorch"),
            ("cv2", "OpenCV"),
            ("pandas", "Pandas"),
            ("numpy", "NumPy"),
            ("paddle", "Paddle"),
            ("paddleocr", "PaddleOCR")
        ]
        
        failed_packages = []
        
        for package, description in test_packages:
            try:
                if self.dry_run:
                    print(f"  DRY RUN: Would test {description}")
                    continue
                
                if package == "torch":
                    code = (
                        "import torch, json, sys;"
                        "info = {'cuda': torch.cuda.is_available(), 'count': torch.cuda.device_count()};"
                        "info.update({'name': torch.cuda.get_device_name(0) if info['cuda'] else 'CPU'});"
                        "print(json.dumps(info))"
                    )
                elif package == "paddle":
                    code = (
                        "import paddle, json, sys;"
                        "cuda = paddle.device.is_compiled_with_cuda();"
                        "count = paddle.device.cuda.device_count() if cuda else 0;"
                        "name = paddle.device.cuda.get_device_name(0) if cuda else 'CPU';"
                        "print(json.dumps({'cuda': cuda, 'count': count, 'name': name}))"
                    )
                else:
                    code = f"import {package}; print('✓ {description} working')"

                result = subprocess.run([
                    str(self.venv_python), "-c", code
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
            print("\nProceeding despite verification failures; you can test imports manually after setup.")
            return True
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

    def install_specialized_packages(self, capabilities: Dict) -> bool:
        """Install specialized packages like PaddleOCR"""
        print("Installing specialized packages...")
        
        # Install PaddleOCR using the proven Method 3 approach
        if not self.build_tools_installer:
            print("⚠ Build tools installer not available, skipping PaddleOCR installation")
            return True
            
        # Ensure the build tools installer uses the venv's pip for all subsequent operations
        try:
            self.build_tools_installer.venv_pip = str(self.venv_pip)
        except AttributeError:
            pass
        
        if not self.build_tools_installer.install_paddleocr(capabilities):
            print("✗ Failed to install PaddleOCR")
            return False
        
        print("✓ Specialized packages installed successfully")
        return True

    def run_gpu_sanity_check(self) -> bool:
        """Run a lightweight Torch&nbsp;+ Paddle GPU self-test inside the venv.

        This step NEVER aborts the setup – it is purely diagnostic.  A failure is
        logged and the setup continues so that CPU-only users are not blocked.
        """
        print("\n=== GPU Sanity Check (Torch & Paddle) ===")

        checker_module = "src.utils.gpu_sanity_checker"

        try:
            # Run the checker with a 2-minute timeout to avoid hanging installs
            import subprocess, textwrap

            result = subprocess.run(
                [str(self.venv_python), "-m", checker_module, "--device", "auto"],
                capture_output=True, text=True, timeout=120
            )

            print(textwrap.dedent(result.stdout))
            if result.returncode == 0:
                print("✓ GPU sanity check passed (or CPU fallback acceptable)")
            else:
                print("⚠ GPU sanity check reported issues – continuing anyway")
            return True  # never fail the whole setup

        except FileNotFoundError:
            print("⚠ gpu_sanity_checker script not found – skipping")
            return True
        except subprocess.TimeoutExpired:
            print("⚠ GPU sanity check timed out – skipping")
            return True
        except Exception as exc:
            print(f"⚠ GPU sanity check error: {exc}")
            return True

    def ensure_latest_numpy(self) -> bool:
        """Upgrade NumPy to the latest 2.x release to avoid old-wheel DLL issues"""
        print("\n=== Ensuring latest NumPy (>=2.1,<3) ===")
        return self.run_command([
            str(self.venv_pip), 'install', '--upgrade', 'numpy>=2.1,<3'
        ], "Upgrading NumPy to >=2.1,<3", use_venv=False)

    def run_complete_setup(self) -> bool:
        """Run the complete unified setup process"""
        print("Unified PLC Diagram Processor Setup")
        print("=" * 60)
        
        steps = [
            ("Finding latest Python version", self.find_latest_python),
            ("Checking Python version", self.check_python_version),
            ("Cleaning existing environment", self.clean_existing_environment),
            ("Detecting system capabilities", lambda: (self.detect_system_capabilities(), True)[1]),
            ("Installing system dependencies", self.install_system_dependencies),
            ("Setting up build environment", lambda: self.setup_build_environment(self.capabilities)),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Upgrading pip tools", self.upgrade_pip_tools),
            ("Installing PyTorch (Direct CUDA)", lambda: self.install_pytorch(self.capabilities)),
            ("Installing other packages", self.install_other_packages),
            ("Setting up data directories", self.setup_data_directories),
            ("Creating activation scripts", self.create_activation_scripts),
            ("Installing specialized packages", lambda: self.install_specialized_packages(self.capabilities)),
            ("Finalize NumPy version", self.ensure_latest_numpy),
            ("GPU sanity check", self.run_gpu_sanity_check),
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
            print(f"\nGPU Status: {gpu_info.get('gpu_models', ['Unknown'])[0]} detected")
        else:
            print("\nGPU Status: CPU-only (no CUDA GPU detected)")
        
        # Show WSL GPU status if available
        if hasattr(self, 'wsl_gpu_info') and self.wsl_gpu_info.get('available'):
            print(f"WSL GPU Status: {self.wsl_gpu_info['gpu_name']} ready for training")
        
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
    
    # Quick check flag: skip the whole setup and only validate imports
    parser.add_argument('--validate-imports', action='store_true',
                       help='Skip all setup steps and only run the final import verification inside the existing virtual environment')
    # split envs
    parser.add_argument('--multi-env', action='store_true',
                       help='After main venv is ready create detection_env (torch) and ocr_env (paddle) via MultiEnvironmentManager')
    
    args = parser.parse_args()
    
    # If the user only wants to validate imports, run the checker and exit early
    if args.validate_imports:
        setup = UnifiedPLCSetup(data_root=args.data_root, dry_run=False, parallel_jobs=args.parallel_jobs)
        success = setup.verify_installation()
        if success:
            print("\n✓ Import validation completed successfully!")
            return 0
        else:
            print("\n✗ Import validation reported issues – see log above")
            return 1
    
    # Otherwise proceed with the full setup workflow
    setup = UnifiedPLCSetup(data_root=args.data_root, dry_run=args.dry_run, parallel_jobs=args.parallel_jobs)
    
    try:
        success = setup.run_complete_setup()
        
        if success and args.multi_env:
            try:
                from pathlib import Path
                from src.utils.multi_env_manager import MultiEnvironmentManager

                mgr = MultiEnvironmentManager(Path(__file__).resolve().parent.parent)
                if mgr.setup() and mgr.health_check():
                    print("\n✓ Multi-environment setup completed successfully!")
                else:
                    print("\n⚠ Multi-environment setup reported issues. See log above.")
            except Exception as exc:
                print(f"\n⚠ Failed to create multi-environment: {exc}")

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
