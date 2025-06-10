#!/usr/bin/env python3
"""
Enhanced Setup script for PLC Diagram Processor
Handles automatic GPU detection, build tools installation, and robust package management
"""

import os
import sys
import subprocess
import platform
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import our enhanced setup modules directly
setup_dir = Path(__file__).resolve().parent
sys.path.append(str(setup_dir))

from gpu_detector import GPUDetector
from build_tools_installer import BuildToolsInstaller
from package_installer import RobustPackageInstaller

class EnhancedPLCSetup:
    """Enhanced setup with automatic GPU detection and robust package installation"""
    
    def __init__(self, data_root: Optional[str] = None):
        self.project_root = project_root
        self.data_root = Path(data_root).absolute() if data_root else self.project_root.parent / 'plc-data'
        self.venv_name = 'yolovenv'
        self.venv_path = self.project_root / self.venv_name
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
        
        # Initialize components
        self.gpu_detector = GPUDetector()
        self.build_tools_installer = BuildToolsInstaller()
        self.package_installer = RobustPackageInstaller()
        
        print(f"Enhanced PLC Diagram Processor Setup")
        print(f"Project root: {self.project_root}")
        print(f"Data root: {self.data_root}")
        print(f"Virtual environment: {self.venv_path}")
        print(f"System: {self.system}")
        print()
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        min_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version < min_version:
            print(f":( Python {min_version[0]}.{min_version[1]}+ required, but {current_version[0]}.{current_version[1]} found")
            return False
        
        print(f":) Python {current_version[0]}.{current_version[1]} detected")
        return True
    
    def detect_system_capabilities(self) -> Dict:
        """Detect GPU and build tools capabilities"""
        print("=== System Capability Detection ===")
        
        # Detect GPU capabilities
        gpu_info = self.gpu_detector.detect_gpu_capabilities()
        
        # Check build tools status
        build_tools_status = self.build_tools_installer.check_build_tools_status()
        
        return {
            "gpu_info": gpu_info,
            "build_tools_status": build_tools_status
        }
    
    def setup_build_environment(self, capabilities: Dict) -> bool:
        """Set up build environment if needed"""
        print("\n=== Build Environment Setup ===")
        
        build_tools_status = capabilities["build_tools_status"]
        
        if build_tools_status["needs_installation"]:
            print("Build tools installation required...")
            
            if not self.build_tools_installer.install_build_tools():
                print(":( Build tools installation failed")
                print("You may encounter compilation issues with some packages")
                
                # Ask user if they want to continue
                response = input("Continue with setup anyway? (y/n): ")
                if response.lower() != 'y':
                    return False
        else:
            print("✓ Build tools already available")
        
        return True

    # === WSL POPPLER INTEGRATION METHODS ===
    def _check_wsl_available(self) -> bool:
        """Check if WSL is available on Windows"""
        try:
            result = subprocess.run(['wsl', '--list'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_wsl_gpu_support(self) -> Dict[str, Any]:
        """Check if GPU is available in WSL for training"""
        print("\n=== Checking WSL GPU Support ===")
        
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
        
        # Check for nvidia-smi in WSL
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
                        print(f"GPU found: {gpu_info['gpu_name']}")
                        print(f"Driver version: {gpu_info['driver_version']}")
            else:
                gpu_info['issues'].append("nvidia-smi not found in WSL")
                print("NVIDIA GPU not detected in WSL")
        except subprocess.TimeoutExpired:
            gpu_info['issues'].append("nvidia-smi command timed out")
            print("GPU check timed out")
        except Exception as e:
            gpu_info['issues'].append(f"Error checking GPU: {str(e)}")
            print(f"Error checking GPU: {e}")
        
        # Check for CUDA in WSL
        if gpu_info['nvidia_smi']:
            print("\nChecking CUDA availability in WSL...")
            try:
                result = subprocess.run(
                    ['wsl', '-e', 'bash', '-c', 'nvcc --version 2>/dev/null | grep "release" | awk \'{print $6}\' | cut -c2-'],
                    capture_output=True, text=True, timeout=5
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    gpu_info['cuda_version'] = result.stdout.strip()
                    gpu_info['cuda_available'] = True
                    print(f"CUDA version: {gpu_info['cuda_version']}")
                else:
                    # Try alternative method
                    result = subprocess.run(
                        ['wsl', '-e', 'bash', '-c', 'nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+"'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        gpu_info['cuda_version'] = result.stdout.strip()
                        gpu_info['cuda_available'] = True
                        print(f"CUDA version (from nvidia-smi): {gpu_info['cuda_version']}")
                    else:
                        gpu_info['issues'].append("CUDA not found in WSL")
                        print("CUDA not detected in WSL (training will use CPU)")
            except Exception as e:
                gpu_info['issues'].append(f"Error checking CUDA: {str(e)}")
                print(f"Could not check CUDA: {e}")
        
        return gpu_info

    def _install_poppler_via_wsl(self) -> bool:
        """Install poppler using WSL on Windows with improved error handling"""
        print("\n=== Installing Poppler via WSL ===")
        
        # Check GPU support (informational, don't block installation)
        wsl_gpu_info = self._check_wsl_gpu_support()
        
        # Store GPU info for later reference
        self.wsl_gpu_info = wsl_gpu_info
        
        if wsl_gpu_info['available']:
            print(f"WSL GPU ready: {wsl_gpu_info['gpu_name']}")
        else:
            print("WSL GPU not available - training will use CPU")
        
        # Continue with poppler installation
        print("\n=== Installing Poppler ===")
        
        # Check if poppler is already installed in WSL
        print("Checking if poppler is already installed in WSL...")
        check_cmd = ['wsl', '-e', 'bash', '-c', 'which pdftotext 2>/dev/null']
        
        try:
            result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                print(f"Poppler is already installed in WSL at: {result.stdout.strip()}")
                return self._create_wsl_wrappers()
        except subprocess.TimeoutExpired:
            print("WSL check timed out - WSL might be starting up")
            time.sleep(3)
            # Try once more
            try:
                result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and result.stdout.strip():
                    print("Poppler is already installed in WSL")
                    return self._create_wsl_wrappers()
            except:
                pass
        except Exception as e:
            print(f"Could not check for existing poppler: {e}")
        
        # Check if we can run commands without sudo
        print("\nChecking WSL sudo requirements...")
        test_sudo_cmd = ['wsl', '-e', 'bash', '-c', 'sudo -n true 2>/dev/null']
        try:
            result = subprocess.run(test_sudo_cmd, capture_output=True, text=True, timeout=5)
            passwordless_sudo = (result.returncode == 0)
        except:
            passwordless_sudo = False
        
        if passwordless_sudo:
            print("Passwordless sudo detected, proceeding with automatic installation...")
            return self._run_wsl_poppler_install()
        
        # Need password authentication - try interactive installation
        print("\nWSL requires sudo password for package installation.")
        print("Attempting installation with password prompt...")
        
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
            import tempfile
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
                    print("\nPoppler installed successfully!")
                    return self._create_wsl_wrappers()
            
            print("\nPoppler installation failed")
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
            print("Package list updated successfully")
        except subprocess.CalledProcessError:
            print("Failed to update package list")
            return False
        except subprocess.TimeoutExpired:
            print("Package update timed out")
            return False
        
        # Install poppler-utils
        print("Installing poppler-utils in WSL...")
        install_cmd = ['wsl', '-e', 'bash', '-c', 'sudo apt-get install -y poppler-utils']
        try:
            subprocess.run(install_cmd, check=True, timeout=120)
            print("Poppler-utils installed successfully")
        except subprocess.CalledProcessError:
            print("Failed to install poppler-utils")
            return False
        except subprocess.TimeoutExpired:
            print("Poppler installation timed out")
            return False
        
        return self._create_wsl_wrappers()

    def _create_wsl_wrappers(self) -> bool:
        """Create Windows-accessible poppler wrappers with error handling"""
        print("\nCreating Windows-accessible poppler wrappers...")
        wrapper_dir = self.project_root / 'bin' / 'poppler'
        
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
                    print(f"  Created wrapper: {wrapper_path.name}")
                except Exception as e:
                    print(f"  Failed to create {wrapper_path.name}: {e}")
            
            if not created_wrappers:
                print("\nFailed to create any wrappers")
                return False
            
            # Add to PATH for current session
            current_path = os.environ.get('PATH', '')
            if str(wrapper_dir) not in current_path:
                os.environ['PATH'] = f"{wrapper_dir};{current_path}"
                print(f"\nAdded {wrapper_dir} to PATH for current session")
            
            print("\n" + "="*60)
            print("Poppler Installation Complete!")
            print("="*60)
            print(f"  Wrappers created: {len(created_wrappers)}/{len(poppler_tools)}")
            print(f"  Location: {wrapper_dir}")
            print("\n  To make permanent, add this to your system PATH:")
            print(f"  {wrapper_dir}")
            print("="*60)
            
            # Show GPU status summary if available
            if hasattr(self, 'wsl_gpu_info') and self.wsl_gpu_info['available']:
                print(f"\nGPU Status: {self.wsl_gpu_info['gpu_name']} ready for training!")
            else:
                print("\nGPU Status: Not available - training will use CPU")
            
            return True
            
        except Exception as e:
            print(f"\nError creating wrappers: {e}")
            return False

    def _manual_poppler_instructions(self) -> bool:
        """Provide manual poppler installation instructions"""
        print("\n=== Manual Poppler Installation Required ===")
        print("Please install Poppler manually:")
        print("1. Download from: https://github.com/oschwartz10612/poppler-windows/releases")
        print("2. Extract the archive")
        print("3. Add the 'bin' folder to your system PATH")
        print("   Example: C:\\poppler-xx.xx.x\\Library\\bin")
        
        response = input("\nHave you installed Poppler manually? (y/n): ")
        if response.lower() != 'y':
            print("Please install Poppler and run the setup again.")
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
        """Install dependencies on Windows with WSL integration"""
        print("Windows detected")
        
        # Check for Visual Studio Build Tools first
        print("\n1. Checking for Visual Studio Build Tools...")
        build_status = self.build_tools_installer.check_build_tools_status()
        
        if build_status["needs_installation"]:
            print("Visual Studio Build Tools installation required")
            if not self.build_tools_installer.install_build_tools():
                print("Build tools installation failed, but continuing...")
        else:
            print("Visual Studio Build Tools already available")
        
        # Check for WSL and install poppler
        print("\n2. Setting up Poppler...")
        
        if self._check_wsl_available():
            print("WSL detected - will install poppler automatically")
            
            if not self._install_poppler_via_wsl():
                print("\nFailed to install poppler via WSL")
                print("Falling back to manual installation instructions...")
                return self._manual_poppler_instructions()
        else:
            print("WSL not detected")
            print("\nWSL (Windows Subsystem for Linux) is recommended for automatic poppler installation.")
            print("To install WSL:")
            print("  1. Open PowerShell as Administrator")
            print("  2. Run: wsl --install")
            print("  3. Restart your computer")
            print("  4. Run this setup again")
            
            response = input("\nDo you want to continue with manual poppler installation? (y/n): ")
            if response.lower() != 'y':
                print("\nPlease install WSL and run setup again for automatic installation.")
                return False
            
            return self._manual_poppler_instructions()
        
        return True

    def _install_linux_dependencies(self) -> bool:
        """Install dependencies on Linux"""
        print("Linux detected")
        import shutil
        
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
        import shutil
        
        if not shutil.which('brew'):
            print("Installing Homebrew...")
            if not self._run_command(['/bin/bash', '-c', 
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
            if not self._run_command(command, description):
                return False
        return True

    def _run_command(self, command: List[str], description: str) -> bool:
        """Run a single command"""
        print(f"Running: {description}")
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"  ✓ {description}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ✗ {description} failed: {e}")
            return False
    
    def create_virtual_environment(self) -> bool:
        """Create virtual environment"""
        print("\n=== Virtual Environment Setup ===")
        
        if self.venv_path.exists():
            print(f"Virtual environment already exists at: {self.venv_path}")
            
            if self.venv_python.exists():
                print("✓ Existing virtual environment appears valid")
                response = input("Recreate virtual environment? (y/n): ")
                if response.lower() != 'y':
                    return True
            
            print("Removing existing virtual environment...")
            try:
                import shutil
                shutil.rmtree(self.venv_path)
            except Exception as e:
                print(f"✗ Failed to remove existing environment: {e}")
                return False
        
        print("Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)], check=True)
            print("✓ Virtual environment created successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create virtual environment: {e}")
            return False
    
    def upgrade_pip_tools(self) -> bool:
        """Upgrade pip and essential tools in virtual environment"""
        print("\n=== Upgrading Virtual Environment Tools ===")
        
        if not self.venv_python.exists():
            print("✗ Virtual environment not found")
            return False
        
        tools = ["pip", "setuptools", "wheel"]
        
        for tool in tools:
            print(f"Upgrading {tool}...")
            try:
                if tool == "pip":
                    # Use python -m pip for pip upgrades
                    subprocess.run([
                        str(self.venv_python), "-m", "pip", "install", "--upgrade", "pip"
                    ], check=True, capture_output=True)
                else:
                    subprocess.run([
                        str(self.venv_pip), "install", "--upgrade", tool
                    ], check=True, capture_output=True)
                print(f"✓ {tool} upgraded successfully")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to upgrade {tool}: {e}")
                return False
        
        return True
    
    def _check_nvidia_gpu_fallback(self) -> Dict[str, Any]:
        """Fallback GPU detection using nvidia-smi directly"""
        print("Running fallback GPU detection...")
        
        gpu_fallback_info = {
            "has_nvidia_gpu": False,
            "has_cuda": False,
            "gpu_models": [],
            "cuda_version": None,
            "driver_version": None
        }
        
        # Check nvidia-smi on Windows
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 2:
                            gpu_fallback_info["gpu_models"].append(parts[0])
                            gpu_fallback_info["driver_version"] = parts[1]
                            gpu_fallback_info["has_nvidia_gpu"] = True
                            print(f"✓ Found GPU: {parts[0]}")
                            print(f"✓ Driver version: {parts[1]}")
        except Exception as e:
            print(f"nvidia-smi check failed: {e}")
        
        # Check for CUDA
        if gpu_fallback_info["has_nvidia_gpu"]:
            # Try nvcc first
            try:
                result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    import re
                    for line in result.stdout.split('\n'):
                        match = re.search(r'release (\d+\.\d+)', line)
                        if match:
                            gpu_fallback_info["cuda_version"] = match.group(1)
                            gpu_fallback_info["has_cuda"] = True
                            print(f"✓ CUDA version: {gpu_fallback_info['cuda_version']}")
                            break
            except:
                pass
            
            # If nvcc failed, try nvidia-smi for CUDA version
            if not gpu_fallback_info["has_cuda"]:
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        # Assume CUDA is available if driver is present
                        gpu_fallback_info["has_cuda"] = True
                        gpu_fallback_info["cuda_version"] = "11.8"  # Safe default
                        print("✓ CUDA runtime assumed available (using default version)")
                except:
                    pass
        
        return gpu_fallback_info

    def install_pytorch_for_gpu(self, gpu_info: Dict) -> bool:
        """Install PyTorch with appropriate GPU support - Enhanced with fallback detection"""
        print("\n=== PyTorch Installation ===")
        
        # Check if we have WSL GPU info as well
        has_wsl_gpu = hasattr(self, 'wsl_gpu_info') and self.wsl_gpu_info.get('available', False)
        
        # If main GPU detection failed but WSL detected GPU, try fallback detection
        if (not gpu_info.get("has_nvidia_gpu", False) or not gpu_info.get("has_cuda", False)) and has_wsl_gpu:
            print("Main GPU detection inconclusive, but WSL GPU detected.")
            print("Running additional GPU checks...")
            
            fallback_gpu_info = self._check_nvidia_gpu_fallback()
            
            # If fallback found GPU, update our gpu_info
            if fallback_gpu_info["has_nvidia_gpu"]:
                print("✓ Fallback GPU detection successful!")
                gpu_info.update(fallback_gpu_info)
            else:
                print(":( Fallback GPU detection also failed")
        
        # Determine PyTorch installation type
        has_gpu = gpu_info.get("has_nvidia_gpu", False)
        has_cuda = gpu_info.get("has_cuda", False)
        
        if has_gpu and has_cuda:
            print(":) NVIDIA GPU with CUDA detected!")
            if gpu_info.get("gpu_models"):
                print(f"   GPU: {gpu_info['gpu_models'][0]}")
            if gpu_info.get("cuda_version"):
                print(f"   CUDA: {gpu_info['cuda_version']}")
            
            # Get optimized PyTorch command
            description, command = self.gpu_detector.get_pytorch_install_command()
            print(f"   {description}")
            
        elif has_wsl_gpu:
            print(":) Using WSL GPU information for PyTorch selection")
            print(f"   WSL GPU: {self.wsl_gpu_info.get('gpu_name', 'Unknown')}")
            
            # Use CUDA version if available
            cuda_version = self.wsl_gpu_info.get('cuda_version', '11.8')
            if cuda_version:
                if cuda_version.startswith('12'):
                    index_url = "https://download.pytorch.org/whl/cu121"
                    print("   Installing PyTorch with CUDA 12.x support")
                else:
                    index_url = "https://download.pytorch.org/whl/cu118"
                    print("   Installing PyTorch with CUDA 11.x support")
                
                command = [str(self.venv_pip), "install", "torch", "torchvision", "--index-url", index_url]
            else:
                # Default to CPU
                command = [str(self.venv_pip), "install", "torch", "torchvision"]
                print("   Installing CPU-only PyTorch (CUDA not detected in WSL)")
                
        else:
            print(":( No CUDA-capable GPU detected")
            print("PyTorch will be installed with CPU-only support")
            
            response = input("Continue with CPU-only PyTorch? (y/n): ")
            if response.lower() != 'y':
                print("Please install CUDA drivers and run setup again for GPU support")
                print("\nGPU Detection Summary:")
                print(f"  Main detection - GPU: {gpu_info.get('has_nvidia_gpu', False)}, CUDA: {gpu_info.get('has_cuda', False)}")
                print(f"  WSL detection - Available: {has_wsl_gpu}")
                return False
            
            command = [str(self.venv_pip), "install", "torch", "torchvision"]

        try:
            print("Installing PyTorch (this may take several minutes)...")
            subprocess.run(command, check=True)
            print("✓ PyTorch installed successfully")
            
            # Verify CUDA availability after installation
            if has_gpu or has_wsl_gpu:
                try:
                    result = subprocess.run([
                        str(self.venv_python), "-c", 
                        "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        print(":) PyTorch CUDA verification:")
                        for line in result.stdout.strip().split('\n'):
                            print(f"   {line}")
                    else:
                        print(":( Could not verify PyTorch CUDA status")
                except Exception as e:
                    print(f":( PyTorch verification failed: {e}")
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ PyTorch installation failed: {e}")
            return False
    
    def install_other_packages(self) -> bool:
        """Install other packages using robust installer"""
        print("\n=== Installing Other Dependencies ===")
        
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            print(f"✗ Requirements file not found: {requirements_file}")
            return False
        
        # Parse requirements and exclude PyTorch packages (already installed)
        pytorch_packages = {"torch", "torchvision", "torchaudio"}
        
        with open(requirements_file, 'r') as f:
            all_packages = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    package_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].split('>')[0].split('<')[0].split('[')[0].lower()
                    if package_name not in pytorch_packages:
                        all_packages.append(line)
        
        if not all_packages:
            print("✓ No additional packages to install")
            return True
        
        print(f"Installing {len(all_packages)} packages with robust strategies...")
        
        # Use our robust package installer
        success = self.package_installer.install_packages(all_packages)
        
        if success:
            print("✓ All packages installed successfully")
        else:
            print(":( Some packages failed to install")
            print("Check the output above for details")
        
        return success
    
    def verify_installation(self) -> bool:
        """Verify that key packages are working"""
        print("\n=== Installation Verification ===")
        
        test_packages = [
            ("torch", "PyTorch"),
            ("cv2", "OpenCV"),
            ("paddleocr", "PaddleOCR"),
            ("ultralytics", "Ultralytics YOLO"),
            ("fitz", "PyMuPDF")
        ]
        
        failed_packages = []
        
        for package, description in test_packages:
            try:
                result = subprocess.run([
                    str(self.venv_python), "-c", f"import {package}; print('✓ {description} working')"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print(result.stdout.strip())
                else:
                    print(f"✗ {description} failed to import")
                    failed_packages.append(description)
            except subprocess.TimeoutExpired:
                print(f":( {description} import test timed out")
                failed_packages.append(description)
            except Exception as e:
                print(f"✗ {description} test failed: {e}")
                failed_packages.append(description)
        
        if failed_packages:
            print(f"\n:( {len(failed_packages)} packages failed verification:")
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
                directory.mkdir(parents=True, exist_ok=True)
                print(f"✓ {directory}")
            except Exception as e:
                print(f"✗ Failed to create {directory}: {e}")
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
            with open(activate_script, 'w') as f:
                f.write(content)
            
            if self.system != 'windows':
                os.chmod(activate_script, 0o755)
            
            print(f"✓ Created: {activate_script}")
            return True
        except Exception as e:
            print(f"✗ Failed to create activation script: {e}")
            return False
    
    def run_complete_setup(self) -> bool:
        """Run the complete enhanced setup process"""
        print("Enhanced PLC Diagram Processor Setup")
        print("=" * 60)
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Detecting system capabilities", lambda: (self.detect_system_capabilities(), True)[1]),
            ("Installing system dependencies", self.install_system_dependencies),
            ("Setting up build environment", lambda: self.setup_build_environment(self.capabilities)),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Upgrading pip tools", self.upgrade_pip_tools),
            ("Installing PyTorch with GPU support", lambda: self.install_pytorch_for_gpu(self.capabilities["gpu_info"])),
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
        print("✓ ENHANCED SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Show GPU status
        if self.capabilities and self.capabilities["gpu_info"]["has_nvidia_gpu"]:
            gpu_info = self.capabilities["gpu_info"]
            print(f"\n GPU Status: {gpu_info['gpu_models'][0]} with CUDA {gpu_info.get('cuda_version', 'Unknown')}")
            print(f"PyTorch installation: {gpu_info['pytorch_index_url']}")
        else:
            print("\n GPU Status: CPU-only (no CUDA GPU detected)")
        
        # Show WSL GPU status if available
        if hasattr(self, 'wsl_gpu_info') and self.wsl_gpu_info['available']:
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
    parser = argparse.ArgumentParser(description='Enhanced PLC Diagram Processor Setup')
    parser.add_argument('--data-root', type=str,
                       help='Custom data root directory (default: ../plc-data)')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run system capability tests')
    
    args = parser.parse_args()
    
    setup = EnhancedPLCSetup(data_root=args.data_root)
    
    if args.test_only:
        print("Running system capability tests only...")
        capabilities = setup.detect_system_capabilities()
        
        print("\n=== Test Results ===")
        gpu_info = capabilities["gpu_info"]
        build_info = capabilities["build_tools_status"]
        
        print(f"GPU Support: {'✓' if gpu_info['has_nvidia_gpu'] else '✗'}")
        if gpu_info['has_nvidia_gpu']:
            print(f"  GPU: {gpu_info['gpu_models'][0]}")
            print(f"  CUDA: {gpu_info.get('cuda_version', 'Not detected')}")
        
        print(f"Build Tools: {'✓' if not build_info['needs_installation'] else '✗'}")
        if build_info['needs_installation']:
            print(f"  Recommended: {build_info['installation_method']}")
        
        return 0
    
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
