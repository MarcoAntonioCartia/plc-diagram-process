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
from typing import Dict, List, Optional, Tuple

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
            print(f"✗ Python {min_version[0]}.{min_version[1]}+ required, but {current_version[0]}.{current_version[1]} found")
            return False
        
        print(f"✓ Python {current_version[0]}.{current_version[1]} detected")
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
                print("⚠ Build tools installation failed")
                print("You may encounter compilation issues with some packages")
                
                # Ask user if they want to continue
                response = input("Continue with setup anyway? (y/n): ")
                if response.lower() != 'y':
                    return False
        else:
            print("✓ Build tools already available")
        
        return True
    
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
    
    def install_pytorch_for_gpu(self, gpu_info: Dict) -> bool:
        """Install PyTorch with appropriate GPU support"""
        print("\n=== PyTorch Installation ===")
        
        description, command = self.gpu_detector.get_pytorch_install_command()
        print(description)
        
        # If no GPU detected, ask user for confirmation
        if not gpu_info["has_nvidia_gpu"] or not gpu_info["has_cuda"]:
            print("\n⚠ No CUDA-capable GPU detected")
            print("PyTorch will be installed with CPU-only support")
            response = input("Continue with CPU-only PyTorch? (y/n): ")
            if response.lower() != 'y':
                print("Please install CUDA drivers and run setup again for GPU support")
                return False
        
        try:
            print("Installing PyTorch (this may take several minutes)...")
            subprocess.run(command, check=True)
            print("✓ PyTorch installed successfully")
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
            print("⚠ Some packages failed to install")
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
