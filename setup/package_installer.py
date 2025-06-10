"""
Robust Package Installer for PLC Diagram Processor
Handles package installation with multiple fallback strategies to avoid compilation issues
"""

import subprocess
import sys
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

class RobustPackageInstaller:
    """Handles robust package installation with multiple fallback strategies"""
    
    def __init__(self):
        self.system_info = {
            "platform": platform.system(),
            "architecture": platform.architecture()[0],
            "python_version": sys.version_info[:2]
        }
        
        # Package installation strategies
        self.installation_strategies = [
            "wheel_only",           # Try wheel-only first
            "wheel_with_fallback",  # Wheel-only with alternative sources
            "conda_forge",          # Use conda-forge if available
            "source_with_compiler", # Compile from source (last resort)
        ]
        
        # Problematic packages that often need compilation
        self.problematic_packages = {
            "numpy": {
                "alternatives": ["numpy==1.24.3"],  # Known working version
                "wheel_sources": ["https://pypi.org/simple/"],
                "conda_name": "numpy"
            },
            "scipy": {
                "alternatives": ["scipy==1.10.1"],
                "wheel_sources": ["https://pypi.org/simple/"],
                "conda_name": "scipy"
            },
            "opencv-python": {
                "alternatives": ["opencv-python-headless"],
                "wheel_sources": ["https://pypi.org/simple/"],
                "conda_name": "opencv"
            },
            "paddlepaddle": {
                "alternatives": ["paddlepaddle==2.5.2"],
                "wheel_sources": ["https://pypi.org/simple/"],
                "conda_name": None  # Not available in conda
            }
        }
    
    def install_packages_from_requirements(self, requirements_file: Path, 
                                         strategy: Optional[str] = None) -> bool:
        """
        Install packages from requirements file with robust error handling
        
        Args:
            requirements_file: Path to requirements.txt file
            strategy: Installation strategy to use (None for auto)
            
        Returns:
            True if installation was successful
        """
        if not requirements_file.exists():
            print(f"✗ Requirements file not found: {requirements_file}")
            return False
        
        print(f" Installing packages from {requirements_file}")
        
        # Parse requirements file
        packages = self._parse_requirements_file(requirements_file)
        if not packages:
            print("⚠ No packages found in requirements file")
            return True
        
        print(f"Found {len(packages)} packages to install")
        
        # Install packages with robust strategy
        return self._install_packages_robust(packages, strategy)
    
    def install_packages(self, packages: List[str], strategy: Optional[str] = None) -> bool:
        """
        Install a list of packages with robust error handling
        
        Args:
            packages: List of package specifications
            strategy: Installation strategy to use (None for auto)
            
        Returns:
            True if installation was successful
        """
        if not packages:
            return True
        
        print(f" Installing {len(packages)} packages")
        return self._install_packages_robust(packages, strategy)
    
    def _parse_requirements_file(self, requirements_file: Path) -> List[str]:
        """Parse requirements file and extract package specifications"""
        packages = []
        
        try:
            with open(requirements_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Handle -r includes (recursive requirements)
                    if line.startswith('-r '):
                        include_file = requirements_file.parent / line[3:].strip()
                        if include_file.exists():
                            packages.extend(self._parse_requirements_file(include_file))
                        continue
                    
                    # Handle other pip options
                    if line.startswith('-'):
                        continue
                    
                    # Extract package name (handle comments at end of line)
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    
                    if line:
                        packages.append(line)
        
        except Exception as e:
            print(f"⚠ Error parsing requirements file: {e}")
        
        return packages
    
    def _install_packages_robust(self, packages: List[str], strategy: Optional[str] = None) -> bool:
        """Install packages using robust strategies with fallbacks"""
        
        failed_packages = []
        successful_packages = []
        
        for package in packages:
            print(f"\n Installing {package}...")
            
            success = False
            package_name = self._extract_package_name(package)
            
            # Try different strategies
            strategies_to_try = [strategy] if strategy else self.installation_strategies
            
            for install_strategy in strategies_to_try:
                if install_strategy is None:
                    continue
                
                print(f"  Trying strategy: {install_strategy}")
                
                if install_strategy == "wheel_only":
                    success = self._install_wheel_only(package)
                elif install_strategy == "wheel_with_fallback":
                    success = self._install_wheel_with_fallback(package, package_name)
                elif install_strategy == "conda_forge":
                    success = self._install_via_conda(package, package_name)
                elif install_strategy == "source_with_compiler":
                    success = self._install_from_source(package)
                
                if success:
                    print(f"  ✓ {package} installed successfully via {install_strategy}")
                    successful_packages.append(package)
                    break
                else:
                    print(f"  ✗ {install_strategy} failed for {package}")
            
            if not success:
                print(f"  ✗ All strategies failed for {package}")
                failed_packages.append(package)
        
        # Summary
        print(f"\n Installation Summary:")
        print(f"  ✓ Successful: {len(successful_packages)}")
        print(f"  ✗ Failed: {len(failed_packages)}")
        
        if failed_packages:
            print(f"\nFailed packages:")
            for pkg in failed_packages:
                print(f"  - {pkg}")
            
            # Provide guidance for failed packages
            self._provide_failure_guidance(failed_packages)
        
        return len(failed_packages) == 0
    
    def _extract_package_name(self, package_spec: str) -> str:
        """Extract package name from package specification"""
        # Handle various formats: package, package==version, package>=version, etc.
        import re
        match = re.match(r'^([a-zA-Z0-9_-]+)', package_spec)
        return match.group(1) if match else package_spec
    
    def _install_wheel_only(self, package: str) -> bool:
        """Install package using wheel-only strategy"""
        try:
            cmd = [
                sys.executable, "-m", "pip", "install",
                "--only-binary=all",
                "--no-compile",
                package
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print("    ⚠ Installation timed out")
            return False
        except Exception as e:
            print(f"    ⚠ Wheel-only installation error: {e}")
            return False
    
    def _install_wheel_with_fallback(self, package: str, package_name: str) -> bool:
        """Install package with wheel fallback strategies"""
        
        # Try standard wheel installation first
        if self._install_wheel_only(package):
            return True
        
        # Try alternative package versions if available
        if package_name in self.problematic_packages:
            alternatives = self.problematic_packages[package_name]["alternatives"]
            
            for alt_package in alternatives:
                print(f"    Trying alternative: {alt_package}")
                if self._install_wheel_only(alt_package):
                    return True
        
        # Try with different index URLs
        index_urls = [
            "https://pypi.org/simple/",
            "https://pypi.python.org/simple/",
        ]
        
        for index_url in index_urls:
            try:
                cmd = [
                    sys.executable, "-m", "pip", "install",
                    "--only-binary=all",
                    "--index-url", index_url,
                    package
                ]
                
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )
                
                if result.returncode == 0:
                    return True
                    
            except:
                continue
        
        return False
    
    def _install_via_conda(self, package: str, package_name: str) -> bool:
        """Install package via conda if available"""
        
        # Check if conda is available
        try:
            subprocess.run(["conda", "--version"], capture_output=True, check=True)
        except:
            return False  # Conda not available
        
        # Get conda package name
        conda_name = package_name
        if package_name in self.problematic_packages:
            conda_name = self.problematic_packages[package_name].get("conda_name", package_name)
        
        if conda_name is None:
            return False  # Package not available in conda
        
        try:
            cmd = ["conda", "install", "-c", "conda-forge", "-y", conda_name]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print("    ⚠ Conda installation timed out")
            return False
        except Exception as e:
            print(f"    ⚠ Conda installation error: {e}")
            return False
    
    def _install_from_source(self, package: str) -> bool:
        """Install package from source (requires compiler)"""
        
        # Check if we have a compiler available
        if self.system_info["platform"] == "Windows":
            # Check for MSVC compiler
            try:
                subprocess.run(["cl"], capture_output=True, check=True)
            except:
                print("    ⚠ No compiler available for source installation")
                return False
        
        try:
            cmd = [
                sys.executable, "-m", "pip", "install",
                "--no-binary", package,
                package
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=900  # 15 minutes
            )
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print("    ⚠ Source installation timed out")
            return False
        except Exception as e:
            print(f"    ⚠ Source installation error: {e}")
            return False
    
    def _provide_failure_guidance(self, failed_packages: List[str]):
        """Provide guidance for packages that failed to install"""
        
        print(f"\n💡 Installation Guidance:")
        print(f"=" * 50)
        
        for package in failed_packages:
            package_name = self._extract_package_name(package)
            
            print(f"\nFor {package}:")
            
            if package_name in self.problematic_packages:
                alternatives = self.problematic_packages[package_name]["alternatives"]
                print(f"  Try alternative versions:")
                for alt in alternatives:
                    print(f"    pip install {alt}")
            
            if package_name == "numpy":
                print(f"  Numpy compilation issues:")
                print(f"    1. Install Visual Studio Build Tools")
                print(f"    2. Try: pip install --only-binary=all numpy==1.24.3")
                print(f"    3. Or use conda: conda install numpy")
            
            elif package_name == "opencv-python":
                print(f"  OpenCV alternatives:")
                print(f"    1. pip install opencv-python-headless")
                print(f"    2. pip install --only-binary=all opencv-python")
            
            elif package_name in ["paddlepaddle", "paddleocr"]:
                print(f"  PaddlePaddle installation:")
                print(f"    1. Check Python version compatibility")
                print(f"    2. Try CPU version: pip install paddlepaddle==2.5.2")
                print(f"    3. Visit: https://www.paddlepaddle.org.cn/install/quick")
            
            else:
                print(f"  General troubleshooting:")
                print(f"    1. Update pip: python -m pip install --upgrade pip")
                print(f"    2. Try wheel-only: pip install --only-binary=all {package}")
                print(f"    3. Check package documentation for alternatives")
    
    def verify_installation(self, packages: List[str]) -> Dict[str, bool]:
        """Verify that packages were installed correctly"""
        
        print(f"\n🔍 Verifying package installations...")
        
        results = {}
        
        for package in packages:
            package_name = self._extract_package_name(package)
            
            try:
                # Try to import the package
                __import__(package_name)
                results[package_name] = True
                print(f"  ✓ {package_name}")
            except ImportError:
                # Try common alternative import names
                alt_names = {
                    "opencv-python": "cv2",
                    "pillow": "PIL",
                    "pyyaml": "yaml",
                    "scikit-learn": "sklearn"
                }
                
                alt_name = alt_names.get(package_name)
                if alt_name:
                    try:
                        __import__(alt_name)
                        results[package_name] = True
                        print(f"  ✓ {package_name} (as {alt_name})")
                        continue
                    except ImportError:
                        pass
                
                results[package_name] = False
                print(f"  ✗ {package_name}")
        
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        print(f"\nVerification: {successful}/{total} packages working")
        
        return results

def main():
    """Test the package installer"""
    installer = RobustPackageInstaller()
    
    # Test with a small set of packages
    test_packages = [
        "numpy>=1.21.0",
        "opencv-python>=4.6.0",
        "PyMuPDF>=1.23.0"
    ]
    
    print("Testing robust package installation...")
    success = installer.install_packages(test_packages)
    
    if success:
        print("\n✓ All packages installed successfully")
        installer.verify_installation(test_packages)
    else:
        print("\n⚠ Some packages failed to install")

if __name__ == "__main__":
    main()
