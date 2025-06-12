"""
PaddleOCR 3.x Standalone Installation Test Suite - Fixed Version
===============================================

This script tests various PaddleOCR installation methods and verifies functionality
independently from the main project setup system.

Author: Assistant AI
Date: January 2025
"""

import subprocess
import sys
import os
import importlib
import platform
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class PaddleOCRInstallationTester:
    """Comprehensive PaddleOCR installation testing suite."""
    
    def __init__(self, test_env_name: str = "paddleocr_test_env"):
        self.test_env_name = test_env_name
        self.platform_info = self._get_platform_info()
        self.test_results = {}
        self.install_log = []
        
    def _get_platform_info(self) -> Dict[str, str]:
        """Get comprehensive platform information."""
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
            "processor": platform.processor() or "Unknown"
        }
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log messages with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {level}: {message}"
        print(log_message)
        self.install_log.append(log_message)
    
    def run_command(self, command: List[str], timeout: int = 300, capture_output: bool = True) -> Tuple[bool, str]:
        """Run a command and return success status and output."""
        try:
            self.log(f"Running command: {' '.join(command)}")
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                check=False
            )
            
            output = ""
            if capture_output:
                output = result.stdout + result.stderr
            
            success = result.returncode == 0
            if not success:
                self.log(f"Command failed with return code {result.returncode}", "ERROR")
                if output:
                    self.log(f"Error output: {output[:500]}...", "ERROR")
            
            return success, output
            
        except subprocess.TimeoutExpired:
            self.log(f"Command timed out after {timeout} seconds", "ERROR")
            return False, "Command timed out"
        except Exception as e:
            self.log(f"Command execution failed: {str(e)}", "ERROR")
            return False, str(e)
    
    def create_virtual_environment(self) -> bool:
        """Create a fresh virtual environment for testing."""
        self.log("Creating virtual environment for testing...")
        
        # Remove existing test environment if it exists
        if os.path.exists(self.test_env_name):
            self.log("Removing existing test environment...")
            import shutil
            shutil.rmtree(self.test_env_name, ignore_errors=True)
        
        # Create new virtual environment
        success, output = self.run_command([
            sys.executable, "-m", "venv", self.test_env_name
        ])
        
        if success:
            self.log("Virtual environment created successfully")
        else:
            self.log("Failed to create virtual environment", "ERROR")
        
        return success
    
    def get_venv_python(self) -> str:
        """Get the Python executable path for the virtual environment."""
        if platform.system() == "Windows":
            return os.path.join(self.test_env_name, "Scripts", "python.exe")
        else:
            return os.path.join(self.test_env_name, "bin", "python")
    
    def get_venv_pip(self) -> str:
        """Get the pip executable path for the virtual environment."""
        if platform.system() == "Windows":
            return os.path.join(self.test_env_name, "Scripts", "pip.exe")
        else:
            return os.path.join(self.test_env_name, "bin", "pip")
    
    def test_installation_method_1(self) -> bool:
        """Test Method 1: Standard PaddleOCR 3.x installation"""
        self.log("Testing Method 1: Standard PaddleOCR 3.x installation")
        
        pip_path = self.get_venv_pip()
        
        # Skip pip upgrade if it fails (not critical)
        success, _ = self.run_command([pip_path, "install", "--upgrade", "pip"])
        if not success:
            self.log("Pip upgrade failed, continuing anyway...", "WARNING")
        
        # Install paddlepaddle 3.0
        self.log("Installing PaddlePaddle 3.0...")
        success, output = self.run_command([
            pip_path, "install", "paddlepaddle==3.0.0",
            "-i", "https://www.paddlepaddle.org.cn/packages/stable/cpu/"
        ], timeout=600)
        
        if not success:
            self.log("Failed to install PaddlePaddle", "ERROR")
            return False
        
        # Install paddleocr 3.x
        self.log("Installing PaddleOCR 3.x...")
        success, output = self.run_command([
            pip_path, "install", "paddleocr"
        ], timeout=600)
        
        if success:
            self.log("Method 1: Installation successful")
        else:
            self.log("Method 1: Installation failed", "ERROR")
        
        return success
    
    def test_installation_method_2(self) -> bool:
        """Test Method 2: Direct PyPI installation"""
        self.log("Testing Method 2: Direct PyPI installation")
        
        pip_path = self.get_venv_pip()
        
        # Install everything from PyPI
        success, output = self.run_command([
            pip_path, "install", "paddlepaddle==3.0.0", "paddleocr"
        ], timeout=600)
        
        if success:
            self.log("Method 2: Installation successful")
        else:
            self.log("Method 2: Installation failed", "ERROR")
        
        return success
    
    def test_installation_method_3(self) -> bool:
        """Test Method 3: Pre-install dependencies approach"""
        self.log("Testing Method 3: Pre-install dependencies approach")
        
        pip_path = self.get_venv_pip()
        
        # Install common dependencies first
        dependencies = [
            "numpy>=1.19.3,<2.0.0",
            "opencv-python>=4.6.0",
            "pillow>=8.2.0",
            "pyyaml>=6.0",
            "shapely>=1.7.0",
            "pyclipper>=1.2.0"
        ]
        
        self.log("Installing dependencies first...")
        for dep in dependencies:
            success, _ = self.run_command([pip_path, "install", dep])
            if not success:
                self.log(f"Failed to install dependency: {dep}", "WARNING")
        
        # Install paddlepaddle
        success, _ = self.run_command([
            pip_path, "install", "paddlepaddle==3.0.0"
        ])
        
        if not success:
            self.log("Failed to install PaddlePaddle", "ERROR")
            return False
        
        # Install paddleocr
        success, _ = self.run_command([
            pip_path, "install", "paddleocr", "--no-deps"
        ])
        
        if not success:
            # Try with deps if no-deps fails
            success, _ = self.run_command([
                pip_path, "install", "paddleocr"
            ])
        
        if success:
            self.log("Method 3: Installation successful")
        else:
            self.log("Method 3: Installation failed", "ERROR")
        
        return success
    
    def test_paddleocr_functionality(self) -> bool:
        """Test basic PaddleOCR functionality."""
        self.log("Testing PaddleOCR functionality...")
        
        python_path = self.get_venv_python()
        
        # Create a simple test script (fixed Unicode issues)
        test_script = f"""
import sys
sys.path.insert(0, '{os.getcwd()}')

try:
    # Test basic import
    print("Testing imports...")
    from paddleocr import PaddleOCR
    print("[OK] PaddleOCR imported successfully")
    
    # Test initialization
    print("Testing initialization...")
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        show_log=False
    )
    print("[OK] PaddleOCR initialized successfully")
    
    # Test with a simple text image (we'll create one)
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a simple test image
    img = Image.new('RGB', (200, 50), color='white')
    draw = ImageDraw.Draw(img)
    try:
        # Try to use a basic font
        font = ImageFont.load_default()
    except:
        font = None
    
    draw.text((10, 10), "Test OCR", fill='black', font=font)
    img.save('test_image.png')
    print("[OK] Test image created")
    
    # Test OCR on the image
    print("Testing OCR recognition...")
    result = ocr.ocr('test_image.png')
    print("[OK] OCR processing completed")
    print(f"Result type: {{type(result)}}")
    
    if result and len(result) > 0:
        print("[OK] OCR returned results")
        print(f"Number of results: {{len(result)}}")
    else:
        print("[WARN] OCR returned empty results (may be normal for simple test)")
    
    print("SUCCESS: All functionality tests passed!")
    
except ImportError as e:
    print(f"IMPORT ERROR: {{e}}")
    sys.exit(1)
except Exception as e:
    print(f"RUNTIME ERROR: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        
        # Write test script to temporary file with UTF-8 encoding
        with open("test_paddleocr.py", "w", encoding='utf-8') as f:
            f.write(test_script)
        
        try:
            # Run the test script
            success, output = self.run_command([
                python_path, "test_paddleocr.py"
            ], timeout=300)
            
            if success and "SUCCESS: All functionality tests passed!" in output:
                self.log("[OK] PaddleOCR functionality test passed")
                return True
            else:
                self.log("[FAIL] PaddleOCR functionality test failed", "ERROR")
                self.log(f"Output: {output}", "ERROR")
                return False
                
        finally:
            # Clean up test files
            for file in ["test_paddleocr.py", "test_image.png"]:
                if os.path.exists(file):
                    os.remove(file)
    
    def run_comprehensive_test(self) -> Dict[str, bool]:
        """Run comprehensive installation tests."""
        self.log("Starting comprehensive PaddleOCR installation tests")
        self.log(f"Platform: {self.platform_info}")
        
        methods = [
            ("Method 1: Standard 3.x", self.test_installation_method_1),
            ("Method 2: Direct PyPI", self.test_installation_method_2),
            ("Method 3: Pre-deps", self.test_installation_method_3),
        ]
        
        results = {}
        
        for method_name, method_func in methods:
            self.log(f"\n{'='*50}")
            self.log(f"Testing {method_name}")
            self.log(f"{'='*50}")
            
            # Create fresh environment for each test
            if self.create_virtual_environment():
                install_success = method_func()
                
                if install_success:
                    # Test functionality
                    func_success = self.test_paddleocr_functionality()
                    results[method_name] = func_success
                else:
                    results[method_name] = False
            else:
                results[method_name] = False
            
            # Clean up between tests
            if os.path.exists(self.test_env_name):
                import shutil
                shutil.rmtree(self.test_env_name, ignore_errors=True)
        
        return results
    
    def generate_report(self, results: Dict[str, bool]) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("PaddleOCR Installation Test Report")
        report.append("=" * 50)
        report.append(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Platform: {self.platform_info['platform']}")
        report.append(f"Python Version: {self.platform_info['python_version']}")
        report.append(f"Architecture: {self.platform_info['architecture']}")
        report.append("")
        
        report.append("Test Results:")
        report.append("-" * 30)
        
        successful_methods = []
        for method, success in results.items():
            status = "[PASS]" if success else "[FAIL]"
            report.append(f"{method}: {status}")
            if success:
                successful_methods.append(method)
        
        report.append("")
        report.append("Recommendations:")
        report.append("-" * 30)
        
        if successful_methods:
            report.append(f"[OK] Successful installation methods: {len(successful_methods)}")
            report.append(f"[OK] Recommended method: {successful_methods[0]}")
            
            if "Method 3: Pre-deps" in successful_methods:
                report.append("\nRecommended installation commands:")
                report.append("```bash")
                report.append("# Create virtual environment")
                report.append("python -m venv paddleocr_env")
                report.append("# Activate environment (Windows)")
                report.append("paddleocr_env\\Scripts\\activate")
                report.append("# Install dependencies first")
                report.append("pip install numpy>=1.19.3,<2.0.0")
                report.append("pip install opencv-python>=4.6.0")
                report.append("pip install pillow>=8.2.0")
                report.append("pip install pyyaml>=6.0")
                report.append("pip install shapely>=1.7.0")
                report.append("pip install pyclipper>=1.2.0")
                report.append("# Install PaddlePaddle 3.0")
                report.append("pip install paddlepaddle==3.0.0")
                report.append("# Install PaddleOCR")
                report.append("pip install paddleocr --no-deps")
                report.append("```")
        else:
            report.append("[FAIL] No installation methods succeeded")
            report.append("[FAIL] Manual investigation required")
        
        report.append("")
        report.append("Installation Log Summary:")
        report.append("-" * 30)
        error_count = len([log for log in self.install_log if "ERROR" in log])
        warning_count = len([log for log in self.install_log if "WARNING" in log])
        report.append(f"Total log entries: {len(self.install_log)}")
        report.append(f"Errors: {error_count}")
        report.append(f"Warnings: {warning_count}")
        
        return "\n".join(report)
    
    def save_detailed_logs(self, filepath: str) -> None:
        """Save detailed installation logs."""
        with open(filepath, "w", encoding='utf-8') as f:
            f.write("Detailed PaddleOCR Installation Logs\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Platform Information:\n")
            for key, value in self.platform_info.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("Detailed Logs:\n")
            f.write("-" * 30 + "\n")
            for log_entry in self.install_log:
                f.write(log_entry + "\n")


def main():
    """Main test execution function."""
    print("PaddleOCR 3.x Installation Test Suite - Fixed Version")
    print("=" * 50)
    
    # Initialize tester
    tester = PaddleOCRInstallationTester()
    
    try:
        # Run comprehensive tests
        results = tester.run_comprehensive_test()
        
        # Generate and save report
        report = tester.generate_report(results)
        print("\n" + report)
        
        # Save detailed report
        report_file = f"paddleocr_installation_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, "w", encoding='utf-8') as f:
            f.write(report)
        
        # Save detailed logs
        log_file = f"paddleocr_installation_logs_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        tester.save_detailed_logs(log_file)
        
        print(f"\nReport saved to: {report_file}")
        print(f"Detailed logs saved to: {log_file}")
        
        # Determine overall success
        successful_methods = [method for method, success in results.items() if success]
        if successful_methods:
            print(f"\nSUCCESS: {len(successful_methods)} installation methods work!")
            print(f"Recommended: {successful_methods[0]}")
            return True
        else:
            print("\nFAILURE: No installation methods succeeded")
            print("Check the detailed logs for troubleshooting information")
            return False
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return False
    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 