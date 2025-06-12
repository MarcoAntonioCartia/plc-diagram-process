"""
Enhanced Build Tools Installer for Windows
Automatically installs Visual Studio Build Tools and Rust/Cargo to resolve compilation issues
Includes comprehensive C++ compiler detection and VS environment activation
"""

import subprocess
import sys
import os
import platform
import tempfile
import urllib.request
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, List

class BuildToolsInstaller:
    """Handles automatic installation of Visual Studio Build Tools on Windows"""
    
    def __init__(self):
        self.system_info = {
            "platform": platform.system(),
            "architecture": platform.architecture()[0]
        }
        self.system_info["is_admin"] = self._check_admin_rights()
        self.vs_installer_url = "https://aka.ms/vs/17/release/vs_buildtools.exe"
        self.temp_dir = Path(tempfile.gettempdir()) / "plc_setup"
        self.temp_dir.mkdir(exist_ok=True)
    
    def _check_admin_rights(self) -> bool:
        """Check if running with administrator privileges"""
        if self.system_info["platform"] != "Windows":
            return True  # Not applicable on non-Windows
        
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except:
            return False
    
    def check_build_tools_status(self) -> Dict:
        """
        Enhanced check of build tools installation status including Rust/Cargo
        
        Returns:
            Dictionary with comprehensive build tools status information
        """
        print(" Checking enhanced build tools status...")
        
        status = {
            "has_visual_studio": False,
            "has_build_tools": False,
            "has_msvc_compiler": False,
            "has_windows_sdk": False,
            "has_rust": False,
            "has_cargo": False,
            "compiler_paths": [],
            "vcvars_paths": [],
            "rust_version": None,
            "cargo_version": None,
            "needs_installation": True,
            "needs_rust_installation": True,
            "installation_method": None,
            "vs_environment_working": False
        }
        
        if self.system_info["platform"] != "Windows":
            status["needs_installation"] = False
            status["installation_method"] = "not_windows"
            return status
        
        # Check for MSVC compiler
        msvc_paths = self._find_msvc_compiler()
        if msvc_paths:
            status["has_msvc_compiler"] = True
            status["compiler_paths"] = msvc_paths
            status["needs_installation"] = False
            print("✓ MSVC compiler found")
        
        # Check for Visual Studio installations
        vs_installations = self._find_visual_studio_installations()
        if vs_installations:
            status["has_visual_studio"] = True
            status["needs_installation"] = False
            print("✓ Visual Studio installation found")
        
        # Check for Build Tools specifically
        build_tools_installations = self._find_build_tools_installations()
        if build_tools_installations:
            status["has_build_tools"] = True
            status["needs_installation"] = False
            print("✓ Visual Studio Build Tools found")
        
        # Check for Windows SDK
        if self._check_windows_sdk():
            status["has_windows_sdk"] = True
            print("✓ Windows SDK found")
        
        # Check for vcvars64.bat scripts
        vcvars_paths = self._find_vcvars_scripts()
        if vcvars_paths:
            status["vcvars_paths"] = vcvars_paths
            status["vs_environment_working"] = self._test_vs_environment_activation(vcvars_paths[0])
            if status["vs_environment_working"]:
                print("✓ VS environment activation working")
        
        # Check for Rust and Cargo
        rust_status = self._check_rust_cargo()
        status.update(rust_status)
        
        # Determine installation method if needed
        if status["needs_installation"]:
            status["installation_method"] = self._determine_installation_method()
        
        return status
    
    def _find_msvc_compiler(self) -> List[str]:
        """Find MSVC compiler (cl.exe) in common locations"""
        compiler_paths = []
        
        # Common MSVC paths
        msvc_base_paths = [
            "C:\\Program Files\\Microsoft Visual Studio",
            "C:\\Program Files (x86)\\Microsoft Visual Studio",
            "C:\\BuildTools\\VC\\Tools\\MSVC"
        ]
        
        for base_path in msvc_base_paths:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    if "cl.exe" in files:
                        compiler_paths.append(os.path.join(root, "cl.exe"))
        
        # Check PATH
        try:
            result = subprocess.run(["where", "cl"], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip() and line.strip() not in compiler_paths:
                        compiler_paths.append(line.strip())
        except:
            pass
        
        return compiler_paths
    
    def _find_visual_studio_installations(self) -> List[Dict]:
        """Find Visual Studio installations using vswhere"""
        installations = []
        
        try:
            # Try to find vswhere
            vswhere_paths = [
                "C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe",
                "C:\\Program Files\\Microsoft Visual Studio\\Installer\\vswhere.exe"
            ]
            
            vswhere_exe = None
            for path in vswhere_paths:
                if os.path.exists(path):
                    vswhere_exe = path
                    break
            
            if vswhere_exe:
                result = subprocess.run([
                    vswhere_exe, "-latest", "-products", "*", 
                    "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property", "installationPath"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            installations.append({"path": line.strip(), "type": "visual_studio"})
        except:
            pass
        
        return installations
    
    def _find_build_tools_installations(self) -> List[Dict]:
        """Find Build Tools installations"""
        installations = []
        
        build_tools_paths = [
            "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools",
            "C:\\Program Files\\Microsoft Visual Studio\\2022\\BuildTools",
            "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools",
            "C:\\Program Files\\Microsoft Visual Studio\\2019\\BuildTools",
            "C:\\BuildTools"
        ]
        
        for path in build_tools_paths:
            if os.path.exists(path):
                vc_tools_path = os.path.join(path, "VC", "Tools", "MSVC")
                if os.path.exists(vc_tools_path):
                    installations.append({"path": path, "type": "build_tools"})
        
        return installations
    
    def _check_windows_sdk(self) -> bool:
        """Check for Windows SDK installation"""
        sdk_paths = [
            "C:\\Program Files (x86)\\Windows Kits\\10",
            "C:\\Program Files\\Windows Kits\\10"
        ]
        
        for path in sdk_paths:
            if os.path.exists(path):
                include_path = os.path.join(path, "Include")
                if os.path.exists(include_path):
                    return True
        
        return False
    
    def _determine_installation_method(self) -> str:
        """Determine the best installation method"""
        if not self.system_info["is_admin"]:
            return "need_admin"
        
        # Check if chocolatey is available
        try:
            subprocess.run(["choco", "--version"], capture_output=True, check=True)
            return "chocolatey"
        except:
            pass
        
        # Check if winget is available
        try:
            subprocess.run(["winget", "--version"], capture_output=True, check=True)
            return "winget"
        except:
            pass
        
        # Default to direct download
        return "direct_download"
    
    def install_build_tools(self, method: Optional[str] = None) -> bool:
        """
        Install Visual Studio Build Tools
        
        Args:
            method: Installation method ('chocolatey', 'winget', 'direct_download', or None for auto)
            
        Returns:
            True if installation was successful
        """
        if self.system_info["platform"] != "Windows":
            print("ℹ Build tools installation not needed on non-Windows systems")
            return True
        
        status = self.check_build_tools_status()
        if not status["needs_installation"]:
            print("✓ Build tools already installed")
            return True
        
        if not self.system_info["is_admin"]:
            print("⚠ Administrator privileges required for build tools installation")
            print("Please run the setup script as administrator or install manually:")
            print("1. Download: https://aka.ms/vs/17/release/vs_buildtools.exe")
            print("2. Run with: --add Microsoft.VisualStudio.Workload.VCTools")
            return False
        
        installation_method = method or status["installation_method"]
        
        print(f" Installing Visual Studio Build Tools using {installation_method}...")
        
        if installation_method == "chocolatey":
            return self._install_via_chocolatey()
        elif installation_method == "winget":
            return self._install_via_winget()
        elif installation_method == "direct_download":
            return self._install_via_direct_download()
        else:
            print(f"⚠ Unknown installation method: {installation_method}")
            return False
    
    def _install_via_chocolatey(self) -> bool:
        """Install build tools via Chocolatey"""
        try:
            print("Installing via Chocolatey...")
            result = subprocess.run([
                "choco", "install", "visualstudio2022buildtools", 
                "--package-parameters", "--add Microsoft.VisualStudio.Workload.VCTools",
                "-y"
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            if result.returncode == 0:
                print("✓ Build tools installed successfully via Chocolatey")
                return True
            else:
                print(f"✗ Chocolatey installation failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("⚠ Installation timed out")
            return False
        except Exception as e:
            print(f"✗ Chocolatey installation error: {e}")
            return False
    
    def _install_via_winget(self) -> bool:
        """Install build tools via winget"""
        try:
            print("Installing via winget...")
            result = subprocess.run([
                "winget", "install", "Microsoft.VisualStudio.2022.BuildTools",
                "--silent", "--accept-package-agreements", "--accept-source-agreements"
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            if result.returncode == 0:
                print("✓ Build tools installed successfully via winget")
                return True
            else:
                print(f"✗ winget installation failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("⚠ Installation timed out")
            return False
        except Exception as e:
            print(f"✗ winget installation error: {e}")
            return False
    
    def _install_via_direct_download(self) -> bool:
        """Install build tools via direct download"""
        try:
            print("Downloading Visual Studio Build Tools installer...")
            installer_path = self.temp_dir / "vs_buildtools.exe"
            
            # Download installer
            urllib.request.urlretrieve(self.vs_installer_url, installer_path)
            print("✓ Installer downloaded")
            
            # Run installer
            print("Running installer (this may take several minutes)...")
            result = subprocess.run([
                str(installer_path),
                "--quiet", "--wait",
                "--add", "Microsoft.VisualStudio.Workload.VCTools",
                "--add", "Microsoft.VisualStudio.Component.Windows10SDK.19041"
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            # Clean up
            try:
                installer_path.unlink()
            except:
                pass
            
            if result.returncode == 0:
                print("✓ Build tools installed successfully")
                return True
            else:
                print(f"✗ Installation failed with return code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠ Installation timed out")
            return False
        except Exception as e:
            print(f"✗ Direct download installation error: {e}")
            return False
    
    def verify_installation(self) -> bool:
        """Verify that build tools were installed correctly"""
        print(" Verifying build tools installation...")
        
        status = self.check_build_tools_status()
        
        if status["has_msvc_compiler"]:
            print("✓ MSVC compiler verification passed")
            return True
        else:
            print("✗ MSVC compiler not found after installation")
            return False
    
    def get_compiler_environment(self) -> Dict[str, str]:
        """Get environment variables needed for compilation"""
        env_vars = {}
        
        if self.system_info["platform"] != "Windows":
            return env_vars
        
        # Try to find vcvarsall.bat and get environment
        vcvarsall_paths = []
        
        # Common locations for vcvarsall.bat
        base_paths = [
            "C:\\Program Files\\Microsoft Visual Studio",
            "C:\\Program Files (x86)\\Microsoft Visual Studio",
            "C:\\BuildTools"
        ]
        
        for base_path in base_paths:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    if "vcvarsall.bat" in files:
                        vcvarsall_paths.append(os.path.join(root, "vcvarsall.bat"))
        
        if vcvarsall_paths:
            try:
                # Use the first found vcvarsall.bat
                vcvarsall = vcvarsall_paths[0]
                
                # Run vcvarsall and capture environment
                result = subprocess.run([
                    "cmd", "/c", f'"{vcvarsall}" x64 && set'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key.strip()] = value.strip()
            except Exception as e:
                print(f"⚠ Could not get compiler environment: {e}")
        
        return env_vars

    def _find_vcvars_scripts(self) -> List[str]:
        """Find vcvars64.bat scripts"""
        vcvars_paths = []
        
        possible_paths = [
            "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat",
            "C:\\Program Files\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat",
            "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat",
            "C:\\Program Files\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat",
            "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat",
            "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                vcvars_paths.append(path)
        
        return vcvars_paths

    def _test_vs_environment_activation(self, vcvars_path: str) -> bool:
        """Test if VS environment can be activated"""
        try:
            test_cmd = f'"{vcvars_path}" && where cl.exe'
            result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True, timeout=10)
            return result.returncode == 0 and result.stdout.strip()
        except:
            return False

    def _check_rust_cargo(self) -> Dict:
        """Check for Rust and Cargo installation"""
        rust_status = {
            "has_rust": False,
            "has_cargo": False,
            "rust_version": None,
            "cargo_version": None,
            "needs_rust_installation": True
        }
        
        # Check for Rust
        try:
            result = subprocess.run(["rustc", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                rust_status["has_rust"] = True
                rust_status["rust_version"] = result.stdout.strip()
                print(f"✓ Rust found: {rust_status['rust_version']}")
        except:
            pass
        
        # Check for Cargo
        try:
            result = subprocess.run(["cargo", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                rust_status["has_cargo"] = True
                rust_status["cargo_version"] = result.stdout.strip()
                print(f"✓ Cargo found: {rust_status['cargo_version']}")
        except:
            pass
        
        # Update needs_rust_installation
        if rust_status["has_rust"] and rust_status["has_cargo"]:
            rust_status["needs_rust_installation"] = False
        
        return rust_status

    def install_rust_cargo(self, method: Optional[str] = None) -> bool:
        """
        Install Rust and Cargo
        
        Args:
            method: Installation method ('chocolatey', 'winget', 'rustup', or None for auto)
            
        Returns:
            True if installation was successful
        """
        print(" Installing Rust and Cargo...")
        
        if self.system_info["platform"] != "Windows":
            print("ℹ Using system package manager for Rust installation")
            return self._install_rust_linux()
        
        # Check current status
        rust_status = self._check_rust_cargo()
        if not rust_status["needs_rust_installation"]:
            print("✓ Rust and Cargo already installed")
            return True
        
        # Determine installation method
        if method is None:
            if self._is_chocolatey_available():
                method = "chocolatey"
            elif self._is_winget_available():
                method = "winget"
            else:
                method = "rustup"
        
        print(f" Installing Rust via {method}...")
        
        if method == "chocolatey":
            return self._install_rust_via_chocolatey()
        elif method == "winget":
            return self._install_rust_via_winget()
        elif method == "rustup":
            return self._install_rust_via_rustup()
        else:
            print(f"⚠ Unknown Rust installation method: {method}")
            return False

    def _is_chocolatey_available(self) -> bool:
        """Check if Chocolatey is available"""
        try:
            subprocess.run(["choco", "--version"], capture_output=True, check=True, timeout=5)
            return True
        except:
            return False

    def _is_winget_available(self) -> bool:
        """Check if winget is available"""
        try:
            subprocess.run(["winget", "--version"], capture_output=True, check=True, timeout=5)
            return True
        except:
            return False

    def _install_rust_via_chocolatey(self) -> bool:
        """Install Rust via Chocolatey"""
        try:
            print("Installing Rust via Chocolatey...")
            result = subprocess.run([
                "choco", "install", "rust", "-y"
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                print("✓ Rust installed successfully via Chocolatey")
                return True
            else:
                print(f"✗ Chocolatey Rust installation failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("⚠ Rust installation timed out")
            return False
        except Exception as e:
            print(f"✗ Chocolatey Rust installation error: {e}")
            return False

    def _install_rust_via_winget(self) -> bool:
        """Install Rust via winget"""
        try:
            print("Installing Rust via winget...")
            result = subprocess.run([
                "winget", "install", "Rustlang.Rust.MSVC",
                "--silent", "--accept-package-agreements", "--accept-source-agreements"
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                print("✓ Rust installed successfully via winget")
                return True
            else:
                print(f"✗ winget Rust installation failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("⚠ Rust installation timed out")
            return False
        except Exception as e:
            print(f"✗ winget Rust installation error: {e}")
            return False

    def _install_rust_via_rustup(self) -> bool:
        """Install Rust via rustup (official installer)"""
        try:
            print("Installing Rust via rustup...")
            
            # Download rustup-init.exe
            rustup_url = "https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe"
            rustup_path = self.temp_dir / "rustup-init.exe"
            
            print("Downloading rustup-init.exe...")
            urllib.request.urlretrieve(rustup_url, rustup_path)
            
            # Run rustup-init with default settings
            print("Running rustup installer...")
            result = subprocess.run([
                str(rustup_path), "-y", "--default-toolchain", "stable"
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            # Clean up
            try:
                rustup_path.unlink()
            except:
                pass
            
            if result.returncode == 0:
                print("✓ Rust installed successfully via rustup")
                
                # Add Rust to PATH for current session
                cargo_bin = os.path.expanduser("~/.cargo/bin")
                if os.path.exists(cargo_bin):
                    current_path = os.environ.get('PATH', '')
                    if cargo_bin not in current_path:
                        os.environ['PATH'] = f"{cargo_bin};{current_path}"
                        print(f"✓ Added {cargo_bin} to PATH")
                
                return True
            else:
                print(f"✗ rustup installation failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠ Rust installation timed out")
            return False
        except Exception as e:
            print(f"✗ rustup installation error: {e}")
            return False

    def _install_rust_linux(self) -> bool:
        """Install Rust on Linux systems"""
        try:
            print("Installing Rust via curl | sh...")
            result = subprocess.run([
                "curl", "--proto", "=https", "--tlsv1.2", "-sSf", 
                "https://sh.rustup.rs", "|", "sh", "-s", "--", "-y"
            ], shell=True, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("✓ Rust installed successfully")
                return True
            else:
                print(f"✗ Rust installation failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"✗ Rust installation error: {e}")
            return False

    def install_with_vs_environment(self, package: str, venv_pip: str) -> bool:
        """
        Install a package with VS environment activated
        
        Args:
            package: Package name to install
            venv_pip: Path to virtual environment pip
            
        Returns:
            True if installation was successful
        """
        print(f" Installing {package} with VS environment...")
        
        # Find vcvars script
        vcvars_paths = self._find_vcvars_scripts()
        if not vcvars_paths:
            print("✗ No vcvars64.bat script found")
            return False
        
        vcvars_path = vcvars_paths[0]
        
        try:
            # Create batch script for installation
            install_script = f'''@echo off
call "{vcvars_path}"
"{venv_pip}" install {package}
'''
            
            script_path = self.temp_dir / "install_with_vs.bat"
            with open(script_path, 'w') as f:
                f.write(install_script)
            
            # Run installation with VS environment
            result = subprocess.run([str(script_path)], capture_output=True, text=True, timeout=1800)
            
            # Clean up
            try:
                script_path.unlink()
            except:
                pass
            
            if result.returncode == 0:
                print(f"✓ {package} installed successfully with VS environment")
                return True
            else:
                print(f"✗ {package} installation failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⚠ {package} installation timed out")
            return False
        except Exception as e:
            print(f"✗ {package} installation error: {e}")
            return False

def main():
    """Test the build tools installer"""
    installer = BuildToolsInstaller()
    
    # Check current status
    status = installer.check_build_tools_status()
    print(f"\nBuild Tools Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Install if needed
    if status["needs_installation"]:
        print(f"\nBuild tools installation needed")
        print(f"Recommended method: {status['installation_method']}")
        
        # Ask user for confirmation
        response = input("\nInstall Visual Studio Build Tools? (y/n): ")
        if response.lower() == 'y':
            success = installer.install_build_tools()
            if success:
                installer.verify_installation()
    else:
        print("\n✓ Build tools are already installed")

if __name__ == "__main__":
    main()
