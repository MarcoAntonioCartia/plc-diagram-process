"""
Build Tools Installer for Windows
Automatically installs Visual Studio Build Tools to resolve compilation issues
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
        Check current status of build tools installation
        
        Returns:
            Dictionary with build tools status information
        """
        print(" Checking build tools status...")
        
        status = {
            "has_visual_studio": False,
            "has_build_tools": False,
            "has_msvc_compiler": False,
            "has_windows_sdk": False,
            "compiler_paths": [],
            "needs_installation": True,
            "installation_method": None
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
