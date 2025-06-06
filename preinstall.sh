#!/bin/bash

echo "=== PLC Diagram Processor Setup ==="
echo "Detecting platform..."
OS="$(uname -s)"

echo "Installing system-level dependencies..."

if [[ "$OS" == "Linux" ]]; then
    echo "Linux detected"
    
    # Detect Linux distribution
    if command -v apt &> /dev/null; then
        echo "Using APT package manager (Ubuntu/Debian)"
        sudo apt update
        # Install Python development headers and build tools
        sudo apt install -y python3-dev python3-pip build-essential
        # Install Poppler (for pdf2image)
        sudo apt install -y poppler-utils
        # Install OpenCV dependencies
        sudo apt install -y libglib2.0-0 libsm6 libxrender1 libxext6
        
    elif command -v yum &> /dev/null; then
        echo "Using YUM package manager (CentOS/RHEL)"
        # Install Python development headers and build tools
        sudo yum install -y python3-devel gcc gcc-c++ make
        # Install Poppler (for pdf2image)
        sudo yum install -y poppler-utils
        # Install OpenCV dependencies
        sudo yum install -y glib2-devel libSM-devel libXrender-devel libXext-devel
        
    elif command -v dnf &> /dev/null; then
        echo "Using DNF package manager (Fedora)"
        # Install Python development headers and build tools
        sudo dnf install -y python3-devel gcc gcc-c++ make
        # Install Poppler (for pdf2image)
        sudo dnf install -y poppler-utils
        # Install OpenCV dependencies
        sudo dnf install -y glib2-devel libSM-devel libXrender-devel libXext-devel
        
    else
        echo "Unknown Linux distribution. Please install manually:"
        echo "- Python development headers (python3-dev/python3-devel)"
        echo "- Build tools (build-essential/gcc/gcc-c++/make)"
        echo "- Poppler utilities"
        echo "- OpenCV system dependencies"
        exit 1
    fi
    
elif [[ "$OS" == "Darwin" ]]; then
    echo "macOS detected"
    # Check for Homebrew
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Installing..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    # Install dependencies
    brew install poppler
    brew install opencv
    # macOS usually has development headers available by default with Xcode Command Line Tools
    if ! command -v gcc &> /dev/null; then
        echo "Installing Xcode Command Line Tools..."
        xcode-select --install
    fi
    
else
    echo "Unsupported or Windows environment detected."
    echo "Please install the following manually:"
    echo "1. Python development headers and build tools"
    echo "2. Poppler from: https://github.com/oschwartz10612/poppler-windows/releases"
    echo "   Extract to: bin/poppler/Library/bin/ and add to PATH"
    echo "3. Visual Studio Build Tools for C++ compilation"
    exit 1
fi

echo ""
echo "Setting up Python environment..."

# Create virtual environment if not already created
if [ ! -d "yolovenv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv yolovenv
else
    echo "Virtual environment already exists"
fi

# Activate venv
echo "Activating virtual environment..."
source yolovenv/bin/activate

# Upgrade pip and install build tools
echo "Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "=== Setup Complete ==="
echo "Virtual environment created and activated at: ./yolovenv"
echo "All dependencies installed successfully!"
echo ""
echo "To activate the environment in future sessions, run:"
echo "source yolovenv/bin/activate"