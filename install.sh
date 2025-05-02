#!/bin/bash
# Thirdeye Installation Script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "========================================"
echo "  Thirdeye Deepfake Detection System"
echo "  Installation Script"
echo "========================================"
echo -e "${NC}"

# Check if this is being run on Windows
if [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo -e "${YELLOW}Windows detected. This script might not work correctly.${NC}"
    echo -e "${YELLOW}For Windows, we recommend manually creating a virtual environment and installing from requirements.txt:${NC}"
    echo -e "${BLUE}python -m venv thirdeye_env${NC}"
    echo -e "${BLUE}thirdeye_env\\Scripts\\activate${NC}"
    echo -e "${BLUE}pip install -r requirements.txt${NC}"
    echo -e "${YELLOW}Continue anyway? [y/N]${NC}"
    read -r CONTINUE
    CONTINUE=${CONTINUE:-N}
    if [[ ! $CONTINUE =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Python 3.8 or higher
echo -e "${YELLOW}Checking Python version...${NC}"
if command_exists python3; then
    PYTHON_CMD="python3"
else
    if command_exists python; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}Python not found.${NC}"
        echo -e "${YELLOW}Please install Python 3.8 or higher and try again.${NC}"
        exit 1
    fi
fi

# Verify Python version
PY_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)

if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 8 ]; then
    echo -e "${GREEN}Python $PY_VERSION is installed and meets requirements.${NC}"
else
    echo -e "${RED}Python 3.8 or higher is required, but $PY_VERSION was found.${NC}"
    echo -e "${YELLOW}Please install Python 3.8 or higher and try again.${NC}"
    exit 1
fi

# Check for pip
echo -e "${YELLOW}Checking for pip...${NC}"
if $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
    echo -e "${GREEN}pip is installed as a module.${NC}"
    PIP_CMD="$PYTHON_CMD -m pip"
else
    echo -e "${RED}pip not found. Installing pip...${NC}"
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $PYTHON_CMD get-pip.py
    rm get-pip.py
    PIP_CMD="$PYTHON_CMD -m pip"
fi

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
$PIP_CMD install --upgrade pip

# Check for ffmpeg
echo -e "${YELLOW}Checking for ffmpeg...${NC}"
if command_exists ffmpeg; then
    echo -e "${GREEN}ffmpeg is installed.${NC}"
else
    echo -e "${RED}ffmpeg not found.${NC}"
    echo -e "${YELLOW}ffmpeg is required for video processing.${NC}"
    echo -e "${YELLOW}Please install ffmpeg manually using your package manager:${NC}"
    echo -e "${BLUE}  - Debian/Ubuntu: sudo apt-get install ffmpeg${NC}"
    echo -e "${BLUE}  - macOS: brew install ffmpeg${NC}"
    echo -e "${BLUE}  - Windows: choco install ffmpeg${NC}"
    echo -e "${YELLOW}Continue installation without ffmpeg? [y/N]${NC}"
    read -r CONTINUE
    CONTINUE=${CONTINUE:-N}
    if [[ ! $CONTINUE =~ ^[Yy]$ ]]; then
        echo -e "${RED}Installation aborted.${NC}"
        exit 1
    fi
fi

# Create requirements.txt
echo -e "${YELLOW}Creating requirements.txt...${NC}"
cat > requirements.txt << EOF
# Core ML frameworks
tensorflow>=2.8.0
numpy>=1.22.0
scikit-learn>=1.1.0
pandas>=1.4.0

# Computer vision and video processing
opencv-python>=4.7.0
face-recognition>=1.3.0
moviepy>=1.0.3
dlib>=19.24.0

# Data visualization
matplotlib>=3.5.0
seaborn>=0.12.0

# Utilities and configuration
pyyaml>=6.0
tqdm>=4.64.0
typing-extensions>=4.4.0

# GPU acceleration (optional)
tensorflow-gpu>=2.8.0

# Testing and development
pytest>=7.0.0
black>=22.0.0
pylint>=2.12.0
EOF

# Ask if virtual environment should be created
echo -e "${YELLOW}Do you want to install Thirdeye in a virtual environment? (recommended) [Y/n]${NC}"
read -r USE_VENV
USE_VENV=${USE_VENV:-Y}

if [[ $USE_VENV =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Installing virtualenv...${NC}"
    $PIP_CMD install virtualenv
    
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    $PYTHON_CMD -m virtualenv thirdeye_env
    
    # Activate virtual environment
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source thirdeye_env/bin/activate
    
    # Update commands for activated environment
    PYTHON_CMD="python"
    PIP_CMD="pip"
    
    echo -e "${GREEN}Virtual environment created and activated.${NC}"
    echo -e "${YELLOW}To activate the virtual environment in the future, run:${NC}"
    echo -e "${BLUE}source thirdeye_env/bin/activate${NC}"
fi

# Install from requirements.txt
echo -e "${YELLOW}Installing packages from requirements.txt...${NC}"
$PIP_CMD install -r requirements.txt

# Create directory structure
echo -e "${YELLOW}Creating directory structure...${NC}"
mkdir -p Data/{TRAIN/{REAL_RAW,DF_RAW,REAL_CLIPS,DF_CLIPS,REAL_SAMPLES,DF_SAMPLES,REAL_MV,DF_MV},TEST/{REAL_RAW,DF_RAW,REAL_CLIPS,DF_CLIPS,REAL_SAMPLES,DF_SAMPLES,REAL_MV,DF_MV},UNKNOWN/{UNKNOWN_RAW,UNKNOWN_FPS,UNKNOWN_CLIPS,UNKNOWN_SAMPLES}} Figures Saved_Models

# Installation complete
echo -e "${GREEN}"
echo "========================================"
echo "  Thirdeye installation complete!"
echo "========================================"
echo -e "${NC}"

if [[ $USE_VENV =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Remember to activate the virtual environment before using Thirdeye:${NC}"
    echo -e "${BLUE}source thirdeye_env/bin/activate${NC}"
fi

echo -e "${YELLOW}To test the installation, run:${NC}"
echo -e "${BLUE}python example.py${NC}"

exit 0