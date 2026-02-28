#!/bin/bash
set -x
set -v 

LOCK2A_DIR="$HOME/LOCK2A"
REPO_URL="https://github.com/Lmah0/Capstone-LOCK-2A.git" 
VENV_NAME="venv"

# Create LOCK2A directory 
if [ -d "$LOCK2A_DIR" ]; then

    read -p "Directory $LOCK2A_DIR already exists. Do you want to remove it and continue? (y/n): " answer
    
    if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
        echo "Directory $LOCK2A_DIR already exists. Nuking it..."
        rm -rf $LOCK2A_DIR
        echo "Creating directory: $LOCK2A_DIR"
        mkdir -p $LOCK2A_DIR    
    else
        echo "Exiting."
        exit
    fi
fi

mkdir -p $LOCK2A_DIR

# Navigate to LOCK2A
cd $LOCK2A_DIR || { echo "Failed to access LOCK2A"; exit 1; }

echo "Updating system and installing GStreamer dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-gi \
    python3-gst-1.0 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    libgirepository1.0-dev \
    libcairo2-dev \
    pkg-config

# Clone capstone repo 
if [ ! -d "Capstone-LOCK-2A" ]; then
    echo "Cloning repository..."
    git clone "$REPO_URL"
else
    echo "Repository already exists. Skipping clone."
fi

# Setup Python virtual environment
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating Python virtual environment ($VENV_NAME)..."
    python3 -m venv --system-site-packages "$VENV_NAME"
    echo "$(ls)" 
else
    echo "Virtual environment ($VENV_NAME) already exists."
fi

#Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install pycairo PyGObject

# Setup .bashrc file
echo "Setting up .bashrc..."

#Add comment to .bashrc
echo -e "\n# LOCK2A Aliases and Functions" >> "$HOME/.bashrc"

# Add alias to cd to ~/LOCK2A
echo "alias cd-lock='cd ~/LOCK2A/Capstone-LOCK-2A'" >> "$HOME/.bashrc"

# Add alias to run server.py
echo "alias run-server='python3 ~/LOCK2A/Capstone-LOCK-2A/Drone/flightComputer/server.py'" >> "$HOME/.bashrc"

# Add alias to run source venv
echo "alias source-lock='source ~/LOCK2A/venv/bin/activate'" >> "$HOME/.bashrc"

# Add bash function to run the mavproxy.py
echo "function runMavMod() {
    mavproxy_path=\"LOCK2A/venv/lib/python3.11/site-packages/MAVProxy/mavproxy.py\"
    if [ ! -f \"\$mavproxy_path\" ]; then
        echo \"Error: Module \$1 not found in MAVProxy modules directory.\"
        return 1
    fi
    python3 mavproxy.py --out=udp:127.0.0.1:5006\"
}" >> "$HOME/.bashrc"

echo "Applying new aliases and functions..."
source "$HOME/.bashrc"


echo -e "\nSetup complete!"
