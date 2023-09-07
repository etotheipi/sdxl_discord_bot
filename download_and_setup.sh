#!/bin/bash

# Hardcoded paths
CKPT_DIR="./ComfyUI/models/checkpoints"
SDXL_BASE="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
SDXL_REFINER="https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors"
VENV_NAME="sdxlbot"

# We depend on two git submodules.  They are not checked out by default, get them now.
git submodule update --init

# Check if 'conda' command is available
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Installing Miniconda3..."

    # Download and install Miniconda (assuming Linux 64-bit here; adapt for other OS)
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b -p ~/miniconda

    # Initialize conda for bash (this may be specific to your shell)
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"

    # Update conda base environment
    conda update -n base conda
else
    echo "Conda is already installed."
fi


# Check if the environment exists
conda info --envs | awk '{print $1}' | grep -x $VENV_NAME &> /dev/null

if [ $? -eq 0 ]; then
    echo "Conda environment '$VENV_NAME' already exists."
else
    echo "Creating new Conda environment '$VENV_NAME' with Python 3.10..."
    conda create -n $VENV_NAME python=3.10 -y
fi


# Activate the environment
eval "$(conda shell.bash hook)"
conda activate $VENV_NAME

# Install packages from requirements.txt if the file exists
conda install -y -c nvidia cuda-toolkit
pip install -r ComfyUI/requirements.txt
pip install -r requirements.txt


# Download the model files
echo "---------------------------------------------------"
echo "Downloading StableDiffusion XL 1.0 BASE model..."
if [ -f "$CKPT_DIR/sd_xl_base_1.0.safetensors" ]; then
    echo "Base model already downloaded, skipping"
    echo "NOTE: if a previous download was interrupted, this could be erroneous.  Use 'rm $CKPT_DIR/sd_xl_base_1.0.safetensors' to reset"
else
    wget $SDXL_BASE -P $CKPT_DIR
fi

echo "---------------------------------------------------"
echo "Downloading StableDiffusion XL 1.0 REFINER model..."
if [ -f "$CKPT_DIR/sd_xl_refiner_1.0.safetensors" ]; then
    echo "Refiner model already downloaded, skipping"
    echo "NOTE: if a previous download was interrupted, this could be erroneous.  Use 'rm $CKPT_DIR/sd_xl_refiner_1.0.safetensors' to reset"
else
    wget $SDXL_REFINER -P $CKPT_DIR
fi
echo "---------------------------------------------------"

# Deactivate virtual environment
conda deactivate

echo ""
echo "Done!"
echo ""
echo "To start the SDXL bot in one command, use the script ./run_bot.sh"
echo "To use the newly-configured environment any other way, use \"conda activate $VENV_NAME\""
