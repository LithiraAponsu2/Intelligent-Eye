#!/bin/sh

# Activate Miniconda (if not already activated)
# eval "$(conda shell.bash hook)"

# Define paths and variables
# PROJECT_DIR="."
# CONDA_ENV_NAME="myenv"
# CONDA_ENV_PATH="$CONDA_ENV_NAME"

# Create and activate Conda environment
# conda create -y -n violationenv python=3.10
conda activate violationenv

# echo "requirements start"

# # Install required Python packages
# pip3 install -r requirements.txt

# echo "requirements done"

# Remove existing Conda environment archive if it exists
file="CONDA_ENV_NAME.tar.gz"
if [ -f "$file" ]; then
  echo "Removing existing $file file"
  rm -rf $file
fi

# Package Conda environment
conda pack -n violationenv -o $file

# Display final directory structure
echo "Final directory structure:"
tree 

# Deactivate Conda environment
conda deactivate
