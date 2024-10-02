#!/bin/bash

# MAKE THIS SCRIPT EXECUTABLE BY RUNNING: chmod +x install_miniconda.sh

# Step 1: Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# Step 2: Make the Miniconda installer script executable
chmod +x $HOME/miniconda.sh

# Step 3: Run the Miniconda installer script with -b (batch mode) and -p (prefix to install location)
bash $HOME/miniconda.sh -b -p $HOME/miniconda

# Step 4: Remove the installer script
rm $HOME/miniconda.sh

# Step 5: Initialize Conda in the shell
$HOME/miniconda/bin/conda init

# Step 6: Source .bashrc to update the PATH
source ~/.bashrc

# Step 7: Ensure Conda is available in the current shell session
source $HOME/miniconda/etc/profile.d/conda.sh

# Step 8: Check if Conda is installed properly
conda --version

# Step 9: Set channel priority
conda config --set channel_priority flexible

echo "Miniconda installation completed"

