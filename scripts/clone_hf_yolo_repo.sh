#!/bin/bash

# Set the repository URL
REPO_URL="https://huggingface.co/MnLgt/yolo-human-parse"

# Set the directory name for the cloned repo
REPO_NAME="yolo-human-parse"

# Navigate to the parent directory
cd ..

# Create the models directory if it doesn't exist
mkdir -p models

# Navigate to the models directory
cd models

# Create and navigate to the repository directory
mkdir "$REPO_NAME"
cd "$REPO_NAME"

# Initialize a new git repository
git init

# Add the remote repository
git remote add origin "$REPO_URL"

# Enable sparse-checkout
git config core.sparseCheckout true

# Create the sparse-checkout file
echo "/*" > .git/info/sparse-checkout
echo "!checkpoint*" >> .git/info/sparse-checkout

# Pull the repository
git pull origin main

echo "Repository cloned successfully into ../models/$REPO_NAME, excluding directories starting with 'checkpoint'."

# Navigate back to the original directory
cd ../../segment