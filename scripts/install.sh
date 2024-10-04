!#/bin/bash 

echo "Checking your installing in the venv"
# Check if inside a virtual environment or a Conda environment
if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_PREFIX" ]]; then
    echo "Not running inside a virtual or Conda environment"
    exit 1
fi

echo "Upgrading pip"
pip install -U pip -qqq

# Check if sudo is installed
if ! command -v sudo &> /dev/null
then
    echo "sudo is not installed. Installing sudo..."
    apt-get update && apt-get install -y sudo
    echo "sudo has been installed."
else
    echo "sudo is already installed."
fi

if ! command -v git-lfs &> /dev/null
then
    echo "git-lfs not found. Installing..."
    sudo apt-get update && sudo apt-get install -y git-lfs
    echo "Installing git lfs"
    git lfs install
else
    echo "git-lfs is already installed"
fi

echo "Cloning GroundingDINO repo"
git clone https://github.com/IDEA-Research/GroundingDINO.git

# Install segment-anything 
echo "Installing segment-anything"
pip install git+https://github.com/facebookresearch/segment-anything.git -qqq

echo "Installing requirements"
pip install -r requirements.txt -qqq

# python -m pip install -e segment_anything
echo "Installing GroundingDINO"
python -m pip install -e GroundingDINO -qqq

echo "Installing diffusers"
pip install --upgrade diffusers[torch] -qqq

echo "Installing other dependencies"
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel -qqq

echo "Downloading weights"
# check if weights dir exists
if [ ! -d "weights" ]; then
    mkdir weights
fi

# check if groundingdino weights exists
if [ ! -f "weights/groundingdino_swint_ogc.pth" ]; then
    wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth -O weights/groundingdino_swint_ogc.pth
fi

# check if sam_vit_h_4b8939.pth exists
if [ ! -f "weights/sam_vit_h_4b8939.pth" ]; then
    wget https://huggingface.co/spaces/mrtlive/segment-anything-model/resolve/main/sam_vit_h_4b8939.pth -O weights/sam_vit_h_4b8939.pth
fi

echo "Registering Venv"
pip install ipykernel ipywidgets -qqq
python -m ipykernel install --user --name groundingdino

echo "Adding git --global config"
git config --global credential.helper store
git config --global user.name "jordan davis"
git config --global user.email "jordandavis16@gmail.com"