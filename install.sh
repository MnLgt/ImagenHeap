!#/bin/bash 

echo "Checking your installing in the venv"
# Check if inside a virtual environment or a Conda environment
if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_PREFIX" ]]; then
    echo "Not running inside a virtual or Conda environment"
    exit 1
fi

echo "Installing git lfs"
git lfs install

echo "Cloning GroundingDINO repo"
git clone https://github.com/IDEA-Research/GroundingDINO.git

# Install segment-anything 
pip install git+https://github.com/facebookresearch/segment-anything.git

pip install -r requirements.txt

# python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
pip install --upgrade diffusers[torch]
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel

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
pip install ipykernel ipywidgets
python -m ipykernel install --user --name groundingdino

echo "Adding git --global config"
git config --global credential.helper store
git config --global user.name "jordan davis"
git config --global user.email "jordandavis16@gmail.com"