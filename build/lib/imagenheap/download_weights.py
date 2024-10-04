import os
import urllib.request
from pathlib import Path

def download_file(url, filename):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

def download_weights():
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    weights = {
        "groundingdino_swint_ogc.pth": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
        "sam_vit_h_4b8939.pth": "https://huggingface.co/spaces/mrtlive/segment-anything-model/resolve/main/sam_vit_h_4b8939.pth"
    }

    for filename, url in weights.items():
        file_path = weights_dir / filename
        if not file_path.exists():
            download_file(url, file_path)
        else:
            print(f"{filename} already exists, skipping download.")

if __name__ == "__main__":
    download_weights()