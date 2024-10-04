import os
import warnings

import wandb
import yaml
from dotenv import find_dotenv, load_dotenv
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
from utils import get_device
from pathlib import Path

load_dotenv(find_dotenv())
wandb_key = os.getenv("WANDB_API_KEY")

device = get_device()

# os.environ["CUDA_HOME"] = "/usr/local/cuda-12.0"
CURDIR = Path().home() / "SPAICE/SEGMENT/segment"

# CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
CONFIG_DIR = CURDIR.parent / "configs"

# TRAIN_CONFIG_PATH = os.path.join(CONFIG_DIR, "train_config.yml")
TRAIN_CONFIG_PATH = CONFIG_DIR / "train_config.yml"

# YOLO_CONFIG_PATH = os.path.join(CONFIG_DIR, "fashion_people_detection_no_person.yml")
YOLO_CONFIG_PATH = CONFIG_DIR / "fashion_people_detection_no_person.yml"
assert YOLO_CONFIG_PATH.exists(), f"{YOLO_CONFIG_PATH} does not exist"

PRETRAINED_MODEL_PATH = "yolov8n-seg.pt"

TRAIN_DATA_DIR = CURDIR.parent / "train_data"

ULTRALYTICS_DIR = TRAIN_DATA_DIR / "ultralytics"

# warnings.filterwarnings(action="ignore", category=UserWarning)


def get_train_config(sweep=False):
    with TRAIN_CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


def main():
    # Initialize WandB
    project = "human_parsing_Sep_2024"
    project_dir = ULTRALYTICS_DIR / project
    
    # Train Settings
    train_config = get_train_config()

    wandb.init(dir=TRAIN_DATA_DIR, project=project)

    # Load a model
    model = YOLO(
        PRETRAINED_MODEL_PATH, task="segment"
    )  # Load a pretrained model (recommended for training)

    # # Add WandB callback
    add_wandb_callback(model)

    # Train the model
    model.train(project=project_dir, 
        data=str(YOLO_CONFIG_PATH), **train_config
    )

    # Finish the W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
