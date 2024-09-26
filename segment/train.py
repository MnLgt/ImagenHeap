import os
import warnings

import wandb
import yaml
from dotenv import find_dotenv, load_dotenv
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
from utils import get_device
from pathlib import Path

device = get_device()

# os.environ["CUDA_HOME"] = "/usr/local/cuda-12.0"
CURDIR = Path().parent.absolute()

# CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
CONFIG_DIR = CURDIR / "../configs"

# TRAIN_CONFIG_PATH = os.path.join(CONFIG_DIR, "train_config.yml")
TRAIN_CONFIG_PATH = CONFIG_DIR / "train_config.yml"

# YOLO_CONFIG_PATH = os.path.join(CONFIG_DIR, "fashion_people_detection_no_person.yml")
YOLO_CONFIG_PATH = CONFIG_DIR / "fashion_people_detection_no_person.yml"

PRETRAINED_MODEL_PATH = "yolov8n-seg.pt"

TRAIN_DATA_DIR = CURDIR / "../train_data"

ULTRALYTICS_DIR = TRAIN_DATA_DIR / "ultralytics"

# WANDB_DIR = os.path.join(os.path.dirname(__file__), "..", "train_data", "wandb")
WANDB_DIR = TRAIN_DATA_DIR / "wandb"

load_dotenv(find_dotenv())

wandb_key = os.getenv("WANDB_API_KEY")

# warnings.filterwarnings(action="ignore", category=UserWarning)


def get_train_config(sweep=False):
    with open(TRAIN_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def main():
    # Initialize WandB
    project = "human_parsing_Sep_2024"
    project_dir = ULTRALYTICS_DIR / project
    # Train Settings
    train_config = get_train_config()

    wandb.init(dir=WANDB_DIR, project=project)

    # Load a model
    model = YOLO(
        PRETRAINED_MODEL_PATH, task="segment"
    )  # Load a pretrained model (recommended for training)

    # # Add WandB callback
    add_wandb_callback(model)

    # fraction=0.1,
    # cache=True,
    # save=False,
    # plots=False,
    # epochs=1,

    # Train the model
    model.train(
        project=project_dir, data=YOLO_CONFIG_PATH, device=device, **train_config
    )

    # Finish the W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
