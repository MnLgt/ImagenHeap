import os
import warnings

import wandb
import yaml
from dotenv import find_dotenv, load_dotenv
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback

# os.environ["CUDA_HOME"] = "/usr/local/cuda-12.0"

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")

TRAIN_CONFIG_PATH = os.path.join(CONFIG_DIR, "train_config.yml")

YOLO_CONFIG_PATH = os.path.join(CONFIG_DIR, "fashion_people_detection.yml")

PRETRAINED_MODEL_PATH = (
    "/home/ubuntu/SPAICE/SEGMENT/models/yolo-human-parse/yolo-human-parse-epoch-125.pt"
)
PRETRAINED_MODEL_PATH = "/home/ubuntu/SPAICE/SEGMENT/weights/yolov8x-seg.pt"

load_dotenv(find_dotenv())

wandb_key = os.getenv("WANDB_API_KEY")

warnings.filterwarnings(action="ignore", category=UserWarning)


def get_train_config(sweep=False):
    with open(TRAIN_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def main():
    # Initialize WandB
    project = "human_parsing_from_scratch"

    # Train Settings
    train_config = get_train_config()

    wandb.init(dir='training',project=project)

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
    results = model.train(project=project, data=YOLO_CONFIG_PATH, **train_config)

    # Finish the W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
