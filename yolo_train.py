# %%
import os

os.environ["CUDA_HOME"] = "/usr/local/cuda-12.0"
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import os
import wandb
import yaml
import warnings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
wandb_key = os.getenv("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = wandb_key

warnings.filterwarnings(action="ignore", category=UserWarning)


def main():
    # Initialize WandB
    project = "human_parsing_test"
    yaml_file = "configs/fashion_people_detection.yml"
    pretrained = "weights/yolov8x-seg.pt"

    # Training Settings
    epochs = 20
    imgsz = 640
    bs = 16
    workers = os.cpu_count()
    half = False

    wandb.init(project=project)

    # Load a model
    model = YOLO(
        pretrained, task="segment"
    )  # Load a pretrained model (recommended for training)
    # model = YOLO(model_file).load(pretrained)  # build from YAML and transfer weights

    # Load labels from YAML configuration file
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
        class_labels = config["names"]  # Adjust the key based on your YAML structure

    # Add WandB callback
    add_wandb_callback(model)

    # Train the model
    results = model.train(
        project=project,
        data=yaml_file,
        epochs=epochs,
        imgsz=imgsz,
        batch=bs,
        workers=workers,
        cache=True,
        half=half,
    )

    # Finish the W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
