# %%
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
    project = "human_parsing"

    wandb.init(project=project)

    # Load a model
    pretrained = "weights/yolov8n-seg.pt"
    model = YOLO(
        pretrained, task="segment"
    )  # Load a pretrained model (recommended for training)

    # Load labels from YAML configuration file
    yaml_file = "configs/fashion_people_detection.yml"
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
        class_labels = config["names"]  # Adjust the key based on your YAML structure

    # Add WandB callback
    add_wandb_callback(model)

    # Set the batch size and number of workers
    bs = 80
    workers = os.cpu_count()

    # Train the model
    results = model.train(
        project=project,
        data=yaml_file,
        epochs=1,
        imgsz=1024,
        batch=bs,
        workers=8,
        augment=True,
        device=[0],
        patience=20,
        save_period=1,
        cache=True,
    )

    # Finish the W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
