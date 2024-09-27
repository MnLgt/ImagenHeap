import os
from typing import List, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from utils import resize_image_pil
from dataclasses import dataclass
from pathlib import Path
import sys
from PIL import Image

sys.path.append("..")
from utilities.convert.coco_to_yolo.coco_to_yolo import coco_to_yolo
from utilities.logger_config import get_logger

logger = get_logger()


# Make the train and val directories for images and labels
@dataclass
class YoloDirs:
    """
    A dataclass representing the directory structure for YOLO dataset.

    Attributes:
        train_img (Path): Path to the training images directory.
        train_lbl (Path): Path to the training labels directory.
        val_img (Path): Path to the validation images directory.
        val_lbl (Path): Path to the validation labels directory.
    """

    train_img: Path
    train_lbl: Path
    val_img: Path
    val_lbl: Path


def make_yolo_dirs(dataset: Path) -> YoloDirs:
    """
    Create the YOLO dataset directory structure and return a YoloDirs instance.

    This function creates the necessary directories for a YOLO dataset,
    including separate directories for training and validation images and labels.

    Args:
        dataset (Path): The root directory for the YOLO dataset.

    Returns:
        YoloDirs: An instance of YoloDirs containing the paths to all created directories.

    Raises:
        OSError: If there's an error creating the directories.
    """
    img_dir = dataset / "images"
    lbl_dir = dataset / "labels"

    dirs = {
        "train_img": img_dir / "train",
        "train_lbl": lbl_dir / "train",
        "val_img": img_dir / "val",
        "val_lbl": lbl_dir / "val",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return YoloDirs(**dirs)


def get_lines(md, image_width, image_height):
    lines = []
    for row in md:
        label_id = row.get("label_id")

        # Get the coco polygons
        coco_polygons = row.get("polygons")

        # Convert the coco polygons to yolo format
        yolo_polygons = coco_to_yolo(coco_polygons, image_width, image_height)

        # Convert the yolo polygons to a string
        yolo_polygons_str = " ".join([str(coord) for coord in yolo_polygons])

        # Create the yolo line
        yolo_line = f"{label_id} {yolo_polygons_str}"

        lines.append(yolo_line)

    return lines


def write_image_and_text_file(
    image: Image.Image,
    image_name: str,
    lines: List[float],
    image_dir: Path,
    label_dir: Path,
):
    # Save images as jpegs
    image_uuid = Path(image_name).stem  # Get the image UUID

    # Save images as jpegs
    image_name = image_uuid + ".jpg"
    image_path = image_dir / image_name

    # Save labels as text files
    text_name = image_uuid + ".txt"
    text_path = label_dir / text_name

    image.save(image_path)

    with open(text_path, "w") as f:
        f.write("\n".join(lines))


def format_and_write(row, image_dir: Path, label_dir: Path, metadata_col: str = 'metadata'):
    try:
        image = row.get("image")
        image = resize_image_pil(image)

        md = row.get(metadata_col)
        if md:
            image_name = row["image_id"]
            lines = get_lines(md, image.width, image.height)
            write_image_and_text_file(image, image_name, lines, image_dir, label_dir)
    except Exception as e:
        logger.info(f"Error processing {row.get('image_id', 'Unknown ID')}: {str(e)}")


def check_directory_contents(image_dir: Path, label_dir: Path):
    # Create sets of all image and label filenames
    image_files: Set[Path] = set(
        [file.relative_to(image_dir) for file in image_dir.glob("*.jpg")]
    )
    label_files: Set[Path] = set(
        [file.relative_to(label_dir) for file in label_dir.glob("*.txt")]
    )

    # Convert image filenames to the expected label filenames
    expected_label_files: Set[Path] = {
        image.with_suffix(".txt") for image in image_files
    }

    # Find mismatches between expected and actual label files
    missing_labels: Set[Path] = expected_label_files - label_files
    extra_labels: Set[Path] = label_files - expected_label_files

    # Report findings
    if not missing_labels and not extra_labels:
        logger.info(
            f"All checks passed: Every image in '{image_dir.name}' has a corresponding label in '{label_dir.name}'."
        )
    else:
        if missing_labels:
            logger.warning(
                f"Missing label files in '{label_dir.name}': {', '.join(str(label) for label in missing_labels)}"
            )
        if extra_labels:
            logger.warning(
                f"Extra label files in '{label_dir.name}' not corresponding to images: {', '.join(str(label) for label in extra_labels)}"
            )


def check_all_directories(yolo_dirs: YoloDirs):
    assert any(yolo_dirs.train_img.iterdir()), "Image Training directory is empty."
    assert any(yolo_dirs.train_lbl.iterdir()), "Label Training directory is empty."
    
    assert any(yolo_dirs.val_img.iterdir()), "Image Validation directory is empty."
    assert any(yolo_dirs.val_lbl.iterdir()), "Label Validation directory is empty."
    
    
    logger.info("Checking training data...")
    check_directory_contents(yolo_dirs.train_img, yolo_dirs.train_lbl)

    logger.info("Checking validation data...")
    check_directory_contents(yolo_dirs.val_img, yolo_dirs.val_lbl)


def load_and_split_dataset(repo_id: str, workers: int, cache_dir: Path) -> tuple:
    try:
        ds = load_dataset(
            repo_id,
            split="train",
            trust_remote_code=True,
            num_proc=workers,
            cache_dir=str(cache_dir),
        )
        ds = ds.train_test_split(train_size=0.9)
        return ds["train"], ds["test"]
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}", exc_info=True)
        raise


def process_dataset(executor, dataset, img_dir, lbl_dir, dataset_type):
    futures = []
    for row in tqdm(dataset, desc=f"Processing {dataset_type} Data"):
        futures.append(executor.submit(format_and_write, row, img_dir, lbl_dir))

    # Wait for all futures to complete
    for future in as_completed(futures):
        try:
            future.result()  # This will raise the exception if one occurred
        except Exception as e:
            logger.error(
                f"Error in processing a {dataset_type.lower()} item: {str(e)}",
                exc_info=True,
            )


def process_dataset(
    executor: ProcessPoolExecutor,
    dataset,
    img_dir: Path,
    lbl_dir: Path,
    dataset_type: str,
) -> List[Exception]:
    futures = []
    errors = []
    for row in tqdm(dataset, desc=f"Processing {dataset_type} Data"):
        futures.append(executor.submit(format_and_write, row, img_dir, lbl_dir))

    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            logger.error(
                f"Error processing {dataset_type} item: {str(e)}", exc_info=True
            )
            errors.append(e)

    return errors


def main():
    repo_id = "MnLgt/fashion_people_detections_v2"
    parent = Path(__file__).parent.parent
    dataset_dir = parent / "datasets/fashion_people_detection"
    workers = os.cpu_count()
    cache_dir = Path(__file__).parent.parent / "hf_cache"
    try:
        # Load and split dataset
        train, val = load_and_split_dataset(repo_id, workers, cache_dir)

        # Make directories
        yolo_dirs = make_yolo_dirs(dataset_dir)
        if not isinstance(yolo_dirs, YoloDirs):
            raise TypeError(f"Expected YOLODirs, but got {type(yolo_dirs)}")

        # Process datasets
        with ProcessPoolExecutor(max_workers=workers) as executor:
            train_errors = process_dataset(
                executor, train, yolo_dirs.train_img, yolo_dirs.train_lbl, "Training"
            )
            val_errors = process_dataset(
                executor, val, yolo_dirs.val_img, yolo_dirs.val_lbl, "Validation"
            )

        # Log summary of errors
        if train_errors or val_errors:
            logger.warning(
                f"Encountered {len(train_errors)} errors in training data and {len(val_errors)} errors in validation data."
            )
        else:
            logger.info("All items processed successfully.")

        # Check directories
        try:
            check_all_directories(yolo_dirs)
        except Exception as e:
            logger.error(f"Error checking directories: {str(e)}", exc_info=True)
            raise

    except Exception as e:
        logger.critical(f"Critical error in main function: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        raise
