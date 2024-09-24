# %%
import sys

sys.path.append("..")

import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)

import os
from typing import List, Dict, Any, Callable
from pathlib import Path
from PIL import Image
from segment.utils import resize_image_pil
from segment.dino_script import DinoDetector
from segment.sam_script import get_sam_results
from segment.utils import get_device
from segment.sam_results import SAMResults
from diffusers.utils import load_image
from segment.sam_script import get_sam_results
from segment.utils import get_coco_style_polygons
import yaml
from datasets import load_dataset, Dataset
from typing import List, Dict
from datasets import Dataset, concatenate_datasets
from typing import List, Dict
from PIL import Image
import math
from tqdm.auto import tqdm
from segment.utils import *
import pandas as pd
import random


# disable datasets.map progress bar
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ******* NOTE ******
# The initial create_dataset notebook used a pytorch dataloader which was much faster but more complex to implement.
# When we have time, we should update this to use a dataloader


# Function for resizing an image to a specific size without changing the aspect ratio
def load_resize_image(image_path: str | Image.Image, size: int) -> Image.Image:
    if isinstance(image_path, str):
        image_pil = load_image(image_path).convert("RGB")
    else:
        image_pil = image_path.convert("RGB")

    image_pil = resize_image_pil(image_pil, size)
    return image_pil


# Convert the results to SamResults formatted results
def load_all_in_sam(images, unformatted_results, labels_dict, **kwargs):
    results = []
    for image, uf in zip(images, unformatted_results):
        try:
            result = SAMResults(
                image, labels_dict, **uf, person_masks_only=False, **kwargs
            ).formatted_results
        except Exception as e:
            print(f"One Image had an error: {e}")
            result = []

        results.append(result)
    return results


def get_metadata(
    text_prompt,
    image_paths: List[str | Image.Image],
    labels_dict: Dict[str, int],
    size: int = 1024,
    **kwargs,
) -> List[Dict]:
    device = get_device()

    assert bool(image_paths), "No images provided"

    # Load and resize the images
    images = [load_resize_image(im, size) for im in image_paths]

    # Get the boxes from the prompts using DINO
    dino_results = DinoDetector(images, text_prompt, **kwargs)
    dino_results.run()
    
    # Get the masks from the images and boxes using SAM
    unformatted_results = get_sam_results(images, dino_results, text_prompt)

    return load_all_in_sam(images, unformatted_results, labels_dict, **kwargs)


# loading yaml config file
def load_yaml(path):
    with open(path, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data


# Get the labels dictionary from the config file
def get_labels_dict(config_path):
    data = load_yaml(config_path)
    labels_dict = data.get("names")
    labels_dict = {v: k for k, v in labels_dict.items()}
    return labels_dict


def filter_list_in_column(dataset: Dataset, column_name: str, filter_function):
    """
    Apply a filter function to a column containing lists of dictionaries.

    :param dataset: The input dataset
    :param column_name: The name of the column containing lists to filter
    :param filter_function: A function that takes a dictionary and returns True to keep it, False to filter it out
    :return: A new dataset with filtered lists in the specified column
    """

    def transform_fn(example):
        filtered_list = [item for item in example[column_name] if filter_function(item)]
        return {column_name: filtered_list}

    return dataset.map(transform_fn)


def score_filter(item, score_cutoff=0.5):
    return item.get("score", 0) > score_cutoff


def remove_masks(batch_results):
    for b in batch_results:
        for row in b:
            row.pop("mask", None)
    return batch_results


def process_batch(
    batch: Dict[str, Any],
    text_prompt: str,
    labels_dict: Dict[str, Any],
    get_metadata_func: Callable,
    batch_size: int = 32,
    **kwargs
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process a batch of images and retrieve metadata.

    Args:
        batch (Dict[str, Any]): The input batch containing image paths or PIL Images.
        text_prompt (str): The text prompt to use for metadata retrieval.
        labels_dict (Dict[str, Any]): Dictionary of labels for metadata retrieval.
        get_metadata_func (Callable): Function to retrieve metadata for a batch of images.
        batch_size (int, optional): Size of sub-batches for processing. Defaults to 32.
        **kwargs: Additional keyword arguments to pass to get_metadata_func.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary containing the 'metadata' key with a list of metadata results.
    """
    image_paths = batch["image"]
    results = []

    with tqdm(total=len(image_paths), desc="Processing Images", unit="img") as pbar:
        for i in range(0, len(image_paths), batch_size):
            sub_batch = image_paths[i:i + batch_size]
            pbar.set_description(f"Sub Batch: {i // batch_size + 1}")

            batch_results = get_metadata_func(
                text_prompt=text_prompt,
                image_paths=sub_batch,
                labels_dict=labels_dict,
                **kwargs
            )

            if batch_results:
                remove_masks(batch_results)
                results.extend(batch_results)

            pbar.update(len(sub_batch))

    return {"metadata": results}


# process all batches
def process_dataset(
    dataset,
    text_prompt,
    labels_dict,
    get_metadata_func,
    batch_size=1000,
    sub_batch_size=32,
    **kwargs,
):
    # Calculate the number of batches
    num_batches = math.ceil(len(dataset) / batch_size)

    # Process the dataset in batches
    pbar = tqdm(range(num_batches), position=0, leave=True)

    batches = []
    for i in pbar:
        pbar.set_description(f"Processing Batch: {i}")
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(dataset))

        # Process this batch
        batch_results = dataset.select(range(start_idx, end_idx)).map(
            lambda batch: process_batch(
                batch, text_prompt, labels_dict, get_metadata_func, sub_batch_size, **kwargs
            ),
            batched=True,
            batch_size=batch_size,
        )

        batches.append(batch_results)

    # Update the dataset with the new results
    return concatenate_datasets(batches)


def sanity_check(
    ds, row=102, mask_row=2, image_col="image", metadata_col="mask_metadata"
):
    image = ds[row][image_col]
    image = resize_image_pil(image)

    polygons = ds[row][metadata_col][mask_row]["polygons"]
    label = ds[row][metadata_col][mask_row]["label"]
    score = ds[row][metadata_col][mask_row]["score"]
    
    try:
        image_url = ds[row]["image_url"]
        print(f"Image URL: {image_url}")
    except KeyError:
        pass

    mask = convert_coco_polygons_to_mask(polygons, 1024, 1024)
    mask_image = Image.fromarray(mask)
    overlay = overlay_mask(image, mask_image, opacity=0.8)

    print(f"Label: {label}")
    print(f"Score: {score}")
    
    display(overlay.resize((512, 512)))


class CreateSegmentationDataset:
    """
    A class for creating and managing segmentation datasets based on text prompts.

    This class processes a given dataset to extract segmentation masks, bounding boxes,
    and other metadata based on text prompts derived from a configuration file. It supports
    efficient batch processing and provides methods for filtering results and performing
    sanity checks.

    Attributes:
        ds (Dataset): The input dataset to be processed.
        metadata_col (str): The name of the column where metadata will be stored.
        config_path (str): Path to the configuration file (YAML format).
        config (dict): Loaded configuration data.
        labels_dict (dict): Dictionary mapping label names to their corresponding indices.
        text_prompt (str): Text prompt generated from the labels for segmentation.
        bs (int): Batch size for processing.
        sub_bs (int): Sub-batch size for processing within each batch.
        processed_ds (Dataset): The processed dataset with added segmentation metadata.

    Args:
        ds (Dataset): The input dataset to be processed.
        config_path (str): Path to the configuration file (YAML format).
        metadata_col (str, optional): Name of the column to store metadata. Defaults to 'metadata'.
        bs (int, optional): Batch size for processing. Defaults to 100.
        sub_bs (int, optional): Sub-batch size for processing. Defaults to 8.
    """

    def __init__(
        self,
        ds: Dataset,
        config_path: str,
        metadata_col: str = "metadata",
        bs: int = 64,
        sub_bs: int = 8,
        box_threshold: int = 0.3,
        text_threshold: int = 0.25,
        iou_threshold: int = 0.8,
    ):
        self.ds = ds
        self.metadata_col = metadata_col
        self.config_path = config_path
        self.config = self._load_yaml(config_path)
        self.labels_dict = self._get_labels_dict(config_path)
        self.text_prompt = self._get_text_prompt()
        self.bs = bs
        self.sub_bs = sub_bs
        self.processed_ds = None
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.iou_threshold = iou_threshold
        

    def _load_yaml(self, path: str) -> Dict:
        """Load YAML configuration file."""
        with open(path, "r") as file:
            return yaml.safe_load(file)

    def _get_labels_dict(self, config_path: str) -> Dict[str, int]:
        """Get labels dictionary from the config file."""
        data = self._load_yaml(config_path)
        labels_dict = data.get("names", {})
        return {v: k for k, v in labels_dict.items()}

    def _get_text_prompt(self) -> str:
        """Generate text prompt from labels."""
        labels = list(self.labels_dict.keys())
        return " . ".join(labels)

    def process(self):
        """
        Process the dataset to extract segmentation metadata.

        This method applies the segmentation model to the dataset in batches,
        adding the resulting metadata to a new copy of the dataset.
        """
        self.processed_ds = process_dataset(
            self.ds,
            self.text_prompt,
            self.labels_dict,
            get_metadata,  # This function should be imported or defined elsewhere
            batch_size=self.bs,
            sub_batch_size=self.sub_bs,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            iou_threshold=self.iou_threshold,
        )

    def filter_scores(self, score_cutoff: float):
        """
        Filter the processed dataset based on segmentation scores.

        Args:
            score_cutoff (float): The minimum score to keep a segmentation result.

        Raises:
            AssertionError: If there is no processed dataset to filter.
        """
        assert self.processed_ds is not None, "There is no processed dataset to filter"

        self.processed_ds = filter_list_in_column(
            self.processed_ds,
            self.metadata_col,
            lambda x: x.get("score", 0) > score_cutoff,
        )

    def check_results(self, row: int = None, mask_row: int = None):
        """
        Perform a sanity check by displaying a segmentation mask overlay.

        Args:
            row (int, optional): The dataset row to check. If None, a random row is selected.
            mask_row (int, optional): The mask index to check within the selected row.
                                      If None, a random mask is selected.

        Raises:
            AssertionError: If there is no processed dataset to check.
        """
        assert self.processed_ds is not None, "There is no processed dataset to check"

        if row is None:
            row = random.randint(0, len(self.processed_ds) - 1)

        if mask_row is None:
            mask_row = random.randint(
                0, len(self.processed_ds[row][self.metadata_col]) - 1
            )

        sanity_check(
            self.processed_ds,
            row,
            mask_row,
            image_col="image",
            metadata_col=self.metadata_col,
        )
