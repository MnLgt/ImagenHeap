# %%
import sys

sys.path.append("..")

import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)

import math
import os
from typing import Any, Callable, Dict, List, Union

from datasets import Dataset, concatenate_datasets
from datasets.utils.logging import disable_progress_bar
from PIL import Image
from tqdm.auto import tqdm

from segment.components.detect.DetectDino import DetectDino
from segment.components.segment.SegmentSam import SegmentSam
from segment.format_results import format_all_results
from segment.utils import *
from segment.utils import get_device

# disable datasets.map progress bar
disable_progress_bar()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sam = SegmentSam()
dino = DetectDino()

def remove_masks(batch_results):
    for b in batch_results:
        for row in b:
            row.pop("mask", None)
    return batch_results

def get_metadata(
    text_prompt: Union[str, List[str]],
    image_paths: List[str | Image.Image],
    image_size: int = 1024,
    **kwargs,
) -> List[Dict]:
    device = get_device()

    assert bool(image_paths), "No images provided"

    # Load and resize the images
    images = [load_resize_image(im, image_size) for im in image_paths]

    # Get the boxes from the prompts using DINO
    dino.process(images, text_prompt, **kwargs)

    # Get the masks from the images and boxes using SAM
    boxes = dino.boxes
    phrases = dino.phrases
    
    sam_results = sam.process(images, boxes, phrases)
    
    dino.reset()

    return format_all_results(sam_results)


def process_batch(
    batch: Dict[str, Any],
    text_prompt: str,
    get_metadata_func: Callable,
    batch_size: int = 32,
    **kwargs,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process a batch of images and retrieve metadata.

    Args:
        batch (Dict[str, Any]): The input batch containing image paths or PIL Images.
        text_prompt (str): The text prompt to use for metadata retrieval.
        get_metadata_func (Callable): Function to retrieve metadata for a batch of images.
        batch_size (int, optional): Size of sub-batches for processing. Defaults to 32.
        **kwargs: Additional keyword arguments to pass to get_metadata_func.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary containing the 'metadata' key with a list of metadata results.
    """
    image_paths = batch["image"]
    results = []

    with tqdm(
        total=len(image_paths),
        desc="Processing Images",
        unit="img",
        leave=False,
        position=1,
    ) as pbar:
        for i in range(0, len(image_paths), batch_size):
            sub_batch = image_paths[i : i + batch_size]
            pbar.set_description(f"Sub Batch: {i // batch_size + 1}")

            batch_results = get_metadata_func(
                text_prompt=text_prompt,
                image_paths=sub_batch,
                **kwargs,
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
    get_metadata_func,
    batch_size=1000,
    sub_batch_size=32,
    **kwargs,
):
    # Calculate the number of batches
    num_batches = math.ceil(len(dataset) / batch_size)


    # Process the dataset in batches
    batches = []
    with tqdm(
        total=num_batches,
        desc="Processing Batches",
        unit="batch",
        position=0,
        leave=False,
    ) as pbar:
        for i in range(num_batches):
            pbar.set_description(f"Processing Batch: {i}")
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(dataset))

            # Process this batch
            batch_results = dataset.select(range(start_idx, end_idx)).map(
                lambda batch: process_batch(
                    batch,
                    text_prompt,
                    get_metadata_func,
                    sub_batch_size,
                    **kwargs,
                ),
                batched=True,
                batch_size=batch_size,
            )

            batches.append(batch_results)

            pbar.update(1)

    # Update the dataset with the new results
    return concatenate_datasets(batches)


class CreateSegmentationDataset:
    """
    A class for creating and managing segmentation datasets based on text prompts.

    This class processes a given dataset to extract segmentation masks, bounding boxes,
    and other metadata based on text prompts. It supports efficient batch processing
    and provides methods for filtering results and performing sanity checks.

    Attributes:
        ds (Dataset): The input dataset to be processed.
        metadata_col (str): The name of the column where metadata will be stored.
        text_prompt (Union[str, List[str]]): Text prompt(s) for segmentation.
        bs (int): Batch size for processing.
        sub_bs (int): Sub-batch size for processing within each batch.
        ds (Dataset): The processed dataset with added segmentation metadata.

    Args:
        ds (Dataset): The input dataset to be processed.
        text_prompt (Union[str, List[str]]): Text prompt(s) for segmentation.
        metadata_col (str, optional): Name of the column to store metadata. Defaults to 'metadata'.
        bs (int, optional): Batch size for processing. Defaults to 64.
        sub_bs (int, optional): Sub-batch size for processing. Defaults to 8.
        box_threshold (float, optional): Threshold for box detection. Defaults to 0.3.
        text_threshold (float, optional): Threshold for text detection. Defaults to 0.25.
        iou_threshold (float, optional): Threshold for IOU. Defaults to 0.8.
    """

    def __init__(
        self,
        ds: Dataset,
        text_prompt: Union[str, List[str]],
        metadata_col: str = "metadata",
        bs: int = 64,
        sub_bs: int = 8,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        iou_threshold: float = 0.8,
        image_size=1024,
        **kwargs,
    ):
        self.ds = ds
        self.metadata_col = metadata_col
        self.text_prompt = self._process_text_prompt(text_prompt)
        self.bs = bs
        self.sub_bs = sub_bs
        self.ds = None
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.iou_threshold = iou_threshold
        self.kwargs = kwargs

    def _process_text_prompt(self, text_prompt: Union[str, List[str]]) -> str:
        """Process the text prompt."""
        if isinstance(text_prompt, str):
            return text_prompt
        elif isinstance(text_prompt, list):
            return ".".join(text_prompt)
        else:
            raise ValueError("text_prompt must be a string or a list of strings")


    def process(self):
        """
        Process the dataset to extract segmentation metadata.

        This method applies the segmentation model to the dataset in batches,
        adding the resulting metadata to a new copy of the dataset.
        """
        self.ds = process_dataset(
            self.ds,
            self.text_prompt,
            get_metadata,  # This function should be imported or defined elsewhere
            batch_size=self.bs,
            sub_batch_size=self.sub_bs,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            iou_threshold=self.iou_threshold,
            **self.kwargs,
        )

