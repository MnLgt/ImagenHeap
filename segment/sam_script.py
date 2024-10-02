import sys

sys.path.append("..")

import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)

import os
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

# segment anything
from segment_anything import build_sam
from segment_anything.utils.transforms import ResizeLongestSide

from segment.dino_script import DinoDetector
from segment.utils import get_device, load_resize_image

DEVICE = get_device()

CURDIR = os.path.dirname(__file__)

WEIGHTS_DIR = os.path.join(CURDIR, "..", "weights")

# SAM
SAM_CHECKPOINT = os.path.join(WEIGHTS_DIR, "sam_vit_h_4b8939.pth")
ckpt_repo_id = "ShilongLiu/GroundingDINO"


# determine if tensor is empty
def is_empty(tensor):
    return tensor.size(0) == 0


@lru_cache(maxsize=1)
def get_sam_model():
    # SAM PREDICTOR
    sam = build_sam(checkpoint=SAM_CHECKPOINT)
    return sam.to(device=DEVICE)


def sam_transform_image(image, device="cpu"):
    if isinstance(image, Image.Image):
        image = np.array(image)

    transform = ResizeLongestSide(1024)
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device)
    return image.permute(2, 0, 1).contiguous()


def sam_prepare_boxes(
    boxes, transform, image_size: Tuple[int, int], device: str = DEVICE
):
    return transform.apply_boxes_torch(boxes, image_size).to(device)


def sam_prepare_row(image, boxes, image_size: Tuple[int, int], device: str = DEVICE):
    return dict(image=image, boxes=boxes, original_size=image_size)


def sam_detect(sam_model, images, boxes, image_sizes, multimask_output=False):
    transform = ResizeLongestSide(1024)
    sam_boxes = [
        sam_prepare_boxes(box, transform, image_size)
        for box, image_size in zip(boxes, image_sizes)
    ]
    batched_input = [
        sam_prepare_row(image, box, image_size)
        for image, box, image_size in zip(
            images,
            sam_boxes,
            image_sizes,
        )
    ]
    batched_output = sam_model(batched_input, multimask_output=multimask_output)
    return batched_output


def sam_process(
    boxes, phrases, sam_model, sam_images, image_sizes, multimask_output=False, **kwargs
):
    assert bool(boxes) == bool(phrases), "Boxes and phrases must be provided together"

    batch_size = sam_images.size(0)

    # Create a mask for valid (non-empty) boxes
    valid_mask = torch.tensor([not is_empty(box) for box in boxes], dtype=torch.bool)

    # Initialize output with None dictionaries for all entries
    output = [
        {"masks": None, "boxes": None, "scores": None, "phrases": None}
        for _ in range(batch_size)
    ]

    if valid_mask.any():
        # Process only valid entries
        valid_sam_images = sam_images[valid_mask]
        valid_boxes = [box for box, is_valid in zip(boxes, valid_mask) if is_valid]

        # Run SAM on valid images
        sam_outputs = sam_detect(
            sam_model,
            valid_sam_images,
            valid_boxes,
            image_sizes=image_sizes,
            multimask_output=multimask_output,
        )
        valid_masks = [output.get("masks") for output in sam_outputs]

        valid_scores = [output.get("iou_predictions") for output in sam_outputs]

        # Filter phrases manually
        valid_phrases = [
            phrase for phrase, is_valid in zip(phrases, valid_mask) if is_valid
        ]

        # Fill in the output for valid entries
        valid_indices = valid_mask.nonzero().squeeze(1)

        for i, (mask, box, score, phrase) in enumerate(
            zip(valid_masks, valid_boxes, valid_scores, valid_phrases)
        ):
            output[valid_indices[i].item()] = {
                "masks": mask,
                "boxes": box,
                "scores": score,
                "phrases": phrase,
            }

    return output


@torch.no_grad()
def get_sam_results(
    images: List[Image.Image],
    boxes: List[torch.Tensor],
    phrases: List[str],
    device: torch.device = DEVICE,
    max_image_side: int = 1024,
    **kwargs,
) -> List[dict]:
    sam_model = get_sam_model()

    # Get the longest side of the
    resized_images = [load_resize_image(image, max_image_side) for image in images]

    #
    # H, W format
    image_sizes = [(size[1], size[0]) for size in [image.size for image in images]]

    # dino_images = torch.stack([transform_image_dino(image) for image in images])
    sam_images = torch.stack(
        [sam_transform_image(image, device) for image in resized_images]
    )

    # dino_images = dino_images.to(DEVICE)
    sam_images = sam_images.to(DEVICE)

    unformatted_results = sam_process(
        boxes, phrases, sam_model, sam_images, image_sizes, **kwargs
    )

    return unformatted_results
