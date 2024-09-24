import sys

sys.path.append("..")

import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)

import os
from functools import lru_cache
from typing import List, Tuple, Callable

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
import torch
import torchvision
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)
from PIL import Image

from segment.sam_results import format_boxes, format_scores
from segment.utils import get_device, image_handler, load_resize_image

DEVICE = get_device()

CURDIR = os.path.dirname(__file__)

WEIGHTS_DIR = os.path.join(CURDIR, "..", "weights")

# Dino
DINO_DIR = os.path.join(CURDIR, "..", "GroundingDINO")
DINO_CHECKPOINT = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
DINO_CONFIG = os.path.join(DINO_DIR, "groundingdino/config/GroundingDINO_SwinT_OGC.py")


def caption_handler(caption: str, images: torch.Tensor) -> List[str]:
    """
    Process a single caption for multiple images represented as a tensor.

    Args:
    caption (str): The caption to be processed.
    images (torch.Tensor): A tensor of images.

    Returns:
    List[str]: A list of processed captions, one for each image in the tensor.

    Raises:
    ValueError: If the caption is empty or images tensor is empty.
    TypeError: If images is not a torch.Tensor.
    """
    if not caption:
        raise ValueError("Caption cannot be empty.")
    if not isinstance(images, torch.Tensor):
        raise TypeError("Images must be a torch.Tensor.")
    if images.numel() == 0:
        raise ValueError("Images tensor cannot be empty.")

    processed_caption = caption.lower().strip()
    if not processed_caption.endswith("."):
        processed_caption += "."

    num_images = images.size(0)  # Assuming the first dimension is the number of images
    return [processed_caption] * num_images


def load_dino_model(
    config_path: str, checkpoint_path: str, device: torch.device
) -> torch.nn.Module:
    """
    Load and initialize the grounding model.

    Args:
        config_path (str): Path to the model configuration file.
        checkpoint_path (str): Path to the model checkpoint file.
        device (torch.device): Device to load the model onto.

    Returns:
        torch.nn.Module: The loaded and initialized model.
    """
    args = SLConfig.fromfile(config_path)
    args.device = device

    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    load_result = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(f"Model loading result: {load_result}")

    model.eval()
    return model.to(device)


@lru_cache(maxsize=1)
def get_dino_model() -> torch.nn.Module:
    """
    Get the GroundedDINO model, using caching to avoid reloading.

    Returns:
        torch.nn.Module: The loaded grounding model.
    """
    return load_dino_model(DINO_CONFIG, DINO_CHECKPOINT, DEVICE)


def transform_image_dino(image_pil: Image.Image) -> torch.tensor:

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image


def run_dino(
    dino_model: Callable,
    images: torch.tensor,
    caption: str,
    box_threshold: float = 0.3,
    text_threshold: int = 0.25,
    **kwargs,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:

    # Process caption
    processed_captions = caption_handler(caption, images)

    # Ensure images is a batch (first dimension is batch size)
    images = images.unsqueeze(0) if len(images.shape) == 3 else images

    with torch.no_grad():
        outputs = dino_model(images, captions=processed_captions)

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()
    prediction_boxes = outputs["pred_boxes"].cpu()

    mask = prediction_logits.max(dim=2)[0] > box_threshold

    tokenizer = dino_model.tokenizer
    tokenized = tokenizer(caption)

    # Vectorized operations
    batch_logits = [logits[m] for logits, m in zip(prediction_logits, mask)]
    batch_boxes = [boxes[m] for boxes, m in zip(prediction_boxes, mask)]

    # Compute max logits once
    max_logits = [logits.max(dim=1) for logits in batch_logits]

    filtered_batch_boxes = []
    filtered_predicts_batch = []
    filtered_phrases_batch = []

    for batch_idx, (batch_logit, max_logit, boxes) in enumerate(
        zip(batch_logits, max_logits, batch_boxes)
    ):
        logit_mask = max_logit.values > text_threshold

        filtered_batch_boxes.append(boxes[logit_mask])
        filtered_predicts_batch.append(max_logit.values[logit_mask])

        filtered_phrases = [
            get_phrases_from_posmap(logit == max_val, tokenized, tokenizer).replace(
                ".", ""
            )
            for logit, max_val in zip(
                batch_logit[logit_mask], max_logit.values[logit_mask]
            )
        ]
        filtered_phrases_batch.append(filtered_phrases)

    return filtered_batch_boxes, filtered_predicts_batch, filtered_phrases_batch


def format_dino(
    batch_boxes_filt,
    batch_scores,
    batch_pred_phrases,
    image_size=(1024, 1024),
    iou_threshold=0.8,
):
    batch_final_boxes = []
    batch_final_scores = []
    batch_final_phrases = []

    # Process each item in the batch
    for boxes_filt, scores, pred_phrases in zip(
        batch_boxes_filt, batch_scores, batch_pred_phrases
    ):
        # Adjust box coordinates based on image size
        H, W = image_size[1], image_size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        # Non-maximum suppression (NMS)
        nms_idx = (
            torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
        )
        final_boxes = boxes_filt[nms_idx]
        final_scores = scores[nms_idx]
        final_phrases = [pred_phrases[idx] for idx in nms_idx]

        # Collect results for this batch element
        batch_final_boxes.append(final_boxes)
        batch_final_scores.append(final_scores)
        batch_final_phrases.append(final_phrases)

    return batch_final_boxes, batch_final_scores, batch_final_phrases


class DinoResults:
    def __init__(self, boxes, scores, phrases):
        self.boxes = boxes
        self.scores = scores
        self.phrases = phrases

    def asdict(self):
        boxes = format_boxes(self.boxes[0])
        scores = format_scores(self.scores[0])
        phrases = self.phrases[0]
        return [
            dict(box=box, score=score, label=phrases[idx])
            for idx, (box, score) in enumerate(zip(boxes, scores))
        ]


def get_dino_results(
    images: str | Image.Image | List[Image.Image],
    text_prompt: str,
    device: torch.device = DEVICE,
    box_threshold=0.3,
    text_threshold=0.25,
    iou_threshold=0.8,
    **kwargs,
) -> DinoResults:
    """
    Run the DINO model on an image and text prompt.

    Args:
    image (Image.Image): A list of images to process.
    text_prompt (str): The text prompt to use.
    device (torch.device): The device to run the model on.
    box_threshold (float): The box threshold.
    text_threshold (float): The text threshold.
    iou_threshold (float): The IoU threshold.

    Returns:
    tuple: A tuple containing the boxes, scores, and phrases.
    """
    images = image_handler(images)
    # Load DINO model
    dino_model = get_dino_model()

    with torch.no_grad():
        dino_images = torch.stack([transform_image_dino(image) for image in images])
        dino_images = dino_images.to(device)

    # Process with DINO model
    boxes, scores, phrases = run_dino(
        dino_model,
        dino_images,
        text_prompt,
        box_threshold,
        text_threshold=text_threshold,
    )
    boxes, scores, phrases = format_dino(
        boxes, scores, phrases, iou_threshold=iou_threshold
    )
    results = DinoResults(boxes, scores, phrases)
    return results


if __name__ == "__main__":
    from pathlib import Path

    from PIL import Image

    from segment.utils import resize_image_pil

    def load_image(image_path):
        image_pil = Image.open(image_path).convert("RGB")
        image_pil = resize_image_pil(image_pil, 1024)
        return image_pil

    image_dir = Path("../datasets/fashion_people_detection/images/val")
    image_paths = list(image_dir.glob("*.jpg"))

    images = [load_image(path) for path in image_paths[:20]]

    results = get_dino_results(images, "legs", DEVICE)
    print(f"Boxes: {results.boxes}")
    print(f"Scores: {results.scores}")
    print(f"Phrases: {results.phrases}")
