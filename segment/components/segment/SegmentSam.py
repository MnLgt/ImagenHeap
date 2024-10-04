import os
import warnings
from functools import lru_cache
from typing import List, Tuple, Dict, Any, Union

import numpy as np
import torch
from PIL import Image
from segment_anything import build_sam
from segment_anything.utils.transforms import ResizeLongestSide
from segment.utils import get_device
from segment.utils import load_resize_image
from segment.components.base import Component


DEVICE = get_device()

CURDIR = os.getcwd()

WEIGHTS_DIR = os.path.join(CURDIR, "weights")

# SAM
SAM_CHECKPOINT = os.path.join(WEIGHTS_DIR, "sam_vit_h_4b8939.pth")
ckpt_repo_id = "ShilongLiu/GroundingDINO"


class SegmentSam(Component):

    def __init__(
        self,
        weights_dir: str = WEIGHTS_DIR,
        device: str = None,
    ):
        super().__init__("segment")
        self.weights_dir = weights_dir
        self.device = device or get_device()
        self.model = None
        self.transform = ResizeLongestSide(1024)

    @lru_cache(maxsize=1)
    def load_model(self):
        sam_checkpoint = os.path.join(self.weights_dir, "sam_vit_h_4b8939.pth")
        sam = build_sam(checkpoint=sam_checkpoint)
        self.model = sam.to(device=self.device)

    def transform_image(self, image: Image.Image) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = np.array(image)

        image = self.transform.apply_image(image)
        image = torch.as_tensor(image, device=self.device)
        return image.permute(2, 0, 1).contiguous()

    def prepare_boxes(
        self, boxes: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        return self.transform.apply_boxes_torch(boxes, image_size).to(self.device)

    def prepare_input(
        self, image: torch.Tensor, boxes: torch.Tensor, image_size: Tuple[int, int]
    ) -> Dict:
        return {"image": image, "boxes": boxes, "original_size": image_size}

    def detect(
        self,
        images: List[torch.Tensor],
        boxes: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
        multimask_output: bool = False,
    ) -> List[Dict]:
        sam_boxes = [
            self.prepare_boxes(box, image_size)
            for box, image_size in zip(boxes, image_sizes)
        ]
        batched_input = [
            self.prepare_input(image, box, image_size)
            for image, box, image_size in zip(images, sam_boxes, image_sizes)
        ]
        return self.model(batched_input, multimask_output=multimask_output)

    @torch.no_grad()
    def _process(
        self,
        boxes: List[torch.Tensor],
        phrases: List[str],
        sam_images: torch.Tensor,
        image_sizes: List[Tuple[int, int]],
        multimask_output: bool = False,
        **kwargs,
    ) -> Dict[str, List]:
        assert len(boxes) == len(phrases), "Boxes and phrases must be provided together"
        batch_size = sam_images.size(0)

        # Create a mask for valid (non-empty) boxes
        valid_mask = torch.tensor([box.numel() > 0 for box in boxes], dtype=torch.bool)

        # Initialize output lists
        output_masks = [None] * batch_size
        output_boxes = [None] * batch_size
        output_scores = [None] * batch_size
        output_phrases = [None] * batch_size

        if valid_mask.any():
            # Process only valid entries
            valid_sam_images = sam_images[valid_mask]
            valid_boxes = [box for box, is_valid in zip(boxes, valid_mask) if is_valid]
            valid_image_sizes = [
                size for size, is_valid in zip(image_sizes, valid_mask) if is_valid
            ]

            # Run SAM on valid images
            sam_outputs = self.detect(
                valid_sam_images, valid_boxes, valid_image_sizes, multimask_output
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
                index = valid_indices[i].item()
                output_masks[index] = mask
                output_boxes[index] = box
                output_scores[index] = score
                output_phrases[index] = phrase

        return {
            "masks": output_masks,
            "boxes": output_boxes,
            "scores": output_scores,
            "phrases": output_phrases,
        }

    @torch.no_grad()
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        warnings.filterwarnings(action="ignore", category=UserWarning)
        warnings.filterwarnings(action="ignore", category=FutureWarning)

        if not self.validate_input(data):
            raise ValueError("Invalid input data")

        images = data.pop("images", [])
        boxes = data.pop("boxes", [])
        phrases = data.pop("phrases", [])
        max_image_side = data.pop("max_image_side", 1024)

        resized_images = [load_resize_image(image, max_image_side) for image in images]
        image_sizes = [(size[1], size[0]) for size in [image.size for image in images]]
        sam_images = torch.stack(
            [self.transform_image(image) for image in resized_images]
        ).to(self.device)

        results = self._process(boxes, phrases, sam_images, image_sizes, **data)
        results["images"] = images  # Add original images to the results
        return results
