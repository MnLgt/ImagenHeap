import os
from typing import Any, Dict, List
from functools import lru_cache
from typing import List, Tuple

import torch
import torchvision
from PIL import Image
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)
import groundingdino.datasets.transforms as T
from groundingdino.config import GroundingDINO_SwinT_OGC


from imagenheap.utilities.box_rescaler import BoxRescaler, BoxFormat, CoordinateSystem
from imagenheap.utils import get_device
from imagenheap.components.base import Component
from imagenheap.utils import load_resize_image, load_image
import logging

logger = logging.getLogger(__name__)

# jupyter
CURDIR = os.path.dirname(__file__)

WEIGHTS_DIR = os.path.join(CURDIR, "..", "..", "..", "weights")

# Dino
DINO_CHECKPOINT = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
DINO_CONFIG = GroundingDINO_SwinT_OGC.__file__


class DetectDino(Component):
    def __init__(self, weights_dir: str = WEIGHTS_DIR, device: str = None):
        super().__init__("detect")
        self.weights_dir = weights_dir
        self.device = device or get_device()
        self.model = None
        self.transform = self._get_transform()
        self.boxes = []
        self.scores = []
        self.tokens = []
        self.phrases = []

    def load_model(self):
        if self.model is None:
            checkpoint_path = DINO_CHECKPOINT

            args = SLConfig.fromfile(DINO_CONFIG)
            args.device = self.device

            model = build_model(args)
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            load_result = model.load_state_dict(
                clean_state_dict(checkpoint["model"]), strict=False
            )
            logger.info(f"Model loading result: {load_result}")

            model.eval()
            self.model = model.to(self.device)
        else:
            logger.info("Model already loaded")

    def _get_transform(self):
        return T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _transform_image(self, image_pil: Image.Image) -> torch.Tensor:
        image, _ = self.transform(image_pil, None)
        return image

    @staticmethod
    def _text_prompt_handler(text_prompt: str, num_images: int) -> List[str]:
        if isinstance(text_prompt, list):
            text_prompt = ".".join(text_prompt)

        processed_text_prompt = text_prompt.lower().strip()
        if not processed_text_prompt.endswith("."):
            processed_text_prompt += "."
        return [processed_text_prompt] * num_images

    @lru_cache(maxsize=1)
    def _tokenize_prompt(self, text_prompt: str):
        return self.model.tokenizer(text_prompt, return_offsets_mapping=True)

    def _get_prompt_from_token(
        self, tokenized, text_prompt: str, target_input_id: int, splitter: str = "."
    ) -> str:
        try:
            target_index = tokenized.input_ids.index(target_input_id)
        except ValueError:
            return f"Input ID {target_input_id} not found in the tokenized text."

        start, end = tokenized.offset_mapping[target_index]

        current_position = 0
        for phrase in text_prompt.split(splitter):
            phrase_end = current_position + len(phrase)
            if current_position <= start < phrase_end:
                return phrase.strip()
            current_position = phrase_end + len(splitter)

        return "Phrase not found"

    def _process(
        self,
        images: torch.Tensor,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], List[List[str]], List[List[str]]
    ]:
        processed_text_prompts = self._text_prompt_handler(text_prompt, images.size(0))

        with torch.no_grad():
            outputs = self.model(images, captions=processed_text_prompts)

        prediction_logits = outputs["pred_logits"].cpu().sigmoid()
        prediction_boxes = outputs["pred_boxes"].cpu()

        mask = prediction_logits.max(dim=2)[0] > box_threshold

        tokenized = self._tokenize_prompt(text_prompt)

        batch_results = []

        for logits, boxes, m in zip(prediction_logits, prediction_boxes, mask):
            filtered_logits = logits[m]
            filtered_boxes = boxes[m]

            max_logits, max_indices = filtered_logits.max(dim=1)
            logit_mask = max_logits > text_threshold

            batch_result = {
                "boxes": filtered_boxes[logit_mask],
                "scores": max_logits[logit_mask],
                "prompt_tokens": [
                    get_phrases_from_posmap(
                        logit == max_val, tokenized, self.model.tokenizer
                    ).replace(".", "")
                    for logit, max_val in zip(
                        filtered_logits[logit_mask], max_logits[logit_mask]
                    )
                ],
                "text_prompts": [
                    self._get_prompt_from_token(
                        tokenized,
                        text_prompt,
                        tokenized.input_ids[max_idx.item()],
                        splitter=".",
                    )
                    for max_idx in max_indices[logit_mask]
                ],
            }
            batch_results.append(batch_result)

        return (
            [result["boxes"] for result in batch_results],
            [result["scores"] for result in batch_results],
            [result["prompt_tokens"] for result in batch_results],
            [result["text_prompts"] for result in batch_results],
        )

    @torch.no_grad()
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_input(data):
            raise ValueError("Invalid input data")

        assert isinstance(data, dict), "Input data must be a dictionary"
        assert "images" in data, "Input data must contain 'images'"
        images = data["images"]
        text_prompt = data["text_prompt"]
        max_image_side = data.get("max_image_side", 1024)
        box_threshold = data.get("box_threshold", 0.3)
        text_threshold = data.get("text_threshold", 0.25)
        iou_threshold = data.get("iou_threshold", 0.8)
        return_tensors = data.get("return_tensors", False)

        if isinstance(images, (str, Image.Image)):
            images = [images]

        if isinstance(text_prompt, list):
            text_prompt = ".".join(text_prompt)

        # If using without SAM then we can load the images without resizing
        # However since this is primarily being used in conjunction with SAM we will resize all images
        # If we just want to use Dino by itself we load the images at normal size, get the sizes, and then resize them
        # The format method will adjust the boxes back to the original image size coordinates

        pil_images = [load_image(image) for image in images]

        # Get the original image sizes
        image_sizes = [image.size for image in pil_images]

        # Resize images to the maximum side
        resized_images = [
            load_resize_image(image, max_image_side) for image in pil_images
        ]

        dino_images = torch.stack(
            [self._transform_image(image) for image in resized_images]
        ).to(self.device)

        boxes, scores, prompt_tokens, pred_prompts = self._process(
            dino_images, text_prompt, box_threshold, text_threshold
        )

        results = self._format_results(
            boxes,
            scores,
            prompt_tokens,
            pred_prompts,
            image_sizes,
            max_image_side,
            iou_threshold,
            return_tensors,
        )
        results.update({"images": images})
        return results

    @staticmethod
    def _resize_image(image: Image.Image, max_side: int) -> Image.Image:
        w, h = image.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            return image.resize((new_w, new_h), Image.LANCZOS)
        return image

    def _format_results(
        self,
        batch_boxes,
        batch_scores,
        batch_prompt_tokens,
        batch_pred_prompts,
        image_sizes,
        max_image_side,
        iou_threshold,
        return_tensors=False,
    ):
        results = []

        for boxes, scores, prompt_tokens, pred_prompts, original_size in zip(
            batch_boxes,
            batch_scores,
            batch_prompt_tokens,
            batch_pred_prompts,
            image_sizes,
        ):
            rescaler = BoxRescaler(original_size, max_image_side)

            if len(boxes) > 0:
                boxes = rescaler.rescale(
                    boxes,
                    from_format=BoxFormat.XYWH,
                    to_format=BoxFormat.XYXY,
                    from_system=CoordinateSystem.NORMALIZED,
                    to_system=CoordinateSystem.ABSOLUTE,
                )

                nms_idx = (
                    torchvision.ops.nms(boxes, scores, iou_threshold).numpy().tolist()
                )

                boxes = boxes[nms_idx]
                scores = scores[nms_idx]
                tokens = [prompt_tokens[idx] for idx in nms_idx]
                phrases = [pred_prompts[idx] for idx in nms_idx]
            else:
                # Create empty tensors/lists for images with no detections
                boxes = torch.empty((0, 4), dtype=torch.float32)
                scores = torch.empty(0, dtype=torch.float32)
                tokens = []
                phrases = []

            self.boxes.append(boxes)
            self.scores.append(scores)
            self.tokens.append(tokens)
            self.phrases.append(phrases)

            if not return_tensors:
                boxes = boxes.tolist()
                scores = scores.tolist()

            image_results = [
                {"box": box, "score": score, "label": token, "phrase": phrase}
                for box, score, token, phrase in zip(boxes, scores, tokens, phrases)
            ]

            results.append(image_results)

        results = {
            "boxes": self.boxes,
            "scores": self.scores,
            "tokens": self.tokens,
            "phrases": self.phrases,
        }
        # Reset the internal state
        self.reset()

        return results

    def reset(self):
        self.boxes = []
        self.scores = []
        self.tokens = []
        self.phrases = []
