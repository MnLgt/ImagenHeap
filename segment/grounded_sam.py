import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)

import os

os.environ["CUDA_HOME"] = "/usr/local/cuda-12.0"

from functools import lru_cache

import numpy as np
import torchvision
from segment_anything.utils.transforms import ResizeLongestSide


import numpy as np

# diffusers
import torch
import torchvision

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)
from PIL import Image

# segment anything
from segment_anything import build_sam
from segment_anything.utils.transforms import ResizeLongestSide
from segment.utils import get_device


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)
    _ = model.eval()
    return model


device = get_device()


ckpt_repo_id = "ShilongLiu/GroundingDINO"


@lru_cache(maxsize=1)
def get_grounding_model():
    ckpt_filename = "weights/groundingdino_swint_ogc.pth"
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    groundingdino_model = load_model(config_file, ckpt_filename, device=device).to(
        device
    )
    return groundingdino_model.to(device)


@lru_cache(maxsize=1)
def get_sam_model():
    sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
    # SAM PREDICTOR
    sam = build_sam(checkpoint=sam_checkpoint)
    return sam.to(device=device)


model = get_grounding_model()
sam = get_sam_model()

# DEFAULT THRESHOLDS
box_threshold = 0.3
text_threshold = 0.25
iou_threshold = 0.8


def caption_handler(caption, images):
    # Preprocess captions: lowercase, strip spaces, and ensure ending with a period
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    # multiply the captions for the number of images
    processed_captions = [caption] * len(images)
    return processed_captions


def transform_image_dino(image_pil):

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image


def run_dino(images, caption, **kwargs):
    processed_captions = caption_handler(caption, images)

    # Ensure images is a batch (first dimension is batch size)
    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    with torch.no_grad():
        # Obtain model outputs for the batch of images and captions
        outputs = model(images, captions=processed_captions)

    prediction_logits = (
        outputs["pred_logits"].cpu().sigmoid()
    )  # prediction_logits.shape = (num_batch, nq, 256)
    prediction_boxes = outputs[
        "pred_boxes"
    ].cpu()  # prediction_boxes.shape = (num_batch, nq, 4)

    # import ipdb; ipdb.set_trace()
    mask = (
        prediction_logits.max(dim=2)[0] > box_threshold
    )  # mask: torch.Size([num_batch, 256])

    bboxes_batch = []
    predicts_batch = []
    phrases_batch = []  # list of lists
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    for i in range(prediction_logits.shape[0]):
        logits = prediction_logits[i][mask[i]]  # logits.shape = (n, 256)
        phrases = [
            get_phrases_from_posmap(
                logit == torch.max(logit), tokenized, tokenizer
            ).replace(".", "")
            for logit in logits  # logit is a tensor of shape (256,) torch.Size([256])  # torch.Size([7, 256])
        ]
        boxes = prediction_boxes[i][mask[i]]  # boxes.shape = (n, 4)
        phrases_batch.append(phrases)
        bboxes_batch.append(boxes)
        predicts_batch.append(logits.max(dim=1)[0])

    return bboxes_batch, predicts_batch, phrases_batch


def format_dino(
    batch_boxes_filt, batch_scores, batch_pred_phrases, image_size=(1024, 1024)
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


def transform_image_sam(image, device="cpu"):
    if isinstance(image, Image.Image):
        image = np.array(image)

    transform = ResizeLongestSide(sam.image_encoder.img_size)
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device)
    return image.permute(2, 0, 1).contiguous()


def sam_prepare_boxes(boxes, image_size):
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    return resize_transform.apply_boxes_torch(boxes, image_size).to(device)


def sam_prepare_row(image, boxes, image_size):
    return dict(image=image, boxes=boxes, original_size=image_size)


def run_sam(images, boxes, image_size=(1024, 1024)):

    sam_boxes = [sam_prepare_boxes(box, image_size) for box in boxes]
    batched_input = [
        sam_prepare_row(image, box, image_size) for image, box in zip(images, sam_boxes)
    ]
    batched_output = sam(batched_input, multimask_output=False)
    return batched_output


# determine if tensor is empty
def is_empty(tensor):
    return tensor.size(0) == 0


def run_grounded_sam_batch(
    dino_images,
    sam_images,
    text_prompt,
    box_thresh=0.3,
    text_thresh=0.25,
    iou_thresh=0.8,
):
    global box_threshold, text_threshold, iou_threshold
    box_threshold = box_thresh
    text_threshold = text_thresh
    iou_threshold = iou_thresh

    # Process with DINO model
    boxes, scores, phrases = run_dino(dino_images, text_prompt)
    boxes, scores, phrases = format_dino(boxes, scores, phrases)

    # Prepare to keep track of outputs, respecting original order
    batch_size = len(dino_images)
    rows = [None] * batch_size  # Initialize with None for all entries

    # Determine if any images do not have detections
    # We do this by looking for any empty bounding boxes
    valid_indices = [i for i, box in enumerate(boxes) if not is_empty(box)]

    # Run any images with detections through SAM
    if valid_indices:
        valid_sam_images = sam_images[valid_indices]
        valid_boxes = [boxes[i] for i in valid_indices]

        # Process non-empty entries with SAM
        sam_outputs = run_sam(valid_sam_images, valid_boxes)
        valid_masks = [output.get("masks") for output in sam_outputs]

        # Filter for scores and phrases that have detections
        valid_scores = [scores[i] for i in valid_indices]
        valid_phrases = [phrases[i] for i in valid_indices]

        # Place processed results back in the correct order
        for idx, mask, box, score, phrase in zip(
            valid_indices, valid_masks, valid_boxes, valid_scores, valid_phrases
        ):
            rows[idx] = {
                "masks": mask,
                "boxes": box,
                "scores": score,
                "phrases": phrase,
            }

    # Fill in rows with None for images without detections
    for i in range(batch_size):
        if rows[i] is None:
            rows[i] = {"masks": None, "boxes": None, "scores": None, "phrases": None}

    return rows
