import os

os.environ["CUDA_HOME"] = "/usr/local/cuda-12.0"
import random
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
import argparse

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
import numpy as np
# diffusers
import torch
import torchvision
from diffusers.utils import make_image_grid
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (clean_state_dict,
                                                    get_phrases_from_posmap)
from PIL import Image, ImageDraw, ImageFont
# segment anything
from segment_anything import SamPredictor, build_sam
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision import transforms as TorchTransforms
# BLIP
from transformers import BlipForConditionalGeneration, BlipProcessor


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


def get_grounding_output(
    model, images, caption, box_threshold, text_threshold, with_logits=True
):
    # Preprocess captions: lowercase, strip spaces, and ensure ending with a period
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    # multiply the captions for the number of images
    processed_captions = [caption] * len(images)

    with torch.no_grad():
        # Ensure images is a batch (first dimension is batch size)
        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        # Obtain model outputs for the batch of images and captions
        outputs = model(images, captions=processed_captions)
        logits = outputs["pred_logits"].cpu().sigmoid()  # (batch_size, nq, 256)
        boxes = outputs["pred_boxes"].cpu()  # (batch_size, nq, 4)

    batch_boxes_filt = []
    batch_scores = []
    batch_pred_phrases = []

    # Tokenizer
    tokenizer = model.tokenizer

    # Process each item in the batch
    for idx, (logit, box, caption) in enumerate(zip(logits, boxes, processed_captions)):
        # Filter output
        filt_mask = logit.max(dim=1)[0] > box_threshold
        logits_filt = logit[filt_mask]  # num_filt, 256
        boxes_filt = box[filt_mask]  # num_filt, 4

        # Tokenize caption
        tokenized = tokenizer(caption)

        # Build predictions
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenizer
            )
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
            scores.append(logit.max().item())

        # Collect results for this batch element
        batch_boxes_filt.append(boxes_filt)
        batch_scores.append(torch.Tensor(scores))
        batch_pred_phrases.append(pred_phrases)

    return batch_boxes_filt, batch_scores, batch_pred_phrases


# %%
def draw_mask(mask, draw, random_color=True):
    if random_color:
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            153,
        )
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)


def draw_masks(masks, image_size):
    mask_image = Image.new("RGBA", image_size, color=(0, 0, 0, 0))

    mask_draw = ImageDraw.Draw(mask_image)
    for mask in masks:
        draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)
    return mask_image


def draw_box(box, draw, label):
    # random color
    color = tuple(np.random.randint(0, 255, size=3).tolist())

    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color, width=2)

    if label:
        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((box[0], box[1]), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (box[0], box[1], w + box[0], box[1] + h)
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1]), str(label), fill="white")

        draw.text((box[0], box[1]), label)


def draw_boxes(image_pil, boxes_filt, pred_phrases):
    image_draw = ImageDraw.Draw(image_pil)

    for box, label in zip(boxes_filt, pred_phrases):
        draw_box(box, image_draw, label)

    image_pil = image_pil.convert("RGBA")
    image_pil.alpha_composite(mask_image)
    return image_pil


# %%
config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
ckpt_repo_id = "ShilongLiu/GroundingDINO"


ckpt_filename = "weights/groundingdino_swint_ogc.pth"
sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
output_dir = "outputs"
device = "cuda" if torch.cuda.is_available() else "cpu"


blip_processor = None
blip_model = None
groundingdino_model = load_model(config_file, ckpt_filename, device=device)
sam_predictor = None


# SAM PREDICTOR
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)


# %%
def transform_image_dino(image_pil):

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image


def transform_image_sam(image_pil):
    image_numpy = np.array(image_pil)
    sam_transform = ResizeLongestSide(sam.image_encoder.img_size)
    input_image = sam_transform.apply_image(image_numpy)
    input_image_torch = torch.as_tensor(input_image, device=device)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    return input_image_torch


def dino_preds(
    transformed_images,  # Assuming batch of images
    text_prompts,  # Assuming list of text prompts
    box_threshold,
    text_threshold,
    iou_threshold,
    image_size,
):
    # Run grounding dino model
    batch_boxes_filt, batch_scores, batch_pred_phrases = get_grounding_output(
        groundingdino_model,
        transformed_images,
        text_prompts,
        box_threshold,
        text_threshold,
    )

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


def sam_preds(images, batch_boxes_filt):
    batch_masks = []

    for image, boxes_filt in zip(images, batch_boxes_filt):
        if isinstance(image, torch.Tensor):
            image_size = image.shape[-2:]
            sam_predictor.set_torch_image(image, image_size)
        else:
            image = np.array(image)
            image_size = image.shape[:2]
            sam_predictor.set_image(image)

        # Transform boxes according to the specific image size
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            boxes_filt, image_size
        ).to(device)

        # Predict masks for the transformed boxes
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # Collect masks for each batch element
        batch_masks.append(masks)

    return batch_masks


def run_grounded_sam(
    input_image,
    text_prompt,
    box_threshold=0.3,
    text_threshold=0.25,
    iou_threshold=0.8,
    sam_image=None,
):
    # load image
    if isinstance(input_image, torch.Tensor):
        transformed_image = input_image
        height, width = tuple(transformed_image.shape[-2:])
        image_size = (width, height)
    else:
        image_pil = input_image.convert("RGB")
        image_size = image_pil.size
        transformed_image = transform_image_dino(image_pil)

    ## DINO Ops
    boxes_filt, scores, pred_phrases = dino_preds(
        transformed_image,
        text_prompt,
        box_threshold,
        text_threshold,
        iou_threshold,
        image_size,
    )

    ## SAM Ops
    if sam_image is not None:
        masks = sam_preds(sam_image, boxes_filt)
    else:
        masks = sam_preds(input_image, boxes_filt)

    return dict(
        masks=masks,
        boxes=boxes_filt,
        label_scores=pred_phrases,
    )
