import os

os.environ["CUDA_HOME"] = "/usr/local/cuda-12.0"
import random
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
import argparse
from diffusers.utils import make_image_grid
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

# segment anything
from segment_anything import build_sam, SamPredictor
import numpy as np

# diffusers
import torch

# BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration

config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
ckpt_repo_id = "ShilongLiu/GroundingDINO"


ckpt_filename = "weights/groundingdino_swint_ogc.pth"
sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
output_dir = "outputs"
device = "cuda" if torch.cuda.is_available() else "cpu"


blip_processor = None
blip_model = None
groundingdino_model = None
sam_predictor = None


def transform_image(image_pil):

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image


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
    model, image, caption, box_threshold, text_threshold, with_logits=True
):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer
        )
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


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


def run_grounded_sam(
    input_image,
    text_prompt,
    box_threshold,
    text_threshold,
    iou_threshold,
):
    global groundingdino_model, sam_predictor

    # load image
    image_pil = input_image.convert("RGB")
    transformed_image = transform_image(image_pil)

    if groundingdino_model is None:
        groundingdino_model = load_model(config_file, ckpt_filename, device=device)

    # run grounding dino model
    boxes_filt, scores, pred_phrases = get_grounding_output(
        groundingdino_model,
        transformed_image,
        text_prompt,
        box_threshold,
        text_threshold,
    )

    size = image_pil.size

    # process boxes
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()

    # nms
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]

    if sam_predictor is None:
        # initialize SAM
        assert sam_checkpoint, "sam_checkpoint is not found!"
        sam = build_sam(checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)

    image = np.array(image_pil)
    sam_predictor.set_image(image)

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        boxes_filt, image.shape[:2]
    ).to(device)

    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # masks: [1, 1, 512, 512]

    mask_image = Image.new("RGBA", size, color=(0, 0, 0, 0))

    mask_draw = ImageDraw.Draw(mask_image)
    for mask in masks:
        draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)

    image_draw = ImageDraw.Draw(image_pil)

    for box, label in zip(boxes_filt, pred_phrases):
        draw_box(box, image_draw, label)

    image_pil = image_pil.convert("RGBA")
    image_pil.alpha_composite(mask_image)
    return dict(
        box_image=image_pil,
        mask_image=mask_image,
        masks=masks,
        boxes=boxes_filt,
        label_scores=pred_phrases,
    )


def get_results(
    input_image,
    labels,
    box_threshold=0.3,
    text_threshold=0.25,
    iou_threshold=0.8,
):
    text_prompt = " . ".join(labels)
    input_image = input_image.convert("RGB")
    return run_grounded_sam(
        input_image,
        text_prompt,
        box_threshold,
        text_threshold,
        iou_threshold,
    )
