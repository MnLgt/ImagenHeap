import numpy as np
from PIL import ImageDraw

from segment.utils import get_coco_style_polygons, unload_box, unload_mask
from segment.visualizer import display_image_with_masks_and_boxes
from itertools import groupby
from operator import itemgetter

# determine whether two masks overlap
import numpy as np
from PIL import Image


def format_scores(scores):
    return [score.squeeze().cpu().numpy().tolist() for score in scores]


def format_masks(masks):
    return [unload_mask(mask) for mask in masks]


def format_boxes(boxes):
    return [unload_box(box) for box in boxes]


def format_results(
    labels,
    scores,
    boxes,
    masks,
    labels_dict,
    polygons,
):
    results_dict = []
    for row in zip(labels, scores, boxes, masks, polygons):
        label, score, box, mask, polygon = row
        label_id = labels_dict[label]
        results_row = dict(
            label=label,
            score=score,
            mask=mask,
            box=box,
            label_id=label_id,
            polygons=polygon,
        )
        results_dict.append(results_row)

    results_dict = sorted(results_dict, key=lambda x: x["label"])

    return results_dict


class SAMResults:

    def __init__(
        self,
        image,
        labels_dict,
        masks=None,
        boxes=None,
        scores=None,
        phrases=None,
        person_masks_only=True,
        overlap_thresh=0.8,
        labels_to_dedupe=[
            "hair",
            "face",
            "neck",
            "back",
            "outfit",
            "phone",
            "hat",
            "bag",
        ],
        include_polygons=True,
        **kwargs,
    ):
        self.image = image
        self.masks = format_masks(masks)
        self.boxes = format_boxes(boxes)
        self.scores = format_scores(scores)
        self.polygons = self.get_polygons()
        self.labels = phrases
        self.labels_dict = labels_dict
        self.person_masks_only = person_masks_only
        self.formatted_results = format_results(
            self.labels,
            self.scores,
            self.boxes,
            self.masks,
            self.labels_dict,
            self.polygons if include_polygons else [None] * len(self.masks),
            self.person_masks_only,
            overlap_thresh=overlap_thresh,
            labels_to_dedupe=labels_to_dedupe,
        )

    def get_polygons(self):
        return [get_coco_style_polygons(mask) for mask in self.masks]

    def display_results(self, **kwargs):
        if len(self.masks) < 4:
            cols = len(self.masks)
        else:
            cols = 4
        return display_image_with_masks_and_boxes(
            self.image, self.formatted_results, cols=cols, **kwargs
        )

    def get_mask(self, mask_label):
        assert mask_label in self.labels, "Mask label not found in results"
        return [f for f in self.formatted_results if f.get("label") == mask_label]
