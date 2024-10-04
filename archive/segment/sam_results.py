import numpy as np
from PIL import ImageDraw

from imagenheap.utils import get_coco_style_polygons, unload_box, unload_mask
from imagenheap.visualizer import visualizer
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


def compute_mask_overlap(mask1, mask2):
    """
    Compute the overlap score between two binary mask images using IoU.

    Args:
    mask1 (PIL.Image): First binary mask image
    mask2 (PIL.Image): Second binary mask image

    Returns:
    float: IoU score (0-1) representing the overlap between masks
    """
    # Convert PIL images to numpy arrays
    mask1_array = np.array(mask1.convert("1"))
    mask2_array = np.array(mask2.convert("1"))

    # Compute intersection and union
    intersection = np.logical_and(mask1_array, mask2_array)
    union = np.logical_or(mask1_array, mask2_array)

    # Compute IoU score
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


def deduplicate_labels(
    mask_list, labels=["hair", "face", "neck", "back", "outfit", "phone", "hat", "bag"]
):
    """
    Deduplicate items in the mask_list that have labels matching the labels.
    For each special label, only the item with the highest score is kept.

    Args:
    mask_list (list): List of dictionaries containing 'label', 'mask', and 'mask_score'
    labels (list): List of labels to be deduplicated

    Returns:
    list: Processed list with special label items deduplicated
    """
    result = []
    special_items = {label: {"item": None, "score": -1} for label in labels}

    for item in mask_list:
        label = item["label"].split("_")[
            0
        ]  # Assuming label format is "category_details"

        if label in labels:
            # If this item has a higher score, update the special_items dict
            if item["score"] > special_items[label]["score"]:
                special_items[label] = {"item": item, "score": item["score"]}
        else:
            # If not a special label, add to result list as is
            result.append(item)

    # Add the highest scored item for each special label to the result
    for label, data in special_items.items():
        if data["item"] is not None:
            result.append(data["item"])

    return result


def process_mask_list(mask_list, overlap_thresh=0.8):
    """
    Process a list of mask dictionaries, group by row, and remove duplicates.

    Args:
    mask_list (list): List of dictionaries containing 'label', 'mask', and 'mask_score'

    Returns:
    list: Processed list with duplicates removed
    """
    # Sort the list by row (assuming the row is part of the label)
    sorted_list = sorted(mask_list, key=lambda x: x["label"].split("_")[0])

    # Group by row
    grouped = groupby(sorted_list, key=lambda x: x["label"].split("_")[0])

    result = []
    for _, group in grouped:
        group_list = list(group)

        # If there's only one item in the group, add it to the result
        if len(group_list) == 1:
            result.append(group_list[0])
            continue

        # Compare masks within the group
        while group_list:
            current = group_list.pop(0)
            to_remove = []

            for other in group_list:
                overlap = compute_mask_overlap(current["mask"], other["mask"])
                if overlap > overlap_thresh:  # 80% overlap threshold
                    if current["score"] >= other["score"]:
                        to_remove.append(other)
                    else:
                        to_remove.append(current)
                        current = other
                        break

            # Remove duplicates
            for item in to_remove:
                if item in group_list:
                    group_list.remove(item)

            # Add the kept item to the result
            if current not in result:
                result.append(current)

    return result


def remove_non_person_masks(person_mask, formatted_results, overlap_thresh=0.8):
    return [
        f
        for f in formatted_results
        if f.get("label") == "person"
        or compute_mask_overlap(person_mask, f.get("mask")) > overlap_thresh
    ]


def draw_box(image, box, color="purple", width=3):
    draw = ImageDraw.Draw(image.copy())
    draw.rectangle(box, width=width, outline=color)
    return image


def format_results(
    labels,
    scores,
    boxes,
    masks,
    labels_dict,
    polygons,
    person_masks_only=True,
    overlap_thresh=0.8,
    labels_to_dedupe=None,
):

    # check that the person mask is present
    if person_masks_only:
        assert "person" in labels, "Person mask not present in results"

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

    if person_masks_only:
        # Get the person mask
        person_mask = [f for f in results_dict if f.get("label") == "person"][0]["mask"]

        assert person_mask is not None, "Person mask not found in results"

        # Remove any results that do no overlap with the person
        # The purpose of this is to ensure that you are not getting masks for objects that are not on the person
        results_dict = remove_non_person_masks(person_mask, results_dict)

    if labels_to_dedupe:
        results_dict = deduplicate_labels(results_dict, labels_to_dedupe)

    return process_mask_list(results_dict, overlap_thresh)


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
        return visualizer(self.image, self.formatted_results, cols=cols, **kwargs)

    def get_mask(self, mask_label):
        assert mask_label in self.labels, "Mask label not found in results"
        return [f for f in self.formatted_results if f.get("label") == mask_label]
