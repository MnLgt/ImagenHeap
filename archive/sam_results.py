import numpy as np

from SEGMENT.imagenheap.utils import (display_image_with_masks, unload_box,
                                   unload_mask)


def format_labels(label_scores):
    return [label.replace(")", "").split("(")[0] for label in label_scores]


def format_scores(label_scores):
    return [label.replace(")", "").split("(")[1] for label in label_scores]


def format_masks(masks):
    return [unload_mask(mask) for mask in masks]


def format_boxes(boxes):
    return [unload_box(box) for box in boxes]


# determine whether two masks overlap
def masks_overlap(mask1, mask2):
    return np.any(np.logical_and(mask1, mask2))


def remove_non_person_masks(person_mask, formatted_results):
    return [
        f
        for f in formatted_results
        if f.get("label") == "person" or masks_overlap(person_mask, f.get("mask"))
    ]


def format_results(labels, scores, boxes, masks, labels_dict):

    # check that the person mask is present
    assert "person" in labels, "Person mask not present in results"

    results_dict = []
    for row in zip(labels, scores, boxes, masks):
        label, score, box, mask = row
        label_id = labels_dict[label]
        results_row = dict(
            label=label, score=score, mask=mask, box=box, label_id=label_id
        )
        results_dict.append(results_row)

    results_dict = sorted(results_dict, key=lambda x: x["label"])

    # Get the person mask
    person_mask = [f for f in results_dict if f.get("label") == "person"][0]["mask"]

    assert person_mask is not None, "Person mask not found in results"

    # Remove any results that do no overlap with the person
    results_dict = remove_non_person_masks(person_mask, results_dict)

    return results_dict


class SAMResults:

    def __init__(
        self,
        image,
        labels_dict,
        box_image=None,
        mask_image=None,
        masks=None,
        boxes=None,
        label_scores=None,
        **kwargs
    ):
        self.image = image
        self.labels_dict = labels_dict
        self.box_image = box_image
        self.mask_image = mask_image
        self.masks = format_masks(masks)
        self.boxes = format_boxes(boxes)
        self.scores = format_scores(label_scores)
        self.labels = format_labels(label_scores)
        self.labels_dict = labels_dict
        self.formatted_results = format_results(
            self.labels, self.scores, self.boxes, self.masks, self.labels_dict
        )

    def display_results(self):
        if len(self.masks) < 4:
            cols = len(self.masks)
        else:
            cols = 4
        return display_image_with_masks(self.image, self.formatted_results, cols=cols)

    def get_mask(self, mask_label):
        assert mask_label in self.labels, "Mask label not found in results"
        return [f for f in r.formatted_results if f.get("label") == mask_label]
