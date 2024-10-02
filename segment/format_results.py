
# determine whether two masks overlap

from segment.utils import unload_box, unload_mask, get_coco_style_polygons
from typing import List 
from PIL import Image

def format_scores(scores):
    return [score.squeeze().cpu().numpy().tolist() for score in scores]


def format_masks(masks):
    return [unload_mask(mask) for mask in masks]

def format_polygons(masks: List[Image.Image]) -> List[List[List[float]]]:
    return [get_coco_style_polygons(mask) for mask in masks]

def format_boxes(boxes):
    return [unload_box(box) for box in boxes]


def format_results(
    masks,
    boxes,
    scores,
    phrases,
    **kwargs
):
    results_dict = []
    
    masks = format_masks(masks)
    polygons = format_polygons(masks)
    boxes = format_boxes(boxes)
    scores = format_scores(scores)
    
    for mask, box, score, phrase, polygon in zip(masks, boxes, scores, phrases, polygons):
        results_row = dict(
            mask=mask,
            box=box,
            score=score,
            phrase=phrase,
            polygons=polygon,
            
            
        )
        results_dict.append(results_row)

    results_dict = sorted(results_dict, key=lambda x: x["phrase"])

    return results_dict

def format_all_results(sam_results, **kwargs):
    results = []
    for result in sam_results:
        try:
            result = format_results(**result)
        except Exception as e:
            print(f"One Image had an error: {e}")
            result = []

        results.append(result)
    return results