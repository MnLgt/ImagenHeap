import torch
from typing import List, Dict, Any
from imagenheap.utils import get_coco_style_polygons
from PIL import Image
import numpy as np


def unload_mask(mask):
    # permute the mask to the right order
    mask = mask.permute(1, 2, 0)

    mask = mask.cpu().numpy().squeeze()
    mask = mask.astype(np.uint8) * 255
    return Image.fromarray(mask).convert("L")


def unload_box(box):
    return box.cpu().numpy().tolist()

class ResultFormatter:
    @staticmethod
    def _format_field(data: List[List[Any]], formatter_func):
        return [
            [formatter_func(item) if item is not None else None for item in row]
            for row in data
        ]

    @classmethod
    def format_scores(cls, scores: List[List[Any]]) -> List[List[float]]:
        return cls._format_field(scores, lambda x: x.squeeze().cpu().numpy().tolist())

    @classmethod
    def format_masks(cls, masks: List[List[Any]]) -> List[List[Any]]:
        return cls._format_field(masks, unload_mask)

    @classmethod
    def format_polygons(cls, masks: List[List[Any]]) -> List[List[Any]]:
        return cls._format_field(masks, get_coco_style_polygons)

    @classmethod
    def format_boxes(cls, boxes: List[List[Any]]) -> List[List[Any]]:
        return cls._format_field(boxes, unload_box)

    @classmethod
    def format_results(
        cls, sam_results: Dict[str, List[Any]], include_polygons: bool = True
    ) -> List[List[Dict[str, Any]]]:
        num_images = len(sam_results["images"])
        formatted_results = []

        formatted_fields = {
            "box": cls.format_boxes(sam_results["boxes"]),
            "score": cls.format_scores(sam_results["scores"]),
            "phrase": sam_results["phrases"],
        }

        if "masks" in sam_results:
            formatted_fields["mask"] = cls.format_masks(sam_results["masks"])
            if include_polygons:
                formatted_fields["polygon"] = cls.format_polygons(
                    formatted_fields["mask"]
                )

        for img_idx in range(num_images):
            img_results = []
            for item_idx in range(len(formatted_fields["box"][img_idx])):
                if formatted_fields["box"][img_idx][item_idx] is not None:
                    result_row = {
                        "image_index": img_idx,
                        **{
                            field: values[img_idx][item_idx]
                            for field, values in formatted_fields.items()
                        },
                    }

                    # Include any additional fields from sam_results
                    for key, value in sam_results.items():
                        if key not in ["images"] + [
                            f"{k}es" if k.endswith("x") else f"{k}s"
                            for k in formatted_fields.keys()
                        ]:
                            field_name = key[:-1] if key.endswith("s") else key
                            result_row[field_name] = (
                                value[img_idx][item_idx]
                                if isinstance(value[img_idx], list)
                                else value[img_idx]
                            )

                    img_results.append(result_row)

            formatted_results.append(sorted(img_results, key=lambda x: x["phrase"]))

        return formatted_results

    @classmethod
    def format_all_results(
        cls, sam_results: Dict[str, List[Any]], **kwargs
    ) -> List[List[Dict[str, Any]]]:
        try:
            return cls.format_results(sam_results, **kwargs)
        except Exception as e:
            print(f"Error formatting results: {e}")
            return []
