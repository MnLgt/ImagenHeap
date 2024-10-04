from typing import List, Dict, Any
from PIL import Image
from segment.utils import unload_box, unload_mask, get_coco_style_polygons


class ResultFormatter:
    @staticmethod
    def format_scores(scores: List[List[Any]]) -> List[List[float]]:
        return [
            [
                score.squeeze().cpu().numpy().tolist() if score is not None else None
                for score in img_scores
            ]
            for img_scores in scores
        ]

    @staticmethod
    def format_masks(masks: List[List[Any]]) -> List[List[Any]]:
        return [
            [unload_mask(mask) if mask is not None else None for mask in img_masks]
            for img_masks in masks
        ]

    @staticmethod
    def format_polygons(masks: List[List[Any]]) -> List[List[Any]]:
        return [
            [
                get_coco_style_polygons(mask) if mask is not None else None
                for mask in img_masks
            ]
            for img_masks in masks
        ]

    @staticmethod
    def format_boxes(boxes: List[List[Any]]) -> List[List[Any]]:
        return [
            [unload_box(box) if box is not None else None for box in img_boxes]
            for img_boxes in boxes
        ]

    def format_results(
        self, sam_results: Dict[str, List[Any]]
    ) -> List[List[Dict[str, Any]]]:
        formatted_results = []
        num_images = len(sam_results["images"])

        # Format special fields
        no_masks = False
        if "masks" in sam_results:
            formatted_masks = self.format_masks(sam_results["masks"])
            formatted_polygons = self.format_polygons(formatted_masks)
        else:
            no_masks = True

        formatted_boxes = self.format_boxes(sam_results["boxes"])
        formatted_scores = self.format_scores(sam_results["scores"])

        for img_idx in range(num_images):
            img_results = []
            for item_idx in range(len(sam_results["boxes"][img_idx])):
                result_row = {
                    "image_index": img_idx,
                    "box": formatted_boxes[img_idx][item_idx],
                    "score": formatted_scores[img_idx][item_idx],
                    "phrase": sam_results["phrases"][img_idx][item_idx],
                }

                if not no_masks:
                    result_row.update({"mask": formatted_masks[img_idx][item_idx]})
                    result_row.update(
                        {"polygons": formatted_polygons[img_idx][item_idx]}
                    )

                # Include any additional fields from sam_results
                for key, value in sam_results.items():
                    if key not in ["images", "masks", "boxes", "scores", "phrases"]:
                        if key[-1] == "s":
                            key = key[:-1]
                        if isinstance(value[img_idx], list):
                            result_row[key] = value[img_idx][item_idx]
                        else:
                            result_row[key] = value[img_idx]

                if result_row["box"] is not None:  # Only add results for valid entries
                    img_results.append(result_row)

            img_results = sorted(img_results, key=lambda x: x["phrase"])
            formatted_results.append(img_results)

        return formatted_results

    def format_all_results(
        self, sam_results: Dict[str, List[Any]], **kwargs
    ) -> List[List[Dict[str, Any]]]:
        try:
            results = self.format_results(sam_results)
        except Exception as e:
            print(f"Error formatting results: {e}")
            results = []

        return results
