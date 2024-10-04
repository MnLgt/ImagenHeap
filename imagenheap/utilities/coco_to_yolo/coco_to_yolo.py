# coco_to_yolo_converter.py

import numpy as np
from typing import List, Tuple, Union
from dataclasses import dataclass
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Polygon:
    points: List[Point]


def find_min_distance_indices(arr1: np.ndarray, arr2: np.ndarray) -> Tuple[int, int]:
    """
    Find a pair of indices with the shortest distance between two arrays of points.

    Args:
        arr1 (np.ndarray): Array of shape (N, 2) representing N points.
        arr2 (np.ndarray): Array of shape (M, 2) representing M points.

    Returns:
        Tuple[int, int]: A pair of indices (i, j) where i is the index in arr1 and j is the index in arr2.
    """
    distances = np.sum((arr1[:, np.newaxis, :] - arr2[np.newaxis, :, :]) ** 2, axis=2)
    return np.unravel_index(np.argmin(distances), distances.shape)


def merge_segments(segments: List[List[float]]) -> List[Point]:
    """
    Merge multiple segments into a single polygon.

    Args:
        segments (List[List[float]]): List of segments, where each segment is a list of coordinates [x1, y1, x2, y2, ...].

    Returns:
        List[Point]: A list of Point objects representing the merged polygon.
    """
    polygon_segments = [np.array(segment).reshape(-1, 2) for segment in segments]
    merged_polygon = []

    for i in range(len(polygon_segments)):
        if i == 0:
            merged_polygon.extend([Point(x, y) for x, y in polygon_segments[i]])
        else:
            prev_segment = polygon_segments[i - 1]
            curr_segment = polygon_segments[i]
            idx1, idx2 = find_min_distance_indices(prev_segment, curr_segment)
            merged_polygon.extend([Point(x, y) for x, y in curr_segment[idx2:]])
            merged_polygon.extend([Point(x, y) for x, y in curr_segment[:idx2]])

    return merged_polygon


def normalize_coordinates(
    polygon: List[Point], image_width: int, image_height: int
) -> List[float]:
    """
    Normalize polygon coordinates based on image dimensions.

    Args:
        polygon (List[Point]): List of Point objects representing the polygon.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        List[float]: Normalized coordinates as a flat list [x1, y1, x2, y2, ...].
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    return [
        coordinate / dimension
        for point in polygon
        for coordinate, dimension in zip(
            (point.x, point.y), (image_width, image_height)
        )
    ]


def coco_to_yolo(
    coco_polygons: List[List[float]], image_width: int, image_height: int
) -> List[float]:
    """
    Convert COCO-style polygons to YOLO-style normalized coordinates.

    Args:
        coco_polygons (List[List[float]]): COCO-style polygon annotations.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        List[float]: YOLO-style normalized coordinates as a flat list.
    """
    if not coco_polygons:
        raise ValueError("Empty polygon list")

    if len(coco_polygons) > 1:
        merged_polygon = merge_segments(coco_polygons)
    else:
        merged_polygon = [
            Point(coco_polygons[0][i], coco_polygons[0][i + 1])
            for i in range(0, len(coco_polygons[0]), 2)
        ]

    return normalize_coordinates(merged_polygon, image_width, image_height)


def process_batch(batch: List[Tuple[List[List[float]], int, int]]) -> List[List[float]]:
    """
    Process a batch of COCO polygons.

    Args:
        batch (List[Tuple[List[List[float]], int, int]]): List of tuples containing
            (coco_polygons, image_width, image_height).

    Returns:
        List[List[float]]: List of YOLO-style normalized coordinates for each input in the batch.
    """
    results = []
    for coco_polygons, image_width, image_height in batch:
        try:
            yolo_polygons = coco_to_yolo(coco_polygons, image_width, image_height)
            results.append(yolo_polygons)
        except Exception as e:
            logger.error(f"Error processing polygon: {e}")
            results.append(None)
    return results


def convert_dataset(
    coco_dataset: List[Tuple[List[List[float]], int, int]], batch_size: int = 1000
) -> List[Union[List[float], None]]:
    """
    Convert a large COCO dataset to YOLO format using parallel processing.

    Args:
        coco_dataset (List[Tuple[List[List[float]], int, int]]): List of tuples containing
            (coco_polygons, image_width, image_height) for each image in the dataset.
        batch_size (int): Number of items to process in each batch.

    Returns:
        List[Union[List[float], None]]: List of YOLO-style normalized coordinates or None for failed conversions.
    """
    results = []
    total_batches = (len(coco_dataset) + batch_size - 1) // batch_size

    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(0, len(coco_dataset), batch_size):
            batch = coco_dataset[i : i + batch_size]
            futures.append(executor.submit(process_batch, batch))

        for i, future in enumerate(as_completed(futures)):
            batch_results = future.result()
            results.extend(batch_results)
            logger.info(f"Processed batch {i+1}/{total_batches}")

    return results


# Example usage
if __name__ == "__main__":
    # Simple example
    coco_polygons = [[0, 0, 100, 0, 100, 100, 0, 100], [50, 50, 75, 50, 75, 75, 50, 75]]
    image_width, image_height = 200, 200
    yolo_polygons = coco_to_yolo(coco_polygons, image_width, image_height)
    print("Single conversion result:", yolo_polygons)

    # Large dataset example
    large_dataset = [(coco_polygons, image_width, image_height) for _ in range(10000)]
    results = convert_dataset(large_dataset)
    print(f"Processed {len(results)} items. First result: {results[0]}")
