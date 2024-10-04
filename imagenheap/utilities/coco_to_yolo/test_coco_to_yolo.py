import unittest
from coco_to_yolo import coco_to_yolo, convert_dataset, Point


class TestCocoToYoloConverter(unittest.TestCase):

    def test_single_polygon_conversion(self):
        coco_polygons = [[0, 0, 100, 0, 100, 100, 0, 100]]
        image_width, image_height = 200, 200
        expected_output = [0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5]
        self.assertAlmostEqual(
            coco_to_yolo(coco_polygons, image_width, image_height), expected_output
        )

    def test_multi_polygon_conversion(self):
        coco_polygons = [
            [0, 0, 100, 0, 100, 100, 0, 100],
            [50, 50, 75, 50, 75, 75, 50, 75],
        ]
        image_width, image_height = 200, 200
        result = coco_to_yolo(coco_polygons, image_width, image_height)
        self.assertEqual(
            len(result), 16
        )  # Expect 8 points (16 coordinates) after merging

    def test_empty_polygon_list(self):
        with self.assertRaises(ValueError):
            coco_to_yolo([], 200, 200)

    def test_invalid_image_dimensions(self):
        coco_polygons = [[0, 0, 100, 0, 100, 100, 0, 100]]
        with self.assertRaises(ValueError):
            coco_to_yolo(coco_polygons, 0, 200)
        with self.assertRaises(ValueError):
            coco_to_yolo(coco_polygons, 200, 0)

    def test_large_dataset_conversion(self):
        large_dataset = [
            ([[0, 0, 100, 0, 100, 100, 0, 100]], 200, 200) for _ in range(1000)
        ]
        results = convert_dataset(large_dataset, batch_size=100)
        self.assertEqual(len(results), 1000)
        self.assertIsNotNone(results[0])


if __name__ == "__main__":
    unittest.main()
