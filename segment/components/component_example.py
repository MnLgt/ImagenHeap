from typing import Dict, Any, List
from PIL import Image
from segment.components.base import Component

InputRequirements = {"images": List[Image.Image]}


class ImageSize(Component):
    def __init__(self, name: str = "image_sizer"):
        super().__init__(name)
        # self.set_input_requirements(InputRequirements)
        # self.set_output_keys(["images", "sizes"])

    def load_model(self):
        # No model to load for this component
        pass

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_input(data):
            raise ValueError("Invalid input data")

        images = data["images"]
        sizes = [image.size for image in images]

        # Create a new dictionary with the original data and the new sizes
        return {**data, "sizes": sizes}


from transformers import AutoImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
from functools import lru_cache


class Depth(Component):
    def __init__(self):
        super().__init__("depth")
        self.image_processor = None
        self.model = None

    def get_image_processor(self):
        return AutoImageProcessor.from_pretrained("facebook/dpt-dinov2-small-kitti")

    @lru_cache(maxsize=1)
    def load_model(self):
        self.image_processor = self.get_image_processor()
        self.model = DPTForDepthEstimation.from_pretrained(
            "facebook/dpt-dinov2-small-kitti"
        )

    def _process(self, images):

        # prepare image for the model
        inputs = self.image_processor(images=images, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=images[0].size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # visualize the prediction
        output = prediction.squeeze().cpu().numpy()

        depth_images = []

        for output in outputs:
            formatted = (output * 255 / np.max(output)).astype("uint8")
            depth = Image.fromarray(formatted)
            depth_images.append(depth)
        return depth_images

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:

        images = data.get("images", None)
        assert images is not None, "Images not found in data."

        depth_images = self._process(images)

        return data.update({"depth_images": depth_images})
