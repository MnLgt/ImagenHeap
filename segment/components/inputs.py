import os
from abc import ABC, abstractmethod
from typing import List, Union, overload
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from segment.components.base import Component
from segment.utils import load_image

ImageInput = Union[Path, List[Path], str, List[str], Image.Image, List[Image.Image]]


@dataclass
class ImageLoader(Component):
    image: ImageInput

    @overload
    def process(self, image: None = None) -> List[Image.Image]: ...

    @overload
    def process(self, image: ImageInput) -> List[Image.Image]: ...

    def process(self, image: Union[ImageInput, None] = None) -> List[Image.Image]:
        """
        Processes the input image(s) and returns a list of PIL Image objects.

        Args:
            image: Input image(s). If None, uses the image attribute.

        Returns:
            A list of PIL Image objects.

        Raises:
            ValueError: If the input image is not a supported type.
        """
        image_input = image if image is not None else self.image
        return self._process_image_input(image_input)

    def _process_image_input(self, image_input: ImageInput) -> List[Image.Image]:
        if isinstance(image_input, (str, Path)):
            return [self._load_single_image(image_input)]
        elif isinstance(image_input, list):
            return [self._load_single_image(img) for img in image_input]
        elif isinstance(image_input, Image.Image):
            return [image_input]
        else:
            raise ValueError(
                "Image must be a string, Path, list of strings/Paths, PIL Image, or list of PIL Images."
            )

    @staticmethod
    def _load_single_image(img: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(img, (str, Path)):
            return load_image(str(img))
        elif isinstance(img, Image.Image):
            return img
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
