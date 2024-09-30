from abc import ABC, abstractmethod
from typing import Dict, Any, List, TypeVar, Union
from PIL import Image, ImageOps
import os
from typing import Callable, Union
import requests
from dataclasses import dataclass

ImageInput = TypeVar("ImageInput", str, List[str], Image.Image, List[Image.Image])


def load_image(
    image: Union[str, Image.Image],
    convert_method: Callable[[Image.Image], Image.Image] = None,
) -> Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `Image.Image`):
            The image to convert to the PIL Image format.
        convert_method (Callable[[Image.Image], Image.Image], optional):
            A conversion method to apply to the image after loading it.
            When set to `None` the image will be converted "RGB".

    Returns:
        `Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
            )
    elif isinstance(image, Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
        )

    image = ImageOps.exif_transpose(image)

    if convert_method is not None:
        image = convert_method(image)
    else:
        image = image.convert("RGB")

    return image


class Component(ABC):
    @abstractmethod
    def process(self, data: Any = None) -> Any:
        """
        Processes the input data and returns the output data.

        Args:
            data: Input data.

        Returns:
            Processed data.
        """
        pass


@dataclass
class ImageLoader(Component):
    image: ImageInput

    def process(self, image: ImageInput = None) -> List[Image.Image]:
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

        if isinstance(image_input, str):
            return [load_image(image_input)]
        elif isinstance(image_input, List):
            return [
                load_image(img) if isinstance(img, str) else img for img in image_input
            ]
        elif isinstance(image_input, Image.Image):
            return [image_input]
        else:
            raise ValueError(
                "Image must be a string, list of strings, PIL Image, or list of PIL Images."
            )
