from typing import Union, List
from pathlib import Path
from PIL import Image

ImageInput = Union[Path, List[Path], str, List[str], Image.Image, List[Image.Image]]