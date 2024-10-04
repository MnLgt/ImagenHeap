import os
from pathlib import Path

# Default weights directory
DEFAULT_WEIGHTS_DIR = Path(os.environ.get("IMAGENHEAP_WEIGHTS_DIR", Path.home() / ".imagenheap" / "weights"))

# Weight file names
GROUNDINGDINO_WEIGHTS = "groundingdino_swint_ogc.pth"
SAM_WEIGHTS = "sam_vit_h_4b8939.pth"

# Ensure the weights directory exists
DEFAULT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)