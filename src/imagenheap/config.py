import os
from pathlib import Path
import sys


def get_package_root():
    """Get the root directory of the installed package."""
    return Path(__file__).parent


# Default weights directory: inside the package by default
DEFAULT_WEIGHTS_DIR = os.environ.get(
    "IMAGENHEAP_WEIGHTS_DIR", str(get_package_root() / "weights")
)

# Ensure DEFAULT_WEIGHTS_DIR is a Path object
DEFAULT_WEIGHTS_DIR = Path(DEFAULT_WEIGHTS_DIR)

# Weight file names
GROUNDINGDINO_WEIGHTS = "groundingdino_swint_ogc.pth"
SAM_WEIGHTS = "sam_vit_h_4b8939.pth"

# Ensure the weights directory exists
DEFAULT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Print the weights directory for debugging
print(f"Weights directory: {DEFAULT_WEIGHTS_DIR}", file=sys.stderr)
