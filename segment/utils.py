import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as F
import time 
from diffusers.utils import load_image

def time_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        print(f"{round(duration,2)} seconds: {func.__name__}")
        return result

    return wrapper

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"
    return device


def display_image_with_masks(image, results, cols=4):
    # Convert PIL Image to numpy array
    image_np = np.array(image)

    # Check image dimensions
    if image_np.ndim != 3:
        raise ValueError("Image must be a 3-dimensional array")

    # Number of masks
    n = len(results)
    rows = (n + cols - 1) // cols  # Calculate required number of rows

    # Setting up the plot
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n == 1:
        axs = np.array([[axs]])
    elif rows == 1:
        axs = np.array([axs])
    else:
        axs = axs.reshape(rows, cols)

    for i, result in enumerate(results):
        mask = result["mask"]
        label = result["label"]
        score = float(result["score"])

        row = i // cols
        col = i % cols

        # Convert PIL mask to numpy array
        mask_np = np.array(mask)

        # Check mask dimensions
        if mask_np.ndim != 2:
            raise ValueError("Mask must be a 2-dimensional array")

        # Create an overlay where mask is True
        overlay = np.zeros_like(image_np)
        overlay[mask_np > 0] = [0, 0, 255]  # Applying blue color on the mask area

        # Combine the image and the overlay
        combined = image_np.copy()
        indices = np.where(mask_np > 0)
        combined[indices] = combined[indices] * 0.5 + overlay[indices] * 0.5

        # Show the combined image
        ax = axs[row, col]
        ax.imshow(combined)
        ax.axis("off")
        ax.set_title(f"Label: {label}, Score: {score:.2f}", fontsize=12)
        rect = patches.Rectangle(
            (0, 0),
            image_np.shape[1],
            image_np.shape[0],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    # Hide unused subplots if the total number of masks is not a multiple of cols
    for idx in range(i + 1, rows * cols):
        row = idx // cols
        col = idx % cols
        axs[row, col].axis("off")

    plt.tight_layout()
    plt.show()


# def unload_mask(mask):
#     mask = mask.cpu().numpy().squeeze()
#     mask = mask.astype(np.uint8) * 255
#     return Image.fromarray(mask)


def unload_mask(mask):
    # permute the mask to the right order
    mask = mask.permute(1, 2, 0)

    mask = mask.cpu().numpy().squeeze()
    mask = mask.astype(np.uint8) * 255
    return Image.fromarray(mask).convert("L")


def unload_box(box):
    return box.cpu().numpy().tolist()


def format_results(labels, scores, boxes, masks):
    results_dict = []
    for row in zip(labels, scores, boxes, masks):
        label, score, mask, box = row
        results_row = dict(label=label, score=score, mask=mask, box=box)
        results_dict.append(results_row)
    return results_dict


def get_coco_style_polygons(mask):
    """
    Extracts polygons from a binary mask in COCO style format.

    Parameters:
    - mask: A binary mask (numpy array or PIL Image).

    Returns:
    - List of polygons, where each polygon is represented as a flat list of points.
    """
    if isinstance(mask, Image.Image):
        mask = np.array(mask)

    # Ensure mask is binary
    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    coco_polygons = []
    for contour in contours:
        # Flatten the contour array
        flattened = contour.flatten()
        coco_polygons.append(flattened.tolist())

    return coco_polygons


def convert_coco_polygons_to_mask(polygons, height, width):
    """
    Converts COCO polygons back into a boolean mask.

    Parameters:
    - polygons: List of polygons where each polygon is represented as a flat list of coordinates [x1, y1, x2, y2, ...].
    - height: The height of the mask to be generated.
    - width: The width of the mask to be generated.

    Returns:
    - mask: A boolean mask with polygons filled in.
    """
    # Create an empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    # Fill the polygons
    for polygon in polygons:
        # Reshape polygon to a sequence of points
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        # Draw the polygon
        cv2.fillPoly(mask, [pts], 255)
    # Convert to boolean
    return mask > 0


def resize_preserve_aspect_ratio(image, max_side=512):
    width, height = image.size
    scale = min(max_side / width, max_side / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return image.resize((new_width, new_height))


def overlay_mask(image, mask, opacity=0.5):
    """
    Takes in a PIL image and a PIL boolean image mask. Overlay the mask on the image
    and color the mask with a low opacity blue with hex #88CFF9.
    """
    # Convert the boolean mask to an image with alpha channel
    alpha = mask.convert("L").point(lambda x: 255 if x == 255 else 0, mode="1")

    # Choose the color
    r, g, b = (128, 0, 128)

    color_mask = Image.new("RGBA", mask.size, (r, g, b, int(opacity * 255)))
    mask_rgba = Image.composite(
        color_mask, Image.new("RGBA", mask.size, (0, 0, 0, 0)), alpha
    )

    # Create a new RGBA image to overlay the mask on
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

    # Paste the mask onto the overlay
    overlay.paste(mask_rgba, (0, 0))

    # Create a new image to return by blending the original image and the overlay
    result = Image.alpha_composite(image.convert("RGBA"), overlay)

    # Convert the result back to the original mode and return it
    return result.convert(image.mode)


def pad_to_fixed_size(img, size=(640, 640)):
    width, height = img.size
    # Calculate padding
    left = (size[0] - width) // 2
    top = (size[1] - height) // 2
    right = size[0] - width - left
    bottom = size[1] - height - top

    # Apply padding
    img_padded = F.pad(img, padding=(left, top, right, bottom))
    return img_padded


def resize_image_pil(image_pil, max_size=1024):
    # Ensure image is in RGB
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")

    # Calculate new dimensions preserving aspect ratio
    width, height = image_pil.size
    scale = min(max_size / width, max_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)

    # Calculate padding needed to reach 1024x1024
    pad_width = (max_size - new_width) // 2
    pad_height = (max_size - new_height) // 2

    # Apply padding symmetrically
    image_pil = ImageOps.expand(
        image_pil,
        border=(
            pad_width,
            pad_height,
            max_size - new_width - pad_width,
            max_size - new_height - pad_height,
        ),
        fill=(0, 0, 0),
    )

    return image_pil

# Function for resizing an image to a specific size without changing the aspect ratio
def load_resize_image(image_path: str | Image.Image, size: int) -> Image.Image:
    if isinstance(image_path, str):
        image_pil = load_image(image_path).convert("RGB")
    else:
        image_pil = image_path.convert("RGB")

    image_pil = resize_image_pil(image_pil, size)
    return image_pil