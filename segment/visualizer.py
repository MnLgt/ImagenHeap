import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from segment.utils import convert_coco_polygons_to_mask
from PIL import Image


def visualizer(
    image,
    results,
    box_label="box",
    mask_label="mask",
    prompt_label="phrase",
    score_label="score",
    cols=4,
    **kwargs,
):
    # Convert PIL Image to numpy array
    image_np = np.array(image)

    # Check image dimensions
    if image_np.ndim != 3:
        raise ValueError("Image must be a 3-dimensional array")

    # Number of results
    n = len(results)

    # If there are fewer images than cols, set cols to n
    cols = cols if cols <= n else n

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
        label = result[prompt_label]
        score = float(result[score_label])

        row = i // cols
        col = i % cols

        # Create a copy of the original image
        combined = image_np.copy()
        if mask_label not in result and "polygons" in result:
            polygons = result["polygons"]
            mask = convert_coco_polygons_to_mask(polygons, 1024, 1024)
            mask_image = Image.fromarray(mask)
            result[mask_label] = mask_image

        # Draw mask if present
        if mask_label in result:
            mask = result[mask_label]
            # Convert PIL mask to numpy array
            mask_np = np.array(mask)

            # Check mask dimensions
            if mask_np.ndim != 2:
                raise ValueError("Mask must be a 2-dimensional array")

            # Create an overlay where mask is True
            overlay = np.zeros_like(image_np)
            overlay[mask_np > 0] = [0, 0, 255]  # Applying blue color on the mask area

            # Combine the image and the overlay
            indices = np.where(mask_np > 0)
            combined[indices] = combined[indices] * 0.5 + overlay[indices] * 0.5

        # Show the combined image
        ax = axs[row, col]
        ax.imshow(combined)
        ax.axis("off")
        ax.set_title(f"Label: {label}, Score: {score:.2f}", fontsize=12)

        # Draw bounding box if present
        if box_label in result:
            bbox = result[box_label]
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="r", facecolor="none"
            )
            ax.add_patch(rect)

    # Hide unused subplots if the total number of results is not a multiple of cols
    for idx in range(i + 1, rows * cols):
        row = idx // cols
        col = idx % cols
        axs[row, col].axis("off")

    plt.tight_layout()
    plt.show()
