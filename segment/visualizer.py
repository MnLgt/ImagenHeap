import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from segment.utils import convert_coco_polygons_to_mask
from PIL import Image
import seaborn as sns


def overlay_mask(image, mask, opacity=0.5):
    """
    Takes in a PIL image and a PIL boolean image mask. Overlay the mask on the image
    and color the mask with a low opacity blue with hex #88CFF9.
    """
    # Convert the boolean mask to an image with alpha channel
    alpha = mask.convert("L").point(lambda x: 255 if x == 255 else 0, mode="1")

    # Choose the color
    r, g, b = (128, 0, 128)  # Purple color

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


def visualizer(
    image,
    results,
    box_label="box",
    mask_label="mask",
    prompt_label="phrase",
    score_label="score",
    cols=3,
    **kwargs,
):
    # Ensure image is a PIL Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Ensure results is a list
    if not isinstance(results, list):
        results = [results]

    # Number of results
    n = len(results)

    # If there are fewer images than cols, set cols to n
    cols = min(cols, n)
    rows = (n + cols - 1) // cols

    # Set up the plot with a dark background
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(6 * cols, 6 * rows + 1))
    gs = GridSpec(rows, cols, figure=fig)

    # Use a modern color palette
    colors = sns.color_palette("husl", n_colors=8)

    for i, result in enumerate(results):
        row = i // cols
        col = i % cols

        # Create a copy of the original image
        combined = image.copy()

        # Handle polygon to mask conversion
        if mask_label not in result and "polygons" in result:
            polygons = result["polygons"]
            mask = convert_coco_polygons_to_mask(polygons, image.height, image.width)
            result[mask_label] = Image.fromarray(mask)

        # Draw mask if present
        if mask_label in result:
            mask = result[mask_label]
            if isinstance(mask, np.ndarray):
                mask = Image.fromarray(mask)

            # Ensure mask size matches image size
            if mask.size != image.size:
                print(
                    f"Warning: Mask size {mask.size} doesn't match image size {image.size}. Resizing mask."
                )
                mask = mask.resize(image.size)

            # Apply the overlay
            combined = overlay_mask(combined, mask)

        # Convert to numpy array for matplotlib
        combined_np = np.array(combined)

        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(combined_np)
        ax.axis("off")

        # Draw bounding box if present
        if box_label in result:
            bbox = result[box_label]
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=colors[i % len(colors)],
                facecolor="none",
            )
            ax.add_patch(rect)

        # Add metadata as an inset
        metadata = {
            k: v
            for k, v in result.items()
            if (k not in [mask_label, box_label, "polygons"])
            and (isinstance(v, (str, float, int)))
        }
        if score := metadata.get("score", None):
            metadata["score"] = f"{score:.2f}"

        metadata_text = "\n".join(
            [f"{key.title()}: {value}" for key, value in metadata.items()]
        )

        # Create an inset axes for the metadata
        inset_ax = ax.inset_axes([0.05, 0.05, 0.9, 0.2])
        inset_ax.axis("off")
        inset_ax.text(
            0,
            0,
            metadata_text,
            fontsize=14,
            color="white",
            linespacing=1.5,
            bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", pad=5),
        )

    # Adjust layout and display
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
