import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import math


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_matplotlib(rgb):
    return tuple(c / 255 for c in rgb)


def overlay_mask(image, mask, opacity=0.5, color=(136 / 255, 207 / 255, 249 / 255)):
    """
    Takes in a PIL image and a PIL boolean image mask. Overlay the mask on the image
    and color the mask with the specified color and opacity.

    :param image: PIL Image object
    :param mask: PIL Image object (boolean mask)
    :param opacity: float, opacity of the overlay (0-1)
    :param color: str or tuple, color of the overlay (hex string or RGB tuple with values 0-1 or 0-255)
    :return: PIL Image object
    """
    # Normalize the color
    r, g, b = color

    # Convert the boolean mask to an image with alpha channel
    alpha = mask.convert("L").point(lambda x: 255 if x == 255 else 0, mode="1")

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


def get_colors():
    return [
        "#FF5252",
        "#FF4081",
        "#E040FB",
        "#7C4DFF",
        "#536DFE",
        "#448AFF",
        "#64FFDA",
        "#69F0AE",
        "#EEFF41",
        "#FFAB40",
    ]


def add_padding(image, padding):
    """Add padding to an image."""
    w, h = image.size
    pad = int(padding * min(w, h))
    new_w, new_h = w + 2 * pad, h + 2 * pad
    result = Image.new(image.mode, (new_w, new_h), (0, 0, 0))  # Black background
    result.paste(image, (pad, pad))
    return result


def visualizer(
    image,
    results,
    box_label="box",
    mask_label="mask",
    prompt_label="phrase",
    score_label="score",
    cols=3,
    padding=0.1,  # New parameter for padding
):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if not isinstance(results, list):
        results = [results]

    n = len(results)
    cols = min(cols, n)
    rows = math.ceil(n / cols)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), squeeze=False)
    colors = get_colors()

    for i, result in enumerate(results):
        row, col = divmod(i, cols)
        ax = axes[row, col]

        color_rgb = hex_to_rgb(colors[i % len(colors)])
        color_mpl = rgb_to_matplotlib(color_rgb)
        combined = image.copy()

        if mask_label in result:
            mask = result[mask_label]
            if isinstance(mask, np.ndarray):
                mask = Image.fromarray(mask)
            if mask.size != image.size:
                mask = mask.resize(image.size)
            combined = overlay_mask(combined, mask, opacity=0.6, color=color_rgb)

        # Add padding to the image
        img_with_padding = add_padding(combined, padding)

        ax.imshow(np.array(img_with_padding))
        ax.axis("off")

        if box_label in result:
            x1, y1, x2, y2 = result[box_label]
            # Adjust box coordinates for padding
            pad_pixels = int(padding * min(combined.size))
            rect = patches.Rectangle(
                (x1 + pad_pixels, y1 + pad_pixels),
                x2 - x1,
                y2 - y1,
                linewidth=3,
                edgecolor=color_mpl,
                facecolor="none",
            )
            ax.add_patch(rect)

        metadata = {
            k: v
            for k, v in result.items()
            if k not in [mask_label, box_label, "polygons", "image_index"]
            and isinstance(v, (str, float, int))
        }
        if score := metadata.get(score_label):
            metadata[score_label] = f"{score:.2f}"

        metadata_text = "\n".join(
            f"{key.title()}: {value}" for key, value in sorted(metadata.items())
        )
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

    for i in range(n, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()
