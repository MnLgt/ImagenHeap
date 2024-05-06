
import os
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import load_dataset
from utils import convert_coco_to_yolo_polygons
import io 


def convert_coco_to_yolo_polygons(coco_polygons, image_width, image_height):
    """
    Converts COCO style polygons to a normalized YOLO style format, outputting a single list
    of normalized coordinates representing the perimeter of the segmentation.

    Parameters:
    - coco_polygons: List of polygons, each represented as a flat list of points (x1, y1, x2, y2, ..., xn, yn).
    - image_width: The width of the original image.
    - image_height: The height of the original image.

    Returns:
    - Single list of normalized coordinates in the format [x1, y1, x2, y2, ..., xn, yn] for all polygons concatenated.
    """
    yolo_coordinates = []
    for polygon in coco_polygons:
        for i in range(0, len(polygon), 2):
            x_normalized = polygon[i] / image_width
            y_normalized = polygon[i + 1] / image_height
            yolo_coordinates.extend([x_normalized, y_normalized])

    return yolo_coordinates

# Make the train and val directories for images and labels 
def make_yolo_dirs(parent_dir):
    train_dir = os.path.join(parent_dir, "images", "train")
    train_labels = os.path.join(parent_dir, "labels", "train")

    val_dir = os.path.join(parent_dir, "images", "val")
    val_labels = os.path.join(parent_dir, "labels", "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)
    os.makedirs(val_labels, exist_ok=True)
    return train_dir, val_dir

def get_lines(md, image_width, image_height):
    lines = []
    for row in md:
        label = row.get("label")
        label_id = row.get("label_id")
        coco_polygons = row.get("polygons")
        yolo_polygons = convert_coco_to_yolo_polygons(coco_polygons, image_width, image_height)
        yolo_polygons_str = " ".join([str(coord) for coord in yolo_polygons])
        yolo_line = f"{label_id} {yolo_polygons_str}"
        lines.append(yolo_line)
    return lines

def write_image_and_text_file(image, image_name, lines, output_dir):
    # Save images as jpegs
    image_uuid = image_name.split(".")[0]
    image_name = f"{image_uuid}.jpg"
    image_path = os.path.join(output_dir,image_name)

    text_name = f"{image_uuid}.txt"
    text_output_dir = output_dir.replace("images", "labels")
    text_path = os.path.join(text_output_dir, text_name)
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(image_path)
    with open(text_path, "w") as f:
        f.write("\n".join(lines))

def format_and_write(row, output_dir):
    try:
        image = row.get('image')
        md = row.get('mask_metadata')
        if md:
            image_name = row['image_id']
            lines = get_lines(md, image.width, image.height)
            write_image_and_text_file(image, image_name, lines, output_dir)
    except Exception as e:
        print(f"Error processing {row.get('image_id', 'Unknown ID')}: {str(e)}")

def main():
    repo_id = "jordandavis/fashion_people_detections"
    parent_dir = "datasets/fashion_people_detection"
    workers = os.cpu_count()

    # Load Dataset
    ds = load_dataset(repo_id, split='train', trust_remote_code=True, num_proc=workers)

    # Split
    ds = ds.train_test_split(train_size=0.9)
    train = ds["train"]
    val = ds["test"]

    # Make directories
    train_dir, val_dir = make_yolo_dirs(parent_dir)

    # Parallel processing
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for row in tqdm(train):
            futures.append(executor.submit(format_and_write, row, train_dir))
        for row in tqdm(val):
            futures.append(executor.submit(format_and_write, row, val_dir))
        
        # Wait for all futures to complete
        for future in as_completed(futures):
            pass

if __name__ == "__main__":
    main()
