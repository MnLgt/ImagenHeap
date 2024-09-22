import os
from PIL import Image, ImageOps
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import load_dataset
from utils import resize_image_pil
import io
import numpy as np


def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """
    Merge multi segments to one list. Find the coordinates with min distance between each segment, then connect these
    coordinates with one thin line to merge all segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


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
    label_scores = {}  # Dictionary to keep track of the best scores per label

    for row in md:
        label = row.get("label")
        
        # don't include person since it overlaps with other masks
        if label != 'person':
            label_id = row.get("label_id")
    
            coco_polygons = row.get("polygons")
            score = row.get("score", 0)  # Default score
    
            # Normalize and reshape polygon coordinates
            if len(coco_polygons) > 1:
                yolo_polygons = merge_multi_segment(coco_polygons)
                yolo_polygons = (
                    (
                        np.concatenate(yolo_polygons, axis=0)
                        / np.array([image_width, image_height])
                    )
                    .reshape(-1)
                    .tolist()
                )
            else:
                yolo_polygons = [j for i in coco_polygons for j in i]
                yolo_polygons = (
                    (
                        np.array(yolo_polygons).reshape(-1, 2)
                        / np.array([image_width, image_height])
                    )
                    .reshape(-1)
                    .tolist()
                )
    
            yolo_polygons_str = " ".join([str(coord) for coord in yolo_polygons])
            yolo_line = f"{label_id} {yolo_polygons_str}"
            
            lines.append(yolo_line)

    return lines


def write_image_and_text_file(image, image_name, lines, output_dir):
    # Save images as jpegs
    image_uuid = image_name.split(".")[0]
    image_name = f"{image_uuid}.jpg"
    image_path = os.path.join(output_dir, image_name)

    text_name = f"{image_uuid}.txt"
    text_output_dir = output_dir.replace("images", "labels")
    text_path = os.path.join(text_output_dir, text_name)

    image.save(image_path)

    with open(text_path, "w") as f:
        f.write("\n".join(lines))


def format_and_write(row, output_dir):
    try:
        image = row.get("image")
        image = resize_image_pil(image)

        md = row.get("mask_metadata")
        if md:
            image_name = row["image_id"]
            lines = get_lines(md, image.width, image.height)
            write_image_and_text_file(image, image_name, lines, output_dir)
    except Exception as e:
        print(f"Error processing {row.get('image_id', 'Unknown ID')}: {str(e)}")


def check_directory_contents(image_dir, label_dir):
    # Create sets of all image and label filenames
    image_files = {f for f in os.listdir(image_dir) if f.endswith(".jpg")}
    label_files = {f for f in os.listdir(label_dir) if f.endswith(".txt")}

    # Convert image filenames to the expected label filenames
    expected_label_files = {image.replace(".jpg", ".txt") for image in image_files}

    # Find mismatches between expected and actual label files
    missing_labels = expected_label_files - label_files
    extra_labels = label_files - expected_label_files

    # Report findings
    if not missing_labels and not extra_labels:
        print(
            f"All checks passed for {image_dir} and {label_dir}: Every image has a corresponding label."
        )
    else:
        if missing_labels:
            print(f"Missing label files in {label_dir}: {missing_labels}")
        if extra_labels:
            print(
                f"Extra label files in {label_dir} not corresponding to images: {extra_labels}"
            )


def check_all_directories(train_image_dir, val_image_dir):
    train_label_dir = train_image_dir.replace("images", "labels")
    val_label_dir = val_image_dir.replace("images", "labels")

    print("Checking training data...")
    check_directory_contents(train_image_dir, train_label_dir)

    print("Checking validation data...")
    check_directory_contents(val_image_dir, val_label_dir)


def main():
    repo_id = "jordandavis/fashion_people_detections"
    parent_dir = os.path.join(os.path.dirname(__file__),"..","datasets/fashion_people_detection")
    workers = os.cpu_count()

    # Load Dataset
    ds = load_dataset(
        repo_id, split="train", trust_remote_code=True, num_proc=workers, cache_dir='hf_cache'
    )

    # Split
    ds = ds.train_test_split(train_size=0.9)
    train = ds["train"]
    val = ds["test"]

    # Make directories
    train_dir, val_dir = make_yolo_dirs(parent_dir)

    # Parallel processing
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []

        # Process training data
        for row in tqdm(train, desc="Processing Training Data"):
            futures.append(executor.submit(format_and_write, row, train_dir))

        # Wait for all training futures to complete
        for future in as_completed(futures):
            # Optionally handle or log errors from futures here
            if future.exception():
                print("Error in processing a training item:", future.exception())

        # Clear the list of futures for validation processing
        futures = []

        # Process validation data
        for row in tqdm(val, desc="Processing Validation Data"):
            futures.append(executor.submit(format_and_write, row, val_dir))

        # Wait for all validation futures to complete
        for future in as_completed(futures):
            # Optionally handle or log errors from futures here
            if future.exception():
                print("Error in processing a validation item:", future.exception())

    # Check directories
    check_all_directories(train_dir, val_dir)


if __name__ == "__main__":
    main()
    