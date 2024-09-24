# %%
import sys

sys.path.append("..")

import os 
from create_dataset import CreateSegmentationDataset, get_labels_dict
from segment.utils import get_device
from datasets import load_dataset
from create_dataset import filter_list_in_column

# disable datasets.map progress bar
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
# The number of images loaded on each batch
batch_size = 512

# The number of images processed on the gpu
sub_batch_size = 8

# The number of workers for the dataloader
num_workers = os.cpu_count()

# The directory for hugging face cache
cache_dir = "hf_cache"

# The image dataset ID and split to load
dataset_id = "MnLgt/fashion_num_people"
split = "train"

# The labels from the yolo config file
config_path = "configs/fashion_people_detection_no_person.yml"

# Thresholds for dino and sam
box_threshold = 0.3
iou_threshold = 0.8
text_threshold = 0.35

# %%
ds = load_dataset(
    dataset_id,
    split=split,
    trust_remote_code=True,
    cache_dir=cache_dir,
    streaming=False,
    num_proc=num_workers,
)
print(f"Total Rows: {ds.num_rows}")

# %%

# Filter for images with one person
a = ds['num_people']
b = [i for i, x in enumerate(a) if x > 0]
c = ds.select(b)
ds = c
print(f"Total Rows After Filtering: {ds.num_rows}")



# # %%
# p = CreateSegmentationDataset(ds, config_path, bs=batch_size, sub_bs=sub_batch_size, box_threshold=box_threshold, text_threshold=text_threshold, iou_threshold=iou_threshold)

# # %%
# p.process()

# # %%
# # remove any items from the md lists that are empty
# def remove_none(item):
#     return bool(item)

# p.processed_ds = filter_list_in_column(p.processed_ds, 'metadata', remove_none)

# # filter items with a score
# p.filter_scores(score_cutoff=0.8)

# # %%
# p.processed_ds

# # %%



# # %%
# p.check_results()

# # %%


# %%



