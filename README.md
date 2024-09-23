# Install
```
chmod +x ./install.sh
./install.sh
```

# Download the dataset 
- Make sure to have your hf key in the env

```
python segment/download_dataset.py
```

### Segment With Text Prompt 

- See example notebook for more details


### Project Structure ###

THe purpose of this module:
    - Get the segmentation masks from images using text prompts 
    - Create a dataset of the masks and the corresponding bounding boxes
    - Train a yolo model on the masks
    - Push the model and dataset to the hub

CONFIG
The config directory includes the config files in yolo format.  

They should include paths to the train and val images dir 
A dictionary of label_ids and labels
Colors - Optional - A list of the colors to use for each corresponding label

PARSE - Get the masks and boxes for those labels 

CONVERT - Make a dataset 

PUSH_TO_HUB - push dataset to hub PUSH_TO_HUB

CONVERT - Convert the dataset to yolo format 

TRAIN - Train yolo model 

PUSH_TO_HUB - Push yolo model to hub

