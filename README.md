### Install
```
chmod +x ./install.sh
./install.sh
```

### Download the dataset 
- Make sure to have your hf key in the env

```
python segment/download_dataset.py
```

### <h1>Segment With Text Prompt</h1>

- See example notebook for more details


### Project Structure ###

The purpose of this module: 
- Get all of the segmentation masks for the images in the dataset based on text prompts 
- Train a yolo model on the segmentation masks 
- Push the dataset and the model to the hub 


### Yolo To Do ###
Build tools for dataset quality control: 
- bar chart of class distribution
- Add a filter for the dataset to filter out images below a certain score 