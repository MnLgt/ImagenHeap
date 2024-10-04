import random
import warnings
from typing import Any, Dict, List, Union
import yaml

from datasets import Dataset
from imagenheap.components.detect.DetectDino import DetectDino
from imagenheap.components.segment.SegmentSam import SegmentSam
from imagenheap.components.base import Component
from imagenheap.components.component_manager import ComponentManager
from imagenheap.components.data_manager import DataManager
from imagenheap.components.training_manager import TrainingManager
from imagenheap.format_results import ResultFormatter
from imagenheap.utilities.logger_config import get_logger
from imagenheap.visualizer import visualizer

logger = get_logger()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class DotDict(dict):
    """Dot notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ImagenHeap:
    def __init__(self):
        self.data_manager = DataManager()
        self.component_manager = ComponentManager()
        self.training_manager = TrainingManager()
        self.result_formatter = ResultFormatter()
        self.batch_size = None
        self.dataset = None
        self.images = None
        self.processed_results = None
        self.formatted_results = None
        self.component_configs = {}

    def load_config(self, config_path: str):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Register components
        for component in config.get("components", []):
            component_class = globals()[component["name"]]
            self.register_component(component_class())

        # Set pipeline
        if "pipeline" in config:
            self.set_pipeline(config["pipeline"])

        # Load dataset
        if "dataset" in config:
            self.load_dataset(config["dataset"]["path"])

        # Set batch size
        if "batch_size" in config:
            self.set_batch_size(config["batch_size"])

        # Set component arguments
        if "component_args" in config:
            for component, args in config["component_args"].items():
                self.add_args(component).update(args)

    def load_dataset(self, ds: Union[str, Dataset]) -> Dataset:
        self.dataset = self.data_manager.load(ds)
        self.images = self.dataset["image"]
        return self.dataset

    def add_args(self, component_name: str):
        """Get configuration object for a specific component."""
        if component_name not in self.component_configs:
            self.component_configs[component_name] = DotDict()
        return self.component_configs[component_name]

    def get_initial_data(self):
        return {
            "images": self.images,
        }

    def set_batch_size(self, batch_size: int = 8):
        self.component_manager.set_batch_size(batch_size)
        self.batch_size = batch_size

    def format_results(self, **kwargs):
        if self.processed_results is not None:
            self.formatted_results = self.result_formatter.format_all_results(
                self.processed_results, **kwargs
            )

    def process_dataset(self) -> Dict[str, Any]:
        if self.component_manager.batch_size is None:
            self.set_batch_size()

        if self.dataset is None or self.images is None:
            raise ValueError("Dataset not loaded. Please load a dataset first.")

        initial_data = self.get_initial_data()

        self.processed_results = self.component_manager.process(
            initial_data, self.component_configs
        )
        self.format_results()

    def get_formatted_results(self) -> List[Dict[str, Any]]:
        if self.formatted_results is None:
            raise ValueError(
                "No formatted results available. Please process the dataset first."
            )
        return self.formatted_results

    def visualize(self, index: int = None, cols=3, **kwargs):
        if index is None:
            index = random.randint(0, len(self.images) - 1)

        if self.images is None or self.formatted_results is None:
            raise ValueError(
                "Images or results not available. Please load and process the dataset first."
            )

        image = self.images[index]
        result = self.formatted_results[index]

        num_items = len(result)
        if num_items == 0:
            logger.warning("No items detected in the image.")
            return
        else:
            cols = min(cols, num_items)

        visualizer(image, result, cols=cols, **kwargs)

    def train_model(self, model_config: Dict[str, Any]):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Please load a dataset first.")
        self.training_manager.train_model(self.dataset, model_config)

    def push_to_hub(
        self, repo_id: str, token: str, commit_message: str = "md", private: bool = True
    ):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Please load a dataset first.")
        self.data_manager.push_to_hub(repo_id, token, commit_message, private)

    def set_pipeline(self, pipeline: List[str]):
        self.component_manager.set_pipeline(pipeline)

    def register_component(self, component: Component):
        self.component_manager.register_component(component)

    def run(self):
        self.processed_results = None
        self.process_dataset()
        return self.get_formatted_results()


# Usage example
if __name__ == "__main__":
    ih = ImagenHeap()
    ih.load_config("/workspace/SEGMENT/configs/imagenheap/config.yml")
    results = ih.run()
    # Further processing or visualization can be done here
