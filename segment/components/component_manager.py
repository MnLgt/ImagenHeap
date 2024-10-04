import torch
from typing import Dict, Any, List, Callable, Set
from tqdm.auto import tqdm
from segment.components.base import Component
from utilities.logger_config import get_logger

logger = get_logger()


class ComponentManager:
    def __init__(self):
        self.components: Dict[str, Component] = {}
        self.pipeline: List[str] = []
        self.loaded_components: Set[str] = set()
        self.batch_size: int = None

    def register_component(self, component: Component) -> None:
        self.components[component.name] = component

    def get_component(self, name: str) -> Component:
        return self.components.get(name)

    def set_pipeline(self, pipeline: List[str]) -> None:
        self.pipeline = pipeline

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = max(1, batch_size)  # Ensure batch size is at least 1

    def create_batches(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        num_items = len(next(iter(data.values())))
        return [
            {k: v[i : i + self.batch_size] for k, v in data.items()}
            for i in range(0, num_items, self.batch_size)
        ]

    def merge_batch_results(
        self, batch_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        merged = {k: [] for k in batch_results[0].keys()}
        for batch in batch_results:
            for k, v in batch.items():
                merged[k].extend(v)
        return merged

    def validate_pipeline(self) -> bool:
        for i in range(len(self.pipeline) - 1):
            current_component = self.get_component(self.pipeline[i])
            next_component = self.get_component(self.pipeline[i + 1])

            if not all(
                key in current_component.output_keys
                for key in next_component.input_requirements
            ):
                logger.error(
                    f"Invalid pipeline: {current_component.name} does not produce all required inputs for {next_component.name}"
                )
                return False
        return True

    def process_component(
        self,
        component: Component,
        data: Dict[str, Any],
        component_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:

        # load the component model
        component.load_model()

        batches = self.create_batches(data)
        total_items = len(next(iter(data.values())))
        batch_results = []

        desc = f"Using {component.name.title()}"
        with tqdm(total=total_items, position=1, leave=True, desc=desc) as batch_pbar:
            for batch in batches:
                if component_kwargs:
                    batch.update(component_kwargs)

                batch_result = component.process(batch)
                batch_results.append(batch_result)

                num_items = len(next(iter(batch.values())))
                batch_pbar.update(num_items)

        component.unload_model()
        return self.merge_batch_results(batch_results)

    def process(
        self, initial_data: Dict[str, Any], component_kwargs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not self.validate_pipeline():
            raise ValueError("Invalid pipeline configuration")

        data = initial_data
        num_components = len(self.pipeline)

        with tqdm(total=num_components, desc="Overall Progress") as pbar:
            for component_name in self.pipeline:
                component = self.get_component(component_name)
                component_config = component_kwargs.get(component_name, {})

                data = self.process_component(component, data, component_config)
                pbar.update(1)

        return data
