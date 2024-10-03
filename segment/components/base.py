from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from PIL import Image


class Component(ABC):
    def __init__(self, name: str):
        self.name = name
        self.config: Dict[str, Any] = {}
        self.input_requirements: Dict[str, type] = {}
        self.output_keys: List[str] = []

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data and returns the output data.

        Args:
            data: Input data dictionary.

        Returns:
            Processed data dictionary.
        """
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def unload_model(self):
        pass

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configures the component with the given configuration.

        Args:
            config: Configuration dictionary.
        """
        self.config.update(config)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the current configuration of the component.

        Returns:
            Current configuration dictionary.
        """
        return self.config

    def set_input_requirements(self, requirements: Dict[str, type]) -> None:
        """
        Sets the input requirements for the component.

        Args:
            requirements: Dictionary of input keys and their expected types.
        """
        self.input_requirements = requirements

    def set_output_keys(self, keys: List[str]) -> None:
        """
        Sets the keys for the output data that this component produces.

        Args:
            keys: List of output data keys.
        """
        self.output_keys = keys

    def validate_input(self, data: Dict[str, Any]) -> bool:
        """
        Validates the input data against the component's requirements.

        Args:
            data: Input data dictionary.

        Returns:
            True if input is valid, False otherwise.
        """
        for key, expected_type in self.input_requirements.items():
            if key not in data or not isinstance(data[key], expected_type):
                print(f"Invalid input: {key} should be of type {expected_type}")
                return False
        return True

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        return self.__str__()
