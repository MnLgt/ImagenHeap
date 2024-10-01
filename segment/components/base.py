from abc import ABC, abstractmethod
from typing import Any


class Component(ABC):
    @abstractmethod
    def process(self, data: Any = None) -> Any:
        """
        Processes the input data and returns the output data.

        Args:
            data: Input data.

        Returns:
            Processed data.
        """
        pass
