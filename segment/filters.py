from typing import Callable

from datasets import Dataset

from segment.utilities.logger_config import get_logger

from typing import Callable

from datasets import Dataset

logger = get_logger()


def score_filter(item: dict, score_cutoff: float = 0.5) -> bool:
    """
    Filter an item based on its score.

    :param item: The dictionary item to filter
    :param score_cutoff: The minimum score to keep an item
    :return: True if the item's score is above the cutoff, False otherwise
    """
    return item.get("score", 0) > score_cutoff


def remove_empty_items(item: dict, column_name: str) -> bool:
    """
    Filter out items with empty values in a specified column.

    :param item: The dictionary item to filter
    :param column_name: The name of the column to check for emptiness
    :return: True if the item's specified column is not empty, False otherwise
    """
    return bool(item[column_name])


class Filter:
    @staticmethod
    def filter_rows_of_dataset(dataset: Dataset, filter_function: Callable) -> Dataset:
        """
        Apply a filter function to the rows of a dataset.

        :param dataset: The input dataset
        :param filter_function: A function that takes a dictionary and returns True to keep it, False to filter it out
        :return: A new dataset with filtered rows
        """
        return dataset.filter(filter_function)

    @staticmethod
    def filter_list_in_column(
        dataset: Dataset, column_name: str, filter_function: Callable
    ) -> Dataset:
        """
        Apply a filter function to a column containing lists of dictionaries.

        :param dataset: The input dataset
        :param column_name: The name of the column containing lists to filter
        :param filter_function: A function that takes a dictionary and returns True to keep it, False to filter it out
        :return: A new dataset with filtered lists in the specified column
        """

        def transform_fn(example):
            filtered_list = [
                item for item in example[column_name] if filter_function(item)
            ]
            return {column_name: filtered_list}

        return dataset.map(transform_fn)
