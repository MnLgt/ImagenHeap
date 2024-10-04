from typing import Union
from datasets import Dataset
from huggingface_hub import create_repo
from datasets import load_dataset
import os


class DataManager:
    def load(
        self,
        dataset: Union[str, Dataset],
        split="train",
        num_proc=os.cpu_count(),
        **kwargs,
    ) -> Dataset:
        # Placeholder for loading dataset
        if isinstance(dataset, str):
            print(f"Loading dataset: {dataset}")
            return load_dataset(
                dataset,
                trust_remote_code=True,
                split=split,
                num_proc=num_proc,
                **kwargs,
            )
        else:
            return dataset

    def push_to_hub(self, repo_id, token, commit_message="md", private=True):
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=private,
            token=token,
        )

        self.ds.push_to_hub(repo_id, commit_message=commit_message, token=token)

        print(f"Pushed Dataset to Hub: {repo_id}")
