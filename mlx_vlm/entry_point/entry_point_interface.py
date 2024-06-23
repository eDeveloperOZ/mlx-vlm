from abc import ABC, abstractmethod
from huggingface_hub import snapshot_download
from pathlib import Path
from typing import Optional

from mlx_vlm.core.utils import parse_arguments
from mlx_vlm.core.tokenizer import Tokenizer
from mlx_vlm.media_engine.media_engine_interface import MediaEngineInterface
from mlx_vlm.models.base_model.model_interface import ModelInterface     

class EntryPointInterface():
    def __init__(self):
        self.args = parse_arguments()
        self.model_path = self.get_model_path(self.args.model)
        self.tokenizer = Tokenizer(self.model_path)

    @abstractmethod
    def initMediaEngine(self) -> MediaEngineInterface:
        pass

    @abstractmethod
    def initModel(self) -> ModelInterface:
        pass

    @abstractmethod
    def execute(self):
        pass

    def get_model_path(self, model_name: str, revision: Optional[str] = None) -> Path:
        """
        Ensures the model is available locally. If the path does not exist locally,
        it is downloaded from the Hugging Face Hub.

        Args:
            path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
            revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

        Returns:
            Path: The path to the model.
        """
        model_path = Path(model_name)
        if not model_path.exists():
            model_path = Path(
                snapshot_download(
                    repo_id=model_name,
                    revision=revision,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                        "*.txt",
                    ],
                )
                )
        return model_path
