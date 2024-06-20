import inspect
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from language import LanguageBase
from vision import VisionBase


class BaseModel(nn.Module):
    def __init__(self, vision_config, language_config):
        super().__init__()
        self.vision = VisionBase.from_dict(vision_config)
        self.language = LanguageBase.from_dict(language_config)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def from_pretrained(cls, path_or_hf_repo: str):
        raise NotImplementedError("Subclasses must implement this method.")