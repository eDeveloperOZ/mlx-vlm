from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import glob
import json

import mlx.core as mx
import mlx.nn as nn

from .language_base import LanguageBase, LanguageConfig
from .vision_base import VisionBase, VisionConfig
from .model_interface import ModelInterface
from ...core.error_handler import ErrorHandler
from ...core.logger import Logger
from ...core.config_manager import ConfigManager


logger = Logger()

class BaseModel(ModelInterface, ABC):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config['vision_config'] = {}
        self.config['text_config'] = {}
        self.config['vision_config'] = VisionConfig.from_dict(self.config['vision_config'])
        self.config['text_config'] = LanguageConfig.from_dict(self.config['text_config'])
        

    @abstractmethod
    def init_model_config(self):
        raise NotImplementedError("Subclass must implement this method")

    @abstractmethod
    def _create_language_model(self, config: LanguageConfig) -> LanguageBase:
        return LanguageBase(self.config['text_config'])
    
    @abstractmethod
    def _set_language_config(self):
        raise NotImplementedError("Subclass must implement this method")

    @abstractmethod
    def _create_vision_model(self, config: VisionConfig) -> VisionBase:
        return VisionBase(self.config['vision_config'])
    
    @abstractmethod
    def _set_vision_config(self):
        raise NotImplementedError("Subclass must implement this method")


    @abstractmethod
    def _create_multimodal_projector(self) -> nn.Module:
        # This method creates a multimodal projector that transforms the vision model's hidden states
    # to match the text model's hidden size, enabling the integration of visual and textual features.
        raise NotImplementedError("Subclass must implement this method")

    @abstractmethod
    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        pass

    @abstractmethod
    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> Tuple[mx.array, Any]:
        pass

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            input_ids = input_data.get('input_ids')
            pixel_values = input_data.get('pixel_values')
            attention_mask = input_data.get('attention_mask')
            cache = input_data.get('cache')

            logits, new_cache = self(input_ids, pixel_values, attention_mask, cache)
            return {'logits': logits, 'cache': new_cache}
        except Exception as e:
            self.error_handler.handle_exception(e, "Error processing input in BaseModel")
    @abstractmethod
    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        raise NotImplementedError("Subclass must implement this method")

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_path: str) -> 'BaseModel':
        config = ConfigManager(model_path).get_config()
        model = cls(config)
        weights = model._load_weights(model_path)
        sanitized_weights = model.sanitize(weights)
        model.load_weights(file_or_weights=sanitized_weights) 
        return model

    @staticmethod
    def _load_weights(model_path: Path) -> Dict[str, mx.array]:
        weight_files = glob.glob(str(model_path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {model_path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))
        return weights
        