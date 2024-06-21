from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any

import mlx.core as mx
import mlx.nn as nn

from .language_base import LanguageBase, LanguageConfig
from .model_interface import ModelInterface
from ...core.error_handler import ErrorHandler
from ...core.logger import Logger


logger = Logger()

class BaseModel(ModelInterface, ABC):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.language_model = self._create_language_model(config['text_config'])
        self.vision_model = self._create_vision_model(config['vision_config'])
        self.multimodal_projector = self._create_multimodal_projector(config)

    @abstractmethod
    def init_model_config(self):
        raise NotImplementedError("Subclass must implement this method")

    @abstractmethod
    def _create_language_model(self, config: LanguageConfig) -> LanguageBase:
        pass

    @abstractmethod
    def _create_vision_model(self, config: Dict[str, Any]) -> nn.Module:
        pass

    @abstractmethod
    def _create_multimodal_projector(self, config: Dict[str, Any]) -> nn.Module:
        pass

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

    def load_weights(self, weights: Dict[str, mx.array]) -> None:
        try:
            sanitized_weights = self.sanitize(weights)
            self.load_weights(list(sanitized_weights.items()))
        except Exception as e:
            self.error_handler.handle_exception(e, "Error loading weights in BaseModel")

    @abstractmethod
    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path_or_hf_repo: str) -> 'BaseModel':
        pass