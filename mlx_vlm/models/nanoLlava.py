import inspect
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from transformers import AutoConfig

import mlx.core as mx
import mlx.nn as nn

from .base_model.base_model import BaseModel
from .base_model.language_base import LanguageConfig, LanguageBase
from .base_model.vision_base import VisionConfig, VisionBase
from mlx_vlm.core.config_manager import ConfigManager

class LlavaMultiModalProjector(nn.Module):
    def __init__(self, vision_hidden_size: int, text_hidden_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(vision_hidden_size, text_hidden_size, bias=True)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(text_hidden_size, text_hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x

    
@dataclass
class ModelConfig:
    text_config: LanguageConfig
    vision_config: VisionConfig
    model_type: str
    auto_map: dict
    hidden_size: int
    mm_hidden_size: int
    mm_vision_tower: str
    mm_projector_type: str = "mlp2x_gelu"
    ignore_index: int = -100
    image_token_index: int = -200
    vocab_size: int = 151936

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Model(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config.vision_config, config.text_config)
        self.image_token_index = -200  # Adjust as needed
        self.init_model_config()

    """
    Check the LanguageConfig and VisionConfig values of the base model and 
    update the values with the appropriate values for the ModelConfig
    """
    def init_model_config(self):
        ModelConfig.vision_config.num_hidden_layers = 27 
        ModelConfig.vision_config.hidden_size = 1152
        ModelConfig.vision_config.intermediate_size = 4304
        ModelConfig.vision_config.num_attention_heads = 16
        ModelConfig.vision_config.image_size = 384
        ModelConfig.vision_config.patch_size = 14
        ModelConfig.vision_config.projection_dim = 768
        ModelConfig.vision_config.vocab_size = 32000
        ModelConfig.vision_config.num_channels = 3
        ModelConfig.vision_config.layer_norm_eps = 1e-6

    def _create_multimodal_projector(self) -> nn.Module:
        return LlavaMultiModalProjector(
            self.vision.config.hidden_size,
            self.language.config.hidden_size
        )

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        if pixel_values is None:
            return self.language(input_ids)

        inputs_embeds = self.language.embed_tokens(input_ids)

        _, hidden_states = self.vision(
            pixel_values.transpose(0, 2, 3, 1),
            output_hidden_states=True
        )

        selected_image_feature = hidden_states[-2]
        selected_image_feature = selected_image_feature[:, 1:]

        image_features = self.multimodal_projector(selected_image_feature)

        final_inputs_embeds = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds, attention_mask

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        num_images, num_image_patches, embed_dim = image_features.shape
        image_positions = mx.where(input_ids[0] == self.image_token_index)[0]

        if len(image_positions) != num_images:
            raise ValueError(
                f"The number of image tokens ({len(image_positions)}) does not "
                f"match the number of image inputs ({num_images})."
            )

        text_segments = []
        start_idx = 0

        for position in image_positions:
            text_segments.append(inputs_embeds[:, start_idx:position])
            start_idx = position + 1

        image_embeddings = mx.split(image_features, image_features.shape[0])
        final_embeddings = [v for p in zip(text_segments, image_embeddings) for v in p]
        final_embeddings += [inputs_embeds[:, start_idx:]]

        return mx.concatenate(final_embeddings, axis=1)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> Tuple[mx.array, Any]:
        input_embeddings, attention_mask = self.get_input_embeddings(input_ids, pixel_values, attention_mask)
        logits, cache = self.language(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=input_embeddings,
            mask=attention_mask
        )
        return logits, cache

    @classmethod
    def from_pretrained(cls, model_path):
        config = ConfigManager(model_path).get_config()
        vision_config = AutoConfig.from_pretrained(config["mm_vision_tower"])
        text_config = AutoConfig.from_pretrained(config["language_model"])
        vision_config = vision_config.to_dict()
        text_config = text_config.to_dict()
        config["vision_config"] = vision_config["vision_config"]
        config["text_config"] = text_config

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        weights
