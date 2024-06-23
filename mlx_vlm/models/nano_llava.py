import inspect
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from transformers import AutoConfig
import re

import mlx.core as mx
import mlx.nn as nn

from .base_model.base_model import BaseModel
from .base_model.language_base import LanguageConfig, LanguageBase
from .base_model.vision_base import VisionConfig, VisionBase
from mlx_vlm.core.config_manager import ConfigManager

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
        text_config = LanguageConfig.from_dict(params.get("text_config", {}))
        vision_config = VisionConfig.from_dict(params.get("vision_config", {}))
        return cls(
            text_config=text_config,
            vision_config=vision_config,
            **{k: v for k, v in params.items() if k in inspect.signature(cls).parameters}
        )

class NanoLlava(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        self.init_model_config()
        self.language = self._create_language_model(self.config['text_config'])
        self.vision = self._create_vision_model(self.config['vision_config'])
        self.multimodal_projector = self._create_multimodal_projector()


    def init_model_config(self):
        vision_config = AutoConfig.from_pretrained(self.config["mm_vision_tower"])
        text_config = AutoConfig.from_pretrained(self.config["language_model"])
        vision_config = vision_config.to_dict()
        text_config = text_config.to_dict()
        self.config["vision_config"] = vision_config["vision_config"]
        self.config["text_config"] = text_config
        self._set_vision_config()
        self._set_language_config()

    """
    Initialize the model vision and language configs with the 
    sepecific values applicable to the model
    """
    def _set_vision_config(self):
        self.config['vision_config']['num_hidden_layers']   = 27 
        self.config['vision_config']['hidden_size'] = 1152
        self.config['vision_config']['intermediate_size'] = 4304
        self.config['vision_config']['num_attention_heads'] = 16
        self.config['vision_config']['image_size'] = 384
        self.config['vision_config']['patch_size'] = 14
        self.config['vision_config']['projection_dim'] = 768
        self.config['vision_config']['vocab_size'] = 32000
        self.config['vision_config']['num_channels'] = 3
        self.config['vision_config']['layer_norm_eps'] = 1e-6

    def _set_language_config(self):
        self.config['text_config']['rope_theta'] = 1000000
        self.config['text_config']['rope_traditional'] = False
        self.config['text_config']['rope_scaling'] = None
        self.config['text_config']['tie_word_embeddings'] = True

    def _create_multimodal_projector(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.config['vision_config']['hidden_size'], self.config['text_config']['hidden_size']),
            nn.GELU(),
            nn.Linear(self.config['text_config']['hidden_size'], self.config['text_config']['hidden_size'])
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

        selected_image_feature = hidden_states[-1].astype(pixel_values.dtype)
        assert selected_image_feature.shape[-2] == 729


        image_features = self.multimodal_projector(selected_image_feature)

        final_inputs_embeds = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds, attention_mask

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        num_images, num_image_patches, embed_dim = image_features.shape
        image_positions = mx.where(input_ids[0] == self.config.image_token_index)[0]

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

    def _sanitize_language_weights(self, weights):
        if (
            self.config['text_config']['tie_word_embeddings']
            and "language_model.model.lm_head.weight" not in weights
        ):
            weights["language_model.model.lm_head.weight"] = weights[
                "language_model.model.embed_tokens.weight"
            ]
        return self.language.sanitize(weights)

    def _sanitize_vision_weights(self, weights):
        return self.vision.sanitize(weights)

    def sanitize(self, weights):


        weights = {
            (
                f"{k.split('.', 1)[1]}"
                if re.match(r"^model\.vision_tower", k)
                else (
                    f"mm_projector.linear_1.{k.split('.')[-1]}"
                    if re.match(r"^model\.mm_projector\.0", k)
                    else (
                        f"mm_projector.linear_2.{k.split('.')[-1]}"
                        if re.match(r"^model\.mm_projector\.2", k)
                        else (
                            f"language_model.model.{k}"
                            if re.match(r"^lm_head", k)
                            else (
                                f"language_model.{k}"
                                if re.match(r"^model\.(embed_tokens|norm|layers)", k)
                                else k
                            )
                        )
                    )
                )
            ): v
            for k, v in weights.items()
        }

        weights = {
            (
                f"vision_tower.vision_tower.vision_model.head.attention.in_proj.bias"
                if re.match(
                    r"^vision_tower\.vision_tower\.vision_model\.head\.attention\.in_proj_bias",
                    k,
                )
                else (
                    f"vision_tower.vision_tower.vision_model.head.attention.in_proj.weight"
                    if re.match(
                        r"^vision_tower\.vision_tower\.vision_model\.head\.attention\.in_proj_weight",
                        k,
                    )
                    else k
                )
            ): v
            for k, v in weights.items()
        }
        weights = self._sanitize_vision_weights(weights)
        weights = self._sanitize_language_weights(weights)

        return weights
