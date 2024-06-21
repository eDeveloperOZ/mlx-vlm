import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from .base_model.base_model import BaseModel
from .base_model.language_base import LanguageBase, LanguageConfig
from .vision import VisionModel, VisionConfig

class PaLIGEMMAMultiModalProjector(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.linear = nn.Linear(
            config['vision_config']['hidden_size'],
            config['vision_config']['projection_dim'],
            bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)

class PaLIGEMMA(BaseModel):
    def _create_language_model(self, config: LanguageConfig) -> LanguageBase:
        return LanguageBase(config)

    def _create_vision_model(self, config: Dict[str, Any]) -> nn.Module:
        return VisionModel(VisionConfig.from_dict(config))

    def _create_multimodal_projector(self, config: Dict[str, Any]) -> nn.Module:
        return PaLIGEMMAMultiModalProjector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        if pixel_values is None:
            return self.language_model(input_ids)

        inputs_embeds = self.language_model.embed_tokens(input_ids)

        hidden_state, _, _ = self.vision_model(
            pixel_values.transpose(0, 2, 3, 1).astype(inputs_embeds.dtype),
            output_hidden_states=True,
        )

        image_features = hidden_state[None, :].astype(pixel_values.dtype)
        image_features = self.multimodal_projector(image_features)

        final_inputs_embeds, final_attention_mask_4d = self._prepare_inputs_for_multimodal(
            image_features, inputs_embeds, input_ids, attention_mask
        )
        return final_inputs_embeds, final_attention_mask_4d

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> Tuple[mx.array, Any]:
        input_embeddings, final_attention_mask_4d = self.get_input_embeddings(
            input_ids, pixel_values, attention_mask
        )

        logits, cache = self.language_model(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=input_embeddings,
            mask=final_attention_mask_4d,
        )
        return logits, cache

    def _prepare_inputs_for_multimodal(
        self, image_features, inputs_embeds, input_ids, attention_mask
    ):
        # Implementation similar to the original PaLI-GEMMA model
        # ...

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        # Implementation similar to the original PaLI-GEMMA model
        # ...

    @classmethod
    def from_pretrained(cls, path_or_hf_repo: str) -> 'PaLIGEMMA':
        path = Path(path_or_hf_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                    ],
                )
            )

        with open(path / "config.json", "r") as f:
            config = json.load(f)

        model = cls(config)
        weights = {}
        for weight_file in path.glob("*.safetensors"):
            weights.update(mx.load(str(weight_file)))

        sanitized_weights = model.sanitize(weights)
        model.load_weights(list(sanitized_weights.items()))
        return model