import inspect
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class VisionBase(nn.Module):
    def __init__(
        self,
        model_type: str,
        num_hidden_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        image_size: int,
        patch_size: int,
        num_channels: int = 3,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.model_type = model_type
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.layer_norm_eps = layer_norm_eps

        self.embed_dim = hidden_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_positions = self.num_patches

        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        self.layers = nn.Sequential(
            *[self._make_encoder_layer() for _ in range(num_hidden_layers)]
        )
        self.post_layernorm = nn.LayerNorm(hidden_size)

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def _make_encoder_layer(self):
        return nn.Sequential(
            nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps),
            self._make_attention(),
            nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps),
            self._make_mlp(),
        )

    def _make_attention(self):
        return nn.MultiHeadAttention(
            self.hidden_size,
            num_heads=self.num_attention_heads,
            bias=True,
        )

    def _make_mlp(self):
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.intermediate_size),
            nn.GELU(approx="fast"),
            nn.Linear(self.intermediate_size, self.hidden_size),
        )

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        batch_size = x.shape[0]
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        position_ids = mx.array(np.arange(self.num_positions)[None, :])
        embeddings = patch_embeddings + self.position_embedding(position_ids)

        encoder_states = (embeddings,) if output_hidden_states else None
        for layer in self.layers:
            embeddings = layer(embeddings)
            if output_hidden_states:
                encoder_states = encoder_states + (embeddings,)

        pooler_output = self.post_layernorm(embeddings[:, 0, :])
        return pooler_output, embeddings, encoder_states

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                continue
            elif "patch_embedding.weight" in k:
                if self._check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[k] = v
        return sanitized_weights

    def _check_array_shape(self, arr):
        shape = arr.shape
        if len(shape) != 4:
            return False
        out_channels, kH, KW, _ = shape
        return (out_channels >= kH) and (out_channels >= KW) and (kH == KW)