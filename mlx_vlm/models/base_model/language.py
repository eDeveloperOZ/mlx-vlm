import inspect
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class LanguageBase(nn.Module):
    def __init__(
        self,
        model_type: str,
        vocab_size: int,
        max_position_embeddings: int,
        pad_token_id: int,
        num_hidden_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.layer_norm_eps = layer_norm_eps

        self.embed_dim = hidden_size

        self.word_embedding = nn.Embedding(vocab_size, self.embed_dim, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_position_embeddings, self.embed_dim)

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
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        batch_size, seq_length = input_ids.shape
        position_ids = mx.array(np.arange(seq_length)[None, :])
        embeddings = self.word_embedding(input_ids) + self.position_embedding(position_ids)

        encoder_states = (embeddings,) if output_hidden_states else None
        for layer in self.layers:
            embeddings = layer(embeddings, attention_mask)
            if output_hidden_states:
                encoder_states = encoder_states + (embeddings,)

        pooler_output = self.post_layernorm(embeddings[:, 0, :])
        return pooler_output, embeddings, encoder_states

    def sanitize(self, weights):
        return weights