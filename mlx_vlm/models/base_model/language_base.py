from abc import ABC
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

@dataclass
class LanguageConfig:
    model_type: str = "default"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    max_position_embeddings: int = 512
    vocab_size: int = 30522
    num_key_value_heads: Optional[int] = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000
    rope_traditional: bool = False
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 0

    @classmethod
    def from_dict(cls, params: dict):
        return cls(**{k: v for k, v in params.items() if k in cls.__annotations__})

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

class LanguageBase(ABC):
    def __init__(self, config: LanguageConfig):
        self.config = LanguageConfig.from_dict(config)
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.position_embedding = nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size)
        
        self.layers = [self._create_transformer_block() for _ in range(self.config.num_hidden_layers)]
        
        self.norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def _create_transformer_block(self):
        return {
            'self_attn': self._create_attention(),
            'input_layernorm': nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps),
            'post_attention_layernorm': nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps),
            'mlp': self._create_mlp(),
        }

    def _create_attention(self):
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        return {
            'num_heads': self.config.num_attention_heads,
            'num_kv_heads': self.config.num_key_value_heads,
            'scale': head_dim ** -0.5,
            'q_proj': nn.Linear(self.config.hidden_size, self.config.num_attention_heads * head_dim, bias=True),
            'k_proj': nn.Linear(self.config.hidden_size, self.config.num_key_value_heads * head_dim, bias=True),
            'v_proj': nn.Linear(self.config.hidden_size, self.config.num_key_value_heads * head_dim, bias=True),
            'o_proj': nn.Linear(self.config.num_attention_heads * head_dim, self.config.hidden_size, bias=False),
            'rope': nn.RoPE(head_dim, traditional=self.config.rope_traditional, base=self.config.rope_theta),
        }

    def _create_mlp(self):
        return {
            'gate_proj': nn.Linear(self.config.hidden_size, self.config.intermediate_size, bias=False),
            'down_proj': nn.Linear(self.config.intermediate_size, self.config.hidden_size, bias=False),
            'up_proj': nn.Linear(self.config.hidden_size, self.config.intermediate_size, bias=False),
        }

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        x = self._embed(input_ids, position_ids)
        x, cache = self._encode(x, attention_mask, cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, cache

    def _embed(self, input_ids: mx.array, position_ids: Optional[mx.array] = None) -> mx.array:
        x = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = mx.arange(input_ids.shape[1])[None, :]
        x = x + self.position_embedding(position_ids)
        return x

    def _encode(
        self, 
        x: mx.array, 
        attention_mask: Optional[mx.array], 
        cache: Optional[Tuple[mx.array, mx.array]]
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        if attention_mask is None and x.shape[1] > 1:
            attention_mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            attention_mask = attention_mask.astype(x.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            x, cache[i] = self._transformer_block_forward(x, layer, attention_mask, cache[i])

        return x, cache

    def _transformer_block_forward(
        self,
        x: mx.array,
        layer: dict,
        attention_mask: Optional[mx.array],
        cache: Optional[Tuple[mx.array, mx.array]],
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        r, cache = self._attention_forward(layer['input_layernorm'](x), layer['self_attn'], mask=attention_mask, cache=cache)
        h = x + r
        r = self._mlp_forward(layer['post_attention_layernorm'](h), layer['mlp'])
        out = h + r
        return out, cache

    def _attention_forward(
        self,
        x: mx.array,
        attn: dict,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, L, _ = x.shape

        queries = attn['q_proj'](x)
        keys = attn['k_proj'](x)
        values = attn['v_proj'](x)

        queries = queries.reshape(B, L, attn['num_heads'], -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, attn['num_kv_heads'], -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, attn['num_kv_heads'], -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            queries = attn['rope'](queries, offset=key_cache.shape[2])
            keys = attn['rope'](keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = attn['rope'](queries)
            keys = attn['rope'](keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=attn['scale'], mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return attn['o_proj'](output), (keys, values)

    def _mlp_forward(self, x: mx.array, mlp: dict) -> mx.array:
        return mlp['down_proj'](nn.silu(mlp['gate_proj'](x)) * mlp['up_proj'](x))

    def sanitize(self, weights):
        return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

    @staticmethod
    def from_dict(params: dict) -> 'LanguageBase':
        config = LanguageConfig.from_dict(params)
        return LanguageBase(config)

