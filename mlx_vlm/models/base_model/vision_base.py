from abc import ABC, abstractmethod
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

@dataclass
class VisionConfig:
    model_type: str = "default"
    num_hidden_layers: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    image_size: int = 224
    patch_size: int = 16
    layer_norm_eps: float = 1e-12
    num_channels: int = 3
    projection_dim: Optional[int] = None
    vocab_size: Optional[int] = None

    @classmethod
    def from_dict(cls, params: dict):
        return cls(**{k: v for k, v in params.items() if k in cls.__annotations__})
    
class VisionBase(ABC):
    def __init__(self, config: VisionConfig):
        self.config = config
        self.patch_embedding = nn.Conv2d(
            in_channels=self.config['num_channels'],
            out_channels=self.config['hidden_size'],
            kernel_size=self.config['patch_size'],
            stride=self.config['patch_size'],
        )
        self.num_patches = (self.config['image_size'] // self.config['patch_size']) ** 2
        self.position_embedding = nn.Embedding(self.num_patches + 1, self.config['hidden_size'])
        self.class_embedding = mx.zeros((self.config['hidden_size'],))
        
        self.encoder_layers = [self._create_encoder_layer() for _ in range(self.config['num_hidden_layers'])]
        
        self.post_layernorm = nn.LayerNorm(self.config['hidden_size'])

    def _create_encoder_layer(self):
        return {
            'self_attn': self._create_attention(),
            'layer_norm1': nn.LayerNorm(self.config['hidden_size'], eps=self.config['layer_norm_eps']),
            'mlp': self._create_mlp(),
            'layer_norm2': nn.LayerNorm(self.config['hidden_size'], eps=self.config['layer_norm_eps'])
        }

    def _create_attention(self):
        dims = self.config['hidden_size']
        num_heads = self.config['num_attention_heads']
        return {
            'num_heads': num_heads,
            'scale': (dims // num_heads) ** -0.5,
            'q_proj': nn.Linear(dims, dims),
            'k_proj': nn.Linear(dims, dims),
            'v_proj': nn.Linear(dims, dims),
            'out_proj': nn.Linear(dims, dims)
        }

    def _create_mlp(self):
        return {
            'activation_fn': nn.GELU(approx="fast"),
            'fc1': nn.Linear(self.config['hidden_size'], self.config['intermediate_size']),
            'fc2': nn.Linear(self.config['intermediate_size'], self.config['hidden_size'])
        }

    def __call__(self, x: mx.array, output_hidden_states: Optional[bool] = None) -> mx.array:
        x = self._embed(x)
        encoder_output, encoder_states = self._encode(x, output_hidden_states)
        pooler_output = self.post_layernorm(encoder_output[:, 0, :])
        return pooler_output, encoder_output, encoder_states

    def _embed(self, x: mx.array) -> mx.array:
        B = x.shape[0]
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        
        cls_embeddings = mx.broadcast_to(self.class_embedding, (B, 1, self.config.hidden_size))
        embeddings = mx.concatenate((cls_embeddings, patch_embeddings), axis=1)
        
        position_ids = mx.array(np.arange(self.num_patches + 1)[None, :])
        embeddings += self.position_embedding(position_ids)
        
        return embeddings

    def _encode(self, x: mx.array, output_hidden_states: Optional[bool] = None) -> Tuple[mx.array, Optional[Tuple[mx.array, ...]]]:
        encoder_states = (x,) if output_hidden_states else None
        
        for layer in self.encoder_layers:
            x = self._encoder_layer_forward(x, layer)
            if output_hidden_states:
                encoder_states = encoder_states + (x,)
        
        return x, encoder_states

    def _encoder_layer_forward(self, x: mx.array, layer: dict) -> mx.array:
        y = layer['layer_norm1'](x)
        y = self._attention_forward(y, layer['self_attn'])
        x = x + y
        y = layer['layer_norm2'](x)
        y = self._mlp_forward(y, layer['mlp'])
        return x + y

    def _attention_forward(self, x: mx.array, attn: dict) -> mx.array:
        B, L, _ = x.shape
        queries = attn['q_proj'](x)
        keys = attn['k_proj'](x)
        values = attn['v_proj'](x)

        queries = queries.reshape(B, L, attn['num_heads'], -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, attn['num_heads'], -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, attn['num_heads'], -1).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=attn['scale'])
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return attn['out_proj'](output)

    def _mlp_forward(self, x: mx.array, mlp: dict) -> mx.array:
        return mlp['fc2'](mlp['activation_fn'](mlp['fc1'](x)))

    def sanitize(self, weights: dict) -> dict:
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position ids
                continue
            elif "patch_embedding.weight" in k:
                # PyTorch conv2d weight tensors have shape:
                #   [out_channels, in_channels, kH, KW]
                # MLX conv2d expects the weight be of shape:
                #   [out_channels, kH, KW, in_channels]
                if self._check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights

    @staticmethod
    def _check_array_shape(arr: mx.array) -> bool:
        shape = arr.shape
        if len(shape) != 4:
            return False
        out_channels, kH, KW, _ = shape
        return (out_channels >= kH) and (out_channels >= KW) and (kH == KW)