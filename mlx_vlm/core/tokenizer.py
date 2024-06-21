import json
from pathlib import Path
from functools import partial
from transformers import AutoTokenizer

from mlx_vlm.tokenizer_utils import (
    NaiveStreamingDetokenizer,
    SPMStreamingDetokenizer,
    BPEStreamingDetokenizer,
    TokenizerWrapper,
)

def _match(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return len(a) == len(b) and all(k in b and _match(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_match(ai, bi) for ai, bi in zip(a, b))
    return a == b

def _is_spm_decoder(decoder):
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ],
    }
    return _match(_target_description, decoder)

def _is_spm_decoder_no_space(decoder):
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ],
    }
    return _match(_target_description, decoder)

def _is_bpe_decoder(decoder):
    _target_description = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": False,
    }
    return _match(_target_description, decoder)

def load_tokenizer(model_path, return_tokenizer=True, tokenizer_config_extra={}):
    """
    Load a huggingface tokenizer and try to infer the type of streaming
    detokenizer to use.

    Note, to use a fast streaming tokenizer, pass a local file path rather than
    a Hugging Face repo ID.

    Args:
        model_path (str or Path): Path to the model directory or Hugging Face repo ID.
        return_tokenizer (bool): If True, return a TokenizerWrapper. If False, return only the detokenizer class.
        tokenizer_config_extra (dict): Extra configuration options for the tokenizer.

    Returns:
        TokenizerWrapper or detokenizer_class: Depending on the return_tokenizer parameter.
    """
    model_path = Path(model_path)
    detokenizer_class = NaiveStreamingDetokenizer

    tokenizer_file = model_path / "tokenizer.json"
    if tokenizer_file.exists():
        with tokenizer_file.open() as f:
            tokenizer_content = json.load(f)
        if "decoder" in tokenizer_content:
            if _is_spm_decoder(tokenizer_content["decoder"]):
                detokenizer_class = SPMStreamingDetokenizer
            elif _is_spm_decoder_no_space(tokenizer_content["decoder"]):
                detokenizer_class = partial(SPMStreamingDetokenizer, trim_space=False)
            elif _is_bpe_decoder(tokenizer_content["decoder"]):
                detokenizer_class = BPEStreamingDetokenizer

    if return_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), **tokenizer_config_extra)
        return TokenizerWrapper(tokenizer, detokenizer_class)
    else:
        return detokenizer_class

class Tokenizer:
    def __init__(self, model_path, tokenizer_config_extra={}):
        self.tokenizer = load_tokenizer(model_path, return_tokenizer=True, tokenizer_config_extra=tokenizer_config_extra)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id