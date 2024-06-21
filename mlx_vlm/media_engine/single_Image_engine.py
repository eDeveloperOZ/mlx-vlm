
import numpy as np
from PIL import Image
import mlx.nn as nn
from typing import Optional, Callable
import mlx.core as mx
from transformers import PreTrainedTokenizer
import time

from .base_image_processor import BaseImageProcessor, expand2square
from ..core.error_handler import ErrorHandler
from ..core.logger import Logger

logger = Logger()

class SingleImageMediaEngine(BaseImageProcessor):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        # self.image_size = image_size

    def generate(
        self,
        model: nn.Module,
        processor: PreTrainedTokenizer,
        image: str,
        prompt: str,
        image_processor=None,
        temp: float = 0.0,
        max_tokens: int = 100,
        verbose: bool = False,
        formatter: Optional[Callable] = None,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = None,
        top_p: float = 1.0,
    ) -> str:
        """
        Generate text from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       temp (float): The temperature for sampling (default 0).
       max_tokens (int): The maximum number of tokens (default 100).
       verbose (bool): If ``True``, print tokens and timing information
           (default ``False``).
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       repetition_penalty (float, optional): The penalty factor for repeating tokens.
        repetition_context_size (int, optional): The number of tokens to consider for repetition penalty.
    """
        if verbose:
            print("=" * 10)
            print("Image:", image, "\n")
            print("Prompt:", prompt)

        if image_processor is not None:
            prompt_tokens = mx.array(processor.encode(prompt))
            tokenizer = processor
        else:
            prompt_tokens = mx.array(processor.tokenizer.encode(prompt))
            tokenizer = processor.tokenizer

        image_token_index = model.config.image_token_index
        input_ids, pixel_values, mask = self.prepare_inputs(
            image_processor, processor, image, prompt, image_token_index
        )
        logits, cache = model(input_ids, pixel_values, mask)
        logits = logits[:, -1, :]
        y, _ = self.sample(logits, temp, top_p)

        tic = time.perf_counter()
        detokenizer = processor.detokenizer
        detokenizer.reset()

        detokenizer.add_token(y.item())
        for (token, prob), n in zip(
            self.generate_step(
                model.language_model,
                logits,
                mask,
                cache,
                temp,
                repetition_penalty,
                repetition_context_size,
                top_p,
            ),
            range(max_tokens),
        ):
            token = token.item()
            if n == 0:
                prompt_time = time.perf_counter() - tic
                tic = time.perf_counter()

            if token == tokenizer.eos_token_id:
                break

            detokenizer.add_token(token)

            if verbose:
                if formatter:
                    # We have to finalize so that the prob corresponds to the last segment
                    detokenizer.finalize()
                    formatter(detokenizer.last_segment, prob.item())
                else:
                    print(detokenizer.last_segment, end="", flush=True)

        token_count = n + 1
        detokenizer.finalize()

        if verbose:
            print(detokenizer.last_segment, flush=True)
            gen_time = time.perf_counter() - tic
            print("=" * 10)
            if token_count == 0:
                print("No tokens generated for this prompt")
                return
            prompt_tps = prompt_tokens.size / prompt_time
            gen_tps = (token_count - 1) / gen_time

            print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
            print(f"Generation: {gen_tps:.3f} tokens-per-sec")

        return detokenizer.text

