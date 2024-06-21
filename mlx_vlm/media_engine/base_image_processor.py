from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Union
from PIL import Image
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.image_processing_utils import get_size_dict
from transformers.image_utils import ChannelDimension, PILImageResampling, load_image
from transformers import AutoImageProcessor
import mlx.nn as nn
import mlx.core as mx
import numpy as np
from typing import Optional, Callable, Generator

from mlx_vlm.core.tokenizer import load_tokenizer
from .media_engine_interface import MediaEngineInterface
from mlx_vlm.core.utils import top_p_sampling, apply_repetition_penalty

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class BaseImageProcessor(MediaEngineInterface, ABC):
    def __init__(
        self,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        size: Tuple[int, int] = (384, 384),
        crop_size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        rescale_factor: float = 1 / 255,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        model_path: str = None
    ):
        super().__init__()
        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = get_size_dict(
            crop_size or {"height": size[0], "width": size[1]},
            default_to_square=True,
            param_name="crop_size"
        )
        self.init_engine(model_path)
    
    def init_engine(self, model_path: str)-> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        self.engine_config = {"trust_remote_code": True}
        self.processor = AutoImageProcessor.from_pretrained(model_path, **self.engine_config)
        detokenizer_class = load_tokenizer(model_path, return_tokenizer=False)
        if "tokenizer" in self.processor.__dict__.keys():
            self.processor.detokenizer = detokenizer_class(self.processor.tokenizer)
        else:
            self.processor.detokenizer = detokenizer_class(self.processor)
        
        

    @abstractmethod
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
    ) -> Any:
        pass

    def load_engine(self, model_path: str, engine_type: str) -> None:
        # Implement engine loading logic here
        pass
    
    def generate_step(
        self,
        model: nn.Module,
        prompt: mx.array,
        mask: mx.array,
        cache=None,
        temp: float = 0.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = 20,
        top_p: float = 1.0,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
        """
        A generator producing text based on the given prompt from the model.

        Args:
            prompt (mx.array): The input prompt.
            model (nn.Module): The model to use for generation.
            temp (float): The temperature for sampling, if 0 the argmax is used.
            repetition_penalty (float, optional): The penalty factor for repeating tokens.
            repetition_context_size (int, optional): The number of tokens to consider for repetition penalty (default 20).
            top_p (float, optional): Nulceus sampling, higher means model considers more less likely words

        Yields:
            Generator[Tuple[mx.array, mx.array]]: A generator producing
            one token and probability per call.
        """

        if repetition_penalty and (
            repetition_penalty < 0 or not isinstance(repetition_penalty, float)
        ):
            raise ValueError(
                f"repetition_penalty must be a non-negative float, got {repetition_penalty}"
            )

        y, prob = self.sample(prompt, temp, top_p)

        repetition_context = prompt.tolist()

        if repetition_context_size:
            repetition_context = repetition_context[-repetition_context_size:]

        while True:
            logits, cache = model(y[None], mask=mask, cache=cache)
            logits = logits[:, -1, :]

            if repetition_penalty:
                logits = apply_repetition_penalty(
                    logits, repetition_context, repetition_penalty
                )
                y, prob = self.sample(logits, temp, top_p)
                repetition_context.append(y.item())
            else:
                y, prob = self.sample(logits, temp, top_p)

            if repetition_context_size:
                if len(repetition_context) > repetition_context_size:
                    repetition_context = repetition_context[-repetition_context_size:]
            yield y, prob

    def sample(logits: mx.array, temp: float, top_p: float) -> Tuple[mx.array, float]:
        softmax_logits = mx.softmax(logits)

        if temp == 0:
            token = mx.argmax(logits, axis=-1)
        else:
            if top_p > 0 and top_p < 1.0:
                token = top_p_sampling(logits, top_p, temp)
            else:
                token = mx.random.categorical(logits * (1 / temp))

        prob = softmax_logits[0, token]
        return token, prob  
    
    def prepare_inputs(image_processor, processor, image, prompt, image_token_index):

        mask = None
        if isinstance(image, str):
            image = load_image(image)

        if image_processor is not None:
            text_chunks = [processor(chunk).input_ids for chunk in prompt.split("<image>")]
            input_ids = mx.array([text_chunks[0] + [image_token_index] + text_chunks[1]])
            pixel_values = image_processor.preprocess(images=[image])[0]
            pixel_values = mx.array(np.expand_dims(pixel_values, axis=0))
        else:
            inputs = processor(prompt, image, return_tensors="np")
            pixel_values = mx.array(inputs["pixel_values"])
            input_ids = mx.array(inputs["input_ids"])
            mask = mx.array(inputs["attention_mask"])
        return input_ids, pixel_values, mask