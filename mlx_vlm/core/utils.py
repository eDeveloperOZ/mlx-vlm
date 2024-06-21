import argparse
from PIL import Image
from pathlib import Path
from io import BytesIO
from typing import Union
import requests
import mlx.core as mx

def parse_arguments():
    print(f'I AM PARSING ARGUMENTS')
    parser = argparse.ArgumentParser(description="MLX Vision Language Model")
    
    # Common arguments
    parser.add_argument("--model", type=str, default="qnguyen3/nanoLLaVA",
                        help="The path to the local model directory or Hugging Face repo.")
    parser.add_argument("--image", type=str,
                        default="http://images.cocodataset.org/val2017/000000039769.jpg",
                        help="URL or path of the image to process.")
    parser.add_argument("--prompt", type=str, default="What are these?",
                        help="Message to be processed by the model.")
    parser.add_argument("-e", "--engine", type=str, choices=["single", "batch", "video"],
                        default="single",
                        help="The engine to use for processing the input, single(default), batch or video")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum number of tokens to generate.")
    parser.add_argument("--temp", type=float, default=0.3,
                        help="Temperature for sampling.")
    parser.add_argument("--verbose", action="store_true",
                        help="Detailed output.")
    
    # Add a subparser for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # CLI mode
    cli_parser = subparsers.add_parser("cli", help="Run in CLI mode")
    
    # Chat UI mode
    chat_parser = subparsers.add_parser("chat", help="Run in Chat UI mode")
    
    # Generate mode
    generate_parser = subparsers.add_parser("generate", help="Run in Generate mode")
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.error("You must specify a mode: cli, chat, or generate")
    
    return args

def load_image(image_source: Union[str, Path, BytesIO]):
    """
    Helper function to load an image from either a URL or file.
    """
    if isinstance(image_source, BytesIO):
        # for base64 encoded images
        try:
            return Image.open(image_source)
        except IOError as e:
            raise ValueError(f"Failed to load image from BytesIO with error: {e}")
    elif image_source.startswith(("http://", "https://")):
        try:
            response = requests.get(image_source, stream=True)
            response.raise_for_status()
            return Image.open(response.raw)
        except Exception as e:
            raise ValueError(
                f"Failed to load image from URL: {image_source} with error {e}"
            )
    elif Path(image_source).is_file():
        try:
            return Image.open(image_source)
        except IOError as e:
            raise ValueError(f"Failed to load image {image_source} with error: {e}")
    else:
        raise ValueError(
            f"The image {image_source} must be a valid URL or existing file."
        )
    
def get_message_json(model_name, prompt):
    """
    Get the appropriate JSON message based on the specified model.

    Args:
        model_name (str): The model for which to generate the message. Options: 'Idefics 2', 'nanollava', 'llava'.
        prompt (str): The text prompt to be included in the message.
        *args: Additional positional arguments (unused).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        dict: A dictionary representing the JSON message for the specified model.
    """
    if model_name.lower() == "idefics2":
        message = {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    elif model_name.lower() in ["llava-qwen2", "llava"]:
        message = {"role": "user", "content": f"<image>\n{prompt}"}
    elif model_name.lower() == "multi_modality":
        message = {"role": "user", "content": f"<image>{prompt}"}
    elif model_name.lower() == "paligemma":
        message = prompt
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return message

def top_p_sampling(logits: mx.array, top_p: float, temperature: float) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logits: The logits from the model's output.
        top_p: The cumulative probability threshold for top-p filtering.
        temperature: Temperature parameter for softmax distribution reshaping.
    Returns:
        token selected based on the top-p criterion.
    """
    if (
        logits.dtype == mx.bfloat16
    ):  # workaround for unable to load kernel contiguous_scan_inclusive_sum_bfloat16_bfloat16
        logits = logits.astype(mx.float32)

    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460
    probs = mx.softmax(logits / temperature, axis=-1)

    # sort probs in ascending order
    sorted_indices = mx.argsort(probs, axis=-1)
    sorted_probs = probs[..., sorted_indices.squeeze(0)]

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # select tokens with cumulative probs below threshold
    top_probs = mx.where(
        cumulative_probs > 1 - top_p,
        sorted_probs,
        mx.zeros_like(sorted_probs),
    )

    sorted_token = mx.random.categorical(mx.log(top_probs))
    token = sorted_indices.squeeze(0)[sorted_token]

    return token

def apply_repetition_penalty(logits: mx.array, generated_tokens: any, penalty: float):
    """
    Apply repetition penalty to specific logits based on the given context.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        logits (mx.array): The logits produced by the language model.
        generated_tokens (any): A list of N previous tokens.
        penalty (float): The repetition penalty factor to be applied.

    Returns:
        logits (mx.array): Logits with repetition penalty applied to generated tokens.
    """
    if len(generated_tokens) > 0:
        indices = mx.array([token for token in generated_tokens])
        selected_logits = logits[:, indices]
        selected_logits = mx.where(
            selected_logits < 0, selected_logits * penalty, selected_logits / penalty
        )
        logits[:, indices] = selected_logits
    return logits