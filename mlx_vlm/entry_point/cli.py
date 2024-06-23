import glob
from importlib import import_module
import mlx.core as mx
import codecs
import importlib

from mlx_vlm.core.config_manager import ConfigManager
from mlx_vlm.entry_point.entry_point_interface import EntryPointInterface
from mlx_vlm.core.constants import MODELS_MAP
from mlx_vlm.media_engine.single_Image_engine import SingleImageMediaEngine
from mlx_vlm.core.logger import Logger
from mlx_vlm.core.utils import load_image, get_message_json

logging = Logger()

class CLI(EntryPointInterface):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = None
        self.engine = None
        self.initModel()
        self.initMediaEngine()

    def initMediaEngine(self):
        if self.args.engine == "single":
            self.engine = SingleImageMediaEngine(self.model_path)
        # TODO: Add support for other media engines: batch, video and real-time
        else:
            raise ValueError(f"Media engine {self.args.engine} not supported")

    def initModel(self):
        config = ConfigManager(self.model_path).get_config()
        model_type = config.get("model_type")
        model_type = MODELS_MAP.get(model_type, model_type)

        model_file = ''.join(['_' + char.lower() if char.isupper() else char for char in model_type]).lstrip('_')
        model_module = importlib.import_module(f"mlx_vlm.models.{model_file}")

        model_class_name = model_file.replace("_", " ").title().replace(" ", "")
        model_class = getattr(model_module, model_class_name)(config)

        self.model = model_class.from_pretrained(self.model_path)


    def execute(self):
        image = load_image(self.args.image)
        prompt = codecs.decode(self.args.prompt, "unicode_escape")

        # Use the processor from the engine
        processor = self.engine.processor

        if "chat_template" in processor.__dict__.keys():
            prompt = processor.apply_chat_template(
                [get_message_json(self.model.config.model_type, prompt)],
                tokenize=False,
                add_generation_prompt=True,
            )
        elif "tokenizer" in processor.__dict__.keys():
            if self.model.config.model_type != "paligemma":
                prompt = processor.tokenizer.apply_chat_template(
                    [get_message_json(self.model.config.model_type, prompt)],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            raise ValueError(
                "Error: processor does not have 'chat_template' or 'tokenizer' attribute."
            )

        output = self.engine.generate(
            self.model,
            self.engine.processor,  # Pass the processor here
            image,
            prompt,
            temp=self.args.temp,
            max_tokens=self.args.max_tokens,
            verbose=self.args.verbose
        )

        print("Model Output:", output)