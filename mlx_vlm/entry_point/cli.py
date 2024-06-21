import glob
from importlib import import_module
import mlx_vlm.core as mx
import codecs

from mlx_vlm.core.config_manager import ConfigManager
from mlx_vlm.entry_point.entry_point_interface import EntryPointInterface
from mlx_vlm.core.constants import MODELS_MAP
from mlx_vlm.media_engine.single_Image_engine import SingleImageMediaEngine
from mlx_vlm.core.logger import Logger
from mlx_vlm.core.utils import load_image, get_message_json

logging = Logger()

class CLI(EntryPointInterface):
    def __init__(self):
        super().__init__()
        self.model_path = self.get_model_path(self.args.model)
        self.model = None
        self.engine = None
        self.initModel()
        self.initMediaEngine()

    def initMediaEngine(self):
        if self.args.engine == "single":
            self.engine = SingleImageMediaEngine(self.model_path)
        else:
            raise ValueError(f"Media engine {self.args.engine} not supported")

    def initModel(self):
        config = ConfigManager(self.model_path).get_config()
        quantization = config.get("quantization", None)

        weight_files = glob.glob(str(self.model_path / "*.safetensors"))
        if not weight_files:
            # TODO: WTF?! find a better way to handle this!! 
            logging.info(message= f"""
No safetensors found in {self.model_path}
Create safetensors using the following code:
```
from transformers import AutoModelForCausalLM, AutoProcessor

model_id= "<huggingface_model_id>"
model = AutoModelForCausalLM.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

model.save_pretrained("<local_dir>")
processor.save_pretrained("<local_dir>")
```
Then use the <local_dir> as the --hf-path in the convert script.
```
python -m mlx_vlm.convert --hf-path <local_dir> --mlx-path <mlx_dir>
```
        """)
            logging.error(f"No safetensors found in {self.model_path}")
        
        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        if "language_config" in config:
            config["text_config"] = config["language_config"]
            del config["language_config"]
        
        model_type = config.get("model_type")
        model_type = MODELS_MAP.get(model_type, model_type)
        try:
            model_module = import_module(f"mlx_vlm.models.{model_type}")
            model_class = model_module.Model
            model_config_class = model_module.ModelConfig

            # Load and prepare configs
            vision_config = model_module.VisionConfig.from_dict(config["vision_config"])
            text_config = model_module.TextConfig.from_dict(config["text_config"])
            
            model_config = model_config_class.from_dict(config)
            model_config.vision_config = vision_config
            model_config.text_config = text_config

            # Initialize the model
            self.model = model_class(model_config)

            # Load weights
            weight_files = glob.glob(str(self.model_path / "*.safetensors"))
            if not weight_files:
                raise FileNotFoundError(f"No safetensors found in {self.model_path}")

            weights = {}
            for wf in weight_files:
                weights.update(mx.load(wf))

            # Apply model-specific sanitization if available
            if hasattr(self.model, "sanitize"):
                weights = self.model.sanitize(weights)

            # Load the weights into the model
            self.model.load_weights(list(weights.items()))

            # Set the model to evaluation mode
            self.model.eval()


            
        except ImportError:
            # TODO: Create adding model guide and provide it here as a link 
            logging.error(f"Model type {model_type} not supported (YET!)")


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

if __name__ == "__main__":
    CLI().execute()