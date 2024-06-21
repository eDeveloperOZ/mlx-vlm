import json
import os

from .logger import Logger
logging = Logger()

class ConfigManager:
    def __init__(self, model_path=None):
        self.config_path = model_path / "config.json"
        self.config = self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, 'r') as config_file:
                return json.load(config_file)
        except FileNotFoundError:
            logging.error(f"Config file not found at {self.config_path}, exiting...")
            
    def get_config(self):
        return self.config