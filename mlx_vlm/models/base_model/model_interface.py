import mlx.nn as nn

from ...core.error_handler import ErrorHandler
from ...core.logger import Logger


logger = Logger()

class ModelInterface(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.error_handler = ErrorHandler(logger)

    def process(self):
        self.error_handler.handle_not_implemented_error("Subclasses must implement this method.")

    def load_weights(self):
        self.error_handler.handle_not_implemented_error("Subclasses must implement this method.")

    def get_input_embeddings(self):
        self.error_handler.handle_not_implemented_error("Subclasses must implement this method.")
