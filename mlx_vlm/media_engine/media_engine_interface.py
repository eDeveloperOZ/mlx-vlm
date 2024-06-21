from abc import ABC, abstractmethod
from typing import Any

from ..core.error_handler import ErrorHandler
from ..core.logger import Logger

logger = Logger()

class MediaEngineInterface(ABC):
    def __init__(self):
        self.error_handler = ErrorHandler(logger)

    @abstractmethod
    def process(self, input: Any) -> Any:
        raise NotImplementedError("Subclass must implement this method")
