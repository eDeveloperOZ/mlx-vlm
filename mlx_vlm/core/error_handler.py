from .logger import Logger

class ErrorHandler:
    """
    A class to handle errors across the MLX_VLM framework.
    This class provides methods to log and handle exceptions in a standardized way.
    """

    def __init__(self, logger):
        """
        Initializes the ErrorHandler with a logger.
        
        :param logger: A logger instance to log error messages.
        """
        self.logger = logger

    def handle_exception(self, e: Exception, message: str = "An error occurred"):
        """
        Handles exceptions by logging them with a custom message.
        
        :param e: The exception that was raised.
        :param message: A custom message to log with the exception information.
        """
        self.logger.error(f"{message}: {str(e)}")
        raise e
    
    def handle_not_implemented_error(self, message: str):
        """
        Handles not implemented errors by logging them with a custom message.
        
        :param message: A custom message to log with the exception information.
        """
        self.logger.error(f"{message}")
        raise NotImplementedError(message)

    def log_warning(self, message: str):
        """
        Logs a warning message.
        
        :param message: The warning message to log.
        """
        self.logger.warn(message)
