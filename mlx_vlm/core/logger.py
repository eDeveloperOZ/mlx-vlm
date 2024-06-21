class Logger:
    def info(self, message: str):
        print(f"INFO: {message}")

    def warn(self, message: str):
        print(f"WARNING: {message}")

    def error(self, message: str):
        print(f"ERROR: {message}")
        print("exiting...")
        exit(1)

    def debug(self, message: str):
        print(f"DEBUG: {message}")