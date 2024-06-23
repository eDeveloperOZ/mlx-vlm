import sys

from mlx_vlm.entry_point.cli import CLI
# TODO: Implement the chat_ui and generate classes
# from mlx_vlm.entry_point.chat_ui import ChatUI
# from mlx_vlm.entry_point.generate import Generate
from mlx_vlm.core.utils import parse_arguments
from mlx_vlm.core.logger import Logger


def main():
    args = parse_arguments()
    
    try:
        if args.mode == "cli":
            entry_point = CLI(args)
        # TODO: Implement the chat_ui and generate classes
        # elif args.mode == "chat":
        #     entry_point = ChatUI(args)
        # elif args.mode == "generate":
        #     entry_point = Generate()
        else:
            raise ValueError(f"Invalid mode: {args.mode}")
        print(f'\nexecuting the entry point!!')
        entry_point.execute()
    except Exception as e:
        # TODO: change this to a logger
        # TODO: change this only if the --verbose flag is set
        import traceback
        print("Execution stack trace:")
        traceback.print_exc()
        print(f"An error occurred in entry point: {e}")                  

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred in main: {e}")
