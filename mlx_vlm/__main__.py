from mlx_vlm.entry_point.cli import CLI
from mlx_vlm.entry_point.chat_ui import ChatUI
# TODO: Implement the generate classes
# from mlx_vlm.entry_point.generate import Generate
from mlx_vlm.core.utils import parse_arguments

def main():
    print(f"\nI AM HERE At main\n")
    args = parse_arguments()
    print(f"\nArguments parsed: {args}\n")
    try:
        if args.mode == "cli":
            entry_point = CLI()
        elif args.mode == "chat":
            entry_point = ChatUI()
        elif args.mode == "generate":
            # TODO: Implement the generate classes
            pass
        else:
            raise ValueError("Invalid mode")

        entry_point.execute()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()