from mlx_vlm.core.utils import parse_arguments
from mlx_vlm.entry_point.cli import CLI
from mlx_vlm.entry_point.chat_ui import ChatUI

def main():
    print("\nI AM HERE At main\n")
    args = parse_arguments()
    print(f"\nArguments parsed: {args}\n")
    try:
        if args.mode == "cli":
            entry_point = CLI()
        elif args.mode == "chat":
            entry_point = ChatUI()
        elif args.mode == "generate":
            # TODO: Implement the generate classes
            print("Generate mode not implemented yet")
            return
        else:
            raise ValueError(f"Invalid mode: {args.mode}")

        entry_point.execute()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()