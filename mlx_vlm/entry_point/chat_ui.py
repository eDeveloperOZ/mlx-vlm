from mlx_vlm.entry_point.entry_point_interface import EntryPointInterface
import gradio as gr
from mlx_vlm.chat_ui import chat, generate

class ChatUI(EntryPointInterface):
    def run_chat(self):
        demo = gr.ChatInterface(
            fn=chat,
            title="MLX-VLM Chat UI",
            additional_inputs=[
                gr.Slider(minimum=0, maximum=1, step=0.1, value=0.1, label="Temperature"),
                gr.Slider(minimum=128, maximum=4096, step=1, value=200, label="Max new tokens"),
            ],
            description=f"Now Running {self.args.model}",
            multimodal=True,
        )
        demo.launch(inbrowser=True)

if __name__ == "__main__":
    ChatUI().execute()