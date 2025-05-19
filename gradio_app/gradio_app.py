import gradio as gr
from chatbox.model import ChatBot
from dotenv import load_dotenv
import os
load_dotenv()


options = ["Naruto", "Sauske", "Sakura"]
def chatbot_interface(name):
    character_chatbox = ChatBot("Vovia/chatbot",
                                name=name,
                                huggingface_token=os.getenv('HUGGINGFACE_TOKEN'),
                                )
    def chatboxes(message,history):
        output = character_chatbox.chatting(message,history)
        output = output['content'].strip()
        return output

    return chatboxes





def main():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character ChatBot</h1>")
                gr.Markdown("### Please choose one option:")
                choice = gr.Radio(choices=options, label="Select a character", type="value", value="Naruto")
                start_button = gr.Button("Start Chat")
                chatbot_ui = gr.ChatInterface(fn=None, visible=False)

                def start_chat(character):
                    chatbot_fn = chatbot_interface(character)
                    chatbot_ui.fn = chatbot_fn
                    chatbot_ui.visible = True
                    return gr.update(visible=True)

                start_button.click(fn=start_chat, inputs=choice, outputs=chatbot_ui)

    demo.launch()

if __name__ == "__main__":
    main()
