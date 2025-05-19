import gradio as gr
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chatbox.model import ChatBot
from dotenv import load_dotenv

load_dotenv()

def chat_function(message, history):
    chatbox = ChatBot(
            "Vovia/chatbot",
            huggingface_token=os.getenv("HUGGINGFACE_TOKEN")
        )

    output = chatbox.chatting(message, history)
    return output['content'].strip()

def main():
    with gr.Blocks() as demo:
        gr.HTML("<h1>Character ChatBot</h1>")
        gr.ChatInterface(chat_function)

    demo.launch(share=True)

if __name__ == "__main__":
    main()
