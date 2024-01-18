import os
import string
import copy
import gradio as gr
import PIL.Image
import torch
from transformers import BitsAndBytesConfig, pipeline
import re
import time
try: 
   import flash_attn
   use_flash_attention_2=True
except:
   use_flash_attention_2=False


DESCRIPTION = "# TinyLLaVA 🌋"

model_id = "checkpoints/tinyllava-v1.0-1.1b-hf"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
print(use_flash_attention_2)
pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config, "use_flash_attention_2":use_flash_attention_2})



def extract_response_pairs(text):
    turns = re.split(r'(USER:|ASSISTANT:)', text)[1:]
    turns = [turn.strip() for turn in turns if turn.strip()]
    conv_list = []
    for i in range(0, len(turns[1::2]), 2):
        if i + 1 < len(turns[1::2]):
            conv_list.append([turns[1::2][i].lstrip(":"), turns[1::2][i + 1].lstrip(":")])

    return conv_list



def add_text(history, text):
  history = history.append([text, None])
  return history, text

def infer(image, prompt,
            temperature,
            length_penalty,
            repetition_penalty,
            max_length,
            min_length,
            top_p):

  outputs = pipe(images=image, prompt=prompt,
                  generate_kwargs={"temperature":temperature,
                  "length_penalty":length_penalty,
                  "repetition_penalty":repetition_penalty,
                  "max_length":max_length,
                  "min_length":min_length,
                  "top_p":top_p})
  inference_output = outputs[0]["generated_text"]
  return inference_output



def bot(history_chat, text_input, image,
            temperature,
            length_penalty,
            repetition_penalty,
            max_length,
            min_length,
            top_p):
                
    if text_input == "":
        gr.Warning("Please input text")

    if image==None:
        gr.Warning("Please input image or wait for image to be uploaded before clicking submit.")
    chat_history = " ".join(history_chat) # history as a str to be passed to model
    chat_history = chat_history + f"USER: <image>\n{text_input}\nASSISTANT:" # add text input for prompting
    inference_result = infer(image, chat_history,
            temperature,
            length_penalty,
            repetition_penalty,
            max_length,
            min_length,
            top_p)
    # return inference and parse for new history
    chat_val = extract_response_pairs(inference_result)
    
    # create history list for yielding the last inference response
    chat_state_list = copy.deepcopy(chat_val)
    chat_state_list[-1][1] = "" # empty last response
    
    # add characters iteratively
    for character in chat_val[-1][1]:
        chat_state_list[-1][1] += character
        time.sleep(0.05)
        # yield history but with last response being streamed
        yield chat_state_list


css = """
  #mkd {
    height: 500px;
    overflow: auto;
    border: 1px solid #ccc;
  }
  """
with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.Markdown("""## LLaVA, one of the greatest multimodal chat models is now available in Transformers with 4-bit quantization! ⚡️
    See the docs here: https://huggingface.co/docs/transformers/main/en/model_doc/llava.""")
    chatbot = gr.Chatbot(label="Chat", show_label=False)
    gr.Markdown("Input image and text and start chatting 👇")
    with gr.Row():

      image = gr.Image(type="pil")
      text_input = gr.Text(label="Chat Input", show_label=False, max_lines=3, container=False)

    history_chat = gr.State(value=[])

    with gr.Accordion(label="Advanced settings", open=False):
        temperature = gr.Slider(
            label="Temperature",
            info="Used with nucleus sampling.",
            minimum=0.5,
            maximum=1.0,
            step=0.1,
            value=1.0,
        )
        length_penalty = gr.Slider(
            label="Length Penalty",
            info="Set to larger for longer sequence, used with beam search.",
            minimum=-1.0,
            maximum=2.0,
            step=0.2,
            value=1.0,
        )
        repetition_penalty = gr.Slider(
            label="Repetition Penalty",
            info="Larger value prevents repetition.",
            minimum=1.0,
            maximum=5.0,
            step=0.5,
            value=1.5,
        )
        max_length = gr.Slider(
            label="Max Length",
            minimum=1,
            maximum=500,
            step=1,
            value=200,
        )
        min_length = gr.Slider(
            label="Minimum Length",
            minimum=1,
            maximum=100,
            step=1,
            value=1,
        )
        top_p = gr.Slider(
            label="Top P",
            info="Used with nucleus sampling.",
            minimum=0.5,
            maximum=1.0,
            step=0.1,
            value=0.9,
        )
    chat_output = [
        chatbot,
        history_chat
    ]


    chat_inputs = [
        image,
        text_input,
        temperature,
        length_penalty,
        repetition_penalty,
        max_length,
        min_length,
        top_p,
        history_chat
    ]
    with gr.Row():
      clear_chat_button = gr.Button("Clear")
      cancel_btn = gr.Button("Stop Generation")
      chat_button = gr.Button("Submit", variant="primary")
      
    chat_event1 = chat_button.click(add_text, [chatbot, text_input], [chatbot, text_input]).then(bot, [chatbot, text_input,
                                                                                           image, temperature,
        length_penalty,
        repetition_penalty,
        max_length,
        min_length,
        top_p], chatbot)
    
    chat_event2 = text_input.submit(
        add_text,
        [chatbot, text_input],
        [chatbot, text_input]
    ).then(
        fn=bot,
        inputs=[chatbot, text_input, image, temperature,
        length_penalty,
        repetition_penalty,
        max_length,
        min_length,
        top_p],
        outputs=chatbot
    )
    clear_chat_button.click(
        fn=lambda: ([], []),
        inputs=None,
        outputs=[
            chatbot,
            history_chat
        ],
        queue=False,
        api_name="clear",
    )
    image.change(
        fn=lambda: ([], []),
        inputs=None,
        outputs=[
            chatbot,
            history_chat
        ],
        queue=False)
    cancel_btn.click(
        None, [], [], 
        cancels=[chat_event1, chat_event2]
    )  
    examples = [["./images/baklava.png", "How to make this pastry?"],["./images/bee.png","Describe this image."]]
    gr.Examples(examples=examples, inputs=[image, text_input, chat_inputs])


    

if __name__ == "__main__":
    demo.queue(max_size=10).launch(debug=True)