# from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
# import torch
# from llava.model.builder import load_pretrained_model
# from transformers
# tokenizer, model, image_processor, context_len = load_pretrained_model('checkpoints/tinyllava-v1.0-1.1b-lora', 'TinyLlama/TinyLlama-1.1B-Chat-V1.0', 'llava')
# model = LlavaLlamaForCausalLM.from_pretrained('checkpoints/tinyllava-v1.0-1.1b/')
# print(model)
# torch.save(model.state_dict(), 'model_state_dict.bin')
# print(torch.load('model_state_dict.bin').keys())

import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer

model_id = "checkpoints/tinyllava-v1.0-1.1b-hf"

# prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
# image_file = "images/llava_logo.png"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to('cuda')

processor = AutoProcessor.from_pretrained(model_id)

# raw_image = Image.open(requests.get(image_file, stream=True).raw)
# raw_image = Image.open(image_file)
# inputs = processor(prompt, raw_image, return_tensors='pt').to('cuda', torch.float16)

# output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
# print(processor.decode(output[0][2:], skip_special_tokens=True))

model_max_length = 512
tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=None,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
    )
# tokenizer.pad_token = tokenizer.unk_token
print(f'pad_token:{tokenizer.pad_token}')
print(f'unk_token:{tokenizer.unk_token}')
num_added_tokens = tokenizer.add_tokens("[SEG]")
seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
print(f'seg_token_idx:{seg_token_idx}')