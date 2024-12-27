import logging
import os
from loader import load_data
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from config import config
import os
from qwen_vl_utils import process_vision_info
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)
model_name = "/root/autodl-tmp/WWW2025-MDSIRC/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto", output_hidden_states=True
).eval()
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
tokenizer = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "train/sha/0a856d3d-6939-4a3b-b80c-46ff0ddb324d-461-0.jpg",
                        },
                        {"type": "text", "text": "Hello, describe this picture"},
                    ],
                }
            ]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
print(text)
print(image_inputs)
inputs = tokenizer(
    text=[text],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(device)
res = model(**inputs)
# 检查输出对象中的隐藏状态
hidden_states = res.hidden_states  # 这是一个 tuple，每一层的输出为一个张量
last_hidden_state = hidden_states[-1]  # 获取最后一层的隐藏状态
print("Last hidden state shape:", last_hidden_state.shape)
print(model)