# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
#
# model.save_pretrained("./Qwen2.5-1.5B")
# tokenizer.save_pretrained("./Qwen2.5-1.5B")



# model_name = "Qwen/Qwen-VL-Chat-Int4"
#
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#
# model.save_pretrained("./Qwen-VL-Int4")
# tokenizer.save_pretrained("./Qwen-VL-Int4")

# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-2B", torch_dtype="auto", device_map="auto"
# )
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B")
#
# model.save_pretrained("./Qwen2-VL-2B")
# processor.save_pretrained("./Qwen2-VL-2B")

# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7ßB", torch_dtype="auto", device_map="auto"
# )
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# model.save_pretrained("./Qwen2-VL-7B-Instruct")
# processor.save_pretrained("./Qwen2-VL-7B-Instruct")

from modelscope import Qwen2VLForConditionalGeneration, AutoProcessor
from modelscope import snapshot_download
import os

os.environ['MODELSCOPE_CACHE'] = '/root/autodl-tmp/WWW2024-MDSIRC/modelscope-cache'
model_dir = snapshot_download("qwen/Qwen2-VL-7B-Instruct")

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     model_dir,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained(model_dir)

model.save_pretrained("./Qwen2-VL-7B-Instruct")
processor.save_pretrained("./Qwen2-VL-7B-Instruct")