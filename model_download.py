from transformers import AutoModelForCausalLM, AutoTokenizer

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

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-2B", torch_dtype="auto", device_map="auto"
# )
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B")
#
# model.save_pretrained("./Qwen2-VL-2B")
# processor.save_pretrained("./Qwen2-VL-2B")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

model.save_pretrained("./Qwen2-VL-7B-Instruct")
processor.save_pretrained("./Qwen2-VL-7B-Instruct")