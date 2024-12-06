from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

model.save_pretrained("./Qwen2.5-1.5B")
tokenizer.save_pretrained("./Qwen2.5-1.5B")
