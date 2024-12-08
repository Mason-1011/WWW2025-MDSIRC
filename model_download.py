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