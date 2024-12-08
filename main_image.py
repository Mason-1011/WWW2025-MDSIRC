from config import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluator import ImageEvaluator
from loader import load_data
import torch

def eval_origin_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    model_name = "/root/autodl-tmp/WWW2025-MDSIRC/Qwen-VL-Int4"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    image_classify_qwenvl = ImageEvaluator(config, model, tokenizer)
    data_iter = load_data(config, "image", "train/train.json")
    image_classify_qwenvl.eval_QWEN_VL(data_iter)


if __name__ == "__main__":
    # eval_origin_model() // 不进行任何微调，准确度约为0.2
    pass