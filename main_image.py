from config import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluator import ImageEvaluator
from loader import load_data
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
import logging

def eval_origin_model(logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    # model_name = "/root/autodl-tmp/WWW2025-MDSIRC/Qwen-VL-Int4"
    # model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    #
    # image_classify_qwenvl = ImageEvaluator(config, model, tokenizer)
    # data_iter = load_data(config, "image", "train/train.json")
    # image_classify_qwenvl.eval_QWEN_VL(data_iter)

    model_name = "/root/autodl-tmp/WWW2025-MDSIRC/Qwen2-VL-2B"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    min_pixels = 256 * 28 * 28
    max_pixels = 1560 * 32 * 32
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
    image_classify_qwenvl = ImageEvaluator(config, model, processor, logger)
    data_iter = load_data(config, "image", "train/train.json")
    image_classify_qwenvl.eval_QWEN2_VL(data_iter)


if __name__ == "__main__":
    # 创建一个日志文件名（你可以根据需要自定义文件名）
    log_file = 'log/log_eval_Qwen2-VL-Int4.log'

    # 配置日志，输出到控制台和文件
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(),  # 输出到控制台
                            logging.FileHandler(log_file, mode='a', encoding='utf-8')  # 输出到文件，追加模式
                        ])
    logger = logging.getLogger(__name__)
    eval_origin_model(logger) # 不进行任何微调，准确度约为0.4