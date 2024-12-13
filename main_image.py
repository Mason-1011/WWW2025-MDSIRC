from config import config
from evaluator import ImageEvaluator
from loader import load_data
import torch
import logging
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from model import ImageModel, choose_optimizer
import os
import glob

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

def save_checkpoint(model, optimizer, epoch, save_dir, task_name, max_checkpoints=5, interval=10):
    """保存模型断点，支持数量上限管理和永久保存"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 保存文件路径
    checkpoint_path = os.path.join(save_dir, f"{task_name}_checkpoint_epoch_{epoch}.pth")
    # 仅保存添加的分类块
    # state = {
    #     "model_state": {
    #         k: v for k, v in model.state_dict().items() if "encoder" not in k and "processor" not in k
    #     },
    #     "optimizer_state": optimizer.state_dict(),
    #     "epoch": epoch,
    # }
    state = {
        # 保存模型中非 encoder 和 processor 的权重
        "model_state": {
            k: v for k, v in model.state_dict().items() if "encoder" not in k and "processor" not in k
        },
        # 保存 LoRA 权重
        "lora_state": {
            k: v for k, v in model.encoder.state_dict().items() if "lora_" in k
        },
        # 保存优化器状态
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved for epoch {epoch} at {checkpoint_path}")

    # 保留永久保存的模型（如每10个epoch）
    if epoch % interval == 0:
        print(f"Permanent checkpoint saved for epoch {epoch}")
        return

    # 管理普通断点数量，删除多余的
    all_checkpoints = sorted(glob.glob(os.path.join(save_dir, f"{task_name}_checkpoint_epoch_*.pth")), key=os.path.getmtime)
    normal_checkpoints = [ckpt for ckpt in all_checkpoints if f"{task_name}_checkpoint_epoch_{interval}" not in ckpt]

    if len(normal_checkpoints) > max_checkpoints:
        oldest_checkpoint = normal_checkpoints[0]
        os.remove(oldest_checkpoint)
        print(f"Removed oldest checkpoint: {oldest_checkpoint}")

def load_checkpoint(model, optimizer, save_dir, task_name):
    """加载最新模型断点"""
    all_checkpoints = sorted(glob.glob(os.path.join(save_dir, f"{task_name}_checkpoint_epoch_*.pth")), key=os.path.getmtime)
    if not all_checkpoints:
        print("No checkpoints found, starting training from scratch.")
        return 0

    latest_checkpoint = all_checkpoints[-1]
    checkpoint = torch.load(latest_checkpoint, map_location="cuda" if torch.cuda.is_available() else "cpu")
    # 加载分类块权重
    model.load_state_dict(
        {k: v for k, v in checkpoint["model_state"].items() if k in model.state_dict()},
        strict=False # 允许部分 key 不匹配
    )
    # 加载lora权重
    if "lora_state" in checkpoint:
        model.encoder.load_state_dict(
            {k: v for k, v in checkpoint["lora_state"].items() if k in model.encoder.state_dict()},
            strict=False
        )
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"Checkpoint loaded from {latest_checkpoint} at epoch {checkpoint['epoch']}")
    return checkpoint["epoch"]

def train_Qwen2(config, logger, task_name, epochs = 40, save_dir="/root/autodl-tmp/WWW2025-MDSIRC/checkpoints", max_checkpoints=5, interval=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    model = ImageModel(config).to(device).to(dtype=torch.bfloat16)
    data_iter_train = load_data(config, "image", "/root/autodl-tmp/WWW2025-MDSIRC/train/train_image.json")
    data_iter_valid = load_data(config, "image", "/root/autodl-tmp/WWW2025-MDSIRC/train/valid_image.json")

    # 加载优化器
    optimizer = choose_optimizer(config, model)

    evaluator = ImageEvaluator(config, model, logger = logger)

    # 加载断点
    start_epoch = load_checkpoint(model, optimizer, save_dir, task_name)

    model.train()
    for epoch in range(start_epoch, epochs):
        steps = 0
        for sample in data_iter_train:
            label = [config["image_label_map"][i] for i in sample["label"]]
            output_id = torch.tensor(label).to(device)
            loss, logits = model(sample, output_id)
            loss.backward()
            optimizer.step()

            if steps % 350 == 0 and steps > 0:
                logger.info(f"---------------当前epoch: {epoch}")
                logger.info(f"训练损失: {loss}")
                evaluator.eval_Qwen2_VL_classify_block(data_iter_valid)
            steps += len(sample["label"])

        # 保存断点
        save_checkpoint(model, optimizer, epoch + 1, save_dir, task_name, max_checkpoints, interval)



if __name__ == "__main__":
    # 创建一个日志文件名（你可以根据需要自定义文件名）
    task_name = "12-13"
    log_file = f'/root/autodl-tmp/WWW2025-MDSIRC/log/{task_name}_log_eval_Qwen2-VL.log'

    # 配置日志，输出到控制台和文件
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(),  # 输出到控制台
                            logging.FileHandler(log_file, mode='a', encoding='utf-8', delay=True)  # 输出到文件，追加模式
                        ])
    logger = logging.getLogger(__name__)
    # eval_origin_model(logger) # 不进行任何微调，准确度约为0.4
    train_Qwen2(config, logger, task_name)